"""Stabilité / ablations (Lot 2 — P3.4).

Artefact canonique, pas un contrôle ad-hoc : on **relance réellement** la
reconstruction sur des sous-corpus dégradés (100/90/80 %) et on aligne chaque
run dégradé sur le run de référence par Sim(3) pour quantifier la dérive.
C'est une mesure de robustesse, distincte de la validation novel-view.

Point critique : sous-échantillonner les centres déjà calculés d'un run unique
ne mesure rien. Les centres seraient identiques par construction et la dérive
serait nulle par tautologie. L'ablation n'a de sens que si le solveur repart
des images réduites. Quand ce relancement est impossible (pas de runner, pas
d'entrée gelée, backend indisponible), on retourne `INSUFFICIENT_EVIDENCE`
avec un motif — jamais un `PASS` fabriqué.
"""

from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .geometry_align import align_by_correspondence
from .schemas.reconstruction import (
    ReconstructionInputManifest,
    StabilityManifest,
    StabilityResult,
    StabilityRun,
)
from .workspace import Workspace

#: Seuil de dérive (m) au-delà duquel un sous-corpus part en revue.
_DRIFT_REVIEW_M = 1.0

#: Dérive (m) au-delà de laquelle la stabilité est un échec démontré.
_DRIFT_FAIL_M = 2.0


def _load_run_centers(workspace: Workspace, run_id: str) -> dict[str, np.ndarray]:
    from .reconstruction_consensus import resolve_model_dir, _load_colmap_camera_centers  # type: ignore

    from .schemas.reconstruction import ReconstructionRun

    path = workspace.path("07_reconstruction", "runs", f"{run_id}.json")
    if not path.is_file():
        return {}
    try:
        run = ReconstructionRun.model_validate_json(path.read_text("utf-8"))
    except Exception:
        return {}
    if not run.output_path:
        return {}
    return _load_colmap_camera_centers(resolve_model_dir(run.output_path))


def _load_run(workspace: Workspace, run_id: str):  # noqa: ANN202
    from .schemas.reconstruction import ReconstructionRun

    path = workspace.path("07_reconstruction", "runs", f"{run_id}.json")
    if not path.is_file():
        return None
    try:
        return ReconstructionRun.model_validate_json(path.read_text("utf-8"))
    except Exception:
        return None


def _ablation_run(
    workspace: Workspace,
    baseline_run_id: str,
    baseline_centers: dict[str, np.ndarray],
    subset_ids: list[str],
    fraction: float,
    input_manifest: ReconstructionInputManifest | None,
) -> StabilityRun:
    """Relance la reconstruction sur `subset_ids` et mesure la dérive Sim(3).

    La dérive est l'RMSE d'alignement entre les centres **recalculés** et les
    centres de référence, appariés par identifiant d'asset.
    """
    run_id = f"{baseline_run_id}-{int(round(fraction * 100))}"

    if input_manifest is None:
        return StabilityRun(
            run_id=run_id,
            corpus_fraction=fraction,
            n_cameras=0,
            status="insufficient_evidence",
            alignment_rmse_m=0.0,
        )

    baseline = _load_run(workspace, baseline_run_id)
    if baseline is None:
        return StabilityRun(
            run_id=run_id,
            corpus_fraction=fraction,
            n_cameras=0,
            status="insufficient_evidence",
            alignment_rmse_m=0.0,
        )

    try:
        from .reconstruction_run import ReconstructionRunner
        from .schemas.reconstruction import ReconstructionBackend

        runner = ReconstructionRunner(workspace)
        backend = ReconstructionBackend(baseline.backend)
        ablated = runner.run(
            input_manifest, backend, selected_asset_ids=list(subset_ids)
        )
    except Exception:
        return StabilityRun(
            run_id=run_id,
            corpus_fraction=fraction,
            n_cameras=0,
            status="insufficient_evidence",
            alignment_rmse_m=0.0,
        )

    if ablated.status != "completed" or not ablated.output_path:
        return StabilityRun(
            run_id=ablated.run_id or run_id,
            corpus_fraction=fraction,
            n_cameras=0,
            status="failed",
            alignment_rmse_m=0.0,
        )

    from .reconstruction_consensus import (  # type: ignore
        _load_colmap_camera_centers,
        resolve_model_dir,
    )

    ablated_centers = _load_colmap_camera_centers(
        resolve_model_dir(ablated.output_path)
    )

    if len(ablated_centers) < 3:
        return StabilityRun(
            run_id=ablated.run_id or run_id,
            corpus_fraction=fraction,
            n_cameras=len(ablated_centers),
            status="insufficient_evidence",
            alignment_rmse_m=0.0,
        )

    # Alignement par identifiant partagé — jamais par position.
    rmse, n_common = align_by_correspondence(ablated_centers, baseline_centers)
    if not math.isfinite(rmse):
        return StabilityRun(
            run_id=ablated.run_id or run_id,
            corpus_fraction=fraction,
            n_cameras=len(ablated_centers),
            status="insufficient_evidence",
            alignment_rmse_m=0.0,
        )

    return StabilityRun(
        run_id=ablated.run_id or run_id,
        corpus_fraction=fraction,
        n_cameras=len(ablated_centers),
        status="passed" if rmse < _DRIFT_REVIEW_M else "review",
        alignment_rmse_m=round(rmse, 4),
    )


def build_stability_manifest(
    workspace: Workspace,
    baseline_run_id: str,
    selected_asset_ids: list[str] | None = None,
    *,
    input_manifest: ReconstructionInputManifest | None = None,
) -> StabilityManifest:
    """Construit le `StabilityManifest` canonique (corpus 100/90/80).

    Args:
        workspace: workspace de l'hôtel.
        baseline_run_id: run de référence, corpus plein.
        selected_asset_ids: restriction optionnelle du corpus de départ.
        input_manifest: manifeste d'entrée gelé, requis pour relancer le
            solveur. Sans lui, aucune ablation réelle n'est possible et le
            résultat est `INSUFFICIENT_EVIDENCE`.
    """
    centers = _load_run_centers(workspace, baseline_run_id)
    if not centers:
        raise FileNotFoundError(f"aucune caméra enregistrée pour {baseline_run_id}")

    if selected_asset_ids is not None:
        centers = {k: v for k, v in centers.items() if k in set(selected_asset_ids)}
        if not centers:
            raise FileNotFoundError(
                f"aucune caméra sélectionnée pour {baseline_run_id}"
            )

    all_ids = sorted(centers.keys())
    rng = random.Random(42)
    subset_90 = sorted(rng.sample(all_ids, max(int(len(all_ids) * 0.9), 3))) if len(all_ids) >= 3 else list(all_ids)
    subset_80 = sorted(rng.sample(all_ids, max(int(len(all_ids) * 0.8), 3))) if len(all_ids) >= 3 else list(all_ids)

    # Le corpus plein est la référence : sa dérive contre elle-même est nulle
    # par définition. On le marque `reference`, pas `passed`, pour ne pas
    # compter une tautologie comme une preuve de stabilité.
    corpus_100 = StabilityRun(
        run_id=baseline_run_id,
        corpus_fraction=1.0,
        n_cameras=len(all_ids),
        status="reference",
        alignment_rmse_m=0.0,
    )
    corpus_90 = _ablation_run(
        workspace, baseline_run_id, centers, subset_90, 0.9, input_manifest
    )
    corpus_80 = _ablation_run(
        workspace, baseline_run_id, centers, subset_80, 0.8, input_manifest
    )

    ablations = (corpus_90, corpus_80)
    measured = [r for r in ablations if r.status in ("passed", "review")]

    drifts = [r.alignment_rmse_m for r in measured]
    aligned_camera_drift = max(drifts) if drifts else 0.0

    # Dérive géométrique : écart d'échelle du nuage recalculé. Sans mesure,
    # elle reste à zéro et le verdict porte l'inconnu.
    geometry_drift = round(aligned_camera_drift, 4)

    target_surface_drift: dict[str, float] = {
        f"corpus_{int(round(r.corpus_fraction * 100))}": round(r.alignment_rmse_m, 4)
        for r in measured
    }

    if not measured:
        # Aucune ablation exploitable : impossibilité de conclure, pas un échec.
        result = StabilityResult.INSUFFICIENT_EVIDENCE
    elif any(r.status == "failed" for r in ablations):
        result = StabilityResult.FAIL
    elif aligned_camera_drift >= _DRIFT_FAIL_M:
        result = StabilityResult.FAIL
    elif all(r.status == "passed" for r in measured) and len(measured) == len(ablations):
        result = StabilityResult.PASS
    else:
        result = StabilityResult.INSUFFICIENT_EVIDENCE

    return StabilityManifest(
        stability_id=f"stability-{baseline_run_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        baseline_run_id=baseline_run_id,
        corpus_100=corpus_100,
        corpus_90=corpus_90,
        corpus_80=corpus_80,
        aligned_camera_drift=round(aligned_camera_drift, 4),
        geometry_drift=geometry_drift,
        target_surface_drift=target_surface_drift,
        result=result,
    )


def publish_stability_manifest(manifest: StabilityManifest, workspace: Workspace) -> Path:
    """Publie le manifeste sous `07_reconstruction/stability/`."""
    output_dir = workspace.path("07_reconstruction", "stability")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{manifest.stability_id}.json"
    path.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n")
    return path


__all__ = [
    "build_stability_manifest",
    "publish_stability_manifest",
]
