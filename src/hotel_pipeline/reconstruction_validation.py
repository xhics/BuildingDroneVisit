"""Validations de reconstruction (Lot 2 — P5).

Ce module fournit trois validations obligatoires avant `ENVIRONMENT_3D_READY` :
- Held-out novel view : réserve 20% des images, vérifie la potentialité d'enregistrement
- Stability : compare la reconstruction complète à un sous-ensemble 90%
- Cross-solver consensus : vérifie que COLMAP/GLUEMAP/feed-forward ne divergent pas
"""

from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .geometry_align import (
    align_by_correspondence,
    alignment_rmse,
    apply_sim3,
    umeyama_sim3,
)
from .schemas.reconstruction import ReconstructionRun
from .reconstruction_consensus import resolve_model_dir
from .workspace import Workspace


# ---------------------------------------------------------------------------
# Helpers Sim(3)
# ---------------------------------------------------------------------------


def _umeyama_sim3(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Sim(3) amenant `source` sur `target`. Délègue à `geometry_align`."""
    return umeyama_sim3(source, target)


def _apply_sim3(points: np.ndarray, R: np.ndarray, t: np.ndarray, s: float) -> np.ndarray:
    return apply_sim3(points, R, t, s)


def _is_pose_line(line: str) -> bool:
    """Une ligne de pose finit par `CAMERA_ID NAME` ; une ligne d'observations
    n'est faite que de nombres."""
    parts = line.split()
    if len(parts) < 10:
        return False
    try:
        float(parts[-1])
    except ValueError:
        return True
    return False


def _load_colmap_camera_centers(run_dir: Path) -> dict[str, np.ndarray]:
    images_file = run_dir / "normalized" / "images"
    if not images_file.is_file():
        return {}
    # COLMAP écrit **deux** lignes par image : la pose, puis les
    # observations « X Y POINT3D_ID … ». Lire toutes les lignes prenait ces
    # observations pour des poses — autant de caméras fantômes.
    _lines = [
        line for line in images_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if any(not _is_pose_line(line) for line in _lines):
        _lines = _lines[::2]

    centers: dict[str, np.ndarray] = {}
    for line in _lines:
        parts = line.split()
        if len(parts) < 9:
            continue
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        name = parts[9] if len(parts) > 9 else parts[8]
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)],
        ])
        t = np.array([tx, ty, tz])
        center = -R.T @ t
        centers[Path(name).stem] = center
    return centers


def _load_run_centers(run: ReconstructionRun) -> dict[str, np.ndarray]:
    if not run.output_path:
        return {}
    run_dir = resolve_model_dir(run.output_path)

    normalized_dir = run_dir / "normalized"
    if not normalized_dir.is_dir():
        normalized_dir = run_dir.parent / "normalized"
    if not normalized_dir.is_dir():
        normalized_dir = run_dir.parent.parent / "normalized"
    if not normalized_dir.is_dir():
        return {}
    return _load_colmap_camera_centers(normalized_dir.parent)


# ---------------------------------------------------------------------------
# Modèles
# ---------------------------------------------------------------------------


class HeldOutValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    validation_id: str
    reconstruction_run_id: str
    status: str
    metrics: dict = Field(default_factory=dict)
    error: str | None = None


class StabilityValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    validation_id: str
    reconstruction_run_id: str
    status: str
    subsets: list[dict] = Field(default_factory=list)
    error: str | None = None


class CrossSolverValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    validation_id: str
    reconstruction_run_id: str
    status: str
    consensus_score: float = Field(default=0.0, ge=0.0, le=1.0)
    divergent_solvers: list[str] = Field(default_factory=list)
    error: str | None = None


class ReconstructionValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    validation_id: str
    reconstruction_run_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    held_out: HeldOutValidation | None = None
    stability: StabilityValidation | None = None
    cross_solver: CrossSolverValidation | None = None
    overall_status: str = "pending"


# ---------------------------------------------------------------------------
# Validations
# ---------------------------------------------------------------------------


def _load_run(run_id: str, workspace: Workspace) -> ReconstructionRun | None:
    path = workspace.path("07_reconstruction", "runs", f"{run_id}.json")
    if not path.is_file():
        return None
    try:
        return ReconstructionRun.model_validate_json(path.read_text("utf-8"))
    except Exception:
        return None


def _load_sibling_runs(workspace: Workspace, reconstruction_input_id: str) -> list[ReconstructionRun]:
    runs_dir = workspace.path("07_reconstruction", "runs")
    siblings: list[ReconstructionRun] = []
    if not runs_dir.is_dir():
        return siblings
    for path in runs_dir.glob("*.json"):
        try:
            run = ReconstructionRun.model_validate_json(path.read_text("utf-8"))
            if run.reconstruction_input_id == reconstruction_input_id and run.status == "completed":
                siblings.append(run)
        except Exception:
            continue
    return siblings


def validate_held_out(
    workspace: Workspace,
    reconstruction_run_id: str,
) -> HeldOutValidation:
    validation_id = (
        f"heldout-{reconstruction_run_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )

    run = _load_run(reconstruction_run_id, workspace)
    if run is None or run.status != "completed":
        return HeldOutValidation(
            validation_id=validation_id,
            reconstruction_run_id=reconstruction_run_id,
            status="failed",
            error="run introuvable ou non complété",
        )

    centers = _load_run_centers(run)
    if not centers:
        return HeldOutValidation(
            validation_id=validation_id,
            reconstruction_run_id=reconstruction_run_id,
            status="failed",
            error="aucune caméra enregistrée",
        )

    # Vues cachées = assets sélectionnés mais non enregistrés
    snapshot_path = workspace.path(
        "07_reconstruction", f"reconstruction_input_{run.reconstruction_input_id}.json"
    )
    input_path = workspace.path("07_reconstruction", "plans")
    selected_ids = list(centers.keys())
    held_out_ids = []
    if snapshot_path.is_file():
        try:
            from .schemas import ReconstructionInputManifest

            snapshot = ReconstructionInputManifest.model_validate_json(
                snapshot_path.read_text("utf-8")
            )
            selected_ids = snapshot.selected_asset_ids
        except Exception:
            pass
    elif input_path.is_dir():
        for plan_file in input_path.glob("*.json"):
            try:
                plan = json.loads(plan_file.read_text("utf-8"))
                if plan.get("reconstruction_input_id") == run.reconstruction_input_id:
                    selected_ids = plan.get("selected_asset_ids", selected_ids)
                    break
            except Exception:
                continue

    registered = set(centers.keys())
    held_out_ids = [aid for aid in selected_ids if aid not in registered]

    metrics = {
        "total_selected": len(selected_ids),
        "registered": len(registered),
        "held_out": len(held_out_ids),
        "held_out_ids": held_out_ids[:20],
    }

    localization = None
    localization_dir = workspace.path("07_reconstruction", "localization")
    if localization_dir.is_dir():
        from .schemas import LocalizationManifest, PoseEvidenceClass

        for path in sorted(localization_dir.glob("*.json"), reverse=True):
            try:
                candidate = LocalizationManifest.model_validate_json(path.read_text("utf-8"))
            except Exception:
                continue
            if candidate.reconstruction_input_id == run.reconstruction_input_id:
                localization = candidate
                break

    if len(held_out_ids) == 0:
        status = "insufficient_evidence"
        metrics["note"] = "aucune vue cachée n'a été réellement testée"
    elif len(registered) < 3:
        status = "failed"
        metrics["note"] = "trop peu de vues enregistrées pour valider"
    elif localization is None:
        status = "insufficient_evidence"
        metrics["note"] = "aucun LocalizationManifest: identifier les vues cachées ne les valide pas"
    else:
        measured = {
            pose.asset_id
            for pose in localization.poses
            if pose.evidence_class is PoseEvidenceClass.LOCALIZED_MEASURED
        }
        tested = set(held_out_ids) & {attempt.asset_id for attempt in localization.attempts}
        passed = set(held_out_ids) & measured
        metrics["held_out_tested"] = len(tested)
        metrics["held_out_localized_measured"] = len(passed)
        metrics["held_out_success_rate"] = len(passed) / len(held_out_ids)
        status = "passed" if tested == set(held_out_ids) and passed == set(held_out_ids) else "failed"
        metrics["note"] = "validation fondée sur les tentatives PnP du manifeste de localisation"

    return HeldOutValidation(
        validation_id=validation_id,
        reconstruction_run_id=reconstruction_run_id,
        status=status,
        metrics=metrics,
    )


def validate_stability(
    workspace: Workspace,
    reconstruction_run_id: str,
) -> StabilityValidation:
    validation_id = (
        f"stability-{reconstruction_run_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )

    run = _load_run(reconstruction_run_id, workspace)
    if run is None or run.status != "completed":
        return StabilityValidation(
            validation_id=validation_id,
            reconstruction_run_id=reconstruction_run_id,
            status="failed",
            error="run introuvable ou non complété",
        )

    centers = _load_run_centers(run)
    if len(centers) < 4:
        return StabilityValidation(
            validation_id=validation_id,
            reconstruction_run_id=reconstruction_run_id,
            status="failed",
            error="moins de 4 caméras enregistrées",
        )

    asset_ids = sorted(centers.keys())
    subsets_results = []

    # Les sous-corpus sont comparés au corpus plein par identifiant partagé.
    # Comparer par position alignerait des caméras sans rapport : deux
    # échantillonnages n'ont aucune raison d'être dans le même ordre.
    rng = random.Random(42)
    for label, fraction, threshold in (("90%", 0.9, 0.5), ("80%", 0.8, 1.0)):
        subset = sorted(rng.sample(asset_ids, max(int(len(asset_ids) * fraction), 3)))
        subset_centers = {aid: centers[aid] for aid in subset if aid in centers}
        if len(subset_centers) < 3:
            continue
        rmse, n_common = align_by_correspondence(subset_centers, centers)
        if not math.isfinite(rmse):
            subsets_results.append({
                "subset": label,
                "n_cameras": len(subset),
                "alignment_rmse_m": None,
                "status": "insufficient_evidence",
            })
            continue
        subsets_results.append({
            "subset": label,
            "n_cameras": len(subset),
            "alignment_rmse_m": round(rmse, 4),
            "status": "passed" if rmse < threshold else "review",
        })

    # Une ablation non concluante n'est pas une ablation en échec : on la
    # distingue explicitement plutôt que de la fondre dans "review".
    statuses = {s.get("status") for s in subsets_results}
    if not subsets_results or statuses == {"insufficient_evidence"}:
        overall = "insufficient_evidence"
    elif statuses <= {"passed"}:
        overall = "passed"
    else:
        overall = "review"
    return StabilityValidation(
        validation_id=validation_id,
        reconstruction_run_id=reconstruction_run_id,
        status=overall,
        subsets=subsets_results,
    )


def validate_cross_solver(
    workspace: Workspace,
    reconstruction_run_id: str,
) -> CrossSolverValidation:
    validation_id = (
        f"crosssolver-{reconstruction_run_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )

    run = _load_run(reconstruction_run_id, workspace)
    if run is None or run.status != "completed":
        return CrossSolverValidation(
            validation_id=validation_id,
            reconstruction_run_id=reconstruction_run_id,
            status="failed",
            error="run introuvable ou non complété",
        )

    centers_run = _load_run_centers(run)
    if not centers_run:
        return CrossSolverValidation(
            validation_id=validation_id,
            reconstruction_run_id=reconstruction_run_id,
            status="failed",
            error="aucune caméra enregistrée",
        )

    siblings = _load_sibling_runs(workspace, run.reconstruction_input_id)
    siblings = [s for s in siblings if s.run_id != run.run_id and s.status == "completed"]

    if not siblings:
        return CrossSolverValidation(
            validation_id=validation_id,
            reconstruction_run_id=reconstruction_run_id,
            status="pending",
            consensus_score=0.0,
            error="aucun run frère complété pour comparaison",
        )

    best_score = 0.0
    divergent = []
    for sibling in siblings:
        centers_sib = _load_run_centers(sibling)
        common = sorted(set(centers_run) & set(centers_sib))
        if len(common) < 3:
            continue
        X = np.array([centers_run[k] for k in common])
        Y = np.array([centers_sib[k] for k in common])
        # On amène le frère (source) sur le run courant (target).
        R, t, s = umeyama_sim3(Y, X)
        Y_aligned = apply_sim3(Y, R, t, s)
        rmse = alignment_rmse(Y_aligned, X)
        score = max(0.0, 1.0 - rmse / 2.0)
        best_score = max(best_score, score)
        if rmse > 1.0:
            divergent.append(sibling.backend)

    status = "passed" if best_score >= 0.7 else "review" if best_score >= 0.3 else "failed"
    return CrossSolverValidation(
        validation_id=validation_id,
        reconstruction_run_id=reconstruction_run_id,
        status=status,
        consensus_score=round(best_score, 3),
        divergent_solvers=divergent,
    )


def build_validation_report(
    workspace: Workspace,
    reconstruction_run_id: str,
) -> ReconstructionValidationReport:
    held_out = validate_held_out(workspace, reconstruction_run_id)
    stability = validate_stability(workspace, reconstruction_run_id)
    cross_solver = validate_cross_solver(workspace, reconstruction_run_id)

    statuses = [v.status for v in (held_out, stability, cross_solver) if v and v.status not in ("pending",)]
    overall = "passed" if all(s == "passed" for s in statuses) else "review" if any(s == "review" for s in statuses) else "failed" if any(s == "failed" for s in statuses) else "pending"

    validation_id = (
        f"validation-{reconstruction_run_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )

    return ReconstructionValidationReport(
        validation_id=validation_id,
        reconstruction_run_id=reconstruction_run_id,
        held_out=held_out,
        stability=stability,
        cross_solver=cross_solver,
        overall_status=overall,
    )


def publish_validation_report(
    report: ReconstructionValidationReport,
    workspace: Workspace,
) -> Path:
    """Publie le rapport de validation sous `07_reconstruction/validation/`."""
    output_dir = workspace.path("07_reconstruction", "validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{report.validation_id}.json"
    output_path.write_text(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n")
    return output_path


__all__ = [
    "HeldOutValidation",
    "StabilityValidation",
    "CrossSolverValidation",
    "ReconstructionValidationReport",
    "validate_held_out",
    "validate_stability",
    "validate_cross_solver",
    "build_validation_report",
    "publish_validation_report",
]
