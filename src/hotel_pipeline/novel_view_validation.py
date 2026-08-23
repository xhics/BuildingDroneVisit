"""Validation novel-view sur vues cachées (Lot 2 — Porte C).

Cette porte ne vérifie pas une moyenne : elle détecte l'hallucination. Le
plan de vues cachées (`HoldoutPlan`) est une **stratégie**, pas un pourcentage
fixe. Sur 200 vues, 20 % va bien ; sur une façade à trois vues indépendantes,
retirer 20 % peut supprimer l'unique observation qui rend le problème
reconstructible. `preserve_reconstructibility=True` interdit donc de rendre
l'ensemble d'entraînement non reconstructible.

Mesurer l'hallucination exige de **rendre** la scène depuis la pose d'une vue
cachée et de comparer ce rendu à la photographie réelle (SSIM, LPIPS, inliers,
IoU de silhouette). Tant qu'aucun moteur de rendu dense n'est raccordé, ces
métriques ne sont pas calculables — et une proximité de centres caméra n'en est
pas un substitut : un modèle qui hallucine place ses caméras tout aussi près
les unes des autres. La porte retourne donc `metrics_measured=False` avec un
motif, ce que `fidelity_gate` traduit en `INSUFFICIENT_EVIDENCE` plutôt qu'en
PASS ou en FAIL.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .sparse_reprojection import measure_held_out
from .schemas.reconstruction import (
    HoldoutPlan,
    HoldoutStrategy,
    NovelViewCriteria,
    NovelViewValidationGate,
)
from .workspace import Workspace


def _load_run(workspace: Workspace, run_id: str):  # noqa: ANN202
    from .schemas.reconstruction import ReconstructionRun

    path = workspace.path("07_reconstruction", "runs", f"{run_id}.json")
    if not path.is_file():
        return None
    try:
        return ReconstructionRun.model_validate_json(path.read_text("utf-8"))
    except Exception:
        return None


def _load_run_centers(workspace: Workspace, run_id: str) -> dict[str, np.ndarray]:
    """Centres de caméra (asset_id -> XYZ) depuis la sortie COLMAP normalisée."""
    from .reconstruction_consensus import _load_colmap_camera_centers  # type: ignore

    run = _load_run(workspace, run_id)
    if run is None or not run.output_path:
        return {}
    run_dir = Path(run.output_path)
    if not run_dir.is_dir():
        run_dir = run_dir.parent
    return _load_colmap_camera_centers(run_dir)


def _viewpoint_of(workspace: Workspace, asset_id: str) -> str:
    try:
        from .schemas import AssetManifest

        assets = AssetManifest.model_validate_json(
            workspace.assets_path.read_text("utf-8")
        )
        by_id = {a.id: a for a in assets.assets}
        a = by_id.get(asset_id)
        if a is not None and a.viewpoint_cluster:
            return a.viewpoint_cluster
    except Exception:
        pass
    return asset_id


def _select_held_out(
    workspace: Workspace,
    run_id: str,
    plan: HoldoutPlan,
    selected_asset_ids: list[str],
    registered_asset_ids: set[str],
) -> tuple[list[str], list[str]]:
    """Retourne (train_ids, held_out_ids) selon la stratégie.

    `preserve_reconstructibility` garantit que `train_ids` reste
    reconstructible : on ne retire jamais la dernière vue d'un composant,
    ni une vue unique sur une cible.
    """
    import random

    rng = random.Random(42)

    if plan.strategy is HoldoutStrategy.LEAVE_ONE_VIEWPOINT_OUT:
        # Retire une vue par groupe de viewpoints, jamais la dernière d'un groupe.
        held_out: list[str] = []
        by_vp: dict[str, list[str]] = {}
        for aid in sorted(registered_asset_ids):
            vp = _viewpoint_of(workspace, aid)
            by_vp.setdefault(vp, []).append(aid)
        for vp in sorted(by_vp):
            members = sorted(by_vp[vp])
            if len(members) > 1:
                held_out.append(rng.choice(members))
    elif plan.strategy is HoldoutStrategy.K_FOLD:
        ids = sorted(registered_asset_ids)
        k = max(2, int(round(1.0 / max(plan.benchmark_profile, 0.1))))
        fold = rng.randrange(k)
        held_out = [ids[i] for i in range(len(ids)) if i % k == fold]
    else:  # STRATIFIED_BY_TARGET
        ids = sorted(registered_asset_ids)
        n = int(len(ids) * plan.benchmark_profile)
        held_out = sorted(rng.sample(ids, max(0, min(n, len(ids) - 1))))

    train = [a for a in selected_asset_ids if a not in set(held_out)]

    # Un ensemble d'entraînement sous 3 vues n'est plus reconstructible :
    # l'ablation détruirait la mesure qu'elle prétend faire.
    if plan.preserve_reconstructibility and len(train) < 3:
        train = list(selected_asset_ids)
        held_out = []

    return train, sorted(held_out)


def _dense_renderer_available() -> bool:
    """Un moteur de rendu dense est-il raccordé ?

    Le rendu novel-view exige un modèle dense (splats gaussiens, maillage
    texturé, NeRF) et un rasteriseur. Aucun n'est raccordé à ce stade — d'où
    la mesure creuse ci-dessous, qui répond à la même question sur les seules
    observations disponibles.
    """
    return False


def build_novel_view_gate(
    workspace: Workspace,
    reconstruction_run_id: str,
    holdout_plan: HoldoutPlan | None = None,
    selected_asset_ids: list[str] | None = None,
) -> NovelViewValidationGate:
    """Construit la porte novel-view (Gate C) pour un run de reconstruction.

    Le plan de vues cachées est calculé réellement — c'est lui qui pilotera le
    rendu comparatif. Les métriques restent non mesurées tant qu'aucun moteur
    de rendu n'est disponible, et la porte le déclare explicitement.
    """
    if holdout_plan is None:
        holdout_plan = HoldoutPlan(strategy=HoldoutStrategy.LEAVE_ONE_VIEWPOINT_OUT)

    run = _load_run(workspace, reconstruction_run_id)
    if run is None:
        raise FileNotFoundError(f"run introuvable : {reconstruction_run_id}")

    registered = set(_load_run_centers(workspace, reconstruction_run_id).keys())
    if selected_asset_ids is None:
        selected_asset_ids = sorted(registered)

    train_ids, held_out_ids = _select_held_out(
        workspace, reconstruction_run_id, holdout_plan, selected_asset_ids, registered
    )

    criteria = NovelViewCriteria()

    if not registered:
        reason = (
            "aucune caméra enregistrée pour ce run : aucune pose de vue cachée "
            "à rendre"
        )
    elif not held_out_ids:
        reason = (
            "aucune vue cachée retenue : le corpus est trop petit pour en "
            "retirer une sans le rendre non reconstructible"
        )
    elif not _dense_renderer_available():
        reason = (
            "aucun moteur de rendu dense raccordé : SSIM, LPIPS, inliers et IoU "
            "de silhouette exigent un rendu comparé aux vues cachées"
        )
    else:  # pragma: no cover - branche activée avec un moteur de rendu
        reason = None

    # --- mesure creuse : les observations tiennent lieu de photographie ----
    #
    # Le rendu dense reste absent, mais la question de la porte C — « le
    # modèle prédit-il une vue qu'il n'a pas vue ? » — se pose déjà sur les
    # observations 2D de la reconstruction : un point fabriqué pour satisfaire
    # l'entraînement se projette loin de l'endroit où la vue cachée l'a vu.
    if reason is None or "moteur de rendu" in reason:
        run_dir = _run_directory(workspace, reconstruction_run_id)
        measurement = (
            measure_held_out(run_dir, held_out_ids, train_ids)
            if run_dir is not None
            else None
        )
        if measurement is not None:
            return NovelViewValidationGate(
                holdout_plan=holdout_plan,
                feature_inliers=_clamp(measurement["feature_inliers"]),
                silhouette_iou=_clamp(measurement["silhouette_iou"]),
                reprojection_px=max(0.0, measurement["reprojection_px"]),
                structural_similarity=_clamp(measurement["structural_similarity"]),
                # SSIM et LPIPS exigent une image rendue : ils restent aux
                # défauts du schéma, et rien ne prétend les avoir mesurés.
                pass_criteria=criteria,
                metrics_measured=True,
                unmeasured_reason=(
                    "mesure creuse : reprojection sur observations ; "
                    "SSIM et LPIPS exigent un rendu dense"
                ),
                held_out_asset_ids=held_out_ids,
            )
        reason = (
            "reprojection creuse impossible : ni observations 2D ni "
            "intrinsèques exploitables dans la sortie du run"
        )

    # Aucune métrique inventée : les défauts du schéma restent en place et
    # `metrics_measured=False` empêche toute interprétation en PASS.
    return NovelViewValidationGate(
        holdout_plan=holdout_plan,
        pass_criteria=criteria,
        metrics_measured=False,
        unmeasured_reason=reason,
        held_out_asset_ids=held_out_ids,
    )


def _clamp(value: float) -> float:
    """Ramène une part dans [0, 1] : le schéma le contraint."""
    return float(min(1.0, max(0.0, value)))


def _run_directory(workspace: Workspace, run_id: str) -> Path | None:
    from .reconstruction_consensus import resolve_model_dir

    run = _load_run(workspace, run_id)
    if run is None or not run.output_path:
        return None
    return resolve_model_dir(run.output_path)


def publish_novel_view_gate(gate: NovelViewValidationGate, workspace: Workspace) -> Path:
    """Publie la porte sous `07_reconstruction/novel_view/`."""
    output_dir = workspace.path("07_reconstruction", "novel_view")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_ref = gate.holdout_plan.strategy.value if gate.holdout_plan else "gate"
    path = output_dir / f"novel_view_{run_ref}.json"
    path.write_text(json.dumps(gate.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n")
    return path


__all__ = [
    "build_novel_view_gate",
    "publish_novel_view_gate",
]
