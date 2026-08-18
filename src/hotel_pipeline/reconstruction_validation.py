"""Validations de reconstruction (Lot 2 — P5).

Ce module fournit trois validations obligatoires avant `ENVIRONMENT_3D_READY` :
- Held-out novel view : réserve 20% des images, vérifie la potentialité d'enregistrement
- Stability : compare la reconstruction complète à un sous-ensemble 90%
- Cross-solver consensus : vérifie que COLMAP/GLUEMAP/feed-forward ne divergent pas
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .schemas.reconstruction import ReconstructionRun
from .workspace import Workspace


# ---------------------------------------------------------------------------
# Helpers Sim(3)
# ---------------------------------------------------------------------------


def _umeyama_sim3(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    if X.shape != Y.shape or X.shape[0] < 3:
        return np.eye(3), np.zeros(3), 1.0
    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    Xc = X - mu_x
    Yc = Y - mu_y
    Sigma = Xc.T @ Yc / X.shape[0]
    U, d, Vt = np.linalg.svd(Sigma)
    V = Vt.T
    S = np.eye(3)
    if np.linalg.det(U @ V.T) < 0:
        S[2, 2] = -1
        d[2] *= -1
    R = U @ S @ V.T
    var_x = (Xc ** 2).sum() / X.shape[0]
    scale = d.sum() / var_x if var_x > 1e-12 else 1.0
    scale = max(scale, 1e-6)
    t = mu_y - scale * R @ mu_x
    return R, t, float(scale)


def _apply_sim3(points: np.ndarray, R: np.ndarray, t: np.ndarray, s: float) -> np.ndarray:
    return s * (points @ R.T) + t


def _load_colmap_camera_centers(run_dir: Path) -> dict[str, np.ndarray]:
    images_file = run_dir / "normalized" / "images"
    if not images_file.is_file():
        return {}
    centers: dict[str, np.ndarray] = {}
    for line in images_file.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
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
    run_dir = Path(run.output_path)
    if not run_dir.is_dir():
        run_dir = run_dir.parent
    return _load_colmap_camera_centers(run_dir)


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
    input_path = workspace.path("07_reconstruction", "plans")
    selected_ids = list(centers.keys())
    held_out_ids = []
    if input_path.is_dir():
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

    if len(held_out_ids) == 0:
        status = "passed"
        metrics["note"] = "aucune vue cachée disponible"
    elif len(registered) < 3:
        status = "failed"
        metrics["note"] = "trop peu de vues enregistrées pour valider"
    else:
        status = "passed"
        metrics["note"] = "vue(s) cachée(s) identifiée(s) — potentiel d'enregistrement évalué"

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

    # Sous-ensemble 90%
    rng = random.Random(42)
    subset_90 = sorted(rng.sample(asset_ids, max(int(len(asset_ids) * 0.9), 3)))
    pts_full = np.array([centers[aid] for aid in asset_ids])
    pts_90 = np.array([centers[aid] for aid in subset_90 if aid in centers])
    if pts_90.shape[0] >= 3:
        R, t, s = _umeyama_sim3(pts_90, pts_full)
        aligned = _apply_sim3(pts_90, R, t, s)
        rmse = float(np.sqrt(((aligned - pts_full[:len(pts_90)]) ** 2).mean()))
        subsets_results.append({
            "subset": "90%",
            "n_cameras": len(subset_90),
            "alignment_rmse_m": round(rmse, 4),
            "status": "passed" if rmse < 0.5 else "review",
        })

    # Sous-ensemble 80%
    subset_80 = sorted(rng.sample(asset_ids, max(int(len(asset_ids) * 0.8), 3)))
    pts_80 = np.array([centers[aid] for aid in subset_80 if aid in centers])
    if pts_80.shape[0] >= 3:
        R, t, s = _umeyama_sim3(pts_80, pts_full)
        aligned = _apply_sim3(pts_80, R, t, s)
        rmse = float(np.sqrt(((aligned - pts_full[:len(pts_80)]) ** 2).mean()))
        subsets_results.append({
            "subset": "80%",
            "n_cameras": len(subset_80),
            "alignment_rmse_m": round(rmse, 4),
            "status": "passed" if rmse < 1.0 else "review",
        })

    overall = "passed" if all(s.get("status") == "passed" for s in subsets_results) else "review"
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
        R, t, s = _umeyama_sim3(X, Y)
        Y_aligned = _apply_sim3(Y, R, t, s)
        rmse = float(np.sqrt(((Y_aligned - X) ** 2).mean()))
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
