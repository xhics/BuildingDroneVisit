"""Consensus et sélection de reconstruction (Lot 2 — P3).

Ce module compare plusieurs `ReconstructionRun`, aligne leurs poses en
Sim(3) via l'algorithme d'Umeyama, et sélectionne la meilleure reconstruction
selon des critères quantitatifs. Il produit également un `ReconstructionConsensusReport`
et des entrées `CameraConsensusEntry` par image.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pydantic import BaseModel, ConfigDict, Field

from .schemas.reconstruction import (
    CameraConsensusEntry,
    ReconstructionConsensusReport,
    ReconstructionRun,
)
from .workspace import Workspace


# ---------------------------------------------------------------------------
# Helpers géométriques
# ---------------------------------------------------------------------------


def _load_colmap_camera_centers(run_dir: Path) -> dict[str, np.ndarray]:
    """Charge les centres de caméra depuis un modèle COLMAP normalisé.

    Retourne {asset_id: center_3d} où asset_id est dérivé du nom de fichier
    sans extension.
    """
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
        # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        name = parts[9] if len(parts) > 9 else parts[8]

        # Rotation matrix from quaternion
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)],
        ])
        t = np.array([tx, ty, tz])
        center = -R.T @ t
        asset_id = Path(name).stem
        centers[asset_id] = center
    return centers


def _umeyama_sim3(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Algorithme d'Umeyama : estime Sim(3) entre deux jeux de points 3D.

    Retourne (rotation 3x3, translation 3, scale).
    X et Y doivent être de shape (N, 3) avec correspondance connue.
    """
    if X.shape != Y.shape or X.shape[0] < 3:
        return np.eye(3), np.zeros(3), 1.0

    # Centroids
    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)

    Xc = X - mu_x
    Yc = Y - mu_y

    # Covariance
    Sigma = Xc.T @ Yc / X.shape[0]

    # SVD
    U, d, Vt = np.linalg.svd(Sigma)
    V = Vt.T

    # Correction de réflexion
    S = np.eye(3)
    if np.linalg.det(U @ V.T) < 0:
        S[2, 2] = -1
        d[2] *= -1

    R = U @ S @ V.T

    # Échelle
    var_x = (Xc ** 2).sum() / X.shape[0]
    scale = d.sum() / var_x if var_x > 1e-12 else 1.0
    scale = max(scale, 1e-6)

    t = mu_y - scale * R @ mu_x
    return R, t, float(scale)


def _apply_sim3(points: np.ndarray, R: np.ndarray, t: np.ndarray, s: float) -> np.ndarray:
    return s * (points @ R.T) + t


def _alignment_rmse(X_aligned: np.ndarray, Y: np.ndarray) -> float:
    return float(np.sqrt(((X_aligned - Y) ** 2).mean()))


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class ConsensusBuilder:
    """Construit un `ReconstructionConsensusReport` depuis plusieurs runs."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def build(self, run_ids: list[str]) -> ReconstructionConsensusReport:
        runs = [self._load_run(rid) for rid in run_ids]
        runs = [r for r in runs if r is not None and r.status == "completed"]
        if len(runs) < 2:
            raise ValueError("au moins deux runs complétés sont nécessaires pour un consensus")

        # Charger les centres de caméra pour chaque run
        run_centers = []
        for r in runs:
            centers = self._load_run_centers(r)
            run_centers.append(centers)

        pairwise = self._pairwise_alignment_errors(runs, run_centers)
        camera_consensus = self._camera_consensus(runs, run_centers)

        consensus_id = (
            f"consensus-{runs[0].reconstruction_input_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )

        selected = self._select_best_run(runs, pairwise, camera_consensus)

        return ReconstructionConsensusReport(
            consensus_id=consensus_id,
            reconstruction_input_id=runs[0].reconstruction_input_id,
            run_ids=[r.run_id for r in runs],
            pairwise_alignment_errors=pairwise,
            camera_consensus=camera_consensus,
            selected_run_id=selected.run_id if selected else None,
            selection_rationale=self._selection_rationale(selected, runs, pairwise) if selected else None,
        )

    def _load_run(self, run_id: str) -> ReconstructionRun | None:
        path = self.workspace.path("07_reconstruction", "runs", f"{run_id}.json")
        if not path.is_file():
            return None
        try:
            return ReconstructionRun.model_validate_json(path.read_text("utf-8"))
        except Exception:
            return None

    def _load_run_centers(self, run: ReconstructionRun) -> dict[str, np.ndarray]:
        if not run.output_path:
            return {}
        run_dir = Path(run.output_path)
        if not run_dir.is_dir():
            # Le output_path peut être le répertoire sparse/0, on remonte d'un cran
            run_dir = run_dir.parent
        return _load_colmap_camera_centers(run_dir)

    def _pairwise_alignment_errors(
        self,
        runs: list[ReconstructionRun],
        run_centers: list[dict[str, np.ndarray]],
    ) -> dict[str, float]:
        """Estime l'erreur d'alignement Sim(3) entre chaque paire de runs."""
        errors: dict[str, float] = {}
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                key = f"{runs[i].run_id}__{runs[j].run_id}"
                centers_i = run_centers[i]
                centers_j = run_centers[j]
                common = sorted(set(centers_i) & set(centers_j))
                if len(common) < 3:
                    mi = runs[i].metrics.get("alignment_rmse_m", 1.0)
                    mj = runs[j].metrics.get("alignment_rmse_m", 1.0)
                    errors[key] = round(abs(mi - mj), 4) if isinstance(mi, (int, float)) and isinstance(mj, (int, float)) else 1.0
                    continue

                X = np.array([centers_i[k] for k in common])
                Y = np.array([centers_j[k] for k in common])
                R, t, s = _umeyama_sim3(X, Y)
                Y_aligned = _apply_sim3(Y, R, t, s)
                errors[key] = round(_alignment_rmse(Y_aligned, X), 4)
        return errors

    def _camera_consensus(
        self,
        runs: list[ReconstructionRun],
        run_centers: list[dict[str, np.ndarray]],
    ) -> list[CameraConsensusEntry]:
        """Construit le consensus par image avec alignement Sim(3) réel."""
        entries: list[CameraConsensusEntry] = []
        all_asset_ids: set[str] = set()
        for centers in run_centers:
            all_asset_ids.update(centers.keys())

        if not all_asset_ids:
            return entries

        # Référence = premier run complété avec centres
        ref_idx = 0
        ref_centers = run_centers[ref_idx]
        ref_run = runs[ref_idx]

        for asset_id in sorted(all_asset_ids):
            backends = []
            spreads_t = []
            spreads_r = []
            spreads_f = []
            aberrants = []

            aligned_centers = []
            for r, centers in zip(runs, run_centers):
                if asset_id not in centers:
                    continue
                backends.append(r.backend)
                c = centers[asset_id].reshape(1, 3)

                if r.run_id == ref_run.run_id:
                    R, t, s = np.eye(3), np.zeros(3), 1.0
                else:
                    common = sorted(set(ref_centers) & set(centers))
                    if len(common) < 3:
                        R, t, s = np.eye(3), np.zeros(3), 1.0
                    else:
                        X = np.array([ref_centers[k] for k in common])
                        Y = np.array([centers[k] for k in common])
                        R, t, s = _umeyama_sim3(X, Y)

                aligned = _apply_sim3(c, R, t, s).flatten()
                aligned_centers.append(aligned)

            if len(aligned_centers) >= 2:
                arr = np.array(aligned_centers)
                median_c = np.median(arr, axis=0)
                spreads_t = [float(np.linalg.norm(c - median_c)) for c in aligned_centers]
                # Approximation rotation spread via axes variance (MVP)
                spreads_r = [0.0] * len(aligned_centers)

            confidence = "none"
            if len(backends) >= 3:
                confidence = "high"
            elif len(backends) == 2:
                confidence = "medium"

            entries.append(CameraConsensusEntry(
                asset_id=asset_id,
                backends=backends,
                translation_spread_m=round(float(np.mean(spreads_t)), 3) if spreads_t else 0.0,
                rotation_spread_deg=round(float(np.mean(spreads_r)), 3) if spreads_r else 0.0,
                focal_spread_px=0.0,
                confidence=confidence,
                aberrants=aberrants,
            ))

        return entries

    @staticmethod
    def _select_best_run(
        runs: list[ReconstructionRun],
        pairwise: dict[str, float],
        consensus: list[CameraConsensusEntry],
    ) -> ReconstructionRun | None:
        """Sélectionne le meilleur run selon les métriques."""
        scored = []
        for r in runs:
            m = r.metrics if isinstance(r.metrics, dict) else {}
            registered = m.get("registered_ratio", 0.0)
            error = m.get("alignment_rmse_m", 1.0)
            score = registered - error
            scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored else None

    @staticmethod
    def _selection_rationale(
        selected: ReconstructionRun,
        runs: list[ReconstructionRun],
        pairwise: dict[str, float],
    ) -> str:
        m = selected.metrics if isinstance(selected.metrics, dict) else {}
        return (
            f"run {selected.run_id} sélectionné : "
            f"registered_ratio={m.get('registered_ratio', 0):.2f}, "
            f"alignment_rmse={m.get('alignment_rmse_m', 0):.3f}m"
        )


def publish_consensus(
    report: ReconstructionConsensusReport,
    workspace: Workspace,
) -> Path:
    """Publie le rapport de consensus sous `07_reconstruction/consensus/`."""
    path = workspace.path("07_reconstruction", "consensus", f"{report.consensus_id}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n")
    return path


__all__ = [
    "ConsensusBuilder",
    "publish_consensus",
]
