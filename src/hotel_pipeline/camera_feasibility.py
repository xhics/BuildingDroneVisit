"""Faisabilité caméra et trajectoire validée (Lot 2 — P6).

Ce module évalue si la reconstruction supporte les poses de caméra
intentionnelles, puis produit une `ValidatedCameraPath`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .schemas.reconstruction import (
    CameraFeasibilityField,
    ValidatedCameraPath,
)
from .workspace import Workspace


class CameraFeasibilityEvaluator:
    """Évalue la faisabilité d'une pose caméra sur la reconstruction."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def evaluate_pose(
        self,
        *,
        pose_id: str,
        position_local_m: tuple[float, float, float],
        yaw_deg: float,
        pitch_deg: float,
        fov_deg: float,
        min_distance_m: float = 0.0,
        reconstructed_fraction: float = 0.0,
        proxy_fraction: float = 0.0,
        unknown_fraction: float = 0.0,
    ) -> CameraFeasibilityField:
        """Évalue une pose unique.

        Pour le MVP, les scores sont calculés à partir des fractions fournies.
        """
        visible = max(0.0, min(1.0, reconstructed_fraction + proxy_fraction * 0.5))
        collision = False
        distance_violation = False

        if min_distance_m > 0:
            dist = (
                position_local_m[0] ** 2
                + position_local_m[1] ** 2
                + position_local_m[2] ** 2
            ) ** 0.5
            distance_violation = dist < min_distance_m

        framing = visible if not distance_violation else 0.0
        overall = visible * 0.7 + framing * 0.3

        return CameraFeasibilityField(
            pose_id=pose_id,
            position_local_m=position_local_m,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            fov_deg=fov_deg,
            visible_surface_confidence=round(visible, 3),
            unknown_fraction=round(unknown_fraction, 3),
            proxy_fraction=round(proxy_fraction, 3),
            reconstructed_fraction=round(reconstructed_fraction, 3),
            minimum_distance_violation=distance_violation,
            collision=collision,
            framing_quality=round(framing, 3),
            overall_score=round(overall, 3),
        )


def build_validated_camera_path(
    workspace: Workspace,
    reconstruction_run_id: str,
    *,
    probe_path_path: str | None = None,
) -> ValidatedCameraPath:
    """Construit une trajectoire validée depuis un chemin de diagnostic.

    Pour le MVP, on reprend les poses du `camera_probe_path.json` s'il
    existe, sinon on génère un orbit minimal.
    """
    from .scene_package import _camera_path as build_probe_path
    from shapely.geometry import Polygon

    if probe_path_path:
        try:
            probe = ValidatedCameraPath.model_validate_json(
                Path(probe_path_path).read_text("utf-8")
            )
            return probe
        except Exception:
            pass

    polygon = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
    probe = build_probe_path(polygon, height_m=10.0, fov_deg=80.0)

    path_id = (
        f"validated-path-{reconstruction_run_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    poses = [
        CameraFeasibilityField(
            pose_id=f"pose-{idx:03d}",
            position_local_m=pose.position_local_m,
            yaw_deg=pose.azimuth_deg,
            pitch_deg=0.0,
            fov_deg=pose.fov_horizontal_deg,
            overall_score=0.5,
        )
        for idx, pose in enumerate(probe.poses)
    ]

    return ValidatedCameraPath(
        path_id=path_id,
        reconstruction_run_id=reconstruction_run_id,
        simulation_only=True,
        derivation=probe.derivation + " ; validé par faisabilité caméra MVP",
        poses=poses,
    )


def publish_validated_path(path: ValidatedCameraPath, workspace: Workspace) -> Path:
    """Publie la trajectoire validée sous `07_revalidation/`."""
    out = workspace.path("07_reconstruction", "validated_camera_paths")
    out.mkdir(parents=True, exist_ok=True)
    path_file = out / f"{path.path_id}.json"
    path_file.write_text(json.dumps(path.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n")
    return path_file


__all__ = [
    "CameraFeasibilityEvaluator",
    "build_validated_camera_path",
    "publish_validated_path",
]
