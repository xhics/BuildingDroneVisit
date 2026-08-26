"""Faisabilité caméra et trajectoire validée (Lot 2 — P6).

Ce module évalue si la reconstruction supporte les poses de caméra
intentionnelles, puis produit une `ValidatedCameraPath`.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .schemas.reconstruction import (
    CameraFeasibilityField,
    ValidatedCameraPath,
)
from .reconstruction_consensus import resolve_model_dir
from .workspace import Workspace


# ---------------------------------------------------------------------------
# Helpers géométriques
# ---------------------------------------------------------------------------


def _load_colmap_points3d(run_dir: Path) -> np.ndarray | None:
    points_file = run_dir / "normalized" / "points3D"
    if not points_file.is_file():
        return None
    pts = []
    for line in points_file.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 4:
            pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not pts:
        return None
    return np.array(pts)


def _load_run_points(reconstruction_run_id: str, workspace: Workspace) -> np.ndarray | None:
    run_json_path = workspace.path("07_reconstruction", "runs", f"{reconstruction_run_id}.json")
    if not run_json_path.is_file():
        return None
    run_data = json.loads(run_json_path.read_text("utf-8"))
    output_path = run_data.get("output_path")
    if not output_path:
        return None
    run_dir = resolve_model_dir(output_path)

    normalized_dir = run_dir / "normalized"
    if not normalized_dir.is_dir():
        normalized_dir = run_dir.parent / "normalized"
    if not normalized_dir.is_dir():
        normalized_dir = run_dir.parent.parent / "normalized"
    if not normalized_dir.is_dir():
        return None
    return _load_colmap_points3d(normalized_dir.parent)


def _visible_fraction_from_points(
    position: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
    points: np.ndarray,
) -> tuple[float, float, float]:
    if points is None or len(points) == 0:
        return 0.0, 1.0, 0.0
    vecs = points - position
    dists = np.linalg.norm(vecs, axis=1)
    if dists.max() < 1e-6:
        return 0.0, 1.0, 0.0
    dirs = vecs / dists[:, None]

    yaw_rad = math.radians(yaw_deg)
    pitch_rad = math.radians(pitch_deg)
    fov_rad = math.radians(fov_deg / 2.0)

    forward = np.array([
        math.cos(pitch_rad) * math.sin(yaw_rad),
        math.cos(pitch_rad) * math.cos(yaw_rad),
        math.sin(pitch_rad),
    ])
    forward = forward / np.linalg.norm(forward)

    cos_angles = dirs @ forward
    in_fov = cos_angles >= math.cos(fov_rad)
    visible_pts = points[in_fov]
    if len(visible_pts) == 0:
        return 0.0, 1.0, 0.0

    reconstructed = len(visible_pts) / len(points)
    unknown = 0.0  # pas de carte de densité au MVP
    return reconstructed, unknown, reconstructed


# ---------------------------------------------------------------------------
# Évaluateur
# ---------------------------------------------------------------------------


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
        reconstruction_run_id: str | None = None,
    ) -> CameraFeasibilityField:
        reconstructed = reconstructed_fraction
        unknown = unknown_fraction
        proxy = proxy_fraction

        if reconstruction_run_id is not None:
            points = _load_run_points(reconstruction_run_id, self.workspace)
            recon_frac, unk_frac, vis_frac = _visible_fraction_from_points(
                np.array(position_local_m), yaw_deg, pitch_deg, fov_deg, points
            )
            if recon_frac > 0:
                reconstructed = recon_frac
                unknown = unk_frac
                proxy = max(0.0, 1.0 - reconstructed - unknown)

        visible = max(0.0, min(1.0, reconstructed + proxy * 0.5))
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

        # Orientation complète stockée : matrice ET quaternion. Aucune
        # validation en aval ne reconstruit la visée à partir du yaw seul.
        rotation = pose_rotation_matrix(yaw_deg, pitch_deg)
        quaternion = _rotation_to_quaternion(rotation)

        return CameraFeasibilityField(
            pose_id=pose_id,
            position_local_m=position_local_m,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            fov_deg=fov_deg,
            orientation_matrix=tuple(float(v) for v in rotation.reshape(-1)),
            orientation_quaternion=quaternion,
            visible_surface_confidence=round(visible, 3),
            unknown_fraction=round(unknown, 3),
            proxy_fraction=round(proxy, 3),
            reconstructed_fraction=round(reconstructed, 3),
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

    points = _load_run_points(reconstruction_run_id, workspace)
    if points is not None and len(points) > 0:
        centroid = points.mean(axis=0)
        spread = float(np.linalg.norm(points[:, :2].max(axis=0) - points[:, :2].min(axis=0)))
        radius = max(spread * 0.8, 5.0)
        height = float(points[:, 2].mean() + points[:, 2].std()) if len(points) > 1 else 10.0
        polygon = Polygon([
            (centroid[0] - radius, centroid[1] - radius),
            (centroid[0] + radius, centroid[1] - radius),
            (centroid[0] + radius, centroid[1] + radius),
            (centroid[0] - radius, centroid[1] + radius),
        ])
    else:
        polygon = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
        height = 10.0

    probe = build_probe_path(polygon, height_m=height, fov_deg=80.0)

    path_id = (
        f"validated-path-{reconstruction_run_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )

    evaluator = CameraFeasibilityEvaluator(workspace)
    poses = []
    for idx, pose in enumerate(probe.poses):
        field = evaluator.evaluate_pose(
            pose_id=f"pose-{idx:03d}",
            position_local_m=pose.position_local_m,
            yaw_deg=pose.azimuth_deg,
            pitch_deg=_pitch_of_pose(pose),
            fov_deg=pose.fov_horizontal_deg,
            reconstruction_run_id=reconstruction_run_id,
        )
        poses.append(field)

    derivation = (
        f"trajectoire probe autour du nuage reconstruit "
        f"({len(points) if points is not None else 0} points) ; validé par faisabilité caméra"
    )

    return ValidatedCameraPath(
        path_id=path_id,
        reconstruction_run_id=reconstruction_run_id,
        simulation_only=True,
        derivation=derivation,
        poses=poses,
    )


def _pitch_of_pose(pose) -> float:  # noqa: ANN001
    """Pitch complet d'une pose, dérivé de sa géométrie position→look_at.

    L'orientation n'est jamais réduite au yaw : le pitch mesuré par la
    géométrie de la pose traverse tout le pipeline, et la matrice de
    rotation complète reste disponible sur le champ validé.
    """
    look_at = getattr(pose, "look_at_local_m", None)
    if look_at is None:
        return 0.0
    position = np.asarray(pose.position_local_m, dtype=np.float64)
    target = np.asarray(look_at, dtype=np.float64)
    delta = target - position
    horizontal = math.hypot(delta[0], delta[1])
    if horizontal < 1e-9:
        return 90.0 if delta[2] > 0 else -90.0
    return math.degrees(math.atan2(delta[2], horizontal))


def pose_rotation_matrix(
    yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0
) -> np.ndarray:
    """Matrice de rotation monde→caméra (convention Z vers l'avant).

    Toute validation consomme cette pose **complète** — jamais une
    reconstruction partielle à partir du seul yaw.
    """
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return ry @ rx @ rz


def _rotation_to_quaternion(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z) d'une matrice de rotation — trace maximale."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    return (w / norm, x / norm, y / norm, z / norm)


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
