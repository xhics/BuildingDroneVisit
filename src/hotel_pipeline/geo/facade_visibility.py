"""Preuve de visibilité pixel par pixel pour les texels de façade.

Ce module remplace la chaîne « projeter puis remplir les trous » par une
chaîne « observer puis refuser ce qui n'est pas observé ». Un atlas troué
est le résultat attendu ; le proxy mesuré reprend sa place sous les trous.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..logging import get_logger

log = get_logger("facade-visibility")

TEXEL_M_FACADE = 0.12
MIN_PIXELS_PER_M = 2.0
MAX_INCIDENCE_DEG = 65.0
PROXY_DEPTH_TOLERANCE_M = 0.25
LIDAR_CLASSES = (3, 4, 5)
LIDAR_OCCLUSION_MARGIN_M = 1.5
POSE_MAX_ERROR_M = 0.5
POSE_MAX_ERROR_PX = 12


class TexelStatus(str, Enum):
    OBSERVED_CONSENSUS = "OBSERVED_CONSENSUS"
    OBSERVED_SINGLE = "OBSERVED_SINGLE"
    REJECTED_DISAGREEMENT = "REJECTED_DISAGREEMENT"
    REJECTED_OCCLUDED = "REJECTED_OCCLUDED"
    REJECTED_SEMANTIC = "REJECTED_SEMANTIC"
    REJECTED_POSE = "REJECTED_POSE"
    REJECTED_RESOLUTION = "REJECTED_RESOLUTION"
    UNOBSERVED = "UNOBSERVED"

    @property
    def is_observed(self) -> bool:
        return self in (TexelStatus.OBSERVED_CONSENSUS, TexelStatus.OBSERVED_SINGLE)


REJECTION_ORDER = (
    "semantic_absent",
    "semantic_not_building",
    "occluded_by_proxy",
    "occluded_by_lidar",
    "pose_error",
    "resolution",
    "incidence",
    "behind_camera",
    "outside_image",
)


@dataclass(frozen=True)
class FacadeTexelCandidate:
    source_view: str
    col: int
    row: int
    u_m: float
    v_norm: float
    pixel_xy: tuple[int, int]
    wall_depth_m: float
    proxy_first_hit_depth_m: float
    proxy_first_hit_face_id: int | None
    lidar_depth_m: float | None
    semantic_visible: bool
    semantic_source: str | None
    pose_error_px: float | None
    pose_error_m: float | None
    local_gsd_m: float | None
    incidence_deg: float | None
    sharpness: float | None
    colour_rgb: tuple[float, float, float] | None
    rejections: tuple[str, ...] = ()


@dataclass
class ProxyDepth:
    """Z-buffer du proxy 3D pour une vue donnée."""

    width: int
    height: int
    depth: np.ndarray
    face_id_map: np.ndarray | None = None

    def hit(self, x: int, y: int) -> tuple[float, int | None]:
        if 0 <= y < self.height and 0 <= x < self.width:
            d = float(self.depth[y, x])
            fid = int(self.face_id_map[y, x]) if self.face_id_map is not None else None
            return d, fid
        return float("inf"), None

    @classmethod
    def render(
        cls,
        camera,
        triangles: Sequence[np.ndarray],
        face_ids: Sequence[int],
        width: int,
        height: int,
    ) -> ProxyDepth:
        depth = np.full((height, width), np.inf, dtype=np.float64)
        face_map = np.full((height, width), -1, dtype=np.int32)

        for tri, fid in zip(triangles, face_ids):
            tri = np.asarray(tri, dtype=np.float64)
            if tri.shape != (3, 3):
                continue
            screen, z = camera.project(tri)
            if screen is None:
                continue
            xs = screen[:, 0]
            ys = screen[:, 1]
            x0 = int(np.floor(xs.min()))
            x1 = int(np.ceil(xs.max()))
            y0 = int(np.floor(ys.min()))
            y1 = int(np.ceil(ys.max()))
            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(width - 1, x1)
            y1 = min(height - 1, y1)
            if x1 < x0 or y1 < y0:
                continue

            s0, s1, s2 = screen
            z0, z1, z2 = z
            den = (s1[0] - s0[0]) * (s2[1] - s0[1]) - (s2[0] - s0[0]) * (s1[1] - s0[1])
            if abs(den) < 1e-8:
                continue
            inv_den = 1.0 / den

            for py in range(y0, y1 + 1):
                for px in range(x0, x1 + 1):
                    p = np.array([px + 0.5, py + 0.5], dtype=np.float64)
                    v1 = (p[0] - s0[0]) * (s2[1] - s0[1]) - (s2[0] - s0[0]) * (p[1] - s0[1])
                    v2 = (s1[0] - s0[0]) * (p[1] - s0[1]) - (p[0] - s0[0]) * (s1[1] - s0[1])
                    u = v1 * inv_den
                    v = v2 * inv_den
                    w = 1.0 - u - v
                    if u < -1e-4 or v < -1e-4 or w < -1e-4:
                        continue
                    if z0 > 0 and z1 > 0 and z2 > 0:
                        inv_z = w * (1.0 / z0) + u * (1.0 / z1) + v * (1.0 / z2)
                        z_pixel = 1.0 / inv_z
                    else:
                        z_pixel = w * z0 + u * z1 + v * z2
                    if z_pixel < depth[py, px]:
                        depth[py, px] = z_pixel
                        face_map[py, px] = fid

        return cls(width=width, height=height, depth=depth, face_id_map=face_map)


@dataclass
class LidarOcclusion:
    """Carte de profondeur LiDAR, splattée, pour détecter les occulteurs."""

    depth: np.ndarray
    valid: np.ndarray

    @classmethod
    def from_window(
        cls,
        laz_window,
        camera,
        scene_origin_xyz: np.ndarray,
        focal_px: float,
        width: int,
        height: int,
        margin_m: float = LIDAR_OCCLUSION_MARGIN_M,
    ) -> LidarOcclusion | None:
        if laz_window is None or len(laz_window) == 0:
            return None

        mask = np.isin(laz_window.classification, list(LIDAR_CLASSES))
        if not mask.any():
            return None

        x = laz_window.x[mask]
        y = laz_window.y[mask]
        z = laz_window.z[mask]

        points = np.column_stack([x, y, z]) - scene_origin_xyz

        screen, depth = camera.project(points)
        if screen is None:
            return None

        depth_map = np.full((height, width), np.inf, dtype=np.float64)
        valid = np.zeros((height, width), dtype=bool)

        for i in range(len(points)):
            px, py = round(screen[i, 0]), round(screen[i, 1])
            if not (0 <= px < width and 0 <= py < height):
                continue
            d = float(depth[i])
            if not math.isfinite(d) or d <= 0.0:
                continue
            r_px = max(1, min(6, math.ceil(focal_px * 0.5 / max(d, 1e-6))))
            for dy in range(-r_px, r_px + 1):
                for dx in range(-r_px, r_px + 1):
                    if dx * dx + dy * dy > r_px * r_px:
                        continue
                    sy, sx = py + dy, px + dx
                    if 0 <= sy < height and 0 <= sx < width and d < depth_map[sy, sx]:
                        depth_map[sy, sx] = d
                        valid[sy, sx] = True

        return cls(depth=depth_map, valid=valid)


def measure_facade_alignment(
    camera,
    plane,
    proxy_depth: ProxyDepth | None = None,
    building_mask: np.ndarray | None = None,
) -> tuple[float, float, int]:
    """Écart médian entre l'arête haute projetée du mur et le masque building.

    Retourne (error_px, error_m, columns_used).
    """
    if building_mask is None:
        return float("inf"), float("inf"), 0
    height, width = building_mask.shape
    top_edge = np.array([
        plane.point(0.0, 1.0),
        plane.point(plane.length_m, 1.0),
    ])
    screen, screen_z = camera.project(top_edge)
    if screen is None:
        return float("inf"), float("inf"), 0

    x0, x1 = max(0, int(np.floor(screen[0, 0]))), min(width - 1, int(np.ceil(screen[1, 0])))
    if x1 < x0:
        return float("inf"), float("inf"), 0

    dx = screen[1, 0] - screen[0, 0]
    dy = screen[1, 1] - screen[0, 1]
    dz = screen_z[1] - screen_z[0]
    errors_px = []
    for x in range(x0, x1 + 1):
        if dx == 0:
            y_proj = screen[0, 1]
            z_proj = float(screen_z[0])
        else:
            t = (x - screen[0, 0]) / dx
            y_proj = screen[0, 1] + t * dy
            z_proj = float(screen_z[0] + t * dz)
        y_proj_int = round(y_proj)
        if not (0 <= y_proj_int < height):
            continue
        col_mask = building_mask[:, x]
        if not col_mask.any():
            continue
        if proxy_depth is not None:
            hit_depth, _ = proxy_depth.hit(x, y_proj_int)
            if hit_depth != float("inf") and hit_depth < z_proj - PROXY_DEPTH_TOLERANCE_M:
                continue
        y_mask = round(np.argmax(col_mask))
        errors_px.append(abs(y_proj - y_mask))

    if not errors_px:
        return float("inf"), float("inf"), 0

    median_px = float(np.median(errors_px))
    local_gsd = _estimate_local_gsd(camera, plane)
    error_m = median_px * local_gsd if local_gsd > 0 else float("inf")
    return median_px, error_m, len(errors_px)


def _estimate_local_gsd(camera, plane) -> float:
    centre = plane.point(plane.length_m * 0.5, 0.5)
    to_cam = np.asarray(camera.position, dtype=np.float64) - centre
    dist = float(np.linalg.norm(to_cam))
    if dist < 1e-6:
        return 0.12
    focal = getattr(camera, "f", None)
    if not focal:
        return 0.12
    return dist / max(focal, 1e-6)


def admit(
    candidate: FacadeTexelCandidate,
    policy: dict | None = None,
) -> tuple[bool, str | None, str | None]:
    """Point unique de décision, avec codes de refus ordonnés."""
    policy = policy or {}

    def check(code: str, condition: bool) -> bool:
        return condition

    if check("semantic_absent", not candidate.semantic_visible and candidate.semantic_source is None):
        return False, "semantic_absent", None
    if check("semantic_not_building", not candidate.semantic_visible and candidate.semantic_source == "not_building"):
        return False, "semantic_not_building", None
    if check("occluded_by_proxy", candidate.proxy_first_hit_face_id is not None and candidate.wall_depth_m > candidate.proxy_first_hit_depth_m + PROXY_DEPTH_TOLERANCE_M):
        return False, "occluded_by_proxy", None
    if check("occluded_by_lidar", candidate.lidar_depth_m is not None and candidate.wall_depth_m > candidate.lidar_depth_m + LIDAR_OCCLUSION_MARGIN_M):
        return False, "occluded_by_lidar", None
    if check("pose_error", candidate.pose_error_m is not None and candidate.pose_error_m > POSE_MAX_ERROR_M):
        return False, "pose_error", None
    if check("resolution", candidate.local_gsd_m is not None and candidate.local_gsd_m > TEXEL_M_FACADE / MIN_PIXELS_PER_M):
        return False, "resolution", None
    if check("incidence", candidate.incidence_deg is not None and candidate.incidence_deg > MAX_INCIDENCE_DEG):
        return False, "incidence", None
    if check("behind_camera", candidate.wall_depth_m <= 0.5):
        return False, "behind_camera", None

    return True, None, None


__all__ = [
    "REJECTION_ORDER",
    "FacadeTexelCandidate",
    "LidarOcclusion",
    "ProxyDepth",
    "TexelStatus",
    "admit",
    "measure_facade_alignment",
]
