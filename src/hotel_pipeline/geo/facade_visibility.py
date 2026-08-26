"""Preuve de visibilité pixel par pixel pour les texels de façade.

Ce module remplace la chaîne « projeter puis remplir les trous » par une
chaîne « observer puis refuser ce qui n'est pas observé ». Un atlas troué
est le résultat attendu ; le proxy mesuré reprend sa place sous les trous.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..logging import get_logger
from ..schemas.canonical_states import TexelStatus

log = get_logger("facade-visibility")

TEXEL_M_FACADE = 0.12
MIN_PIXELS_PER_M = 2.0
MAX_INCIDENCE_DEG = 65.0
PROXY_DEPTH_TOLERANCE_M = 0.25
LIDAR_CLASSES = (3, 4, 5)
LIDAR_OCCLUSION_MARGIN_M = 1.5
POSE_MAX_ERROR_M = 0.5
POSE_MAX_ERROR_PX = 12


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
    facade_face_ids: set[int] | None = None,
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

    # Problème 43 : échantillonner sur toute l'étendue, indépendamment du sens
    x_min_px = max(0, int(np.floor(min(screen[0, 0], screen[1, 0]))))
    x_max_px = min(width - 1, int(np.ceil(max(screen[0, 0], screen[1, 0]))))
    if x_max_px < x_min_px:
        return float("inf"), float("inf"), 0

    dx = screen[1, 0] - screen[0, 0]
    dy = screen[1, 1] - screen[0, 1]
    dz = screen_z[1] - screen_z[0]
    errors_px = []
    for x_px in range(x_min_px, x_max_px + 1):
        if abs(dx) < 1e-6:
            t = 0.5
        else:
            t = (x_px - screen[0, 0]) / dx
        t = max(0.0, min(1.0, t))
        y_proj = screen[0, 1] + t * dy
        z_proj = float(screen_z[0] + t * dz)
        y_proj_int = int(round(y_proj))
        if not (0 <= y_proj_int < height):
            continue

        # Problèmes 42 et 44 : vérifier que le pixel proxy appartient à cette façade
        if proxy_depth is not None and facade_face_ids is not None:
            hit_depth, hit_fid = proxy_depth.hit(x_px, y_proj_int)
            if hit_fid is not None and hit_fid >= 0 and hit_fid not in facade_face_ids:
                continue
            if hit_depth != float("inf") and hit_depth < z_proj - PROXY_DEPTH_TOLERANCE_M:
                continue

        # Problème 42 : comparer au masque observé autour de la projection,
        # pas au pixel le plus haut de la colonne
        col_mask = building_mask[:, x_px]
        if not col_mask.any():
            continue
        mask_ys = np.where(col_mask)[0]
        closest_y = int(mask_ys[np.argmin(np.abs(mask_ys - y_proj))])
        errors_px.append(abs(y_proj - closest_y))

    if not errors_px:
        return float("inf"), float("inf"), 0

    median_px = float(np.median(errors_px))
    local_gsd = _estimate_local_gsd(camera, plane)
    error_m = median_px * local_gsd if local_gsd > 0 else float("inf")
    return median_px, error_m, len(errors_px)


def _estimate_local_gsd(camera, plane) -> float:
    """GSD local par jacobien numérique de la projection au centre de la façade.

    Remplace le calcul ponctuel par une estimation de la déformation pixel/mètre
    dans les directions (u, z) du plan, ce qui rend le GSD insensible à un
    déplacement artificiel du point d'échantillonnage.
    """
    eps = 1e-3
    u = plane.length_m * 0.5
    v_norm = 0.5
    p0 = plane.point(u, v_norm)
    p_u = plane.point(u + eps, v_norm)
    z_abs = v_norm * plane.top_z(u)
    dz = eps
    v_norm_z = min(1.0, v_norm + dz / max(plane.top_z(u), 1e-6))
    p_z = plane.point(u, v_norm_z)

    pts = np.array([p0, p_u, p_z])
    screen, _ = camera.project(pts)
    if screen is None or screen.shape[0] < 3:
        to_cam = np.asarray(camera.position, dtype=np.float64) - p0
        dist = float(np.linalg.norm(to_cam))
        focal = getattr(camera, "f", None)
        if dist < 1e-6 or not focal:
            return 0.12
        return dist / max(focal, 1e-6)

    dx_du = (screen[1, 0] - screen[0, 0]) / eps
    dy_du = (screen[1, 1] - screen[0, 1]) / eps
    dx_dz = (screen[2, 0] - screen[0, 0]) / dz
    dy_dz = (screen[2, 1] - screen[0, 1]) / dz

    jac_x = max(math.sqrt(dx_du * dx_du + dy_du * dy_du), 1e-6)
    jac_z = max(math.sqrt(dx_dz * dx_dz + dy_dz * dy_dz), 1e-6)
    gsd = 1.0 / min(jac_x, jac_z)
    return max(0.005, min(gsd, 0.5))


def local_facade_reprojection_gate(
    measurements: Sequence[tuple[str, str, float, float, int]],
    *,
    max_error_px: float = POSE_MAX_ERROR_PX,
    max_error_m: float = POSE_MAX_ERROR_M,
) -> dict:
    """Require every measured (facade, view) pair to satisfy local alignment."""
    rows = []
    for facade_id, view_id, error_px, error_m, samples in measurements:
        passed = samples > 0 and error_px <= max_error_px and error_m <= max_error_m
        rows.append({
            "facade_id": facade_id, "view_id": view_id,
            "error_px": float(error_px), "error_m": float(error_m),
            "samples": int(samples), "passed": passed,
        })
    return {
        "status": "measured" if rows else "unavailable",
        "passed": bool(rows) and all(row["passed"] for row in rows),
        "measurements": rows,
        "max_error_px": max_error_px,
        "max_error_m": max_error_m,
    }


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
    "local_facade_reprojection_gate",
]
