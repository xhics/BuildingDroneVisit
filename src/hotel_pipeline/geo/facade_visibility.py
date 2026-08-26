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
    """Z-buffer du proxy 3D pour une vue donnée.

    Convention unique du pipeline : ``depth`` est un **Z espace caméra** —
    l'abscisse le long de l'axe optique — jamais une distance euclidienne,
    jamais un z-buffer GPU normalisé. Toutes les comparaisons de profondeur
    du pipeline se font dans cette même unité.
    """

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
        z_near_m: float = 0.05,
    ) -> ProxyDepth:
        """Rastérise le proxy avec clip near-plane et découpage au bord.

        Aucun sommet derrière la caméra n'atteint la projection : chaque
        triangle est découpé en espace caméra contre z = z_near (0, 1 ou 2
        triangles), puis le polygone écran est découpé contre le rectangle
        image — jamais clampé.
        """
        from ..render_engine import (
            clip_polygon_to_image,
            clip_triangle_near,
        )
        import shapely

        has_contract = hasattr(camera, "R") and hasattr(camera, "t")
        depth = np.full((height, width), np.inf, dtype=np.float64)
        face_map = np.full((height, width), -1, dtype=np.int32)

        for tri, fid in zip(triangles, face_ids):
            tri = np.asarray(tri, dtype=np.float64)
            if tri.shape != (3, 3):
                continue
            if has_contract:
                cam_tri = tri @ camera.R.T + camera.t
                for piece in clip_triangle_near(cam_tri, z_near_m):
                    safe_z = np.where(np.abs(piece[:, 2]) < 1e-9, 1e-9, piece[:, 2])
                    normalized = piece[:, :2] / safe_z[:, None]
                    try:
                        screen = np.asarray(
                            camera.img_from_cam(
                                np.column_stack([normalized, np.ones(len(piece))])
                            ),
                            dtype=float,
                        ).reshape((-1, 2))
                    except TypeError:
                        screen = np.asarray(
                            camera.img_from_cam(normalized), dtype=float
                        ).reshape((-1, 2))
                    polygon = clip_polygon_to_image(screen, width, height)
                    if len(polygon) < 3:
                        continue
                    plane_point_cam = piece[0]
                    plane_normal_cam = np.cross(piece[1] - piece[0], piece[2] - piece[0])
                    norm_len = float(np.linalg.norm(plane_normal_cam))
                    if norm_len < 1e-12:
                        continue
                    plane_normal_cam /= norm_len
                    denom_const = float(plane_normal_cam @ plane_point_cam)
            else:
                # Caméra historique (duck typing ``project``) : pas d'espace
                # caméra explicite. On saute tout triangle dont un sommet est
                # derrière la caméra — l'explosion near-plane reste exclue —
                # et on découpe le polygone écran au rectangle image.
                screen, zcam = camera.project(tri)
                if screen is None:
                    continue
                zcam = np.asarray(zcam, dtype=float).reshape(-1)
                if np.any(zcam <= z_near_m):
                    continue
                polygon = clip_polygon_to_image(
                    np.asarray(screen, dtype=float).reshape((-1, 2)), width, height
                )
                if len(polygon) < 3:
                    continue

                min_x = max(int(np.floor(polygon[:, 0].min())), 0)
                max_x = min(int(np.ceil(polygon[:, 0].max())), width - 1)
                min_y = max(int(np.floor(polygon[:, 1].min())), 0)
                max_y = min(int(np.ceil(polygon[:, 1].max())), height - 1)
                if max_x < min_x or max_y < min_y:
                    continue
                _fill_proxy_pixels(
                    polygon,
                    lambda px: _legacy_depth(camera, tri, screen, zcam, px),
                    depth,
                    face_map,
                    fid,
                    min_x,
                    max_x,
                    min_y,
                    max_y,
                    z_near_m,
                )
                continue

            min_x = max(int(np.floor(polygon[:, 0].min())), 0)
            max_x = min(int(np.ceil(polygon[:, 0].max())), width - 1)
            min_y = max(int(np.floor(polygon[:, 1].min())), 0)
            max_y = min(int(np.ceil(polygon[:, 1].max())), height - 1)
            if max_x < min_x or max_y < min_y:
                continue

            gx, gy = np.meshgrid(
                np.arange(min_x, max_x + 1) + 0.5,
                np.arange(min_y, max_y + 1) + 0.5,
            )
            inside = shapely.contains_xy(
                shapely.Polygon(polygon), gx.ravel(), gy.ravel()
            ).reshape(gx.shape)
            if not inside.any():
                continue

            # Profondeur exacte par rayon en espace caméra : le pixel définit
            # la droite (xn, yn, 1) ; son intersection avec le plan du
            # triangle donne le Z perspective-exact.
            rays_world = np.asarray([
                camera.ray_from_pixel(u, v) for u, v in zip(gx.ravel(), gy.ravel())
            ])
            rays_cam = rays_world @ camera.R.T
            rays_cam /= np.maximum(rays_cam[:, 2:3], 1e-12)
            denom = rays_cam @ plane_normal_cam
            safe_denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
            z_pixels = (denom_const / safe_denom).reshape(gx.shape)

            region = depth[min_y:max_y + 1, min_x:max_x + 1]
            closer = inside & (z_pixels > z_near_m) & (z_pixels < region)
            if not closer.any():
                continue
            region[closer] = z_pixels[closer]
            face_region = face_map[min_y:max_y + 1, min_x:max_x + 1]
            face_region[closer] = fid

        return cls(width=width, height=height, depth=depth, face_id_map=face_map)


def _legacy_ray_depth(camera, tri, pixel_xy):  # noqa: ANN001
    """Profondeur Z par intersection rayon/triangle pour une caméra legacy.

    Le rayon part de ``camera.position`` vers le centre du pixel reconstruit
    par les vecteurs droite/haut de la caméra ; l'intersection avec le plan
    du triangle donne le Z espace caméra (produit avec l'axe avant).
    """
    import numpy as np

    ox, oy = pixel_xy
    direction = (
        camera.fwd
        + ((ox - camera.width / 2) / camera.f) * camera.right
        - ((oy - camera.height / 2) / camera.f) * camera.up
    )
    direction = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return None
    direction /= norm
    e1 = tri[1] - tri[0]
    e2 = tri[2] - tri[0]
    normal = np.cross(e1, e2)
    norm_len = float(np.linalg.norm(normal))
    if norm_len < 1e-12:
        return None
    normal /= norm_len
    denom = float(direction @ normal)
    if abs(denom) < 1e-12:
        return None
    t = float((tri[0] - np.asarray(camera.position)) @ normal) / denom
    if t <= 0.0:
        return None
    hit = np.asarray(camera.position) + t * direction
    return float((hit - np.asarray(camera.position)) @ camera.fwd)


def _legacy_depth(camera, tri, screen, zcam, pixel_xy):  # noqa: ANN001
    """Depth for both rich legacy cameras and minimal project-only cameras."""
    if all(hasattr(camera, name) for name in ("fwd", "right", "up", "f", "width", "height")):
        return _legacy_ray_depth(camera, tri, pixel_xy)
    p = np.asarray(pixel_xy, dtype=float)
    a, b, c = np.asarray(screen, dtype=float)
    denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(float(denom)) < 1e-12:
        return None
    w0 = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / denom
    w1 = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / denom
    w2 = 1.0 - w0 - w1
    inv_depth = w0 / zcam[0] + w1 / zcam[1] + w2 / zcam[2]
    return None if inv_depth <= 0 else float(1.0 / inv_depth)


def _fill_proxy_pixels(
    polygon,
    depth_of_pixel,  # noqa: ANN001 - callable (px_tuple) -> float | None
    depth,
    face_map,
    fid,
    min_x,
    max_x,
    min_y,
    max_y,
    z_near_m,
):
    """Remplit le proxy pour une caméra legacy, pixel par pixel clippé."""
    import shapely

    gx, gy = np.meshgrid(
        np.arange(min_x, max_x + 1) + 0.5,
        np.arange(min_y, max_y + 1) + 0.5,
    )
    inside = shapely.contains_xy(
        shapely.Polygon(polygon), gx.ravel(), gy.ravel()
    ).reshape(gx.shape)
    if not inside.any():
        return
    rows, cols = np.where(inside)
    for row, col in zip(rows, cols):
        value = depth_of_pixel((gx[row, col], gy[row, col]))
        if value is None or not math.isfinite(value) or value <= z_near_m:
            continue
        y_index, x_index = min_y + row, min_x + col
        if value < depth[y_index, x_index]:
            depth[y_index, x_index] = value
            face_map[y_index, x_index] = fid


def _focal_of(camera):  # noqa: ANN001
    model = str(getattr(camera, "model", "PINHOLE")).upper()
    params = np.asarray(getattr(camera, "params", [1.0, 0.0, 0.0]), dtype=float)
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        return float(params[0]), float(params[0])
    return float(params[0]), float(params[1])


def _principal_of(camera):  # noqa: ANN001
    model = str(getattr(camera, "model", "PINHOLE")).upper()
    params = np.asarray(getattr(camera, "params", [1.0, 0.0, 0.0]), dtype=float)
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        return float(params[1]), float(params[2])
    return float(params[2]), float(params[3])


@dataclass
class RegisteredView:
    """Une vue enregistrée : tout ce qu'une validation peut consommer.

    L'image, la caméra canonique, les masques sémantiques et **les deux
    cartes de profondeur** voyagent ensemble. Le textureur ne reconstruit
    plus rien : il demande à cet objet ce que le pixel établit.

    Les profondeurs sont toutes exprimées en Z espace caméra — la même
    convention que ``ProxyDepth``, que ``LidarOcclusion`` et que le
    z-buffer : aucune conversion implicite ne traîne entre modules.
    """

    asset_id: str
    camera: object                      # CanonicalCamera ou compatible
    image: np.ndarray | None = None
    semantic_mask: np.ndarray | None = None
    proxy_depth: "ProxyDepth | None" = None
    lidar_depth: "LidarOcclusion | None" = None
    pose_error_px: float | None = None

    def occludes(self, pixel_xy: tuple[int, int], surface_z: float) -> bool:
        """Le pixel voit-il autre chose que la surface visée ?

        Comparaison strictement en Z espace caméra, tolérances distinctes
        pour le proxy et le LiDAR ; verdict identique quelle que soit la
        distance euclidienne du point hors axe.
        """
        x, y = int(pixel_xy[0]), int(pixel_xy[1])
        proxy = self.proxy_depth
        if proxy is not None:
            hit_z, hit_fid = proxy.hit(x, y)
            if hit_fid is not None and hit_fid >= 0 and surface_z > hit_z + PROXY_DEPTH_TOLERANCE_M:
                return True
        lidar = self.lidar_depth
        if lidar is not None and 0 <= y < lidar.valid.shape[0] and 0 <= x < lidar.valid.shape[1]:
            if lidar.valid[y, x] and surface_z > float(lidar.depth[y, x]) + LIDAR_OCCLUSION_MARGIN_M:
                return True
        return False


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


def local_projection_jacobian(camera, point_world, u_direction=None, delta_m=0.05):
    point = np.asarray(point_world, dtype=np.float64)
    direction_u = np.asarray(u_direction, dtype=np.float64) if u_direction is not None else np.array([1.0, 0.0, 0.0])
    norm_u = float(np.linalg.norm(direction_u))
    if norm_u < 1e-9:
        return 0.0, 0.0
    direction_u = direction_u / norm_u
    base_screen, _ = camera.project(point[None, :])
    if base_screen is None:
        return 0.0, 0.0
    base_screen = np.asarray(base_screen, dtype=np.float64).reshape(-1, 2)
    def _pixels_per_m(delta):
        screen, _ = camera.project((point + delta)[None, :])
        if screen is None:
            return 0.0
        displacement = float(np.linalg.norm(np.asarray(screen, dtype=np.float64).reshape(-1, 2)[0] - base_screen[0]))
        return displacement / max(delta_m, 1e-9)
    px_per_m_u = _pixels_per_m(direction_u * delta_m)
    px_per_m_v = _pixels_per_m(np.array([0.0, 0.0, delta_m]))
    return px_per_m_u, px_per_m_v


def effective_gsd_m(camera, point_world, u_direction=None, delta_m=0.05):
    px_u, px_v = local_projection_jacobian(camera, point_world, u_direction, delta_m)
    best = min(p for p in (px_u, px_v) if p > 0.0) if (px_u > 0 or px_v > 0) else 0.0
    if best <= 0.0:
        return None
    return 1.0 / best


def _estimate_local_gsd(camera, plane) -> float:
    centre = plane.point(plane.length_m * 0.5, 0.5)
    gsd = effective_gsd_m(camera, centre, plane.along)
    if gsd is not None:
        return gsd
    to_cam = np.asarray(camera.position, dtype=np.float64) - centre
    dist = float(np.linalg.norm(to_cam))
    if dist < 1e-6:
        return 0.12
    focal = getattr(camera, "f", None)
    if not focal:
        return 0.12
    return dist / max(focal, 1e-6)


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
    "effective_gsd_m",
    "local_projection_jacobian",
    "measure_facade_alignment",
    "local_facade_reprojection_gate",
]
