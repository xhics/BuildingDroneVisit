"""Deterministic CPU rasterizer for independent holdout validation."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from .geo.facade_visibility import ProxyDepth


@dataclass(frozen=True)
class DenseRender:
    rgb: np.ndarray
    depth: np.ndarray
    normal: np.ndarray
    surface_id: np.ndarray
    silhouette: np.ndarray


def rasterize_canonical_mesh(
    camera,
    triangles: Sequence[np.ndarray],
    surface_ids: Sequence[int],
    width: int,
    height: int,
    colours: Sequence[tuple[int, int, int]] | None = None,
) -> DenseRender:
    """Render the canonical triangles using the same perspective z-buffer."""
    proxy = ProxyDepth.render(camera, triangles, surface_ids, width, height)
    ids = proxy.face_id_map
    assert ids is not None
    silhouette = ids >= 0
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    normal = np.zeros((height, width, 3), dtype=np.float32)
    colour_by_id: dict[int, tuple[int, int, int]] = {}
    normal_by_id: dict[int, np.ndarray] = {}
    for index, (triangle, sid) in enumerate(zip(triangles, surface_ids)):
        tri = np.asarray(triangle, dtype=float)
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        norm = float(np.linalg.norm(n))
        if norm > 1e-12:
            normal_by_id[int(sid)] = n / norm
        colour_by_id[int(sid)] = colours[index] if colours else (180, 180, 180)
    for sid in np.unique(ids[silhouette]):
        selected = ids == sid
        rgb[selected] = colour_by_id.get(int(sid), (180, 180, 180))
        normal[selected] = normal_by_id.get(int(sid), np.zeros(3))
    return DenseRender(rgb, proxy.depth, normal, ids.copy(), silhouette)


def silhouette_iou(rendered: np.ndarray, observed: np.ndarray) -> float | None:
    """Pixel IoU of two actual binary silhouettes; never a bbox surrogate."""
    a, b = np.asarray(rendered, bool), np.asarray(observed, bool)
    if a.shape != b.shape:
        raise ValueError("rendered and observed masks must have identical shapes")
    union = np.logical_or(a, b).sum()
    if union == 0:
        return None
    return float(np.logical_and(a, b).sum() / union)


def depth_error_m(rendered_depth: np.ndarray, observed_depth: np.ndarray) -> float | None:
    valid = np.isfinite(rendered_depth) & np.isfinite(observed_depth) & (observed_depth > 0)
    if not valid.any():
        return None
    return float(np.median(np.abs(rendered_depth[valid] - observed_depth[valid])))


def visible_surface_fraction(render: DenseRender, surface_ids: Sequence[int]) -> float:
    """Fraction of z-buffer pixels owned by requested, truly visible surfaces."""
    foreground = render.silhouette
    if not foreground.any():
        return 0.0
    visible = np.isin(render.surface_id, np.asarray(list(surface_ids), dtype=int)) & foreground
    return float(visible.sum() / foreground.sum())


__all__ = ["DenseRender", "depth_error_m", "rasterize_canonical_mesh", "silhouette_iou", "visible_surface_fraction"]
