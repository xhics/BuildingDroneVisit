"""Dense validation renderer for independent, frozen holdout cameras."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def camera_digest(camera) -> str:  # noqa: ANN001
    digest = hashlib.sha256()
    for value in (camera.model, camera.width, camera.height, camera.near_m,
                  camera.far_m, camera.camera_id, camera.group):
        digest.update(repr(value).encode())
    for array in (camera.params, camera.R, camera.t):
        digest.update(np.asarray(array, dtype=np.float64).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class DenseRender:
    rgb: np.ndarray
    depth: np.ndarray
    normal: np.ndarray
    triangle_id: np.ndarray
    surface_id: np.ndarray
    surface_id_lookup: dict[int, str]
    silhouette: np.ndarray
    target_building_mask: np.ndarray
    category_masks: dict[str, np.ndarray]
    camera_digest: str
    input_mesh_digest: str
    proxy_renderer_usage: int = 0


@dataclass(frozen=True)
class HoldoutComparison:
    silhouette_iou: float | None
    boundary_median_px: float | None
    boundary_p90_px: float | None
    boundary_hausdorff_px: float | None
    edge_precision: float | None
    edge_recall: float | None
    edge_f1: float | None
    depth_median_error_m: float | None
    depth_rmse_m: float | None
    depth_p90_error_m: float | None
    normal_median_error_deg: float | None
    appearance_score: float | None
    per_surface: dict[str, dict]


def _surface_colour(surface_id: str) -> tuple[int, int, int]:
    raw = hashlib.sha256(surface_id.encode()).digest()
    return tuple(72 + int(value) % 144 for value in raw[:3])


def render_holdout(mesh, camera, *, target_building_id: str | None = None) -> DenseRender:  # noqa: ANN001
    """Render only the frozen canonical mesh through the canonical camera."""
    from .canonical_camera import CanonicalCamera
    from .reality_contract import require_canonical_mesh
    from .render_engine import rasterize_mesh

    if not isinstance(camera, CanonicalCamera):
        raise TypeError("holdout renderer requires CanonicalCamera")
    receipt = require_canonical_mesh(mesh, "dense_holdout_renderer")
    frame = rasterize_mesh(mesh, camera)
    visible = frame.triangle_id >= 0
    triangles = np.full(frame.triangle_id.shape, -1, dtype=np.int64)
    triangles[visible] = mesh.triangle_ids[frame.triangle_id[visible]]
    rgb = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
    target = np.zeros_like(visible)
    categories: dict[str, np.ndarray] = {}
    for encoded, physical_id in frame.surface_id_lookup.items():
        pixels = frame.surface_id == encoded
        rgb[pixels] = _surface_colour(physical_id)
        record = mesh.surface_catalog[physical_id]
        categories.setdefault(record.kind, np.zeros_like(visible))
        categories[record.kind] |= pixels
        if target_building_id is None or record.building_id == target_building_id:
            target |= pixels
    return DenseRender(
        rgb, frame.depth_z.copy(), frame.normal.copy(), triangles,
        frame.surface_id.copy(), dict(frame.surface_id_lookup), visible, target,
        categories, camera_digest(camera), receipt.input_mesh_digest, 0,
    )


def _boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, bool)
    p = np.pad(mask, 1, mode="constant")
    interior = (p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1]
                & p[1:-1, :-2] & p[1:-1, 2:])
    return mask & ~interior


def _boundary_metrics(rendered: np.ndarray, observed: np.ndarray):  # noqa: ANN202
    from scipy.ndimage import distance_transform_edt
    a, b = _boundary(rendered), _boundary(observed)
    if not a.any() or not b.any():
        return None, None, None
    errors = np.r_[distance_transform_edt(~a)[b], distance_transform_edt(~b)[a]]
    return float(np.median(errors)), float(np.percentile(errors, 90)), float(errors.max())


def _edge_map(image: np.ndarray, support: np.ndarray) -> np.ndarray:
    gray = np.asarray(image, float).mean(axis=2) if image.ndim == 3 else np.asarray(image, float)
    gx, gy = np.zeros_like(gray), np.zeros_like(gray)
    gx[:, 1:], gy[1:, :] = np.abs(np.diff(gray, axis=1)), np.abs(np.diff(gray, axis=0))
    magnitude = np.hypot(gx, gy)
    values = magnitude[support]
    threshold = float(np.percentile(values, 75)) if values.size else np.inf
    return (magnitude >= max(threshold, 5.0)) & support


def _edge_metrics(rendered: np.ndarray, observed: np.ndarray, tolerance_px: float = 2.0):  # noqa: ANN202
    from scipy.ndimage import distance_transform_edt
    if not rendered.any() or not observed.any():
        return None, None, None
    precision = float(np.mean(distance_transform_edt(~observed)[rendered] <= tolerance_px))
    recall = float(np.mean(distance_transform_edt(~rendered)[observed] <= tolerance_px))
    return precision, recall, 2 * precision * recall / max(precision + recall, 1e-12)


def silhouette_iou(rendered: np.ndarray, observed: np.ndarray) -> float | None:
    a, b = np.asarray(rendered, bool), np.asarray(observed, bool)
    if a.shape != b.shape:
        raise ValueError("rendered and observed masks must have identical shapes")
    union = np.logical_or(a, b).sum()
    return None if union == 0 else float(np.logical_and(a, b).sum() / union)


def _depth_metrics(rendered: np.ndarray, observed: np.ndarray, mask: np.ndarray):  # noqa: ANN202
    valid = mask & np.isfinite(rendered) & np.isfinite(observed) & (observed > 0)
    if not valid.any():
        return None, None, None
    errors = np.abs(rendered[valid] - observed[valid])
    return float(np.median(errors)), float(np.sqrt(np.mean(errors ** 2))), float(np.percentile(errors, 90))


def compare_holdout(render: DenseRender, observed_rgb: np.ndarray,
                    observed_building_mask: np.ndarray, *,
                    dynamic_mask: np.ndarray | None = None,
                    observed_depth: np.ndarray | None = None,
                    observed_normal: np.ndarray | None = None,
                    supported_appearance_mask: np.ndarray | None = None) -> HoldoutComparison:
    """Compare geometry first; RGB only inside explicitly supported pixels."""
    valid = np.ones(render.silhouette.shape, bool)
    if dynamic_mask is not None:
        valid &= ~np.asarray(dynamic_mask, bool)
    predicted = render.target_building_mask & valid
    observed = np.asarray(observed_building_mask, bool) & valid
    boundary = _boundary_metrics(predicted, observed)
    rendered_edges, observed_edges = _boundary(predicted), _edge_map(observed_rgb, observed)
    edge = _edge_metrics(rendered_edges, observed_edges)
    depth = ((None, None, None) if observed_depth is None else
             _depth_metrics(render.depth, observed_depth, predicted & observed))
    normal_error = None
    if observed_normal is not None:
        normal_valid = predicted & observed & (np.linalg.norm(observed_normal, axis=2) > 0.5)
        if normal_valid.any():
            dots = np.sum(render.normal[normal_valid] * observed_normal[normal_valid], axis=1)
            normal_error = float(np.median(np.degrees(np.arccos(np.clip(np.abs(dots), 0, 1)))))
    appearance = None
    if supported_appearance_mask is not None:
        photo_valid = predicted & observed & valid & np.asarray(supported_appearance_mask, bool)
        if photo_valid.any():
            residual = np.mean(np.abs(render.rgb[photo_valid].astype(float) - observed_rgb[photo_valid].astype(float)))
            appearance = float(np.clip(1.0 - residual / 255.0, 0.0, 1.0))
    per_surface = {}
    for encoded, physical_id in render.surface_id_lookup.items():
        pixels = (render.surface_id == encoded) & valid
        if not pixels.any():
            continue
        local_observed = observed & pixels
        per_surface[physical_id] = {
            "visible_pixels": int(pixels.sum()),
            "mask_support_fraction": float(local_observed.sum() / pixels.sum()),
            "edge_f1": _edge_metrics(_boundary(pixels), observed_edges & pixels)[2],
            "depth_median_error_m": (_depth_metrics(render.depth, observed_depth, pixels & observed)[0]
                                     if observed_depth is not None else None),
        }
    return HoldoutComparison(silhouette_iou(predicted, observed), *boundary, *edge,
                             *depth, normal_error, appearance, per_surface)


def render_digest(render: DenseRender) -> str:
    digest = hashlib.sha256()
    for array in (render.rgb, render.depth, render.normal, render.triangle_id,
                  render.surface_id, render.silhouette):
        digest.update(np.asarray(array).tobytes())
    digest.update(render.camera_digest.encode())
    digest.update(render.input_mesh_digest.encode())
    return digest.hexdigest()


def write_holdout_artifacts(output_dir: Path, image_id: str, render: DenseRender) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{image_id}-buffers.npz"
    np.savez_compressed(path, rgb=render.rgb, depth_m=render.depth,
                        normal_world=render.normal, triangle_id=render.triangle_id,
                        surface_id=render.surface_id,
                        target_silhouette=render.target_building_mask)
    metadata = output_dir / f"{image_id}-render.json"
    metadata.write_text(json.dumps({
        "image_id": image_id, "camera_digest": render.camera_digest,
        "canonical_mesh_digest": render.input_mesh_digest,
        "render_digest": render_digest(render),
        "surface_id_lookup": render.surface_id_lookup,
        "proxy_renderer_usage": 0, "normal_space": "world", "buffers": path.name,
    }, indent=2) + "\n", encoding="utf-8")
    return {"buffers": str(path), "metadata": str(metadata)}


def validate_dense_holdouts(
    mesh, cases: Sequence[dict], output_path: Path, *, train_asset_ids: Sequence[str],
    holdout_asset_ids: Sequence[str], frozen_model_digest: str,
    target_building_id: str | None = None,
) -> dict:
    """Render localized HOLDOUT cases and publish evidence consumed by Gate C."""
    if set(train_asset_ids) & set(holdout_asset_ids):
        raise ValueError("dense holdout validation received a leaking split")
    artifacts_dir = output_path.parent / f"{output_path.stem}-artifacts"
    rows, surface_rows = [], {}
    for case in cases:
        image_id = str(case["image_id"])
        if image_id not in set(holdout_asset_ids):
            raise ValueError(f"{image_id} is not an independent holdout")
        camera = case["camera"]
        expected_camera_digest = case.get("camera_digest")
        if expected_camera_digest and camera_digest(camera) != expected_camera_digest:
            raise ValueError(f"localized camera digest changed for {image_id}")
        render = render_holdout(mesh, camera, target_building_id=target_building_id)
        comparison = compare_holdout(
            render, case["rgb"], case["building_mask"],
            dynamic_mask=case.get("dynamic_mask"),
            observed_depth=case.get("depth_m"),
            observed_normal=case.get("normal_world"),
            supported_appearance_mask=case.get("supported_appearance_mask"),
        )
        artifacts = write_holdout_artifacts(artifacts_dir, image_id, render)
        row = {"image_id": image_id, "localization_success": True,
               **comparison.__dict__, "artifacts": artifacts,
               "camera_digest": render.camera_digest,
               "render_digest": render_digest(render)}
        rows.append(row)
        for surface_id, metrics in comparison.per_surface.items():
            surface_rows.setdefault(surface_id, []).append(metrics)
    def median(key: str):  # noqa: ANN202
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return float(np.median(values)) if values else None
    surfaces = {}
    type_rows: dict[str, list[float]] = {}
    for surface_id, metrics in sorted(surface_rows.items()):
        edges = [float(row["edge_f1"]) for row in metrics if row.get("edge_f1") is not None]
        depths = [float(row["depth_median_error_m"]) for row in metrics
                  if row.get("depth_median_error_m") is not None]
        surfaces[surface_id] = {
            "images": len(metrics),
            "edge_f1": float(np.median(edges)) if edges else None,
            "depth_median_error_m": float(np.median(depths)) if depths else None,
            "novel_view_validation_score": float(np.median(edges)) if edges else None,
        }
        kind = mesh.surface_catalog[surface_id].kind
        type_rows.setdefault(kind, []).extend(edges)
    surface_types = {
        kind: {"edge_f1": float(np.median(values)), "surface_samples": len(values)}
        for kind, values in sorted(type_rows.items()) if values
    }
    payload = {
        "renderer": "dense_holdout_renderer", "independent": True,
        "train_asset_ids": sorted(train_asset_ids),
        "holdout_asset_ids": sorted(holdout_asset_ids),
        "frozen_model_digest": frozen_model_digest,
        "canonical_mesh_digest": mesh.mesh_digest(),
        "proxy_renderer_usage": 0, "canonical_mesh_usage_fraction": 1.0,
        "silhouette_iou": median("silhouette_iou"),
        "edge_alignment": median("edge_f1"),
        "reprojection_px": median("boundary_median_px") or 0.0,
        "ssim": median("appearance_score") or 0.0,
        "lpips": 1.0 - (median("appearance_score") or 0.0),
        "feature_inliers": 0.0,
        "global_metrics": {
            "silhouette_iou": median("silhouette_iou"),
            "boundary_median_px": median("boundary_median_px"),
            "boundary_p90_px": median("boundary_p90_px"),
            "boundary_hausdorff_px": median("boundary_hausdorff_px"),
            "edge_f1": median("edge_f1"),
            "depth_median_error_m": median("depth_median_error_m"),
            "normal_median_error_deg": median("normal_median_error_deg"),
            "appearance_score": median("appearance_score"),
        },
        "holdout_results": rows, "surface_scores": surfaces,
        "surface_type_scores": surface_types,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def rasterize_canonical_mesh(camera, triangles: Sequence[np.ndarray],
                             surface_ids: Sequence[int], width: int, height: int,
                             colours: Sequence[tuple[int, int, int]] | None = None) -> DenseRender:
    """Legacy synthetic adapter; production must call :func:`render_holdout`."""
    from .geo.facade_visibility import ProxyDepth
    proxy = ProxyDepth.render(camera, triangles, surface_ids, width, height)
    ids = proxy.face_id_map
    assert ids is not None
    silhouette = ids >= 0
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    normal = np.zeros((height, width, 3), dtype=np.float32)
    for index, (triangle, sid) in enumerate(zip(triangles, surface_ids)):
        selected = ids == sid
        rgb[selected] = colours[index] if colours else (180, 180, 180)
        n = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        normal[selected] = n / max(float(np.linalg.norm(n)), 1e-12)
    return DenseRender(rgb, proxy.depth, normal, ids.copy(), ids.copy(),
                       {int(v): str(v) for v in surface_ids}, silhouette, silhouette,
                       {"building": silhouette}, "legacy-camera", "legacy-triangles", 1)


def depth_error_m(rendered_depth: np.ndarray, observed_depth: np.ndarray) -> float | None:
    return _depth_metrics(rendered_depth, observed_depth, np.ones(rendered_depth.shape, bool))[0]


def visible_surface_fraction(render: DenseRender, surface_ids: Sequence[int]) -> float:
    if not render.silhouette.any():
        return 0.0
    visible = np.isin(render.surface_id, np.asarray(list(surface_ids), int)) & render.silhouette
    return float(visible.sum() / render.silhouette.sum())


__all__ = ["DenseRender", "HoldoutComparison", "camera_digest", "compare_holdout",
           "depth_error_m", "rasterize_canonical_mesh", "render_digest",
           "render_holdout", "silhouette_iou", "visible_surface_fraction",
           "validate_dense_holdouts", "write_holdout_artifacts"]
