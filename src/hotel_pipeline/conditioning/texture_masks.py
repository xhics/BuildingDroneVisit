"""Masques sémantiques dédiés à la texture de façade.

Deux sources sont lues, par ordre de priorité :
1. ``11_conditioning/texture_view_masks.json`` et ses rasters associés, produits
   par ``semantic_detection.py --purpose texture`` ;
2. ``11_conditioning/semantic_observations.json``, en repli.

Chaque vue retourne un masque ``building`` (booléen), un masque ``occluders``,
la fidélité du masque, les classes présentes et les régions d'enseigne non
tranchées à ce stade.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..workspace import Workspace

OCCLUDER_CLASSES = frozenset({
    "tree_evergreen", "tree_deciduous", "bush", "car", "truck", "bus",
    "person", "bicycle", "fence", "pole", "lamp_post", "road_sign",
    "mobiliary", "hvac_unit", "flower_pot",
})


@dataclass
class TextureViewMask:
    asset_id: str
    building: np.ndarray | None
    occluders: np.ndarray | None
    fidelity: str
    classes_present: list[str]
    sign_regions: list[dict[str, Any]]
    image_path: Path | None = None
    width: int = 0
    height: int = 0


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text("utf-8"))


def _load_raster(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        image = Image.open(path)
        if image.mode not in ("1", "L", "P"):
            image = image.convert("L")
        return np.asarray(image, dtype=bool)
    except Exception:
        return None


def _polygon_to_mask(points: list[list[float]], width: int, height: int) -> np.ndarray:
    import cv2

    mask = np.zeros((height, width), dtype=np.uint8)
    if len(points) < 3:
        return np.zeros((height, width), dtype=bool)
    pts = np.asarray([[point[0], point[1]] for point in points], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def _build_from_semantic_observations(workspace: Workspace) -> dict[str, TextureViewMask]:
    path = workspace.path("11_conditioning", "semantic_observations.json")
    payload = _read_json(path)
    if not payload:
        return {}

    input_paths = {
        str(item.get("asset_id")): workspace.root / str(item.get("path"))
        for item in payload.get("inputs", [])
    }
    by_asset: dict[str, list[dict]] = {}
    for observation in payload.get("observations", []):
        by_asset.setdefault(str(observation.get("asset_id")), []).append(observation)

    masks: dict[str, TextureViewMask] = {}
    for asset_id, observations in by_asset.items():
        image_path = input_paths.get(asset_id)
        if image_path is None or not image_path.is_file():
            continue
        with Image.open(image_path) as src:
            width, height = src.size

        building_polygons = [
            item.get("segmentation_2d", {}).get("points") or []
            for item in observations
            if item.get("class") == "building"
        ]
        if not any(len(points) >= 3 for points in building_polygons):
            continue

        building_mask = np.zeros((height, width), dtype=bool)
        occluder_mask = np.zeros((height, width), dtype=bool)
        sign_regions: list[dict[str, Any]] = []
        classes_present: list[str] = []

        for points in building_polygons:
            if len(points) >= 3:
                building_mask |= _polygon_to_mask(points, width, height)

        for item in observations:
            cls = str(item.get("class", ""))
            if cls == "building":
                continue
            classes_present.append(cls)
            points = item.get("segmentation_2d", {}).get("points") or []
            if len(points) < 3:
                continue
            if cls in OCCLUDER_CLASSES:
                occluder_mask |= _polygon_to_mask(points, width, height)
            elif cls in ("sign", "logo"):
                sign_regions.append({
                    "class": cls,
                    "points": points,
                    "decision": "pending",
                })

        has_holes = bool(occluder_mask.any())
        fidelity = "polygon_no_holes" if not has_holes else "polygon_with_occluders"

        masks[asset_id] = TextureViewMask(
            asset_id=asset_id,
            building=building_mask,
            occluders=occluder_mask,
            fidelity=fidelity,
            classes_present=sorted(set(classes_present)),
            sign_regions=sign_regions,
            image_path=image_path,
            width=width,
            height=height,
        )
    return masks


def _build_from_texture_masks_json(workspace: Workspace) -> dict[str, TextureViewMask]:
    path = workspace.path("11_conditioning", "texture_view_masks.json")
    payload = _read_json(path)
    if not payload:
        return {}

    masks: dict[str, TextureViewMask] = {}
    for view in payload.get("views", []):
        asset_id = str(view.get("asset_id"))
        raster_path = workspace.path("11_conditioning", "texture_view_masks", view.get("raster", ""))
        building = _load_raster(raster_path)
        if building is None:
            building = _polygon_to_mask(
                view.get("building_polygon", []),
                view.get("width", 0),
                view.get("height", 0),
            )
        occluders = _polygon_to_mask(
            view.get("occluders_polygon", []),
            view.get("width", 0),
            view.get("height", 0),
        ) if view.get("occluders_polygon") else None
        masks[asset_id] = TextureViewMask(
            asset_id=asset_id,
            building=building,
            occluders=occluders,
            fidelity=view.get("fidelity", "unknown"),
            classes_present=view.get("classes_present", []),
            sign_regions=view.get("sign_regions", []),
            image_path=Path(view["image_path"]) if view.get("image_path") else None,
            width=view.get("width", 0),
            height=view.get("height", 0),
        )
    return masks


def load_texture_masks(workspace: Workspace) -> dict[str, TextureViewMask]:
    masks = _build_from_texture_masks_json(workspace)
    if masks:
        return masks
    return _build_from_semantic_observations(workspace)


__all__ = ["OCCLUDER_CLASSES", "TextureViewMask", "load_texture_masks"]
