"""Masques sémantiques dédiés à la texture de façade.

Trois sources sont lues, par ordre de priorité :
1. ``11_conditioning/texture_view_masks_store/index.json`` — le magasin
   primaire écrit par :func:`save_texture_masks` : rasters binaires PNG
   accompagnés de ``width``, ``height``, checksum du raster et checksum de
   l'image RGB associée. Le JSON polygonal devient secondaire ;
2. ``11_conditioning/texture_view_masks.json`` et ses rasters associés,
   produits par ``semantic_detection.py --purpose texture`` ;
3. ``11_conditioning/semantic_observations.json``, en repli.

Contrat fail-closed : une vue sans vrai masque ``building`` ne peut pas
texturer la façade. Le complément d'un masque d'occludeurs n'est **jamais**
considéré comme un masque bâtiment. Un masque dont la géométrie pixel ne
correspond pas à l'image RGB est rejeté sauf transformation explicite
``T_mask_to_canonical_image`` fournie dans la provenance.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..logging import get_logger
from ..workspace import Workspace

log = get_logger("texture-masks")

OCCLUDER_CLASSES = frozenset({
    "tree_evergreen", "tree_deciduous", "bush", "car", "truck", "bus",
    "person", "bicycle", "fence", "pole", "lamp_post", "road_sign",
    "mobiliary", "hvac_unit", "flower_pot",
})

MASK_STORE_CONTRACT_VERSION = 2
MASK_STORE_DIR = ("11_conditioning", "texture_view_masks_store")


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
    building_checksum: str | None = None
    occluders_checksum: str | None = None
    image_checksum: str | None = None
    transform: dict[str, Any] | None = None

    @property
    def mask_id(self) -> str:
        return self.asset_id


def mask_checksum(mask: np.ndarray) -> str:
    packed = np.packbits(np.asarray(mask, dtype=bool), bitorder="big").tobytes()
    return "sha256:" + hashlib.sha256(packed).hexdigest()


def save_texture_masks(workspace: Workspace, masks: dict[str, TextureViewMask]) -> Path:
    store = workspace.path(*MASK_STORE_DIR)
    store.mkdir(parents=True, exist_ok=True)
    entries = []
    for asset_id in sorted(masks):
        mask = masks[asset_id]
        entry = {
            "mask_id": mask.mask_id,
            "asset_id": asset_id,
            "width": int(mask.width),
            "height": int(mask.height),
            "fidelity": mask.fidelity,
            "classes_present": list(mask.classes_present),
            "sign_regions": list(mask.sign_regions),
            "image_path": str(mask.image_path) if mask.image_path else None,
            "image_checksum": mask.image_checksum,
            "transform": mask.transform,
            "building_raster": None,
            "building_checksum": None,
            "occluders_raster": None,
            "occluders_checksum": None,
        }
        asset_dir = store / asset_id
        if mask.building is not None:
            asset_dir.mkdir(parents=True, exist_ok=True)
            raster_rel = f"{asset_id}/building.png"
            _write_bool_png(asset_dir / "building.png", mask.building)
            entry["building_raster"] = raster_rel
            entry["building_checksum"] = mask_checksum(mask.building)
            entry["width"] = int(mask.building.shape[1])
            entry["height"] = int(mask.building.shape[0])
        if mask.occluders is not None:
            asset_dir.mkdir(parents=True, exist_ok=True)
            raster_rel = f"{asset_id}/occluders.png"
            _write_bool_png(asset_dir / "occluders.png", mask.occluders)
            entry["occluders_raster"] = raster_rel
            entry["occluders_checksum"] = mask_checksum(mask.occluders)
        entries.append(entry)
    payload = {"contract_version": MASK_STORE_CONTRACT_VERSION, "views": entries}
    index_path = store / "index.json"
    index_path.write_text(json.dumps(payload, indent=2), "utf-8")
    return index_path


def _write_bool_png(path: Path, mask: np.ndarray) -> None:
    binary = np.where(np.asarray(mask, dtype=bool), 255, 0).astype(np.uint8)
    Image.fromarray(binary, mode="L").save(path)


def _load_bool_png(path: Path, expected_checksum: str | None = None) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        image = Image.open(path)
        array = np.asarray(image.convert("L"))
    except OSError:
        return None
    mask = array > 127
    if expected_checksum and mask_checksum(mask) != expected_checksum:
        log.warning("checksum de masque divergent : %s rejeté", path)
        return None
    return mask


def align_mask_to_image(mask: np.ndarray, image_shape: tuple[int, ...], transform: dict[str, Any] | None = None) -> np.ndarray | None:
    target_hw = (int(image_shape[0]), int(image_shape[1]))
    if (int(mask.shape[0]), int(mask.shape[1])) == target_hw:
        return np.asarray(mask, dtype=bool).copy()
    if not transform:
        return None
    kind = str(transform.get("type", ""))
    import cv2
    h, w = target_hw
    source = np.asarray(mask, dtype=np.uint8)
    if kind in ("affine", "T_mask_to_canonical_image") and transform.get("matrix"):
        matrix = np.asarray(transform["matrix"], dtype=np.float64)
        if matrix.shape != (2, 3):
            return None
        warped = cv2.warpAffine(source, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return warped.astype(bool)
    if kind == "resize":
        src_w = int(transform.get("source_width", 0))
        src_h = int(transform.get("source_height", 0))
        if src_w != int(mask.shape[1]) or src_h != int(mask.shape[0]):
            return None
        resized = cv2.resize(source, (w, h), interpolation=cv2.INTER_NEAREST)
        return resized.astype(bool)
    return None


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text("utf-8"))


def _build_from_store_index(workspace: Workspace) -> dict[str, TextureViewMask]:
    payload = _read_json(workspace.path(*MASK_STORE_DIR, "index.json"))
    if not payload or int(payload.get("contract_version", 0)) < MASK_STORE_CONTRACT_VERSION:
        return {}
    store = workspace.path(*MASK_STORE_DIR)
    masks = {}
    for view in payload.get("views", []):
        asset_id = str(view.get("asset_id"))
        building = None
        if view.get("building_raster"):
            building = _load_bool_png(store / view["building_raster"], view.get("building_checksum"))
        occluders = None
        if view.get("occluders_raster"):
            occluders = _load_bool_png(store / view["occluders_raster"], view.get("occluders_checksum"))
        width = int(view.get("width", 0))
        height = int(view.get("height", 0))
        if building is not None:
            width, height = int(building.shape[1]), int(building.shape[0])
        masks[asset_id] = TextureViewMask(
            asset_id=asset_id,
            building=building,
            occluders=occluders,
            fidelity=str(view.get("fidelity", "unknown")),
            classes_present=list(view.get("classes_present", [])),
            sign_regions=list(view.get("sign_regions", [])),
            image_path=Path(view["image_path"]) if view.get("image_path") else None,
            width=width,
            height=height,
            building_checksum=view.get("building_checksum"),
            occluders_checksum=view.get("occluders_checksum"),
            image_checksum=view.get("image_checksum"),
            transform=view.get("transform"),
        )
    return masks


def _build_from_semantic_observations(workspace: Workspace) -> dict[str, TextureViewMask]:
    path = workspace.path("11_conditioning", "semantic_observations.json")
    payload = _read_json(path)
    if not payload:
        return {}
    input_paths = {str(item.get("asset_id")): workspace.root / str(item.get("path")) for item in payload.get("inputs", [])}
    by_asset = {}
    for observation in payload.get("observations", []):
        by_asset.setdefault(str(observation.get("asset_id")), []).append(observation)
    masks = {}
    for asset_id, observations in by_asset.items():
        image_path = input_paths.get(asset_id)
        if image_path is None or not image_path.is_file():
            continue
        with Image.open(image_path) as src:
            width, height = src.size
        building_polygons = [item.get("segmentation_2d", {}).get("points") or [] for item in observations if item.get("class") == "building"]
        if not any(len(points) >= 3 for points in building_polygons):
            continue
        building_mask = _merge_polygons_into_mask(building_polygons, width, height)
        occluder_mask = _merge_polygons_into_mask([item.get("segmentation_2d", {}).get("points") or [] for item in observations if item.get("class") in OCCLUDER_CLASSES], width, height)
        sign_regions = []
        classes_present = []
        for item in observations:
            cls = str(item.get("class", ""))
            if cls == "building":
                continue
            classes_present.append(cls)
            points = item.get("segmentation_2d", {}).get("points") or []
            if len(points) < 3:
                continue
            if cls in ("sign", "logo"):
                sign_regions.append({"class": cls, "points": points, "decision": "pending"})
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
            building_checksum=mask_checksum(building_mask),
            occluders_checksum=mask_checksum(occluder_mask),
        )
    return masks


def _build_from_texture_masks_json(workspace: Workspace) -> dict[str, TextureViewMask]:
    path = workspace.path("11_conditioning", "texture_view_masks.json")
    payload = _read_json(path)
    if not payload:
        return {}
    masks = {}
    for view in payload.get("views", []):
        asset_id = str(view.get("asset_id"))
        width = int(view.get("width", 0))
        height = int(view.get("height", 0))
        building = None
        raster_path = workspace.path("11_conditioning", "texture_view_masks", view.get("raster", "") or "")
        if view.get("raster") and raster_path.is_file():
            candidate = _load_bool_png(raster_path, view.get("building_checksum"))
            if candidate is not None and (candidate.shape[1], candidate.shape[0]) == (width, height):
                building = candidate
            elif candidate is not None:
                log.warning("masque %s : raster %dx%d != %dx%d annonce — rejete", asset_id, candidate.shape[1], candidate.shape[0], width, height)
        if building is None and width > 0 and height > 0:
            building = _merge_polygons_into_mask(view.get("building_polygons") or [view.get("building_polygon") or []], width, height)
        occluders = None
        if view.get("occluders_raster"):
            occluders = _load_bool_png(workspace.path("11_conditioning", "texture_view_masks", view["occluders_raster"]), view.get("occluders_checksum"))
        elif view.get("occluders_polygon") and width > 0 and height > 0:
            occluders = _merge_polygons_into_mask([view.get("occluders_polygon") or []], width, height)
        masks[asset_id] = TextureViewMask(
            asset_id=asset_id,
            building=building,
            occluders=occluders,
            fidelity=view.get("fidelity", "unknown"),
            classes_present=view.get("classes_present", []),
            sign_regions=view.get("sign_regions", []),
            image_path=Path(view["image_path"]) if view.get("image_path") else None,
            width=width,
            height=height,
            building_checksum=view.get("building_checksum"),
            occluders_checksum=view.get("occluders_checksum"),
            image_checksum=view.get("image_checksum"),
            transform=view.get("transform"),
        )
    return masks


def load_texture_masks(workspace: Workspace) -> dict[str, TextureViewMask]:
    masks = _build_from_store_index(workspace)
    if masks:
        return masks
    masks = _build_from_texture_masks_json(workspace)
    if masks:
        return masks
    return _build_from_semantic_observations(workspace)


def _merge_polygons_into_mask(polygons: list[list[list[float]]], width: int, height: int) -> np.ndarray:
    import cv2
    mask = np.zeros((height, width), dtype=bool)
    if width <= 0 or height <= 0:
        return mask
    all_points = []
    for polygon in polygons:
        pts = np.asarray(polygon, dtype=np.int32)
        if pts.shape[0] >= 3:
            all_points.append(pts)
    if all_points:
        cv2.fillPoly(np.zeros((height, width), dtype=np.uint8), all_points, 1, mask=mask)
    return mask


__all__ = ["OCCLUDER_CLASSES", "TextureViewMask", "align_mask_to_image", "load_texture_masks", "mask_checksum", "save_texture_masks"]
