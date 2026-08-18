"""Preprocessing et masques pour le Lot 2 — P1.

Ce module génère les masques SfM (sky, people, cars, water, etc.) et
les images normalisées pour la reconstruction. Les masques sont stockés
comme `DerivedArtifact` et ne modifient jamais les images originales.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .schemas.reconstruction import ReconstructionInputManifest
from .workspace import Workspace


class MaskSet(BaseModel):
    """Jeu de masques binaires pour les assets sélectionnés."""

    model_config = ConfigDict(extra="forbid")

    mask_set_id: str
    reconstruction_input_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    mask_classes: list[str] = Field(min_length=1)
    asset_mask_paths: dict[str, str] = Field(default_factory=dict)
    sha256: str = Field(min_length=64, max_length=64)


def _mask_sky(image: np.ndarray) -> np.ndarray:
    """Masque le ciel par seuillage couleur HSV (MVP heuristique)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([80, 20, 150])
    upper = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _mask_water(image: np.ndarray) -> np.ndarray:
    """Masque l'eau par seuillage couleur (MVP heuristique)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([80, 30, 80])
    upper = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _generate_mask(image_path: Path, mask_classes: list[str]) -> np.ndarray:
    """Génère un masque binaire combiné pour les classes demandées."""
    image = cv2.imread(str(image_path))
    if image is None:
        return np.zeros((10, 10), dtype=np.uint8)

    combined = np.zeros(image.shape[:2], dtype=np.uint8)
    for cls in mask_classes:
        if cls == "sky":
            combined = cv2.bitwise_or(combined, _mask_sky(image))
        elif cls == "water":
            combined = cv2.bitwise_or(combined, _mask_water(image))
        elif cls == "people":
            combined = cv2.bitwise_or(combined, _mask_people(image))
        elif cls == "cars":
            combined = cv2.bitwise_or(combined, _mask_cars(image))
        elif cls == "large_reflections":
            combined = cv2.bitwise_or(combined, _mask_large_reflections(image))
        elif cls == "signage":
            combined = cv2.bitwise_or(combined, _mask_signage(image))
        elif cls == "mobile_furniture":
            combined = cv2.bitwise_or(combined, _mask_mobile_furniture(image))
    return combined


def _mask_people(image: np.ndarray) -> np.ndarray:
    """MVP: pas de détecteur people intégré, retourne masque vide."""
    return np.zeros(image.shape[:2], dtype=np.uint8)


def _mask_cars(image: np.ndarray) -> np.ndarray:
    """MVP: pas de détecteur cars intégré, retourne masque vide."""
    return np.zeros(image.shape[:2], dtype=np.uint8)


def _mask_large_reflections(image: np.ndarray) -> np.ndarray:
    """MVP: pas de détecteur reflections intégré, retourne masque vide."""
    return np.zeros(image.shape[:2], dtype=np.uint8)


def _mask_signage(image: np.ndarray) -> np.ndarray:
    """MVP: pas de détecteur signage intégré, retourne masque vide."""
    return np.zeros(image.shape[:2], dtype=np.uint8)


def _mask_mobile_furniture(image: np.ndarray) -> np.ndarray:
    """MVP: pas de détecteur mobile_furniture intégré, retourne masque vide."""
    return np.zeros(image.shape[:2], dtype=np.uint8)


def generate_mask_set(
    workspace: Workspace,
    input_manifest: ReconstructionInputManifest,
    *,
    mask_classes: list[str] | None = None,
) -> str:
    """Génère un jeu de masques binaires pour les assets sélectionnés.

    Args:
        workspace: workspace de l'hôtel
        input_manifest: manifeste d'entrée
        mask_classes: classes à masquer (sky, people, cars, water, etc.)

    Returns:
        SHA-256 du jeu de masques
    """
    if mask_classes is None:
        mask_classes = ["sky", "people", "cars", "water"]

    mask_dir = workspace.path("05_colmap", "preprocessed", "masks")
    mask_dir.mkdir(parents=True, exist_ok=True)

    hasher = hashlib.sha256()
    asset_mask_paths: dict[str, str] = {}

    for asset_id in input_manifest.selected_asset_ids:
        mask_path = mask_dir / f"{asset_id}.png"
        image_path = _resolve_image_path(workspace, asset_id)
        if image_path and image_path.is_file():
            mask = _generate_mask(image_path, mask_classes)
            cv2.imwrite(str(mask_path), mask)
        else:
            mask_path.write_bytes(b"")
        relative = str(mask_path.relative_to(workspace.path("05_colmap")))
        asset_mask_paths[asset_id] = relative
        hasher.update(relative.encode("utf-8"))

    digest = hasher.hexdigest()
    mask_set = MaskSet(
        mask_set_id=f"mask-{digest[:16]}",
        reconstruction_input_id=input_manifest.reconstruction_input_id,
        mask_classes=mask_classes,
        asset_mask_paths=asset_mask_paths,
        sha256=digest,
    )

    output_path = workspace.path("05_colmap", "preprocessed", "mask_set.json")
    output_path.write_text(json.dumps(mask_set.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n")
    return digest


def _resolve_image_path(workspace: Workspace, asset_id: str) -> Path | None:
    """Résout le chemin image d'un asset depuis le manifeste."""
    try:
        from .schemas import AssetManifest
        assets = AssetManifest.model_validate_json(workspace.assets_path.read_text("utf-8"))
        by_id = {a.id: a for a in assets.assets}
        asset = by_id.get(asset_id)
        if asset and asset.local_path:
            return workspace.path(asset.local_path)
    except Exception:
        pass
    return None


def publish_mask_set(mask_set: MaskSet, workspace: Workspace) -> Path:
    """Publie le MaskSet sous `05_colmap/preprocessed/`."""
    output_path = workspace.path("05_colmap", "preprocessed", f"mask_set_{mask_set.mask_set_id}.json")
    output_path.write_text(json.dumps(mask_set.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n")
    return output_path


__all__ = [
    "MaskSet",
    "generate_mask_set",
    "publish_mask_set",
]
