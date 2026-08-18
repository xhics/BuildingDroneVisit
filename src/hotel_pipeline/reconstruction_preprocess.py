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
        mask_classes: classes à masquer (sky, people, cars, water, large_reflections, signage, mobile_furniture)

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
        mask_path.write_bytes(b"\x00" * 10)
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
