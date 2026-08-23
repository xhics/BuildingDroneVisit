"""Prétraitement / masquage (Lot 2 — P1.2).

Produit un `PreprocessManifest` reproductible **avant** tout matching : masques
(sky, personnes, véhicules, eau, réflexions spéculaires), normalisation EXIF/
orientation, graines de modèle caméra depuis EXIF, et cohortes temporelles.
Sans cela, LightGlue risque de fabriquer des arêtes sur des voitures, de la
végétation ou des reflets de piscine.

Les masques binaires sont générés dans `reconstruction_preprocess.py` ; ce
module orchestre le manifeste dérivé complet et le rend digest-trackable.
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


class CameraSeed(BaseModel):
    """Graine de modèle caméra depuis EXIF (à calibrer ensuite).

    `fx`/`fy` sont en **pixels**, pas en millimètres : COLMAP attend une
    focale en pixels. La conversion exige la largeur du capteur, absente de la
    plupart des EXIF ; on la dérive de `FocalLengthIn35mmFilm` quand elle est
    présente, sinon la graine est déclarée non calibrable.
    """

    model_config = ConfigDict(extra="forbid")

    asset_id: str
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    width: int | None = None
    height: int | None = None
    calibrable: bool = False
    #: Origine de la focale : "focal_35mm", ou None si non calibrable.
    focal_source: str | None = None
    #: Focale brute EXIF en mm, conservée pour audit.
    focal_mm: float | None = None


class NormalizationReport(BaseModel):
    """Normalisation EXIF/orientation — constatée, jamais supposée.

    `orientation_uniform` est un **constat** issu de la lecture du tag EXIF
    Orientation de chaque image, pas une intention. L'affirmer sans avoir
    inspecté les fichiers inventerait un fait.
    """

    model_config = ConfigDict(extra="forbid")

    images_normalized: int = Field(ge=0)
    orientation_uniform: bool = True
    color_space: str = "srgb"
    notes: list[str] = Field(default_factory=list)
    #: Images dont l'orientation EXIF a pu être lue.
    orientation_inspected: int = Field(default=0, ge=0)
    #: Valeurs d'orientation EXIF distinctes rencontrées.
    orientation_values: list[int] = Field(default_factory=list)


class TemporalCohort(BaseModel):
    """Une cohorte temporelle d'assets."""

    model_config = ConfigDict(extra="forbid")

    name: str
    asset_ids: list[str] = Field(default_factory=list)


class PreprocessManifest(BaseModel):
    """Manifeste de prétraitement reproductible (P1.2)."""

    model_config = ConfigDict(extra="forbid")

    preprocess_id: str = Field(min_length=1)
    reconstruction_input_id: str = Field(min_length=1)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    mask_set_digest: str | None = Field(default=None, min_length=64, max_length=64)
    masked_asset_ids: list[str] = Field(default_factory=list)
    exif_normalization: NormalizationReport
    camera_model_seeds: dict[str, CameraSeed] = Field(default_factory=dict)
    temporal_cohorts: list[TemporalCohort] | None = None


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_preprocess_manifest(
    workspace: Workspace,
    input_manifest: ReconstructionInputManifest,
    *,
    mask_set_digest: str | None = None,
    mask_classes: list[str] | None = None,
) -> PreprocessManifest:
    """Construit le PreprocessManifest depuis le manifeste d'entrée.

    Génère les masques (si absents), extrait les graines EXIF, et regroupe
    les cohortes temporelles depuis le manifeste d'assets.
    """
    from .reconstruction_preprocess import generate_mask_set

    if mask_set_digest is None:
        mask_set_digest = generate_mask_set(
            workspace, input_manifest, mask_classes=mask_classes
        )
    masked_asset_ids = list(input_manifest.selected_asset_ids)

    camera_model_seeds: dict[str, CameraSeed] = {}
    for asset_id in input_manifest.selected_asset_ids:
        seed = _extract_camera_seed(workspace, asset_id)
        if seed is not None:
            camera_model_seeds[asset_id] = seed

    temporal_cohorts = _build_temporal_cohorts(input_manifest)

    return PreprocessManifest(
        preprocess_id=f"pre-{input_manifest.reconstruction_input_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        reconstruction_input_id=input_manifest.reconstruction_input_id,
        mask_set_digest=mask_set_digest,
        masked_asset_ids=masked_asset_ids,
        exif_normalization=_inspect_orientation(
            workspace, input_manifest.selected_asset_ids
        ),
        camera_model_seeds=camera_model_seeds,
        temporal_cohorts=temporal_cohorts,
    )


def _inspect_orientation(
    workspace: Workspace, asset_ids: list[str]
) -> NormalizationReport:
    """Constate l'orientation EXIF réelle du corpus sélectionné."""
    from .schemas import AssetManifest

    try:
        assets = AssetManifest.model_validate_json(
            workspace.assets_path.read_text("utf-8")
        )
        by_id = {a.id: a for a in assets.assets}
    except Exception:
        by_id = {}

    values: set[int] = set()
    inspected = 0
    unreadable = 0

    for asset_id in asset_ids:
        asset = by_id.get(asset_id)
        if asset is None or not asset.local_path:
            unreadable += 1
            continue
        img_path = workspace.path(asset.local_path)
        if not img_path.is_file():
            unreadable += 1
            continue
        try:
            from PIL import Image

            with Image.open(img_path) as img:
                exif = img._getexif()
            # 274 = tag EXIF Orientation.
            orientation = int(exif.get(274, 1)) if exif else 1
            values.add(orientation)
            inspected += 1
        except Exception:
            unreadable += 1

    notes: list[str] = []
    if inspected:
        notes.append(f"orientation EXIF lue sur {inspected} image(s)")
    if unreadable:
        notes.append(f"{unreadable} image(s) sans orientation lisible")
    if not inspected:
        notes.append("aucune orientation constatée : uniformité non établie")

    return NormalizationReport(
        images_normalized=inspected,
        orientation_uniform=bool(inspected) and len(values) <= 1,
        color_space="srgb",
        notes=notes,
        orientation_inspected=inspected,
        orientation_values=sorted(values),
    )


def _extract_camera_seed(workspace: Workspace, asset_id: str) -> CameraSeed | None:
    try:
        from .schemas import AssetManifest
        from PIL import Image, ExifTags

        assets = AssetManifest.model_validate_json(
            workspace.assets_path.read_text("utf-8")
        )
        by_id = {a.id: a for a in assets.assets}
        asset = by_id.get(asset_id)
        if asset is None or not asset.local_path:
            return None
        img_path = workspace.path(asset.local_path)
        if not img_path.is_file():
            return None
        with Image.open(img_path) as img:
            exif = img._getexif()
            pixel_size = img.size  # lu avant fermeture du fichier
        if not exif:
            return CameraSeed(
                asset_id=asset_id,
                width=pixel_size[0],
                height=pixel_size[1],
                cx=pixel_size[0] / 2.0,
                cy=pixel_size[1] / 2.0,
                calibrable=False,
            )
        exif_data = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}

        def _as_float(value: Any) -> float | None:
            if value is None:
                return None
            if isinstance(value, tuple) and len(value) == 2 and value[1]:
                return float(value[0]) / float(value[1])
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        focal_mm = _as_float(exif_data.get("FocalLength"))
        width = exif_data.get("ExifImageWidth") or exif_data.get("ImageWidth")
        height = exif_data.get("ExifImageHeight") or exif_data.get("ImageLength")

        # Dimensions de repli : le pixel réel de l'image décodée.
        if not width or not height:
            width, height = pixel_size

        if not width or not height:
            return CameraSeed(asset_id=asset_id, calibrable=False, focal_mm=focal_mm)

        width = int(width)
        height = int(height)

        # La focale 35 mm équivalente donne le champ de vision sans connaître
        # le capteur : fx_px = f35 * largeur_px / 36 (largeur du plein format).
        focal_35 = _as_float(exif_data.get("FocalLengthIn35mmFilm"))
        if focal_35 and focal_35 > 0:
            fx = focal_35 * float(max(width, height)) / 36.0
            return CameraSeed(
                asset_id=asset_id,
                fx=fx,
                fy=fx,
                cx=width / 2.0,
                cy=height / 2.0,
                width=width,
                height=height,
                calibrable=True,
                focal_source="focal_35mm",
                focal_mm=focal_mm,
            )

        # Focale en mm sans largeur de capteur : non convertible en pixels.
        # On enregistre le fait brut plutôt qu'une intrinsèque fausse d'un
        # facteur ~100 à 500.
        return CameraSeed(
            asset_id=asset_id,
            cx=width / 2.0,
            cy=height / 2.0,
            width=width,
            height=height,
            calibrable=False,
            focal_mm=focal_mm,
        )
    except Exception:
        return None


def _build_temporal_cohorts(
    input_manifest: ReconstructionInputManifest,
) -> list[TemporalCohort] | None:
    if not input_manifest.temporal_cohorts:
        return None
    return [
        TemporalCohort(name=name, asset_ids=ids)
        for name, ids in input_manifest.temporal_cohorts.items()
    ]


def publish_preprocess_manifest(manifest: PreprocessManifest, workspace: Workspace) -> Path:
    """Publie le PreprocessManifest sous `07_reconstruction/preprocess/`."""
    output_dir = workspace.path("07_reconstruction", "preprocess")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{manifest.preprocess_id}.json"
    path.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n")
    return path


__all__ = [
    "CameraSeed",
    "NormalizationReport",
    "TemporalCohort",
    "PreprocessManifest",
    "build_preprocess_manifest",
    "publish_preprocess_manifest",
]
