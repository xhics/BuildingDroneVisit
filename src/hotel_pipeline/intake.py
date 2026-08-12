"""Inventaire et qualification des photos (plan directeur §9 ; complément §4).

Deux règles structurantes :

1. une image reste `reference_only` tant que ses droits ne permettent pas son
   usage en reconstruction — c'est le schéma qui l'impose, pas la discipline ;
2. la version de l'entrée (avant/après la rénovation de 2024) n'est pas
   déductible visuellement sans référence datée. C'est un verrou humain.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from .logging import get_logger
from .schemas import (
    Asset,
    AssetCategory,
    AssetManifest,
    EntranceVersion,
    ExteriorInterior,
    Rights,
)

log = get_logger("intake")

ASSET_MANIFEST_NAME = "asset_manifest.json"

#: Colonnes acceptées dans un inventaire CSV. `id` et `rights` sont obligatoires.
CSV_COLUMNS = (
    "id",
    "source",
    "source_url_or_id",
    "rights",
    "category",
    "capture_year",
    "exterior_or_interior",
    "entrance_version",
    "file",
)


class IntakeError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _enum(value: str, enum_cls, default):
    text = (value or "").strip()
    if not text:
        return default
    try:
        return enum_cls(text)
    except ValueError as exc:
        allowed = ", ".join(sorted(m.value for m in enum_cls))
        raise IntakeError(f"valeur {text!r} invalide ; attendu l'un de : {allowed}") from exc


def load_csv(csv_path: Path, images_root: Path | None = None) -> list[Asset]:
    """Charge un inventaire CSV en assets validés.

    Les droits sont obligatoires et sans valeur par défaut permissive : une
    ligne sans `rights` est refusée plutôt que supposée exploitable.
    """
    if not csv_path.is_file():
        raise IntakeError(f"inventaire introuvable : {csv_path}")

    assets: list[Asset] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        unknown = set(reader.fieldnames or []) - set(CSV_COLUMNS)
        if unknown:
            raise IntakeError(
                f"colonnes inconnues : {sorted(unknown)} ; attendu : {list(CSV_COLUMNS)}"
            )

        for line_no, row in enumerate(reader, start=2):
            asset_id = (row.get("id") or "").strip()
            if not asset_id:
                raise IntakeError(f"ligne {line_no} : colonne 'id' vide")

            rights_raw = (row.get("rights") or "").strip()
            if not rights_raw:
                raise IntakeError(
                    f"ligne {line_no} ({asset_id}) : 'rights' est obligatoire — "
                    f"un asset sans droits établis ne peut pas entrer dans le pipeline"
                )

            try:
                rights = _enum(rights_raw, Rights, Rights.UNKNOWN)
                category = _enum(row.get("category", ""), AssetCategory, AssetCategory.OTHER)
                exterior = _enum(
                    row.get("exterior_or_interior", ""), ExteriorInterior, ExteriorInterior.UNKNOWN
                )
                entrance = _enum(
                    row.get("entrance_version", ""), EntranceVersion, EntranceVersion.UNKNOWN
                )
            except IntakeError as exc:
                raise IntakeError(f"ligne {line_no} ({asset_id}) : {exc}") from exc

            checksum = "0" * 64
            file_ref = (row.get("file") or "").strip()
            if file_ref and images_root:
                file_path = images_root / file_ref
                if not file_path.is_file():
                    raise IntakeError(f"ligne {line_no} ({asset_id}) : fichier absent {file_path}")
                checksum = sha256_file(file_path)

            year_raw = (row.get("capture_year") or "").strip()

            assets.append(
                Asset(
                    id=asset_id,
                    source=(row.get("source") or "inconnu").strip(),
                    source_url_or_id=(row.get("source_url_or_id") or file_ref or "—").strip(),
                    rights=rights,
                    ai_eligible=False,
                    confidence=0.5,
                    category=category,
                    capture_year=int(year_raw) if year_raw else None,
                    checksum=checksum,
                    exterior_or_interior=exterior,
                    entrance_version=entrance,
                    # Jamais accordé à l'import : l'éligibilité production est
                    # une décision explicite, prise après revue des droits.
                    production_eligible=False,
                )
            )

    return assets


def promote(manifest: AssetManifest, asset_ids: list[str]) -> list[str]:
    """Marque des assets comme éligibles production.

    Le validateur du schéma refuse la promotion d'un asset aux droits
    insuffisants ; l'erreur remonte telle quelle.
    """
    promoted: list[str] = []
    for asset_id in asset_ids:
        asset = next((a for a in manifest.assets if a.id == asset_id), None)
        if asset is None:
            raise IntakeError(f"asset inconnu : {asset_id!r}")
        updated = asset.model_copy(update={"production_eligible": True})
        Asset.model_validate(updated.model_dump())  # refuse si les droits ne suivent pas
        manifest.assets[manifest.assets.index(asset)] = updated
        promoted.append(asset_id)
    return promoted


def coverage(manifest: AssetManifest) -> dict[str, int]:
    """Compte ce qui conditionne la suite du pipeline."""
    eligible = manifest.production_eligible()
    exteriors = [a for a in eligible if a.exterior_or_interior is ExteriorInterior.EXTERIOR]
    return {
        "total": len(manifest.assets),
        "production_eligible": len(eligible),
        "exterior_eligible": len(exteriors),
        "exterior_post_2024": len(
            [a for a in exteriors if a.entrance_version is EntranceVersion.POST_2024]
        ),
        "entrance_version_unknown": len(
            [a for a in exteriors if a.entrance_version is EntranceVersion.UNKNOWN]
        ),
    }
