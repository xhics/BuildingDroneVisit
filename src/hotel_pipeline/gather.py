"""Collecte et tri d'un corpus (plan directeur §9, §11).

Enchaîne : collecte multi-sources → téléchargement → déduplication → qualité →
classification → manifeste d'assets. Chaque étape est facultative et dégradable :
l'absence d'une clé réduit la couverture, elle n'interrompt pas le traitement
(plan directeur §6).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import requests

from .collectors import CollectedImage, download, to_asset
from .logging import get_logger
from .schemas import Asset, AssetManifest, ExteriorInterior
from .triage import basic_scores, group_duplicates, normalised_quality, phash

log = get_logger("gather")


@dataclass
class SourceReport:
    """Ce qu'une source a rendu, et pourquoi elle a échoué le cas échéant."""

    name: str
    collected: int = 0
    downloaded: int = 0
    skipped_reason: str | None = None


@dataclass
class GatherReport:
    sources: list[SourceReport] = field(default_factory=list)
    duplicates: int = 0
    flagged_quality: int = 0

    def as_dict(self) -> dict:
        return {
            "sources": [
                {
                    "name": s.name,
                    "collected": s.collected,
                    "downloaded": s.downloaded,
                    "skipped_reason": s.skipped_reason,
                }
                for s in self.sources
            ],
            "duplicates": self.duplicates,
            "flagged_quality": self.flagged_quality,
        }


def _images_dir(workspace, source: str) -> Path:  # noqa: ANN001
    """Tout arrive en `reference_only` ; la promotion déplace ensuite."""
    return workspace.path("02_images", "reference_only", source)


def collect_sources(
    lat: float,
    lon: float,
    place_query: str | None,
    radius_m: int = 300,
    website_url: str | None = None,
) -> tuple[list[CollectedImage], list[SourceReport]]:
    """Interroge chaque source configurée, sans jamais bloquer sur une absence."""
    from .collectors import mapillary, places, streetview, website

    images: list[CollectedImage] = []
    reports: list[SourceReport] = []

    def attempt(name: str, fn) -> None:
        if not _configured(name):
            reports.append(SourceReport(name, skipped_reason="clé absente"))
            log.info("source %s ignorée : clé absente", name)
            return
        try:
            found = fn()
        except (requests.RequestException, RuntimeError) as exc:
            reports.append(SourceReport(name, skipped_reason=str(exc)))
            log.warning("source %s indisponible : %s", name, exc)
            return
        images.extend(found)
        reports.append(SourceReport(name, collected=len(found)))

    attempt("mapillary", lambda: mapillary.collect(lat, lon, radius_m))
    attempt("street_view", lambda: streetview.collect(lat, lon))
    if place_query:
        attempt("places", lambda: places.collect(place_query))
    if website_url:
        attempt("website", lambda: website.collect(website_url))

    return images, reports


def _configured(source: str) -> bool:
    """Le site officiel ne requiert aucune clé : seule son URL le conditionne."""
    required = {
        "mapillary": "MAPILLARY_TOKEN",
        "street_view": "GOOGLE_MAPS_API_KEY",
        "places": "GOOGLE_PLACES_API_KEY",
        "website": None,
    }[source]
    return required is None or bool(os.environ.get(required, "").strip())


def download_all(
    images: list[CollectedImage], workspace, reports: list[SourceReport]  # noqa: ANN001
) -> list[CollectedImage]:
    """Télécharge chaque image, en signant l'URL au dernier moment."""
    from .collectors import places, streetview

    signers = {"street_view": streetview.sign_url, "places": places.sign_url}
    by_name = {r.name: r for r in reports}
    downloaded: list[CollectedImage] = []

    for image in images:
        target = _images_dir(workspace, image.source) / f"{image.source_id}.jpg"
        signed = signers.get(image.source)
        original_url = image.url
        try:
            if signed:
                image.url = signed(image)
            download(image, target)
            downloaded.append(image)
            if image.source in by_name:
                by_name[image.source].downloaded += 1
        except (requests.RequestException, RuntimeError) as exc:
            log.warning("échec de téléchargement %s : %s", image.asset_id, exc)
        finally:
            # L'URL signée ne doit jamais atteindre le manifeste : elle porte
            # la clé d'API.
            image.url = original_url

    return downloaded


def read_signs(
    assets: list[Asset], reader, expected: list[str], excluded: list[str]  # noqa: ANN001
) -> dict[str, int]:
    """Lit l'enseigne des images et en déduit l'appartenance à l'établissement.

    C'est la réponse au risque nº1 du §3 : lire « Mortagne » disqualifie une
    image, lire « WelcomINNS » la confirme. Décisif pour les images sans
    position caméra, que la géométrie ne peut pas trancher.
    """
    from .schemas import PropertyMatchStatus
    from .triage import evaluate

    counts = {"match": 0, "mismatch": 0, "uncertain": 0}

    for index, asset in enumerate(assets):
        if not asset.local_path or not Path(asset.local_path).is_file():
            continue
        try:
            text = reader.read(Path(asset.local_path))
        except (OSError, ValueError, RuntimeError) as exc:
            log.warning("OCR impossible pour %s : %s", asset.id, exc)
            continue

        reading = evaluate(text, expected, excluded)
        assets[index] = asset.model_copy(
            update={"property_match_status": reading.status, "sign_text": text[:500] or None}
        )
        counts[reading.status.value] += 1

    log.info(
        "OCR d'enseigne : %d confirmée(s), %d disqualifiée(s), %d indéterminée(s)",
        counts["match"],
        counts["mismatch"],
        counts["uncertain"],
    )
    return counts


def triage(
    assets: list[Asset], classifier=None, quality_issues=None  # noqa: ANN001
) -> GatherReport:
    """Déduplique, note la qualité et classe les assets, en place."""
    report = GatherReport()

    hashes: dict[str, str] = {}
    for asset in assets:
        if asset.local_path and Path(asset.local_path).is_file():
            try:
                hashes[asset.id] = phash(Path(asset.local_path))
            except OSError as exc:
                log.warning("pHash impossible pour %s : %s", asset.id, exc)

    groups = group_duplicates(hashes)
    report.duplicates = len(groups) - len(set(groups.values()))

    for index, asset in enumerate(assets):
        updates: dict = {}
        if asset.id in hashes:
            updates["phash"] = hashes[asset.id]
            updates["duplicate_group"] = groups.get(asset.id)

        path = Path(asset.local_path) if asset.local_path else None
        if path and path.is_file():
            try:
                updates["quality_score"] = normalised_quality(basic_scores(path))
            except (OSError, ValueError) as exc:
                log.warning("qualité non mesurable pour %s : %s", asset.id, exc)

            if quality_issues and path.name in quality_issues.flagged():
                updates["quality_score"] = 0.0
                report.flagged_quality += 1

            if classifier is not None:
                try:
                    result = classifier.classify(path)
                    updates["exterior_or_interior"] = result.exterior_or_interior
                    updates["category"] = result.category
                    updates["confidence"] = result.exterior_confidence
                except (OSError, ValueError, RuntimeError) as exc:
                    log.warning("classification impossible pour %s : %s", asset.id, exc)

        if updates:
            assets[index] = asset.model_copy(update=updates)

    return report


def build_manifest(
    hotel_id: str, images: list[CollectedImage], assume_rights: bool
) -> AssetManifest:
    return AssetManifest(
        hotel_id=hotel_id,
        assets=[to_asset(image, assume_rights=assume_rights) for image in images],
    )


def summarise(manifest: AssetManifest) -> dict[str, int]:
    """Compte ce qui décide de la route, pas ce qui flatte le corpus.

    `exterior` recense les vues d'extérieur ; `sees_building` recense celles qui
    cadrent réellement l'hôtel. Seul le second chiffre a un sens pour le Gate
    photo-first — le premier inclut la chaussée, l'autoroute et les voisins.
    """
    exteriors = [
        a for a in manifest.assets if a.exterior_or_interior is ExteriorInterior.EXTERIOR
    ]
    seeing = [a for a in manifest.assets if a.sees_building]
    seeing_exterior = [a for a in seeing if a in exteriors]

    return {
        "total": len(manifest.assets),
        "exterior": len(exteriors),
        "exterior_unique": len({a.duplicate_group or a.id for a in exteriors}),
        "sees_building": len(seeing),
        "sees_building_unique": len({a.duplicate_group or a.id for a in seeing_exterior}),
        "usable_rights": len([a for a in manifest.assets if a.usable_in_production]),
        "encumbered": len([a for a in manifest.assets if a.rights_encumbered]),
    }
