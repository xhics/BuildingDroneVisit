"""Socle commun aux collecteurs (plan directeur §9).

Chaque source étiquette ses droits **à la source**, jamais par défaut permissif.
La conversion en `Asset` est centralisée ici pour qu'aucun collecteur ne puisse
inventer sa propre politique de droits.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import requests

from ..logging import get_logger
from ..schemas import Asset, AssetCategory, ExteriorInterior, Rights

log = get_logger("collect")

DOWNLOAD_TIMEOUT = 60


@dataclass(frozen=True)
class SourcePolicy:
    """Droits et attribution d'une source, décidés une fois pour toutes."""

    name: str
    rights: Rights
    #: L'usage en production requiert-il une décision explicite de l'opérateur ?
    requires_assumption: bool
    attribution: str | None = None
    note: str | None = None


#: Politique de droits par source. Le tableau est la référence unique ;
#: modifier un droit se fait ici, pas au fil du code.
POLICIES: dict[str, SourcePolicy] = {
    "mapillary": SourcePolicy(
        "mapillary",
        Rights.OPEN_DATA,
        requires_assumption=False,
        attribution="© contributeurs Mapillary, CC BY-SA 4.0",
        note="licence ouverte, attribution obligatoire",
    ),
    "street_view": SourcePolicy(
        "street_view",
        Rights.PUBLIC_UNCLEARED,
        requires_assumption=True,
        attribution="© Google Street View",
        note="conditions Google Maps Platform restrictives pour l'usage dérivé",
    ),
    "places": SourcePolicy(
        "places",
        Rights.PUBLIC_UNCLEARED,
        requires_assumption=True,
        attribution="© déposants Google Places",
        note="droits détenus par les déposants",
    ),
    "commons": SourcePolicy(
        "commons",
        Rights.OPEN_DATA,
        requires_assumption=False,
        attribution="© contributeurs Wikimedia Commons",
        note="licences ouvertes, attribution selon la fiche du fichier",
    ),
    "flickr": SourcePolicy(
        "flickr",
        Rights.OPEN_DATA,
        requires_assumption=False,
        attribution="© auteur Flickr, licence Creative Commons",
        note="collecte restreinte aux licences CC et domaine public",
    ),
    "tripadvisor": SourcePolicy(
        "tripadvisor",
        Rights.PUBLIC_UNCLEARED,
        requires_assumption=True,
        attribution="© déposants TripAdvisor",
        note="droits des déposants, attribution imposée par TripAdvisor",
    ),
    "hotel": SourcePolicy(
        "hotel",
        Rights.OWNED,
        requires_assumption=False,
        note="fourni ou autorisé par l'établissement",
    ),
    "website": SourcePolicy(
        "website",
        Rights.UNKNOWN,
        requires_assumption=True,
        attribution="© site officiel de l'établissement",
        note="site officiel — à clarifier avec l'établissement",
    ),
}


@dataclass
class CollectedImage:
    """Une image rapportée par un collecteur, avant qualification."""

    source: str
    source_id: str
    url: str
    captured_year: int | None = None
    heading_deg: float | None = None
    lat: float | None = None
    lon: float | None = None
    extra: dict[str, str] = field(default_factory=dict)
    local_path: Path | None = None

    #: Trajet dont cette vue fait partie, quand la source le publie. Deux vues
    #: d'une même séquence ont un recouvrement **plausible** — pas prouvé : un
    #: véhicule tourne. `None` signifie « non rendu », jamais « aucune ».
    sequence_id: str | None = None

    #: Le cap est-il observé (imagerie de roulage) ou choisi par nous
    #: (extraction d'un panorama sphérique) ? Voir `Asset.heading_is_measured`.
    heading_is_measured: bool = True

    @property
    def asset_id(self) -> str:
        return f"{self.source}-{self.source_id}"


class Collector(Protocol):
    """Contrat d'un collecteur de médias."""

    name: str

    def collect(self, lat: float, lon: float, radius_m: int) -> list[CollectedImage]: ...


def download(image: CollectedImage, destination: Path) -> Path:
    """Télécharge une image si elle n'est pas déjà présente.

    L'idempotence est structurelle : un second passage ne retélécharge rien,
    ce qui rend la collecte rejouable sans coût d'API supplémentaire.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        image.local_path = destination
        return destination

    response = requests.get(image.url, timeout=DOWNLOAD_TIMEOUT, stream=True)
    response.raise_for_status()

    tmp = destination.with_suffix(destination.suffix + ".part")
    with tmp.open("wb") as handle:
        for chunk in response.iter_content(1 << 16):
            handle.write(chunk)
    tmp.replace(destination)

    image.local_path = destination
    return destination


def to_asset(image: CollectedImage, assume_rights: bool = False) -> Asset:
    """Convertit une image collectée en `Asset`, droits appliqués depuis POLICIES.

    `assume_rights` reflète la décision de l'opérateur d'assumer l'usage de
    sources aux droits non établis. Elle n'est appliquée qu'aux sources qui
    l'exigent, et laisse une trace dans l'asset.
    """
    policy = POLICIES[image.source]
    encumbered = bool(assume_rights and policy.requires_assumption)

    checksum = (
        _sha256(image.local_path)
        if image.local_path and image.local_path.is_file()
        else "0" * 64
    )

    return Asset(
        id=image.asset_id,
        source=image.source,
        source_url_or_id=image.url or image.source_id,
        rights=policy.rights,
        ai_eligible=False,
        confidence=0.5,
        category=AssetCategory.OTHER,
        capture_year=image.captured_year,
        checksum=checksum,
        exterior_or_interior=ExteriorInterior.UNKNOWN,
        attribution=policy.attribution,
        heading_deg=image.heading_deg,
        heading_is_measured=image.heading_is_measured,
        camera_lat=image.lat,
        camera_lon=image.lon,
        local_path=str(image.local_path) if image.local_path else None,
        rights_encumbered=encumbered,
        rights_note=policy.note,
        production_eligible=False,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
