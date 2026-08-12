"""Collecteur Street View Static.

Deux points de conception :

1. l'endpoint `metadata` est **gratuit** et dit si un panorama existe avant de
   déclencher une requête image facturée. Il est donc systématiquement appelé
   d'abord ;
2. Street View ne rend pas une liste d'images mais une vue par cap. Les caps
   sont donc échantillonnés autour du bâtiment, et la déduplication en aval
   écarte les vues redondantes.
"""

from __future__ import annotations

import requests

from ..config import secret
from ..logging import get_logger
from ..providers.cache import cached_call, ensure_online
from .base import CollectedImage

log = get_logger("streetview")

IMAGE_URL = "https://maps.googleapis.com/maps/api/streetview"
METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
TIMEOUT = 30

#: Caps échantillonnés, en degrés. Huit vues suffisent à couvrir un bâtiment.
DEFAULT_HEADINGS: tuple[int, ...] = (0, 45, 90, 135, 180, 225, 270, 315)
SIZE = "640x640"
FOV = 80

name = "street_view"


def _metadata(lat: float, lon: float, radius_m: int, key: str) -> dict:
    def fetch() -> dict:
        response = requests.get(
            METADATA_URL,
            params={"location": f"{lat},{lon}", "radius": radius_m, "key": key},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    return cached_call(f"streetview-meta::{lat:.6f}::{lon:.6f}::{radius_m}", fetch)


def collect(
    lat: float, lon: float, radius_m: int = 60, headings: tuple[int, ...] = DEFAULT_HEADINGS
) -> list[CollectedImage]:
    """Vues Street View autour d'un point, un cap par image."""
    ensure_online("Street View")
    key = secret("GOOGLE_MAPS_API_KEY")

    metadata = _metadata(lat, lon, radius_m, key)
    status = metadata.get("status")
    if status != "OK":
        log.warning("Street View sans panorama utilisable (%s)", status)
        return []

    pano_id = metadata.get("pano_id", "")
    pano_lat = (metadata.get("location") or {}).get("lat", lat)
    pano_lon = (metadata.get("location") or {}).get("lng", lon)
    year = _year(metadata.get("date"))

    images = [
        CollectedImage(
            source=name,
            source_id=f"{pano_id or 'pano'}-{heading:03d}",
            # La clé n'est pas stockée dans l'URL du manifeste : elle est
            # ajoutée au moment du téléchargement seulement.
            url=(
                f"{IMAGE_URL}?size={SIZE}&location={lat},{lon}"
                f"&heading={heading}&fov={FOV}&pitch=0&radius={radius_m}"
            ),
            captured_year=year,
            heading_deg=float(heading),
            lat=pano_lat,
            lon=pano_lon,
            extra={"pano_id": pano_id},
        )
        for heading in headings
    ]

    log.info("Street View : %d vue(s) depuis le panorama %s", len(images), pano_id or "?")
    return images


def sign_url(image: CollectedImage) -> str:
    """Ajoute la clé à l'URL, au moment du téléchargement uniquement."""
    return f"{image.url}&key={secret('GOOGLE_MAPS_API_KEY')}"


def _year(date: str | None) -> int | None:
    """`date` est au format AAAA-MM."""
    if not date:
        return None
    try:
        return int(date.split("-")[0])
    except (ValueError, IndexError):
        return None
