"""Collecteur Wikimedia Commons par géolocalisation.

Aucune clé requise, licences ouvertes et explicites : c'est la source la moins
coûteuse à interroger. Le volume attendu autour d'un hôtel d'autoroute est
faible, mais chaque image y est exploitable sans arbitrage de droits.
"""

from __future__ import annotations

import requests

from ..logging import get_logger
from ..providers.cache import cached_call, ensure_online
from .base import CollectedImage

log = get_logger("commons")

API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "hotel-pipeline/0.1 (reconstruction 3D, contact via dépôt)"
TIMEOUT = 30

name = "commons"


def collect(lat: float, lon: float, radius_m: int = 500, limit: int = 100) -> list[CollectedImage]:
    """Images géolocalisées autour d'un point."""
    ensure_online("Wikimedia Commons")

    params = {
        "action": "query",
        "format": "json",
        "generator": "geosearch",
        "ggscoord": f"{lat}|{lon}",
        # L'API plafonne le rayon de géorecherche à 10 km.
        "ggsradius": min(radius_m, 10_000),
        "ggslimit": limit,
        "ggsnamespace": 6,  # espace Fichier
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|size",
        "iiurlwidth": 2048,
    }

    def fetch() -> dict:
        response = requests.get(
            API_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT
        )
        response.raise_for_status()
        return response.json()

    payload = cached_call(f"commons::{lat:.5f}::{lon:.5f}::{radius_m}::{limit}", fetch)
    pages = (payload.get("query") or {}).get("pages") or {}

    images: list[CollectedImage] = []
    for page in pages.values():
        info = (page.get("imageinfo") or [{}])[0]
        url = info.get("thumburl") or info.get("url")
        if not url:
            continue

        metadata = info.get("extmetadata") or {}
        images.append(
            CollectedImage(
                source=name,
                source_id=str(page.get("pageid")),
                url=url,
                captured_year=_year(metadata),
                extra={
                    "title": page.get("title", ""),
                    "licence": (metadata.get("LicenseShortName") or {}).get("value", ""),
                    "artist": (metadata.get("Artist") or {}).get("value", "")[:200],
                },
            )
        )

    log.info("Wikimedia Commons : %d image(s) dans un rayon de %d m", len(images), radius_m)
    return images


def _year(metadata: dict) -> int | None:
    raw = (metadata.get("DateTimeOriginal") or metadata.get("DateTime") or {}).get("value", "")
    for token in str(raw).replace("-", " ").replace(":", " ").split():
        if token.isdigit() and 1900 <= int(token) <= 2100:
            return int(token)
    return None
