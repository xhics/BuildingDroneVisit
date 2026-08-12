"""Collecteur Mapillary (API Graph v4).

Source la plus utile du dispositif : c'est la seule imagerie de rue sous
licence ouverte (CC BY-SA), donc la seule qui alimente la reconstruction sans
décision de droits. Sa couverture décide en pratique de la viabilité du
photo-first.
"""

from __future__ import annotations

from datetime import datetime, timezone

import requests

from ..config import secret
from ..logging import get_logger
from ..providers.cache import cached_call, ensure_online
from .base import CollectedImage

log = get_logger("mapillary")

GRAPH_URL = "https://graph.mapillary.com/images"
TIMEOUT = 60
FIELDS = "id,captured_at,compass_angle,geometry,thumb_2048_url,is_pano"

name = "mapillary"


def _bbox(lat: float, lon: float, radius_m: int) -> str:
    """Boîte englobante approchée — l'API Graph filtre par bbox, pas par rayon."""
    import math

    dlat = radius_m / 110_540.0
    dlon = radius_m / (111_320.0 * math.cos(math.radians(lat)))
    return f"{lon - dlon},{lat - dlat},{lon + dlon},{lat + dlat}"


def collect(lat: float, lon: float, radius_m: int = 300, limit: int = 200) -> list[CollectedImage]:
    """Images Mapillary dans un rayon autour du bâtiment."""
    ensure_online("Mapillary")
    token = secret("MAPILLARY_TOKEN")
    bbox = _bbox(lat, lon, radius_m)

    def fetch() -> dict:
        response = requests.get(
            GRAPH_URL,
            params={"fields": FIELDS, "bbox": bbox, "limit": limit},
            headers={"Authorization": f"OAuth {token}"},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    # La clé de cache exclut le jeton : deux jetons différents ne doivent pas
    # produire deux entrées, et aucun secret ne doit atterrir sur le disque.
    payload = cached_call(f"mapillary::{bbox}::{limit}", fetch)

    images: list[CollectedImage] = []
    for entry in payload.get("data", []):
        url = entry.get("thumb_2048_url")
        if not url:
            continue

        geometry = entry.get("geometry") or {}
        coordinates = geometry.get("coordinates") or [None, None]

        images.append(
            CollectedImage(
                source=name,
                source_id=str(entry["id"]),
                url=url,
                captured_year=_year(entry.get("captured_at")),
                heading_deg=_normalise_heading(entry.get("compass_angle")),
                lon=coordinates[0],
                lat=coordinates[1],
                extra={"is_pano": str(entry.get("is_pano", False))},
            )
        )

    log.info("Mapillary : %d image(s) dans un rayon de %d m", len(images), radius_m)
    return images


def _year(captured_at: int | None) -> int | None:
    """`captured_at` est un horodatage en millisecondes."""
    if not captured_at:
        return None
    try:
        return datetime.fromtimestamp(int(captured_at) / 1000, tz=timezone.utc).year
    except (ValueError, OSError, OverflowError):
        return None


def _normalise_heading(angle: float | None) -> float | None:
    if angle is None:
        return None
    return float(angle) % 360.0
