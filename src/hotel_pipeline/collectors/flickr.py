"""Collecteur Flickr, restreint aux licences Creative Commons.

Intérêt propre à cette source : l'API filtre **par licence**. On peut donc
n'obtenir que des images réellement exploitables en reconstruction, sans
arbitrage de droits — comme Mapillary, mais avec des points de vue de piétons
plutôt que de véhicules.
"""

from __future__ import annotations

import requests

from ..config import secret
from ..logging import get_logger
from ..providers.cache import cached_call, ensure_online
from .base import CollectedImage

log = get_logger("flickr")

API_URL = "https://api.flickr.com/services/rest/"
TIMEOUT = 30

#: Identifiants de licences Creative Commons et domaine public.
#: 1 BY-NC-SA · 2 BY-NC · 3 BY-NC-ND · 4 BY · 5 BY-SA · 6 BY-ND
#: 7 domaine public · 9 CC0 · 10 domaine public marqué
CC_LICENSES = "1,2,3,4,5,6,7,9,10"

name = "flickr"


def collect(
    lat: float, lon: float, radius_m: int = 500, limit: int = 100
) -> list[CollectedImage]:
    """Photos géolocalisées sous licence ouverte autour d'un point."""
    ensure_online("Flickr")
    key = secret("FLICKR_API_KEY")

    params = {
        "method": "flickr.photos.search",
        "api_key": key,
        "lat": lat,
        "lon": lon,
        # L'API attend un rayon en kilomètres, plafonné à 32.
        "radius": min(radius_m / 1000.0, 32),
        "radius_units": "km",
        "license": CC_LICENSES,
        "extras": "url_l,url_o,geo,date_taken,license,owner_name",
        "per_page": min(limit, 500),
        "format": "json",
        "nojsoncallback": 1,
    }

    def fetch() -> dict:
        response = requests.get(API_URL, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()

    payload = cached_call(f"flickr::{lat:.5f}::{lon:.5f}::{radius_m}::{limit}", fetch)

    if payload.get("stat") != "ok":
        raise RuntimeError(f"Flickr : {payload.get('message', 'réponse inattendue')}")

    images: list[CollectedImage] = []
    for entry in (payload.get("photos") or {}).get("photo", []):
        url = entry.get("url_o") or entry.get("url_l")
        if not url:
            continue

        images.append(
            CollectedImage(
                source=name,
                source_id=str(entry["id"]),
                url=url,
                captured_year=_year(entry.get("datetaken")),
                lat=_float(entry.get("latitude")),
                lon=_float(entry.get("longitude")),
                extra={
                    "title": (entry.get("title") or "")[:200],
                    "owner": entry.get("ownername", ""),
                    "license": str(entry.get("license", "")),
                },
            )
        )

    log.info("Flickr : %d image(s) sous licence ouverte", len(images))
    return images


def _year(taken: str | None) -> int | None:
    try:
        return int(str(taken)[:4]) if taken else None
    except ValueError:
        return None


def _float(value) -> float | None:  # noqa: ANN001
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result != 0.0 else None
