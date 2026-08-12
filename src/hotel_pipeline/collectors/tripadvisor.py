"""Collecteur TripAdvisor via la Content API.

L'API expose `/location/{id}/photos`. Le volume est faible — quelques clichés
par établissement — mais ce sont des photos de voyageurs, donc des points de
vue différents des visuels promotionnels du site officiel.

Les droits appartiennent aux déposants, et TripAdvisor impose l'attribution :
la source relève donc de la décision d'assumer l'usage.
"""

from __future__ import annotations

import requests

from ..config import secret
from ..logging import get_logger
from ..providers.cache import cached_call, ensure_online
from .base import CollectedImage

log = get_logger("tripadvisor")

BASE_URL = "https://api.content.tripadvisor.com/api/v1"
TIMEOUT = 30

name = "tripadvisor"


def _get(path: str, params: dict) -> dict:
    response = requests.get(
        f"{BASE_URL}/{path}",
        params=params,
        headers={"accept": "application/json", "Referer": "https://localhost"},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def find_location(query: str, key: str) -> dict | None:
    payload = cached_call(
        f"tripadvisor-search::{query}",
        lambda: _get(
            "location/search",
            {"key": key, "searchQuery": query, "category": "hotels", "language": "fr"},
        ),
    )
    results = payload.get("data") or []
    return results[0] if results else None


def collect(query: str, limit: int = 20) -> list[CollectedImage]:
    """Photos d'un établissement identifié par recherche textuelle."""
    ensure_online("TripAdvisor")
    key = secret("TRIPADVISOR_API_KEY")

    location = find_location(query, key)
    if location is None:
        log.warning("aucune fiche TripAdvisor pour %r", query)
        return []

    location_id = location["location_id"]
    log.info(
        "fiche TripAdvisor %s : %s", location_id, location.get("name", "?")
    )

    payload = cached_call(
        f"tripadvisor-photos::{location_id}::{limit}",
        lambda: _get(
            f"location/{location_id}/photos", {"key": key, "limit": limit, "language": "fr"}
        ),
    )

    images: list[CollectedImage] = []
    for entry in payload.get("data") or []:
        # `images` propose plusieurs tailles ; on prend la plus grande utile.
        sizes = entry.get("images") or {}
        best = sizes.get("original") or sizes.get("large") or sizes.get("medium") or {}
        url = best.get("url")
        if not url:
            continue

        images.append(
            CollectedImage(
                source=name,
                source_id=str(entry.get("id")),
                url=url,
                captured_year=_year(entry.get("published_date")),
                extra={
                    "caption": (entry.get("caption") or "")[:200],
                    "attribution": ((entry.get("user") or {}).get("username") or ""),
                    "album": entry.get("album", ""),
                },
            )
        )

    log.info("TripAdvisor : %d photo(s)", len(images))
    return images


def _year(published: str | None) -> int | None:
    if not published:
        return None
    try:
        return int(str(published)[:4])
    except ValueError:
        return None
