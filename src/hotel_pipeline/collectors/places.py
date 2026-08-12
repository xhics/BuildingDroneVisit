"""Collecteur Google Places — photos déposées sur la fiche de l'établissement.

Source la plus directement liée à l'hôtel, donc la plus utile pour dater
l'entrée rénovée — mais les droits appartiennent aux déposants.
"""

from __future__ import annotations

import requests

from ..config import secret
from ..logging import get_logger
from ..providers.cache import cached_call, ensure_online
from .base import CollectedImage

log = get_logger("places")

SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
MEDIA_URL = "https://places.googleapis.com/v1"
TIMEOUT = 30
MAX_WIDTH = 1600

name = "places"


def find_place(query: str, key: str) -> dict | None:
    """Résout un établissement par recherche textuelle."""

    def fetch() -> dict:
        response = requests.post(
            SEARCH_URL,
            json={"textQuery": query},
            headers={
                "X-Goog-Api-Key": key,
                "X-Goog-FieldMask": (
                    "places.id,places.displayName,places.formattedAddress,places.photos"
                ),
                "Content-Type": "application/json",
            },
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    payload = cached_call(f"places-search::{query}", fetch)
    places = payload.get("places") or []
    return places[0] if places else None


def collect(query: str, limit: int = 20) -> list[CollectedImage]:
    """Photos de la fiche Places d'un établissement."""
    ensure_online("Places")
    key = secret("GOOGLE_PLACES_API_KEY")

    place = find_place(query, key)
    if place is None:
        log.warning("aucune fiche Places pour %r", query)
        return []

    log.info(
        "fiche Places : %s — %s",
        (place.get("displayName") or {}).get("text", "?"),
        place.get("formattedAddress", "?"),
    )

    images = []
    for index, photo in enumerate((place.get("photos") or [])[:limit]):
        photo_name = photo.get("name")
        if not photo_name:
            continue
        images.append(
            CollectedImage(
                source=name,
                source_id=f"{place['id']}-{index:03d}",
                url=f"{MEDIA_URL}/{photo_name}/media?maxWidthPx={MAX_WIDTH}",
                extra={
                    "place_id": place["id"],
                    "attributions": ", ".join(
                        a.get("displayName", "") for a in (photo.get("authorAttributions") or [])
                    ),
                },
            )
        )

    log.info("Places : %d photo(s)", len(images))
    return images


def sign_url(image: CollectedImage) -> str:
    return f"{image.url}&key={secret('GOOGLE_PLACES_API_KEY')}"
