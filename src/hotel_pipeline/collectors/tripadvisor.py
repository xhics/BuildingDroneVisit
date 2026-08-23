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

#: Le quota TripAdvisor est **mensuel et plafonné** (1 000 appels sur l'offre
#: gratuite). Une réponse expirée serait redemandée et consommerait le quota
#: pour un contenu qui n'a pas bougé : les photos d'un établissement changent
#: à l'échelle de l'année, pas de la semaine. On conserve donc les réponses
#: bien au-delà du TTL par défaut (7 jours), et `--refresh` reste le moyen
#: explicite de forcer une nouvelle interrogation.
CACHE_TTL_SECONDS = 365 * 24 * 3600

#: Maximum accepté par `/location/{id}/photos`. Le quota se compte en
#: **appels**, pas en photos : demander 5 clichés coûte exactement le même
#: appel que d'en demander 50. Plafonner par défaut à 20 laissait donc des
#: photos sur la table sans rien économiser.
MAX_PHOTOS_PER_CALL = 50


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
        ttl=CACHE_TTL_SECONDS,
    )
    results = payload.get("data") or []
    return results[0] if results else None


def collect(query: str, limit: int = MAX_PHOTOS_PER_CALL) -> list[CollectedImage]:
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
        ttl=CACHE_TTL_SECONDS,
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


def photo_url(provider_id: str, resolution: str = "original") -> str:
    """Adresse d'un cliché, relue depuis la réponse déjà en cache.

    La Content API ne sait pas rendre **un** cliché par identifiant : elle
    publie la liste des photos d'un établissement. On relit donc la réponse
    mémorisée par `collect` plutôt que de dépenser un appel de quota par
    image — c'est précisément ce que le cache d'un an sert à éviter.

    `resolution` est déjà traduite dans le vocabulaire de la source par
    `PROVIDER_RESOLUTIONS` : « thumbnail », « small », « medium », « large »
    ou « original ».
    """
    from ..providers.cache import get_cache

    cache = get_cache()
    for key in cache.iterkeys():
        if not str(key).startswith("tripadvisor-photos::"):
            continue
        payload = cache.get(key) or {}
        for entry in payload.get("data") or []:
            if str(entry.get("id")) != str(provider_id):
                continue
            sizes = entry.get("images") or {}
            chosen = sizes.get(resolution) or {}
            url = chosen.get("url")
            if url:
                return str(url)
            raise ValueError(
                f"taille {resolution!r} absente pour le cliché "
                f"{provider_id!r} ; connues : "
                f"{sorted(k for k, v in sizes.items() if (v or {}).get('url'))}"
            )

    raise ValueError(
        f"cliché TripAdvisor {provider_id!r} absent du cache : relancez "
        "`assets discover` pour réinterroger la fiche"
    )
