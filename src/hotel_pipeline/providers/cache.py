"""Cache des appels externes (plan directeur §18, complément §3.2).

Le cache vit côté poste local : placé sur le Container Disk d'une VM détruite
à chaque session, il n'aurait aucun effet.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import diskcache

#: Une semaine. Les empreintes de bâtiments et les adresses bougent peu.
DEFAULT_TTL_SECONDS = 7 * 24 * 3600


class OfflineError(RuntimeError):
    """Appel réseau tenté alors que le mode hors ligne est actif."""


def ensure_online(what: str) -> None:
    """Interdit tout appel réseau si ``HOTEL_PIPELINE_OFFLINE=1``.

    Garde-fou de la suite de tests : le §17 du plan directeur exige des tests
    unitaires « rapides et sans réseau ». Plutôt que de compter sur la
    discipline des auteurs de tests, l'appel échoue bruyamment.
    """
    if os.environ.get("HOTEL_PIPELINE_OFFLINE") == "1":
        raise OfflineError(f"mode hors ligne actif — appel {what} refusé")


def cache_dir() -> Path:
    return Path(os.environ.get("HOTEL_PIPELINE_CACHE", ".cache")).resolve()


_cache: diskcache.Cache | None = None


def get_cache() -> diskcache.Cache:
    global _cache
    if _cache is None:
        _cache = diskcache.Cache(str(cache_dir()))
    return _cache


def cached_call(key: str, producer: Callable[[], Any], ttl: int = DEFAULT_TTL_SECONDS) -> Any:
    """Retourne la valeur en cache, ou l'obtient et la mémorise.

    ``HOTEL_PIPELINE_NO_CACHE=1`` court-circuite entièrement le cache.
    """
    if os.environ.get("HOTEL_PIPELINE_NO_CACHE") == "1":
        return producer()

    cache = get_cache()
    hit = cache.get(key, default=None)
    if hit is not None:
        return hit

    value = producer()
    cache.set(key, value, expire=ttl)
    return value
