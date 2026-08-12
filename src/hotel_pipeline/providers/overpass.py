"""Empreintes de bâtiments et stationnements via Overpass (plan directeur §9).

L'instance publique Overpass est limitée en débit et souvent congestionnée
(complément §5). Acceptable pour un hôtel, fragile à l'échelle : le repli par
extrait OSM régional est un point ouvert, pas une urgence du Lot 1.
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests

from ..logging import get_logger
from .cache import cached_call, ensure_online

log = get_logger("overpass")

#: Miroirs essayés dans l'ordre. L'instance principale est régulièrement
#: congestionnée et répond alors 429 ou 504 (complément §5).
MIRRORS: tuple[str, ...] = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)
USER_AGENT = "hotel-pipeline/0.1 (reconstruction 3D, contact via dépôt)"
TIMEOUT = 180
ATTEMPTS_PER_MIRROR = 2
BACKOFF_SECONDS = 5

#: Codes traduisant une congestion passagère, non une requête fautive.
TRANSIENT_STATUS = frozenset({429, 502, 503, 504})


class OverpassError(RuntimeError):
    pass


def _endpoints() -> tuple[str, ...]:
    override = os.environ.get("OVERPASS_URL", "").strip()
    return (override,) if override else MIRRORS


def _post(url: str, ql: str) -> dict[str, Any]:
    response = requests.post(
        url, data={"data": ql}, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT
    )
    if response.status_code in TRANSIENT_STATUS:
        raise OverpassError(f"{url} occupé ({response.status_code})")
    response.raise_for_status()
    payload = response.json()
    if "elements" not in payload:
        raise OverpassError(f"réponse sans 'elements' depuis {url}")
    return payload


def _query(ql: str) -> dict[str, Any]:
    """Interroge Overpass, en essayant chaque miroir avec reprise.

    Un 504 sur l'instance publique est un incident courant, pas une erreur de
    requête : il ne doit pas faire échouer une étape de plusieurs minutes.
    """
    endpoints = _endpoints()
    ensure_online(f"Overpass {endpoints[0]}")

    failures: list[str] = []
    for url in endpoints:
        for attempt in range(1, ATTEMPTS_PER_MIRROR + 1):
            try:
                return _post(url, ql)
            except (OverpassError, requests.RequestException) as exc:
                failures.append(f"{url} (essai {attempt}) : {exc}")
                log.warning("Overpass indisponible — %s", exc)
                if attempt < ATTEMPTS_PER_MIRROR:
                    time.sleep(BACKOFF_SECONDS * attempt)

    raise OverpassError(
        "aucun miroir Overpass n'a répondu.\n  "
        + "\n  ".join(failures)
        + "\n  Réessayez plus tard, ou fixez OVERPASS_URL vers une instance dédiée."
    )


def features_around(lat: float, lon: float, radius_m: int = 500) -> list[dict[str, Any]]:
    """Bâtiments et stationnements dans un rayon, avec leur géométrie.

    ``out geom`` fournit les coordonnées des nœuds directement, ce qui évite un
    second aller-retour pour résoudre les références.
    """
    ql = f"""
    [out:json][timeout:{TIMEOUT}];
    (
      way["building"](around:{radius_m},{lat},{lon});
      way["amenity"="parking"](around:{radius_m},{lat},{lon});
      relation["building"](around:{radius_m},{lat},{lon});
    );
    out geom tags;
    """.strip()

    payload = cached_call(f"overpass::{lat:.6f}::{lon:.6f}::{radius_m}", lambda: _query(ql))
    elements = payload["elements"]
    log.info("Overpass a retourné %d éléments dans un rayon de %d m", len(elements), radius_m)
    return elements
