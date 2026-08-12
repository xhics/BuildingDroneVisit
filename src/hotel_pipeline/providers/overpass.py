"""Empreintes de bâtiments et stationnements via Overpass (plan directeur §9).

L'instance publique Overpass est limitée en débit et souvent congestionnée
(complément §5). Acceptable pour un hôtel, fragile à l'échelle : le repli par
extrait OSM régional est un point ouvert, pas une urgence du Lot 1.
"""

from __future__ import annotations

import os
from typing import Any

import requests

from ..logging import get_logger
from .cache import cached_call, ensure_online

log = get_logger("overpass")

DEFAULT_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "hotel-pipeline/0.1 (reconstruction 3D, contact via dépôt)"
TIMEOUT = 120


class OverpassError(RuntimeError):
    pass


def _query(ql: str) -> dict[str, Any]:
    url = os.environ.get("OVERPASS_URL", DEFAULT_URL)
    ensure_online(f"Overpass {url}")
    response = requests.post(
        url, data={"data": ql}, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT
    )
    response.raise_for_status()
    payload = response.json()
    if "elements" not in payload:
        raise OverpassError("réponse Overpass sans 'elements'")
    return payload


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
