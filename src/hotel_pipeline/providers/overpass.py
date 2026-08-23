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


def roads_around(lat: float, lon: float, radius_m: int = 350) -> list[dict[str, Any]]:
    """Voies carrossables et accès dans un rayon.

    Les allées de service et les accès de stationnement sont inclus
    volontairement : ce sont eux qui pénètrent la propriété et peuvent offrir
    des points de vue que la voie publique n'a pas.
    """
    ql = f"""
    [out:json][timeout:{TIMEOUT}];
    (
      way["highway"~"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|living_street)$"](around:{radius_m},{lat},{lon});
    );
    out geom tags;
    """.strip()

    payload = cached_call(f"overpass-roads::{lat:.6f}::{lon:.6f}::{radius_m}", lambda: _query(ql))
    elements = payload["elements"]
    log.info("réseau routier : %d voie(s) dans un rayon de %d m", len(elements), radius_m)
    return elements


def way_by_id(way_id: int) -> list[dict[str, Any]]:
    """Résout une voie précise, avec sa géométrie complète.

    Le cache de collecte ne contient que bâtiments et stationnements : y
    chercher une voie rendrait une absence qui n'en est pas une.
    """
    ql = f"""
    [out:json][timeout:{TIMEOUT}];
    way({way_id});
    out geom tags;
    """.strip()

    payload = cached_call(f"overpass-way::{way_id}", lambda: _query(ql))
    return payload["elements"]


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

#: Emprises décrivant le sol et les plantations d'un site. La requête des
#: bâtiments ne les demande pas : sur le pilote, cinquante-six éléments
#: collectés ne portaient pas un seul tag de végétation, alors que pelouses et
#: massifs sont cartographiés en amont. Ce qui n'est pas demandé n'arrive pas.
GROUND_SELECTORS: tuple[str, ...] = (
    'way["landuse"~"^(grass|meadow|forest|village_green|greenfield|farmland)$"]',
    'way["natural"~"^(wood|scrub|grassland|tree_row|water|wetland)$"]',
    'way["leisure"~"^(garden|park|pitch|golf_course|playground)$"]',
    'way["surface"~"^(grass|dirt|gravel|paving_stones|concrete|asphalt)$"]',
    # Un arbre isolé est un nœud, non une emprise : il compte pourtant, car il
    # occulte une façade à lui seul devant une entrée.
    'node["natural"="tree"]',
    'relation["landuse"~"^(grass|forest|meadow)$"]',
    'relation["leisure"~"^(garden|park)$"]',
)


def ground_around(
    lat: float, lon: float, radius_m: int = 300
) -> list[dict[str, Any]]:
    """Sol, plantations et arbres isolés dans un rayon, avec leur géométrie.

    Séparé de `features_around` à dessein : élargir la requête des bâtiments
    aurait laissé la clé de cache inchangée, et les anciennes réponses — sans
    végétation — auraient été resservies indéfiniment. Une requête distincte a
    sa propre clé, et le cache existant reste valide pour ce qu'il décrit.
    """
    selectors = "\n      ".join(
        f"{selector}(around:{radius_m},{lat},{lon});" for selector in GROUND_SELECTORS
    )
    ql = f"""
    [out:json][timeout:{TIMEOUT}];
    (
      {selectors}
    );
    out geom tags;
    """.strip()

    payload = cached_call(
        f"overpass-ground::{lat:.6f}::{lon:.6f}::{radius_m}", lambda: _query(ql)
    )
    elements = payload["elements"]
    log.info(
        "Overpass a retourné %d élément(s) de sol dans un rayon de %d m",
        len(elements),
        radius_m,
    )
    return elements
