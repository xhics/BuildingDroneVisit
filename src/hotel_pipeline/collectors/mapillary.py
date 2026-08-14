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

#: Taille de page demandée à l'API.
PAGE_SIZE = 200

#: Plafond de sécurité : une zone urbaine dense peut contenir des milliers de
#: clichés, dont l'immense majorité ne regarde pas le bâtiment.
MAX_IMAGES = 1500

name = "mapillary"


def _bbox(lat: float, lon: float, radius_m: int) -> str:
    """Boîte englobante approchée — l'API Graph filtre par bbox, pas par rayon."""
    import math

    dlat = radius_m / 110_540.0
    dlon = radius_m / (111_320.0 * math.cos(math.radians(lat)))
    return f"{lon - dlon},{lat - dlat},{lon + dlon},{lat + dlat}"


def collect(
    lat: float,
    lon: float,
    radius_m: int = 300,
    page_size: int = PAGE_SIZE,
    max_images: int = MAX_IMAGES,
) -> list[CollectedImage]:
    """Images Mapillary dans un rayon autour du bâtiment.

    La pagination est indispensable : l'API rend une page bornée, et sur un
    rayon large elle est atteinte avant d'avoir couvert la zone. S'arrêter à la
    première page tronque silencieusement le corpus — et écarte des vues
    proches du bâtiment au profit d'images lointaines arrivées en tête.
    """
    ensure_online("Mapillary")
    token = secret("MAPILLARY_TOKEN")
    bbox = _bbox(lat, lon, radius_m)

    def fetch_all() -> list[dict]:
        entries: list[dict] = []
        url = GRAPH_URL
        params: dict | None = {"fields": FIELDS, "bbox": bbox, "limit": page_size}

        while url and len(entries) < max_images:
            response = requests.get(
                url,
                params=params,
                headers={"Authorization": f"OAuth {token}"},
                timeout=TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
            entries.extend(payload.get("data", []))

            # Le curseur `next` porte déjà tous ses paramètres.
            url = ((payload.get("paging") or {}).get("next")) or ""
            params = None

        return entries[:max_images]

    # La clé de cache exclut le jeton : deux jetons différents ne doivent pas
    # produire deux entrées, et aucun secret ne doit atterrir sur le disque.
    data = cached_call(f"mapillary::{bbox}::{page_size}::{max_images}", fetch_all)

    images: list[CollectedImage] = []
    for entry in data:
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


def image_metadata(image_ids: list[str]) -> dict[str, dict]:
    """Métadonnées d'images précises — séquence comprise.

    Les assets historiques ne portent ni `sequence_id` ni provenance : leur
    corrélation ne peut donc pas être affirmée depuis le corpus. Elle se
    demande à la source, image par image.
    """
    import requests

    ensure_online("Mapillary Graph")
    token = secret("MAPILLARY_TOKEN")
    fields = "id,captured_at,sequence,compass_angle,computed_compass_angle,geometry"
    found: dict[str, dict] = {}

    for image_id in image_ids:
        response = requests.get(
            f"https://graph.mapillary.com/{image_id}",
            params={"fields": fields},
            headers={"Authorization": f"OAuth {token}"},
            timeout=TIMEOUT,
        )
        if response.status_code == 404:
            continue
        response.raise_for_status()
        found[image_id] = response.json()

    log.info("métadonnées Mapillary : %d/%d image(s) retrouvée(s)", len(found), len(image_ids))
    return found


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
