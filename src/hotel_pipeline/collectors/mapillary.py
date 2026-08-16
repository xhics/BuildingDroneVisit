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
from ..providers import transport
from ..providers.cache import cached_call, ensure_online
from .base import CollectedImage

log = get_logger("mapillary")

GRAPH_URL = "https://graph.mapillary.com/images"
TIMEOUT = 60
#: `sequence` vient dans la **même** requête : l'enrichissement de séquence
#: n'exige donc aucun appel supplémentaire. Sans lui, la continuité restait
#: inconnue, et tout besoin l'exigeant demeurait borné à l'aperçu.
#: `sequence`, `camera_*` et les dimensions viennent dans la **même** requête :
#: aucun appel supplémentaire. Sans le champ de vision ni la largeur, aucun
#: cadrage n'était calculable, et les 194 images du pilote restaient bornées à
#: l'aperçu quoi qu'elles montrent.
FIELDS = (
    "id,captured_at,compass_angle,geometry,thumb_2048_url,is_pano,sequence,"
    "camera_type,camera_parameters,width,height"
)

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

        page = 0
        while url and len(entries) < max_images:
            page += 1
            # Chaque page est une requête : les fondre en une seule sous-
            # estimait le coût d'un facteur égal au nombre de pages.
            response = transport.get(
                "mapillary", transport.Stage.COARSE_SEARCH, url,
                params=params,
                headers={"Authorization": f"OAuth {token}"},
                timeout=TIMEOUT,
                page=page,
                what="Mapillary Graph (recherche)",
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
                sequence_id=(
                    str(entry["sequence"]) if entry.get("sequence") else None
                ),
                camera_type=entry.get("camera_type"),
                fov_deg=_horizontal_fov(entry),
                width_px=entry.get("width"),
                height_px=entry.get("height"),
                extra={"is_pano": str(entry.get("is_pano", False))},
            )
        )

    log.info("Mapillary : %d image(s) dans un rayon de %d m", len(images), radius_m)
    return images


#: Résolutions que l'API Graph sait rendre. Une autre valeur ne se devine pas :
#: `thumb_9999_url` ne renverrait rien, et le silence passerait pour une panne.
THUMBNAIL_FIELDS: dict[str, str] = {
    "thumb_256": "thumb_256_url",
    "thumb_1024": "thumb_1024_url",
    "thumb_2048": "thumb_2048_url",
    "thumb_original": "thumb_original_url",
}


def thumbnail_url(image_id: str, resolution: str = "thumb_2048") -> str:
    """Adresse d'une vignette, **redemandée au moment du téléchargement**.

    Mapillary ne publie pas d'URL durable : celle-ci vaut quelques minutes.
    C'est pourquoi aucun manifeste n'en conserve — seul de quoi la reconstruire
    y figure, et cette fonction est le point où elle réapparaît.
    """
    field = THUMBNAIL_FIELDS.get(resolution)
    if field is None:
        raise ValueError(
            f"résolution {resolution!r} inconnue de l'API Graph ; "
            f"disponibles : {sorted(THUMBNAIL_FIELDS)}"
        )

    # Résolution d'adresse : un appel **distinct** de celui qui suivra sur le
    # CDN. Les confondre ferait passer un HEAD Mapillary pour une requête là
    # où il en coûte deux.
    response = transport.get(
        "mapillary", transport.Stage.URL_RESOLUTION,
        f"https://graph.mapillary.com/{image_id}",
        params={"fields": field},
        headers={"Authorization": f"OAuth {secret('MAPILLARY_TOKEN')}"},
        timeout=TIMEOUT,
        what="Mapillary Graph (adresse)",
    )
    response.raise_for_status()
    url = response.json().get(field)
    if not url:
        raise ValueError(
            f"image {image_id} : l'API ne rend pas de {resolution!r} — la vue "
            "existe peut-être dans une autre résolution"
        )
    return url


def image_metadata(image_ids: list[str]) -> dict[str, dict]:
    """Métadonnées d'images précises — séquence comprise.

    Les assets historiques ne portent ni `sequence_id` ni provenance : leur
    corrélation ne peut donc pas être affirmée depuis le corpus. Elle se
    demande à la source, image par image.
    """
    token = secret("MAPILLARY_TOKEN")
    fields = "id,captured_at,sequence,compass_angle,computed_compass_angle,geometry"
    found: dict[str, dict] = {}

    for image_id in image_ids:
        # Un appel par image : c'est le coût de l'enrichissement, et il doit
        # figurer au registre comme tel.
        response = transport.get(
            "mapillary", transport.Stage.METADATA_ENRICHMENT,
            f"https://graph.mapillary.com/{image_id}",
            what="Mapillary Graph (métadonnées)",
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


#: Champ horizontal d'une image sphérique : elle voit tout autour.
PANORAMIC_FOV_DEG = 360.0


def _horizontal_fov(entry: dict) -> float | None:
    """Champ de vision horizontal, **dérivé** des paramètres publiés.

    Mapillary ne publie pas d'angle : il publie un rapport focal, exprimé en
    fraction de la plus grande dimension de l'image. L'angle s'en déduit, et
    sans lui aucun cadrage n'est calculable — la vue reste utilisable, mais
    rien ne dit ce qu'elle montre.

    Rend `None` dès qu'un élément manque. Supposer un objectif produirait un
    tri fondé sur une caméra imaginaire, ce qui est pire que l'ignorance.
    """
    import math

    if entry.get("is_pano"):
        return PANORAMIC_FOV_DEG

    parameters = entry.get("camera_parameters") or []
    width, height = entry.get("width"), entry.get("height")
    if not parameters or not width or not height:
        return None

    focal_ratio = parameters[0]
    if not isinstance(focal_ratio, (int, float)) or focal_ratio <= 0:
        return None

    # Le rapport est relatif à la plus grande dimension ; l'angle horizontal se
    # mesure donc sur la largeur rapportée à cette même référence.
    longest = max(width, height)
    focal_px = focal_ratio * longest
    return round(2.0 * math.degrees(math.atan((width / 2.0) / focal_px)), 2)
