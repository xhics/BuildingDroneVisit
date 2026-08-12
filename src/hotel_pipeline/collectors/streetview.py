"""Collecteur Street View multi-position (Lot 1B §7).

Le collecteur précédent tournait un unique panorama sur huit caps. L'étape 2 a
chiffré le résultat : 8 fichiers, 8 photographies, **1 seul point de vue**.
Huit rotations ne produisent aucun angle nouveau.

Ce collecteur cherche donc des **positions** indépendantes : il échantillonne
le réseau routier autour du bâtiment, interroge le panorama existant à chaque
point, déduplique par identifiant de panorama, puis calcule un cap dirigé vers
l'empreinte.

Économie d'appels : l'endpoint `metadata` est gratuit et dit s'il existe un
panorama, où et à quelle date. Toute la sélection s'y fait ; l'endpoint image,
lui facturé, n'est appelé que pour les panoramas retenus.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import requests
from shapely import wkt as shapely_wkt
from shapely.geometry import Point
from shapely.ops import nearest_points

from ..config import secret
from ..logging import get_logger
from ..providers.cache import cached_call, ensure_online
from ..visibility import bearing_deg, haversine_m
from ..schemas.policy import DEFAULT_POLICY, PipelinePolicy
from .base import CollectedImage

log = get_logger("streetview")

IMAGE_URL = "https://maps.googleapis.com/maps/api/streetview"
METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
TIMEOUT = 30

SIZE = "640x640"
FOV = DEFAULT_POLICY.collection.image_fov_deg

#: Vue élargie, réservée aux positions révélant la transition route–entrée–
#: stationnement. Elle ne crée pas un point de vue supplémentaire.
WIDE_FOV = DEFAULT_POLICY.collection.wide_fov_deg

#: Pas d'échantillonnage du réseau. Un panorama Street View est espacé d'une
#: dizaine de mètres : échantillonner plus finement multiplie les appels sans
#: révéler de position nouvelle.
SAMPLE_SPACING_M = DEFAULT_POLICY.collection.sample_spacing_m

#: Rayon de recherche autour de chaque point échantillonné.
SNAP_RADIUS_M = DEFAULT_POLICY.collection.snap_radius_m

#: Au-delà, un panorama est trop loin pour porter du détail de façade.
MAX_PANORAMA_DISTANCE_M = DEFAULT_POLICY.collection.max_panorama_distance_m

name = "street_view"


@dataclass
class Panorama:
    pano_id: str
    lat: float
    lon: float
    date: str | None
    copyright: str | None


def sample_road_network(
    elements: list[dict], spacing_m: float = SAMPLE_SPACING_M
) -> list[tuple[float, float]]:
    """Points régulièrement espacés le long des voies.

    L'échantillonnage se fait en mètres réels, segment par segment, pour que
    la densité ne dépende pas de la finesse de numérisation OSM.
    """
    samples: list[tuple[float, float]] = []

    for element in elements:
        geometry = element.get("geometry") or []
        for start, end in zip(geometry, geometry[1:]):
            length = haversine_m(start["lat"], start["lon"], end["lat"], end["lon"])
            if length <= 0:
                continue
            steps = max(1, int(length // spacing_m))
            for step in range(steps):
                ratio = step / steps
                samples.append(
                    (
                        start["lat"] + (end["lat"] - start["lat"]) * ratio,
                        start["lon"] + (end["lon"] - start["lon"]) * ratio,
                    )
                )

    log.info("réseau échantillonné : %d point(s) tous les %.0f m", len(samples), spacing_m)
    return samples


def panorama_at(lat: float, lon: float, key: str, radius_m: int = SNAP_RADIUS_M) -> Panorama | None:
    """Panorama le plus proche d'un point, via l'endpoint gratuit."""

    def fetch() -> dict:
        response = requests.get(
            METADATA_URL,
            params={"location": f"{lat},{lon}", "radius": radius_m, "key": key},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    payload = cached_call(f"streetview-meta::{lat:.5f}::{lon:.5f}::{radius_m}", fetch)
    if payload.get("status") != "OK":
        return None

    location = payload.get("location") or {}
    return Panorama(
        pano_id=payload.get("pano_id", ""),
        lat=location.get("lat", lat),
        lon=location.get("lon", location.get("lng", lon)),
        date=payload.get("date"),
        copyright=payload.get("copyright"),
    )


def collect(
    lat: float,
    lon: float,
    building_wkt: str | None = None,
    road_elements: list[dict] | None = None,
    radius_m: int | None = None,
    policy: PipelinePolicy = DEFAULT_POLICY,
) -> list[CollectedImage]:
    """Vues Street View depuis des positions indépendantes.

    Sans réseau routier fourni, le collecteur se rabat sur le panorama le plus
    proche du bâtiment : c'est le comportement dégradé, pas le nominal.
    """
    ensure_online("Street View")
    key = secret("GOOGLE_MAPS_API_KEY")

    if road_elements is None:
        from ..providers.overpass import roads_around

        road_elements = roads_around(lat, lon, radius_m or policy.collection.road_radius_m)

    samples = sample_road_network(road_elements, policy.collection.sample_spacing_m)
    if not samples:
        samples = [(lat, lon)]

    # Déduplication par identifiant de panorama : plusieurs points
    # d'échantillonnage tombent nécessairement sur le même panorama.
    panoramas: dict[str, Panorama] = {}
    for sample_lat, sample_lon in samples:
        try:
            panorama = panorama_at(
                sample_lat, sample_lon, key, policy.collection.snap_radius_m
            )
        except requests.RequestException as exc:
            log.warning("métadonnées indisponibles en %.5f,%.5f : %s", sample_lat, sample_lon, exc)
            continue
        if panorama and panorama.pano_id and panorama.pano_id not in panoramas:
            panoramas[panorama.pano_id] = panorama

    log.info("panoramas distincts trouvés : %d", len(panoramas))

    if building_wkt is None:
        return [_image_for(p, heading=0.0, distance=0.0) for p in panoramas.values()]

    building = shapely_wkt.loads(building_wkt)
    images: list[CollectedImage] = []
    too_far = 0

    for panorama in panoramas.values():
        target = nearest_points(building, Point(panorama.lon, panorama.lat))[0]
        distance = haversine_m(panorama.lat, panorama.lon, target.y, target.x)
        if distance > policy.collection.max_panorama_distance_m:
            too_far += 1
            continue

        # Cap dirigé vers l'empreinte, et non huit caps fixes.
        heading = bearing_deg(panorama.lat, panorama.lon, target.y, target.x)
        images.append(_image_for(panorama, heading, distance, policy.collection.image_fov_deg))

    log.info(
        "Street View : %d position(s) cadrant le bâtiment, %d écartée(s) pour distance",
        len(images),
        too_far,
    )
    return images


def _image_for(
    panorama: Panorama, heading: float, distance: float, fov: int = FOV
) -> CollectedImage:
    return CollectedImage(
        source=name,
        source_id=panorama.pano_id or f"{panorama.lat:.5f}_{panorama.lon:.5f}",
        url=(
            f"{IMAGE_URL}?size={SIZE}&pano={panorama.pano_id}"
            f"&heading={heading:.1f}&fov={fov}&pitch=0"
        ),
        captured_year=_year(panorama.date),
        heading_deg=heading % 360.0,
        # Le cap est dirigé par nous vers l'empreinte : il exprime une
        # intention de cadrage, pas une observation.
        heading_is_measured=False,
        lat=panorama.lat,
        lon=panorama.lon,
        extra={
            "pano_id": panorama.pano_id,
            "date": panorama.date or "",
            "copyright": panorama.copyright or "",
            "distance_m": f"{distance:.1f}",
            "fov": str(fov),
        },
    )


def sign_url(image: CollectedImage) -> str:
    """Ajoute la clé à l'URL, au moment du téléchargement uniquement."""
    return f"{image.url}&key={secret('GOOGLE_MAPS_API_KEY')}"


def _year(date: str | None) -> int | None:
    """`date` est au format AAAA-MM."""
    if not date:
        return None
    try:
        return int(date.split("-")[0])
    except (ValueError, IndexError):
        return None
