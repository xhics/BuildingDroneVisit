"""Collecte de panoramas Street View autour d'une adresse.

Inspiré de ``src/hotel_pipeline/collectors/streetview.py`` : même idée
(échantillonner des positions, interroger l'endpoint gratuit ``metadata``
pour ne payer l'endpoint image que pour les panoramas retenus, dédupliquer
par identifiant de panorama, diriger le cap vers le centre plutôt que huit
caps fixes). Simplifié pour `simple_mode` : pas de réseau routier OSM ni
d'empreinte de bâtiment mesurée — l'échantillonnage se fait sur des cercles
concentriques autour du point géocodé, ce qui suffit pour une adresse
ponctuelle sans les données géospatiales du pipeline principal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import requests

from .geo_utils import bearing_deg, haversine_m, offset_to_latlon

IMAGE_URL = "https://maps.googleapis.com/maps/api/streetview"
METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
TIMEOUT = 20

#: Rayons des cercles d'échantillonnage autour du centre (mètres). Un
#: panorama Street View piéton est presque toujours sur la voie, donc à
#: quelques dizaines de mètres du bâtiment plutôt que collé dessus.
DEFAULT_RING_RADII_M: tuple[float, ...] = (15.0, 30.0, 50.0, 75.0)

#: Points échantillonnés par cercle.
DEFAULT_SAMPLES_PER_RING = 10

#: Au-delà, un panorama est trop loin pour porter du détail utile du bâtiment.
DEFAULT_MAX_DISTANCE_M = 90.0


@dataclass
class Panorama:
    pano_id: str
    lat: float
    lon: float
    date: str | None
    distance_m: float
    #: Cap du panorama vers le centre géocodé — dirige la vue vers le
    #: bâtiment plutôt que de photographier au hasard.
    heading_to_center_deg: float


def _metadata_at(lat: float, lon: float, api_key: str, radius_m: int = 50) -> dict | None:
    resp = requests.get(
        METADATA_URL,
        params={"location": f"{lat},{lon}", "radius": radius_m, "key": api_key},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "OK":
        return None
    return data


def find_panoramas(
    center_lat: float,
    center_lon: float,
    api_key: str,
    *,
    ring_radii_m: tuple[float, ...] = DEFAULT_RING_RADII_M,
    samples_per_ring: int = DEFAULT_SAMPLES_PER_RING,
    max_distance_m: float = DEFAULT_MAX_DISTANCE_M,
) -> list[Panorama]:
    """Panoramas distincts trouvés autour du centre, triés du plus proche au plus loin.

    N'appelle que l'endpoint ``metadata`` (gratuit) : aucun frais tant que
    les images ne sont pas téléchargées.
    """
    found: dict[str, Panorama] = {}
    for radius in ring_radii_m:
        for i in range(samples_per_ring):
            angle = 2 * math.pi * i / samples_per_ring
            east = radius * math.sin(angle)
            north = radius * math.cos(angle)
            lat, lon = offset_to_latlon(center_lat, center_lon, east, north)
            try:
                data = _metadata_at(lat, lon, api_key)
            except requests.RequestException:
                continue
            if not data:
                continue
            pano_id = data.get("pano_id", "")
            if not pano_id or pano_id in found:
                continue
            location = data.get("location") or {}
            p_lat = location.get("lat", lat)
            p_lon = location.get("lng", lon)
            distance = haversine_m(center_lat, center_lon, p_lat, p_lon)
            if distance > max_distance_m:
                continue
            found[pano_id] = Panorama(
                pano_id=pano_id,
                lat=p_lat,
                lon=p_lon,
                date=data.get("date"),
                distance_m=distance,
                heading_to_center_deg=bearing_deg(p_lat, p_lon, center_lat, center_lon),
            )

    return sorted(found.values(), key=lambda p: p.distance_m)


def image_url(
    panorama: Panorama,
    api_key: str,
    *,
    heading: float | None = None,
    fov: int = 90,
    pitch: float = 0.0,
    size: str = "640x640",
) -> str:
    """URL Street View Static, cadrée vers le centre par défaut."""
    effective_heading = panorama.heading_to_center_deg if heading is None else heading
    return (
        f"{IMAGE_URL}?size={size}&pano={panorama.pano_id}&heading={effective_heading:.1f}"
        f"&fov={fov}&pitch={pitch:.1f}&key={api_key}"
    )


def download_image(panorama: Panorama, api_key: str, dest_path: str | Path, **kwargs) -> Path:
    """Télécharge l'image du panorama vers ``dest_path`` et renvoie le chemin."""
    url = image_url(panorama, api_key, **kwargs)
    resp = requests.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    path = Path(dest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    return path


def assign_to_positions(
    panoramas: list[Panorama],
    positions: list[tuple[float, float]],
    *,
    max_distance_m: float = 80.0,
) -> list[Panorama | None]:
    """Pour chaque position de caméra, le panorama réel le plus proche.

    C'est l'ancrage **spatial** : la référence utilisée à un instant donné
    montre l'endroit que la caméra survole réellement, au lieu d'être choisie
    par thème. Sans cela, un survol de l'aile est pourrait s'appuyer sur une
    photo de la façade ouest, et la transition inventerait un raccord entre
    deux endroits sans rapport.

    ``None`` là où aucun panorama n'est assez proche : mieux vaut une étape
    sans référence, assumée, qu'une référence trompeuse.
    """
    return [
        nearest_to(panoramas, lat, lon, max_distance_m=max_distance_m)
        for lat, lon in positions
    ]


def nearest_to(
    panoramas: list[Panorama], lat: float, lon: float, *, max_distance_m: float = 40.0
) -> Panorama | None:
    """Panorama le plus proche d'un point donné, ou None si aucun n'est assez proche."""
    best: Panorama | None = None
    best_distance = max_distance_m
    for panorama in panoramas:
        distance = haversine_m(panorama.lat, panorama.lon, lat, lon)
        if distance <= best_distance:
            best, best_distance = panorama, distance
    return best


__all__ = [
    "Panorama",
    "assign_to_positions",
    "download_image",
    "find_panoramas",
    "image_url",
    "nearest_to",
]
