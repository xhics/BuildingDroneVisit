"""Récupération d'une image satellite statique centrée sur un point."""

from __future__ import annotations

import io
import math

import requests
from PIL import Image

STATIC_MAP_URL = "https://maps.googleapis.com/maps/api/staticmap"

#: Constante standard de la projection Web Mercator (résolution au zoom 0,
#: tuiles de 256 px) — utilisée par toutes les cartes de type Google/OSM.
_MERCATOR_BASE_MPP = 156543.03392


def meters_per_pixel(lat: float, zoom: int, scale: int) -> float:
    """Résolution au sol, en mètres par pixel, à une latitude et un zoom donnés."""
    return (_MERCATOR_BASE_MPP * math.cos(math.radians(lat))) / (2**zoom) / scale


def fetch_satellite_image(
    lat: float,
    lon: float,
    api_key: str,
    *,
    zoom: int = 19,
    size: int = 640,
    scale: int = 2,
) -> tuple[Image.Image, float]:
    """Télécharge une image satellite centrée sur (lat, lon).

    Renvoie ``(image, mètres_par_pixel)``.
    """
    params = {
        "center": f"{lat},{lon}",
        "zoom": zoom,
        "size": f"{size}x{size}",
        "scale": scale,
        "maptype": "satellite",
        "key": api_key,
    }
    resp = requests.get(STATIC_MAP_URL, params=params, timeout=20)
    resp.raise_for_status()
    image = Image.open(io.BytesIO(resp.content)).convert("RGB")
    mpp = meters_per_pixel(lat, zoom, scale)
    return image, mpp
