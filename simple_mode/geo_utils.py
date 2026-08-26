"""Petites conversions géographiques partagées (repère local <-> WGS84).

Approximation plane (sphère locale), suffisante aux échelles de ce module
(quelques dizaines de mètres autour d'une adresse) — pas de reprojection
cartographique complète comme dans `hotel_pipeline`.
"""

from __future__ import annotations

import math

EARTH_RADIUS_M = 6378137.0


def offset_to_latlon(lat0: float, lon0: float, east_m: float, north_m: float) -> tuple[float, float]:
    """Point à (east_m, north_m) du centre (lat0, lon0)."""
    d_lat = north_m / EARTH_RADIUS_M
    d_lon = east_m / (EARTH_RADIUS_M * math.cos(math.radians(lat0)))
    return lat0 + math.degrees(d_lat), lon0 + math.degrees(d_lon)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(min(1.0, a)))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Cap, en degrés, du point 1 vers le point 2."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlambda)
    return math.degrees(math.atan2(x, y)) % 360.0


__all__ = ["bearing_deg", "haversine_m", "offset_to_latlon"]
