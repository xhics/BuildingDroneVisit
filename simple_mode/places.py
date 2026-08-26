"""Estimation grossière de l'étendue d'un lieu, via l'API Google Places.

Le viewport renvoyé par Places n'est pas l'empreinte exacte d'un bâtiment —
c'est la zone que Google juge pertinente à afficher pour ce lieu — mais
c'est un indicateur bon marché pour distinguer un bâtiment isolé d'un grand
complexe (hôtel avec jardins, campus, centre commercial), et pour mettre à
l'échelle les figures de vol en conséquence plutôt que d'appliquer le même
gabarit à tout le monde.
"""

from __future__ import annotations

import requests

from .geo_utils import haversine_m

FIND_PLACE_URL = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
TIMEOUT = 15


def fetch_viewport_extent_m(address: str, api_key: str) -> tuple[float, float] | None:
    """Renvoie ``(largeur_m, hauteur_m)`` du viewport Places pour cette adresse.

    ``None`` si l'API ne trouve rien ou si ``GOOGLE_PLACES_API_KEY`` n'est
    pas activée pour cette clé — l'appelant retombe alors sur le gabarit par
    défaut (bâtiment isolé), pas une erreur bloquante.
    """
    try:
        resp = requests.get(
            FIND_PLACE_URL,
            params={
                "input": address,
                "inputtype": "textquery",
                "fields": "geometry,name",
                "key": api_key,
            },
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return None

    if data.get("status") != "OK" or not data.get("candidates"):
        return None

    viewport = data["candidates"][0].get("geometry", {}).get("viewport")
    if not viewport:
        return None

    northeast, southwest = viewport.get("northeast"), viewport.get("southwest")
    if not northeast or not southwest:
        return None

    width_m = haversine_m(northeast["lat"], southwest["lng"], northeast["lat"], northeast["lng"])
    height_m = haversine_m(southwest["lat"], southwest["lng"], northeast["lat"], southwest["lng"])
    return width_m, height_m


__all__ = ["fetch_viewport_extent_m"]
