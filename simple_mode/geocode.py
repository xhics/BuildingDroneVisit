"""Géocodage d'une adresse via l'API Google Geocoding."""

from __future__ import annotations

import requests

GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"


class GeocodeError(RuntimeError):
    """L'adresse n'a pas pu être résolue en coordonnées."""


def geocode_address(address: str, api_key: str) -> dict:
    """Résout une adresse en coordonnées.

    Renvoie ``{"lat": float, "lon": float, "formatted_address": str}``.
    """
    resp = requests.get(
        GEOCODE_URL, params={"address": address, "key": api_key}, timeout=15
    )
    resp.raise_for_status()
    data = resp.json()
    status = data.get("status")
    if status != "OK" or not data.get("results"):
        raise GeocodeError(
            f"géocodage échoué ({status}) pour {address!r} : "
            f"{data.get('error_message', 'aucun résultat')}"
        )
    result = data["results"][0]
    location = result["geometry"]["location"]
    return {
        "lat": location["lat"],
        "lon": location["lng"],
        "formatted_address": result.get("formatted_address", address),
    }
