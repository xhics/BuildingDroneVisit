"""Géocodage d'adresse (complément d'implémentation §5).

Nominatim résout parfois sur le centroïde de rue plutôt que sur le bâtiment.
Un géocodeur officiel québécois est donc préférable en source primaire.

L'URL du service québécois n'est pas codée en dur : elle est fournie par
``GEOCODER_QC_URL``. Tant qu'elle n'est pas configurée, Nominatim est utilisé
et le manifeste enregistre quel fournisseur a réellement produit la position —
la provenance n'est jamais supposée.
"""

from __future__ import annotations

import os
from typing import Any

import requests

from ..logging import get_logger
from ..schemas.spatial import GeocodeResult
from .cache import cached_call, ensure_online

log = get_logger("geocode")

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "hotel-pipeline/0.1 (reconstruction 3D, contact via dépôt)"
TIMEOUT = 30


class GeocodingError(RuntimeError):
    pass


def _get_json(url: str, params: dict[str, Any]) -> Any:
    ensure_online(f"géocodage {url}")
    response = requests.get(
        url, params=params, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT
    )
    response.raise_for_status()
    return response.json()


def _nominatim(address: str) -> GeocodeResult:
    payload = cached_call(
        f"nominatim::{address}",
        lambda: _get_json(
            NOMINATIM_URL,
            {"q": address, "format": "jsonv2", "limit": 1, "addressdetails": 1},
        ),
    )
    if not payload:
        raise GeocodingError(f"Nominatim n'a rien retourné pour {address!r}")

    top = payload[0]
    return GeocodeResult(
        lat=float(top["lat"]),
        lon=float(top["lon"]),
        provider="nominatim",
        raw_label=top.get("display_name"),
        postcode=(top.get("address") or {}).get("postcode"),
    )


def _quebec(address: str, base_url: str) -> GeocodeResult:
    """Géocodeur officiel québécois, si son URL est configurée.

    Le format de réponse attendu est GeoJSON. Toute divergence lève une erreur
    explicite plutôt que d'inventer une position.
    """
    payload = cached_call(
        f"qc::{base_url}::{address}",
        lambda: _get_json(base_url, {"q": address, "limit": 1}),
    )
    features = payload.get("features") if isinstance(payload, dict) else None
    if not features:
        raise GeocodingError(f"le géocodeur québécois n'a rien retourné pour {address!r}")

    feature = features[0]
    lon, lat = feature["geometry"]["coordinates"][:2]
    properties = feature.get("properties") or {}
    return GeocodeResult(
        lat=float(lat),
        lon=float(lon),
        provider="adresses-quebec",
        raw_label=properties.get("label") or properties.get("adresse"),
        postcode=properties.get("code_postal") or properties.get("postcode"),
    )


def geocode(address: str) -> GeocodeResult:
    """Résout une adresse, source officielle d'abord, Nominatim en secours."""
    base_url = os.environ.get("GEOCODER_QC_URL", "").strip()

    if base_url:
        try:
            return _quebec(address, base_url)
        except (requests.RequestException, GeocodingError, KeyError, ValueError) as exc:
            log.warning("géocodeur québécois indisponible (%s), repli sur Nominatim", exc)

    return _nominatim(address)
