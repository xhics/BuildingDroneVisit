"""Collecteur KartaView (ex-OpenStreetCam), APIs publiques v2.0 et v1.0.

Deuxième source d'imagerie de rue **ouverte** du dispositif, après Mapillary.
Comme elle, elle publie des caps observés — la caméra roulait, personne n'a
choisi la direction après coup — donc des vues dont le contenu se mesure au
lieu de se supposer.

Son intérêt n'est pas le volume mais l'angle : deux sources de roulage
indépendantes ne parcourent pas les mêmes rues aux mêmes dates. C'est
exactement ce qui manque à une reconstruction bornée à une seule façade.

Aucune clé n'est requise : l'authentification KartaView sert à *téléverser*,
pas à lire. Deux points d'entrée publics coexistent, et il faut les deux :

- **v2.0** `api.kartaview.org/2.0/photo/` interrogé par `lat/lng/radius`.
  Réponse riche — URLs absolues, dimensions, niveau de qualité, distance.
  Interrogé par emprise (`bbTopLeft`), il répond « Restricted access! » : ce
  n'est pas un refus d'authentification mais un jeu de paramètres refusé.
- **v1.0** `api.openstreetcam.org/1.0/list/nearby-photos/`, plus pauvre mais
  nettement plus large.

Mesuré sur le pilote, même rayon de 300 m : v1 rend 218 clichés, v2 en rend
70, et **les deux ensembles diffèrent** — 61 propres à v2, 209 propres à v1.
Aucun n'englobe l'autre ; n'en interroger qu'un perdrait des vues. On fusionne
donc par identifiant, en préférant la description v2 quand elle existe.
"""

from __future__ import annotations

import hashlib
from datetime import datetime

import requests

from ..logging import get_logger
from ..providers.cache import cached_call, ensure_online
from .base import CollectedImage

log = get_logger("kartaview")

#: v2.0 publique : filtre par rayon. Description riche.
V2_URL = "https://api.kartaview.org/2.0/photo/"

#: Au-delà de 100, `itemsPerPage` renvoie zéro ligne — un plafond silencieux.
V2_PAGE_SIZE = 100

#: v1.0 publique : plus pauvre, mais couverture nettement plus large.
NEARBY_URL = "https://api.openstreetcam.org/1.0/list/nearby-photos/"

#: Hôte servant les fichiers. Les chemins rendus par l'API sont relatifs.
STORAGE_URL = "https://api.openstreetcam.org/"

#: L'API répond lentement sur les grands rayons ; 300 m a été mesuré à ~10 s,
#: 500 m dépassait 40 s. Le délai est donc large, et le rayon modeste.
TIMEOUT = 90

#: Plafond de sécurité, comme pour Mapillary : une emprise dense peut contenir
#: des milliers de clichés dont la plupart ne regardent pas la cible.
MAX_IMAGES = 1500

#: Version du **parseur** : ce que le code fait des champs reçus. Une réponse
#: identique lue autrement n'est plus la même donnée.
PARSER_VERSION = 2

name = "kartaview"


def contract_digest() -> str:
    """Empreinte de ce qu'on demande et de ce qu'on en fait.

    Même raison que pour Mapillary : une entrée de cache obtenue sous un
    contrat antérieur ne doit pas se relire comme si elle portait les champs
    ajoutés depuis.
    """
    material = f"api=v2.0+v1.0|parser={PARSER_VERSION}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]


def _fetch_v1(lat: float, lon: float, radius_m: int) -> dict:
    response = requests.post(
        NEARBY_URL,
        data={"lat": lat, "lng": lon, "radius": radius_m},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def _fetch_v2(lat: float, lon: float, radius_m: int) -> dict:
    """Pages v2 concaténées. `hasMoreData` borne la pagination."""
    rows: list[dict] = []
    page = 1
    while len(rows) < MAX_IMAGES:
        response = requests.get(
            V2_URL,
            params={
                "lat": lat,
                "lng": lon,
                "radius": radius_m,
                "itemsPerPage": V2_PAGE_SIZE,
                "page": page,
            },
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        result = (response.json() or {}).get("result") or {}
        batch = result.get("data") or []
        if not batch:
            break
        rows.extend(batch)
        if not result.get("hasMoreData"):
            break
        page += 1
    return {"data": rows}


def _normalise_v2(entry: dict) -> dict:
    """Ramène une ligne v2 au vocabulaire commun.

    Les deux APIs décrivent les mêmes clichés sous des noms différents. On
    traduit vers un seul vocabulaire plutôt que de disséminer des `or` dans
    le parseur.
    """
    return {
        "id": entry.get("id"),
        "lat": entry.get("lat"),
        "lng": entry.get("lng"),
        "heading": entry.get("heading"),
        "shot_date": entry.get("shotDate"),
        "sequence_id": entry.get("sequenceId"),
        "sequence_index": entry.get("sequenceIndex"),
        "projection": entry.get("projection"),
        "field_of_view": entry.get("fieldOfView"),
        "username": ((entry.get("user") or {}) or {}).get("username")
        if isinstance(entry.get("user"), dict)
        else entry.get("user"),
        "width": entry.get("width"),
        "height": entry.get("height"),
        "quality_level": entry.get("qualityLevel"),
        # v2 rend des URLs absolues : pas de préfixe de stockage à ajouter.
        "absolute_url": (
            entry.get("fileurlProc")
            or entry.get("fileurl")
            or entry.get("fileurlLTh")
            or entry.get("fileurlTh")
        ),
    }


def _fetch(lat: float, lon: float, radius_m: int) -> dict:
    """Fusionne v2 et v1 par identifiant.

    Aucun des deux ensembles n'englobe l'autre. La description v2 est plus
    riche : elle l'emporte quand le même cliché figure des deux côtés, mais
    l'absence d'une ligne en v2 ne retire jamais celle de v1.
    """
    merged: dict[str, dict] = {}

    try:
        for row in _fetch_v2(lat, lon, radius_m).get("data") or []:
            row = _normalise_v2(row)
            if row.get("id") is not None:
                merged[str(row["id"])] = row
    except requests.RequestException as exc:
        # Une API muette ne doit pas emporter l'autre.
        log.warning("KartaView v2 indisponible : %s", exc)

    try:
        for row in _fetch_v1(lat, lon, radius_m).get("currentPageItems") or []:
            key = str(row.get("id"))
            if key not in merged:
                merged[key] = row
    except requests.RequestException as exc:
        log.warning("KartaView v1 indisponible : %s", exc)

    return {"currentPageItems": list(merged.values())}


def collect(lat: float, lon: float, radius_m: int = 300) -> list[CollectedImage]:
    """Clichés KartaView dans un rayon autour d'un point."""
    ensure_online("KartaView")

    digest = contract_digest()
    payload = cached_call(
        f"kartaview-nearby::{digest}::{lat:.6f},{lon:.6f}::{radius_m}",
        lambda: _fetch(lat, lon, radius_m),
    )

    entries = payload.get("currentPageItems") or []

    images: list[CollectedImage] = []
    for entry in entries[:MAX_IMAGES]:
        url = _best_url(entry)
        if not url:
            continue

        latitude = _as_float(entry.get("lat"))
        longitude = _as_float(entry.get("lng"))
        if latitude is None or longitude is None:
            # Sans position, une vue de roulage n'apporte aucune géométrie et
            # ne se distingue plus d'une photographie quelconque.
            continue

        # `projection` vaut PLANE ou SPHERE. Un panorama sphérique n'a pas de
        # cap propre : le sien décrit le véhicule, pas le cadrage d'une vue
        # qu'on en extrairait. On ne le déclare donc pas mesuré.
        projection = str(entry.get("projection") or "").upper()
        is_planar = projection != "SPHERE"

        heading = _normalise_heading(entry.get("heading"))

        images.append(
            CollectedImage(
                source=name,
                source_id=str(entry.get("id")),
                url=url,
                captured_year=_year(entry.get("shot_date")),
                heading_deg=heading,
                lat=latitude,
                lon=longitude,
                sequence_id=(
                    str(entry["sequence_id"]) if entry.get("sequence_id") else None
                ),
                camera_type=projection.lower() or None,
                fov_deg=_as_float(entry.get("field_of_view")) or None,
                width_px=_as_int(entry.get("width")),
                height_px=_as_int(entry.get("height")),
                heading_is_measured=bool(is_planar and heading is not None),
                extra={
                    "sequence_index": str(entry.get("sequence_index") or ""),
                    "contributor": str(entry.get("username") or ""),
                },
            )
        )

    log.info("KartaView : %d image(s) dans un rayon de %d m", len(images), radius_m)
    return images


def _best_url(entry: dict) -> str | None:
    """Plus grande résolution publiée, sans en inventer une.

    `name` est le fichier traité en pleine taille, `lth_name` une grande
    vignette, `th_name` une petite. Demander une variante que l'API ne rend
    pas produirait un silence qui passerait pour une panne réseau.
    """
    absolute = entry.get("absolute_url")
    if absolute:
        return str(absolute)
    for field_name in ("name", "lth_name", "th_name"):
        value = entry.get(field_name)
        if value:
            return STORAGE_URL + str(value).lstrip("/")
    return None


def _as_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_int(value: object) -> int | None:
    number = _as_float(value)
    return int(number) if number is not None else None


def _normalise_heading(value: object) -> float | None:
    heading = _as_float(value)
    if heading is None:
        return None
    return heading % 360.0


def _year(shot_date: object) -> int | None:
    """Année de prise de vue, ou None.

    L'API rend « 1970-01-01 » pour une date absente : c'est l'époque Unix,
    pas une photographie de 1970. La retenir daterait faussement le corpus.
    """
    if not shot_date:
        return None
    text = str(shot_date)
    try:
        year = datetime.strptime(text[:10], "%Y-%m-%d").year
    except ValueError:
        try:
            year = int(text[:4])
        except ValueError:
            return None
    return None if year <= 1970 else year


def photo_url(provider_id: str, resolution: str = "proc") -> str:
    """Adresse d'un cliché, redemandée à l'API au moment du téléchargement.

    Le manifeste ne conserve aucune URL — c'est une règle du dispositif, et
    elle vaut ici aussi : le chemin de stockage KartaView encode la date et un
    hachage qu'on ne saurait pas reconstruire. On réinterroge donc la fiche du
    cliché, et l'on choisit la variante demandée.

    `resolution` est déjà traduite dans le vocabulaire de la source par
    `PROVIDER_RESOLUTIONS` : « proc » (pleine taille), « lth » (grande
    vignette) ou « th » (petite).
    """
    ensure_online("KartaView")

    payload = cached_call(
        f"kartaview-photo::{contract_digest()}::{provider_id}",
        lambda: _fetch_photo(provider_id),
    )
    entry = payload.get("entry") or {}
    if not entry:
        raise ValueError(f"cliché KartaView {provider_id!r} introuvable")

    by_variant = {
        "proc": entry.get("fileurlProc") or entry.get("fileurl"),
        "lth": entry.get("fileurlLTh"),
        "th": entry.get("fileurlTh"),
    }
    url = by_variant.get(resolution)
    if not url:
        raise ValueError(
            f"variante {resolution!r} absente pour le cliché {provider_id!r} ; "
            f"connues : {sorted(k for k, v in by_variant.items() if v)}"
        )
    return str(url)


def _fetch_photo(provider_id: str) -> dict:
    """Fiche d'un cliché unique, par identifiant."""
    response = requests.get(
        f"{V2_URL}{provider_id}",
        headers={"accept": "application/json"},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    result = (response.json() or {}).get("result") or {}
    data = result.get("data")
    if isinstance(data, list):
        data = data[0] if data else None
    return {"entry": data or {}}
