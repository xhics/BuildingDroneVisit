"""Sourcing des vraies photos d'un établissement, et parcours qui les relie.

Le survol extérieur (``cesium_render``) montre le bâtiment tel qu'il est,
mais s'arrête à ses murs. Pour une visite d'hôtel, l'intérieur ne peut pas
être inventé : une chambre générée sans référence ne ressemble à aucune
chambre de cet hôtel-là. Ce module va donc chercher les **photos réelles**
de l'établissement (API Google Places), les fait classer par un modèle de
vision, puis en déduit un **parcours** — extérieur, entrée, espaces communs,
chambres, équipements, jardins — qui servira de fil conducteur à la vidéo.

Les photos deviennent les points d'ancrage : entre deux d'entre elles, la
génération vidéo n'a plus qu'à inventer une transition courte entre deux
images réelles et thématiquement voisines, le cas où elle est fiable.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
TIMEOUT = 30

#: Ordre de visite. Une visite lisible part de l'extérieur, entre, monte vers
#: l'intimité des chambres, puis s'ouvre sur les équipements et les extérieurs
#: aménagés — plutôt que d'enchaîner les lieux au hasard des photos trouvées.
JOURNEY_ORDER = [
    "exterieur",
    "entree",
    "hall",
    "restaurant",
    "bar",
    "chambre",
    "salle_de_bain",
    "spa",
    "piscine",
    "salle_de_sport",
    "jardin",
    "vue",
]

CATEGORY_LABELS_FR = {
    "exterieur": "Façade et abords",
    "entree": "Entrée",
    "hall": "Hall / réception",
    "restaurant": "Restaurant",
    "bar": "Bar / salon",
    "chambre": "Chambre",
    "salle_de_bain": "Salle de bain",
    "spa": "Spa",
    "piscine": "Piscine",
    "salle_de_sport": "Salle de sport",
    "jardin": "Jardin / terrasse",
    "vue": "Vue depuis l'établissement",
    "autre": "Autre",
}


class HotelSourceError(RuntimeError):
    """La recherche ou le téléchargement des photos a échoué."""


@dataclass
class SourcePhoto:
    path: Path
    #: Catégorie issue du classement visuel (clé de ``CATEGORY_LABELS_FR``).
    category: str = "autre"
    #: Description courte, réutilisée dans les prompts de transition.
    description_fr: str = ""
    #: La photo convient-elle comme **référence d'apparence** pour un rendu ?
    #: Une vue encombrée de texte, d'enseignes ou d'un premier plan massif
    #: contamine la génération : le modèle en recopie les lettrages dans la
    #: vidéo. Constaté sur une photo à massif floral « UNESCO », dont le
    #: lettrage s'est retrouvé incrusté dans le plan aérien.
    clean_reference: bool = False

    @property
    def label_fr(self) -> str:
        return CATEGORY_LABELS_FR.get(self.category, CATEGORY_LABELS_FR["autre"])


@dataclass
class HotelSite:
    query: str
    place_id: str
    display_name: str
    lat: float
    lon: float
    photos: list[SourcePhoto] = field(default_factory=list)

    def journey(self) -> list[SourcePhoto]:
        """Photos ordonnées en parcours de visite.

        Les catégories absentes sont simplement sautées : un établissement
        sans spa ne doit pas produire d'étape vide. Les photos non
        classées ferment la marche plutôt que d'être écartées — elles
        montrent souvent un détail utile.
        """
        ordered: list[SourcePhoto] = []
        for category in JOURNEY_ORDER:
            ordered.extend(p for p in self.photos if p.category == category)
        ordered.extend(p for p in self.photos if p.category not in JOURNEY_ORDER)
        return ordered


def find_place(query: str, api_key: str) -> dict:
    """Résout un nom d'établissement en fiche Places (id, nom, position, photos)."""
    response = requests.post(
        SEARCH_URL,
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": "places.id,places.displayName,places.location,places.photos",
        },
        json={"textQuery": query},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("places"):
        raise HotelSourceError(f"aucun établissement trouvé pour {query!r}")
    return payload["places"][0]


def fetch_photos(
    place: dict, api_key: str, out_dir: str | Path, *, limit: int = 10, max_width: int = 1600
) -> list[SourcePhoto]:
    """Télécharge les photos de l'établissement et renvoie leurs chemins."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    photos: list[SourcePhoto] = []
    for index, entry in enumerate(place.get("photos", [])[:limit]):
        url = f"https://places.googleapis.com/v1/{entry['name']}/media"
        try:
            response = requests.get(
                url, params={"maxWidthPx": max_width, "key": api_key}, timeout=TIMEOUT
            )
            response.raise_for_status()
        except requests.RequestException:
            continue
        path = out_dir / f"source_{index:02d}.jpg"
        path.write_bytes(response.content)
        photos.append(SourcePhoto(path=path))
    if not photos:
        raise HotelSourceError("aucune photo téléchargeable pour cet établissement")
    return photos


_CLASSIFY_PROMPT = (
    "Classe cette photo d'établissement hôtelier dans exactement une catégorie parmi : "
    + ", ".join(JOURNEY_ORDER)
    + ", autre. Indique aussi si elle conviendrait comme référence visuelle pour "
    "générer une image du lieu : elle ne convient PAS si elle contient du texte, un "
    "logo, une enseigne, un massif floral écrit, un panneau, un filigrane, des "
    "personnes au premier plan, ou si le sujet est masqué par un avant-plan encombré. "
    "Réponds en JSON strict : "
    '{"category": "<catégorie>", "description": "<une phrase en français décrivant '
    'le lieu, ses matériaux, ses couleurs et son ambiance>", '
    '"clean_reference": true|false}'
)


def classify_photos(photos: list[SourcePhoto], openai_key: str, *, model: str = "gpt-4o-mini") -> None:
    """Fait décrire et catégoriser chaque photo par un modèle de vision.

    Le classement conditionne l'ordre de visite ; la description sert ensuite
    à écrire des prompts de transition ancrés sur ce que la photo montre
    vraiment, plutôt que sur une chambre d'hôtel générique. Une photo dont
    le classement échoue reste dans le lot, en catégorie ``autre``.
    """
    from openai import AuthenticationError, OpenAI, RateLimitError

    client = OpenAI(api_key=openai_key)
    for photo in photos:
        encoded = base64.b64encode(photo.path.read_bytes()).decode()
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    response_format={"type": "json_object"},
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": _CLASSIFY_PROMPT},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                                },
                            ],
                        }
                    ],
                )
                data = json.loads(response.choices[0].message.content or "{}")
                photo.category = str(data.get("category", "autre"))
                photo.description_fr = str(data.get("description", ""))
                photo.clean_reference = bool(data.get("clean_reference", False))
                break
            except RateLimitError as exc:
                # Plafond de jetons par minute : il suffit d'attendre. Échouer
                # ici gâcherait le classement de tout un lot pour quelques
                # secondes d'attente — et un lot de photos le déclenche
                # facilement, chaque image pesant lourd en jetons.
                if attempt >= 2:
                    raise HotelSourceError(f"classement impossible : {exc}") from exc
                time.sleep(5 * (attempt + 1))
            except AuthenticationError as exc:
                # Celle-ci vaut pour tout le lot : l'avaler classerait chaque
                # photo en « autre » et donnerait un parcours silencieusement
                # vide, sans jamais dire que la clé est morte.
                raise HotelSourceError(f"classement impossible : {exc}") from exc
            except Exception:  # noqa: BLE001 — un échec isolé reste exploitable
                photo.category = "autre"
                break


def load_site(
    query: str, places_key: str, openai_key: str | None, out_dir: str | Path, *, limit: int = 10
) -> HotelSite:
    """Chaîne complète : recherche, téléchargement, classement."""
    place = find_place(query, places_key)
    location = place.get("location", {})
    site = HotelSite(
        query=query,
        place_id=place.get("id", ""),
        display_name=place.get("displayName", {}).get("text", query),
        lat=float(location.get("latitude", 0.0)),
        lon=float(location.get("longitude", 0.0)),
        photos=fetch_photos(place, places_key, out_dir, limit=limit),
    )
    if openai_key:
        classify_photos(site.photos, openai_key)
    return site


__all__ = [
    "CATEGORY_LABELS_FR",
    "JOURNEY_ORDER",
    "HotelSite",
    "HotelSourceError",
    "SourcePhoto",
    "classify_photos",
    "fetch_photos",
    "find_place",
    "load_site",
]
