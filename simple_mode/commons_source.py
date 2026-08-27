"""Photographies libres géolocalisées, via Wikimedia Commons.

Les sources déjà branchées plafonnent vite : Places n'a livré que deux vues
extérieures exploitables du Château Frontenac, et Street View, qui est de la
photographie de voirie, ne cadre jamais un sujet — de près on n'obtient qu'un
pan de mur, d'assez loin le bâtiment disparaît derrière les arbres.

Commons comble ce manque pour les édifices remarquables : des photographies
prises **intentionnellement**, sous des angles variés, souvent par des
photographes soigneux. C'est précisément le type d'image qui a fonctionné en
référence — une vue d'ensemble, dégagée, cadrée sur le bâtiment.

Aucune clé n'est requise. En contrepartie, la couverture est inégale : riche
sur un monument, quasi nulle sur un hôtel ordinaire. Cette source complète
les autres, elle ne les remplace pas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import requests

API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "simple-mode/0.1 (references video, contact via depot)"
TIMEOUT = 30


@dataclass
class CommonsPhoto:
    path: Path
    title: str
    width: int
    height: int
    #: Licence et auteur, à conserver : ces images sont libres mais leur
    #: réutilisation reste conditionnée à l'attribution.
    credit: str = ""


def search(lat: float, lon: float, *, radius_m: int = 400, limit: int = 40) -> list[dict]:
    """Fiches d'images géolocalisées autour d'un point."""
    params = {
        "action": "query",
        "format": "json",
        "generator": "geosearch",
        "ggscoord": f"{lat}|{lon}",
        "ggsradius": min(radius_m, 10_000),
        "ggslimit": limit,
        "ggsnamespace": 6,  # espace Fichier
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|size",
        "iiurlwidth": 1600,
    }
    response = requests.get(
        API_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT
    )
    response.raise_for_status()
    pages = (response.json().get("query") or {}).get("pages") or {}
    return list(pages.values())


def _is_usable(page: dict, *, min_width: int) -> bool:
    """Écarte ce qui n'est pas une photographie exploitable.

    Commons héberge aussi des plans, des blasons, des cartes et des scans
    anciens : passés en référence, ils feraient dériver la génération vers un
    dessin plutôt qu'une photographie.
    """
    info = (page.get("imageinfo") or [{}])[0]
    if info.get("width", 0) < min_width:
        return False
    url = str(info.get("url", "")).lower()
    if not url.endswith((".jpg", ".jpeg", ".png")):
        return False
    title = str(page.get("title", "")).lower()
    excluded = ("map", "carte", "plan", "blason", "coat of arms", "logo", "diagram", "svg")
    return not any(word in title for word in excluded)


def fetch(
    lat: float,
    lon: float,
    out_dir: str | Path,
    *,
    radius_m: int = 400,
    limit: int = 8,
    min_width: int = 1000,
) -> list[CommonsPhoto]:
    """Télécharge les photographies libres les plus grandes du lieu.

    Le tri par taille sert de filtre qualité grossier : sur Commons, une
    image en haute définition a presque toujours été versée délibérément,
    quand les petites sont souvent des vignettes ou des recadrages.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        pages = search(lat, lon, radius_m=radius_m, limit=40)
    except requests.RequestException:
        return []

    usable = [p for p in pages if _is_usable(p, min_width=min_width)]
    usable.sort(key=lambda p: p["imageinfo"][0].get("width", 0), reverse=True)

    photos: list[CommonsPhoto] = []
    for index, page in enumerate(usable[:limit]):
        info = page["imageinfo"][0]
        url = info.get("thumburl") or info.get("url")
        try:
            image = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
            image.raise_for_status()
        except requests.RequestException:
            continue

        destination = out_dir / f"commons_{index:02d}.jpg"
        destination.write_bytes(image.content)
        meta = info.get("extmetadata") or {}
        photos.append(
            CommonsPhoto(
                path=destination,
                title=str(page.get("title", "")).removeprefix("File:"),
                width=int(info.get("width", 0)),
                height=int(info.get("height", 0)),
                credit=str((meta.get("Artist") or {}).get("value", ""))[:120],
            )
        )
    return photos


__all__ = ["CommonsPhoto", "fetch", "search"]
