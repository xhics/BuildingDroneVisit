"""Collecteur du site officiel de l'établissement.

Les photos du site sont les plus susceptibles de montrer le bâtiment sous son
meilleur angle, et de refléter l'entrée dans sa version actuelle. Leurs droits
appartiennent à l'établissement : `Rights.UNKNOWN` tant qu'un accord n'est pas
formalisé, donc soumis à la décision d'assumer l'usage.
"""

from __future__ import annotations

import re
from urllib.parse import urljoin, urlparse

import requests

from ..logging import get_logger
from ..providers.cache import cached_call, ensure_online
from .base import CollectedImage

log = get_logger("website")

TIMEOUT = 30
USER_AGENT = "hotel-pipeline/0.1 (reconstruction 3D, contact via dépôt)"
MAX_PAGES = 12
MIN_DIMENSION_HINT = 400

#: Vignettes, pictogrammes et pixels de suivi n'ont aucune valeur ici.
SKIP_PATTERNS = re.compile(
    r"(logo|icon|favicon|sprite|pixel|spacer|placeholder|avatar|badge)", re.IGNORECASE
)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")

name = "website"


def _fetch(url: str) -> str:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    response.raise_for_status()
    return response.text


def _image_urls(html: str, base_url: str) -> list[str]:
    """Extrait les URL d'images, y compris celles des attributs responsives."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    found: list[str] = []

    for tag in soup.find_all("img"):
        for attribute in ("src", "data-src", "data-lazy-src", "data-original"):
            value = tag.get(attribute)
            if value:
                found.append(urljoin(base_url, value))

        srcset = tag.get("srcset") or tag.get("data-srcset")
        if srcset:
            # La dernière entrée d'un srcset est la plus grande résolution.
            candidates = [part.strip().split(" ")[0] for part in srcset.split(",") if part.strip()]
            if candidates:
                found.append(urljoin(base_url, candidates[-1]))

    # Images de fond déclarées en style inline.
    for match in re.finditer(r"url\(['\"]?([^'\")]+\.(?:jpg|jpeg|png|webp))", html, re.IGNORECASE):
        found.append(urljoin(base_url, match.group(1)))

    return found


def _internal_links(html: str, base_url: str) -> list[str]:
    from bs4 import BeautifulSoup

    domain = urlparse(base_url).netloc
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for tag in soup.find_all("a", href=True):
        url = urljoin(base_url, tag["href"])
        if urlparse(url).netloc == domain and url.split("#")[0] not in links:
            links.append(url.split("#")[0])
    return links


def _is_useful(url: str) -> bool:
    path = urlparse(url).path.lower()
    if not path.endswith(IMAGE_SUFFIXES):
        return False
    return not SKIP_PATTERNS.search(url)


def collect(site_url: str, max_pages: int = MAX_PAGES) -> list[CollectedImage]:
    """Parcourt le site et rapporte ses images exploitables.

    Exploration limitée aux pages internes et bornée par `max_pages` : il s'agit
    de récupérer une galerie, pas d'aspirer un site.
    """
    ensure_online("site officiel")

    visited: set[str] = set()
    queue = [site_url]
    urls: list[str] = []

    while queue and len(visited) < max_pages:
        page = queue.pop(0)
        if page in visited:
            continue
        visited.add(page)

        try:
            html = cached_call(f"website::{page}", lambda p=page: _fetch(p))
        except requests.RequestException as exc:
            log.warning("page inaccessible %s : %s", page, exc)
            continue

        urls.extend(u for u in _image_urls(html, page) if _is_useful(u))
        for link in _internal_links(html, page):
            if link not in visited and len(queue) + len(visited) < max_pages:
                queue.append(link)

    unique = list(dict.fromkeys(urls))
    log.info("site officiel : %d image(s) sur %d page(s)", len(unique), len(visited))

    return [
        CollectedImage(
            source=name,
            source_id=_slug(url, index),
            url=url,
            extra={"page_count": str(len(visited))},
        )
        for index, url in enumerate(unique)
    ]


def _slug(url: str, index: int) -> str:
    stem = urlparse(url).path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", stem)[:48].strip("-")
    return f"{index:03d}-{cleaned}" if cleaned else f"{index:03d}"
