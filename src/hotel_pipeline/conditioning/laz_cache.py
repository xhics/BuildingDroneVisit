"""Lecture unique d'une tuile LiDAR, partagée entre ses consommateurs.

La végétation et le sol lisent le même fichier, chacun de son côté : sur ce
pilote, la classification du sol relisait vingt-trois millions de points pour
vingt secondes, après que l'extraction de végétation en eut déjà parcouru
autant. Le fichier ne change pas entre les deux.

Le cache garde les points d'une **emprise** donnée. Deux appels au même rayon
partagent la lecture ; un rayon plus large relit, ce qui est correct — il
demande des points que le premier n'a pas retenus.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..logging import get_logger

log = get_logger("laz-cache")

#: Nombre de lectures conservées. Une seule suffit en pratique : les
#: consommateurs se suivent sur la même tuile et la même emprise.
MAX_ENTRIES = 2

_CACHE: dict[tuple, "LazWindow"] = {}


@dataclass
class LazWindow:
    """Points d'une tuile dans une fenêtre carrée, par classe."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    classification: np.ndarray
    intensity: np.ndarray | None
    red: np.ndarray | None
    green: np.ndarray | None
    blue: np.ndarray | None

    def __len__(self) -> int:
        return int(self.x.size)


def read_window(
    laz_path: Path,
    centre: tuple[float, float],
    radius_m: float,
    with_colour: bool = True,
) -> LazWindow | None:
    """Lit une fenêtre de la tuile, ou rend la lecture déjà en cache."""
    laz_path = Path(laz_path)
    if not laz_path.is_file():
        return None

    try:
        stamp = laz_path.stat().st_mtime_ns
    except OSError:
        return None

    key = (str(laz_path), stamp, round(centre[0], 1), round(centre[1], 1),
           round(radius_m, 1), with_colour)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    import laspy

    cx, cy = centre
    xs, ys, zs, cls = [], [], [], []
    intensity, red, green, blue = [], [], [], []
    has_colour = with_colour

    with laspy.open(str(laz_path)) as reader:
        for chunk in reader.chunk_iterator(2_000_000):
            x = np.asarray(chunk.x)
            y = np.asarray(chunk.y)
            inside = (np.abs(x - cx) <= radius_m) & (np.abs(y - cy) <= radius_m)
            if not inside.any():
                continue
            xs.append(x[inside])
            ys.append(y[inside])
            zs.append(np.asarray(chunk.z)[inside])
            cls.append(np.asarray(chunk.classification)[inside])
            intensity.append(np.asarray(chunk.intensity)[inside].astype(np.float64))
            if has_colour:
                try:
                    red.append(np.asarray(chunk.red)[inside].astype(np.float64))
                    green.append(np.asarray(chunk.green)[inside].astype(np.float64))
                    blue.append(np.asarray(chunk.blue)[inside].astype(np.float64))
                except (AttributeError, ValueError):
                    has_colour = False

    if not xs:
        return None

    window = LazWindow(
        x=np.concatenate(xs),
        y=np.concatenate(ys),
        z=np.concatenate(zs),
        classification=np.concatenate(cls),
        intensity=np.concatenate(intensity),
        red=np.concatenate(red) if has_colour and red else None,
        green=np.concatenate(green) if has_colour and green else None,
        blue=np.concatenate(blue) if has_colour and blue else None,
    )

    # Le cache reste petit : ces tableaux pèsent lourd, et deux emprises
    # suffisent à couvrir une exécution.
    if len(_CACHE) >= MAX_ENTRIES:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = window
    log.info("tuile lue : %d point(s) dans un rayon de %.0f m", len(window), radius_m)
    return window


def clear() -> None:
    """Vide le cache — utile aux tests et entre deux sites."""
    _CACHE.clear()
