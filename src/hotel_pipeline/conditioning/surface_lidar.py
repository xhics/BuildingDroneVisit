"""Nature du sol déduite du retour LiDAR, là où la carte se tait.

Un plan d'établissement montre surtout du sol, et le générateur laissé libre y
couvre d'asphalte une pelouse ou l'inverse. OpenStreetMap porte cette
information mais rarement aux abords : sur ce pilote, la pelouse cartographiée
la plus proche est à près de trois cents mètres.

Le nuage LiDAR, lui, couvre le site. Deux signaux y distinguent une surface
végétale d'une surface minérale :

- **l'intensité du retour** — mesurée ici, elle vaut environ 36 400 sur les
  emprises minérales d'OpenStreetMap contre 41 500 ailleurs, soit un écart de
  cinq mille unités ;
- **la couleur**, quand la tuile la porte. Elle est peu contrastée sur un
  relevé hivernal — le gazon dormant n'est pas vert — et ne sert donc que
  d'appoint.

La classification reste une déduction : elle sort `indetermine` là où les deux
signaux se contredisent, plutôt que de trancher au hasard.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..logging import get_logger

log = get_logger("conditioning-surface")

#: Classe ASPRS du sol.
GROUND_CLASS = 2

#: Taille de cellule de la carte de sol, en mètres. La maille est fine parce
#: qu'elle sert à **trouver** la frontière, non à la dessiner : le rendu ne
#: consomme pas les cellules mais les contours qu'on en extrait. Un premier
#: essai rendait les cellules elles-mêmes, et il fallait alors les grossir à
#: quatre mètres pour tenir le temps de calcul — au prix de bordures en
#: escalier.
SURFACE_CELL_M = 1.0

#: Points minimaux dans une cellule pour la classer.
MIN_POINTS_PER_CELL = 2

#: Écart minimal à la médiane, en unités d'intensité, pour trancher. En deçà,
#: la cellule est trop proche de la limite pour être attribuée.
INTENSITY_MARGIN = 1500.0

#: Nombre de passes de lissage par vote majoritaire. Une cellule isolée dans
#: un voisinage contraire relève du bruit du capteur, non d'une frontière :
#: mesuré sur ce pilote, dix-neuf pour cent des cellules contredisaient leur
#: entourage et soixante pour cent des plages ne faisaient qu'une ou deux
#: cellules. Un sol réel forme des plages continues — une allée, une pelouse —
#: et c'est cette continuité que le lissage restitue.
SMOOTHING_PASSES = 2

#: Part des voisins qu'il faut voir d'accord pour retourner une cellule. Réglé
#: haut : une frontière franche entre gazon et enrobé doit survivre au lissage.
SMOOTHING_MAJORITY = 0.62

#: En deçà, une plage isolée est absorbée par ce qui l'entoure : quelques
#: cellules perdues au milieu d'un stationnement ne décrivent pas une pelouse.
MIN_PATCH_CELLS = 3

#: Indice de verdure au-delà duquel la couleur confirme une surface végétale.
#: Réglé bas : un relevé d'hiver donne un gazon à peine plus vert que le
#: bitume, et un seuil élevé n'aurait jamais rien classé.
GREENNESS_CONFIRM = 0.02


@dataclass
class SurfaceCell:
    """Une cellule de sol et la nature que les retours lui attribuent."""

    x: float
    y: float
    kind: str
    intensity: float
    greenness: float | None
    points: int
    #: Nature retenue quand le doute est comblé par le voisinage, au moment de
    #: fermer la surface. `kind` garde ce que la mesure a établi.
    filled_kind: str | None = None

    def as_dict(self) -> dict:
        return {
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "kind": self.kind,
            "intensity": round(self.intensity, 0),
            "greenness": None if self.greenness is None else round(self.greenness, 4),
            "points": self.points,
        }


@dataclass
class SurfaceMap:
    """Carte du sol autour d'un site."""

    hotel_id: str
    cells: list[SurfaceCell] = field(default_factory=list)
    cell_m: float = SURFACE_CELL_M
    threshold: float = 0.0
    provenance: dict = field(default_factory=dict)

    def by_kind(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for cell in self.cells:
            counts[cell.kind] = counts.get(cell.kind, 0) + 1
        return counts

    def area_by_kind(self) -> dict[str, float]:
        unit = self.cell_m * self.cell_m
        return {k: round(v * unit, 1) for k, v in self.by_kind().items()}

    def as_dict(self) -> dict:
        return {
            "hotel_id": self.hotel_id,
            "cell_m": self.cell_m,
            "cell_count": len(self.cells),
            "by_kind": self.by_kind(),
            "area_by_kind_m2": self.area_by_kind(),
            "intensity_threshold": round(self.threshold, 0),
            "provenance": self.provenance,
            "cells": [c.as_dict() for c in self.cells],
            "caveats": [
                "la nature du sol est déduite de l'intensité du retour LiDAR, "
                "non d'une classification livrée : une surface humide ou un "
                "matériau clair peut être mal attribué",
                "une cellule dont l'intensité reste proche du seuil sort "
                "`indetermine` plutôt que d'être tranchée au hasard",
                "le relevé est hivernal : la couleur n'y confirme presque rien, "
                "un gazon dormant n'étant guère plus vert que le bitume",
            ],
        }


def _grid_index(cells: list[SurfaceCell], cell_m: float) -> dict:
    """Position de chaque cellule dans la grille, en indices entiers."""
    return {
        (
            int(math.floor(c.x / cell_m)),
            int(math.floor(c.y / cell_m)),
        ): i
        for i, c in enumerate(cells)
    }


def _smooth(cells: list[SurfaceCell], cell_m: float) -> int:
    """Lisse la classification par vote majoritaire du voisinage.

    Le vote porte sur les huit voisins immédiats. Une cellule que son entourage
    contredit franchement est retournée ; une frontière nette, où la moitié du
    voisinage seulement diffère, est préservée.

    Les cellules indéterminées participent au vote sans jamais l'emporter :
    elles marquent un doute, pas une nature.
    """
    if not cells:
        return 0

    # Les centres valent `(col + 0.5) * cell` : arrondir `x / cell` ferait
    # tomber deux cellules voisines sur le même index, et le voisinage
    # ressortait vide — le lissage ne retournait alors aucune cellule.
    index = _grid_index(cells, cell_m)
    changed = 0

    for _ in range(SMOOTHING_PASSES):
        kinds = [c.kind for c in cells]
        updates: list[tuple[int, str]] = []

        for (gx, gy), position in index.items():
            neighbours = [
                kinds[index[(gx + dx, gy + dy)]]
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                if (dx, dy) != (0, 0) and (gx + dx, gy + dy) in index
            ]
            if len(neighbours) < 4:
                continue

            counts: dict[str, int] = {}
            for kind in neighbours:
                counts[kind] = counts.get(kind, 0) + 1
            counts.pop("indetermine", None)
            if not counts:
                continue

            best = max(counts, key=counts.get)
            if best == kinds[position]:
                continue
            if counts[best] / len(neighbours) >= SMOOTHING_MAJORITY:
                updates.append((position, best))

        for position, kind in updates:
            cells[position].kind = kind
        changed += len(updates)
        if not updates:
            break

    return changed


def _absorb_small_patches(cells: list[SurfaceCell], cell_m: float) -> int:
    """Absorbe les plages trop petites dans la nature qui les entoure."""
    if not cells:
        return 0

    index = _grid_index(cells, cell_m)
    seen: set[tuple[int, int]] = set()
    absorbed = 0

    for start in index:
        if start in seen:
            continue
        kind = cells[index[start]].kind
        stack = [start]
        seen.add(start)
        members = []
        while stack:
            key = stack.pop()
            members.append(key)
            gx, gy = key
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    other = (gx + dx, gy + dy)
                    if (
                        other in index
                        and other not in seen
                        and cells[index[other]].kind == kind
                    ):
                        seen.add(other)
                        stack.append(other)

        if len(members) >= MIN_PATCH_CELLS:
            continue

        # La plage est trop petite : elle prend la nature dominante autour.
        around: dict[str, int] = {}
        for gx, gy in members:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    other = (gx + dx, gy + dy)
                    if other in index and other not in members:
                        neighbour = cells[index[other]].kind
                        if neighbour != kind:
                            around[neighbour] = around.get(neighbour, 0) + 1
        if not around:
            continue
        winner = max(around, key=around.get)
        for key in members:
            cells[index[key]].kind = winner
        absorbed += len(members)

    return absorbed


def classify_ground(
    laz_path: Path,
    centre: tuple[float, float],
    radius_m: float = 90.0,
    cell_m: float = SURFACE_CELL_M,
    footprints: list | None = None,
) -> SurfaceMap:
    """Range le sol en cellules végétales, minérales ou indéterminées.

    Le seuil d'intensité n'est pas décrété : il est pris à la médiane des
    cellules du site. Un relevé plus réfléchissant qu'un autre — capteur,
    saison, humidité — déplace toute la distribution, et une valeur absolue
    échouerait sur la tuile suivante.
    """
    from .laz_cache import read_window

    laz_path = Path(laz_path)
    if not laz_path.is_file():
        log.info("tuile LiDAR absente : %s", laz_path)
        return SurfaceMap(hotel_id="unknown")

    # La lecture est partagée : la végétation parcourt la même tuile, et
    # relire vingt-trois millions de points coûtait vingt secondes.
    window = read_window(laz_path, centre, radius_m)
    if window is None:
        log.info("aucun point de sol dans l'emprise")
        return SurfaceMap(hotel_id="unknown")

    ground = window.classification == GROUND_CLASS
    xs_list = [window.x[ground]] if ground.any() else []
    ys_list = [window.y[ground]] if ground.any() else []
    intensity_list = [window.intensity[ground]] if ground.any() else []
    has_colour = window.red is not None
    red_list = [window.red[ground]] if has_colour and ground.any() else []
    green_list = [window.green[ground]] if has_colour and ground.any() else []
    blue_list = [window.blue[ground]] if has_colour and ground.any() else []

    if not xs_list:
        log.info("aucun point de sol dans l'emprise")
        return SurfaceMap(hotel_id="unknown")

    xs = np.concatenate(xs_list)
    ys = np.concatenate(ys_list)
    intensity = np.concatenate(intensity_list)
    greenness = None
    if has_colour and red_list:
        red = np.concatenate(red_list)
        green = np.concatenate(green_list)
        blue = np.concatenate(blue_list)
        total = np.maximum(red + green + blue, 1.0)
        greenness = (2.0 * green - red - blue) / total

    # Les points sous une emprise bâtie ne décrivent pas le sol du site.
    if footprints:
        import shapely
        from shapely.geometry import Polygon

        built = np.zeros(xs.size, dtype=bool)
        for ring in footprints:
            polygon = Polygon(ring)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if not polygon.is_empty:
                built |= shapely.contains_xy(polygon, xs, ys)
        if built.any():
            xs, ys, intensity = xs[~built], ys[~built], intensity[~built]
            if greenness is not None:
                greenness = greenness[~built]

    if xs.size == 0:
        return SurfaceMap(hotel_id="unknown")

    col = np.floor(xs / cell_m).astype(np.int64)
    row = np.floor(ys / cell_m).astype(np.int64)
    order = np.lexsort((col, row))
    col, row = col[order], row[order]
    intensity = intensity[order]
    green_sorted = greenness[order] if greenness is not None else None

    keys = np.stack([row, col], axis=1)
    _, starts = np.unique(keys, axis=0, return_index=True)
    starts = np.sort(starts)
    bounds = np.append(starts, len(keys))

    cell_intensity, cell_green, cell_counts, cell_xy = [], [], [], []
    for begin, end in zip(bounds[:-1], bounds[1:]):
        if end - begin < MIN_POINTS_PER_CELL:
            continue
        cell_intensity.append(float(np.median(intensity[begin:end])))
        cell_green.append(
            float(np.median(green_sorted[begin:end])) if green_sorted is not None else None
        )
        cell_counts.append(int(end - begin))
        cell_xy.append(
            ((col[begin] + 0.5) * cell_m, (row[begin] + 0.5) * cell_m)
        )

    if not cell_intensity:
        return SurfaceMap(hotel_id="unknown")

    values = np.asarray(cell_intensity)
    threshold = float(np.median(values))

    cells: list[SurfaceCell] = []
    for (x, y), value, verdure, count in zip(
        cell_xy, cell_intensity, cell_green, cell_counts
    ):
        if value >= threshold + INTENSITY_MARGIN:
            kind = "vegetal"
        elif value <= threshold - INTENSITY_MARGIN:
            kind = "mineral"
        elif verdure is not None and verdure >= GREENNESS_CONFIRM:
            # La couleur ne tranche pas seule, mais elle départage une cellule
            # que l'intensité laisse dans l'indécision.
            kind = "vegetal"
        else:
            kind = "indetermine"
        cells.append(
            SurfaceCell(
                x=x, y=y, kind=kind, intensity=value, greenness=verdure, points=count
            )
        )

    # La classification cellule par cellule ignore le voisinage : elle produit
    # un damier là où le terrain forme des plages. Le lissage la rend continue,
    # sans jamais déplacer une frontière franche.
    smoothed = _smooth(cells, cell_m)
    absorbed = _absorb_small_patches(cells, cell_m)
    log.info(
        "lissage du sol : %d cellule(s) retournée(s), %d absorbée(s)",
        smoothed,
        absorbed,
    )

    surface = SurfaceMap(
        hotel_id="unknown",
        cells=cells,
        cell_m=cell_m,
        threshold=threshold,
        provenance={
            "laz": str(laz_path),
            "radius_m": radius_m,
            "has_colour": has_colour,
            "intensity_margin": INTENSITY_MARGIN,
            "smoothed_cells": smoothed,
            "absorbed_cells": absorbed,
        },
    )
    log.info("sol classé : %s", surface.by_kind())
    return surface
