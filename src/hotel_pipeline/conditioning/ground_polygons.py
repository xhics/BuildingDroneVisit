"""Contours des plages de sol, en polygones plutôt qu'en damier de cellules.

Une carte de cellules décrit correctement *où* se trouve le gazon, mais elle le
rend en marches d'escalier : la bordure d'une allée devient une succession de
rectangles de la taille de la maille. Affiner la maille ne résout rien — à un
mètre, le pilote produit vingt-sept mille dalles à rasteriser par image, et le
rendu d'une séquence passe de six à plus de vingt minutes.

La bordure est donc extraite comme **contour** : la grille fine sert à trouver
la frontière au demi-pixel près, puis le tracé est simplifié en un polygone de
quelques dizaines de sommets. Trente-huit polygones remplacent les vingt-sept
mille dalles, avec des bords nets là où le damier montrait des créneaux.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..logging import get_logger

log = get_logger("conditioning-ground-polygons")

#: Lissage appliqué à la grille avant extraction, en cellules. Mesuré sur ce
#: pilote : à 1,2, quinze pour cent des plages ressortaient très dentelées —
#: un périmètre trois fois celui d'un disque de même aire — et le sol prenait
#: un aspect fractal que rien dans le terrain ne justifie. Une pelouse a un
#: bord franc ; ses digitations d'un mètre sont du bruit de classification.
CONTOUR_SMOOTHING = 2.6

#: Tolérance de simplification du tracé, en cellules. Au-delà, une bordure
#: courbe se rabat en segments trop longs et l'allée redevient anguleuse.
SIMPLIFY_TOLERANCE = 1.6

#: En deçà, un contour décrit du bruit résiduel plutôt qu'une plage.
MIN_CONTOUR_VERTICES = 20

#: Part d'indéterminé au-delà de laquelle le relevé est tenu pour uniforme :
#: le seuil d'intensité tombe alors au milieu d'une population unique et rien
#: ne le franchit. Le sol est posé sans nature plutôt que de disparaître — un
#: terrain existe même quand on ignore son revêtement.
UNIFORM_GROUND_SHARE = 0.6

#: Aire minimale d'une plage retenue, en mètres carrés. Relevé de douze à
#: quarante : les plages en deçà sont des îlots de quelques cellules qui
#: multiplient les fragments sans décrire de surface reconnaissable.
MIN_PATCH_AREA_M2 = 40.0


@dataclass
class GroundPatch:
    """Une plage de sol d'une seule nature, décrite par son contour."""

    kind: str
    #: Sommets du contour fermé, en CRS projeté.
    ring: list[tuple[float, float]]

    def area_m2(self) -> float:
        if len(self.ring) < 3:
            return 0.0
        xs = np.array([p[0] for p in self.ring])
        ys = np.array([p[1] for p in self.ring])
        return float(abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1))) / 2)

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "vertices": len(self.ring),
            "area_m2": round(self.area_m2(), 1),
        }


@dataclass
class GroundPatches:
    """Les plages de sol d'un site, prêtes au rendu."""

    hotel_id: str
    patches: list[GroundPatch] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    def by_kind(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for patch in self.patches:
            totals[patch.kind] = totals.get(patch.kind, 0.0) + patch.area_m2()
        return {k: round(v, 1) for k, v in totals.items()}

    def as_dict(self) -> dict:
        return {
            "hotel_id": self.hotel_id,
            "patch_count": len(self.patches),
            "area_by_kind_m2": self.by_kind(),
            "provenance": self.provenance,
            "patches": [p.as_dict() for p in self.patches],
            "caveats": [
                "un contour suit la frontière que la grille fine établit : il "
                "la lisse, il ne la mesure pas plus finement que le relevé",
                "une plage sous le seuil d'aire est écartée : quelques mètres "
                "carrés isolés relèvent du bruit du capteur",
                "une cellule que la mesure n'a pas tranchée est rattachée à la "
                "nature qui l'entoure pour fermer la surface ; la carte de "
                "classification, elle, continue de la compter indéterminée",
            ],
        }


def _cover_extent(cells: list, cell_m: float, hotel_id: str) -> GroundPatches:
    """Emprise couverte, en une seule plage sans nature établie."""
    xs = [c.x for c in cells]
    ys = [c.y for c in cells]
    half = cell_m * 0.5
    ring = [
        (min(xs) - half, min(ys) - half),
        (max(xs) + half, min(ys) - half),
        (max(xs) + half, max(ys) + half),
        (min(xs) - half, max(ys) + half),
    ]
    ring.append(ring[0])
    result = GroundPatches(hotel_id=hotel_id)
    result.patches.append(GroundPatch(kind="indetermine_pose", ring=ring))
    result.provenance = {
        "cell_m": cell_m,
        "source_cells": len(cells),
        "uniform_ground": True,
    }
    return result


def _effective_kind(cell) -> str:  # noqa: ANN001
    """Nature retenue pour le rendu, doute comblé compris."""
    return getattr(cell, "filled_kind", None) or cell.kind


def _fill_undecided(cells: list) -> int:
    """Rattache chaque cellule indéterminée à la nature qui l'entoure.

    Le comblement est écrit dans un attribut distinct : la classification
    d'origine reste intacte, et un rapport peut toujours dire combien de
    cellules n'ont pas été tranchées par la mesure.
    """
    index = {
        (int(np.floor(c.x)), int(np.floor(c.y))): i for i, c in enumerate(cells)
    }
    undecided = [
        key for key, position in index.items() if cells[position].kind == "indetermine"
    ]
    if not undecided:
        return 0

    filled = 0
    # Plusieurs passes : un trou large se comble depuis ses bords, de proche
    # en proche, plutôt que de rester vide en son centre.
    for _ in range(6):
        remaining = []
        for key in undecided:
            gx, gy = key
            counts: dict[str, int] = {}
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    other = index.get((gx + dx, gy + dy))
                    if other is None:
                        continue
                    kind = _effective_kind(cells[other])
                    if kind != "indetermine":
                        counts[kind] = counts.get(kind, 0) + 1
            if not counts:
                remaining.append(key)
                continue
            cells[index[key]].filled_kind = max(counts, key=counts.get)
            filled += 1
        if not remaining or len(remaining) == len(undecided):
            break
        undecided = remaining

    return filled


def from_cells(
    cells: list,
    cell_m: float,
    hotel_id: str = "unknown",
    kinds: tuple[str, ...] = ("vegetal", "mineral"),
) -> GroundPatches:
    """Extrait les contours des plages depuis une carte de cellules.

    Chaque nature est traitée séparément : le contour d'une plage de gazon suit
    sa propre frontière, sans hériter des créneaux du minéral voisin.
    """
    try:
        from scipy.ndimage import gaussian_filter
        from skimage.measure import approximate_polygon, find_contours
    except ImportError as exc:  # pragma: no cover - dépend de l'installation
        log.info("extraction de contours indisponible (%s)", exc)
        return GroundPatches(hotel_id=hotel_id)

    if not cells:
        return GroundPatches(hotel_id=hotel_id)

    xs = sorted({c.x for c in cells})
    ys = sorted({c.y for c in cells})
    if len(xs) < 3 or len(ys) < 3:
        return GroundPatches(hotel_id=hotel_id)

    col = {value: i for i, value in enumerate(xs)}
    row = {value: i for i, value in enumerate(ys)}
    x0, y0 = xs[0], ys[0]

    # Une cellule indéterminée n'est pas un trou dans le terrain : la chaussée
    # continue sous le doute. Laissée vide, elle se voyait comme une tache
    # noire au milieu d'une route — quatre pour cent du sol, en petits groupes
    # dispersés. Elle est donc rattachée à la nature qui l'entoure, ce qui
    # ferme la surface sans effacer l'incertitude : le rapport de
    # classification, lui, continue de la compter comme indéterminée.
    resolved = _fill_undecided(cells)
    if resolved:
        log.info("%d cellule(s) indéterminée(s) comblée(s) par voisinage", resolved)

    # Un relevé trop uniforme laisse tout indéterminé : le seuil dérivé tombe
    # au milieu d'une population unique et rien ne franchit la marge. Le sol
    # disparaissait alors entièrement du rendu, ce qui est pire que de le poser
    # sans nature — un terrain existe même quand on ignore son revêtement.
    undecided = sum(1 for c in cells if _effective_kind(c) == "indetermine")
    if undecided > len(cells) * UNIFORM_GROUND_SHARE:
        log.info(
            "sol trop uniforme pour être distingué : %d cellule(s) posées "
            "sans nature",
            len(cells),
        )
        for cell in cells:
            cell.filled_kind = "indetermine_pose"
        # Une grille pleine n'a pas de frontière intérieure : l'extraction de
        # contours n'y trouverait rien. L'emprise couverte est donc posée
        # telle quelle, comme un rectangle unique.
        return _cover_extent(cells, cell_m, hotel_id)

    result = GroundPatches(hotel_id=hotel_id)
    for kind in kinds:
        grid = np.zeros((len(ys), len(xs)), dtype=np.float64)
        for cell in cells:
            if _effective_kind(cell) == kind:
                grid[row[cell.y], col[cell.x]] = 1.0
        if not grid.any():
            continue

        # Le lissage arrondit les créneaux d'une cellule ; le seuil à un demi
        # place ensuite le contour entre deux cellules, au demi-pas près.
        smoothed = gaussian_filter(grid, CONTOUR_SMOOTHING)
        for contour in find_contours(smoothed, 0.5):
            if len(contour) < MIN_CONTOUR_VERTICES:
                continue
            simplified = approximate_polygon(contour, tolerance=SIMPLIFY_TOLERANCE)
            if len(simplified) < 4:
                continue

            ring = [
                (x0 + point[1] * cell_m, y0 + point[0] * cell_m)
                for point in simplified
            ]
            if ring[0] != ring[-1]:
                ring.append(ring[0])

            patch = GroundPatch(kind=kind, ring=ring)
            if patch.area_m2() < MIN_PATCH_AREA_M2:
                continue
            result.patches.append(patch)

    result.provenance = {
        "cell_m": cell_m,
        "source_cells": len(cells),
        "smoothing": CONTOUR_SMOOTHING,
        "simplify_tolerance": SIMPLIFY_TOLERANCE,
    }
    log.info(
        "plages de sol : %d polygone(s) depuis %d cellule(s)",
        len(result.patches),
        len(cells),
    )
    return result
