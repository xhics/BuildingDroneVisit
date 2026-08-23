"""Segmentation des couronnes et tri des objets verticaux fins.

Deux défauts d'un regroupement par simple connexité, l'un et l'autre mesurés
sur ce pilote et visibles au rendu.

**Les massifs sont trop grossiers.** Relier de proche en proche les cellules
occupées fusionne une rangée d'arbres en un bloc de plus de mille mètres
carrés. Le rendu produisait alors de gros pavés verts sans rapport avec des
arbres. La segmentation par bassins versants, amorcée aux sommets locaux du
modèle de hauteur de canopée, rend ici quatre-vingt-douze couronnes de trois
mètres de rayon médian — c'est-à-dire des arbres.

**Un poteau n'est pas un arbre.** Quatorze pour cent des amas retenus faisaient
moins de quatre mètres carrés d'emprise pour plus de trois mètres de haut :
lampadaires, mâts d'enseigne, panneaux. Rendus en vert, ils plaçaient de la
végétation là où le générateur devait voir du mobilier urbain. Ils sont
désormais classés à part, sur leur signature géométrique.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..logging import get_logger

log = get_logger("conditioning-canopy")

#: Résolution du modèle de hauteur de canopée, en mètres.
CHM_RESOLUTION_M = 0.5

#: Rayon de recherche d'un sommet, en cellules. Neuf cellules à cinquante
#: centimètres séparent deux couronnes voisines sans découper la même en deux.
PEAK_WINDOW = 9

#: En deçà, un maximum local n'est pas un sommet d'arbre mais du bruit.
MIN_PEAK_HEIGHT_M = 2.0

#: Emprise maximale d'un objet vertical fin — poteau, mât, panneau.
POLE_MAX_FOOTPRINT_M2 = 3.0

#: Un poteau est **continu** du sol à son sommet. Ce critère est le seul qui
#: le distingue d'une branche isolée : mesurée sans lui, la détection rendait
#: cent quarante-deux objets de moins d'un mètre carré, dont l'écrasante
#: majorité n'étaient que des retours épars dans une couronne. Un mât porte des
#: retours à toutes les hauteurs ; une branche n'en a qu'en altitude.
POLE_MIN_VERTICAL_FILL = 0.55

#: Hauteur du bas d'un poteau : au-delà, l'objet ne part pas du sol et n'est
#: donc pas un mât mais un fragment de houppier.
POLE_MAX_BASE_M = 1.5

#: Hauteur minimale pour qu'un objet fin compte comme mobilier plutôt que buisson.
POLE_MIN_HEIGHT_M = 2.5

#: Un objet fin mais large en haut reste un arbre : la couronne s'étale.
POLE_MAX_SPREAD_RATIO = 0.45

#: Rapport rayon/hauteur plafonné, par nature. Le rayon vient de l'aire du
#: bassin versant et ne connaît pas la hauteur : mesuré sur ce pilote, les
#: arbustes ressortaient à 1,26 — plus larges que hauts — contre 0,46 pour les
#: arbres matures. À l'écran, un buisson paraissait donc plus massif qu'un
#: grand arbre voisin. Les valeurs retenues suivent le port réel : une couronne
#: s'étale sur environ la moitié de sa hauteur, un arbuste reste ramassé.
MAX_RADIUS_RATIO: dict[str, float] = {
    "couronne": 0.55,
    "buisson": 0.90,
    "poteau": 0.40,
}

ObjectKind = Literal["couronne", "poteau", "buisson"]


@dataclass
class CanopyObject:
    """Un objet vertical isolé, et ce que sa forme établit."""

    kind: ObjectKind
    centre: tuple[float, float]
    radius_m: float
    height_m: float
    footprint_m2: float
    points: int

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "centre": [round(c, 2) for c in self.centre],
            "radius_m": round(self.radius_m, 2),
            "height_m": round(self.height_m, 2),
            "footprint_m2": round(self.footprint_m2, 1),
            "points": self.points,
        }


def _classify(
    footprint_m2: float, height_m: float, spread_ratio: float
) -> ObjectKind:
    """Nature d'un objet d'après son emprise, sa hauteur et son étalement.

    Le discriminant est la forme, non l'étiquette : la tuile ne classe pas la
    végétation, et un lampadaire n'y porte aucune marque. Un objet haut, étroit
    et qui ne s'élargit pas vers le sommet est du mobilier ; un objet qui
    s'étale en hauteur est une couronne.
    """
    if (
        footprint_m2 <= POLE_MAX_FOOTPRINT_M2
        and height_m >= POLE_MIN_HEIGHT_M
        and spread_ratio <= POLE_MAX_SPREAD_RATIO
    ):
        return "poteau"
    if height_m < POLE_MIN_HEIGHT_M:
        return "buisson"
    return "couronne"


def _isolated_poles(
    xs: np.ndarray,
    ys: np.ndarray,
    heights: np.ndarray,
    resolution_m: float,
) -> tuple[list[CanopyObject], np.ndarray]:
    """Repère les objets fins et hauts **avant** la segmentation par bassins.

    Un lampadaire ne produit pas de maximum local isolé : le bassin d'un arbre
    voisin l'absorbe, et il ressortait classé en couronne. Mesuré sur ce
    pilote, la segmentation seule n'en retenait qu'un sur les neuf que la
    distribution annonçait.

    Le tri se fait donc en amont, sur la colonne verticale : une cellule dont
    l'emprise reste étroite sur toute sa hauteur est du mobilier, et ses points
    sont retirés du nuage avant que les couronnes ne soient découpées.
    """
    from scipy import ndimage

    x0, y0 = float(xs.min()), float(ys.min())
    col = ((xs - x0) / resolution_m).astype(np.int64)
    row = ((ys - y0) / resolution_m).astype(np.int64)
    width = int(col.max()) + 1
    height = int(row.max()) + 1

    grid = np.zeros((height, width), dtype=np.float64)
    np.maximum.at(grid, (row, col), heights)

    tall = grid >= POLE_MIN_HEIGHT_M
    if not tall.any():
        return [], np.zeros(xs.size, dtype=bool)

    labels, count = ndimage.label(tall)
    poles: list[CanopyObject] = []
    taken = np.zeros(xs.size, dtype=bool)
    point_labels = labels[row, col]

    for identifier in range(1, count + 1):
        cells = int((labels == identifier).sum())
        footprint = cells * resolution_m * resolution_m
        if footprint > POLE_MAX_FOOTPRINT_M2:
            continue

        selected = point_labels == identifier
        if selected.sum() < 4:
            continue
        px, py, ph = xs[selected], ys[selected], heights[selected]
        top = float(ph.max())
        if top < POLE_MIN_HEIGHT_M:
            continue

        # Un mât part du sol : sinon c'est une branche suspendue.
        if float(ph.min()) > POLE_MAX_BASE_M:
            continue

        # Et il est continu : ses retours couvrent la plupart des tranches de
        # hauteur, là où une branche n'occupe qu'une bande étroite.
        slices = max(int(top // 1.0), 1)
        occupied = len({int(value) for value in ph})
        if occupied / slices < POLE_MIN_VERTICAL_FILL:
            continue

        poles.append(
            CanopyObject(
                kind="poteau",
                centre=(float(px.mean()), float(py.mean())),
                radius_m=float(max(np.sqrt(footprint / np.pi), resolution_m)),
                height_m=top,
                footprint_m2=footprint,
                points=int(selected.sum()),
            )
        )
        taken |= selected

    return poles, taken


def segment(
    xs: np.ndarray,
    ys: np.ndarray,
    heights: np.ndarray,
    resolution_m: float = CHM_RESOLUTION_M,
) -> list[CanopyObject]:
    """Isole les objets verticaux d'un nuage, un par couronne.

    Les points sont rasterisés en modèle de hauteur de canopée, les sommets
    locaux servent de germes, et les bassins versants découpent le reste. C'est
    la méthode usuelle en foresterie, et elle vaut ici parce qu'on cherche
    exactement la même chose : séparer des houppiers qui se touchent.
    """
    if xs.size == 0:
        return []

    try:
        from scipy import ndimage
        from skimage.segmentation import watershed
    except ImportError as exc:  # pragma: no cover - dépend de l'installation
        log.info("segmentation indisponible (%s) : aucun objet isolé", exc)
        return []

    # Le mobilier est écarté d'abord : laissé dans le nuage, il serait absorbé
    # par le bassin d'une couronne voisine et rendu comme végétation.
    poles, taken = _isolated_poles(xs, ys, heights, resolution_m)
    if taken.any():
        xs, ys, heights = xs[~taken], ys[~taken], heights[~taken]
    if xs.size == 0:
        return poles

    x0, y0 = float(xs.min()), float(ys.min())
    width = int((xs.max() - x0) / resolution_m) + 1
    height = int((ys.max() - y0) / resolution_m) + 1
    if width < 3 or height < 3:
        return []

    chm = np.zeros((height, width), dtype=np.float64)
    col = ((xs - x0) / resolution_m).astype(np.int64)
    row = ((ys - y0) / resolution_m).astype(np.int64)
    np.maximum.at(chm, (row, col), heights)

    # Une dilatation légère comble les trous entre deux retours du même arbre,
    # sans réunir deux couronnes distinctes.
    smoothed = ndimage.gaussian_filter(ndimage.maximum_filter(chm, size=3), 1.0)

    peaks = (
        smoothed == ndimage.maximum_filter(smoothed, size=PEAK_WINDOW)
    ) & (smoothed > MIN_PEAK_HEIGHT_M)
    markers, count = ndimage.label(peaks)
    if count == 0:
        return []

    labels = watershed(-smoothed, markers, mask=smoothed > 1.0)

    # Chaque point du nuage hérite du bassin de sa cellule.
    point_labels = labels[row, col]

    objects: list[CanopyObject] = []
    for identifier in range(1, count + 1):
        selected = point_labels == identifier
        if selected.sum() < 8:
            continue
        px, py, ph = xs[selected], ys[selected], heights[selected]

        cells = int((labels == identifier).sum())
        footprint = cells * resolution_m * resolution_m
        top = float(np.percentile(ph, 95))
        centre = (float(px.mean()), float(py.mean()))
        radius = float(max(np.sqrt(footprint / np.pi), resolution_m))

        # Étalement : part de l'emprise occupée dans la moitié haute de
        # l'objet. Une couronne s'y déploie, un mât n'y est qu'un point.
        upper = ph >= top * 0.6
        spread = float(upper.sum() / max(selected.sum(), 1))

        kind = _classify(footprint, top, spread)
        # Le rayon est plafonné par la hauteur : un bassin large mais bas
        # décrivait un volume trapu qui écrasait visuellement ses voisins.
        limit = MAX_RADIUS_RATIO.get(kind, 0.55) * top
        objects.append(
            CanopyObject(
                kind=kind,
                centre=centre,
                radius_m=min(radius, max(limit, resolution_m)),
                height_m=top,
                footprint_m2=footprint,
                points=int(selected.sum()),
            )
        )

    objects.extend(poles)

    kinds: dict[str, int] = {}
    for item in objects:
        kinds[item.kind] = kinds.get(item.kind, 0) + 1
    log.info("segmentation : %d objet(s) — %s", len(objects), kinds)
    return objects
