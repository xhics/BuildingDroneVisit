"""Relief des façades, lu dans le nuage plutôt que dans le raster de toiture.

Le modèle de hauteur de canopée est une vue de dessus : il donne l'altitude du
toit, jamais ce qui se passe le long d'un mur. Les murs étaient donc dressés
entre deux hauteurs interpolées, et les décrochements verticaux — un pignon qui
monte, une avancée d'entrée, une aile plus basse — s'aplanissaient.

Le nuage brut, lui, en porte la trace. Mesuré sur ce pilote : six mille cinq
cents retours de classe bâtiment entre un et huit mètres dans l'emprise cible,
dont quarante pour cent à moins de deux mètres d'un bord. Le long des arêtes,
la hauteur varie de un à trois mètres — le pignon central s'y lit.

Le module échantillonne cette variation et rend un profil par arête, assez fin
pour que le mur suive le bâti au lieu de le moyenner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..logging import get_logger

log = get_logger("conditioning-facade")

#: Classe ASPRS du bâti.
BUILDING_CLASS = 6

#: Distance à l'arête en deçà de laquelle un point décrit ce mur, en mètres.
#: Deux mètres et demi captent la façade sans absorber le cœur de la toiture.
WALL_BAND_M = 2.5

#: Longueur d'un segment de mur échantillonné, en mètres. Trois mètres suivent
#: un pignon sans hacher une façade droite.
SEGMENT_M = 3.0

#: Retours minimaux dans un segment pour en tirer une hauteur.
MIN_SEGMENT_POINTS = 6

#: Percentile retenu par segment : le haut du mur, superstructures écartées.
SEGMENT_PERCENTILE = 92.0


@dataclass
class EdgeProfile:
    """Hauteurs relevées le long d'une arête d'emprise."""

    edge_index: int
    #: Hauteurs par segment, du premier sommet vers le second.
    heights: np.ndarray
    points: int

    @property
    def relief_m(self) -> float:
        finite = self.heights[np.isfinite(self.heights)]
        return float(finite.max() - finite.min()) if finite.size else 0.0

    def as_dict(self) -> dict:
        return {
            "edge_index": self.edge_index,
            "segments": int(self.heights.size),
            "relief_m": round(self.relief_m, 2),
            "points": self.points,
        }


@dataclass
class FacadeRelief:
    """Relief de façade d'un volume, arête par arête."""

    feature_id: str
    profiles: dict[int, EdgeProfile] = field(default_factory=dict)

    @property
    def max_relief_m(self) -> float:
        return max((p.relief_m for p in self.profiles.values()), default=0.0)

    def height_along(self, edge_index: int, t: float) -> float | None:
        """Hauteur du mur à une fraction `t` d'une arête, si elle est relevée."""
        profile = self.profiles.get(edge_index)
        if profile is None:
            return None
        heights = profile.heights
        if heights.size == 0:
            return None
        finite = np.isfinite(heights)
        if not finite.any():
            return None

        position = float(np.clip(t, 0.0, 1.0)) * (heights.size - 1)
        low, high = int(np.floor(position)), int(np.ceil(position))
        a, b = heights[low], heights[high]

        if np.isfinite(a) and np.isfinite(b):
            weight = position - low
            return float(a * (1 - weight) + b * weight)
        if np.isfinite(a):
            return float(a)
        if np.isfinite(b):
            return float(b)

        # Les deux bornes manquent — un segment sans retour au milieu d'une
        # arête. Interpoler sur les segments relevés de part et d'autre garde
        # le mur continu : rendre `None` le creusait jusqu'au sol.
        indices = np.flatnonzero(finite)
        return float(np.interp(position, indices, heights[indices]))

    def as_dict(self) -> dict:
        return {
            "feature_id": self.feature_id,
            "edges_profiled": len(self.profiles),
            "max_relief_m": round(self.max_relief_m, 2),
            "profiles": [p.as_dict() for p in self.profiles.values()],
        }


def read_relief(
    laz_path: Path,
    prism,  # noqa: ANN001
    ground_z: float,
    band_m: float = WALL_BAND_M,
    segment_m: float = SEGMENT_M,
    cloud: tuple | None = None,
) -> FacadeRelief:
    """Relève la hauteur du bâti le long de chaque arête d'une emprise.

    Seuls les retours **de classe bâtiment** comptent : un arbre appuyé contre
    une façade ferait monter le mur à sa place.
    """
    from .laz_cache import read_window

    result = FacadeRelief(feature_id=prism.feature_id)
    footprint = prism.footprint
    centre = footprint.mean(axis=0)
    radius = float(
        np.hypot(footprint[:, 0] - centre[0], footprint[:, 1] - centre[1]).max()
    ) + band_m + 5.0

    if cloud is not None:
        xs, ys, zs = cloud
    else:
        # Une lecture par volume relisait le nuage vingt-huit fois, pour cinq
        # minutes là où une seule fenêtre suffit. `apply_relief` la partage.
        window = read_window(
            Path(laz_path), (float(centre[0]), float(centre[1])), radius
        )
        if window is None:
            return result
        built = window.classification == BUILDING_CLASS
        if not built.any():
            return result
        xs, ys, zs = window.x[built], window.y[built], window.z[built] - ground_z

    # Restreindre au voisinage du volume : le reste ne décrit pas ses murs.
    near_volume = (np.abs(xs - centre[0]) <= radius) & (
        np.abs(ys - centre[1]) <= radius
    )
    if not near_volume.any():
        return result
    xs, ys, zs = xs[near_volume], ys[near_volume], zs[near_volume]

    count = len(footprint)
    for index in range(count):
        start = footprint[index]
        end = footprint[(index + 1) % count]
        direction = end - start
        length = float(np.hypot(*direction))
        if length < segment_m * 2:
            continue

        # Projection sur l'arête : abscisse le long du mur, et écart latéral.
        rel_x = xs - start[0]
        rel_y = ys - start[1]
        t = (rel_x * direction[0] + rel_y * direction[1]) / (length * length)
        lateral = np.abs(rel_x * direction[1] - rel_y * direction[0]) / length
        near = (t >= 0.0) & (t <= 1.0) & (lateral <= band_m)
        if near.sum() < MIN_SEGMENT_POINTS * 2:
            continue

        segments = max(int(length / segment_m), 2)
        heights = np.full(segments, np.nan)
        bucket = np.clip((t[near] * segments).astype(int), 0, segments - 1)
        band_z = zs[near]
        for slot in range(segments):
            selected = band_z[bucket == slot]
            if selected.size >= MIN_SEGMENT_POINTS:
                heights[slot] = float(np.percentile(selected, SEGMENT_PERCENTILE))

        if np.isfinite(heights).sum() < 2:
            continue
        result.profiles[index] = EdgeProfile(
            edge_index=index, heights=heights, points=int(near.sum())
        )

    if result.profiles:
        log.info(
            "%s : %d arête(s) profilée(s), relief max %.1f m",
            prism.feature_id,
            len(result.profiles),
            result.max_relief_m,
        )
    return result


def apply_relief(scene, laz_path: Path, ground_z: float | None = None) -> dict:  # noqa: ANN001
    """Relève le relief de façade de chaque volume et l'attache à la scène.

    Le sol de référence vient du terrain quand il est connu ; à défaut, la
    médiane des retours de sol autour du site. Une altitude fausse déplacerait
    tous les murs d'autant.
    """
    from .laz_cache import read_window

    laz_path = Path(laz_path)
    if not laz_path.is_file():
        return {"profiled": 0, "total": len(scene.prisms)}

    if ground_z is None:
        window = read_window(laz_path, scene.centre, 120.0)
        if window is None:
            return {"profiled": 0, "total": len(scene.prisms)}
        ground = window.classification == 2
        if not ground.any():
            return {"profiled": 0, "total": len(scene.prisms)}
        ground_z = float(np.median(window.z[ground]))

    # Une seule fenêtre, assez large pour tous les volumes de la scène.
    span = 0.0
    for prism in scene.prisms:
        centre = prism.footprint.mean(axis=0)
        span = max(
            span,
            float(np.hypot(centre[0] - scene.centre[0], centre[1] - scene.centre[1]))
            + float(
                np.hypot(
                    prism.footprint[:, 0] - centre[0],
                    prism.footprint[:, 1] - centre[1],
                ).max()
            ),
        )

    shared = read_window(laz_path, scene.centre, span + WALL_BAND_M + 10.0)
    cloud = None
    if shared is not None:
        built = shared.classification == BUILDING_CLASS
        if built.any():
            cloud = (
                shared.x[built],
                shared.y[built],
                shared.z[built] - ground_z,
            )

    profiled = 0
    max_relief = 0.0
    for prism in scene.prisms:
        relief = read_relief(laz_path, prism, ground_z, cloud=cloud)
        if relief.profiles:
            prism.facade_relief = relief
            profiled += 1
            max_relief = max(max_relief, relief.max_relief_m)

    log.info("relief de façade : %d/%d volume(s)", profiled, len(scene.prisms))
    return {
        "profiled": profiled,
        "total": len(scene.prisms),
        "ground_z": round(ground_z, 2),
        "max_relief_m": round(max_relief, 2),
    }
