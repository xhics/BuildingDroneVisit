"""Couverture d'apparence, façade par façade (Lot 1B, complément).

`appearance_coverage` valait `"partial" if kind == "FACADE_PRIMARY" else "none"` :
une constante littérale, vraie par accident sur le pilote et fausse partout
ailleurs. Un second site aurait hérité du verdict du premier.

Ce module mesure ce que chaque mur a réellement reçu. Trois conditions
indépendantes, toutes nécessaires :

```text
dans le cadre     l'azimut du point tombe dans le champ de la caméra
de face           la normale extérieure du mur regarde vers la caméra
non masqué        ni l'empreinte elle-même, ni un bâtiment voisin
```

La deuxième condition manquait à toute mesure antérieure : sans elle, un mur
arrière « comptait » depuis l'avant, puisque seul l'angle était vérifié.

Ce que ce module **ne** sait pas : les arbres, les clôtures, les panoramas pris
à l'intérieur d'un autre bâtiment. La couverture géométrique majore donc
toujours la couverture d'apparence, et c'est pourquoi un constat humain reste
requis avant de promouvoir un mur.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from shapely.geometry import LineString, Point

from ..logging import get_logger

log = get_logger("facade-coverage")

#: Points échantillonnés par segment de mur. Onze bornes donnent des dixièmes
#: de mur : plus fin ne se lit plus dans un rapport, moins fin manque une baie.
SAMPLES_PER_SEGMENT = 10

#: Au-delà, un mur occupe trop peu de pixels pour porter une texture. Ce n'est
#: pas une limite de visibilité mais d'exploitabilité.
DEFAULT_MAX_DISTANCE_M = 150.0

#: Tolérance d'intersection : un rayon frôlant un coin ne masque pas un mur.
_GRAZING_M = 0.5


@dataclass(frozen=True)
class FacadeSample:
    """Un point de mur et la direction vers laquelle il regarde."""

    x: float
    y: float
    normal: tuple[float, float]


@dataclass
class FacadeVisibility:
    """Ce qu'une position voit d'un mur, et pourquoi le reste lui échappe."""

    facade_id: str
    observed_fraction: float
    sampled: int
    distance_m: float | None = None
    rejected_out_of_frame: int = 0
    rejected_back_facing: int = 0
    rejected_occluded: int = 0
    rejected_too_far: int = 0

    def as_dict(self) -> dict:
        return {
            "facade_id": self.facade_id,
            "observed_fraction": round(self.observed_fraction, 3),
            "sampled": self.sampled,
            "distance_m": None if self.distance_m is None else round(self.distance_m, 1),
            "rejected": {
                "out_of_frame": self.rejected_out_of_frame,
                "back_facing": self.rejected_back_facing,
                "occluded": self.rejected_occluded,
                "too_far": self.rejected_too_far,
            },
        }


@dataclass
class FacadeCoverage:
    """Couverture cumulée d'un mur par un corpus."""

    facade_id: str
    best_fraction: float = 0.0
    best_subject: str | None = None
    best_distance_m: float | None = None
    union_fraction: float = 0.0
    weighted_union_fraction: float = 0.0
    appearance_union_fraction: float = 0.0
    geometric_support_fraction: float = 0.0
    contributing: list[str] = field(default_factory=list)
    sampled: int = 0

    @property
    def appearance_coverage(self) -> str:
        """Vocabulaire de `zone_confidence.geojson` pour l'apparence photographique."""
        effective = self.weighted_union_fraction if self.weighted_union_fraction > 0.0 else self.appearance_union_fraction
        if effective <= 0.0:
            return "none"
        if effective >= 0.9:
            return "full"
        return "partial"

    @property
    def geometric_support_coverage(self) -> str:
        """Vocabulaire de `zone_confidence.geojson` pour le support géométrique."""
        effective = self.geometric_support_fraction
        if effective <= 0.0:
            return "none"
        if effective >= 0.9:
            return "full"
        return "partial"

    def as_dict(self) -> dict:
        return {
            "facade_id": self.facade_id,
            "appearance_coverage": self.appearance_coverage,
            "appearance_union_fraction": round(self.appearance_union_fraction, 3),
            "geometric_support_coverage": self.geometric_support_coverage,
            "geometric_support_fraction": round(self.geometric_support_fraction, 3),
            "union_fraction": round(self.union_fraction, 3),
            "weighted_union_fraction": round(self.weighted_union_fraction, 3),
            "best_fraction": round(self.best_fraction, 3),
            "best_subject": self.best_subject,
            "best_distance_m": (
                None if self.best_distance_m is None else round(self.best_distance_m, 1)
            ),
            "contributing_subjects": sorted(self.contributing),
            "sampled_points": self.sampled,
        }


def _segments(geometry):  # noqa: ANN001
    parts = geometry.geoms if geometry.geom_type.startswith("Multi") else [geometry]
    for part in parts:
        coords = list(part.coords)
        for start, end in zip(coords, coords[1:]):
            yield start, end


def sample_facade(geometry, footprint) -> list[FacadeSample]:  # noqa: ANN001
    """Échantillonne un mur et oriente chaque point vers l'extérieur.

    La normale extérieure est celle des deux perpendiculaires qui sort de
    l'empreinte. Sans ce test, un mur serait « vu » depuis l'intérieur du
    bâtiment, donc depuis l'autre côté.
    """
    samples: list[FacadeSample] = []
    for (x0, y0), (x1, y1) in _segments(geometry):
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length == 0.0:
            continue
        normal = None
        for candidate in ((dy / length, -dx / length), (-dy / length, dx / length)):
            mid = Point(
                (x0 + x1) / 2 + candidate[0] * 0.5,
                (y0 + y1) / 2 + candidate[1] * 0.5,
            )
            if not footprint.contains(mid):
                normal = candidate
                break
        if normal is None:
            # Segment intérieur — une cour, un patio. Rien n'y regarde dehors.
            continue
        for index in range(SAMPLES_PER_SEGMENT + 1):
            ratio = index / SAMPLES_PER_SEGMENT
            samples.append(FacadeSample(x0 + dx * ratio, y0 + dy * ratio, normal))
    return samples


def _angular_difference(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def visible_points(  # noqa: PLR0913
    samples: list[FacadeSample],
    origin: tuple[float, float],
    footprint,  # noqa: ANN001
    obstacles: list,
    heading_deg: float | None,
    fov_deg: float | None,
    max_distance_m: float = DEFAULT_MAX_DISTANCE_M,
) -> tuple[list[int], FacadeVisibility]:
    """Indices des points du mur qu'une position voit réellement.

    `heading_deg` absent signifie cadrage inconnu, non cadrage total : la
    condition de cadre est alors ignorée, et le rapport le dit.
    """
    ox, oy = origin
    half_fov = (fov_deg / 2.0) if fov_deg else None
    seen: list[int] = []
    out_of_frame = back_facing = occluded = too_far = 0
    nearest: float | None = None

    for index, sample in enumerate(samples):
        vx, vy = ox - sample.x, oy - sample.y
        distance = math.hypot(vx, vy)
        if distance > max_distance_m:
            too_far += 1
            continue
        # De face : la normale extérieure doit pointer vers l'observateur.
        if sample.normal[0] * vx + sample.normal[1] * vy <= 0.0:
            back_facing += 1
            continue
        if heading_deg is not None and half_fov is not None:
            bearing = math.degrees(math.atan2(sample.x - ox, sample.y - oy)) % 360.0
            if _angular_difference(bearing, heading_deg) > half_fov:
                out_of_frame += 1
                continue
        # Décoller le rayon du mur, sinon il intersecte l'empreinte par
        # construction et tout point serait déclaré masqué.
        ray = LineString([
            (ox, oy),
            (sample.x + sample.normal[0] * 0.05, sample.y + sample.normal[1] * 0.05),
        ])
        if ray.intersection(footprint).length > _GRAZING_M:
            occluded += 1
            continue
        if any(ray.intersection(shape).length > _GRAZING_M for shape in obstacles):
            occluded += 1
            continue
        seen.append(index)
        nearest = distance if nearest is None else min(nearest, distance)

    report = FacadeVisibility(
        facade_id="",
        observed_fraction=(len(seen) / len(samples)) if samples else 0.0,
        sampled=len(samples),
        distance_m=nearest,
        rejected_out_of_frame=out_of_frame,
        rejected_back_facing=back_facing,
        rejected_occluded=occluded,
        rejected_too_far=too_far,
    )
    return seen, report


def coverage_from_subjects(  # noqa: PLR0913
    facade_id: str,
    samples: list[FacadeSample],
    subjects,  # noqa: ANN001 — itérable de (id, origin, heading, fov, view_sector?, distance_m?)
    footprint,  # noqa: ANN001
    obstacles: list,
    max_distance_m: float = DEFAULT_MAX_DISTANCE_M,
) -> FacadeCoverage:
    """Couverture d'un mur par un ensemble de positions.

    `union_fraction` compte un point dès qu'**une** vue le montre : c'est ce
    qui permet à plusieurs cadrages partiels de couvrir un mur qu'aucun ne
    montre entier. `best_fraction` reste la meilleure vue seule, car
    reconstituer par recouvrement suppose un recalage que rien n'a mesuré.

    `weighted_union_fraction` applique un facteur de qualité par sujet :
    - la pertinence sectorielle (une vue FRONT compte plus pour FACADE_PRIMARY)
    - la distance (une vue proche contribue davantage qu'une vue à 150 m)
    """
    coverage = FacadeCoverage(facade_id=facade_id, sampled=len(samples))
    if not samples:
        return coverage

    union: set[int] = set()
    weighted_union: dict[int, float] = {}
    for subject in subjects:
        subject_id = subject[0]
        origin = subject[1]
        heading = subject[2]
        fov = subject[3]
        view_sector = subject[4] if len(subject) > 4 else None
        distance_m = subject[5] if len(subject) > 5 else None
        seen, report = visible_points(
            samples, origin, footprint, obstacles, heading, fov, max_distance_m
        )
        if not seen:
            continue
        union.update(seen)
        coverage.contributing.append(subject_id)
        if report.observed_fraction > coverage.best_fraction:
            coverage.best_fraction = report.observed_fraction
            coverage.best_subject = subject_id
            coverage.best_distance_m = report.distance_m
        weight = _subject_quality_weight(view_sector, facade_id, distance_m)
        for sample_index in seen:
            weighted_union[sample_index] = max(
                weighted_union.get(sample_index, 0.0), weight
            )

    coverage.union_fraction = len(union) / len(samples)
    coverage.appearance_union_fraction = coverage.union_fraction
    coverage.weighted_union_fraction = (
        sum(weighted_union.values()) / len(samples) if samples else 0.0
    )
    # geometric_support_fraction reste 0.0 par défaut ; il sera rempli
    # par satellite_completion ou d'autres sources géométriques.
    return coverage


def _subject_quality_weight(  # noqa: ANN001
    view_sector, facade_id: str, distance_m: float | None,
) -> float:
    """Pondération d'un sujet selon son secteur et sa distance.

    Une vue frontale sur la façade observée vaut 1.0. Une vue latérale ou
    éloignée vaut moins : elle contribue au recouvrement, mais avec une
    confiance moindre.
    """
    sector_relevance = 1.0
    if view_sector is not None:
        canonical = {
            "FACADE_PRIMARY": {"front": 1.0, "front_left_corner": 0.85, "front_right_corner": 0.85,
                               "left": 0.4, "right": 0.4, "rear": 0.2, "rear_left_corner": 0.2, "rear_right_corner": 0.2},
            "FACADE_LEFT": {"left": 1.0, "front_left_corner": 0.85, "rear_left_corner": 0.85,
                            "front": 0.4, "rear": 0.4, "right": 0.2, "front_right_corner": 0.2, "rear_right_corner": 0.2},
            "FACADE_RIGHT": {"right": 1.0, "front_right_corner": 0.85, "rear_right_corner": 0.85,
                             "front": 0.4, "rear": 0.4, "left": 0.2, "front_left_corner": 0.2, "rear_left_corner": 0.2},
            "FACADE_REAR": {"rear": 1.0, "rear_left_corner": 0.85, "rear_right_corner": 0.85,
                            "left": 0.4, "right": 0.4, "front": 0.2, "front_left_corner": 0.2, "front_right_corner": 0.2},
        }.get(facade_id, {})
        sector_relevance = sector_relevance.get(view_sector, 0.5)

    distance_decay = 1.0
    if distance_m is not None and distance_m > 0:
        distance_decay = max(0.5, 1.0 - (distance_m / DEFAULT_MAX_DISTANCE_M) * 0.5)

    return round(sector_relevance * distance_decay, 3)


__all__ = [
    "DEFAULT_MAX_DISTANCE_M",
    "SAMPLES_PER_SEGMENT",
    "FacadeCoverage",
    "FacadeSample",
    "FacadeVisibility",
    "coverage_from_subjects",
    "sample_facade",
    "visible_points",
]
