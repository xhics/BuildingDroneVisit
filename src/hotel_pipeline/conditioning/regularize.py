"""Régularisation des emprises : redresser ce qu'un bâtiment a de droit.

Une emprise cartographique est saisie à la main : ses angles ne sont jamais
tout à fait droits, ses murs jamais tout à fait parallèles. Rendue telle
quelle, elle donne des façades légèrement de guingois — un défaut discret mais
que l'œil relève sur un plan large, parce qu'un bâtiment réel, lui, est
d'équerre.

La méthode suit ce que la reconstruction LOD2 appelle la régularisation
(SimpliCity et apparentés) : trouver l'orientation dominante du bâti, puis y
aligner chaque arête qui s'en approche. Une arête franchement oblique — un pan
coupé, une aile en biais — est laissée telle quelle : la régularisation
redresse ce qui devait l'être, elle ne rectifie pas la forme.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..logging import get_logger

log = get_logger("conditioning-regularize")

#: Écart à l'orientation dominante, en degrés, en deçà duquel une arête est
#: tenue pour devant être alignée. Au-delà, elle est délibérément oblique.
ALIGN_TOLERANCE_DEG = 12.0

#: Longueur minimale d'une arête pour peser dans l'orientation dominante.
#: Les micro-segments d'une saisie manuelle ne disent rien de l'équerrage.
MIN_EDGE_WEIGHT_M = 3.0

#: Déplacement maximal d'un sommet, en mètres. Au-delà, la régularisation ne
#: redresse plus : elle déforme.
MAX_SHIFT_M = 1.2


@dataclass
class Regularization:
    """Ce que le redressement a changé sur une emprise."""

    feature_id: str
    dominant_deg: float
    edges_aligned: int
    edges_total: int
    max_shift_m: float

    def as_dict(self) -> dict:
        return {
            "feature_id": self.feature_id,
            "dominant_deg": round(self.dominant_deg, 2),
            "edges_aligned": self.edges_aligned,
            "edges_total": self.edges_total,
            "max_shift_m": round(self.max_shift_m, 3),
        }


def dominant_orientation(footprint: np.ndarray) -> float:
    """Orientation principale d'une emprise, en degrés dans [0, 90).

    Chaque arête vote pour son orientation, pondérée par sa longueur : un mur
    de quatre-vingts mètres dit l'équerrage du bâtiment, un raccord de deux
    mètres non. Les orientations sont ramenées modulo quatre-vingt-dix, deux
    murs perpendiculaires décrivant le même équerrage.
    """
    count = len(footprint)
    if count < 3:
        return 0.0

    # Somme vectorielle sur l'angle quadruplé : c'est ce qui fait coïncider
    # une arête et sa perpendiculaire dans le même vote.
    accumulator = np.zeros(2)
    for index in range(count):
        start = footprint[index]
        end = footprint[(index + 1) % count]
        delta = end - start
        length = float(np.hypot(*delta))
        if length < MIN_EDGE_WEIGHT_M:
            continue
        angle = math.atan2(delta[1], delta[0])
        accumulator += length * np.array([math.cos(4 * angle), math.sin(4 * angle)])

    if np.linalg.norm(accumulator) < 1e-9:
        return 0.0
    degrees = math.degrees(math.atan2(accumulator[1], accumulator[0]) / 4.0) % 90.0
    # Un contour d'équerre donne un angle quadruplé d'un cheveu négatif, que le
    # modulo renvoie à 90° — la même orientation, mais illisible dans un
    # rapport. On la ramène à zéro, dont elle n'a jamais bougé.
    return 0.0 if degrees > 90.0 - 1e-6 else float(degrees)


def regularize(
    footprint: np.ndarray,
    feature_id: str = "unknown",
    tolerance_deg: float = ALIGN_TOLERANCE_DEG,
) -> tuple[np.ndarray, Regularization]:
    """Aligne les arêtes proches de l'orientation dominante.

    Le redressement procède par projection : chaque arête presque alignée est
    ramenée sur la direction voulue, en gardant son milieu. Les sommets
    partagés reçoivent la moyenne des positions que leurs deux arêtes leur
    assignent, ce qui referme le contour sans le déchirer.
    """
    count = len(footprint)
    if count < 4:
        return footprint.copy(), Regularization(feature_id, 0.0, 0, 0, 0.0)

    dominant = dominant_orientation(footprint)
    axes = [
        np.array([math.cos(math.radians(dominant + k * 90.0)),
                  math.sin(math.radians(dominant + k * 90.0))])
        for k in range(2)
    ]

    proposals: list[list[np.ndarray]] = [[] for _ in range(count)]
    aligned = 0

    for index in range(count):
        start = footprint[index]
        end = footprint[(index + 1) % count]
        delta = end - start
        length = float(np.hypot(*delta))
        if length < 1e-6:
            continue

        direction = delta / length
        best_axis, best_cos = None, 0.0
        for axis in axes:
            cosine = abs(float(np.dot(direction, axis)))
            if cosine > best_cos:
                best_axis, best_cos = axis, cosine

        angle_gap = math.degrees(math.acos(min(best_cos, 1.0)))
        if best_axis is None or angle_gap > tolerance_deg:
            continue

        # L'arête pivote autour de son milieu, sur l'axe le plus proche.
        middle = (start + end) * 0.5
        signed = best_axis if float(np.dot(direction, best_axis)) >= 0 else -best_axis
        proposals[index].append(middle - signed * (length * 0.5))
        proposals[(index + 1) % count].append(middle + signed * (length * 0.5))
        aligned += 1

    adjusted = footprint.copy().astype(np.float64)
    shift = 0.0
    for index, candidates in enumerate(proposals):
        if not candidates:
            continue
        target = np.mean(candidates, axis=0)
        move = float(np.hypot(*(target - footprint[index])))
        # Un sommet qu'il faudrait déplacer d'un mètre et demi n'était pas
        # « presque aligné » : le laisser où il est vaut mieux que déformer.
        if move > MAX_SHIFT_M:
            continue
        adjusted[index] = target
        shift = max(shift, move)

    report = Regularization(
        feature_id=feature_id,
        dominant_deg=dominant,
        edges_aligned=aligned,
        edges_total=count,
        max_shift_m=shift,
    )
    log.info(
        "%s : orientation %.1f°, %d/%d arête(s) alignée(s), écart max %.2f m",
        feature_id,
        dominant,
        aligned,
        count,
        shift,
    )
    return adjusted, report


def apply_to_scene(scene, tolerance_deg: float = ALIGN_TOLERANCE_DEG) -> dict:  # noqa: ANN001
    """Redresse les emprises de tous les volumes d'une scène."""
    reports = []
    for prism in scene.prisms:
        adjusted, report = regularize(
            prism.footprint, prism.feature_id, tolerance_deg
        )
        if report.edges_aligned:
            prism.footprint = adjusted
            reports.append(report)

    return {
        "regularized": len(reports),
        "total": len(scene.prisms),
        "max_shift_m": round(max((r.max_shift_m for r in reports), default=0.0), 3),
        "detail": [r.as_dict() for r in reports[:6]],
    }
