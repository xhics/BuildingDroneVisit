"""Poser une caméra contre le LiDAR, sans passer par une autre image.

La reconstruction multivue enchaîne les images : A s'accroche à B, B à C. Quand
la chaîne se rompt — recouvrement insuffisant, façades trop semblables — les
morceaux restent dans des référentiels séparés et rien ne les rapproche.

Ce module renverse la dépendance. Chaque image est confrontée directement au
nuage LiDAR, qui est le même pour toutes : deux vues sans aucune correspondance
mutuelle se retrouvent alors dans un référentiel commun.

**Le principe.** Depuis la position déclarée, on essaie des poses, on projette
les arêtes de toiture mesurées, et l'on mesure combien elles tombent sur des
contours réels de l'image. La comparaison ne passe par aucune association
explicite : c'est un champ de distance qui note chaque pose, ce qui tolère les
segments fragmentés, les extrémités occultées et les correspondances inconnues.

**Ce que la méthode ne fait pas.** Elle ne crée ni texture, ni parallaxe, ni
détail de façade absent du LiDAR. Elle place une caméra — et sur ce pilote,
elle est moins nécessaire depuis que le graphe de toiture porte quatre familles
d'orientation. Sur un bâtiment mitoyen ou une tour, dont les arêtes seraient
plus courtes et moins variées, elle redeviendrait la voie principale.

**Ce qu'elle exige.** Des arêtes visibles dans l'image. Un toit plat, un
bâtiment vu de trop loin ou masqué par la végétation ne donnent rien à aligner,
et le module le dit plutôt que de rendre une pose que rien ne soutient.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..logging import get_logger

log = get_logger("geo-lidar-localize")

#: Rayon de la grille de positions essayées, en mètres autour de la position
#: déclarée. Il couvre l'imprécision d'un relevé de rue sans explorer la ville.
SEARCH_RADIUS_M = 12.0

#: Pas de la grille de positions, en mètres.
POSITION_STEP_M = 4.0

#: Amplitude de la recherche en cap, en degrés de part et d'autre du cap connu.
HEADING_SPAN_DEG = 20.0

#: Pas de la recherche en cap, en degrés.
HEADING_STEP_DEG = 4.0

#: Distance, en pixels, au-delà de laquelle un contour ne compte plus comme
#: soutien. Elle borne le champ de distance : sans plafond, une arête loin de
#: tout contour tirerait la pose vers n'importe où.
SUPPORT_RADIUS_PX = 25.0

#: Part des arêtes projetées qui doit trouver un appui pour qu'une pose soit
#: retenue. En deçà, l'alignement décrit du hasard.
MIN_SUPPORTED_FRACTION = 0.35

#: Écart relatif entre le meilleur score et le suivant, en deçà duquel la pose
#: est tenue pour indécise. Deux poses équivalentes ne localisent rien.
#:
#: Mesuré sur ce pilote, le module distingue franchement une pose juste d'une
#: pose fausse — 0,787 contre 0,000 à cinquante mètres ou soixante degrés
#: d'écart. Mais dans le voisinage immédiat de la bonne pose, plusieurs
#: candidates obtiennent des scores presque égaux : l'ambiguïté est locale et
#: réelle, un pas de quatre mètres et de quatre degrés ne la lève pas.
#:
#: Le seuil provoque donc un refus systématique ici. C'est le comportement
#: voulu — mieux vaut ne pas localiser que livrer une pose parmi plusieurs
#: également plausibles — mais il appelle un raffinement continu autour du
#: meilleur candidat plutôt qu'une grille plus fine, qui multiplierait les
#: ex æquo sans les départager.
DECISIVE_MARGIN = 0.08


@dataclass
class PoseCandidate:
    """Une pose essayée, et ce que les contours en disent."""

    position: np.ndarray
    heading_deg: float
    #: Score moyen d'appui, dans [0, 1] : 1 quand chaque arête tombe sur un
    #: contour, 0 quand aucune n'en approche.
    score: float = 0.0
    supported: int = 0
    projected: int = 0

    @property
    def supported_fraction(self) -> float:
        return self.supported / max(self.projected, 1)

    def as_dict(self) -> dict:
        return {
            "position": [round(float(v), 2) for v in self.position],
            "heading_deg": round(self.heading_deg, 1),
            "score": round(self.score, 4),
            "supported": self.supported,
            "projected": self.projected,
            "supported_fraction": round(self.supported_fraction, 3),
        }


@dataclass
class LocalizationResult:
    """La pose retenue, ou la raison pour laquelle aucune ne l'est."""

    asset_id: str
    best: PoseCandidate | None = None
    runner_up: PoseCandidate | None = None
    decisive: bool = False
    reason: str = ""
    #: Arêtes non utilisées pour poser, servant à valider le résultat.
    holdout_residual_px: float | None = None
    provenance: dict = field(default_factory=dict)

    @property
    def localized(self) -> bool:
        return self.best is not None and self.decisive

    def as_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "localized": self.localized,
            "decisive": self.decisive,
            "best": self.best.as_dict() if self.best else None,
            "runner_up": self.runner_up.as_dict() if self.runner_up else None,
            "holdout_residual_px": (
                round(self.holdout_residual_px, 1)
                if self.holdout_residual_px is not None
                else None
            ),
            "reason": self.reason,
            "provenance": self.provenance,
            "caveats": [
                "la pose vient de l'alignement d'arêtes mesurées sur des "
                "contours : elle ne crée ni texture ni parallaxe",
                "un toit plat ou trop lointain ne donne rien à aligner — "
                "l'absence de pose y est un constat, non un échec",
                "le résidu de contrôle porte sur des arêtes écartées du "
                "calcul : c'est lui qui dit si la pose tient",
            ],
        }


def edge_distance_field(image, radius_px: float = SUPPORT_RADIUS_PX):  # noqa: ANN001
    """Champ de distance aux contours de l'image, plafonné.

    Le plafond compte autant que le champ : une arête projetée loin de tout
    contour ne doit pas peser plus qu'une autre également perdue, sans quoi
    l'optimisation la poursuivrait au détriment des arêtes bien placées.
    """
    import cv2

    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grey, 60, 160)
    # `distanceTransform` mesure la distance au zéro le plus proche : les
    # contours doivent donc valoir zéro, et le reste non nul.
    field_px = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
    return np.minimum(field_px, radius_px)


def _sample_field(field_px, x: float, y: float, radius_px: float) -> float:
    """Distance au contour le plus proche, hors cadre compris."""
    height, width = field_px.shape
    column, row = int(round(x)), int(round(y))
    if not (0 <= column < width and 0 <= row < height):
        return radius_px
    return float(field_px[row, column])


def score_pose(
    ridges,  # noqa: ANN001
    camera,  # noqa: ANN001
    field_px,
    samples_per_edge: int = 12,
    radius_px: float = SUPPORT_RADIUS_PX,
) -> tuple[float, int, int]:
    """Note une pose : combien d'arêtes projetées tombent sur des contours.

    L'arête est échantillonnée sur sa longueur plutôt que réduite à ses
    extrémités : une extrémité occultée ne doit pas disqualifier une arête dont
    le corps s'aligne parfaitement.
    """
    total, supported, projected = 0.0, 0, 0
    for ridge in ridges:
        ratios = np.linspace(0.0, 1.0, samples_per_edge)
        points = np.array(
            [np.asarray(ridge.start) + (np.asarray(ridge.end) - np.asarray(ridge.start)) * t
             for t in ratios]
        )
        screen, depth = camera.project(points)
        if screen is None:
            continue
        visible = depth > 0.5 if depth is not None else np.ones(len(points), dtype=bool)
        if not np.any(visible):
            continue

        projected += 1
        distances = [
            _sample_field(field_px, screen[k][0], screen[k][1], radius_px)
            for k in range(len(points))
            if visible[k]
        ]
        if not distances:
            continue
        # Le score d'une arête : 1 quand elle épouse un contour, 0 quand elle
        # en est à la distance plafond.
        edge_score = 1.0 - float(np.mean(distances)) / radius_px
        total += edge_score
        if edge_score >= 0.5:
            supported += 1

    return (total / max(projected, 1), supported, projected)


def localize(
    ridges,  # noqa: ANN001
    image,
    make_camera,  # noqa: ANN001
    origin: np.ndarray,
    heading_deg: float,
    asset_id: str = "asset",
    holdout: int = 0,
) -> LocalizationResult:
    """Cherche la pose dont les arêtes projetées épousent le mieux les contours.

    `make_camera` reçoit une position et un cap, et rend une caméra projetant
    des points 3D. L'injecter garde ce module indépendant du modèle optique.

    `holdout` réserve des arêtes hors du calcul : elles servent à mesurer si la
    pose tient sur ce qui n'a pas servi à la produire. C'est la seule validation
    honnête d'un alignement — sans elle, un bon score ne prouve que sa propre
    optimisation.
    """
    result = LocalizationResult(asset_id=asset_id)
    usable = list(ridges)
    if not usable:
        result.reason = "aucune arête mesurée : rien à aligner"
        return result

    fitting = usable[holdout:] if holdout else usable
    control = usable[:holdout] if holdout else []
    if not fitting:
        result.reason = "toutes les arêtes réservées au contrôle"
        return result

    field_px = edge_distance_field(image)

    offsets = np.arange(-SEARCH_RADIUS_M, SEARCH_RADIUS_M + 1e-6, POSITION_STEP_M)
    headings = np.arange(
        heading_deg - HEADING_SPAN_DEG,
        heading_deg + HEADING_SPAN_DEG + 1e-6,
        HEADING_STEP_DEG,
    )

    candidates: list[PoseCandidate] = []
    for dx in offsets:
        for dy in offsets:
            position = np.asarray(origin, dtype=np.float64) + np.array([dx, dy, 0.0])
            for candidate_heading in headings:
                camera = make_camera(position, float(candidate_heading))
                score, supported, projected = score_pose(fitting, camera, field_px)
                if projected == 0:
                    continue
                candidates.append(
                    PoseCandidate(
                        position=position,
                        heading_deg=float(candidate_heading),
                        score=score,
                        supported=supported,
                        projected=projected,
                    )
                )

    if not candidates:
        result.reason = (
            "aucune arête ne se projette dans l'image : le bâtiment est hors "
            "cadre ou derrière la caméra"
        )
        return result

    candidates.sort(key=lambda c: -c.score)
    result.best = candidates[0]
    result.runner_up = candidates[1] if len(candidates) > 1 else None
    result.provenance = {
        "poses_tried": len(candidates),
        "ridges_fitting": len(fitting),
        "ridges_holdout": len(control),
        "search_radius_m": SEARCH_RADIUS_M,
        "heading_span_deg": HEADING_SPAN_DEG,
        "support_radius_px": SUPPORT_RADIUS_PX,
    }

    if result.best.supported_fraction < MIN_SUPPORTED_FRACTION:
        result.reason = (
            f"seules {result.best.supported_fraction:.0%} des arêtes trouvent un "
            f"appui (seuil {MIN_SUPPORTED_FRACTION:.0%}) : l'alignement décrit "
            "du hasard"
        )
        return result

    if result.runner_up is not None:
        margin = result.best.score - result.runner_up.score
        if margin < DECISIVE_MARGIN * max(result.best.score, 1e-6):
            result.reason = (
                f"deux poses de score voisin ({result.best.score:.3f} et "
                f"{result.runner_up.score:.3f}) : la localisation n'est pas "
                "tranchée"
            )
            return result

    result.decisive = True
    result.reason = "pose retenue : les arêtes projetées épousent les contours"

    # Contrôle : les arêtes réservées se projettent-elles là où il faut ?
    if control:
        camera = make_camera(result.best.position, result.best.heading_deg)
        score, _supported, projected = score_pose(control, camera, field_px)
        if projected:
            result.holdout_residual_px = (1.0 - score) * SUPPORT_RADIUS_PX

    log.info(
        "%s : %s (%d pose(s) essayée(s), appui %.0f%%)",
        asset_id,
        "localisée" if result.localized else "non localisée",
        len(candidates),
        100 * result.best.supported_fraction,
    )
    return result


__all__ = [
    "DECISIVE_MARGIN",
    "HEADING_SPAN_DEG",
    "HEADING_STEP_DEG",
    "MIN_SUPPORTED_FRACTION",
    "POSITION_STEP_M",
    "SEARCH_RADIUS_M",
    "SUPPORT_RADIUS_PX",
    "LocalizationResult",
    "PoseCandidate",
    "edge_distance_field",
    "localize",
    "score_pose",
]
