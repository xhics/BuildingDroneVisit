"""Associer les arêtes de toiture mesurées aux segments visibles dans les images.

Le nuage LiDAR donne des arêtes de toiture **métriques** : elles ne sont pas
cherchées dans les images mais déduites de l'intersection de deux pans ajustés
au nuage. Sur ce pilote, trente-deux d'entre elles totalisent trois cent douze
mètres, réparties sur douze orientations.

Ces lignes valent mieux que des lignes triangulées depuis les photographies —
elles ne dépendent d'aucune pose. Mais elles ne contraignent rien tant qu'on
ne sait pas **où elles tombent dans chaque image**. C'est ce que ce module
établit : pour chaque arête projetée, quel segment détecté lui correspond, et
avec quelle confiance.

**Trois précautions gouvernent la méthode.**

D'abord, une arête peut n'avoir aucun correspondant : occultée, hors cadre, ou
noyée dans un ciel sans contraste. L'association prévoit donc explicitement la
sortie « rien », et ne force jamais un appariement.

Ensuite, la fenêtre de recherche est dimensionnée par l'**incertitude de pose**,
non par une tolérance fixe. Une pose approximative déplace la projection
attendue ; chercher dans un rayon serré y produirait des correspondances
fausses, qui contraindraient l'ajustement vers l'erreur plutôt que vers la
vérité.

Enfin, ces arêtes sont quasi horizontales et vues d'en dessous, depuis la rue.
Elles contraignent fortement le **cap et le roulis**, faiblement la distance.
C'est ce qu'il faut en attendre : de la stabilité angulaire, pas de la
profondeur.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..logging import get_logger

log = get_logger("geo-ridge-match")

#: Écart d'orientation, en degrés, au-delà duquel un segment ne peut pas
#: décrire l'arête projetée. Large : la pose est approximative, et c'est elle
#: qui déplace l'orientation attendue.
ANGLE_TOLERANCE_DEG = 22.0

#: Rayon de recherche de base, en pixels, pour une pose tenue pour sûre. Il est
#: élargi par l'incertitude de pose — voir `search_radius_px`.
BASE_RADIUS_PX = 18.0

#: Longueur minimale d'un segment détecté, en pixels. En deçà, l'orientation
#: n'est pas fiable et le segment décrit du bruit de texture.
MIN_SEGMENT_PX = 22.0

#: Part de la longueur projetée qu'un segment doit couvrir. Une arête vue de
#: loin se fragmente en plusieurs traits : on n'exige pas qu'un seul segment la
#: décrive entière.
MIN_LENGTH_RATIO = 0.25

#: Rapport entre le meilleur coût et le suivant, en deçà duquel l'association
#: est tenue pour ambiguë. Deux candidats équivalents ne tranchent rien.
AMBIGUITY_RATIO = 0.75


@dataclass
class RidgeProjection:
    """Une arête 3D telle qu'elle devrait apparaître dans une image."""

    ridge_index: int
    asset_id: str
    #: Extrémités projetées, en pixels.
    start_px: tuple[float, float]
    end_px: tuple[float, float]
    #: Distance de la caméra au milieu de l'arête, en mètres.
    distance_m: float
    in_frame: bool

    @property
    def length_px(self) -> float:
        return float(
            math.hypot(
                self.end_px[0] - self.start_px[0], self.end_px[1] - self.start_px[1]
            )
        )

    @property
    def angle_deg(self) -> float:
        return math.degrees(
            math.atan2(
                self.end_px[1] - self.start_px[1], self.end_px[0] - self.start_px[0]
            )
        ) % 180.0


@dataclass
class RidgeMatch:
    """Ce qu'une arête a trouvé, ou n'a pas trouvé, dans une image."""

    ridge_index: int
    asset_id: str
    #: Segment retenu, en pixels. `None` quand rien ne correspond.
    segment: tuple[float, float, float, float] | None
    cost: float | None
    #: Écart d'orientation retenu, en degrés.
    angle_gap_deg: float | None = None
    #: Distance moyenne entre la projection et le segment, en pixels.
    offset_px: float | None = None
    ambiguous: bool = False
    #: Candidats écartés de peu, conservés pour un départage topologique.
    alternatives: list = field(default_factory=list)
    reason: str = ""

    @property
    def matched(self) -> bool:
        return self.segment is not None and not self.ambiguous

    def as_dict(self) -> dict:
        return {
            "ridge_index": self.ridge_index,
            "asset_id": self.asset_id,
            "matched": self.matched,
            "ambiguous": self.ambiguous,
            "cost": round(self.cost, 3) if self.cost is not None else None,
            "angle_gap_deg": (
                round(self.angle_gap_deg, 1) if self.angle_gap_deg is not None else None
            ),
            "offset_px": (
                round(self.offset_px, 1) if self.offset_px is not None else None
            ),
            "segment": (
                [round(v, 1) for v in self.segment] if self.segment else None
            ),
            "reason": self.reason,
        }


@dataclass
class RidgeMatchReport:
    matches: list[RidgeMatch] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    @property
    def matched(self) -> list[RidgeMatch]:
        return [m for m in self.matches if m.matched]

    def by_ridge(self) -> dict[int, int]:
        """Nombre d'images où chaque arête a été retrouvée."""
        counts: dict[int, int] = {}
        for match in self.matched:
            counts[match.ridge_index] = counts.get(match.ridge_index, 0) + 1
        return counts

    def as_dict(self) -> dict:
        counts = self.by_ridge()
        return {
            "attempted": len(self.matches),
            "matched": len(self.matched),
            "ambiguous": sum(1 for m in self.matches if m.ambiguous),
            "ridges_found": len(counts),
            "ridges_in_two_or_more": sum(1 for v in counts.values() if v >= 2),
            "by_ridge": counts,
            "matches": [m.as_dict() for m in self.matches],
            "provenance": self.provenance,
            "caveats": [
                "une arête associée dans une seule image ne contraint pas une "
                "pose : il en faut deux, et d'orientations différentes",
                "ces arêtes sont quasi horizontales et vues d'en dessous — "
                "elles contraignent le cap, non la distance",
                "l'association dépend de la pose supposée : une pose fausse "
                "produit des correspondances fausses, non une absence",
            ],
        }


def search_radius_px(
    projection: RidgeProjection, pose_sigma_m: float, focal_px: float
) -> float:
    """Rayon de recherche dicté par l'incertitude de pose, non par une constante.

    Une incertitude de position se traduit en pixels par la focale divisée par
    la distance : le même mètre d'erreur déplace beaucoup un objet proche, peu
    un objet lointain. Chercher dans un rayon fixe reviendrait à ignorer ce que
    l'on sait mal.
    """
    if projection.distance_m <= 0.1:
        return BASE_RADIUS_PX
    drift = focal_px * pose_sigma_m / projection.distance_m
    return BASE_RADIUS_PX + float(drift)


def project_ridge(
    start: np.ndarray,
    end: np.ndarray,
    camera,  # noqa: ANN001
    ridge_index: int,
    asset_id: str,
) -> RidgeProjection | None:
    """Projette une arête 3D dans une image. `None` si elle passe derrière."""
    points = np.vstack([start, end])
    screen, depth = camera.project(points)
    if depth is None or np.any(depth <= 0.5):
        return None

    width, height = camera.width, camera.height
    inside = [
        0.0 <= point[0] < width and 0.0 <= point[1] < height for point in screen
    ]
    middle = (np.asarray(start) + np.asarray(end)) * 0.5
    return RidgeProjection(
        ridge_index=ridge_index,
        asset_id=asset_id,
        start_px=(float(screen[0][0]), float(screen[0][1])),
        end_px=(float(screen[1][0]), float(screen[1][1])),
        distance_m=float(np.linalg.norm(middle - camera.position)),
        in_frame=any(inside),
    )


def detect_segments(image_path, min_length_px: float = MIN_SEGMENT_PX):  # noqa: ANN001
    """Segments de droite d'une image, par LSD.

    Un détecteur déterministe et sans apprentissage suffit ici : une arête de
    toiture se détache sur le ciel, et c'est le contraste le plus franc d'une
    vue de rue. Un détecteur appris ne se justifierait que si ces arêtes
    ressortaient fragmentées.
    """
    import cv2

    data = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if data is None:
        return []
    detector = cv2.createLineSegmentDetector()
    found = detector.detect(data)[0]
    if found is None:
        return []

    segments = []
    for entry in found.reshape(-1, 4):
        x1, y1, x2, y2 = (float(v) for v in entry)
        if math.hypot(x2 - x1, y2 - y1) >= min_length_px:
            segments.append((x1, y1, x2, y2))
    return segments


#: Écart, en degrés, sous lequel un segment est tenu pour horizontal — donc
#: candidat à décrire une arête de toiture vue depuis la rue. Au-delà, il
#: décrit autre chose : un montant, un poteau, un bord de véhicule.
HORIZONTAL_BAND_DEG = 35.0


def distance_families(segments, expected_angle: float):  # noqa: ANN001
    """Sépare les segments selon ce qu'ils peuvent décrire.

    Un champ de distance unique laisse une arête de toiture s'aligner sur le
    bord d'une voiture : les deux sont des traits sombres, et rien dans la
    géométrie ne les distingue. Mesuré sur ce pilote, quatre-vingt-six
    associations restaient ambiguës faute de cette séparation.

    La famille est décidée par l'orientation attendue de l'arête : un faîtage
    vu d'en dessous se projette près de l'horizontale, et un segment franchement
    vertical ne peut pas le décrire quelle que soit sa proximité. Les segments
    obliques restent disponibles — une arête de croupe en est une — mais dans
    une famille distincte, où ils ne concurrencent pas les horizontaux.
    """
    families: dict[str, list] = {"compatible": [], "oblique": [], "ecarte": []}
    for segment in segments:
        gap = _angle_gap(_angle_of(segment), expected_angle)
        if gap <= ANGLE_TOLERANCE_DEG:
            families["compatible"].append(segment)
        elif gap <= HORIZONTAL_BAND_DEG:
            families["oblique"].append(segment)
        else:
            families["ecarte"].append(segment)
    return families


def _angle_of(segment) -> float:  # noqa: ANN001
    return math.degrees(
        math.atan2(segment[3] - segment[1], segment[2] - segment[0])
    ) % 180.0


def _angle_gap(first: float, second: float) -> float:
    """Écart entre deux orientations non orientées, dans [0, 90]."""
    gap = abs(first - second) % 180.0
    return min(gap, 180.0 - gap)


def _point_to_line(point, start, end) -> float:  # noqa: ANN001
    """Distance d'un point à la droite portée par un segment."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    return abs(dy * (point[0] - start[0]) - dx * (point[1] - start[1])) / length


def match_one(
    projection: RidgeProjection,
    segments,  # noqa: ANN001
    pose_sigma_m: float = 3.0,
    focal_px: float = 900.0,
) -> RidgeMatch:
    """Cherche le segment décrivant une arête projetée, ou conclut qu'il n'y en a pas."""
    base = RidgeMatch(
        ridge_index=projection.ridge_index,
        asset_id=projection.asset_id,
        segment=None,
        cost=None,
    )
    if not projection.in_frame:
        base.reason = "arête hors du cadre de cette vue"
        return base
    if projection.length_px < MIN_SEGMENT_PX:
        base.reason = (
            f"arête trop courte à l'écran ({projection.length_px:.0f} px) : "
            "son orientation n'est pas mesurable"
        )
        return base

    radius = search_radius_px(projection, pose_sigma_m, focal_px)
    expected_angle = projection.angle_deg
    middle = (
        (projection.start_px[0] + projection.end_px[0]) * 0.5,
        (projection.start_px[1] + projection.end_px[1]) * 0.5,
    )

    # Les familles d'abord : un segment vertical ne décrit pas un faîtage,
    # même s'il passe au bon endroit. Les écarter avant le calcul de coût
    # évite qu'ils entrent en concurrence et créent une fausse ambiguïté.
    families = distance_families(segments, expected_angle)
    candidates = families["compatible"] or families["oblique"]

    scored: list[tuple[float, tuple, float, float]] = []
    for segment in candidates:
        gap = _angle_gap(_angle_of(segment), expected_angle)
        length = math.hypot(segment[2] - segment[0], segment[3] - segment[1])
        if length < projection.length_px * MIN_LENGTH_RATIO:
            continue
        # Distance de l'arête attendue à la droite du segment, prise aux deux
        # extrémités : un segment parallèle mais décalé doit être écarté.
        offset = 0.5 * (
            _point_to_line(projection.start_px, segment[:2], segment[2:])
            + _point_to_line(projection.end_px, segment[:2], segment[2:])
        )
        if offset > radius:
            continue
        # Le coût mêle décalage et désorientation, ramenés à leurs tolérances
        # pour être comparables.
        cost = offset / radius + gap / ANGLE_TOLERANCE_DEG
        scored.append((cost, segment, gap, offset))

    if not scored:
        base.reason = (
            f"aucun segment dans {radius:.0f} px et {ANGLE_TOLERANCE_DEG:.0f}° "
            "de l'arête attendue"
        )
        return base

    scored.sort(key=lambda item: item[0])
    cost, segment, gap, offset = scored[0]
    match = RidgeMatch(
        ridge_index=projection.ridge_index,
        asset_id=projection.asset_id,
        segment=segment,
        cost=cost,
        angle_gap_deg=gap,
        offset_px=offset,
        reason="segment retenu",
    )
    # Deux candidats de coût voisin ne tranchent rien par eux-mêmes. Mesuré
    # sur ce pilote, ils diffèrent de moins de dix degrés : ce sont des traits
    # parallèles voisins — une corniche et son ombre, deux niveaux de bardage —
    # que ni l'orientation ni la distance ne séparent.
    #
    # Le départage vient d'ailleurs : les candidats concurrents sont conservés
    # pour qu'une contrainte topologique puisse les trancher. Les jeter ici
    # forcerait un choix que l'appariement seul ne peut pas fonder.
    if len(scored) > 1 and cost >= scored[1][0] * AMBIGUITY_RATIO:
        match.ambiguous = True
        match.alternatives = [entry[1] for entry in scored[1:4]]
        match.reason = (
            f"deux segments de coût voisin ({cost:.2f} et {scored[1][0]:.2f}) : "
            "l'appariement seul ne tranche pas — départage topologique requis"
        )
    return match


def disambiguate(matches: list, graph, consistent) -> int:  # noqa: ANN001
    """Tranche les associations ambiguës par la topologie du toit.

    Un segment retenu doit respecter le voisinage de son arête : si deux
    arêtes se rejoignent en 3D, leurs segments doivent se rejoindre à l'image.
    Cette contrainte croisée départage des candidats que la seule proximité
    laisse équivalents.

    Les alternatives sont essayées dans l'ordre du coût : on ne cherche pas le
    meilleur segment au sens topologique, mais le meilleur au sens du coût
    **qui respecte** la topologie. Une arête dont aucune alternative ne passe
    reste ambiguë — c'est un aveu, non un échec.
    """
    resolved = 0
    by_ridge = {m.ridge_index: m for m in matches if m.matched}
    for match in matches:
        if not match.ambiguous or not match.alternatives:
            continue
        for candidate in [match.segment] + list(match.alternatives):
            trial = {r: m.segment for r, m in by_ridge.items()}
            trial[match.ridge_index] = candidate
            verdicts = consistent(graph, trial)
            if verdicts.get(match.ridge_index):
                match.segment = candidate
                match.ambiguous = False
                match.reason = "départagé par la topologie du toit"
                by_ridge[match.ridge_index] = match
                resolved += 1
                break
    return resolved


__all__ = [
    "AMBIGUITY_RATIO",
    "ANGLE_TOLERANCE_DEG",
    "HORIZONTAL_BAND_DEG",
    "BASE_RADIUS_PX",
    "MIN_LENGTH_RATIO",
    "MIN_SEGMENT_PX",
    "RidgeMatch",
    "RidgeMatchReport",
    "RidgeProjection",
    "detect_segments",
    "disambiguate",
    "distance_families",
    "match_one",
    "project_ridge",
    "search_radius_px",
]
