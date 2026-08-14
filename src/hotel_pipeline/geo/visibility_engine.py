"""Moteur de visibilité multi-rayons (Lot 1B V2, étape 3).

L'ancien contrôle tirait **un** rayon vers le point le plus proche de
l'empreinte : une tour posée devant ce point condamnait la vue entière, et un
hangar masquant les trois quarts de la façade passait inaperçu dès lors qu'il
laissait ce point-là dégagé. La silhouette est ici échantillonnée, et chaque
cellule jugée pour elle-même.

Trois principes s'y appliquent.

La **pondération angulaire** : une cellule vaut sa largeur, non une unité. Un
échantillonnage plus fin ne doit pas peser davantage.

La **profondeur** : un obstacle situé derrière la cible ne masque rien. On
cherche donc d'abord la première intersection avec la cible, puis on ne
confronte que ce qui la précède.

La **preuve verticale** : sans terrain et hauteur des deux côtés, un obstacle
en plan reste un risque. Aucune hauteur n'est supposée — ni celle d'un
véhicule, ni trois mètres par étage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..logging import get_logger
from ..schemas.visibility import (
    HitVerdict,
    LineOfSightStatus,
    ObstacleHit,
    RayAssessment,
    RayPartition,
    VerticalVisibilityStatus,
    VisibilityAssessment,
)

log = get_logger("visibility-engine")

ENGINE_VERSION = "multiray-1.0.0"

#: Méthodes et modèles réellement implémentés. Une politique qui en nomme un
#: autre doit être refusée : le maillage est **uniforme**, et la projection
#: n'est valable que pour une caméra perspective.
SUPPORTED_SAMPLING = frozenset({"uniform_angular_cells"})
SUPPORTED_PROJECTION = frozenset({"pinhole_tangent"})


def check_supported(policy) -> list[str]:  # noqa: ANN001
    """Refuse une politique dont le moteur ne sait pas honorer les réglages."""
    problems = []
    if policy.sampling_method not in SUPPORTED_SAMPLING:
        problems.append(
            f"méthode d'échantillonnage {policy.sampling_method!r} non implémentée ; "
            f"disponibles : {sorted(SUPPORTED_SAMPLING)}"
        )
    if policy.projection_model not in SUPPORTED_PROJECTION:
        problems.append(
            f"modèle de projection {policy.projection_model!r} non implémenté ; "
            f"disponibles : {sorted(SUPPORTED_PROJECTION)}"
        )
    return problems


@dataclass(frozen=True)
class Obstacle:
    """Un voisin, et ce qu'on sait de sa hauteur."""

    feature_id: str
    shape: object  # polygone projeté
    height_m: float | None = None
    ground_m: float | None = None

    #: Référentiel vertical de `ground_m`. Sans lui, sa soustraction à une
    #: autre altitude suppose une origine commune que rien n'établit.
    vertical_crs: str | None = None

    @property
    def height_known(self) -> bool:
        return self.height_m is not None


@dataclass(frozen=True)
class CameraVertical:
    """Ce qu'on sait de la verticale à la caméra.

    Une caméra sans terrain connu ne peut rien prouver : la hauteur d'un
    véhicule de prise de vue est une convention, pas une mesure.
    """

    ground_m: float | None = None
    height_above_ground_m: float | None = None
    provenance: str | None = None
    vertical_crs: str | None = None

    @property
    def elevation_m(self) -> float | None:
        if self.ground_m is None or self.height_above_ground_m is None:
            return None
        return self.ground_m + self.height_above_ground_m


@dataclass(frozen=True)
class TargetVertical:
    """Verticale de la cible, éventuellement mesurée **au point visé**.

    Une hauteur médiane écraserait un bâtiment dont un corps est plus bas
    qu'un autre : c'est le point que le rayon touche qui compte, et le
    WelcomINNS varie de 3,3 m à 13 m selon l'endroit.
    """

    ground_m: float | None = None
    height_m: float | None = None
    provenance: str | None = None
    vertical_crs: str | None = None

    #: Renvoie (terrain, sommet, provenance) au point d'impact. Prioritaire sur
    #: les valeurs scalaires quand il est fourni.
    sampler: object = None

    def at(self, points) -> "TargetVertical":  # noqa: ANN001
        """Relève la verticale au premier point où elle est définie.

        Le rayon touche l'empreinte sur son **bord**, où les rasters n'ont
        souvent pas de valeur : une cellule de bordure est à cheval sur le
        dehors. On sonde donc quelques décimètres plus avant, dans le volume.
        """
        if self.sampler is None or not points:
            return self
        for point in points:
            ground, top = self.sampler(point)
            if ground is not None and top is not None:
                return TargetVertical(
                    ground_m=ground, height_m=max(top - ground, 0.0),
                    provenance=self.provenance,
                )
        return TargetVertical(provenance=self.provenance)


def normalise(angle: float) -> float:
    return angle % 360.0


def bearing_between(origin, point) -> float:  # noqa: ANN001
    """Azimut en degrés, mesuré depuis le nord, en projection.

    En EPSG:2950, `x` est un easting et `y` un northing : l'azimut se calcule
    donc avec `atan2(dx, dy)`, non l'inverse.
    """
    return normalise(math.degrees(math.atan2(point[0] - origin[0], point[1] - origin[1])))


def angular_span(origin, shape) -> tuple[float, float, float, bool]:  # noqa: ANN001
    """Intervalle angulaire réellement occupé par une forme.

    Ni le centroïde ni la boîte ne le donnent : sur un bâtiment oblique, la
    boîte ajoute des dizaines de degrés vides. On prend les azimuts de tous les
    sommets, puis on cherche le plus grand **trou** — l'intervalle occupé est
    son complément. C'est ce qui gère le passage 359° → 0° sans cas
    particulier.
    """  # noqa: D401
    coords = _boundary_coords(shape)
    bearings = sorted({bearing_between(origin, point) for point in coords})
    if len(bearings) < 2:
        return (bearings[0] if bearings else 0.0), (bearings[0] if bearings else 0.0), 0.0, False

    gaps = []
    for index, bearing in enumerate(bearings):
        following = bearings[(index + 1) % len(bearings)]
        gap = normalise(following - bearing)
        gaps.append((gap, bearing, following))

    widest_gap, gap_start, gap_end = max(gaps, key=lambda item: item[0])
    start, end = gap_end, gap_start
    span = normalise(end - start)
    if span == 0.0 and widest_gap > 0:
        span = 360.0 - widest_gap
    return start, end, span, start > end


def _boundary_coords(shape) -> list[tuple[float, float]]:  # noqa: ANN001
    if hasattr(shape, "exterior"):
        return list(shape.exterior.coords)
    if hasattr(shape, "geoms"):
        return [point for part in shape.geoms for point in _boundary_coords(part)]
    return list(shape.coords)


def cells(start: float, span: float, policy) -> list[tuple[float, float]]:  # noqa: ANN001
    """Découpe l'intervalle en cellules **uniformes**, chacune portant sa largeur.

    Le maillage n'est pas adaptatif : toutes les cellules ont la même
    ouverture. La pondération par la largeur n'en reste pas moins nécessaire —
    elle rend le résultat indépendant du pas choisi, et permettra un maillage
    non uniforme sans changer les fractions.

    Le nombre vient de la politique : un pas choisi dans le code changerait les
    fractions sans laisser de trace.
    """
    if span <= 0:
        return []
    count = max(
        policy.min_angular_cells,
        int(math.ceil(span / policy.max_angular_step_deg)),
    )
    width = span / count
    return [(normalise(start + (index + 0.5) * width), width) for index in range(count)]


def _ray(origin, bearing: float, length: float):  # noqa: ANN001
    from shapely.geometry import LineString

    radians = math.radians(bearing)
    return LineString(
        [origin, (origin[0] + length * math.sin(radians), origin[1] + length * math.cos(radians))]
    )


def _first_hit(origin, ray, shape) -> float | None:  # noqa: ANN001
    """Distance à la première intersection d'un rayon avec une forme."""
    from shapely.geometry import Point

    crossing = ray.intersection(shape)
    if crossing.is_empty:
        return None
    return Point(origin).distance(crossing)


def _incomparable_references(
    camera: CameraVertical, obstacle: Obstacle, target: TargetVertical,
    vertical: object,
) -> list[str]:
    """Quelles altitudes ne peuvent pas être comparées sans supposition ?

    Sans référence de site déclarée, on n'exige rien : c'est l'état antérieur,
    et le rendre bloquant périmerait des runs déjà produits sur une source
    verticale unique. Dès qu'une référence existe, elle fait autorité.
    """
    if vertical is None or not getattr(vertical, "is_known", False):
        return []

    problems = []
    for label, declared in (
        ("caméra", camera.vertical_crs),
        (f"obstacle {obstacle.feature_id}", obstacle.vertical_crs),
        ("cible", target.vertical_crs),
    ):
        if declared is None:
            problems.append(f"référentiel vertical non déclaré pour {label}")
        elif not vertical.comparable_with(declared):
            problems.append(
                f"référentiel {declared!r} de {label} sans transformation "
                f"déclarée vers {vertical.crs!r}"
            )
    return problems


def vertical_verdict(
    origin,  # noqa: ANN001
    obstacle: Obstacle,
    obstacle_distance: float,
    target_distance: float,
    camera: CameraVertical,
    target: TargetVertical,
    vertical: object = None,
) -> tuple[bool, VerticalVisibilityStatus, list[str]]:
    """L'obstacle masque-t-il **prouvablement** la cible sur ce rayon ?

    Il faut tout connaître : terrain et hauteur de caméra, de cible et
    d'obstacle. Une seule absence laisse un risque — et l'obstacle ne masque
    que s'il couvre **toute** la bande utile de la cible, un toit dépassant
    au-dessus suffisant à laisser voir la silhouette.

    `vertical` est le `VerticalReference` du site. Les trois altitudes ne se
    soustraient qu'à référentiel identique, ou via une transformation déclarée :
    orthométrique et ellipsoïdal diffèrent ici de plusieurs dizaines de mètres,
    et les mélanger produirait un blocage « prouvé » qui n'existe pas.
    """
    incomparable = _incomparable_references(camera, obstacle, target, vertical)
    if incomparable:
        # Ni blocage ni certitude : on ne sait pas, et on dit pourquoi.
        return False, VerticalVisibilityStatus.UNKNOWN, incomparable

    missing: list[str] = []
    if camera.ground_m is None:
        missing.append("terrain à la caméra")
    if camera.height_above_ground_m is None:
        missing.append("hauteur de caméra")
    if target.ground_m is None:
        missing.append("terrain de la cible")
    if target.height_m is None:
        missing.append("hauteur de la cible")
    if obstacle.ground_m is None:
        missing.append(f"terrain de {obstacle.feature_id}")
    if obstacle.height_m is None:
        missing.append(f"hauteur de {obstacle.feature_id}")

    if missing:
        # « Incomplet » quand une partie est connue, « inconnu » quand rien ne
        # l'est : la nuance dit s'il vaut la peine d'aller chercher le reste.
        status = (
            VerticalVisibilityStatus.UNKNOWN
            if len(missing) == 6
            else VerticalVisibilityStatus.INCOMPLETE
        )
        return False, status, missing

    eye = camera.elevation_m
    obstacle_top = obstacle.ground_m + obstacle.height_m
    target_top = target.ground_m + target.height_m

    # Élévation de la ligne de vue au droit de l'obstacle, si l'on vise le
    # sommet de la cible.
    ratio = obstacle_distance / target_distance if target_distance else 1.0
    line_at_obstacle = eye + (target_top - eye) * ratio

    return obstacle_top >= line_at_obstacle, VerticalVisibilityStatus.FULLY_KNOWN, []


def assess(
    assessment_id: str,
    subject_ref: str,
    target_ref: str,
    origin,  # noqa: ANN001 — (x, y) projeté
    target_shape,  # noqa: ANN001
    obstacles: list[Obstacle],
    policy,  # noqa: ANN001 — VisibilityPolicy
    camera: CameraVertical | None = None,
    target_vertical: TargetVertical | None = None,
    vertical: object = None,
    *,
    crs: str,
) -> VisibilityAssessment:
    """Évalue une ligne de vue, cellule par cellule. **Sans cadrage.**

    Le champ de vision n'entre pas ici : deux recadrages d'un même panorama
    voient la même scène, et faire varier la visibilité avec l'objectif
    reviendrait à déplacer les murs en tournant la caméra. Ce que le cadre
    laisse entrer se mesure dans `frame_target`.
    """
    from shapely.geometry import Point

    camera = camera or CameraVertical()
    target_vertical = target_vertical or TargetVertical()
    precision = policy.output_precision

    start, end, span, crosses = angular_span(origin, target_shape)
    distance = Point(origin).distance(target_shape)

    assessment = VisibilityAssessment(
        assessment_id=assessment_id,
        subject_ref=subject_ref,
        target_ref=target_ref,
        camera_x=round(origin[0], precision),
        camera_y=round(origin[1], precision),
        crs=crs,
        span_start_deg=round(start, precision),
        span_end_deg=round(end, precision),
        angular_span_deg=round(span, precision),
        crosses_north=crosses,
        distance_m=round(distance, precision),
    )
    if span <= 0:
        assessment.status = LineOfSightStatus.INSUFFICIENT_DATA
        return assessment

    # Longueur des rayons dérivée de la cible : un rayon fixe de mille mètres
    # traversait des voisins hors sujet et coûtait pour rien.
    reach = _reach(origin, target_shape)

    weights = {partition: 0.0 for partition in RayPartition}
    rays: list[RayAssessment] = []
    at_risk: set[str] = set()
    blocking: set[str] = set()
    missing_vertical: set[str] = set()
    clear_run = best_clear = 0.0

    for bearing, width in cells(start, span, policy):
        partition, ray = _assess_cell(
            origin, bearing, width, target_shape, obstacles, camera, target_vertical,
            policy, reach, vertical,
        )
        weights[partition] += width
        rays.append(ray)

        if partition is RayPartition.CLEAR_2D:
            clear_run += width
            best_clear = max(best_clear, clear_run)
        else:
            clear_run = 0.0

        # Croisé n'est pas responsable : seuls les verdicts individuels
        # nourrissent les agrégats.
        at_risk.update(ray.at_risk)
        blocking.update(ray.blocking)
        missing_vertical.update(ray.missing_vertical)

    total = sum(weights.values()) or 1.0
    assessment.rays = rays
    assessment.proven_clear_fraction = round(weights[RayPartition.CLEAR_2D] / total, precision)
    assessment.risk_unknown_height_fraction = round(
        weights[RayPartition.RISK_UNKNOWN_HEIGHT] / total, precision
    )
    # La dernière fraction absorbe l'arrondi : trois valeurs arrondies
    # séparément ne totalisent pas 1, et l'invariant refuserait l'évaluation.
    assessment.proven_blocked_fraction = round(
        1.0 - assessment.proven_clear_fraction - assessment.risk_unknown_height_fraction,
        precision,
    )
    assessment.largest_clear_span_deg = round(best_clear, precision)
    assessment.obstacles_at_risk = sorted(at_risk)
    assessment.obstacles_blocking = sorted(blocking)
    assessment.missing_vertical = sorted(missing_vertical)
    assessment.status = _status_of(assessment)
    return assessment


def _reach(origin, target_shape) -> float:  # noqa: ANN001
    """Portée utile d'un rayon : au-delà de la cible, plus rien ne masque."""
    from shapely.geometry import Point

    point = Point(origin)
    farthest = max(point.distance(Point(vertex)) for vertex in _boundary_coords(target_shape))
    return farthest * 1.05 + 1.0


def _assess_cell(
    origin, bearing: float, width: float, target_shape, obstacles: list[Obstacle],  # noqa: ANN001
    camera: CameraVertical, target_vertical: TargetVertical, policy, reach: float,  # noqa: ANN001
    vertical: object = None,
) -> tuple[RayPartition, RayAssessment]:
    precision = policy.output_precision
    ray = _ray(origin, bearing, reach)
    target_distance = _first_hit(origin, ray, target_shape)

    if target_distance is None:
        # La cellule vise la silhouette mais ne la touche pas : un contour
        # concave laisse des directions vides à l'intérieur de l'intervalle.
        return RayPartition.CLEAR_2D, RayAssessment(
            bearing_deg=round(bearing, precision), angular_width_deg=width,
            partition=RayPartition.CLEAR_2D,
        )

    # Seuls comptent les obstacles **avant** la cible : derrière, ils ne
    # masquent rien, et les compter condamnait des vues parfaitement dégagées.
    interposed: list[tuple[float, Obstacle]] = []
    for obstacle in obstacles:
        hit = _first_hit(origin, ray, obstacle.shape)
        if hit is None or hit >= target_distance - policy.intersection_tolerance_m:
            continue
        interposed.append((hit, obstacle))
    interposed.sort(key=lambda item: item[0])

    # La verticale de la cible est relevée là où le rayon la touche, puis un
    # peu plus avant si la bordure n'est pas définie.
    local_target = target_vertical.at(
        [_point_at(origin, bearing, target_distance + offset) for offset in (0.0, 0.5, 1.5, 3.0)]
    )

    hits: list[ObstacleHit] = []
    blocked = False
    incomplete: list[str] = []
    for distance, obstacle in interposed:
        proven, vertical_status, absent = vertical_verdict(
            origin, obstacle, distance, target_distance, camera, local_target,
            vertical,
        )
        verdict = (
            HitVerdict.UNDECIDABLE if absent
            else HitVerdict.BLOCKS if proven
            else HitVerdict.PASSES_UNDER
        )
        hits.append(
            ObstacleHit(
                obstacle_ref=obstacle.feature_id,
                distance_m=round(distance, precision),
                vertical_status=vertical_status,
                verdict=verdict,
                missing_vertical=sorted(set(absent)),
            )
        )
        blocked = blocked or verdict is HitVerdict.BLOCKS
        incomplete.extend(absent)

    common = dict(
        bearing_deg=round(bearing, precision), angular_width_deg=width,
        target_distance_m=round(target_distance, precision), hits=hits,
    )

    # Priorité : blocage prouvé, puis risque, puis libre.
    if blocked:
        return RayPartition.BLOCKED_2_5D, RayAssessment(
            partition=RayPartition.BLOCKED_2_5D,
            vertical_status=VerticalVisibilityStatus.FULLY_KNOWN, **common
        )
    if incomplete:
        return RayPartition.RISK_UNKNOWN_HEIGHT, RayAssessment(
            partition=RayPartition.RISK_UNKNOWN_HEIGHT,
            vertical_status=VerticalVisibilityStatus.INCOMPLETE
            if len(set(incomplete)) < 6
            else VerticalVisibilityStatus.UNKNOWN,
            missing_vertical=sorted(set(incomplete)), **common
        )
    return RayPartition.CLEAR_2D, RayAssessment(
        partition=RayPartition.CLEAR_2D,
        vertical_status=(
            VerticalVisibilityStatus.FULLY_KNOWN if hits
            else VerticalVisibilityStatus.UNKNOWN
        ),
        **common
    )


def _point_at(origin, bearing: float, distance: float) -> tuple[float, float]:  # noqa: ANN001
    radians = math.radians(bearing)
    return (
        origin[0] + distance * math.sin(radians),
        origin[1] + distance * math.cos(radians),
    )


def _within_frame(bearing: float, heading: float, half_fov: float) -> bool:
    """Écart angulaire au cap, en tenant compte du passage par 0°."""
    difference = abs((bearing - heading + 180.0) % 360.0 - 180.0)
    return difference <= half_fov


def _status_of(assessment: VisibilityAssessment) -> LineOfSightStatus:
    if assessment.proven_blocked_fraction >= 1.0 - 1e-9:
        return LineOfSightStatus.BLOCKED
    if assessment.proven_clear_fraction >= 1.0 - 1e-9:
        return LineOfSightStatus.CLEAR
    if assessment.risk_unknown_height_fraction > 0 and assessment.proven_clear_fraction == 0:
        return LineOfSightStatus.AT_RISK
    return LineOfSightStatus.PARTIAL


# --- cadrage ------------------------------------------------------------------


def frame_target(
    assessment_id: str,
    subject_ref: str,
    span_start_deg: float | None,
    angular_span_deg: float | None,
    heading_deg: float | None,
    fov_deg: float | None,
    width_px: int | None,
    height_px: int | None,
    parameters_source: str | None,
    policy,  # noqa: ANN001
    pitch_deg: float | None = None,
    target_vertical_span_deg: float | None = None,
    reason_if_absent: str = "paramètres de caméra absents",
):
    """Ce qu'une caméra perspective laisse entrer de la cible.

    Les deux bornes sont projetées **séparément** par leur propre tangente :
    `tan(span/2)/tan(fov/2)` n'est exact que si la cible est centrée. Une
    façade à 20° du cap n'occupe pas la même largeur qu'au milieu du cadre, et
    c'est précisément près des bords qu'on décide si elle y tient.

    L'intersection avec le cadre est analytique — bornage des angles, non
    comptage d'échantillons.
    """
    from ..schemas.visibility import FramingAssessment

    missing = [
        name
        for name, value in (
            ("cap", heading_deg), ("champ", fov_deg),
            ("largeur", width_px), ("hauteur", height_px),
        )
        if value is None
    ]
    if missing or angular_span_deg is None or span_start_deg is None:
        return FramingAssessment(
            assessment_id=assessment_id, subject_ref=subject_ref,
            heading_deg=heading_deg, fov_deg=fov_deg, pitch_deg=pitch_deg,
            width_px=width_px, height_px=height_px,
            parameters_source=parameters_source,
            horizontal_computable=False,
            horizontal_reason=(
                f"{reason_if_absent} : {missing}" if missing
                else "silhouette sans intervalle angulaire"
            ),
            vertical_computable=False,
            vertical_reason="largeur non calculable, hauteur encore moins",
        )

    half_fov = fov_deg / 2.0
    half_tan = math.tan(math.radians(half_fov))

    # Écarts signés des deux bornes au cap, ramenés dans [-180, 180].
    left = _signed_offset(span_start_deg, heading_deg)
    right = left + angular_span_deg

    unclipped = abs(_tangent(right, half_tan) - _tangent(left, half_tan)) / 2.0

    # Intersection analytique avec le cadre : on borne les angles, pas les
    # échantillons.
    clipped_left = max(left, -half_fov)
    clipped_right = min(right, half_fov)
    if clipped_right <= clipped_left:
        in_frame = 0.0
        clipped = 0.0
    else:
        in_frame = (clipped_right - clipped_left) / angular_span_deg
        clipped = abs(
            _tangent(clipped_right, half_tan) - _tangent(clipped_left, half_tan)
        ) / 2.0

    precision = policy.output_precision
    vertical_fov = 2.0 * math.degrees(math.atan(half_tan * (height_px / width_px)))

    # La hauteur exige une inclinaison : une visée réputée horizontale est une
    # convention, pas une mesure.
    vertical_computable = pitch_deg is not None and target_vertical_span_deg is not None
    expected_height_px = None
    vertical_reason = None
    if vertical_computable:
        half_vertical_tan = math.tan(math.radians(vertical_fov / 2.0))
        top = _signed_offset_value(target_vertical_span_deg / 2.0 - pitch_deg)
        bottom = _signed_offset_value(-target_vertical_span_deg / 2.0 - pitch_deg)
        height_fraction = abs(
            _tangent(top, half_vertical_tan) - _tangent(bottom, half_vertical_tan)
        ) / 2.0
        expected_height_px = int(round(min(height_fraction, 1.0) * height_px))
    else:
        vertical_reason = (
            "inclinaison de visée inconnue"
            if pitch_deg is None
            else "étendue verticale de la cible inconnue"
        )

    return FramingAssessment(
        assessment_id=assessment_id, subject_ref=subject_ref,
        heading_deg=heading_deg, fov_deg=fov_deg,
        vertical_fov_deg=round(vertical_fov, precision),
        pitch_deg=pitch_deg, width_px=width_px, height_px=height_px,
        parameters_source=parameters_source,
        projection_model=policy.projection_model,
        target_in_frame_fraction=round(min(max(in_frame, 0.0), 1.0), precision),
        unclipped_width_fraction=round(unclipped, precision),
        clipped_width_fraction=round(min(clipped, 1.0), precision),
        expected_width_px=int(round(min(clipped, 1.0) * width_px)),
        expected_height_px=expected_height_px,
        horizontal_computable=True,
        vertical_computable=vertical_computable,
        vertical_reason=vertical_reason,
    )


def _signed_offset(bearing: float, heading: float) -> float:
    """Écart signé au cap, dans [-180, 180]."""
    return (bearing - heading + 180.0) % 360.0 - 180.0


def _signed_offset_value(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def _tangent(angle_deg: float, half_tan: float) -> float:
    """Position d'un angle sur la largeur normalisée du cadre.

    Bornée : au-delà de 89°, la tangente diverge, et une cible qui déborde du
    cadre n'a pas besoin d'un nombre infini pour être dite hors champ.
    """
    bounded = max(-89.0, min(89.0, angle_deg))
    return math.tan(math.radians(bounded)) / half_tan


# --- corridors ------------------------------------------------------------------


def sample_line(line, step_m: float) -> list[tuple[str, tuple[float, float]]]:  # noqa: ANN001
    """Échantillonne une ligne projetée, pas régulier et identifiants stables.

    Le point de fermeture d'une boucle n'est pas compté deux fois : la voie
    d'accès du WelcomINNS en est une, et le doublon aurait gonflé son compte
    d'emplacements.
    """
    length = line.length
    if length <= 0:
        return []

    count = max(1, int(math.floor(length / step_m)))
    samples: list[tuple[str, tuple[float, float]]] = []
    seen: set[tuple[float, float]] = set()

    for index in range(count + 1):
        point = line.interpolate(min(index * step_m, length))
        key = (round(point.x, 3), round(point.y, 3))
        if key in seen:
            continue
        seen.add(key)
        samples.append((f"s{index:03d}", (point.x, point.y)))
    return samples


def group_segments(useful_indices: list[int]) -> int:
    """Nombre de segments continus dans une suite d'échantillons utiles.

    Vingt-cinq échantillons d'une même route ne font pas vingt-cinq points de
    vue : ce sont les ruptures qui comptent.
    """
    if not useful_indices:
        return 0
    ordered = sorted(useful_indices)
    segments = 1
    for previous, current in zip(ordered, ordered[1:]):
        if current != previous + 1:
            segments += 1
    return segments
