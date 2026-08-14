"""Ce que la géométrie dit d'un candidat, **avant** toute acquisition.

`assets plan` classait presque tout en `preview_required` : sans mesure de
cadrage, il ne triait pas, il différait. Ce module produit la mesure — et il
la produit sur des **métadonnées**, puisqu'à ce stade aucune image n'existe.

D'où le vocabulaire du schéma, qu'on respecte ici scrupuleusement : ces valeurs
sont des **espérances**, non des mesures. `unclipped_width_fraction` n'est pas
bornée à 1 : une cible plus large que le champ de vision déborde légitimement
du cadre, et écrêter effacerait cette information.

Rien n'est supposé. Une caméra sans champ de vision déclaré ne rend aucune
fraction de cadre — elle rend `None`, et le plan demandera une miniature. Un
site sans contexte spatial ne se mesure pas du tout : projeter dans un
référentiel supposé donnerait des mètres finis et faux.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .logging import get_logger
from .schemas.acquisition import CandidateGeometry

log = get_logger("candidate-geometry")


class GeometryUnavailable(RuntimeError):
    """Aucune mesure n'a été produite, et rien n'a été supposé."""


@dataclass
class GeometryReport:
    """Ce qui a été mesuré, ce qui ne l'a pas été, et pourquoi."""

    measured: int = 0
    skipped: dict[str, str] = field(default_factory=dict)
    with_framing: int = 0
    without_framing: dict[str, int] = field(default_factory=dict)

    #: Besoins dont la cible n'a pas été trouvée. Comptés à part : un besoin
    #: non mesurable n'est pas un besoin non servi, et le rabattre sur le
    #: bâtiment principal serait le défaut qu'on corrige.
    unresolved_targets: dict[str, str] = field(default_factory=dict)

    #: Couples mesurés depuis un côté que le besoin n'accepte pas.
    wrong_sector: int = 0

    #: Couples dont la caméra se trouve dans une zone interdite au besoin.
    forbidden_zone_entries: int = 0

    #: Seuil effectivement appliqué, inscrit au rapport comme à la mesure.
    sector_half_width_deg: float | None = None

    def as_dict(self) -> dict:
        return {
            "measured": self.measured,
            "skipped": self.skipped,
            "framing": {
                "computable": self.with_framing,
                "not_computable": self.without_framing,
            },
            "unresolved_targets": self.unresolved_targets,
            "wrong_sector": self.wrong_sector,
            "forbidden_zone_entries": self.forbidden_zone_entries,
            "sector_half_width_deg": self.sector_half_width_deg,
            "note": (
                "espérances calculées sur métadonnées : aucune image n'a été "
                "acquise, et aucune de ces valeurs n'est une mesure sur pixels"
            ),
        }


def measure(
    candidate,  # noqa: ANN001 — CaptureCandidate
    target_shape,  # noqa: ANN001 — empreinte projetée de **cette** cible
    projection,  # noqa: ANN001 — ProjectionService
    policy,  # noqa: ANN001 — VisibilityPolicy
    obstacles: list | None = None,
    view_sector=None,  # noqa: ANN001
) -> CandidateGeometry:
    """Ce qu'on peut espérer de ce candidat, sans avoir vu son image.

    La distance et l'intervalle angulaire se calculent dès qu'une position
    existe. La taille projetée, elle, exige un champ de vision : sans lui, la
    fraction de cadre reste inconnue plutôt que devinée — c'est la différence
    entre « on ne sait pas » et « on a supposé un objectif ».
    """
    from shapely.geometry import Point

    from .geo import visibility_engine as engine

    if candidate.camera_lat is None or candidate.camera_lon is None:
        raise GeometryUnavailable(
            f"{candidate.candidate_id} : aucune position de caméra"
        )

    origin = projection.point(candidate.camera_lat, candidate.camera_lon)
    distance = Point(origin).distance(target_shape)
    start, _, span, _ = engine.angular_span(origin, target_shape)

    heading = (
        candidate.requested_heading_deg
        if candidate.requested_heading_deg is not None
        else candidate.computed_heading_deg or candidate.original_heading_deg
    )
    offset = None
    if heading is not None:
        centre = engine.normalise(start + span / 2.0)
        offset = abs((centre - heading + 180.0) % 360.0 - 180.0)

    geometry = CandidateGeometry(
        distance_m=round(distance, policy.output_precision),
        angular_span_deg=round(span, policy.output_precision),
        target_offset_deg=round(offset, policy.output_precision) if offset is not None else None,
        view_sector=view_sector,
    )

    framing = _framing(candidate, start, span, heading, policy)
    if framing is not None:
        geometry = geometry.model_copy(update=framing)

    risk, blocking = _occlusion(origin, target_shape, obstacles or [], policy)
    if risk or blocking:
        geometry = geometry.model_copy(
            update={"occlusion_risk": bool(risk), "occluded_by": sorted(blocking)}
        )
    return geometry


def _framing(candidate, start: float, span: float, heading, policy) -> dict | None:  # noqa: ANN001
    """Fraction de cadre espérée, si la caméra dit assez d'elle-même.

    Trois éléments sont nécessaires : un cap, un champ de vision, et la
    largeur de l'image. Il en manque un, et la fraction reste inconnue —
    supposer un objectif produirait un tri fondé sur une caméra imaginaire.
    """
    fov = candidate.requested_fov_deg
    width_px = candidate.advertised_width
    if heading is None or fov is None:
        return None

    framed = engine_frame(candidate, start, span, heading, fov, width_px, policy)
    if framed is None or not framed.horizontal_computable:
        return None

    # Deux grandeurs distinctes, que confondre rendrait un tri faux : la
    # largeur **non écrêtée** dit la taille apparente de la cible, y compris
    # quand elle déborde ; la fraction **dans le cadre** dit ce que l'image
    # contiendra réellement. Une cible deux fois plus large que le champ a une
    # largeur de 2,0 et une fraction dans le cadre de 0,5.
    update = {
        "unclipped_width_fraction": framed.unclipped_width_fraction,
        "clipped_width_fraction": framed.clipped_width_fraction,
        "in_frame_fraction": framed.target_in_frame_fraction,
    }
    if width_px and framed.clipped_width_fraction is not None:
        update["expected_width_px"] = int(
            round(framed.clipped_width_fraction * width_px)
        )
    return update


def engine_frame(candidate, start, span, heading, fov, width_px, policy):  # noqa: ANN001, ANN201
    """Appelle le calcul de cadrage du moteur, qui fait déjà autorité.

    Le refaire ici créerait deux règles pour une question, et la projection
    tangentielle est exactement le genre de calcul qu'on ne veut pas voir
    diverger entre deux modules.
    """
    from .geo import visibility_engine as engine

    return engine.frame_target(
        assessment_id=f"cand-{candidate.candidate_id}",
        subject_ref=candidate.candidate_id,
        span_start_deg=start, angular_span_deg=span,
        heading_deg=heading, fov_deg=fov,
        width_px=width_px, height_px=candidate.advertised_height,
        parameters_source="métadonnées du candidat", policy=policy,
        reason_if_absent="champ de vision non déclaré par la source",
    )


def _occlusion(origin, target_shape, obstacles: list, policy) -> tuple[bool, set]:  # noqa: ANN001
    """Obstacles interposés en plan, sans jamais conclure à un blocage.

    À ce stade, aucune hauteur n'est connue de façon fiable : un obstacle
    rencontré est un **risque**, jamais une preuve. Le transformer en certitude
    écarterait des vues parfaitement dégagées.

    Le segment s'arrête au centroïde de la cible : rien ne peut donc être
    « derrière », et aucun contrôle de profondeur n'est nécessaire ici. La
    mesure complète — plusieurs rayons, profondeur, verticale — appartient au
    moteur de visibilité, qui travaille sur des positions acquises ; ce qu'on
    produit ici est une alerte sur métadonnées, et elle est nommée comme telle.
    """
    from shapely.geometry import LineString

    if not obstacles:
        return False, set()

    centre = target_shape.centroid
    ray = LineString([origin, (centre.x, centre.y)])

    found = set()
    for obstacle in obstacles:
        shape = getattr(obstacle, "shape", None)
        if shape is None or not ray.intersects(shape):
            continue
        found.add(getattr(obstacle, "feature_id", "?"))
    return bool(found), found


def _forbidden_entered(candidate, demand, zones: dict, projection) -> set:  # noqa: ANN001
    """Zones interdites où se trouve la caméra, parmi celles que le besoin nomme.

    Le schéma validait ces références sans que rien ne les fasse agir : une
    vue prise depuis une zone interdite était planifiée comme une autre.
    """
    from shapely.geometry import Point

    refs = getattr(demand, "forbidden_zone_refs", None) or []
    if not refs:
        return set()

    origin = Point(projection.point(candidate.camera_lat, candidate.camera_lon))
    return {ref for ref in refs if ref in zones and zones[ref].contains(origin)}


def measure_all(
    candidates: list,
    manifest,  # noqa: ANN001 — CaptureGeometryManifest
    projection,  # noqa: ANN001
    policy,  # noqa: ANN001
    demands: list,
    obstacles: list | None = None,
    front_azimuth_deg: float | None = None,
    site=None,  # noqa: ANN001
    half_width_deg: float | None = None,
    forbidden_zones: dict | None = None,
) -> tuple[dict, GeometryReport]:
    """Mesure chaque candidat **contre la cible de chaque besoin**.

    La version précédente calculait une géométrie sur le bâtiment principal et
    la recopiait à tous les besoins : un besoin d'entrée était mesuré contre la
    façade entière, et deux secteurs opposés recevaient la même réponse. Ici,
    chaque besoin résout sa propre cible, et un besoin dont la cible n'est pas
    résolue n'est pas mesuré du tout — il n'est pas rabattu sur le bâtiment.
    """
    from .demand_targets import (
        DEFAULT_SECTOR_HALF_WIDTH_DEG, TargetUnresolved, observer_bearing, resolve,
    )

    half_width = (
        half_width_deg if half_width_deg is not None
        else DEFAULT_SECTOR_HALF_WIDTH_DEG
    )
    zones = forbidden_zones or {}
    report = GeometryReport(sector_half_width_deg=half_width)
    measured: dict[tuple[str, str], CandidateGeometry] = {}

    targets = {}
    for demand in demands:
        try:
            targets[demand.demand_id] = resolve(
                demand, manifest, front_azimuth_deg, site, half_width
            )
        except TargetUnresolved as exc:
            report.unresolved_targets[demand.demand_id] = str(exc).split(" : ", 1)[-1]

    for candidate in candidates:
        if candidate.camera_lat is None or candidate.camera_lon is None:
            report.skipped[candidate.candidate_id] = "aucune position de caméra"
            continue

        for demand in demands:
            target = targets.get(demand.demand_id)
            if target is None:
                continue

            try:
                geometry = measure(
                    candidate, target.shape, projection, policy, obstacles,
                )
            except GeometryUnavailable as exc:
                report.skipped[candidate.candidate_id] = str(exc).split(" : ", 1)[-1]
                break

            # Le secteur ne se déduit pas de la distance : une vue excellente
            # prise du mauvais côté ne montre pas la façade demandée.
            origin = projection.point(candidate.camera_lat, candidate.camera_lon)
            bearing = observer_bearing(origin, target.shape)
            if not target.observer_is_admissible(bearing):
                geometry = geometry.model_copy(
                    update={"view_sector": None, "wrong_sector": True}
                )
                report.wrong_sector += 1

            # Le seuil effectif voyage avec la mesure : le modifier doit
            # périmer ce qu'il a produit, et une mesure qui ne le porte pas ne
            # peut pas être confrontée.
            geometry = geometry.model_copy(
                update={"sector_half_width_deg": half_width}
            )

            entered = _forbidden_entered(candidate, demand, zones, projection)
            if entered:
                geometry = geometry.model_copy(
                    update={"forbidden_zones_entered": sorted(entered)}
                )
                report.forbidden_zone_entries += 1

            measured[(candidate.candidate_id, demand.demand_id)] = geometry
            report.measured += 1
            if geometry.unclipped_width_fraction is not None:
                report.with_framing += 1
            else:
                reason = (
                    "cap absent"
                    if candidate.original_heading_deg is None
                    and candidate.requested_heading_deg is None
                    else "champ de vision non déclaré"
                )
                report.without_framing[reason] = (
                    report.without_framing.get(reason, 0) + 1
                )

    log.info(
        "géométrie de candidats : %d couple(s) mesuré(s), %d avec cadrage, "
        "%d hors secteur, %d cible(s) non résolue(s)",
        report.measured, report.with_framing, report.wrong_sector,
        len(report.unresolved_targets),
    )
    return measured, report
