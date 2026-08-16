"""Exécution du moteur de visibilité sur un corpus réel.

Ne mute rien : la projection vers les assets est une commande distincte. Ce
module lit le manifeste géométrique, le corpus et la politique, mesure, et
publie un rapport versionné.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from ..logging import get_logger
from ..schemas.geometry import GeometryResolutionStatus, GeometryRole
from ..schemas.visibility import (
    CorridorVisibilityAssessment,
    LineOfSightStatus,
    UsefulnessVerdict,
    VisibilityRun,
)
from . import visibility_engine as engine

log = get_logger("visibility-run")


def digest(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def base_manifest_digest(manifest) -> str:  # noqa: ANN001
    """Empreinte du manifeste **hors résultats de visibilité**.

    Sans cette normalisation, appliquer un run le périmerait aussitôt : les
    champs qu'il vient d'écrire changeraient l'empreinte qu'il déclare. Tout le
    reste — cap, position, revue humaine, aptitude, scores — reste dans le
    calcul, et doit bien périmer le run.

    Employée par `assess` comme par `apply` : deux définitions divergentes
    rendraient la comparaison inutile.
    """
    from ..schemas.assets import VISIBILITY_PROJECTED_FIELDS

    payload = json.loads(manifest.model_dump_json())
    for asset in payload.get("assets", []):
        for field_name in VISIBILITY_PROJECTED_FIELDS:
            asset.pop(field_name, None)
    return digest(payload)


@dataclass
class RunReport:
    run_id: str = ""
    assets_assessed: int = 0

    #: Contrôles de projection effectués avant tout calcul : emprise, finitude
    #: et aller-retour. Vide quand aucun contexte spatial n'a été fourni.
    projection_check: dict = field(default_factory=dict)

    by_status: dict[str, int] = field(default_factory=dict)
    by_source: dict[str, dict[str, int]] = field(default_factory=dict)
    by_sector: dict[str, dict[str, int]] = field(default_factory=dict)

    #: Ce que devient l'ancien verdict d'occultation : c'est la question qui
    #: motivait tout le sous-lot.
    previously_occluded: int = 0
    previously_occluded_now: dict[str, int] = field(default_factory=dict)

    proven_blocked: int = 0

    #: Deux comptes distincts, que le premier rapport confondait : ceux dont le
    #: statut est exclusivement `at_risk`, et ceux qui portent une part de
    #: risque — les 49 `partial` en portent une aussi.
    at_risk_only: int = 0
    with_risk_fraction: int = 0
    clear: int = 0

    #: Pourquoi les risques sont des risques : quelle donnée verticale manque,
    #: et quels obstacles la portent.
    missing_vertical_counts: dict[str, int] = field(default_factory=dict)

    #: Nombre d'**assets** dont au moins un rayon rencontre cet obstacle sans
    #: pouvoir conclure. Le nom précédent parlait de rayons : il en comptait
    #: bien davantage.
    obstacles_by_affected_assets: dict[str, int] = field(default_factory=dict)

    framing_computable: int = 0
    framing_not_computable: dict[str, int] = field(default_factory=dict)

    corridors_assessed: int = 0
    corridors_useful: dict[str, int] = field(default_factory=dict)
    useful_corridor_details: list[dict] = field(default_factory=list)

    parameters: dict[str, str] = field(default_factory=dict)
    digests: dict[str, str] = field(default_factory=dict)
    enrichment: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "projection_check": self.projection_check,
            "assets": {
                "assessed": self.assets_assessed,
                "by_status": self.by_status,
                "by_source": self.by_source,
                "by_sector": self.by_sector,
                "clear": self.clear,
                "at_risk_only": self.at_risk_only,
                "with_risk_fraction": self.with_risk_fraction,
                "proven_blocked": self.proven_blocked,
                "note": (
                    "`clear` signifie seulement qu'une direction vers l'empreinte "
                    "n'est pas obstruée en plan depuis cette position. Ce n'est ni "
                    "une preuve que la caméra vise le bâtiment, ni qu'il entre dans "
                    "l'image : sans cadrage calculable, aucun de ces assets ne peut "
                    "être promu."
                ),
            },
            "previously_occluded": {
                "total": self.previously_occluded,
                "now": self.previously_occluded_now,
            },
            "why_at_risk": {
                "missing_vertical": self.missing_vertical_counts,
                "obstacles_by_affected_assets": self.obstacles_by_affected_assets,
            },
            "framing": {
                "computable": self.framing_computable,
                "not_computable": self.framing_not_computable,
            },
            "corridors": {
                "assessed": self.corridors_assessed,
                "usefulness": self.corridors_useful,
                "useful": self.useful_corridor_details,
            },
            "vertical_enrichment": self.enrichment,
            "parameters": self.parameters,
            "digests": self.digests,
        }


def _obstacles(manifest, heights: dict) -> list[engine.Obstacle]:  # noqa: ANN001
    from shapely import wkt as shapely_wkt

    obstacles = []
    for geometry in manifest.geometries:
        if geometry.role is not GeometryRole.OBSTACLE_BUILDING:
            continue
        if geometry.resolution_status is not GeometryResolutionStatus.RESOLVED:
            continue
        obstacles.append(
            engine.Obstacle(
                feature_id=geometry.feature_id,
                shape=shapely_wkt.loads(geometry.projected_wkt),
                # Hauteur du tag OSM si elle existe, sinon celle mesurée dans
                # le nuage. Rien n'est estimé : sans mesure, la valeur reste
                # absente et le rayon restera un risque.
                height_m=geometry.height_m or heights.get(geometry.feature_id, {}).get("height_m"),
                ground_m=heights.get(geometry.feature_id, {}).get("ground_m"),
            )
        )
    return obstacles


def _target(manifest):  # noqa: ANN001
    from shapely import wkt as shapely_wkt

    for geometry in manifest.geometries:
        if (
            geometry.role is GeometryRole.TARGET_BUILDING
            and geometry.resolution_status is GeometryResolutionStatus.RESOLVED
        ):
            return shapely_wkt.loads(geometry.projected_wkt), geometry
    return None, None


def run_assessment(
    run_id: str,
    hotel_id: str,
    assets: list,
    manifest,  # noqa: ANN001 — CaptureGeometryManifest
    policy,  # noqa: ANN001 — PipelinePolicy
    digests: dict[str, str],
    front_azimuth_deg: float | None = None,
    target_vertical=None,  # noqa: ANN001 — TargetVertical enrichie
    camera_ground=None,  # noqa: ANN001 — callable (x, y) -> Sample | None
    obstacle_heights: dict | None = None,
    elevation_sources: list | None = None,
    spatial_reference=None,  # noqa: ANN001 — SpatialReferenceContext
) -> tuple[VisibilityRun, RunReport]:
    """Mesure la visibilité de chaque asset situé et de chaque corridor.

    `spatial_reference` est obligatoire, et toute exécution sans contexte est
    refusée avant le premier calcul. Le repli historique `EPSG:2950` n'est pas
    autorisé : un calcul portant sur le mauvais pays devrait rester inactif.
    """
    from shapely.geometry import Point
    from shapely.ops import transform as shapely_transform
    from shapely import wkt as shapely_wkt

    if spatial_reference is None:
        raise ValueError("spatial_reference requis : aucun contexte spatial, aucun calcul")

    disagreements = check_spatial_agreement(manifest, spatial_reference)
    if disagreements:
        raise ValueError("; ".join(disagreements))

    settings = policy.visibility
    unsupported = engine.check_supported(settings)
    if unsupported:
        raise ValueError("; ".join(unsupported))

    target_shape, target_geometry = _target(manifest)
    if target_shape is None:
        raise ValueError("aucune empreinte cible résolue : rien à viser")

    obstacles = _obstacles(manifest, obstacle_heights or {})
    report = RunReport(run_id=run_id)
    report.parameters = {k: str(v) for k, v in settings.model_dump().items()}
    report.digests = dict(digests)

    projection = _projection_for(spatial_reference)

    # Toutes les positions caméra, pas seulement la cible : un asset hors du
    # fuseau produirait des mètres finis et faux.
    positions = [
        (a.camera_lat, a.camera_lon) for a in assets
        if a.camera_lat is not None and a.camera_lon is not None
    ]
    if projection is not None and positions:
        report.projection_check = projection.verify(positions, "positions caméra")

    visibility = VisibilityRun(
        run_id=run_id,
        hotel_id=hotel_id,
        engine_version=engine.ENGINE_VERSION,
        method=settings.sampling_method,
        parameters=report.parameters,
        capture_geometry_digest=digests["capture_geometry"],
        policy_digest=digests["policy"],
        site_manifest_digest=digests["site_manifest"],
        asset_files_digest=digests["asset_files"],
        asset_manifest_digest=digests["asset_manifest"],
        target_digest=target_geometry.geometry_digest,
        obstacles_digest=digests["obstacles"],
        road_geometry_digest=digests["roads"],
        spatial_context_digest=manifest.spatial_context_digest,
        crs=manifest.working_crs,
        elevation_sources=list(elevation_sources or []),
    )

    # --- assets ---------------------------------------------------------------
    for asset in assets:
        if asset.camera_lat is None or asset.camera_lon is None:
            continue
        origin = _project(projection, asset.camera_lat, asset.camera_lon)
        ground = camera_ground(origin) if camera_ground else None
        camera = engine.CameraVertical(
            ground_m=ground.value_m if ground else None,
            # Aucune hauteur d'œil n'est supposée : ni Mapillary ni Street View
            # ne publient celle de leur capteur, et l'inventer transformerait
            # un risque en preuve.
            height_above_ground_m=None,
            provenance=ground.provenance if ground else None,
        )
        assessment = engine.assess(
            f"vis-{asset.id}", asset.id, "BUILDING_MAIN", origin, target_shape,
            obstacles, settings, camera=camera, target_vertical=target_vertical,
            vertical=getattr(spatial_reference, "vertical", None),
            crs=manifest.working_crs,
        )
        visibility.assessments.append(assessment)
        _tally(report, asset, assessment)
        framing = _framing(asset, assessment, settings)
        visibility.framings.append(framing)
        if framing.horizontal_computable:
            report.framing_computable += 1
        else:
            reason = (framing.horizontal_reason or "").split(" :")[0]
            report.framing_not_computable[reason] = (
                report.framing_not_computable.get(reason, 0) + 1
            )

    # --- corridors -------------------------------------------------------------
    by_feature = {g.feature_id: g for g in manifest.geometries}
    for corridor in manifest.corridors:
        geometry = by_feature.get(corridor.feature_id)
        if geometry is None or geometry.resolution_status is not GeometryResolutionStatus.RESOLVED:
            continue
        line = shapely_wkt.loads(geometry.projected_wkt)
        visibility.corridors.append(
            _corridor(
                corridor, line, target_shape, obstacles, settings, report,
                sectors=_sector_reader(target_shape, front_azimuth_deg),
                crs=manifest.working_crs,
            )
        )

    report.corridors_assessed = len(visibility.corridors)
    for corridor in visibility.corridors:
        key = corridor.geometrically_useful.value
        report.corridors_useful[key] = report.corridors_useful.get(key, 0) + 1

    return visibility, report


def check_spatial_agreement(manifest, spatial_reference) -> list[str]:  # noqa: ANN001
    """Le manifeste et le contexte courant décrivent-ils le même espace ?

    Trois égalités, et non une seule. Sans elles, les caméras pouvaient être
    projetées dans le référentiel du contexte pendant que la cible et les
    obstacles restaient stockés dans celui du manifeste : des distances en
    mètres, finies, et fausses de milliers de kilomètres.

    ```text
    contexte courant  = empreinte citée par le manifeste
                      = CRS du manifeste
                      = CRS de chaque géométrie résolue
    ```
    """
    problems: list[str] = []
    current = spatial_reference.context_digest()

    if manifest.spatial_context_digest != current:
        problems.append(
            f"le manifeste géométrique cite le contexte spatial "
            f"{manifest.spatial_context_digest!r}, le contexte courant est "
            f"{current!r} : le référentiel ou le territoire a changé depuis, "
            "relancez « geo resolve »"
        )

    if manifest.working_crs != spatial_reference.working_crs:
        problems.append(
            f"le manifeste est en {manifest.working_crs!r}, le contexte courant "
            f"en {spatial_reference.working_crs!r}"
        )

    mismatched = sorted(
        geometry.feature_id
        for geometry in manifest.geometries
        if geometry.resolution_status is GeometryResolutionStatus.RESOLVED
        and geometry.projected_crs != spatial_reference.working_crs
    )
    if mismatched:
        problems.append(
            f"{len(mismatched)} géométrie(s) hors du référentiel courant "
            f"{spatial_reference.working_crs!r} : {mismatched[:5]}"
        )
    return problems


def _projection_for(spatial_reference):  # noqa: ANN001, ANN201
    """Service de projection du site."""
    if spatial_reference is None:
        raise ValueError("spatial_reference requis : aucun contexte spatial, aucun calcul")

    from .projection import ProjectionService

    return ProjectionService(spatial_reference)


def _project(projection, lat: float, lon: float) -> tuple[float, float]:  # noqa: ANN001
    return projection.point(lat, lon)


def _sector_reader(target_shape, front_azimuth_deg: float | None):  # noqa: ANN001
    """Secteur du bâtiment vu depuis une position, si la façade avant est connue.

    Sans azimut de façade, aucun secteur n'est nommé : les inventer ferait
    croire à une couverture orientée qui n'a pas été établie.
    """
    if front_azimuth_deg is None:
        return None

    from ..sectors import sector_for

    centre = target_shape.centroid

    def read(position) -> str:  # noqa: ANN001
        bearing = engine.bearing_between((centre.x, centre.y), position)
        return sector_for(bearing, front_azimuth_deg).value

    return read


def _tally(report: RunReport, asset, assessment) -> None:  # noqa: ANN001
    report.assets_assessed += 1
    status = assessment.status.value
    report.by_status[status] = report.by_status.get(status, 0) + 1

    source = report.by_source.setdefault(asset.source, {})
    source[status] = source.get(status, 0) + 1
    sector = report.by_sector.setdefault(asset.view_sector.value, {})
    sector[status] = sector.get(status, 0) + 1

    if assessment.status is LineOfSightStatus.CLEAR:
        report.clear += 1
    if assessment.proven_blocked_fraction > 0:
        report.proven_blocked += 1
    if assessment.status is LineOfSightStatus.AT_RISK:
        report.at_risk_only += 1
    if assessment.risk_unknown_height_fraction > 0:
        report.with_risk_fraction += 1

    if asset.occluded_by:
        report.previously_occluded += 1
        report.previously_occluded_now[status] = (
            report.previously_occluded_now.get(status, 0) + 1
        )

    for missing in assessment.missing_vertical:
        report.missing_vertical_counts[missing] = (
            report.missing_vertical_counts.get(missing, 0) + 1
        )
    for obstacle in assessment.obstacles_at_risk:
        report.obstacles_by_affected_assets[obstacle] = (
            report.obstacles_by_affected_assets.get(obstacle, 0) + 1
        )


def _framing(asset, assessment, settings):  # noqa: ANN001
    """Cadrage d'un asset, si ses paramètres le permettent.

    Street View rend un panorama : le cadrage demandé est connu. Mapillary rend
    une image dont les intrinsèques ne figurent pas au manifeste — la largeur y
    reste donc inconnue, faute de champ de vision.
    """
    source = None
    fov = None
    if asset.acquisition is not None:
        fov = asset.acquisition.requested_fov_deg
        source = "acquisition_provenance"

    return engine.frame_target(
        f"vis-{asset.id}", asset.id, assessment.span_start_deg,
        assessment.angular_span_deg, asset.heading_deg, fov,
        asset.width, asset.height, source, settings,
        reason_if_absent=(
            "intrinsèques absentes du manifeste"
            if asset.source == "mapillary"
            else "cadrage demandé non conservé"
        ),
    )


def _corridor(  # noqa: ANN001
    corridor, line, target_shape, obstacles, settings, report,
    sectors=None, crs: str = "",
):
    """Ce qu'une voie promet, échantillon par échantillon.

    Les mesures publiées sont celles d'**un** échantillon — le meilleur —, non
    le maximum de chaque grandeur prise séparément : celui-ci décrirait un
    emplacement qui n'existe pas.
    """
    from ..sectors import sector_for

    samples = engine.sample_line(line, settings.corridor_sample_step_m)
    clear_indices: list[int] = []
    potential_indices: list[int] = []
    at_risk: set[str] = set()
    observable: set[str] = set()
    best = None
    best_score = -1.0
    max_span = 0.0

    for index, (sample_id, position) in enumerate(samples):
        assessment = engine.assess(
            f"corridor-{corridor.corridor_id}-{sample_id}", corridor.corridor_id,
            "BUILDING_MAIN", position, target_shape, obstacles, settings,
            # Le référentiel du manifeste, comme les autres appels : une mesure
            # sans CRS ne se rattache à rien, et le rendre facultatif ici
            # laisserait ce chemin le deviner.
            crs=crs,
        )
        at_risk.update(assessment.obstacles_at_risk)
        max_span = max(max_span, assessment.angular_span_deg or 0.0)

        if assessment.proven_clear_fraction > 0:
            clear_indices.append(index)
        elif assessment.risk_unknown_height_fraction > 0:
            potential_indices.append(index)

        # Le secteur observé depuis cet emplacement : c'est ce qui dit quelle
        # façade la voie peut servir, et non « quelque chose du bâtiment ».
        if sectors is not None and assessment.proven_clear_fraction > 0:
            observable.add(sectors(position))

        # Un même échantillon porte toutes ses mesures.
        score = assessment.proven_clear_fraction
        if score > best_score:
            best_score = score
            best = (sample_id, assessment)

    verdict = UsefulnessVerdict.NOT_USEFUL
    if clear_indices:
        verdict = UsefulnessVerdict.USEFUL
    elif at_risk or not samples:
        # Avec des obstacles tous de hauteur inconnue, conclure « pas utile »
        # serait aussi injustifié que conclure « utile ».
        verdict = UsefulnessVerdict.UNKNOWN

    sample_id, chosen = best if best else (None, None)
    assessment = CorridorVisibilityAssessment(
        assessment_id=f"corridor-{corridor.corridor_id}",
        corridor_id=corridor.corridor_id,
        feature_id=corridor.feature_id,
        samples=len(samples),
        sample_step_m=settings.corridor_sample_step_m,
        proven_clear_segments=engine.group_segments(clear_indices),
        potential_segments=engine.group_segments(potential_indices),
        best_sample_ids=[sample_id] if sample_id else [],
        best_sample_id=sample_id,
        best_clear_fraction=chosen.proven_clear_fraction if chosen else None,
        best_risk_fraction=chosen.risk_unknown_height_fraction if chosen else None,
        best_angular_span_deg=chosen.angular_span_deg if chosen else None,
        best_distance_m=chosen.distance_m if chosen else None,
        max_angular_span_deg=round(max_span, settings.output_precision),
        observable_sectors=sorted(observable),
        obstacles_at_risk=sorted(at_risk),
        geometrically_useful=verdict,
        access_status=corridor.access_status.value,
        rationale=(
            f"{len(samples)} échantillon(s) ; "
            f"{engine.group_segments(clear_indices)} segment(s) dégagé(s), "
            f"{engine.group_segments(potential_indices)} à risque ; "
            f"meilleur emplacement {sample_id or '—'}"
        ),
    )
    if verdict is UsefulnessVerdict.USEFUL:
        report.useful_corridor_details.append(
            {
                "corridor_id": corridor.corridor_id,
                "class": corridor.corridor_class.value,
                "access_status": corridor.access_status.value,
                "best_sample_id": sample_id,
                "best_clear_fraction": assessment.best_clear_fraction,
                "best_angular_span_deg": assessment.best_angular_span_deg,
                "best_distance_m": assessment.best_distance_m,
                "observable_sectors": assessment.observable_sectors,
                "proven_clear_segments": assessment.proven_clear_segments,
            }
        )
    return assessment
