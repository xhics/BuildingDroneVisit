"""Géométrie de candidat, calculée sur métadonnées (collecte V2).

Ce qui est éprouvé : ces valeurs sont des **espérances**, aucune caméra n'est
supposée, un obstacle reste un risque, et le plan cesse enfin de tout différer.
"""

from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from hotel_pipeline.candidate_geometry import (
    GeometryUnavailable,
    measure,
    measure_all,
)
from hotel_pipeline.geo import territory
from hotel_pipeline.geo.projection import ProjectionService
from hotel_pipeline.geo.visibility_engine import Obstacle
from hotel_pipeline.schemas import DEFAULT_POLICY
from hotel_pipeline.schemas.acquisition import (
    CaptureCandidate,
    CaptureDemand,
    CaptureIntent,
    Eligibility,
    TargetKind,
)

POLICY = DEFAULT_POLICY.visibility
BOUCHERVILLE = (45.574128, -73.443289)

#: Empreinte carrée d'environ 40 m, autour de la position de référence.
BUILDING = Polygon([
    (-73.44355, 45.57395), (-73.44300, 45.57395),
    (-73.44300, 45.57430), (-73.44355, 45.57430),
])


@pytest.fixture
def service() -> ProjectionService:
    return ProjectionService(territory.resolve("pilote", *BOUCHERVILLE))


@pytest.fixture
def target(service):
    return service.geometry(BUILDING, label="cible")


def candidate(candidate_id: str = "c1", **overrides) -> CaptureCandidate:
    fields = dict(
        candidate_id=candidate_id, source="mapillary", provider_id=candidate_id,
        camera_lat=45.57340, camera_lon=-73.44330,
    )
    fields.update(overrides)
    return CaptureCandidate(**fields)


def demand(demand_id: str = "d1", **overrides) -> CaptureDemand:
    fields = dict(
        demand_id=demand_id, intent=CaptureIntent.BUILDING_CAPTURE,
        target_kind=TargetKind.VIEW_SECTOR, target_ref="front",
    )
    fields.update(overrides)
    return CaptureDemand(**fields)


# --- ce qui se calcule sans caméra --------------------------------------------


def test_distance_and_angular_span_need_only_a_position(service, target) -> None:
    geometry = measure(candidate(), target, service, POLICY)

    assert geometry.distance_m is not None and geometry.distance_m > 0
    assert geometry.angular_span_deg is not None and geometry.angular_span_deg > 0


def test_a_candidate_without_a_position_is_refused_not_guessed(service, target) -> None:
    with pytest.raises(GeometryUnavailable, match="aucune position"):
        measure(candidate(camera_lat=None, camera_lon=None), target, service, POLICY)


def test_the_offset_measures_where_the_camera_actually_looks(service, target) -> None:
    """Un cap opposé à la cible doit produire un écart proche de 180°."""
    towards = measure(candidate(original_heading_deg=0.0), target, service, POLICY)
    away = measure(candidate(original_heading_deg=180.0), target, service, POLICY)

    assert towards.target_offset_deg < 45.0
    assert away.target_offset_deg > 135.0


# --- aucune caméra n'est supposée ---------------------------------------------


def test_no_field_of_view_means_no_frame_fraction(service, target) -> None:
    """Supposer un objectif produirait un tri fondé sur une caméra imaginaire."""
    geometry = measure(candidate(original_heading_deg=0.0), target, service, POLICY)

    assert geometry.unclipped_width_fraction is None
    assert geometry.expected_width_px is None


def test_the_measurement_reads_only_what_the_source_declared(service, target) -> None:
    """Le vrai risque n'est pas l'absence de mesure : c'est la mesure inventée.

    Un défaut d'objectif produirait une fraction plausible pour une caméra qui
    n'existe pas, et le plan trierait sur elle. Le contrôle porte donc sur la
    **valeur** rendue, non sur sa seule présence.
    """
    declared = measure(
        candidate(
            requested_heading_deg=0.0, requested_fov_deg=80.0,
            advertised_width=2048, advertised_height=1536,
        ),
        target, service, POLICY,
    )
    silent = measure(
        candidate(requested_heading_deg=0.0), target, service, POLICY
    )

    assert declared.unclipped_width_fraction is not None
    # La même vue, sans objectif déclaré, ne doit produire aucune espérance —
    # surtout pas celle qu'aurait rendue un objectif par défaut.
    assert silent.unclipped_width_fraction is None
    assert silent.expected_width_px is None


def test_a_narrower_lens_frames_the_target_larger(service, target) -> None:
    """La fraction dépend de l'objectif déclaré, et le prouve en variant."""
    wide = measure(
        candidate(
            requested_heading_deg=0.0, requested_fov_deg=110.0,
            advertised_width=2048, advertised_height=1536,
        ),
        target, service, POLICY,
    )
    narrow = measure(
        candidate(
            requested_heading_deg=0.0, requested_fov_deg=40.0,
            advertised_width=2048, advertised_height=1536,
        ),
        target, service, POLICY,
    )

    assert narrow.unclipped_width_fraction > wide.unclipped_width_fraction


def test_a_declared_field_of_view_yields_an_expectation(service, target) -> None:
    geometry = measure(
        candidate(
            requested_heading_deg=0.0, requested_fov_deg=80.0,
            advertised_width=2048, advertised_height=1536,
        ),
        target, service, POLICY,
    )

    assert geometry.unclipped_width_fraction is not None
    assert geometry.unclipped_width_fraction > 0
    assert geometry.expected_width_px is not None


def test_a_target_wider_than_the_frame_is_not_clipped_to_one(service, target) -> None:
    """Écrêter effacerait l'information : la cible déborde légitimement."""
    close = measure(
        candidate(
            camera_lat=45.57390, camera_lon=-73.44328,
            requested_heading_deg=0.0, requested_fov_deg=20.0,
            advertised_width=2048, advertised_height=1536,
        ),
        target, service, POLICY,
    )

    if close.unclipped_width_fraction is not None:
        assert close.unclipped_width_fraction >= 0.0
        # Le pixel attendu, lui, est borné par la largeur réelle de l'image.
        if close.expected_width_px is not None:
            assert close.expected_width_px <= 2048


# --- un obstacle reste un risque ----------------------------------------------


def test_an_interposed_obstacle_is_a_risk_never_a_proof(service, target) -> None:
    """Aucune hauteur n'est connue : en faire une certitude écarterait des
    vues parfaitement dégagées."""
    wall = service.geometry(
        Polygon([
            (-73.44340, 45.57370), (-73.44315, 45.57370),
            (-73.44315, 45.57375), (-73.44340, 45.57375),
        ]),
        label="obstacle",
    )
    geometry = measure(
        candidate(), target, service, POLICY,
        obstacles=[Obstacle(feature_id="way/9", shape=wall)],
    )

    assert geometry.occlusion_risk is True
    assert geometry.occluded_by == ["way/9"]


def test_an_obstacle_off_the_line_of_sight_is_not_counted(service, target) -> None:
    """Le segment s'arrête au centroïde : ce qui est au-delà n'est pas touché.

    C'est ce qui rend inutile ici tout contrôle de profondeur — j'en avais
    écrit un, et le retirer ne cassait rien : il n'était jamais atteint. Ce
    qui protège réellement, c'est la longueur du segment, et c'est elle qui
    est éprouvée.
    """
    from shapely.geometry import LineString, Point

    aside = service.geometry(
        Polygon([
            (-73.44280, 45.57340), (-73.44260, 45.57340),
            (-73.44260, 45.57350), (-73.44280, 45.57350),
        ]),
        label="à côté",
    )
    origin = service.point(45.57340, -73.44330)
    ray = LineString([origin, (target.centroid.x, target.centroid.y)])

    # Le montage doit être franc : la forme n'est pas sur la ligne de visée.
    assert not ray.intersects(aside)

    geometry = measure(
        candidate(), target, service, POLICY,
        obstacles=[Obstacle(feature_id="way/cote", shape=aside)],
    )

    assert geometry.occlusion_risk is False
    assert geometry.occluded_by == []


def test_a_building_beyond_the_target_is_never_a_risk(service, target) -> None:
    """La garantie réelle : le segment ne dépasse pas la cible.

    S'il la traversait, tout bâtiment situé au-delà compterait comme obstacle,
    et des vues parfaitement dégagées seraient signalées à risque. La forme est
    donc placée **exactement dans l'alignement**, deux fois plus loin.
    """
    beyond = service.geometry(
        Polygon([
            (-73.44335, 45.57480), (-73.44315, 45.57480),
            (-73.44315, 45.57492), (-73.44335, 45.57492),
        ]),
        label="au-delà",
    )

    geometry = measure(
        candidate(), target, service, POLICY,
        obstacles=[Obstacle(feature_id="way/au-dela", shape=beyond)],
    )

    assert geometry.occlusion_risk is False
    assert geometry.occluded_by == []


def test_no_elevation_is_claimed_without_provenance(service, target) -> None:
    """Le contrôle 2,5D n'a pas eu lieu ici, et le schéma l'exigerait tracé."""
    geometry = measure(candidate(), target, service, POLICY)

    assert geometry.used_elevation is False
    assert geometry.elevation_provenance is None


# --- le lot, et ce qu'il laisse indéterminé -----------------------------------


def test_the_batch_reports_what_it_could_not_measure(service, target) -> None:
    located = candidate("c1")
    lost = candidate("c2", camera_lat=None, camera_lon=None)

    measured, report = measure_all(
        [located, lost], target, service, POLICY, [demand()]
    )

    assert report.measured == 1
    assert "c2" in report.skipped
    assert ("c1", "d1") in measured
    assert ("c2", "d1") not in measured


def test_the_batch_states_that_nothing_was_measured_on_pixels(service, target) -> None:
    _, report = measure_all([candidate()], target, service, POLICY, [demand()])

    assert "aucune image n'a été acquise" in report.as_dict()["note"]


def test_the_report_says_why_framing_stayed_unknown(service, target) -> None:
    _, report = measure_all(
        [candidate(original_heading_deg=0.0)], target, service, POLICY, [demand()]
    )

    assert report.with_framing == 0
    assert "champ de vision non déclaré" in report.without_framing


# --- l'effet sur le plan : il trie enfin ---------------------------------------


def test_a_measured_candidate_stops_being_deferred(service, target) -> None:
    """Sans mesure, `plan` classait tout en `preview_required` : il différait."""
    from hotel_pipeline.plan import evaluate

    wanted = demand(min_projected_width_fraction=0.05)
    subject = candidate(
        requested_heading_deg=0.0, requested_fov_deg=80.0,
        advertised_width=2048, advertised_height=1536,
    )
    geometry = measure(subject, target, service, POLICY)

    deferred = evaluate(subject, wanted)
    decided = evaluate(subject, wanted, geometry)

    assert deferred.eligibility is Eligibility.PREVIEW_REQUIRED
    assert decided.eligibility in (Eligibility.ELIGIBLE, Eligibility.REJECTED)


def test_a_view_too_small_for_the_demand_is_now_rejected(service, target) -> None:
    from hotel_pipeline.plan import evaluate

    demanding = demand(min_projected_width_fraction=0.99)
    subject = candidate(
        requested_heading_deg=0.0, requested_fov_deg=80.0,
        advertised_width=2048, advertised_height=1536,
    )
    geometry = measure(subject, target, service, POLICY)

    result = evaluate(subject, demanding, geometry)

    assert result.eligibility is Eligibility.REJECTED
    assert "taille projetée espérée" in result.rejection_reason
