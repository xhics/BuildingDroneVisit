"""Recherche adaptative : chercher là où il manque (collecte V2).

Ce qui est éprouvé : les qualités opposées restent séparées, l'inconnu reste
inconnu, et le bâtiment ne devient jamais une ancre.
"""

from __future__ import annotations

import pytest

from hotel_pipeline.adaptive_search import (
    ContinuityLevel,
    GeometryAnchor,
    SearchReport,
    SequenceStatus,
    anchors_for,
    coverage_gap_priority,
    measure_candidate,
    open_demands,
    pairwise_parallax_potential,
    select_for_demand,
)
from hotel_pipeline.schemas import (
    DEFAULT_POLICY,
    Asset,
    ClusterRole,
    GeometrySuitability,
    ReconstructionRole,
    ViewSector,
)
from hotel_pipeline.schemas.acquisition import (
    CaptureCandidate,
    CaptureDemand,
    CaptureIntent,
    DemandAssessment,
    DemandAssessmentManifest,
    DemandStatus,
    TargetKind,
)

#: Cible : le bâtiment du pilote.
TARGET = (45.5741, -73.4433)

SEARCH = DEFAULT_POLICY.adaptive_search


def demand(demand_id: str = "obligation:FACADE_REAR", ref: str = "rear", **overrides):
    fields = dict(
        demand_id=demand_id, intent=CaptureIntent.BUILDING_CAPTURE,
        target_kind=TargetKind.VIEW_SECTOR, target_ref=ref,
        viewpoints_required=2,
    )
    fields.update(overrides)
    return CaptureDemand(**fields)


def candidate(candidate_id: str, lat: float, lon: float, **overrides):
    fields = dict(
        candidate_id=candidate_id, source="mapillary", provider_id=candidate_id,
        camera_lat=lat, camera_lon=lon, heading_is_measured=True,
    )
    fields.update(overrides)
    return CaptureCandidate(**fields)


def asset(asset_id: str, lat: float, lon: float, sector=ViewSector.RIGHT, **overrides):
    from hotel_pipeline.review import assessment_fields

    fields = dict(
        id=asset_id, source="mapillary", source_url_or_id=asset_id,
        rights="open_data", ai_eligible=False, confidence=0.9, category="facade",
        checksum="a" * 64, cluster_role=ClusterRole.CANONICAL,
        reconstruction_role=ReconstructionRole.PHOTO_GEOMETRY,
        target_building_visible=True, view_sector=sector,
        camera_lat=lat, camera_lon=lon,
    )
    fields.update(
        assessment_fields(
            GeometrySuitability.PRIMARY, "hm", "façade", ["contrôle"], "a" * 64
        )
    )
    fields.update(overrides)
    return Asset(**fields)


def assessment(**statuses) -> DemandAssessmentManifest:
    return DemandAssessmentManifest(
        hotel_id="h", corpus_digest="c0", demand_digest="d0",
        assessments=[
            DemandAssessment(
                demand_id=demand_id, corpus_digest="c0",
                status=status, viewpoints_found=found,
            )
            for demand_id, (status, found) in statuses.items()
        ],
    )


# --- un besoin satisfait n'est pas recherché ----------------------------------


def test_a_satisfied_demand_is_not_searched() -> None:
    served = demand("obligation:FACADE_RIGHT", "right")
    missing = demand("obligation:FACADE_REAR", "rear")

    still_open = open_demands(
        assessment(**{
            "obligation:FACADE_RIGHT": (DemandStatus.MET, 2),
            "obligation:FACADE_REAR": (DemandStatus.OPEN, 0),
        }),
        [served, missing],
    )

    assert [d.demand_id for d in still_open] == ["obligation:FACADE_REAR"]


def test_an_empty_sector_outranks_a_repetition() -> None:
    """Un secteur sans aucune vue passe avant un secteur perfectible."""
    empty = demand("obligation:FACADE_REAR", "rear")
    partial = demand("obligation:FACADE_RIGHT", "right")

    state = assessment(**{
        "obligation:FACADE_REAR": (DemandStatus.OPEN, 0),
        "obligation:FACADE_RIGHT": (DemandStatus.PARTIALLY_MET, 1),
    })

    assert coverage_gap_priority(empty, state) > coverage_gap_priority(partial, state)


# --- la façade droite n'est pas une ancre pour l'avant ------------------------


def test_a_right_facade_view_anchors_only_its_own_sector() -> None:
    """La confondre ferait chercher de la parallaxe autour d'un point qui ne
    voit pas la cible demandée."""
    right_view = asset("a1", 45.5735, -73.4425, sector=ViewSector.RIGHT)
    viewpoints = {"a1": "vp-1"}
    sectors = {"a1": "right"}

    for_right = anchors_for(
        demand("obligation:FACADE_RIGHT", "right"), [right_view], viewpoints, sectors
    )
    for_front = anchors_for(
        demand("obligation:FACADE_PRIMARY", "front"), [right_view], viewpoints, sectors
    )
    for_rear = anchors_for(
        demand("obligation:FACADE_REAR", "rear"), [right_view], viewpoints, sectors
    )

    assert [a.viewpoint_id for a in for_right] == ["vp-1"]
    assert for_front == []
    assert for_rear == []


def test_only_usable_assets_become_anchors() -> None:
    locked = asset(
        "a1", 45.5735, -73.4425,
        reconstruction_role=ReconstructionRole.CONTEXT_LOCK,
    )

    assert anchors_for(demand(ref="right"), [locked], {}, {}) == []


# --- sans ancre : parallaxe None, jamais le bâtiment ---------------------------


def test_no_anchor_means_no_parallax_never_the_building() -> None:
    """Le bâtiment est une cible géométrique, pas une position caméra."""
    measure = measure_candidate(
        candidate("c1", 45.5748, -73.4433), demand(), anchors=[], priority=3,
        target_lat=TARGET[0], target_lon=TARGET[1], policy=SEARCH,
    )

    assert measure.parallax_utility is None
    assert "aucune ancre géométrique" in measure.parallax_reason
    assert "jamais une position caméra" in measure.parallax_reason
    # La priorité vient alors du manque de couverture.
    assert measure.coverage_gap_priority == 3
    assert measure.sector_novelty is True


def test_an_anchor_yields_raw_measures_and_a_policy_score() -> None:
    """Les deux sont publiés, et ne se confondent pas."""
    anchor = GeometryAnchor("vp-1", "obligation:FACADE_RIGHT", 45.5735, -73.4425, "primary")

    measure = measure_candidate(
        candidate("c1", 45.5745, -73.4440), demand(), [anchor], 2,
        target_lat=TARGET[0], target_lon=TARGET[1], policy=SEARCH,
    )
    published = measure.as_dict()["parallax"]

    # Mesures géométriques : elles survivent à tout changement de préférence.
    assert published["bearing_delta_deg"] is not None
    assert published["baseline_m"] is not None
    # Préférence : étiquetée comme telle, jamais confondue avec une mesure.
    assert published["utility_policy_score"] is not None


def test_the_preference_is_not_monotonic() -> None:
    """Le défaut corrigé : `delta / 90` préférait 100° à 25°.

    À grand écart, deux vues montrent des surfaces très différentes et ne
    partagent plus assez de recouvrement pour qu'un appariement fonctionne.
    C'est de la diversité, pas de la parallaxe exploitable.
    """
    from hotel_pipeline.adaptive_search import parallax_utility

    too_small = parallax_utility(4.0, 30.0, SEARCH)
    preferred = parallax_utility(30.0, 30.0, SEARCH)
    excessive = parallax_utility(110.0, 30.0, SEARCH)

    assert preferred > too_small
    assert preferred > excessive


def test_a_small_angle_loses_against_the_preferred_range() -> None:
    from hotel_pipeline.adaptive_search import parallax_utility

    assert parallax_utility(5.0, 30.0, SEARCH) < parallax_utility(25.0, 30.0, SEARCH)


def test_an_excessive_angle_does_not_win_automatically() -> None:
    """Un candidat à 25° peut être préféré à un candidat à 100°."""
    from hotel_pipeline.adaptive_search import parallax_utility

    assert parallax_utility(25.0, 30.0, SEARCH) > parallax_utility(100.0, 30.0, SEARCH)


def test_a_baseline_too_short_yields_no_utility() -> None:
    """Un écart mesuré sur deux positions confondues n'est pas de la parallaxe."""
    from hotel_pipeline.adaptive_search import parallax_utility

    assert parallax_utility(30.0, 0.5, SEARCH) == 0.0
    assert parallax_utility(30.0, 30.0, SEARCH) == 1.0


def test_changing_the_policy_changes_the_preference() -> None:
    from hotel_pipeline.adaptive_search import parallax_utility
    from hotel_pipeline.schemas.policy import AdaptiveSearchPolicy

    wide = AdaptiveSearchPolicy(
        parallax_preferred_min_deg=60.0, parallax_preferred_max_deg=120.0
    )

    assert parallax_utility(100.0, 30.0, SEARCH) < 1.0
    assert parallax_utility(100.0, 30.0, wide) == 1.0


def test_raw_measures_survive_a_change_of_preference() -> None:
    """Un rapport doit rester lisible après un changement de politique."""
    from hotel_pipeline.schemas.policy import AdaptiveSearchPolicy

    anchor = GeometryAnchor("vp-1", "d", 45.5735, -73.4425, "primary")
    subject = candidate("c1", 45.5745, -73.4440)
    wide = AdaptiveSearchPolicy(
        parallax_preferred_min_deg=60.0, parallax_preferred_max_deg=120.0
    )

    strict = measure_candidate(
        subject, demand(), [anchor], 2, target_lat=TARGET[0],
        target_lon=TARGET[1], policy=SEARCH,
    )
    lenient = measure_candidate(
        subject, demand(), [anchor], 2, target_lat=TARGET[0],
        target_lon=TARGET[1], policy=wide,
    )

    assert strict.bearing_delta_from_anchor_deg == lenient.bearing_delta_from_anchor_deg
    assert strict.baseline_to_anchor_m == lenient.baseline_to_anchor_m
    assert strict.parallax_utility != lenient.parallax_utility


def test_no_angular_threshold_is_written_in_the_module() -> None:
    """Une constante bien testée reste une constante arbitraire."""
    import ast
    import pathlib

    tree = ast.parse(
        pathlib.Path("src/hotel_pipeline/adaptive_search.py").read_text("utf-8")
    )
    numbers = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    }

    # Les rangs de priorité (0–4) et l'arithmétique d'azimut (180, 360) sont
    # légitimes. Un **seuil angulaire** ne l'est pas : ni 90, ni 45, ni 15.
    thresholds = {
        value for value in numbers
        if isinstance(value, (int, float)) and 5 <= value < 180
    }

    assert thresholds == set(), f"seuils angulaires codés en dur : {sorted(thresholds)}"


# --- la séquence : inconnue tant qu'on n'a pas demandé ------------------------


def test_an_unqueried_sequence_gives_none_never_zero() -> None:
    """Zéro dirait « mesuré, et nul » — une affirmation qu'on n'a pas."""
    measure = measure_candidate(
        candidate("c1", 45.5748, -73.4433), demand(), [], 3,
        target_lat=TARGET[0], target_lon=TARGET[1], policy=SEARCH,
    )

    assert measure.continuity_gain is None
    assert measure.sequence_status is SequenceStatus.NOT_QUERIED
    assert "séquence non interrogée" in measure.continuity_reason


def test_a_failed_enrichment_does_not_disqualify_a_promising_view() -> None:
    """Elle interdit d'attribuer un gain de continuité, rien de plus."""
    anchor = GeometryAnchor("vp-1", "d", 45.5735, -73.4425, "primary")

    measure = measure_candidate(
        candidate("c1", 45.5748, -73.4441), demand(), [anchor], 2,
        target_lat=TARGET[0], target_lon=TARGET[1], policy=SEARCH,
        sequence_status=SequenceStatus.QUERY_ERROR,
    )

    assert measure.continuity_gain is None
    assert "indisponible" in measure.continuity_reason
    # La géométrie, elle, reste mesurée.
    assert measure.parallax_utility is not None
    assert measure.rejection_reason is None


def test_a_sequence_neighbour_gains_continuity_without_parallax() -> None:
    """Le cas décisif : proche d'une ancre, donc mauvaise parallaxe, mais elle
    ferme un trou de continuité. La rejeter comme doublon perdrait le chaînage.
    """
    anchor = GeometryAnchor("vp-1", "d", 45.5735, -73.4425, "primary")

    measure = measure_candidate(
        candidate("c1", 45.57352, -73.44252), demand(), [anchor], 2,
        target_lat=TARGET[0], target_lon=TARGET[1], policy=SEARCH,
        sequence_of={"c1": "seq-A"}, anchor_sequences={"seq-A"},
        sequence_status=SequenceStatus.KNOWN,
    )

    assert measure.continuity_gain == 1.0
    assert measure.continuity_level is ContinuityLevel.POTENTIAL
    assert measure.parallax_utility < 0.2
    assert measure.rejection_reason is None


def test_belonging_to_a_sequence_is_never_proof_of_overlap() -> None:
    """Un véhicule tourne : la même séquence ne garantit aucun recouvrement."""
    measure = measure_candidate(
        candidate("c1", 45.5735, -73.4425), demand(), [], 3,
        target_lat=TARGET[0], target_lon=TARGET[1], policy=SEARCH,
        sequence_of={"c1": "seq-A"}, anchor_sequences={"seq-A"},
        sequence_status=SequenceStatus.KNOWN,
    )

    assert measure.continuity_level is ContinuityLevel.POTENTIAL
    assert measure.continuity_level is not ContinuityLevel.ACHIEVED
    assert "non mesuré" in measure.continuity_reason


def test_a_sequence_not_returned_is_distinct_from_an_error() -> None:
    absent = measure_candidate(
        candidate("c1", 45.5748, -73.4433), demand(), [], 3,
        sequence_status=SequenceStatus.NOT_RETURNED,
    )
    failed = measure_candidate(
        candidate("c2", 45.5748, -73.4433), demand(), [], 3,
        sequence_status=SequenceStatus.QUERY_ERROR,
    )

    assert "aucune séquence" in absent.continuity_reason
    assert "échec" in failed.continuity_reason


# --- deux nouveaux candidats : la diversité se mesure entre eux ---------------


def test_two_close_candidates_are_not_two_good_viewpoints() -> None:
    """L'absence d'ancre n'autorise pas à retenir deux vues quasi identiques."""
    first = candidate("c1", 45.5748, -73.4433)
    near = candidate("c2", 45.57481, -73.44331)
    far = candidate("c3", 45.5741, -73.4450)

    close_pair = pairwise_parallax_potential(first, near, *TARGET)
    spread_pair = pairwise_parallax_potential(first, far, *TARGET)

    assert close_pair["bearing_delta_deg"] < spread_pair["bearing_delta_deg"]
    assert close_pair["level"] == "potential"
    assert "aucune image n'ayant été inspectée" in close_pair["note"]


def test_the_second_view_is_chosen_by_marginal_gain() -> None:
    """Sans cela, deux vues côte à côte passeraient toutes deux."""
    candidates = {
        "c1": candidate("c1", 45.5748, -73.4433),
        "c2": candidate("c2", 45.57481, -73.44331),
        "c3": candidate("c3", 45.5741, -73.4450),
    }
    wanted = demand(viewpoints_required=2)
    measures = [
        measure_candidate(
            c, wanted, [], 3, target_lat=TARGET[0], target_lon=TARGET[1],
            policy=SEARCH,
        )
        for c in candidates.values()
    ]

    retained = select_for_demand(measures, candidates, wanted, *TARGET, wanted=2, policy=SEARCH)

    assert len(retained) == 2
    # La seconde retenue est celle qui apporte l'angle, non la voisine.
    assert "c3" in retained


def test_selection_is_independent_of_api_order() -> None:
    """Deux exécutions doivent rendre le même plan."""
    candidates = {
        f"c{i}": candidate(f"c{i}", 45.574 + i * 0.0006, -73.4433 - i * 0.0006)
        for i in range(4)
    }
    wanted = demand(viewpoints_required=2)
    measures = [
        measure_candidate(
            c, wanted, [], 3, target_lat=TARGET[0], target_lon=TARGET[1],
            policy=SEARCH,
        )
        for c in candidates.values()
    ]

    forward = select_for_demand(measures, candidates, wanted, *TARGET, wanted=2, policy=SEARCH)
    backward = select_for_demand(
        list(reversed(measures)), candidates, wanted, *TARGET, wanted=2,
        policy=SEARCH,
    )

    assert forward == backward


# --- ce qui est rejeté avant tout téléchargement ------------------------------


def test_a_candidate_without_a_position_is_rejected() -> None:
    measure = measure_candidate(
        candidate("c1", None, None), demand(), [], 3,
        target_lat=TARGET[0], target_lon=TARGET[1], policy=SEARCH,
    )

    assert measure.rejection_reason == "position de caméra inconnue"
    assert measure.distance_to_target_m is None


def test_an_unmeasured_heading_stays_unmeasured() -> None:
    """Aucune orientation n'est inventée : elle est déclarée, ou inconnue."""
    measure = measure_candidate(
        candidate("c1", 45.5748, -73.4433, heading_is_measured=False),
        demand(), [], 3,
    )

    assert measure.heading_is_measured is False


# --- le rapport distingue quatre issues ---------------------------------------


def test_the_report_separates_four_outcomes() -> None:
    """« Zéro » confondait source muette, source vide, panne et tout rejeté."""
    report = SearchReport(
        source_not_queried={"places": "non configurée"},
        zero_results=["street_view"],
        query_errors={"flickr": "503"},
        all_rejected={"mapillary": 47},
    )
    published = report.as_dict()["outcomes"]

    assert published["source_not_queried"] == {"places": "non configurée"}
    assert published["zero_results"] == ["street_view"]
    assert published["query_errors"] == {"flickr": "503"}
    assert published["all_candidates_rejected"] == {"mapillary": 47}


def test_the_report_states_what_a_potential_continuity_is_not() -> None:
    published = SearchReport().as_dict()

    assert "non qu'elles se recouvrent" in published["note"]
    assert published["bytes_downloaded"] == 0


def test_searching_downloads_nothing() -> None:
    measure = measure_candidate(
        candidate("c1", 45.5748, -73.4433), demand(), [], 3,
        target_lat=TARGET[0], target_lon=TARGET[1], policy=SEARCH,
    )
    report = SearchReport(measures=[measure])

    assert report.as_dict()["bytes_downloaded"] == 0


def test_the_selector_prefers_a_usable_angle_over_an_extreme_one() -> None:
    """Le sélecteur utilisait l'angle brut : il retenait le plus grand.

    Un candidat à 25° partage encore de la surface avec la première vue ; un
    candidat à 100° n'en partage peut-être aucune. Maximiser l'écart
    sélectionnait de la diversité en croyant sélectionner de la parallaxe.
    """
    from hotel_pipeline.adaptive_search import pairwise_parallax_potential

    first = candidate("c1", 45.5748, -73.4433)
    # Positions calculées, non devinées : 43,7° et 121,5° d'écart vers la cible.
    moderate = candidate("c2", 45.57465, -73.44405)
    extreme = candidate("c3", 45.5735, -73.4419)

    moderate_pair = pairwise_parallax_potential(first, moderate, *TARGET)
    extreme_pair = pairwise_parallax_potential(first, extreme, *TARGET)

    # Le montage doit être franc : l'un est dans la plage, l'autre au-delà.
    assert moderate_pair["bearing_delta_deg"] <= SEARCH.parallax_preferred_max_deg
    assert extreme_pair["bearing_delta_deg"] > SEARCH.parallax_preferred_max_deg

    candidates = {c.candidate_id: c for c in (first, moderate, extreme)}
    wanted = demand(viewpoints_required=2)
    measures = [
        measure_candidate(
            c, wanted, [], 3, target_lat=TARGET[0], target_lon=TARGET[1],
            policy=SEARCH,
        )
        for c in candidates.values()
    ]

    retained = select_for_demand(
        measures, candidates, wanted, *TARGET, wanted=2, policy=SEARCH
    )

    assert "c2" in retained
    assert "c3" not in retained
