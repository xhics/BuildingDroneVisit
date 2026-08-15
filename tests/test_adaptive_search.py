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
        target_lat=TARGET[0], target_lon=TARGET[1],
    )

    assert measure.parallax_gain is None
    assert "aucune ancre géométrique" in measure.parallax_reason
    assert "jamais une position caméra" in measure.parallax_reason
    # La priorité vient alors du manque de couverture.
    assert measure.coverage_gap_priority == 3
    assert measure.sector_novelty is True


def test_an_anchor_yields_a_measured_parallax() -> None:
    anchor = GeometryAnchor("vp-1", "obligation:FACADE_RIGHT", 45.5735, -73.4425, "primary")

    aligned = measure_candidate(
        candidate("c1", 45.5734, -73.4426), demand(), [anchor], 2,
        target_lat=TARGET[0], target_lon=TARGET[1],
    )
    opposite = measure_candidate(
        candidate("c2", 45.5748, -73.4441), demand(), [anchor], 2,
        target_lat=TARGET[0], target_lon=TARGET[1],
    )

    assert aligned.parallax_gain is not None
    assert opposite.parallax_gain > aligned.parallax_gain
    assert aligned.distance_to_nearest_anchor_m is not None


# --- la séquence : inconnue tant qu'on n'a pas demandé ------------------------


def test_an_unqueried_sequence_gives_none_never_zero() -> None:
    """Zéro dirait « mesuré, et nul » — une affirmation qu'on n'a pas."""
    measure = measure_candidate(
        candidate("c1", 45.5748, -73.4433), demand(), [], 3,
        target_lat=TARGET[0], target_lon=TARGET[1],
    )

    assert measure.continuity_gain is None
    assert measure.sequence_status is SequenceStatus.NOT_QUERIED
    assert "séquence non interrogée" in measure.continuity_reason


def test_a_failed_enrichment_does_not_disqualify_a_promising_view() -> None:
    """Elle interdit d'attribuer un gain de continuité, rien de plus."""
    anchor = GeometryAnchor("vp-1", "d", 45.5735, -73.4425, "primary")

    measure = measure_candidate(
        candidate("c1", 45.5748, -73.4441), demand(), [anchor], 2,
        target_lat=TARGET[0], target_lon=TARGET[1],
        sequence_status=SequenceStatus.QUERY_ERROR,
    )

    assert measure.continuity_gain is None
    assert "indisponible" in measure.continuity_reason
    # La géométrie, elle, reste mesurée.
    assert measure.parallax_gain is not None
    assert measure.rejection_reason is None


def test_a_sequence_neighbour_gains_continuity_without_parallax() -> None:
    """Le cas décisif : proche d'une ancre, donc mauvaise parallaxe, mais elle
    ferme un trou de continuité. La rejeter comme doublon perdrait le chaînage.
    """
    anchor = GeometryAnchor("vp-1", "d", 45.5735, -73.4425, "primary")

    measure = measure_candidate(
        candidate("c1", 45.57352, -73.44252), demand(), [anchor], 2,
        target_lat=TARGET[0], target_lon=TARGET[1],
        sequence_of={"c1": "seq-A"}, anchor_sequences={"seq-A"},
        sequence_status=SequenceStatus.KNOWN,
    )

    assert measure.continuity_gain == 1.0
    assert measure.continuity_level is ContinuityLevel.POTENTIAL
    assert measure.parallax_gain < 0.2
    assert measure.rejection_reason is None


def test_belonging_to_a_sequence_is_never_proof_of_overlap() -> None:
    """Un véhicule tourne : la même séquence ne garantit aucun recouvrement."""
    measure = measure_candidate(
        candidate("c1", 45.5735, -73.4425), demand(), [], 3,
        target_lat=TARGET[0], target_lon=TARGET[1],
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
        measure_candidate(c, wanted, [], 3, target_lat=TARGET[0], target_lon=TARGET[1])
        for c in candidates.values()
    ]

    retained = select_for_demand(measures, candidates, wanted, *TARGET, wanted=2)

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
        measure_candidate(c, wanted, [], 3, target_lat=TARGET[0], target_lon=TARGET[1])
        for c in candidates.values()
    ]

    forward = select_for_demand(measures, candidates, wanted, *TARGET, wanted=2)
    backward = select_for_demand(
        list(reversed(measures)), candidates, wanted, *TARGET, wanted=2
    )

    assert forward == backward


# --- ce qui est rejeté avant tout téléchargement ------------------------------


def test_a_candidate_without_a_position_is_rejected() -> None:
    measure = measure_candidate(
        candidate("c1", None, None), demand(), [], 3,
        target_lat=TARGET[0], target_lon=TARGET[1],
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
        target_lat=TARGET[0], target_lon=TARGET[1],
    )
    report = SearchReport(measures=[measure])

    assert report.as_dict()["bytes_downloaded"] == 0
