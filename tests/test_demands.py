"""Besoins : instanciation puis évaluation sur le corpus existant.

Sans ces deux étapes, la recherche adaptative devrait redéfinir les objectifs
de couverture dans le collecteur — deux sources d'autorité, qui finiraient par
diverger.
"""

from __future__ import annotations

import json

import pytest

from hotel_pipeline.coverage_obligations import (
    OBLIGATIONS,
    ObligationStatus,
    ObligationWaiver,
)
from hotel_pipeline.demands_assess import assess, counts_towards
from hotel_pipeline.demands_build import DemandsRefused, build, demand_id_for
from hotel_pipeline.schemas import (
    Asset,
    ClusterRole,
    DEFAULT_POLICY,
    GeometrySuitability,
    ObjectState,
    ReconstructionRole,
    ViewSector,
)
from hotel_pipeline.schemas.acquisition import CaptureIntent, DemandStatus, TargetKind
from hotel_pipeline.schemas.site import SiteManifest, SiteObject

COVERAGE = DEFAULT_POLICY.coverage


def site(**states) -> SiteManifest:
    """Un manifeste de site où chaque objet porte l'état demandé."""
    objects = []
    for object_id, state in states.items():
        objects.append(
            SiteObject(
                object_id=object_id, kind=object_id, state=state,
                evidence=["essai"] if state is ObjectState.CONFIRMED else [],
            )
        )
    return SiteManifest(hotel_id="h", objects=objects)


def resolved_site() -> SiteManifest:
    return site(**{
        obligation.object_id: ObjectState.INFERRED for obligation in OBLIGATIONS
    })


def waiver(object_id: str, **overrides) -> ObligationWaiver:
    fields = dict(
        object_id=object_id, status=ObligationStatus.NOT_APPLICABLE,
        decided_by="Hicham", rationale="absent du site", evidence=["visite"],
    )
    fields.update(overrides)
    return ObligationWaiver(**fields)


def asset(asset_id: str, **overrides) -> Asset:
    """Un asset porteur : une aptitude exige son appréciation à l'appui."""
    from hotel_pipeline.review import assessment_fields

    suitability = overrides.pop("geometry_suitability", GeometrySuitability.PRIMARY)
    fields = dict(
        id=asset_id, source="mapillary", source_url_or_id=asset_id,
        rights="open_data", ai_eligible=False, confidence=0.9, category="facade",
        checksum="a" * 64,
        cluster_role=ClusterRole.CANONICAL,
        reconstruction_role=ReconstructionRole.PHOTO_GEOMETRY,
        target_building_visible=True,
        view_sector=ViewSector.FRONT,
        camera_lat=45.5730, camera_lon=-73.4430,
    )
    if suitability is not GeometrySuitability.UNASSESSED:
        fields.update(
            assessment_fields(
                suitability, "hm", "façade cadrée",
                ["contrôle du cadrage"], "a" * 64,
            )
        )
    fields.update(overrides)
    return Asset(**fields)


# --- construction : identifiants stables et déterministes ---------------------


def test_each_mandatory_obligation_yields_a_stable_demand() -> None:
    manifest, report = build("h", resolved_site(), COVERAGE)

    identifiers = {demand.demand_id for demand in manifest.demands}
    for obligation in OBLIGATIONS:
        if obligation.mandatory:
            assert demand_id_for(obligation.object_id) in identifiers

    assert "obligation:FACADE_REAR" in identifiers
    assert "obligation:ENTRANCE_MAIN_CURRENT" in identifiers


def test_two_builds_produce_the_same_identifiers() -> None:
    first, _ = build("h", resolved_site(), COVERAGE)
    second, _ = build("h", resolved_site(), COVERAGE)

    assert [d.demand_id for d in first.demands] == [d.demand_id for d in second.demands]


def test_operator_demands_are_preserved() -> None:
    """Le générateur n'est pas propriétaire du manifeste."""
    from hotel_pipeline.schemas.acquisition import CaptureDemand, CaptureDemandManifest

    custom = CaptureDemand(
        demand_id="operateur-toiture-drone", intent=CaptureIntent.BUILDING_CAPTURE,
        target_kind=TargetKind.VIEW_SECTOR, target_ref="roof",
    )
    existing = CaptureDemandManifest(hotel_id="h", demands=[custom])

    manifest, report = build("h", resolved_site(), COVERAGE, existing=existing)

    assert "operateur-toiture-drone" in {d.demand_id for d in manifest.demands}
    assert report.operator_defined == ["operateur-toiture-drone"]


def test_thresholds_come_from_the_policy_never_from_the_command() -> None:
    """Codés dans le générateur, ils auraient fini par mentir sur la politique."""
    from hotel_pipeline.schemas.policy import CoveragePolicy

    tightened = CoveragePolicy(building_viewpoints_required=5)

    manifest, _ = build("h", resolved_site(), tightened)
    facade = next(d for d in manifest.demands if d.target_ref == "front")

    assert facade.viewpoints_required == 5
    assert facade.min_visible_fraction == tightened.building_min_visible_fraction


def test_building_and_context_do_not_share_their_thresholds() -> None:
    manifest, _ = build("h", resolved_site(), COVERAGE)

    facade = next(d for d in manifest.demands if d.target_ref == "front")
    access = next(d for d in manifest.demands if d.target_ref == "ACCESS_ROAD_MAIN")

    assert facade.viewpoints_required == COVERAGE.building_viewpoints_required
    assert access.viewpoints_required == COVERAGE.context_viewpoints_required
    assert facade.continuity_required > access.continuity_required


# --- un objet non résolu ne devient jamais une dispense ------------------------


def test_an_unresolved_object_yields_an_unresolved_target_not_a_waiver() -> None:
    """Le convertir ferait disparaître un manque en le déclarant sans objet."""
    partial = resolved_site()
    partial.objects = [
        obj.model_copy(update={"state": ObjectState.UNRESOLVED, "evidence": []})
        if obj.object_id == "ENTRANCE_MAIN_CURRENT" else obj
        for obj in partial.objects
    ]

    manifest, report = build("h", partial, COVERAGE)

    assert "ENTRANCE_MAIN_CURRENT" in report.unresolved_target
    assert "ENTRANCE_MAIN_CURRENT" not in report.waived
    assert "ENTRANCE_MAIN_CURRENT" not in report.not_applicable
    assert demand_id_for("ENTRANCE_MAIN_CURRENT") not in {
        d.demand_id for d in manifest.demands
    }


def test_a_declared_waiver_is_honoured() -> None:
    manifest, report = build(
        "h", resolved_site(), COVERAGE, waivers=[waiver("PROPERTY_SIGN")]
    )

    assert "PROPERTY_SIGN" in report.not_applicable
    assert demand_id_for("PROPERTY_SIGN") not in {d.demand_id for d in manifest.demands}


def test_building_without_a_site_manifest_is_refused() -> None:
    with pytest.raises(DemandsRefused, match="aucun manifeste de site"):
        build("h", None, COVERAGE)


def test_building_downloads_nothing() -> None:
    _, report = build("h", resolved_site(), COVERAGE)

    assert report.as_dict()["bytes_downloaded"] == 0


# --- évaluation : ce qui compte, et ce qui ne compte pas -----------------------


def demands_for(*refs: str) -> list:
    manifest, _ = build("h", resolved_site(), COVERAGE)
    return [d for d in manifest.demands if d.target_ref in refs]


def test_nine_files_at_six_positions_count_as_six_viewpoints() -> None:
    """Le décompte porte sur les positions, jamais sur les fichiers."""
    from hotel_pipeline.plan import group_viewpoints

    positions = [
        (45.5730, -73.4430), (45.5732, -73.4432), (45.5734, -73.4434),
        (45.5736, -73.4436), (45.5738, -73.4438), (45.5740, -73.4440),
    ]
    assets = [
        asset(f"a{i}", camera_lat=lat, camera_lon=lon)
        for i, (lat, lon) in enumerate(positions)
    ]
    # Trois doublons exacts de position : neuf fichiers, six points de vue.
    assets += [
        asset(f"b{i}", camera_lat=positions[i][0], camera_lon=positions[i][1])
        for i in range(3)
    ]

    class Subject:
        def __init__(self, a):
            self.candidate_id, self.camera_lat = a.id, a.camera_lat
            self.camera_lon, self.panorama_id = a.camera_lon, None

    grouped = group_viewpoints([Subject(a) for a in assets], separation_m=10.0)

    assert len(assets) == 9
    assert len(set(grouped.values())) == 6

    manifest, _ = assess(
        "h", demands_for("front"), assets, corpus_digest="c0", viewpoints=grouped
    )
    assert manifest.assessments[0].viewpoints_found == 6


def test_a_context_lock_never_counts_as_an_observation() -> None:
    locked = asset("a1", reconstruction_role=ReconstructionRole.CONTEXT_LOCK)

    ok, reason = counts_towards(locked)

    assert ok is False
    assert "non porteur de géométrie" in reason


def test_an_inactive_duplicate_never_counts() -> None:
    duplicate = asset("a1", cluster_role=ClusterRole.INACTIVE)

    ok, reason = counts_towards(duplicate)

    assert ok is False
    assert "déduplication" in reason


def test_an_unassessed_view_never_counts() -> None:
    unassessed = asset("a1", geometry_suitability=GeometrySuitability.UNASSESSED)

    assert counts_towards(unassessed)[0] is False


def test_a_view_of_the_wrong_sector_serves_another_demand() -> None:
    """Un point de vue du coin avant-droit ne donne pas d'ancre à l'arrière."""
    corner = asset("a1", view_sector=ViewSector.FRONT_RIGHT_CORNER)

    manifest, report = assess(
        "h", demands_for("front", "rear"), [corner], corpus_digest="c0"
    )

    by_demand = {a.demand_id: a for a in manifest.assessments}
    assert by_demand["obligation:FACADE_REAR"].viewpoints_found == 0
    assert by_demand["obligation:FACADE_PRIMARY"].viewpoints_found == 0


def test_a_satisfied_demand_is_reported_met() -> None:
    from hotel_pipeline.schemas.policy import CoveragePolicy

    lenient = CoveragePolicy(
        building_viewpoints_required=1, building_continuity_required=0.0
    )
    manifest_demands, _ = build("h", resolved_site(), lenient)
    front = [d for d in manifest_demands.demands if d.target_ref == "front"]

    manifest, report = assess("h", front, [asset("a1")], corpus_digest="c0")

    assert manifest.assessments[0].status is DemandStatus.MET
    assert "obligation:FACADE_PRIMARY" not in report.open_demands


def test_a_demand_requiring_continuity_is_never_met_on_an_existing_corpus() -> None:
    """La continuité se mesure sur les images ; rien ici ne les ouvre."""
    manifest, _ = assess("h", demands_for("front"), [asset("a1")], corpus_digest="c0")

    assessment = manifest.assessments[0]
    assert assessment.continuity_achieved is None
    assert assessment.status is not DemandStatus.MET
    assert "continuité non mesurée" in assessment.rationale


def test_an_unmeasured_continuity_is_none_never_zero() -> None:
    """Zéro dirait « mesurée, et nulle » — une affirmation qu'on n'a pas."""
    manifest, _ = assess("h", demands_for("front"), [asset("a1")], corpus_digest="c0")

    assert manifest.assessments[0].continuity_achieved is not 0.0  # noqa: F632
    assert manifest.assessments[0].continuity_achieved is None


def test_an_unresolved_target_is_not_unreachable() -> None:
    """L'un dit qu'on ne sait pas viser, l'autre qu'aucune vue n'existera."""
    demands = demands_for("front")

    manifest, report = assess(
        "h", demands, [], corpus_digest="c0",
        unresolved_targets={demands[0].demand_id: "objet non résolu"},
    )

    assessment = manifest.assessments[0]
    assert assessment.status is DemandStatus.OPEN
    assert assessment.status is not DemandStatus.UNREACHABLE
    assert demands[0].demand_id in report.unresolved_targets


def test_assessing_downloads_nothing() -> None:
    _, report = assess("h", demands_for("front"), [asset("a1")], corpus_digest="c0")

    assert report.as_dict()["bytes_downloaded"] == 0


def test_the_report_names_the_open_demands_for_the_search() -> None:
    """C'est ce rapport qui dira à la recherche quels secteurs sont déficitaires."""
    _, report = assess(
        "h", demands_for("front", "rear", "left"), [asset("a1")], corpus_digest="c0"
    )

    assert "obligation:FACADE_REAR" in report.open_demands
    assert "obligation:FACADE_LEFT" in report.open_demands
