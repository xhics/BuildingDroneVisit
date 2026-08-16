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


def site(geometry: set[str] | None = None, **states) -> SiteManifest:
    """Un manifeste de site aux identifiants **préfixés**, comme le réel.

    `welcominns-boucherville:PARKING_HOTEL` — c'est cette forme qui a révélé
    la jointure fausse : indexer par identifiant puis chercher par type ne
    pouvait jamais aboutir.
    """
    with_geometry = geometry or set()
    objects = []
    for kind, state in states.items():
        objects.append(
            SiteObject(
                object_id=f"h:{kind}", kind=kind, state=state,
                evidence=["essai"] if state is ObjectState.CONFIRMED else [],
                geometry_wkt=(
                    "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"
                    if kind in with_geometry else None
                ),
            )
        )
    return SiteManifest(hotel_id="h", objects=objects)


def resolved_site() -> SiteManifest:
    """Tous les objets présents, inférés, et géoréférencés."""
    kinds = {obligation.object_id for obligation in OBLIGATIONS}
    return site(
        geometry=kinds,
        **{kind: ObjectState.INFERRED for kind in kinds},
    )


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


def test_an_unresolved_object_keeps_its_demand() -> None:
    """« La cible n'est pas résolue » n'est pas « le besoin n'existe pas ».

    Supprimer le besoin donnait un système sûr et bloqué : la découverte ne
    cherchait jamais cette cible, et rien ne pouvait la débloquer.
    """
    partial = resolved_site()
    partial.objects = [
        obj.model_copy(
            update={"state": ObjectState.UNRESOLVED, "evidence": [],
                    "geometry_wkt": None}
        )
        if obj.kind == "ENTRANCE_MAIN_CURRENT" else obj
        for obj in partial.objects
    ]

    manifest, report = build("h", partial, COVERAGE)

    assert demand_id_for("ENTRANCE_MAIN_CURRENT") in {
        d.demand_id for d in manifest.demands
    }
    assert "ENTRANCE_MAIN_CURRENT" in report.unresolved_target
    assert "ENTRANCE_MAIN_CURRENT" not in report.waived
    assert "ENTRANCE_MAIN_CURRENT" not in report.not_applicable


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

    mesure = manifest.assessments[0].continuity_achieved

    # `is not 0.0` comparait des **identités d'objets** : vrai pour presque
    # tout, y compris un zéro calculé. L'assertion ne disait donc rien de ce
    # que sa docstring annonce.
    assert mesure is None
    assert mesure != 0.0, "zéro affirmerait une mesure qu'on n'a pas faite"


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


# --- la jointure par type, et ce que sa fausseté coûtait -----------------------


def test_objects_resolve_by_kind_not_by_instance_identifier() -> None:
    """Le défaut trouvé à l'exécution réelle.

    Les identifiants sont préfixés du site ; les indexer puis chercher par
    type ne pouvait jamais aboutir. Trois objets présents — dont un
    géoréférencé — étaient déclarés absents, et deux besoins obligatoires
    disparaissaient du manifeste.
    """
    from hotel_pipeline.site_resolution import Resolution, resolve_site_object

    manifest = site(
        geometry={"PARKING_HOTEL"},
        PARKING_HOTEL=ObjectState.INFERRED,
        PROPERTY_SIGN=ObjectState.INFERRED,
        ENTRANCE_MAIN_CURRENT=ObjectState.UNRESOLVED,
    )

    parking = resolve_site_object(manifest, "PARKING_HOTEL")
    sign = resolve_site_object(manifest, "PROPERTY_SIGN")
    entrance = resolve_site_object(manifest, "ENTRANCE_MAIN_CURRENT")

    assert parking.resolution is Resolution.TARGETABLE
    assert sign.resolution is Resolution.NO_GEOMETRY
    assert entrance.resolution is Resolution.UNRESOLVED
    # L'identifiant d'instance complet est conservé : le type ne désigne rien.
    assert parking.object_id == "h:PARKING_HOTEL"


def test_an_absent_kind_is_distinguished_from_an_unresolved_one() -> None:
    """« Aucun objet de ce type » et « présent, non résolu » diffèrent."""
    from hotel_pipeline.site_resolution import Resolution, resolve_site_object

    manifest = site(PROPERTY_SIGN=ObjectState.UNRESOLVED)

    assert resolve_site_object(manifest, "PROPERTY_SIGN").resolution is (
        Resolution.UNRESOLVED
    )
    assert resolve_site_object(manifest, "PARKING_HOTEL").resolution is (
        Resolution.ABSENT
    )


def test_two_instances_of_a_singleton_are_refused_not_silently_chosen() -> None:
    """Deux entrées principales ne sont pas un choix, c'est une contradiction."""
    from hotel_pipeline.schemas.site import SiteObject
    from hotel_pipeline.site_resolution import AmbiguousSiteObject, resolve_site_object

    manifest = site(ENTRANCE_MAIN_CURRENT=ObjectState.INFERRED)
    manifest.objects.append(
        SiteObject(
            object_id="h:ENTRANCE_MAIN_CURRENT-bis",
            kind="ENTRANCE_MAIN_CURRENT", state=ObjectState.INFERRED,
        )
    )

    with pytest.raises(AmbiguousSiteObject, match="2 instances"):
        resolve_site_object(manifest, "ENTRANCE_MAIN_CURRENT")


def test_an_inferred_object_without_geometry_is_not_targetable() -> None:
    from hotel_pipeline.site_resolution import Resolution, resolve_site_object

    manifest = site(PROPERTY_SIGN=ObjectState.INFERRED)

    resolved = resolve_site_object(manifest, "PROPERTY_SIGN")
    assert resolved.resolution is Resolution.NO_GEOMETRY
    assert resolved.exists is True
    assert resolved.is_targetable is False


# --- applicabilité : obligatoire quand l'objet existe -------------------------


def test_a_parking_that_exists_produces_its_obligation() -> None:
    """« Facultatif » laissait un stationnement géoréférencé n'exiger rien."""
    manifest, report = build("h", resolved_site(), COVERAGE)

    assert demand_id_for("PARKING_HOTEL") in {d.demand_id for d in manifest.demands}
    assert "PARKING_HOTEL" not in report.not_applicable


def test_a_parking_whose_absence_is_established_is_not_applicable() -> None:
    without = site(**{
        obligation.object_id: ObjectState.INFERRED
        for obligation in OBLIGATIONS
        if obligation.object_id != "PARKING_HOTEL"
    })

    manifest, report = build("h", without, COVERAGE)

    assert demand_id_for("PARKING_HOTEL") not in {d.demand_id for d in manifest.demands}
    assert "PARKING_HOTEL" in report.not_applicable


def test_every_mandatory_obligation_survives_an_unresolved_target() -> None:
    """Les sept obligations obligatoires doivent toutes apparaître."""
    unresolved_everywhere = site(**{
        obligation.object_id: ObjectState.UNRESOLVED for obligation in OBLIGATIONS
    })

    manifest, _ = build("h", unresolved_everywhere, COVERAGE)
    identifiers = {d.demand_id for d in manifest.demands}

    for obligation in OBLIGATIONS:
        if obligation.applicability.value == "always":
            assert demand_id_for(obligation.object_id) in identifiers


# --- proxy de recherche : chercher sans conclure ------------------------------


def test_an_unresolved_target_declares_where_to_search() -> None:
    """Ne rien chercher parce que le point exact manque serait un blocage."""
    partial = site(
        geometry={"BUILDING_MAIN"},
        **{o.object_id: ObjectState.INFERRED for o in OBLIGATIONS},
        BUILDING_MAIN=ObjectState.CONFIRMED,
    )

    _, report = build("h", partial, COVERAGE)

    assert report.search_proxies[demand_id_for("ENTRANCE_MAIN_CURRENT")] == (
        "FACADE_PRIMARY"
    )
    assert report.search_proxies[demand_id_for("PROPERTY_SIGN")] == "BUILDING_MAIN"


def test_a_targetable_object_needs_no_proxy() -> None:
    _, report = build("h", resolved_site(), COVERAGE)

    assert demand_id_for("PARKING_HOTEL") not in report.search_proxies


def test_the_pilot_manifest_carries_at_least_the_seven_mandatory_demands() -> None:
    """Non-régression sur le manifeste réel du pilote."""
    from pathlib import Path

    path = Path(
        "work/welcominns-boucherville/01_sources/capture_demands.json"
    )
    if not path.is_file():  # pragma: no cover — dépend du corpus local
        pytest.skip("manifeste du pilote absent")

    payload = json.loads(path.read_text("utf-8"))
    identifiers = {demand["demand_id"] for demand in payload["demands"]}

    for object_id in (
        "FACADE_PRIMARY", "FACADE_LEFT", "FACADE_RIGHT", "FACADE_REAR",
        "ENTRANCE_MAIN_CURRENT", "PROPERTY_SIGN", "ACCESS_ROAD_MAIN",
    ):
        assert demand_id_for(object_id) in identifiers, object_id

    # Le stationnement, lui, a été **dé-résolu** : son association reposait sur
    # la proximité et l'inspection l'a démentie. Son obligation conditionnelle
    # ne joue donc plus — l'exiger ferait chercher un objet dont rien
    # n'établit l'existence.
    assert demand_id_for("PARKING_HOTEL") not in identifiers


# --- l'existence s'établit, elle ne se présume pas -----------------------------


def test_an_unresolved_conditional_object_is_pending_not_demanded() -> None:
    """Le gabarit instancie **tous** les types : `unresolved` ne prouve rien.

    Créer le besoin ferait consacrer des requêtes à une allée dont rien
    n'établit l'existence ; le dispenser affirmerait son absence.
    """
    manifest, report = build(
        "h",
        site(
            geometry={"PARKING_HOTEL"},
            **{o.object_id: ObjectState.INFERRED for o in OBLIGATIONS
               if o.object_id != "DRIVEWAY_MAIN"},
            DRIVEWAY_MAIN=ObjectState.UNRESOLVED,
        ),
        COVERAGE,
    )

    assert demand_id_for("DRIVEWAY_MAIN") not in {d.demand_id for d in manifest.demands}
    assert "DRIVEWAY_MAIN" in report.pending_applicability
    assert "existence non établie" in report.pending_applicability["DRIVEWAY_MAIN"]
    # Ni dispensée : affirmer l'absence serait une affirmation qu'on n'a pas.
    assert "DRIVEWAY_MAIN" not in report.not_applicable
    assert "DRIVEWAY_MAIN" not in report.waived


def test_an_unresolved_mandatory_object_still_gets_its_demand() -> None:
    """`always` ne dépend d'aucune existence : toute propriété a des façades."""
    manifest, report = build(
        "h",
        site(**{o.object_id: ObjectState.UNRESOLVED for o in OBLIGATIONS}),
        COVERAGE,
    )
    identifiers = {d.demand_id for d in manifest.demands}

    assert demand_id_for("ENTRANCE_MAIN_CURRENT") in identifiers
    assert demand_id_for("FACADE_REAR") in identifiers
    # Le conditionnel, lui, attend.
    assert demand_id_for("PARKING_HOTEL") not in identifiers
    assert "PARKING_HOTEL" in report.pending_applicability


def test_an_absent_conditional_object_is_not_applicable_not_pending() -> None:
    """« Absent du gabarit » est une preuve d'absence ; `unresolved` ne l'est pas."""
    manifest, report = build(
        "h",
        site(**{o.object_id: ObjectState.INFERRED for o in OBLIGATIONS
                if o.object_id != "DRIVEWAY_MAIN"}),
        COVERAGE,
    )

    assert "DRIVEWAY_MAIN" in report.not_applicable
    assert "DRIVEWAY_MAIN" not in report.pending_applicability


def test_an_inferred_conditional_object_is_demanded() -> None:
    manifest, report = build("h", resolved_site(), COVERAGE)

    assert demand_id_for("PARKING_HOTEL") in {d.demand_id for d in manifest.demands}
    assert "PARKING_HOTEL" not in report.pending_applicability


def test_existence_is_distinct_from_instantiation() -> None:
    """Deux questions que le manifeste de site confond volontairement."""
    from hotel_pipeline.site_resolution import resolve_site_object

    instantiated = resolve_site_object(
        site(DRIVEWAY_MAIN=ObjectState.UNRESOLVED), "DRIVEWAY_MAIN"
    )
    established = resolve_site_object(
        site(DRIVEWAY_MAIN=ObjectState.INFERRED), "DRIVEWAY_MAIN"
    )

    assert instantiated.is_instantiated is True
    assert instantiated.exists is False
    assert established.exists is True


def test_the_pilot_holds_seven_demands_after_the_parking_was_unresolved() -> None:
    """Non-régression sur le résultat réel.

    Le pilote portait huit besoins tant que `PARKING_HOTEL` passait pour
    établi. L'inspection de son aperçu a montré le bâtiment 1205 là où l'hôtel
    est au 1195 : l'objet est redevenu `unresolved`, et son obligation
    conditionnelle ne s'applique plus.
    """
    from pathlib import Path

    demands_path = Path(
        "work/welcominns-boucherville/01_sources/capture_demands.json"
    )
    if not demands_path.is_file():  # pragma: no cover — dépend du corpus local
        pytest.skip("manifeste du pilote absent")

    payload = json.loads(demands_path.read_text("utf-8"))
    identifiers = {demand["demand_id"] for demand in payload["demands"]}

    assert len(payload["demands"]) == 7
    assert demand_id_for("PARKING_HOTEL") not in identifiers
    assert demand_id_for("DRIVEWAY_MAIN") not in identifiers
