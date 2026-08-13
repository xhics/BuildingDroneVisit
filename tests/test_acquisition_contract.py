"""Contrat d'acquisition ciblée (Lot 1B V2, étape 1).

Ce qui est éprouvé ici : un candidat n'est pas un asset, un objectif n'est pas
son état, un candidat ne vaut pas la même chose pour deux besoins, une recollecte
n'écrase rien, et rien de tout cela ne repose sur une URL qui expire.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hotel_pipeline import acquisition
from hotel_pipeline.schemas import (
    AcquisitionPlan,
    AcquisitionProvenance,
    Asset,
    CandidateEvaluation,
    CandidateGeometry,
    CandidateManifest,
    CaptureCandidate,
    CaptureDemand,
    CaptureDemandManifest,
    CaptureIntent,
    DemandAssessment,
    DemandStatus,
    Eligibility,
    ExteriorInterior,
    PlanStatus,
    PlannedAcquisition,
    ReviewDecision,
    ReviewEntry,
    Rights,
    TargetKind,
    VolumeStatus,
    capture_identity,
    validate_targets,
)
from hotel_pipeline.schemas.acquisition import REQUIRED_PLAN_DIGESTS
from hotel_pipeline.schemas.assets import DECISION_STATUS, VISIBILITY_OF

FULL_DIGESTS = {name: f"{name[:4]}000000000000" for name in REQUIRED_PLAN_DIGESTS}


def image_bytes(width: int = 64, height: int = 48, noisy: bool = True) -> bytes:
    """Une vraie image JPEG : le contrat en exige désormais une."""
    import io

    from PIL import Image

    image = Image.new("RGB", (width, height), (30, 60, 90))
    if noisy:
        # Une image d'une seule couleur est refusée comme réponse de
        # remplacement : il en faut donc au moins deux.
        for x in range(0, width, 2):
            for y in range(0, height, 3):
                image.putpixel((x, y), (200, 180, 40))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def candidate(candidate_id: str = "mapillary-1", **overrides) -> CaptureCandidate:
    fields = dict(
        candidate_id=candidate_id,
        source="mapillary",
        provider_id="123456789",
        camera_lat=45.5730,
        camera_lon=-73.4433,
        computed_heading_deg=45.0,
        original_heading_deg=47.0,
        sequence_id="seq-a",
        advertised_width=2048,
        advertised_height=1536,
        captured_at=datetime(2024, 3, 1, tzinfo=timezone.utc),
        available_resolutions=["256", "1024", "2048"],
        request_spec={"image_id": "123456789"},
    )
    fields.update(overrides)
    return CaptureCandidate(**fields)


def planned(candidate_id="m-1", **overrides) -> PlannedAcquisition:
    fields = dict(
        candidate_id=candidate_id,
        intents=[CaptureIntent.BUILDING_CAPTURE],
        selection_rationale="secteur avant manquant",
        serves_demands=["front"],
    )
    fields.update(overrides)
    return PlannedAcquisition(**fields)


def evaluation(candidate_id="mapillary-1", demand_id="front", **overrides) -> CandidateEvaluation:
    fields = dict(
        candidate_id=candidate_id,
        demand_id=demand_id,
        intent=CaptureIntent.BUILDING_CAPTURE,
        geometry=CandidateGeometry(distance_m=42.0, unclipped_width_fraction=0.35),
    )
    fields.update(overrides)
    return CandidateEvaluation(**fields)


def provenance(**overrides) -> AcquisitionProvenance:
    fields = dict(
        provider_id="123456789",
        plan_id="plan-1",
        plan_digest="a" * 16,
        candidate_id="mapillary-1",
        intents=[CaptureIntent.BUILDING_CAPTURE],
        run_id="20260813T140000000000Z",
    )
    fields.update(overrides)
    return AcquisitionProvenance(**fields)


def acquired(tmp_path, asset_id="mapillary-1", content=None, **candidate_kw) -> Asset:
    path = tmp_path / f"{asset_id}.jpg"
    path.write_bytes(content if content is not None else image_bytes())
    return acquisition.as_asset(
        candidate(asset_id, **candidate_kw),
        provenance(candidate_id=asset_id),
        path,
        Rights.OPEN_DATA,
    )


def reviewed(tmp_path, asset_id="mapillary-1", content=None) -> Asset:
    """Un asset déjà revu : ce qu'une recollecte ne doit jamais perdre."""
    base = acquired(tmp_path, asset_id, content)
    entry = ReviewEntry(
        decision=ReviewDecision.CONFIRMED,
        decided_by="OpenAI Codex — arbitrage architectural",
        rationale="pylône « HW HÔTEL WELCOMINNS » lisible",
        evidence=["enseigne au tiers gauche du cadre"],
        reviewed_checksum=base.checksum,
    )
    return base.model_copy(
        update={
            "review_history": [entry],
            "target_visibility_decision": entry.decision,
            "review_status": DECISION_STATUS[entry.decision],
            "target_building_visible": VISIBILITY_OF[entry.decision],
            "reviewer": entry.decided_by,
            "review_rationale": entry.rationale,
            "review_evidence": entry.evidence,
        }
    )


# --- l'objectif n'est pas son état ------------------------------------------


def test_a_demand_carries_no_state() -> None:
    """Un objectif est stable ; « satisfait » dépend du corpus du jour."""
    fields = set(CaptureDemand.model_fields)

    assert "status" not in fields
    assert {"target_kind", "target_ref", "viewpoints_required"} <= fields


def test_an_assessment_is_bound_to_a_corpus() -> None:
    with pytest.raises(ValueError):
        DemandAssessment(demand_id="front", corpus_digest="")

    assessed = DemandAssessment(
        demand_id="front", corpus_digest="c" * 16, viewpoints_found=1,
        status=DemandStatus.PARTIALLY_MET,
    )
    assert assessed.corpus_digest


def test_an_unreachable_demand_must_be_justified() -> None:
    """Renoncer sans motif interdit d'y revenir."""
    with pytest.raises(ValueError, match="inatteignable sans motif"):
        DemandAssessment(
            demand_id="rear", corpus_digest="c" * 16, status=DemandStatus.UNREACHABLE
        )


def test_unreachable_demands_are_closed_not_open() -> None:
    """`open_demands()` comptait `unreachable` comme ouvert.

    On aurait cherché indéfiniment ce qu'aucune acquisition ne peut donner.
    """
    manifest = CaptureDemandManifest(
        hotel_id="h",
        demands=[
            CaptureDemand(demand_id="front", intent=CaptureIntent.BUILDING_CAPTURE,
                          target_kind=TargetKind.VIEW_SECTOR, target_ref="front"),
            CaptureDemand(demand_id="rear", intent=CaptureIntent.BUILDING_CAPTURE,
                          target_kind=TargetKind.VIEW_SECTOR, target_ref="rear"),
            CaptureDemand(demand_id="roof", intent=CaptureIntent.BUILDING_CAPTURE,
                          target_kind=TargetKind.VIEW_SECTOR, target_ref="roof"),
        ],
    )
    assessments = [
        DemandAssessment(demand_id="rear", corpus_digest="c" * 16,
                         status=DemandStatus.UNREACHABLE,
                         rationale="façade arrière sur terrain privé sans accès public"),
        DemandAssessment(demand_id="roof", corpus_digest="c" * 16,
                         status=DemandStatus.MET, viewpoints_found=1),
    ]

    outstanding = [d.demand_id for d in manifest.outstanding(assessments)]
    assert outstanding == ["front"]


# --- cibles typées -----------------------------------------------------------


def test_targets_are_checked_against_the_site_and_the_vocabulary() -> None:
    manifest = CaptureDemandManifest(
        hotel_id="h",
        demands=[
            CaptureDemand(demand_id="d1", intent=CaptureIntent.BUILDING_CAPTURE,
                          target_kind=TargetKind.VIEW_SECTOR, target_ref="front"),
            CaptureDemand(demand_id="d2", intent=CaptureIntent.BUILDING_CAPTURE,
                          target_kind=TargetKind.VIEW_SECTOR, target_ref="facade_avant"),
            CaptureDemand(demand_id="d3", intent=CaptureIntent.BUILDING_CAPTURE,
                          target_kind=TargetKind.SITE_OBJECT, target_ref="ENTRANCE_MAIN"),
            CaptureDemand(demand_id="d4", intent=CaptureIntent.CONTEXT_CAPTURE,
                          target_kind=TargetKind.CONTEXT_CORRIDOR, target_ref="ampere-nord"),
            CaptureDemand(demand_id="d5", intent=CaptureIntent.BUILDING_CAPTURE,
                          target_kind=TargetKind.VIEW_SECTOR, target_ref="unknown"),
        ],
    )

    problems = validate_targets(
        manifest,
        site_object_ids={"ENTRANCE_MAIN_CURRENT", "PARKING_HOTEL"},
        corridor_ids={"ampere-sud"},
    )

    assert any("facade_avant" in p and "secteur inconnu" in p for p in problems)
    assert any("objet de site inconnu 'ENTRANCE_MAIN'" in p for p in problems)
    assert any("corridor inconnu 'ampere-nord'" in p for p in problems)
    # `unknown` n'est pas un objectif : ce n'est pas un secteur qu'on vise.
    assert any("d5" in p and "unknown" in p for p in problems)
    assert not any("d1" in p for p in problems)


def test_an_absent_registry_is_not_an_implicit_pass() -> None:
    """Le trou : `and corridors` taisait tout quand le registre était vide.

    Registre absent et registre vide ne disent pas la même chose — le premier
    empêche de valider, le second rend toute référence fausse.
    """
    manifest = CaptureDemandManifest(
        hotel_id="h",
        demands=[
            CaptureDemand(demand_id="d1", intent=CaptureIntent.CONTEXT_CAPTURE,
                          target_kind=TargetKind.CONTEXT_CORRIDOR, target_ref="ampere-nord"),
        ],
    )

    absent = validate_targets(manifest, site_object_ids=set(), corridor_ids=None)
    assert any("invérifiable" in p for p in absent)

    empty = validate_targets(manifest, site_object_ids=set(), corridor_ids=set())
    assert any("corridor inconnu" in p for p in empty)


def test_forbidden_zones_are_validated_too() -> None:
    manifest = CaptureDemandManifest(
        hotel_id="h",
        demands=[
            CaptureDemand(demand_id="roof", intent=CaptureIntent.BUILDING_CAPTURE,
                          target_kind=TargetKind.VIEW_SECTOR, target_ref="roof",
                          forbidden_zone_refs=["roof_gap_12", "roof_gap_99"]),
        ],
    )

    problems = validate_targets(
        manifest, site_object_ids=set(), forbidden_zone_ids={"roof_gap_12"}
    )
    assert any("zone interdite inconnue 'roof_gap_99'" in p for p in problems)
    assert not any("roof_gap_12" in p for p in problems)

    unverifiable = validate_targets(manifest, site_object_ids=set())
    assert any("invérifiable" in p for p in unverifiable)


def test_forbidden_zones_are_designated_not_a_flag() -> None:
    """Un booléen ne disait pas *quelle* zone est interdite."""
    demand = CaptureDemand(
        demand_id="roof", intent=CaptureIntent.BUILDING_CAPTURE,
        target_kind=TargetKind.VIEW_SECTOR, target_ref="roof",
        forbidden_zone_refs=["roof_gap_12", "roof_gap_17"],
    )
    assert "forbidden_close_up" not in CaptureDemand.model_fields
    assert demand.forbidden_zone_refs == ["roof_gap_12", "roof_gap_17"]


# --- un candidat n'est pas un asset, ni un verdict --------------------------


def test_a_candidate_has_neither_file_nor_verdict() -> None:
    fields = set(CaptureCandidate.model_fields)

    assert "checksum" not in fields
    assert "local_path" not in fields
    # Le verdict dépend du besoin : il ne peut pas vivre sur le candidat.
    assert "status" not in fields
    assert "intent" not in fields
    assert "geometry" not in fields
    assert {"provider_id", "available_resolutions", "sequence_id"} <= fields


def test_a_candidate_is_eligible_for_one_demand_and_rejected_for_another() -> None:
    """La même vue cadre la façade, manque l'entrée, et sert le contexte."""
    manifest = CandidateManifest(
        hotel_id="h",
        candidates=[candidate()],
        evaluations=[
            evaluation(demand_id="front"),
            evaluation(demand_id="entrance", eligibility=Eligibility.REJECTED,
                       rejection_reason="entrée hors du champ de vision"),
            evaluation(demand_id="access-road", intent=CaptureIntent.CONTEXT_CAPTURE),
        ],
    )

    assert [e.demand_id for e in manifest.eligible_for("front")] == ["front"]
    assert manifest.eligible_for("entrance") == []
    assert manifest.rejections_by_reason() == {"entrée hors du champ de vision": 1}


def test_a_rejected_evaluation_must_say_why() -> None:
    with pytest.raises(ValueError, match="rejet sans motif"):
        evaluation(eligibility=Eligibility.REJECTED)
    with pytest.raises(ValueError, match="motif de rejet sans rejet"):
        evaluation(rejection_reason="hors rayon")


def test_two_evaluations_of_the_same_pair_are_refused() -> None:
    with pytest.raises(ValueError, match="deux évaluations pour le couple"):
        CandidateManifest(
            hotel_id="h", candidates=[candidate()],
            evaluations=[evaluation(), evaluation()],
        )


def test_an_evaluation_of_an_absent_candidate_is_refused() -> None:
    with pytest.raises(ValueError, match="candidat absent"):
        CandidateManifest(hotel_id="h", evaluations=[evaluation()])


def test_duplicate_candidate_ids_are_refused() -> None:
    with pytest.raises(ValueError, match="dupliqués"):
        CandidateManifest(hotel_id="h", candidates=[candidate(), candidate()])


def test_the_manifest_keeps_the_query_counts() -> None:
    """Sans elles, un zéro ne dit pas si la source a été interrogée."""
    manifest = CandidateManifest(hotel_id="h", queries={"mapillary": 3, "street_view": 0})
    assert manifest.queries["street_view"] == 0


# --- aucune URL persistée ----------------------------------------------------


def test_a_persisted_url_is_refused() -> None:
    with pytest.raises(ValueError, match="contient une URL"):
        candidate(request_spec={"thumb_2048_url": "https://cdn.example/x.jpg"})


def test_a_persisted_secret_is_refused() -> None:
    with pytest.raises(ValueError, match="ressemble à un secret"):
        candidate(request_spec={"api_key": "abcd"})


def test_only_resolutions_and_a_spec_are_kept() -> None:
    kept = candidate()
    assert kept.available_resolutions == ["256", "1024", "2048"]
    assert kept.request_spec == {"image_id": "123456789"}
    assert "thumbnails" not in CaptureCandidate.model_fields


# --- identité d'une prise de vue ---------------------------------------------


def test_a_mapillary_image_is_identified_by_its_id() -> None:
    assert capture_identity("mapillary", "123") == "mapillary-123"


def test_two_framings_of_one_panorama_are_two_captures() -> None:
    """Un panorama n'est pas une image : c'est une sphère qu'on cadre."""
    first = capture_identity("street_view", "PANO1", heading_deg=310.0,
                             fov_deg=60.0, pitch_deg=0.0, size="640x640")
    second = capture_identity("street_view", "PANO1", heading_deg=40.0,
                              fov_deg=60.0, pitch_deg=0.0, size="640x640")
    same = capture_identity("street_view", "PANO1", heading_deg=310.0,
                            fov_deg=60.0, pitch_deg=0.0, size="640x640")

    assert first != second
    assert first == same
    assert first.startswith("street_view-PANO1-")


def test_the_framing_changes_the_identity_field_by_field() -> None:
    base = dict(heading_deg=310.0, fov_deg=60.0, pitch_deg=0.0, size="640x640")
    reference = capture_identity("street_view", "PANO1", **base)

    for field, value in [("fov_deg", 90.0), ("pitch_deg", 10.0), ("size", "1280x1280")]:
        assert capture_identity("street_view", "PANO1", **{**base, field: value}) != reference


# --- empreintes fermées ------------------------------------------------------


def test_an_executable_plan_without_every_digest_is_refused() -> None:
    partial = dict(FULL_DIGESTS)
    del partial["corpus_digest"]

    with pytest.raises(ValueError, match="corpus_digest"):
        AcquisitionPlan(
            plan_id="p", hotel_id="h", status=PlanStatus.EXECUTABLE,
            acquisitions=[planned()],
            **partial,
        )


def test_a_draft_may_be_incomplete_but_never_acquired() -> None:
    draft = AcquisitionPlan(plan_id="p", hotel_id="h")
    assert draft.status is PlanStatus.DRAFT
    assert draft.missing_digests() == list(REQUIRED_PLAN_DIGESTS)

    stale = acquisition.plan_is_current(draft, dict(FULL_DIGESTS))
    assert any("brouillon" in problem for problem in stale)


def test_a_current_plan_matches_every_required_digest() -> None:
    plan = AcquisitionPlan(
        plan_id="p", hotel_id="h", status=PlanStatus.EXECUTABLE,
        acquisitions=[planned()],
        **FULL_DIGESTS,
    )
    assert acquisition.plan_is_current(plan, dict(FULL_DIGESTS)) == []


def test_a_missing_current_digest_is_a_refusal_not_a_silence() -> None:
    """Ignorer une valeur absente laissait passer un plan sans lien avec le site."""
    plan = AcquisitionPlan(
        plan_id="p", hotel_id="h", status=PlanStatus.EXECUTABLE,
        acquisitions=[planned()],
        **FULL_DIGESTS,
    )
    current = dict(FULL_DIGESTS)
    current["road_geometry_digest"] = None

    problems = acquisition.plan_is_current(plan, current)
    assert len(problems) == 1
    assert "road_geometry_digest" in problems[0]


def test_a_diverging_digest_is_detected() -> None:
    plan = AcquisitionPlan(
        plan_id="p", hotel_id="h", status=PlanStatus.EXECUTABLE,
        acquisitions=[planned()],
        **FULL_DIGESTS,
    )
    current = dict(FULL_DIGESTS)
    current["corpus_digest"] = "ffff000000000000"

    problems = acquisition.plan_is_current(plan, current)
    assert len(problems) == 1
    assert problems[0].startswith("corpus_digest")


# --- volume connu et inconnu -------------------------------------------------


def test_an_unknown_size_is_never_counted_as_zero() -> None:
    """Compter l'inconnu comme nul annonçait un volume « exact » faux."""
    plan = AcquisitionPlan(
        plan_id="p", hotel_id="h",
        acquisitions=[
            planned("m-1", expected_bytes=300_000),
            planned("m-2", intents=[CaptureIntent.CONTEXT_CAPTURE],
                    serves_demands=["access-road"]),
        ],
    )

    assert plan.known_bytes == 300_000
    assert plan.unknown_size_items == ["m-2"]
    assert plan.volume_status is VolumeStatus.PARTIAL


def test_volume_is_exact_only_when_every_size_is_known() -> None:
    known = planned("m-1", expected_bytes=1000)
    unknown = planned("m-2")

    assert AcquisitionPlan(plan_id="p", hotel_id="h",
                           acquisitions=[known]).volume_status is VolumeStatus.EXACT
    # Rien de connu et rien d'estimé : « estimé » aurait suggéré un calcul.
    assert AcquisitionPlan(plan_id="p", hotel_id="h",
                           acquisitions=[unknown]).volume_status is VolumeStatus.UNKNOWN


# --- propriétés mesurées, jamais recopiées -----------------------------------


def test_file_properties_are_measured_on_what_was_acquired(tmp_path) -> None:
    """Le fournisseur annonce 2048×1536 ; le fichier, lui, est ce qu'il est."""
    content = image_bytes(64, 48)
    asset = acquired(tmp_path, content=content)

    assert asset.checksum == hashlib.sha256(content).hexdigest()
    assert asset.file_size_bytes == len(content)
    # Mesurées, non recopiées : le candidat annonçait 2048×1536.
    assert (asset.width, asset.height) == (64, 48)


def test_an_unreadable_file_never_becomes_an_asset(tmp_path) -> None:
    """Une acquisition ratée est un échec, pas un asset aux dimensions absentes."""
    with pytest.raises(acquisition.AcquisitionRefused, match="illisible comme image"):
        acquired(tmp_path, content=b"reponse-html-derreur")


def test_an_empty_file_is_refused(tmp_path) -> None:
    with pytest.raises(acquisition.AcquisitionRefused, match="fichier vide"):
        acquired(tmp_path, content=b"")


def test_a_single_colour_placeholder_is_refused(tmp_path) -> None:
    """Street View rend une vignette grise « no imagery » avec un code 200."""
    with pytest.raises(acquisition.AcquisitionRefused, match="une seule couleur"):
        acquired(tmp_path, content=image_bytes(64, 48, noisy=False))


def test_a_wrong_size_is_refused(tmp_path) -> None:
    path = tmp_path / "x.jpg"
    path.write_bytes(image_bytes(64, 48))
    with pytest.raises(acquisition.AcquisitionRefused, match="rendu tronqué"):
        acquisition.measure(path, expected_size=(640, 640))


def test_the_measure_reports_the_real_format(tmp_path) -> None:
    path = tmp_path / "x.jpg"
    path.write_bytes(image_bytes())
    assert acquisition.measure(path).image_format == "JPEG"


def test_exterior_is_never_invented(tmp_path) -> None:
    unknown = acquired(tmp_path, "m-1")
    assert unknown.exterior_or_interior is ExteriorInterior.UNKNOWN

    proven = acquired(tmp_path, "m-2", outdoor_evidence="street_view:source=outdoor")
    assert proven.exterior_or_interior is ExteriorInterior.EXTERIOR


# --- répertoire d'exécution ---------------------------------------------------


@pytest.mark.parametrize(
    "run_id",
    ["../evasion", "run/../..", "/etc", "run id", "", "20260813T140000000000Z/x"],
)
def test_a_malformed_run_id_is_refused(tmp_path, monkeypatch, run_id) -> None:
    """Un identifiant libre permettait de sortir de 02_images/acquisitions."""
    from hotel_pipeline.workspace import Workspace

    monkeypatch.setenv("HOTEL_PIPELINE_WORK", str(tmp_path / "work"))
    workspace = Workspace("hotel-test")

    with pytest.raises(acquisition.AcquisitionRefused, match="identifiant d'exécution"):
        acquisition.run_directory(workspace, run_id)


def test_a_valid_run_stays_inside_the_acquisition_tree(tmp_path, monkeypatch) -> None:
    from hotel_pipeline.workspace import Workspace

    monkeypatch.setenv("HOTEL_PIPELINE_WORK", str(tmp_path / "work"))
    workspace = Workspace("hotel-test")

    target = acquisition.run_directory(workspace, acquisition.new_run_id())
    root = workspace.path("02_images", "acquisitions").resolve()

    assert target.is_relative_to(root)
    assert acquisition.run_directory(workspace, "20260813T140000Z") != target


# --- fusion atomique ---------------------------------------------------------


def test_an_existing_reviewed_asset_survives_a_new_acquisition(tmp_path) -> None:
    existing = [reviewed(tmp_path)]
    before = existing[0].model_dump_json()

    report = acquisition.merge(
        existing, [acquired(tmp_path), acquired(tmp_path, "mapillary-2", image_bytes(80, 60))]
    )

    assert report.added == ["mapillary-2"]
    assert report.unchanged == ["mapillary-1"]
    assert existing[0].model_dump_json() == before
    assert existing[0].review_history


def test_an_id_collision_with_different_content_is_refused(tmp_path) -> None:
    existing = [reviewed(tmp_path)]
    before = [a.model_dump_json() for a in existing]

    other = tmp_path / "autre"
    other.mkdir()
    impostor = acquired(other, "mapillary-1", image_bytes(100, 70))

    with pytest.raises(acquisition.AcquisitionRefused, match="identifiant déjà pris"):
        acquisition.merge(existing, [impostor])

    assert [a.model_dump_json() for a in existing] == before


def test_a_partial_failure_leaves_the_manifest_byte_for_byte(tmp_path) -> None:
    existing = [reviewed(tmp_path)]
    before = [a.model_dump_json() for a in existing]

    good = acquired(tmp_path, "mapillary-9", image_bytes(90, 65))
    other = tmp_path / "bis"
    other.mkdir()
    bad = acquired(other, "mapillary-1", image_bytes(110, 75))

    with pytest.raises(acquisition.AcquisitionRefused):
        acquisition.merge(existing, [good, bad])

    assert [a.model_dump_json() for a in existing] == before
    assert len(existing) == 1


def test_the_same_asset_twice_in_one_batch_is_refused(tmp_path) -> None:
    existing: list[Asset] = []
    with pytest.raises(acquisition.AcquisitionRefused, match="deux fois dans le lot"):
        acquisition.merge(existing, [acquired(tmp_path), acquired(tmp_path)])
    assert existing == []


def test_acquired_files_are_verified_against_their_checksum(tmp_path) -> None:
    assets = [acquired(tmp_path)]
    assert acquisition.verify_acquired(assets) == []

    Path(assets[0].local_path).write_bytes(image_bytes(120, 80))
    problems = acquisition.verify_acquired(assets)
    assert len(problems) == 1
    assert "empreinte du fichier" in problems[0]


def test_a_missing_file_is_reported(tmp_path) -> None:
    assets = [acquired(tmp_path)]
    Path(assets[0].local_path).unlink()
    assert "fichier absent" in acquisition.verify_acquired(assets)[0]


# --- provenance durable ------------------------------------------------------


def test_the_asset_identity_is_the_provider_id_not_a_url(tmp_path) -> None:
    asset = acquired(tmp_path)

    assert asset.source_url_or_id == "123456789"
    assert "http" not in asset.source_url_or_id
    assert asset.acquisition.plan_id == "plan-1"


def test_the_provenance_keeps_both_headings_and_both_positions() -> None:
    kept = AcquisitionProvenance(
        provider_id="1", plan_id="p", plan_digest="d", candidate_id="c",
        queried_lat=45.0, queried_lon=-73.0,
        returned_lat=45.0009, returned_lon=-73.0007,
        original_heading_deg=47.0, computed_heading_deg=45.0,
        requested_heading_deg=310.0, requested_fov_deg=60.0, requested_pitch_deg=5.0,
        advertised_width=2048, advertised_height=1536,
        intents=[CaptureIntent.BUILDING_CAPTURE, CaptureIntent.CONTEXT_CAPTURE],
    )
    assert kept.queried_lat != kept.returned_lat
    assert kept.original_heading_deg != kept.computed_heading_deg
    assert kept.advertised_width == 2048


# --- relations fermées entre manifestes -------------------------------------


def demands_manifest(*demands) -> CaptureDemandManifest:
    return CaptureDemandManifest(
        hotel_id="h",
        demands=list(demands)
        or [
            CaptureDemand(demand_id="front", intent=CaptureIntent.BUILDING_CAPTURE,
                          target_kind=TargetKind.VIEW_SECTOR, target_ref="front"),
            CaptureDemand(demand_id="access-road", intent=CaptureIntent.CONTEXT_CAPTURE,
                          target_kind=TargetKind.SITE_OBJECT, target_ref="ACCESS_ROAD_MAIN"),
        ],
    )


def test_two_assessments_of_one_demand_are_refused() -> None:
    """Dans une liste nue, la seconde écrasait la première sans le dire."""
    from hotel_pipeline.schemas import DemandAssessmentManifest

    with pytest.raises(ValueError, match="évalués deux fois"):
        DemandAssessmentManifest(
            hotel_id="h", corpus_digest="c" * 16, demand_digest="d" * 16,
            assessments=[
                DemandAssessment(demand_id="front", corpus_digest="c" * 16),
                DemandAssessment(demand_id="front", corpus_digest="c" * 16,
                                 status=DemandStatus.MET),
            ],
        )


def test_assessments_of_different_corpora_are_refused() -> None:
    """Un état de couverture ne se compose pas d'instants différents."""
    from hotel_pipeline.schemas import DemandAssessmentManifest

    with pytest.raises(ValueError, match="d'autres corpus"):
        DemandAssessmentManifest(
            hotel_id="h", corpus_digest="c" * 16, demand_digest="d" * 16,
            assessments=[
                DemandAssessment(demand_id="front", corpus_digest="c" * 16),
                DemandAssessment(demand_id="rear", corpus_digest="e" * 16),
            ],
        )


def test_an_assessment_of_an_unknown_demand_is_reported() -> None:
    from hotel_pipeline.schemas import DemandAssessmentManifest

    state = DemandAssessmentManifest(
        hotel_id="h", corpus_digest="c" * 16, demand_digest="d" * 16,
        assessments=[DemandAssessment(demand_id="inexistant", corpus_digest="c" * 16)],
    )
    problems = state.bind(demands_manifest())
    assert any("besoin inconnu" in p for p in problems)


def test_a_state_of_another_hotel_is_reported() -> None:
    from hotel_pipeline.schemas import DemandAssessmentManifest

    state = DemandAssessmentManifest(
        hotel_id="autre", corpus_digest="c" * 16, demand_digest="d" * 16
    )
    assert any("confronté aux besoins" in p for p in state.bind(demands_manifest()))


def test_an_evaluation_must_share_the_intent_of_its_demand() -> None:
    """Évaluer une vue de contexte contre un besoin de bâtiment mesurerait
    l'une avec les exigences de l'autre."""
    from hotel_pipeline.schemas import bind_evaluations

    candidates = CandidateManifest(
        hotel_id="h", candidates=[candidate()],
        evaluations=[
            evaluation(demand_id="front", intent=CaptureIntent.CONTEXT_CAPTURE),
            evaluation(demand_id="inconnu"),
        ],
    )

    problems = bind_evaluations(candidates, demands_manifest())
    assert any("besoin inconnu 'inconnu'" in p for p in problems)
    assert any("≠ 'building_capture' du besoin" in p for p in problems)


def test_a_plan_item_must_rest_on_a_favourable_evaluation() -> None:
    from hotel_pipeline.schemas import bind_plan

    candidates = CandidateManifest(
        hotel_id="h", candidates=[candidate()],
        evaluations=[
            evaluation(demand_id="front", eligibility=Eligibility.REJECTED,
                       rejection_reason="silhouette masquée"),
        ],
    )
    plan = AcquisitionPlan(
        plan_id="p", hotel_id="h",
        acquisitions=[planned("mapillary-1", serves_demands=["front"])],
    )

    problems = bind_plan(plan, candidates, demands_manifest())
    assert any("sans aucune évaluation favorable" in p for p in problems)
    assert any("dont l'évaluation l'a écarté" in p for p in problems)


def test_a_plan_item_without_candidate_is_reported() -> None:
    from hotel_pipeline.schemas import bind_plan

    candidates = CandidateManifest(hotel_id="h")
    plan = AcquisitionPlan(plan_id="p", hotel_id="h", acquisitions=[planned("fantome")])

    assert any("sans candidat correspondant" in p
               for p in bind_plan(plan, candidates, demands_manifest()))


def test_an_unavailable_resolution_is_reported() -> None:
    from hotel_pipeline.schemas import bind_plan

    candidates = CandidateManifest(
        hotel_id="h", candidates=[candidate()], evaluations=[evaluation()]
    )
    plan = AcquisitionPlan(
        plan_id="p", hotel_id="h",
        acquisitions=[planned("mapillary-1", resolution="4096")],
    )

    assert any("résolution '4096' indisponible" in p
               for p in bind_plan(plan, candidates, demands_manifest()))


def test_an_overlap_outside_the_plan_is_reported() -> None:
    """Un recouvrement annoncé avec une image non téléchargée n'existera pas."""
    from hotel_pipeline.schemas import bind_plan

    candidates = CandidateManifest(
        hotel_id="h", candidates=[candidate()], evaluations=[evaluation()]
    )
    plan = AcquisitionPlan(
        plan_id="p", hotel_id="h",
        acquisitions=[planned("mapillary-1", overlap_with=["mapillary-2"])],
    )

    assert any("absent(s) du plan" in p
               for p in bind_plan(plan, candidates, demands_manifest()))


def test_a_coherent_plan_raises_nothing() -> None:
    from hotel_pipeline.schemas import bind_evaluations, bind_plan

    candidates = CandidateManifest(
        hotel_id="h", candidates=[candidate()],
        evaluations=[
            evaluation(demand_id="front"),
            evaluation(demand_id="access-road", intent=CaptureIntent.CONTEXT_CAPTURE),
        ],
    )
    plan = AcquisitionPlan(
        plan_id="p", hotel_id="h",
        acquisitions=[planned("mapillary-1",
                              intents=[CaptureIntent.BUILDING_CAPTURE,
                                       CaptureIntent.CONTEXT_CAPTURE],
                              serves_demands=["front", "access-road"])],
    )

    assert bind_evaluations(candidates, demands_manifest()) == []
    assert bind_plan(plan, candidates, demands_manifest()) == []


# --- intentions multiples ----------------------------------------------------


def test_one_acquisition_can_serve_two_intents() -> None:
    item = planned("m-1", intents=[CaptureIntent.BUILDING_CAPTURE,
                                   CaptureIntent.CONTEXT_CAPTURE],
                   primary_intent=CaptureIntent.BUILDING_CAPTURE,
                   serves_demands=["front", "access-road"])
    assert len(item.intents) == 2


def test_duplicate_or_foreign_intents_are_refused() -> None:
    with pytest.raises(ValueError, match="intentions dupliquées"):
        planned("m-1", intents=[CaptureIntent.BUILDING_CAPTURE,
                                CaptureIntent.BUILDING_CAPTURE])
    with pytest.raises(ValueError, match="intention principale"):
        planned("m-1", intents=[CaptureIntent.BUILDING_CAPTURE],
                primary_intent=CaptureIntent.CONTEXT_CAPTURE)


def test_an_acquisition_must_say_which_demand_it_serves() -> None:
    with pytest.raises(ValueError):
        planned("m-1", serves_demands=[])


# --- identité selon la source ------------------------------------------------


def test_a_mapillary_identity_refuses_a_framing() -> None:
    """Y adjoindre un cadrage fabriquerait une identité sans référent."""
    with pytest.raises(ValueError, match="n'entre pas dans leur identité"):
        capture_identity("mapillary", "123", heading_deg=45.0)


def test_a_street_view_identity_requires_the_whole_framing() -> None:
    with pytest.raises(ValueError, match="cadrage complet"):
        capture_identity("street_view", "PANO1", heading_deg=310.0, fov_deg=60.0)


def test_an_unknown_source_must_declare_its_strategy() -> None:
    from hotel_pipeline.schemas import IdentityStrategy

    with pytest.raises(ValueError, match="déclarez sa stratégie"):
        capture_identity("nouvelle-source", "x")

    assert capture_identity(
        "nouvelle-source", "x", strategy=IdentityStrategy.PROVIDER_IMAGE
    ) == "nouvelle-source-x"


# --- vocabulaire géométrique --------------------------------------------------


def test_the_geometry_speaks_the_official_vocabulary() -> None:
    from hotel_pipeline.schemas import ViewSector

    geometry = CandidateGeometry(view_sector=ViewSector.FRONT, road_ref="way/54581348")
    assert geometry.view_sector is ViewSector.FRONT

    with pytest.raises(ValueError):
        CandidateGeometry(view_sector="façade avant")

    assert "on_road" not in CandidateGeometry.model_fields
    assert "expected_pixels" not in CandidateGeometry.model_fields


def test_an_oversized_target_is_not_clipped_to_the_frame() -> None:
    """Une cible plus large que le champ déborde : l'écrêter effacerait le fait."""
    geometry = CandidateGeometry(unclipped_width_fraction=1.8, expected_width_px=3600)
    assert geometry.unclipped_width_fraction == 1.8


def test_elevation_use_and_provenance_go_together() -> None:
    with pytest.raises(ValueError, match="sans provenance"):
        CandidateGeometry(used_elevation=True)
    with pytest.raises(ValueError, match="sans contrôle d'élévation"):
        CandidateGeometry(elevation_provenance="dtm@20260813T124251Z")

    fine = CandidateGeometry(used_elevation=True,
                             elevation_provenance="dtm@20260813T124251Z")
    assert fine.used_elevation


# --- run_id calendaire --------------------------------------------------------


def test_a_wellformed_but_impossible_date_is_refused(tmp_path, monkeypatch) -> None:
    """`20261340T250000Z` respecte la forme sans désigner aucun instant."""
    from hotel_pipeline.workspace import Workspace

    monkeypatch.setenv("HOTEL_PIPELINE_WORK", str(tmp_path / "work"))
    workspace = Workspace("hotel-test")

    with pytest.raises(acquisition.AcquisitionRefused, match="identifiant d'exécution"):
        acquisition.run_directory(workspace, "20261340T250000Z")
