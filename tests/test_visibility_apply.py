"""Projection d'une exécution de visibilité (Lot 1B V2, étape 4).

Ce qui est éprouvé : rien n'est promu, rien d'humain n'est touché, et une
application interrompue se rejoue sans remuter.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from shapely.geometry import Polygon

from hotel_pipeline.geo import visibility_apply as projection
from hotel_pipeline.geo import visibility_engine as engine
from hotel_pipeline.geo.visibility_run import base_manifest_digest
from hotel_pipeline.schemas import (
    DEFAULT_POLICY,
    Asset,
    AssetManifest,
    ClusterRole,
    ReviewDecision,
    ReviewEntry,
    ReviewStatus,
    Subject,
    TemporalStatus,
    ViewSector,
)
from hotel_pipeline.schemas.assets import DECISION_STATUS, VISIBILITY_PROJECTED_FIELDS, VISIBILITY_OF
from hotel_pipeline.schemas.visibility import (
    ElevationSource,
    FramingAssessment,
    VisibilityRun,
)

POLICY = DEFAULT_POLICY
ORIGIN = (0.0, 0.0)
TARGET = Polygon([(-10, 50), (10, 50), (10, 70), (-10, 70)])


def wall(feature_id: str, y0: float, y1: float, **kwargs) -> engine.Obstacle:
    return engine.Obstacle(
        feature_id, Polygon([(-12, y0), (12, y0), (12, y1), (-12, y1)]), **kwargs
    )


def asset(asset_id: str = "mapillary-1", **overrides) -> Asset:
    fields = dict(
        id=asset_id, source="mapillary", source_url_or_id="1", rights="open_data",
        ai_eligible=False, confidence=0.5, category="facade", checksum="a" * 64,
        camera_lat=45.5730, camera_lon=-73.4433, heading_deg=45.0,
        view_sector=ViewSector.FRONT, cluster_role=ClusterRole.CANONICAL,
        temporal_status=TemporalStatus.AFTER_EVENT, subjects=[Subject.BUILDING],
    )
    fields.update(overrides)
    return Asset(**fields)


def assessment_for(subject: str, obstacles=(), **kwargs):
    return engine.assess(
        f"vis-{subject}", subject, "BUILDING_MAIN", ORIGIN, TARGET,
        list(obstacles), POLICY.visibility, **kwargs
    )


def elevation() -> ElevationSource:
    return ElevationSource(
        kind="raster", role="target_ground", artifact_id="dtm@r1",
        path="06_geo/derived/dtm.tif", sha256="a" * 64,
        horizontal_crs="EPSG:2950", vertical_crs="CGVD 1928",
        sampling_method="rasterio.sample",
    )


def run_for(manifest: AssetManifest, assessments, digests=None) -> VisibilityRun:
    fields = dict(
        run_id="20260813T120000000000Z", hotel_id="h",
        engine_version=engine.ENGINE_VERSION, method="uniform_angular_cells",
        parameters={"max_angular_step_deg": "0.25"},
        capture_geometry_digest="g", policy_digest="p", site_manifest_digest="s",
        asset_files_digest="f", asset_manifest_digest=base_manifest_digest(manifest),
        target_digest="t", obstacles_digest="o", road_geometry_digest="r",
        elevation_sources=[elevation()],
        assessments=list(assessments),
        framings=[
            FramingAssessment(
                assessment_id=a.assessment_id, subject_ref=a.subject_ref,
                horizontal_reason="intrinsèques absentes",
                vertical_reason="inclinaison inconnue",
            )
            for a in assessments
        ],
    )
    fields.update(digests or {})
    return VisibilityRun(**fields)


def current_digests(run: VisibilityRun) -> dict[str, str]:
    return {
        "policy": run.policy_digest,
        "capture_geometry": run.capture_geometry_digest,
        "site_manifest": run.site_manifest_digest,
        "asset_files": run.asset_files_digest,
        "obstacles": run.obstacles_digest,
        "roads": run.road_geometry_digest,
    }


# --- ce que la projection écrit ----------------------------------------------


def test_an_unproven_occlusion_is_cleared() -> None:
    """Les 29 anciennes affirmations reposaient sur une intersection en plan."""
    manifest = AssetManifest(
        hotel_id="h", assets=[asset(occluded_by="way/999")]
    )
    assessment = assessment_for("mapillary-1", [wall("OBSTACLE_1", 20, 25)])

    report, projected = projection.project(
        manifest, run_for(manifest, [assessment]), "d1", POLICY
    )
    updated = projected.assets[0]

    assert updated.occluded_by is None
    assert updated.occlusion_risk_by == ["OBSTACLE_1"]
    assert updated.occlusion_blocked_by == []
    assert updated.line_of_sight_status == "at_risk"
    assert report.former_occlusions[0]["was_occluded_by"] == "way/999"


def test_a_proven_sole_blocker_is_kept() -> None:
    tower = wall("TOUR", 20, 25, height_m=30.0, ground_m=10.0)
    assessment = assessment_for(
        "mapillary-1", [tower],
        camera=engine.CameraVertical(ground_m=10.0, height_above_ground_m=2.5,
                                     provenance="lidar"),
        target_vertical=engine.TargetVertical(ground_m=10.0, height_m=12.0,
                                              provenance="rasters"),
    )
    manifest = AssetManifest(hotel_id="h", assets=[asset()])

    _, projected = projection.project(manifest, run_for(manifest, [assessment]), "d1", POLICY)

    assert projected.assets[0].occluded_by == "TOUR"
    assert projected.assets[0].occlusion_blocked_by == ["TOUR"]
    assert projected.assets[0].line_of_sight_status == "blocked"


def test_a_partial_block_never_fills_the_singular_field() -> None:
    """Le champ singulier ne saurait pas dire « la moitié de la façade »."""
    tower = engine.Obstacle(
        "TOUR", Polygon([(-12, 20), (0, 20), (0, 25), (-12, 25)]),
        height_m=30.0, ground_m=10.0,
    )
    assessment = assessment_for(
        "mapillary-1", [tower],
        camera=engine.CameraVertical(ground_m=10.0, height_above_ground_m=2.5),
        target_vertical=engine.TargetVertical(ground_m=10.0, height_m=12.0),
    )
    manifest = AssetManifest(hotel_id="h", assets=[asset()])

    _, projected = projection.project(manifest, run_for(manifest, [assessment]), "d1", POLICY)

    assert 0 < assessment.proven_blocked_fraction < 1
    assert projected.assets[0].occluded_by is None
    assert projected.assets[0].occlusion_blocked_by == ["TOUR"]


def test_two_blockers_leave_the_singular_field_empty() -> None:
    """Un blocage intégral partagé n'a pas de responsable unique."""
    left = engine.Obstacle(
        "GAUCHE", Polygon([(-12, 20), (0, 20), (0, 25), (-12, 25)]),
        height_m=30.0, ground_m=10.0,
    )
    right = engine.Obstacle(
        "DROITE", Polygon([(0, 20), (12, 20), (12, 25), (0, 25)]),
        height_m=30.0, ground_m=10.0,
    )
    assessment = assessment_for(
        "mapillary-1", [left, right],
        camera=engine.CameraVertical(ground_m=10.0, height_above_ground_m=2.5),
        target_vertical=engine.TargetVertical(ground_m=10.0, height_m=12.0),
    )
    manifest = AssetManifest(hotel_id="h", assets=[asset()])

    _, projected = projection.project(manifest, run_for(manifest, [assessment]), "d1", POLICY)

    assert assessment.proven_blocked_fraction == 1.0
    assert projected.assets[0].occluded_by is None
    assert projected.assets[0].occlusion_blocked_by == ["DROITE", "GAUCHE"]


def test_the_decisive_case_no_eye_height_no_block() -> None:
    """Le test décisif : hauteurs LiDAR connues, hauteur d'œil retirée.

    Aucun blocage ne doit être projeté — c'est exactement l'état du corpus
    réel, où ni Mapillary ni Street View ne publient la hauteur de leur
    capteur.
    """
    tower = wall("TOUR", 20, 25, height_m=30.0, ground_m=10.0)
    assessment = assessment_for(
        "mapillary-1", [tower],
        camera=engine.CameraVertical(ground_m=10.0, height_above_ground_m=None),
        target_vertical=engine.TargetVertical(ground_m=10.0, height_m=12.0),
    )
    manifest = AssetManifest(hotel_id="h", assets=[asset(occluded_by="way/999")])

    _, projected = projection.project(manifest, run_for(manifest, [assessment]), "d1", POLICY)
    updated = projected.assets[0]

    assert assessment.proven_blocked_fraction == 0.0
    assert updated.occluded_by is None
    assert updated.occlusion_blocked_by == []
    assert updated.occlusion_risk_by == ["TOUR"]
    assert "hauteur de caméra" in assessment.missing_vertical


# --- ce que la projection ne touche pas ---------------------------------------


def test_nothing_human_or_semantic_is_touched() -> None:
    entry = ReviewEntry(
        decision=ReviewDecision.CONFIRMED, decided_by="Claude (Opus 5)",
        rationale="pylône lisible", evidence=["enseigne"], reviewed_checksum="a" * 64,
    )
    reviewed = asset(
        occluded_by="way/999",
        review_history=[entry],
        target_visibility_decision=entry.decision,
        review_status=DECISION_STATUS[entry.decision],
        target_building_visible=VISIBILITY_OF[entry.decision],
        reviewer=entry.decided_by, review_rationale=entry.rationale,
        review_evidence=entry.evidence,
        sees_building=True, subject_scores={"building": 0.99},
    )
    manifest = AssetManifest(hotel_id="h", assets=[reviewed])
    assessment = assessment_for("mapillary-1", [wall("OBSTACLE_1", 20, 25)])

    _, projected = projection.project(manifest, run_for(manifest, [assessment]), "d1", POLICY)
    updated = projected.assets[0]

    assert updated.sees_building is True
    assert updated.target_building_visible is True
    assert updated.review_history == reviewed.review_history
    assert updated.review_status is reviewed.review_status
    assert updated.geometry_suitability is reviewed.geometry_suitability
    assert updated.subject_scores == reviewed.subject_scores


def test_only_the_declared_fields_change() -> None:
    manifest = AssetManifest(hotel_id="h", assets=[asset(occluded_by="way/999")])
    assessment = assessment_for("mapillary-1", [wall("OBSTACLE_1", 20, 25)])

    _, projected = projection.project(manifest, run_for(manifest, [assessment]), "d1", POLICY)

    before = json.loads(manifest.assets[0].model_dump_json())
    after = json.loads(projected.assets[0].model_dump_json())
    changed = {k for k in after if before.get(k) != after[k]}

    assert changed <= VISIBILITY_PROJECTED_FIELDS
    # L'empreinte de base est justement celle qui ignore ces champs.
    assert base_manifest_digest(projected) == base_manifest_digest(manifest)


def test_a_role_change_aborts_everything() -> None:
    """La visibilité géométrique ne promeut rien : un rôle qui bouge est un bug."""
    manifest = AssetManifest(hotel_id="h", assets=[asset()])
    assessment = assessment_for("mapillary-1")

    import hotel_pipeline.roles as roles_module

    original = roles_module.role_for
    calls = {"n": 0}

    def shifting(asset_, policy=POLICY):  # noqa: ANN001
        calls["n"] += 1
        from hotel_pipeline.schemas import ReconstructionRole

        if calls["n"] > 1:
            return ReconstructionRole.PHOTO_GEOMETRY, "promu à tort"
        return original(asset_, policy)

    roles_module.role_for = shifting
    try:
        with pytest.raises(projection.ApplicationRefused, match="le rôle passerait"):
            projection.project(manifest, run_for(manifest, [assessment]), "d1", POLICY)
    finally:
        roles_module.role_for = original


# --- vérifications préalables ---------------------------------------------------


def test_a_changed_manifest_perishes_the_run() -> None:
    manifest = AssetManifest(hotel_id="h", assets=[asset()])
    run = run_for(manifest, [assessment_for("mapillary-1")])

    # Une revue humaine survient après la mesure.
    entry = ReviewEntry(
        decision=ReviewDecision.REJECTED, decided_by="Claude (Opus 5)",
        rationale="bâtiment voisin", evidence=["enseigne Tetra Tech"],
        reviewed_checksum="a" * 64,
    )
    reviewed = AssetManifest(
        hotel_id="h",
        assets=[asset(
            review_history=[entry], target_visibility_decision=entry.decision,
            review_status=DECISION_STATUS[entry.decision],
            target_building_visible=VISIBILITY_OF[entry.decision],
            reviewer=entry.decided_by, review_rationale=entry.rationale,
            review_evidence=entry.evidence,
        )],
    )

    problems = projection.verify(run, reviewed, "h", current_digests(run))
    assert any("le manifeste a changé" in p for p in problems)


def test_a_missing_assessment_is_refused() -> None:
    manifest = AssetManifest(hotel_id="h", assets=[asset(), asset("mapillary-2")])
    run = run_for(manifest, [assessment_for("mapillary-1")])

    problems = projection.verify(run, manifest, "h", current_digests(run))
    assert any("évaluations et assets situés diffèrent" in p for p in problems)


def test_a_foreign_hotel_is_refused() -> None:
    manifest = AssetManifest(hotel_id="h", assets=[asset()])
    run = run_for(manifest, [assessment_for("mapillary-1")])

    assert any(
        "appliquée à" in p
        for p in projection.verify(run, manifest, "autre-hotel", current_digests(run))
    )


def test_a_diverging_digest_is_refused() -> None:
    manifest = AssetManifest(hotel_id="h", assets=[asset()])
    run = run_for(manifest, [assessment_for("mapillary-1")])
    current = current_digests(run)
    current["obstacles"] = "autre-empreinte"

    assert any("obstacles_digest" in p for p in projection.verify(run, manifest, "h", current))


# --- idempotence ------------------------------------------------------------------


def test_replaying_the_same_run_is_idempotent() -> None:
    manifest = AssetManifest(hotel_id="h", assets=[asset(occluded_by="way/999")])
    run = run_for(manifest, [assessment_for("mapillary-1", [wall("OBSTACLE_1", 20, 25)])])

    _, projected = projection.project(manifest, run, "d1", POLICY)
    idempotent, divergent = projection.already_applied(projected, run)

    assert idempotent is True
    assert divergent == []


def test_a_hand_edited_field_is_not_idempotent() -> None:
    """Rejouer doit reconstruire un reçu, non couvrir une retouche manuelle."""
    manifest = AssetManifest(hotel_id="h", assets=[asset()])
    run = run_for(manifest, [assessment_for("mapillary-1", [wall("OBSTACLE_1", 20, 25)])])
    _, projected = projection.project(manifest, run, "d1", POLICY)

    tampered = AssetManifest(
        hotel_id="h",
        assets=[projected.assets[0].model_copy(update={"occlusion_risk_by": []})],
    )
    idempotent, divergent = projection.already_applied(tampered, run)

    assert idempotent is False
    assert divergent == ["mapillary-1.occlusion_risk_by"]


def test_another_run_already_applied_is_refused() -> None:
    manifest = AssetManifest(hotel_id="h", assets=[asset(visibility_run_id="autre-run")])
    run = run_for(manifest, [assessment_for("mapillary-1")])

    idempotent, divergent = projection.already_applied(manifest, run)

    assert idempotent is False
    assert any("déjà appliquée" in problem for problem in divergent)


def test_the_receipt_name_is_deterministic() -> None:
    """Une commande interrompue doit retrouver son reçu."""
    assert projection.receipt_name("r1", "d1") == "visibility_application_r1_d1.json"


def test_a_reconstructed_receipt_says_what_it_cannot_know() -> None:
    """Rejoué après coup, il lit un manifeste déjà projeté.

    Zéro occultation retirée n'y signifie pas qu'il n'y en avait aucune.
    """
    report = projection.ApplicationReport(run_id="r", run_digest="d")
    assert report.as_dict()["note"] is None

    report.status = "already_applied"
    assert "n'était plus observable" in report.as_dict()["note"]
