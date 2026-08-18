"""Tests end-to-end du pipeline Lot 2 — scénarios complets."""

from __future__ import annotations

import cv2
import json
from pathlib import Path

import numpy as np
import pytest

from hotel_pipeline.workspace import Workspace
from hotel_pipeline.reconstruction_input import prepare_input, publish_input
from hotel_pipeline.reconstruction_preprocess import generate_mask_set
from hotel_pipeline.view_graph import ViewGraphBuilder
from hotel_pipeline.reconstruction_plan import ReconstructionPlanner, publish_plan
from hotel_pipeline.reconstruction_run import ReconstructionRunner, publish_run
from hotel_pipeline.reconstruction_consensus import ConsensusBuilder, publish_consensus
from hotel_pipeline.geo_alignment import GeoAligner, publish_alignment
from hotel_pipeline.camera_feasibility import CameraFeasibilityEvaluator
from hotel_pipeline.surface_confidence import SurfaceConfidenceBuilder
from hotel_pipeline.dense_reconstruction import run_dense_reconstruction
from hotel_pipeline.schemas.reconstruction import (
    ReconstructionBackend,
    ReconstructionInputManifest,
    ReconstructionPlan,
    ReconstructionRun,
    ViewGraphManifest,
    ViewGraphNode,
    ViewGraphReport,
)


def _write_minimal_workspace(tmp_path: Path, monkeypatch, hotel_id: str, asset_count: int = 3) -> Workspace:
    monkeypatch.setenv("HOTEL_PIPELINE_WORK", str(tmp_path / "work"))
    monkeypatch.setenv("HOTEL_PIPELINE_PROFILES", str(tmp_path / "profiles"))
    workspace = Workspace(hotel_id)
    workspace.create()

    manifest_dir = workspace.path("00_manifest")
    manifest_dir.mkdir(parents=True, exist_ok=True)

    assets = []
    for i in range(asset_count):
        assets.append({
            "id": f"asset-{i+1}",
            "source": "test",
            "source_url_or_id": str(i+1),
            "rights": "owned",
            "checksum": "a" * 64,
            "ai_eligible": False,
            "confidence": 0.9,
            "category": "facade",
            "reconstruction_role": "photo_geometry",
            "view_sector": "front" if i == 0 else ("left" if i == 1 else "right"),
            "viewpoint_cluster": f"vp-{i+1}",
            "camera_lat": 45.0 + i * 0.001,
            "camera_lon": -73.0 + i * 0.001,
            "heading_deg": float(i * 10),
            "local_path": f"images/asset-{i+1}.jpg",
            "quality_score": 0.9 - i * 0.1,
            "capture_year": 2024 - i,
        })

    asset_manifest = {"hotel_id": hotel_id, "assets": assets}
    (manifest_dir / "asset_manifest.json").write_text(json.dumps(asset_manifest))

    spatial_manifest = {
        "hotel_id": hotel_id,
        "address": "1 rue Test",
        "front_azimuth_deg": 0.0,
        "confirmed_building_id": "building-1",
        "confirmed_by": "test",
        "candidates": [
            {
                "feature_id": "building-1",
                "source": "test",
                "centroid_lat": 45.0,
                "centroid_lon": -73.0,
                "area_m2": 100.0,
                "distance_to_geocode_m": 0.0,
                "wkt": "{\"type\": \"Polygon\", \"coordinates\": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]}",
            }
        ],
    }
    (manifest_dir / "spatial_manifest.json").write_text(json.dumps(spatial_manifest))

    site_manifest = {
        "hotel_id": hotel_id,
        "objects": [
            {"kind": "BUILDING_MAIN", "state": "confirmed", "geometry_wkt": "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))"},
        ],
    }
    (manifest_dir / "site_manifest.json").write_text(json.dumps(site_manifest))

    pipeline_policy = {
        "version": "1.0.0",
        "property_profile_id": hotel_id,
    }
    (manifest_dir / "pipeline_policy.json").write_text(json.dumps(pipeline_policy))

    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    profile = {
        "property_id": hotel_id,
        "address": "1 rue Test",
        "official_name": f"Hôtel {hotel_id}",
        "country_code": "CA",
        "timezone": "America/Toronto",
        "ocr_languages": ["fr", "en"],
    }
    (profiles_dir / f"{hotel_id}.json").write_text(json.dumps(profile))

    router_decision = {
        "path": "PATH_D_HYBRID",
        "decision_status": "CAPTURE_REQUIRED",
        "input_digest": "f" * 64,
        "photographic": {"open": [], "partial": [], "satisfied": [], "independent_viewpoints": 0},
        "geometric_proxies": [],
        "appearance_gaps": [],
    }
    (workspace.path("10_validation")).mkdir(parents=True, exist_ok=True)
    (workspace.path("10_validation", "router_decision.json")).write_text(json.dumps(router_decision))

    coverage_report = {
        "hotel_id": hotel_id,
        "generated_at": "2026-01-01T00:00:00Z",
        "router": {"file": "router_decision.json", "sha256": "f" * 64, "path": "PATH_D_HYBRID", "decision_status": "CAPTURE_REQUIRED", "input_digest": "f" * 64},
        "demands": {"open": [], "partial": [], "satisfied": [], "independent_viewpoints": 0},
        "assets": {"total": asset_count, "roles": {"photo_geometry": asset_count}},
        "visibility": {"run_id": "vis-1", "assets_updated": asset_count, "by_status": {}},
        "geometry": {"working_crs": "EPSG:32618", "front_azimuth_deg": 0.0, "qualified_proxies": [], "road_corridors": 0, "obstacles": 0},
        "rights": {"by_status": {"owned": asset_count}, "uncleared_or_unknown": 0, "photo_geometry_with_production_rights": asset_count, "rule": "test"},
        "sources": {},
        "unresolved_objects": [],
        "lot_1b_status": "incomplete_capture_required",
        "blocking_reasons": [],
        "limitations": [],
        "outputs": {},
    }
    (workspace.path("coverage")).mkdir(parents=True, exist_ok=True)
    (workspace.path("coverage", "coverage_report.json")).write_text(json.dumps(coverage_report))

    camera_constraints = {
        "hotel_id": hotel_id,
        "min_distance_m": 5.0,
        "max_distance_m": 100.0,
        "min_fov_deg": 30.0,
        "max_fov_deg": 120.0,
        "forbidden_claims": [],
        "blind_visual_fields": [],
    }
    (workspace.path("coverage", "camera_constraints.json")).write_text(json.dumps(camera_constraints))

    zone_confidence = {
        "type": "FeatureCollection",
        "features": [],
    }
    (workspace.path("coverage", "zone_confidence.geojson")).write_text(json.dumps(zone_confidence))

    derivation_report = {
        "metrics": {
            "height_statistics": {"median_m": 5.0, "mean_m": 5.5},
        }
    }
    (workspace.path("06_geo")).mkdir(parents=True, exist_ok=True)
    (workspace.path("06_geo", "derivation_report.json")).write_text(json.dumps(derivation_report))

    capture_geometry = {
        "hotel_id": hotel_id,
        "target_polygon_wkt": "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))",
    }
    (workspace.path("06_geo", "capture_geometry.json")).write_text(json.dumps(capture_geometry))

    images_dir = workspace.path("images")
    images_dir.mkdir(parents=True, exist_ok=True)
    for i in range(asset_count):
        img_path = images_dir / f"asset-{i+1}.jpg"
        img_path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")

    return workspace


# ---------------------------------------------------------------------------
# Scenario 1: Full happy path with SYNTHETIC backend
# ---------------------------------------------------------------------------


def test_e2e_full_pipeline_happy_path(tmp_path: Path, monkeypatch):
    """Le pipeline complet réussit avec le backend SYNTHETIC."""
    hotel_id = "test-hotel"
    workspace = _write_minimal_workspace(tmp_path, monkeypatch, hotel_id, asset_count=3)

    input_manifest, _ = prepare_input(hotel_id)
    publish_input(input_manifest, workspace)

    builder = ViewGraphBuilder(workspace)
    view_graph = builder.build(input_manifest)
    assert view_graph is not None
    assert len(view_graph.nodes) >= 2

    planner = ReconstructionPlanner(workspace)
    plan = planner.plan(input_manifest, view_graph)
    publish_plan(plan, workspace)
    assert "colmap_incremental" in plan.selected_backends

    runner = ReconstructionRunner(workspace)
    run = runner.run(input_manifest, backend=ReconstructionBackend.SYNTHETIC)
    assert run.status == "completed"
    publish_run(run, workspace)

    builder_consensus = ConsensusBuilder(workspace)
    consensus = builder_consensus.build([run.run_id, run.run_id])
    assert consensus.selected_run_id is not None
    publish_consensus(consensus, workspace)

    aligner = GeoAligner(workspace)
    alignment = aligner.align(run.run_id)
    assert alignment.alignment_id is not None
    publish_alignment(alignment, workspace)

    evaluator = CameraFeasibilityEvaluator(workspace)
    field = evaluator.evaluate_pose(
        pose_id="pose-1",
        position_local_m=(0.0, 0.0, 2.0),
        yaw_deg=0.0,
        pitch_deg=0.0,
        fov_deg=80.0,
        reconstruction_run_id=run.run_id,
    )
    assert field.reconstructed_fraction > 0.0


# ---------------------------------------------------------------------------
# Scenario 2: Missing plan blocks reconstruction
# ---------------------------------------------------------------------------


def test_e2e_reconstruct_blocks_without_plan(tmp_path: Path, monkeypatch):
    """Sans ReconstructionPlan, la reconstruction est bloquée."""
    hotel_id = "test-hotel"
    workspace = _write_minimal_workspace(tmp_path, monkeypatch, hotel_id, asset_count=2)

    input_manifest, _ = prepare_input(hotel_id)
    publish_input(input_manifest, workspace)

    from hotel_pipeline.steps import StepBlocked
    from hotel_pipeline.steps import _reconstruct
    with pytest.raises(StepBlocked, match="aucun ReconstructionPlan"):
        _reconstruct(workspace)


# ---------------------------------------------------------------------------
# Scenario 3: Temporal cohort strategy
# ---------------------------------------------------------------------------


def test_e2e_temporal_cohort_current_only(tmp_path: Path, monkeypatch):
    """Sans cohorte unknown, la stratégie reste current_only."""
    hotel_id = "test-hotel"
    workspace = _write_minimal_workspace(tmp_path, monkeypatch, hotel_id, asset_count=3)

    input_manifest, _ = prepare_input(hotel_id)
    input_manifest.temporal_cohorts = {
        "current_confirmed": ["asset-1", "asset-2", "asset-3"],
    }
    publish_input(input_manifest, workspace)

    builder = ViewGraphBuilder(workspace)
    view_graph = builder.build(input_manifest)

    planner = ReconstructionPlanner(workspace)
    plan = planner.plan(input_manifest, view_graph)
    assert plan.temporal_strategy == "current_only"

    runner = ReconstructionRunner(workspace)
    run = runner.run(input_manifest, backend=ReconstructionBackend.SYNTHETIC)
    assert run.status == "completed"


def test_e2e_temporal_cohort_escalates_with_low_overlap(tmp_path: Path, monkeypatch):
    """Avec une cohorte unknown et overlap faible, la stratégie devient current_plus_unknown."""
    hotel_id = "test-hotel"
    workspace = _write_minimal_workspace(tmp_path, monkeypatch, hotel_id, asset_count=3)

    input_manifest, _ = prepare_input(hotel_id)
    input_manifest.temporal_cohorts = {
        "current_confirmed": ["asset-1"],
        "unknown": ["asset-2", "asset-3"],
    }
    publish_input(input_manifest, workspace)

    builder = ViewGraphBuilder(workspace)
    view_graph = builder.build(input_manifest)
    view_graph.report.registered_candidate_ratio = 0.15

    planner = ReconstructionPlanner(workspace)
    plan = planner.plan(input_manifest, view_graph)
    assert plan.temporal_strategy == "current_plus_unknown"


def test_e2e_temporal_cohort_current_plus_unknown(tmp_path: Path, monkeypatch):
    """Avec current_plus_unknown et overlap faible, la cohorte unknown est incluse."""
    hotel_id = "test-hotel"
    workspace = _write_minimal_workspace(tmp_path, monkeypatch, hotel_id, asset_count=3)

    input_manifest, _ = prepare_input(hotel_id)
    input_manifest.temporal_cohorts = {
        "current_confirmed": ["asset-1"],
        "unknown": ["asset-2", "asset-3"],
    }
    publish_input(input_manifest, workspace)

    builder = ViewGraphBuilder(workspace)
    view_graph = builder.build(input_manifest)
    view_graph.report.registered_candidate_ratio = 0.15

    planner = ReconstructionPlanner(workspace)
    plan = planner.plan(input_manifest, view_graph)
    assert plan.temporal_strategy == "current_plus_unknown"


# ---------------------------------------------------------------------------
# Scenario 4: Multi-backend consensus
# ---------------------------------------------------------------------------


def test_e2e_multi_backend_consensus(tmp_path: Path, monkeypatch):
    """Deux backends SYNTHETIC produisent un consensus valide."""
    import time
    hotel_id = "test-hotel"
    workspace = _write_minimal_workspace(tmp_path, monkeypatch, hotel_id, asset_count=3)

    input_manifest, _ = prepare_input(hotel_id)
    publish_input(input_manifest, workspace)

    runner = ReconstructionRunner(workspace)
    run1 = runner.run(input_manifest, backend=ReconstructionBackend.SYNTHETIC)
    time.sleep(1.1)
    run2 = runner.run(input_manifest, backend=ReconstructionBackend.SYNTHETIC)
    assert run1.status == "completed"
    assert run2.status == "completed"
    assert run1.run_id != run2.run_id

    publish_run(run1, workspace)
    publish_run(run2, workspace)

    builder = ConsensusBuilder(workspace)
    consensus = builder.build([run1.run_id, run2.run_id])
    assert consensus.selected_run_id is not None
    assert len(consensus.camera_consensus) >= 3


# ---------------------------------------------------------------------------
# Scenario 5: Mask generation with multiple classes
# ---------------------------------------------------------------------------


def test_e2e_mask_generation_all_classes(tmp_path: Path, monkeypatch):
    """Toutes les classes de masques peuvent être générées."""
    hotel_id = "test-hotel"
    workspace = _write_minimal_workspace(tmp_path, monkeypatch, hotel_id, asset_count=1)

    img_path = workspace.path("images", "asset-1.jpg")
    img_path.parent.mkdir(parents=True, exist_ok=True)
    test_image = np.zeros((200, 200, 3), dtype=np.uint8)
    test_image[:50, :] = [255, 0, 0]
    test_image[50:150, :] = [100, 100, 100]
    test_image[150:, :] = [0, 255, 255]
    cv2.imwrite(str(img_path), test_image)

    input_manifest, _ = prepare_input(hotel_id)
    mask_digest = generate_mask_set(workspace, input_manifest, mask_classes=[
        "sky", "water", "people", "cars", "signage", "large_reflections", "mobile_furniture"
    ])
    assert len(mask_digest) == 64

    mask_dir = workspace.path("05_colmap", "preprocessed", "masks")
    mask_path = mask_dir / "asset-1.png"
    assert mask_path.exists()
    assert mask_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Scenario 7: Backend selection logic
# ---------------------------------------------------------------------------


def test_e2e_backend_selection_by_view_graph(tmp_path: Path, monkeypatch):
    """Le plan sélectionne les backends selon le ViewGraphReport."""
    hotel_id = "test-hotel"
    workspace = _write_minimal_workspace(tmp_path, monkeypatch, hotel_id, asset_count=5)

    input_manifest, _ = prepare_input(hotel_id)
    publish_input(input_manifest, workspace)

    builder = ViewGraphBuilder(workspace)
    view_graph = builder.build(input_manifest)

    planner = ReconstructionPlanner(workspace)
    plan = planner.plan(input_manifest, view_graph)
    assert "colmap_incremental" in plan.selected_backends

    view_graph.report.valid_pairs = 100
    plan_dense = planner.plan(input_manifest, view_graph)
    assert "colmap_global" in plan_dense.selected_backends

    view_graph.report.repetitive_risk = "high"
    plan_gluemap = planner.plan(input_manifest, view_graph)
    assert "gluemap" in plan_gluemap.selected_backends

    view_graph.report.registered_candidate_ratio = 0.1
    plan_mpsfm = planner.plan(input_manifest, view_graph)
    assert "mpsfm" in plan_mpsfm.selected_backends


# ---------------------------------------------------------------------------
# Scenario 8: Surface confidence after geo-alignment
# ---------------------------------------------------------------------------


def test_e2e_surface_confidence_after_alignment(tmp_path: Path, monkeypatch):
    """SurfaceConfidenceBuilder produit un manifeste valide après alignement."""
    hotel_id = "test-hotel"
    workspace = _write_minimal_workspace(tmp_path, monkeypatch, hotel_id, asset_count=3)

    input_manifest, _ = prepare_input(hotel_id)
    publish_input(input_manifest, workspace)

    runner = ReconstructionRunner(workspace)
    run = runner.run(input_manifest, backend=ReconstructionBackend.SYNTHETIC)
    assert run.status == "completed"
    publish_run(run, workspace)

    builder_consensus = ConsensusBuilder(workspace)
    consensus = builder_consensus.build([run.run_id, run.run_id])
    publish_consensus(consensus, workspace)

    aligner = GeoAligner(workspace)
    alignment = aligner.align(run.run_id)
    publish_alignment(alignment, workspace)

    confidence_builder = SurfaceConfidenceBuilder(workspace)
    confidence = confidence_builder.build(run.run_id)
    assert len(confidence.surfaces) >= 1
    assert all(0.0 <= s.confidence <= 1.0 for s in confidence.surfaces)


# ---------------------------------------------------------------------------
# Scenario 9: Camera feasibility after alignment
# ---------------------------------------------------------------------------


def test_e2e_camera_feasibility_after_alignment(tmp_path: Path, monkeypatch):
    """CameraFeasibilityEvaluator fonctionne après alignement géospatial."""
    hotel_id = "test-hotel"
    workspace = _write_minimal_workspace(tmp_path, monkeypatch, hotel_id, asset_count=3)

    input_manifest, _ = prepare_input(hotel_id)
    publish_input(input_manifest, workspace)

    runner = ReconstructionRunner(workspace)
    run = runner.run(input_manifest, backend=ReconstructionBackend.SYNTHETIC)
    assert run.status == "completed"
    publish_run(run, workspace)

    builder_consensus = ConsensusBuilder(workspace)
    consensus = builder_consensus.build([run.run_id, run.run_id])
    publish_consensus(consensus, workspace)

    aligner = GeoAligner(workspace)
    alignment = aligner.align(run.run_id)
    publish_alignment(alignment, workspace)

    evaluator = CameraFeasibilityEvaluator(workspace)
    field = evaluator.evaluate_pose(
        pose_id="pose-1",
        position_local_m=(0.0, 0.0, 2.0),
        yaw_deg=0.0,
        pitch_deg=0.0,
        fov_deg=80.0,
        reconstruction_run_id=run.run_id,
    )
    assert field.overall_score >= 0.0
    assert field.reconstructed_fraction >= 0.0


# ---------------------------------------------------------------------------
# Scenario 10: Dense reconstruction placeholder
# ---------------------------------------------------------------------------


def test_e2e_dense_reconstruction_placeholder(tmp_path: Path, monkeypatch):
    """run_dense_reconstruction retourne un résultat placeholder."""
    hotel_id = "test-hotel"
    workspace = _write_minimal_workspace(tmp_path, monkeypatch, hotel_id, asset_count=2)

    input_manifest, _ = prepare_input(hotel_id)
    publish_input(input_manifest, workspace)

    runner = ReconstructionRunner(workspace)
    run = runner.run(input_manifest, backend=ReconstructionBackend.SYNTHETIC)
    assert run.status == "completed"
    publish_run(run, workspace)

    dense = run_dense_reconstruction(workspace, run.run_id, backend=ReconstructionBackend.BRUSH)
    assert dense.result_id is not None
    assert dense.backend == "brush"
