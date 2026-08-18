"""Tests pour les modules Lot 2 : view_graph, reconstruction_run, consensus, alignement, dense, faisabilité."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hotel_pipeline.geo_alignment import GeoAligner, publish_alignment
from hotel_pipeline.reconstruction_consensus import ConsensusBuilder, publish_consensus
from hotel_pipeline.reconstruction_run import ReconstructionRunner, publish_run
from hotel_pipeline.view_graph import ViewGraphBuilder, generate_mask_set
from hotel_pipeline.workspace import Workspace
from hotel_pipeline.schemas.reconstruction import (
    ReconstructionConsensusReport,
    ReconstructionInputManifest,
    ReconstructionRun,
    GeoAlignmentManifest,
)


def _write_minimal_workspace(tmp_path: Path, hotel_id: str, asset_count: int = 1) -> Workspace:
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
            "view_sector": "front",
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
        "working_crs": "EPSG:32618",
        "front_azimuth_deg": 0.0,
        "confirmed_building_id": "building-1",
        "candidates": [{"feature_id": "building-1", "centroid_lat": 45.0, "centroid_lon": -73.0}],
    }
    (manifest_dir / "spatial_manifest.json").write_text(json.dumps(spatial_manifest))

    site_manifest = {
        "hotel_id": hotel_id,
        "objects": [
            {"kind": "BUILDING_MAIN", "state": "confirmed", "geometry_wkt": "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))"},
        ],
    }
    (manifest_dir / "site_manifest.json").write_text(json.dumps(site_manifest))

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

    images_dir = workspace.path("05_colmap", "preprocessed", "images")
    images_dir.mkdir(parents=True, exist_ok=True)
    for i in range(asset_count):
        (images_dir / f"asset-{i+1}.jpg").write_bytes(b"\x00" * 10)

    return workspace


# ---------------------------------------------------------------------------
# View Graph
# ---------------------------------------------------------------------------


def test_view_graph_builder_requires_min_two_assets(tmp_path: Path):
    hotel_id = "test-hotel"
    workspace = _write_minimal_workspace(tmp_path, hotel_id, asset_count=1)

    from hotel_pipeline.reconstruction_input import prepare_input
    input_manifest = prepare_input(hotel_id)
    builder = ViewGraphBuilder(workspace)
    with pytest.raises(ValueError, match="au moins deux assets"):
        builder.build(input_manifest)


def test_generate_mask_set_returns_sha256(tmp_path: Path):
    hotel_id = "test-hotel"
    workspace = _write_minimal_workspace(tmp_path, hotel_id, asset_count=1)

    from hotel_pipeline.reconstruction_input import prepare_input
    input_manifest = prepare_input(hotel_id)
    mask_digest = generate_mask_set(workspace, input_manifest)
    assert len(mask_digest) == 64


# ---------------------------------------------------------------------------
# Reconstruction Run
# ---------------------------------------------------------------------------


def test_reconstruction_run_publishes_and_loads(tmp_path: Path):
    workspace = Workspace("test-hotel")
    workspace.create()

    input_manifest = ReconstructionInputManifest(
        reconstruction_input_id="input-1",
        asset_manifest_digest="a" * 64,
        spatial_manifest_digest="b" * 64,
        site_manifest_digest="c" * 64,
        coverage_digest="d" * 64,
        router_decision_digest="e" * 64,
        selected_asset_ids=["asset-1"],
    )

    runner = ReconstructionRunner(workspace)
    run = runner.run(input_manifest)
    assert run.run_id.startswith("run-colmap_incremental-input-1-")
    path = publish_run(run, workspace)
    assert path.exists()

    loaded = ReconstructionRun.model_validate_json(path.read_text("utf-8"))
    assert loaded.run_id == run.run_id
    assert loaded.backend == "colmap_incremental"


# ---------------------------------------------------------------------------
# Consensus
# ---------------------------------------------------------------------------


def test_consensus_builder_requires_two_completed_runs(tmp_path: Path):
    workspace = Workspace("test-hotel")
    workspace.create()

    builder = ConsensusBuilder(workspace)
    with pytest.raises(ValueError, match="au moins deux runs"):
        builder.build(["run-1"])


def test_publish_consensus_writes_json(tmp_path: Path):
    workspace = Workspace("test-hotel")
    workspace.create()

    report = ReconstructionConsensusReport(
        consensus_id="consensus-1",
        reconstruction_input_id="input-1",
        run_ids=["run-1", "run-2"],
    )
    path = publish_consensus(report, workspace)
    assert path.exists()
    assert path.parent.name == "consensus"
    loaded = ReconstructionConsensusReport.model_validate_json(path.read_text("utf-8"))
    assert loaded.consensus_id == "consensus-1"


# ---------------------------------------------------------------------------
# Geo Alignment
# ---------------------------------------------------------------------------


def test_geo_aligner_publishes_alignment(tmp_path: Path):
    workspace = Workspace("test-hotel")
    workspace.create()

    aligner = GeoAligner(workspace)
    manifest = aligner.align("run-1")
    assert manifest.alignment_id.startswith("align-run-1-")
    path = publish_alignment(manifest, workspace)
    assert path.exists()
    assert path.parent.name == "alignment"


# ---------------------------------------------------------------------------
# Dense Reconstruction
# ---------------------------------------------------------------------------


def test_dense_reconstruction_placeholder():
    from hotel_pipeline.dense_reconstruction import DenseReconstructionResult

    result = DenseReconstructionResult(
        result_id="dense-1",
        reconstruction_run_id="run-1",
        backend="brush",
        status="pending",
        error="backend brush non implémenté dans cette phase",
    )
    assert result.result_id == "dense-1"
    assert result.backend == "brush"


# ---------------------------------------------------------------------------
# Camera Feasibility
# ---------------------------------------------------------------------------


def test_camera_feasibility_evaluator_basic():
    from hotel_pipeline.camera_feasibility import CameraFeasibilityEvaluator

    evaluator = CameraFeasibilityEvaluator(Workspace("test-hotel"))
    field = evaluator.evaluate_pose(
        pose_id="pose-1",
        position_local_m=(10.0, 10.0, 10.0),
        yaw_deg=0.0,
        pitch_deg=0.0,
        fov_deg=80.0,
        reconstructed_fraction=0.8,
        proxy_fraction=0.1,
        unknown_fraction=0.1,
    )
    assert field.overall_score > 0.0
    assert field.visible_surface_confidence == pytest.approx(0.85, abs=0.01)


def test_validated_camera_path_builder():
    from hotel_pipeline.camera_feasibility import build_validated_camera_path

    workspace = Workspace("test-hotel")
    workspace.create()
    path = build_validated_camera_path(workspace, "run-1")
    assert len(path.poses) >= 8
    assert path.simulation_only is True
    assert "validé par faisabilité" in path.derivation
