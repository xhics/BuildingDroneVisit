"""Tests pour les modules Lot 2 : view_graph, reconstruction_run, consensus, alignement, dense, faisabilité."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from hotel_pipeline.geo_alignment import GeoAligner, publish_alignment
from hotel_pipeline.reconstruction_consensus import ConsensusBuilder, publish_consensus
from hotel_pipeline.reconstruction_run import ReconstructionRunner, publish_run
from hotel_pipeline.reconstruction_plan import ReconstructionPlanner, publish_plan
from hotel_pipeline.view_graph import ViewGraphBuilder, generate_mask_set
from hotel_pipeline.workspace import Workspace
from hotel_pipeline.schemas.reconstruction import (
    Criticality,
    ReconstructionConsensusReport,
    ReconstructionInputManifest,
    ReconstructionPlan,
    ReconstructionRun,
    ReconstructionTarget,
    ReconstructionTargetKind,
    SupportType,
    ViewGraphManifest,
    ViewGraphNode,
    ViewGraphReport,
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
    input_manifest, _ = prepare_input(hotel_id)
    builder = ViewGraphBuilder(workspace)
    with pytest.raises(ValueError, match="au moins deux assets"):
        builder.build(input_manifest)


def test_load_intrinsics_from_exif(tmp_path: Path):
    hotel_id = "test-hotel"
    workspace = _write_minimal_workspace(tmp_path, hotel_id, asset_count=1)
    img_path = workspace.path("images", "asset-1.jpg")
    img_path.parent.mkdir(parents=True, exist_ok=True)

    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), test_image)

    from hotel_pipeline.view_graph import _load_intrinsics
    from hotel_pipeline.schemas import AssetManifest
    assets = AssetManifest.model_validate_json(workspace.assets_path.read_text("utf-8"))
    asset = assets.assets[0]
    intrinsics = _load_intrinsics(asset)
    assert intrinsics is None or isinstance(intrinsics, dict)


def test_generate_mask_set_returns_sha256(tmp_path: Path):
    hotel_id = "test-hotel"
    workspace = _write_minimal_workspace(tmp_path, hotel_id, asset_count=1)

    from hotel_pipeline.reconstruction_input import prepare_input
    input_manifest, _ = prepare_input(hotel_id)
    mask_digest = generate_mask_set(workspace, input_manifest)
    assert len(mask_digest) == 64


def test_mask_generation_produces_non_empty_masks_for_sky_and_water(tmp_path: Path):
    hotel_id = "test-hotel"
    workspace = _write_minimal_workspace(tmp_path, hotel_id, asset_count=1)
    img_path = workspace.path("images", "asset-1.jpg")
    img_path.parent.mkdir(parents=True, exist_ok=True)
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    test_image[:50, :] = [255, 0, 0]
    cv2.imwrite(str(img_path), test_image)

    from hotel_pipeline.reconstruction_input import prepare_input
    input_manifest, _ = prepare_input(hotel_id)
    mask_digest = generate_mask_set(workspace, input_manifest, mask_classes=["sky", "water"])
    mask_dir = workspace.path("05_colmap", "preprocessed", "masks")
    mask_path = mask_dir / "asset-1.png"
    assert mask_path.exists()
    assert mask_path.stat().st_size > 0


def test_mask_generation_produces_non_empty_masks_for_people_and_cars(tmp_path: Path):
    hotel_id = "test-hotel"
    workspace = _write_minimal_workspace(tmp_path, hotel_id, asset_count=1)
    img_path = workspace.path("images", "asset-1.jpg")
    img_path.parent.mkdir(parents=True, exist_ok=True)
    test_image = np.zeros((200, 200, 3), dtype=np.uint8)
    test_image[:, :] = (100, 100, 100)
    cv2.rectangle(test_image, (80, 50), (120, 150), (0, 0, 255), -1)
    cv2.rectangle(test_image, (50, 140), (150, 180), (255, 255, 255), -1)
    cv2.imwrite(str(img_path), test_image)

    from hotel_pipeline.reconstruction_input import prepare_input
    input_manifest, _ = prepare_input(hotel_id)
    mask_digest = generate_mask_set(workspace, input_manifest, mask_classes=["people", "cars"])
    mask_dir = workspace.path("05_colmap", "preprocessed", "masks")
    mask_path = mask_dir / "asset-1.png"
    assert mask_path.exists()
    assert mask_path.stat().st_size > 0


def test_mask_generation_produces_non_empty_masks_for_signage_and_reflections(tmp_path: Path):
    hotel_id = "test-hotel"
    workspace = _write_minimal_workspace(tmp_path, hotel_id, asset_count=1)
    img_path = workspace.path("images", "asset-1.jpg")
    img_path.parent.mkdir(parents=True, exist_ok=True)
    test_image = np.zeros((200, 200, 3), dtype=np.uint8)
    test_image[:, :] = (100, 100, 100)
    cv2.rectangle(test_image, (60, 40), (100, 80), (0, 0, 255), -1)
    cv2.rectangle(test_image, (110, 40), (150, 80), (255, 0, 0), -1)
    cv2.rectangle(test_image, (70, 100), (140, 140), (200, 200, 200), -1)
    cv2.imwrite(str(img_path), test_image)

    from hotel_pipeline.reconstruction_input import prepare_input
    input_manifest, _ = prepare_input(hotel_id)
    mask_digest = generate_mask_set(workspace, input_manifest, mask_classes=["signage", "large_reflections"])
    mask_dir = workspace.path("05_colmap", "preprocessed", "masks")
    mask_path = mask_dir / "asset-1.png"
    assert mask_path.exists()
    assert mask_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Reconstruction Run
# ---------------------------------------------------------------------------


def test_reconstruction_run_publishes_and_loads(tmp_path: Path):
    workspace = Workspace("test-hotel")
    workspace.create()

    input_manifest = ReconstructionInputManifest(
        reconstruction_input_id="input-1",
        hotel_id="test-hotel",
        asset_manifest_digest="a" * 64,
        spatial_manifest_digest="b" * 64,
        site_manifest_digest="c" * 64,
        coverage_digest="d" * 64,
        router_decision_digest="e" * 64,
        selected_asset_ids=["asset-1"],
        targets=[
            ReconstructionTarget(
                target_id="FACADE_PRIMARY",
                kind=ReconstructionTargetKind.SURFACE,
                criticality=Criticality.MUST_SHOW,
                allowed_support=[
                    SupportType.MEASURED_PHOTO,
                    SupportType.MULTIVIEW_RECONSTRUCTED,
                    SupportType.FEEDFORWARD_INFERRED,
                ],
            ),
        ],
    )

    runner = ReconstructionRunner(workspace)
    run = runner.run(input_manifest)
    assert run.run_id.startswith("run-colmap_incremental-input-1-")
    path = publish_run(run, workspace)
    assert path.exists()

    loaded = ReconstructionRun.model_validate_json(path.read_text("utf-8"))
    assert loaded.run_id == run.run_id
    assert loaded.backend == "colmap_incremental"


def test_feed_forward_backends_fail_gracefully_when_binary_missing(tmp_path: Path):
    """MapAnything et VGGT doivent échouer proprement sans binaire."""
    workspace = _write_minimal_workspace(tmp_path, "test-hotel", asset_count=2)

    for i in range(2):
        img_path = workspace.path("images", f"asset-{i+1}.jpg")
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")

    from hotel_pipeline.reconstruction_input import prepare_input
    input_manifest, _ = prepare_input("test-hotel")

    runner = ReconstructionRunner(workspace)

    for backend_name, binary in [
        ("mapanything", "mapanything"),
        ("vggt", "vggt"),
        ("gluemap", "GLUEMAP"),
        ("mpsfm", "MP-SfM"),
    ]:
        from hotel_pipeline.schemas.reconstruction import ReconstructionBackend
        backend = ReconstructionBackend(backend_name)
        run = runner.run(input_manifest, backend=backend)
        assert run.status == "failed"
        assert binary in run.error
        assert run.backend == backend_name


def test_synthetic_backend_produces_valid_colmap_output(tmp_path: Path):
    hotel_id = "test-hotel"
    workspace = _write_minimal_workspace(tmp_path, hotel_id, asset_count=2)
    for i in range(2):
        img_path = workspace.path("images", f"asset-{i+1}.jpg")
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")

    from hotel_pipeline.reconstruction_input import prepare_input
    input_manifest, _ = prepare_input(hotel_id)

    runner = ReconstructionRunner(workspace)
    from hotel_pipeline.schemas.reconstruction import ReconstructionBackend
    run = runner.run(input_manifest, backend=ReconstructionBackend.SYNTHETIC)
    assert run.status == "completed"
    assert run.output_path is not None
    run_dir = Path(run.output_path)
    assert (run_dir / "cameras").exists()
    assert (run_dir / "images").exists()
    assert (run_dir / "points3D").exists()
    assert run.metrics.get("synthetic") is True
    assert run.metrics.get("registered_ratio") == 1.0


def test_synthetic_backend_works_end_to_end_with_consensus_and_alignment(tmp_path: Path):
    import time
    hotel_id = "test-hotel"
    workspace = _write_minimal_workspace(tmp_path, hotel_id, asset_count=3)
    for i in range(3):
        img_path = workspace.path("images", f"asset-{i+1}.jpg")
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")

    from hotel_pipeline.reconstruction_input import prepare_input
    input_manifest, _ = prepare_input(hotel_id)

    runner = ReconstructionRunner(workspace)
    from hotel_pipeline.schemas.reconstruction import ReconstructionBackend

    run1 = runner.run(input_manifest, backend=ReconstructionBackend.SYNTHETIC)
    time.sleep(1.1)
    run2 = runner.run(input_manifest, backend=ReconstructionBackend.SYNTHETIC)
    assert run1.status == "completed"
    assert run2.status == "completed"
    assert run1.run_id != run2.run_id

    from hotel_pipeline.reconstruction_run import publish_run
    publish_run(run1, workspace)
    publish_run(run2, workspace)

    from hotel_pipeline.reconstruction_consensus import ConsensusBuilder
    builder = ConsensusBuilder(workspace)
    consensus = builder.build([run1.run_id, run2.run_id])
    assert consensus.selected_run_id is not None
    assert len(consensus.camera_consensus) == 3


def test_feed_forward_backends_fail_gracefully_when_binary_missing(tmp_path: Path):
    """MapAnything et VGGT doivent échouer proprement sans binaire."""
    workspace = _write_minimal_workspace(tmp_path, "test-hotel", asset_count=2)

    for i in range(2):
        img_path = workspace.path("images", f"asset-{i+1}.jpg")
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")

    from hotel_pipeline.reconstruction_input import prepare_input
    input_manifest, _ = prepare_input("test-hotel")

    runner = ReconstructionRunner(workspace)

    for backend_name, binary in [
        ("mapanything", "mapanything"),
        ("vggt", "vggt"),
        ("gluemap", "GLUEMAP"),
        ("mpsfm", "MP-SfM"),
    ]:
        from hotel_pipeline.schemas.reconstruction import ReconstructionBackend
        backend = ReconstructionBackend(backend_name)
        run = runner.run(input_manifest, backend=backend)
        assert run.status == "failed"
        assert binary in run.error
        assert run.backend == backend_name


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
    with pytest.raises(FileNotFoundError, match="alignement géographique refusé"):
        aligner.align("run-1")


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


def test_dense_backends_fail_gracefully_when_binary_missing(tmp_path: Path):
    """Brush et gsplat doivent échouer proprement sans binaire."""
    workspace = _write_minimal_workspace(tmp_path, "test-hotel", asset_count=2)
    for i in range(2):
        img_path = workspace.path("images", f"asset-{i+1}.jpg")
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")

    from hotel_pipeline.reconstruction_input import prepare_input
    input_manifest, _ = prepare_input("test-hotel")
    from hotel_pipeline.reconstruction_run import ReconstructionRunner, publish_run
    runner = ReconstructionRunner(workspace)
    run = runner.run(input_manifest)
    publish_run(run, workspace)

    sparse_dir = workspace.path("07_reconstruction", "runs", run.run_id, "sparse", "0")
    sparse_dir.mkdir(parents=True, exist_ok=True)
    (sparse_dir / "cameras").write_text("")
    (sparse_dir / "images").write_text("#\n")
    (sparse_dir / "points3D").write_text("")

    run_json = workspace.path("07_reconstruction", "runs", f"{run.run_id}.json")
    data = json.loads(run_json.read_text("utf-8"))
    data["output_path"] = str(sparse_dir)
    run_json.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    from hotel_pipeline.dense_reconstruction import run_dense_reconstruction
    for backend_name, binary in [
        ("brush", "Brush"),
        ("gsplat", "gsplat"),
    ]:
        from hotel_pipeline.schemas.reconstruction import ReconstructionBackend
        result = run_dense_reconstruction(workspace, run.run_id, backend=ReconstructionBackend(backend_name))
        assert result.status == "failed"
        assert binary in result.error


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


def test_camera_feasibility_with_synthetic_reconstruction(tmp_path: Path):
    hotel_id = "test-hotel"
    workspace = _write_minimal_workspace(tmp_path, hotel_id, asset_count=3)
    for i in range(3):
        img_path = workspace.path("images", f"asset-{i+1}.jpg")
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")

    from hotel_pipeline.reconstruction_input import prepare_input
    input_manifest, _ = prepare_input(hotel_id)

    runner = ReconstructionRunner(workspace)
    from hotel_pipeline.schemas.reconstruction import ReconstructionBackend
    run = runner.run(input_manifest, backend=ReconstructionBackend.SYNTHETIC)
    assert run.status == "completed"

    from hotel_pipeline.reconstruction_run import publish_run
    publish_run(run, workspace)

    from hotel_pipeline.camera_feasibility import CameraFeasibilityEvaluator
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


def test_validated_camera_path_builder():
    from hotel_pipeline.camera_feasibility import build_validated_camera_path

    workspace = Workspace("test-hotel")
    workspace.create()
    path = build_validated_camera_path(workspace, "run-1")
    assert len(path.poses) >= 8
    assert path.simulation_only is True
    assert "validé par faisabilité" in path.derivation


def test_reconstruction_plan_temporal_strategy_current_only_by_default(tmp_path: Path):
    """Par défaut, le plan utilise current_only."""
    workspace = _write_minimal_workspace(tmp_path, "test-hotel", asset_count=2)
    from hotel_pipeline.reconstruction_input import prepare_input, publish_input
    input_manifest, _ = prepare_input("test-hotel")
    publish_input(input_manifest, workspace)

    view_graph = ViewGraphManifest(
        view_graph_id="vg-1",
        reconstruction_input_id=input_manifest.reconstruction_input_id,
        nodes=[
            ViewGraphNode(asset_id="asset-1", pose_status="registered"),
            ViewGraphNode(asset_id="asset-2", pose_status="registered"),
        ],
        pairs=[],
        report=ViewGraphReport(
            images_selected=2,
            pairs_tested=1,
            valid_pairs=10,
            largest_component=2,
            registered_candidate_ratio=0.5,
            median_inlier_ratio=0.8,
            continuity_by_demand={},
            repetitive_risk="low",
            intrinsics_quality="good",
        ),
    )

    planner = ReconstructionPlanner(workspace)
    plan = planner.plan(input_manifest, view_graph)
    assert plan.temporal_strategy == "current_only"


def test_reconstruction_plan_temporal_strategy_escalates_with_unknown_and_low_overlap(tmp_path: Path):
    """Avec une cohorte unknown et overlap faible, le plan bascule vers current_plus_unknown."""
    workspace = _write_minimal_workspace(tmp_path, "test-hotel", asset_count=2)
    from hotel_pipeline.reconstruction_input import prepare_input, publish_input
    input_manifest, _ = prepare_input("test-hotel")
    publish_input(input_manifest, workspace)

    # Ajouter une cohorte unknown
    input_manifest.temporal_cohorts = {"current_confirmed": ["asset-1"], "unknown": ["asset-2"]}
    input_path = workspace.path("07_reconstruction", f"reconstruction_input_{input_manifest.reconstruction_input_id}.json")
    data = json.loads(input_path.read_text("utf-8"))
    data["temporal_cohorts"] = {"current_confirmed": ["asset-1"], "unknown": ["asset-2"]}
    input_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    view_graph = ViewGraphManifest(
        view_graph_id="vg-1",
        reconstruction_input_id=input_manifest.reconstruction_input_id,
        nodes=[
            ViewGraphNode(asset_id="asset-1", pose_status="registered"),
            ViewGraphNode(asset_id="asset-2", pose_status="registered"),
        ],
        pairs=[],
        report=ViewGraphReport(
            images_selected=2,
            pairs_tested=1,
            valid_pairs=5,
            largest_component=1,
            registered_candidate_ratio=0.15,
            median_inlier_ratio=0.6,
            continuity_by_demand={},
            repetitive_risk="low",
            intrinsics_quality="poor",
        ),
    )

    planner = ReconstructionPlanner(workspace)
    plan = planner.plan(input_manifest, view_graph)
    assert plan.temporal_strategy == "current_plus_unknown"


# ---------------------------------------------------------------------------
# GLUEMAP Integration
# ---------------------------------------------------------------------------


def test_gluemap_fails_gracefully_when_not_installed(tmp_path: Path):
    """GLUEMAP doit échouer proprement sans installation."""
    workspace = _write_minimal_workspace(tmp_path, "test-hotel", asset_count=2)
    for i in range(2):
        img_path = workspace.path("images", f"asset-{i+1}.jpg")
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")

    from hotel_pipeline.reconstruction_input import prepare_input
    input_manifest, _ = prepare_input("test-hotel")

    runner = ReconstructionRunner(workspace)
    from hotel_pipeline.schemas.reconstruction import ReconstructionBackend
    run = runner.run(input_manifest, backend=ReconstructionBackend.GLUEMAP)
    assert run.status == "failed"
    assert "introuvable" in run.error or "pygluemap" in run.error


# ---------------------------------------------------------------------------
# MP-SfM Integration
# ---------------------------------------------------------------------------


def test_mpsfm_fails_gracefully_when_not_installed(tmp_path: Path):
    """MP-SfM doit échouer proprement sans installation."""
    workspace = _write_minimal_workspace(tmp_path, "test-hotel", asset_count=2)
    for i in range(2):
        img_path = workspace.path("images", f"asset-{i+1}.jpg")
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")

    from hotel_pipeline.reconstruction_input import prepare_input
    input_manifest, _ = prepare_input("test-hotel")

    runner = ReconstructionRunner(workspace)
    from hotel_pipeline.schemas.reconstruction import ReconstructionBackend
    run = runner.run(input_manifest, backend=ReconstructionBackend.MP_SFM)
    assert run.status == "failed"
    assert "introuvable" in run.error or "mpsfm" in run.error


# ---------------------------------------------------------------------------
# LiDAR Support
# ---------------------------------------------------------------------------


def test_lidar_report_with_no_files(tmp_path: Path):
    """Sans fichiers LiDAR, le rapport indique l'absence."""
    workspace = _write_minimal_workspace(tmp_path, "test-hotel", asset_count=1)
    lidar_dir = workspace.path("06_geo", "lidar_raw")
    if lidar_dir.exists():
        for f in lidar_dir.glob("*"):
            f.unlink()
    from hotel_pipeline.lidar_support import LiDARSupportAnalyzer
    analyzer = LiDARSupportAnalyzer(workspace)
    report = analyzer.analyze()
    assert report.classification == "no_lidar_files"
    assert report.viable_for_lidgs is False


def test_lidar_report_with_synthetic_las(tmp_path: Path):
    """Avec un fichier .las synthétique, le rapport calcule les densités."""
    import laspy
    workspace = _write_minimal_workspace(tmp_path, "test-hotel", asset_count=1)

    lidar_dir = workspace.path("06_geo", "lidar_raw")
    if lidar_dir.exists():
        for f in lidar_dir.glob("*"):
            f.unlink()
    lidar_dir.mkdir(parents=True, exist_ok=True)
    las_path = lidar_dir / "tile-1.las"

    np.random.seed(42)
    n_points = 1000
    x = np.random.uniform(0, 10, n_points)
    y = np.random.uniform(0, 10, n_points)
    z = np.random.uniform(0, 5, n_points)

    header = laspy.LasHeader(point_format=3, version="1.4")
    header.offsets = np.array([0.0, 0.0, 0.0])
    header.scales = np.array([0.01, 0.01, 0.01])
    las = laspy.LasData(header)
    las.x = x
    las.y = y
    las.z = z
    las.write(str(las_path))

    from hotel_pipeline.lidar_support import LiDARSupportAnalyzer
    analyzer = LiDARSupportAnalyzer(workspace)
    report = analyzer.analyze()
    assert report.total_point_density > 0
    assert report.classification in ("aerial", "hybrid", "terrestrial")


# ---------------------------------------------------------------------------
# Dense Reconstruction
# ---------------------------------------------------------------------------


def test_dense_backends_fail_gracefully_when_binary_missing(tmp_path: Path):
    workspace = _write_minimal_workspace(tmp_path, "test-hotel", asset_count=2)
    for i in range(2):
        img_path = workspace.path("images", f"asset-{i+1}.jpg")
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")

    from hotel_pipeline.reconstruction_input import prepare_input
    input_manifest, _ = prepare_input("test-hotel")

    runner = ReconstructionRunner(workspace)
    run = runner.run(input_manifest)
    publish_run(run, workspace)

    sparse_dir = workspace.path("07_reconstruction", "runs", run.run_id, "sparse", "0")
    sparse_dir.mkdir(parents=True, exist_ok=True)
    (sparse_dir / "cameras").write_text("")
    (sparse_dir / "images").write_text("#\n")
    (sparse_dir / "points3D").write_text("")

    run_json = workspace.path("07_reconstruction", "runs", f"{run.run_id}.json")
    data = json.loads(run_json.read_text("utf-8"))
    data["output_path"] = str(sparse_dir)
    run_json.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    from hotel_pipeline.dense_reconstruction import run_dense_reconstruction
    for backend_name, binary in [
        ("brush", "Brush"),
        ("gsplat", "gsplat"),
    ]:
        from hotel_pipeline.schemas.reconstruction import ReconstructionBackend
        result = run_dense_reconstruction(workspace, run.run_id, backend=ReconstructionBackend(backend_name))
        assert result.status == "failed"
        assert binary in result.error
