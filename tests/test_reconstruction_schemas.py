"""Tests pour le Lot 2 — ViewGraph, preprocessing, consensus, alignement, dense, faisabilité caméra."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hotel_pipeline.schemas.reconstruction import (
    AlignmentAnchor,
    CameraConsensusEntry,
    GeoAlignmentManifest,
    PairEvidence,
    ReconstructionConsensusReport,
    ReconstructionInputManifest,
    ReconstructionPlan,
    ReconstructionRun,
    ScenePackageType,
    SurfaceConfidence,
    SurfaceConfidenceManifest,
    ValidatedCameraPath,
    ViewGraphManifest,
    ViewGraphNode,
    ViewGraphReport,
)
from hotel_pipeline.workspace import Workspace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_minimal_workspace(tmp_path: Path, hotel_id: str) -> Workspace:
    workspace = Workspace(hotel_id)
    workspace.create()

    manifest_dir = workspace.path("00_manifest")
    manifest_dir.mkdir(parents=True, exist_ok=True)

    asset_manifest = {
        "hotel_id": hotel_id,
        "assets": [
            {
                "id": "asset-1",
                "source": "test",
                "source_url_or_id": "1",
                "rights": "owned",
                "checksum": "a" * 64,
                "ai_eligible": False,
                "confidence": 0.9,
                "category": "facade",
                "reconstruction_role": "photo_geometry",
                "view_sector": "front",
                "viewpoint_cluster": "vp-1",
                "camera_lat": 45.0,
                "camera_lon": -73.0,
                "heading_deg": 0.0,
                "file_path": "images/asset-1.jpg",
                "quality_score": 0.8,
                "capture_year": 2024,
            },
            {
                "id": "asset-2",
                "source": "test",
                "source_url_or_id": "2",
                "rights": "owned",
                "checksum": "b" * 64,
                "ai_eligible": False,
                "confidence": 0.9,
                "category": "facade",
                "reconstruction_role": "photo_geometry",
                "view_sector": "front",
                "viewpoint_cluster": "vp-1",
                "camera_lat": 45.0,
                "camera_lon": -73.0,
                "heading_deg": 10.0,
                "file_path": "images/asset-2.jpg",
                "quality_score": 0.7,
                "capture_year": 2023,
            },
        ],
    }
    (manifest_dir / "asset_manifest.json").write_text(json.dumps(asset_manifest))

    spatial_manifest = {
        "hotel_id": hotel_id,
        "working_crs": "EPSG:32618",
        "front_azimuth_deg": 0.0,
        "confirmed_building_id": "building-1",
        "candidates": [
            {"feature_id": "building-1", "centroid_lat": 45.0, "centroid_lon": -73.0},
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
        "assets": {"total": 2, "roles": {"photo_geometry": 2}},
        "visibility": {"run_id": "vis-1", "assets_updated": 2, "by_status": {}},
        "geometry": {"working_crs": "EPSG:32618", "front_azimuth_deg": 0.0, "qualified_proxies": [], "road_corridors": 0, "obstacles": 0},
        "rights": {"by_status": {"owned": 2}, "uncleared_or_unknown": 0, "photo_geometry_with_production_rights": 2, "rule": "test"},
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
    for name in ("asset-1.jpg", "asset-2.jpg"):
        (images_dir / name).write_bytes(b"\x00" * 10)

    return workspace


# ---------------------------------------------------------------------------
# Schemas round-trip
# ---------------------------------------------------------------------------


def test_view_graph_node_round_trips():
    node = ViewGraphNode(asset_id="a1", quality_score=0.8)
    payload = node.model_dump(mode="json")
    recovered = ViewGraphNode.model_validate(payload)
    assert recovered.asset_id == "a1"
    assert recovered.quality_score == 0.8


def test_pair_evidence_defaults():
    pair = PairEvidence(image_a="a1", image_b="a2")
    assert pair.status == "failed"
    assert pair.inliers == 0
    assert pair.overlap_estimate == 0.0


def test_view_graph_report_fields():
    report = ViewGraphReport(
        images_selected=10,
        pairs_tested=20,
        valid_pairs=5,
        largest_component=8,
    )
    assert report.registered_candidate_ratio == 0.0
    assert report.repetitive_risk == "none"


def test_reconstruction_run_schema():
    run = ReconstructionRun(
        run_id="run-1",
        reconstruction_input_id="input-1",
        backend="colmap_incremental",
        status="completed",
        metrics={"registered_ratio": 0.85},
    )
    payload = run.model_dump(mode="json")
    recovered = ReconstructionRun.model_validate(payload)
    assert recovered.run_id == "run-1"
    assert recovered.metrics["registered_ratio"] == 0.85


def test_reconstruction_consensus_report_requires_multiple_runs():
    with pytest.raises(Exception):  # Pydantic validation error
        ReconstructionConsensusReport(
            consensus_id="c1",
            reconstruction_input_id="input-1",
            run_ids=["run-1"],
        )


def test_camera_consensus_entry_defaults():
    entry = CameraConsensusEntry(asset_id="a1", backends=["colmap_incremental"])
    assert entry.confidence == "none"
    assert entry.aberrants == []


def test_surface_confidence_defaults():
    sc = SurfaceConfidence(zone_id="FACADE_PRIMARY")
    assert sc.confidence == 0.0
    assert sc.reprojection_error == 0.0
    assert sc.extrapolation_penalty == 0.0


def test_validated_camera_path_requires_poses():
    with pytest.raises(ValueError, match="at least"):
        ValidatedCameraPath(
            path_id="path-1",
            reconstruction_run_id="run-1",
            poses=[],
            derivation="test",
        )


def test_scene_package_type_values():
    assert ScenePackageType.HYBRID_PROXY.value == "hybrid_proxy"
    assert ScenePackageType.RECONSTRUCTED_PHOTO_FIRST.value == "reconstructed_photo_first"


def test_reconstruction_gate_status_values():
    from hotel_pipeline.schemas.reconstruction import ReconstructionGateStatus
    assert ReconstructionGateStatus.ENVIRONMENT_3D_READY.value == "ENVIRONMENT_3D_READY"


def test_alignment_anchor_values():
    assert AlignmentAnchor.FOOTPRINT.value == "footprint"
    assert AlignmentAnchor.LIDAR_ROOF.value == "lidar_roof"


# ---------------------------------------------------------------------------
# GeoAlignmentManifest
# ---------------------------------------------------------------------------


def test_geo_alignment_manifest_round_trip():
    manifest = GeoAlignmentManifest(
        alignment_id="align-1",
        source_reconstruction_id="run-1",
        scale=1.0,
        rotation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        translation={"x": 0.0, "y": 0.0, "z": 0.0},
        horizontal_crs="EPSG:32618",
        footprint_error_m=0.5,
        roof_height_error_m=0.3,
        alignment_rmse_m=0.6,
        anchors=["footprint", "lidar_roof"],
    )
    payload = manifest.model_dump(mode="json")
    recovered = GeoAlignmentManifest.model_validate(payload)
    assert recovered.alignment_id == "align-1"
    assert recovered.scale == 1.0
    assert recovered.alignment_rmse_m == 0.6
