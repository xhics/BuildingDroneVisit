"""Tests pour le Lot 2 — ReconstructionInputManifest et sélection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hotel_pipeline.schemas.reconstruction import (
    ReconstructionInputManifest,
    ReconstructionSelection,
    ReconstructionSelectionManifest,
)
from hotel_pipeline.workspace import Workspace


def test_reconstruction_input_manifest_round_trips():
    manifest = ReconstructionInputManifest(
        reconstruction_input_id="recon-test-20260101T000000Z",
        asset_manifest_digest="a" * 64,
        spatial_manifest_digest="b" * 64,
        site_manifest_digest="c" * 64,
        coverage_digest="d" * 64,
        router_decision_digest="e" * 64,
        selected_asset_ids=["asset-1", "asset-2"],
        excluded_asset_ids=["asset-3"],
        selection_reasons={"asset-3": "droits non clarifiés"},
        mask_set_digest="f" * 64,
        allowed_backends=["colmap_incremental", "gluemap"],
    )
    payload = manifest.model_dump(mode="json")
    recovered = ReconstructionInputManifest.model_validate(payload)
    assert recovered.reconstruction_input_id == manifest.reconstruction_input_id
    assert recovered.selected_asset_ids == ["asset-1", "asset-2"]
    assert recovered.excluded_asset_ids == ["asset-3"]
    assert recovered.selection_reasons == {"asset-3": "droits non clarifiés"}
    assert recovered.allowed_backends == ["colmap_incremental", "gluemap"]


def test_reconstruction_input_rejects_overlap():
    with pytest.raises(ValueError, match="présents à la fois dans selected et excluded"):
        ReconstructionInputManifest(
            reconstruction_input_id="recon-test",
            asset_manifest_digest="a" * 64,
            spatial_manifest_digest="b" * 64,
            site_manifest_digest="c" * 64,
            coverage_digest="d" * 64,
            router_decision_digest="e" * 64,
            selected_asset_ids=["asset-1"],
            excluded_asset_ids=["asset-1", "asset-2"],
        )


def test_reconstruction_selection_manifest_requires_selected():
    with pytest.raises(ValueError, match="au moins un asset doit être sélectionné"):
        ReconstructionSelectionManifest(
            reconstruction_input_id="recon-test",
            selections=[
                ReconstructionSelection(
                    asset_id="asset-1",
                    decision="rejected",
                    reason="flou",
                )
            ],
        )


def test_reconstruction_input_schema_version_is_one():
    manifest = ReconstructionInputManifest(
        reconstruction_input_id="recon-test",
        asset_manifest_digest="a" * 64,
        spatial_manifest_digest="b" * 64,
        site_manifest_digest="c" * 64,
        coverage_digest="d" * 64,
        router_decision_digest="e" * 64,
        selected_asset_ids=["asset-1"],
    )
    assert manifest.contract_version == 1


def test_reconstruction_input_default_backend():
    manifest = ReconstructionInputManifest(
        reconstruction_input_id="recon-test",
        asset_manifest_digest="a" * 64,
        spatial_manifest_digest="b" * 64,
        site_manifest_digest="c" * 64,
        coverage_digest="d" * 64,
        router_decision_digest="e" * 64,
        selected_asset_ids=["asset-1"],
    )
    assert manifest.allowed_backends == ["colmap_incremental"]


def test_prepare_input_requires_workspace(tmp_path):
    hotel_id = "test-hotel"
    workspace = Workspace(hotel_id)
    workspace.create()
    from hotel_pipeline.schemas import AssetManifest, Asset, Rights, ReconstructionRole
    from hotel_pipeline.schemas.enums import AssetCategory

    asset = Asset(
        id="asset-1",
        source="test",
        source_url_or_id="1",
        rights=Rights.OWNED,
        checksum="a" * 64,
        ai_eligible=False,
        confidence=0.9,
        category=AssetCategory.FACADE,
        reconstruction_role=ReconstructionRole.PHOTO_GEOMETRY,
    )
    workspace.write_assets(AssetManifest(hotel_id=hotel_id, assets=[asset]))

    required_files = {
        "spatial_manifest": workspace.spatial_path,
        "site_manifest": workspace.site_path,
        "router_decision": workspace.path("10_validation/router_decision.json"),
        "coverage": workspace.path("coverage/coverage_report.json"),
    }
    for name, path in required_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")

    router_path = required_files["router_decision"]
    router_path.write_text(json.dumps({
        "path": "PATH_D_HYBRID",
        "decision_status": "CAPTURE_REQUIRED",
        "input_digest": "f" * 64,
        "photographic": {"open": [], "partial": [], "satisfied": [], "independent_viewpoints": 0},
        "geometric_proxies": [],
        "appearance_gaps": [],
    }))

    from hotel_pipeline.reconstruction_input import prepare_input
    manifest = prepare_input(hotel_id)
    assert manifest.reconstruction_input_id.startswith(f"recon-{hotel_id}-")
    assert manifest.selected_asset_ids == ["asset-1"]
    assert manifest.excluded_asset_ids == []
    assert manifest.asset_manifest_digest == manifest.asset_manifest_digest  # non-empty SHA-256
    assert len(manifest.asset_manifest_digest) == 64
