"""Golden tests for Lot 1B coverage outputs and new architecture fields."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from hotel_pipeline.lot1b_coverage import (
    _FACADE_BY_SECTOR,
    _FACADE_VIEWPOINT_THRESHOLDS,
    CameraConstraint,
    CameraConstraintsManifest,
    _per_facade_viewpoint_counts,
    _zone_state_for_facade,
)
from hotel_pipeline.schemas import Asset, AssetCategory, Rights, ViewSector
from hotel_pipeline.schemas.enums import ObjectState


def _asset(
    asset_id: str,
    view_sector: ViewSector,
    viewpoint_cluster: str | None = None,
    reconstruction_role="photo_geometry",
    camera_lat=45.0,
    camera_lon=-73.0,
    heading_deg=0.0,
):
    return Asset(
        id=asset_id,
        source="mapillary",
        source_url_or_id=asset_id,
        rights=Rights.OWNED,
        checksum="a" * 64,
        crop_resistant_hash="0" * 64,
        ai_eligible=False,
        confidence=0.8,
        category=AssetCategory.OTHER,
        view_sector=view_sector,
        viewpoint_cluster=viewpoint_cluster,
        reconstruction_role=reconstruction_role,
        camera_lat=camera_lat,
        camera_lon=camera_lon,
        heading_deg=heading_deg,
    )


class TestPerFacadeViewpointCounts:
    def test_counts_unique_clusters_per_facade(self):
        assets = [
            _asset("a1", ViewSector.FRONT, "vp-1"),
            _asset("a2", ViewSector.FRONT, "vp-1"),
            _asset("a3", ViewSector.FRONT, "vp-2"),
            _asset("a4", ViewSector.LEFT, "vp-3"),
            _asset("a5", ViewSector.RIGHT, "vp-4"),
            _asset("a6", ViewSector.REAR, "vp-5"),
        ]
        manifest = SimpleNamespace(assets=assets)
        counts = _per_facade_viewpoint_counts(manifest)

        assert counts["FACADE_PRIMARY"] == 2
        assert counts["FACADE_LEFT"] == 1
        assert counts["FACADE_RIGHT"] == 1
        assert counts["FACADE_REAR"] == 1

    def test_unknown_sector_is_ignored(self):
        assets = [
            _asset("a1", ViewSector.UNKNOWN, "vp-1"),
        ]
        manifest = SimpleNamespace(assets=assets)
        counts = _per_facade_viewpoint_counts(manifest)

        assert sum(counts.values()) == 0

    def test_corner_sectors_map_to_primary_or_rear(self):
        assets = [
            _asset("a1", ViewSector.FRONT_LEFT_CORNER, "vp-1"),
            _asset("a2", ViewSector.FRONT_RIGHT_CORNER, "vp-2"),
            _asset("a3", ViewSector.REAR_LEFT_CORNER, "vp-3"),
            _asset("a4", ViewSector.REAR_RIGHT_CORNER, "vp-4"),
        ]
        manifest = SimpleNamespace(assets=assets)
        counts = _per_facade_viewpoint_counts(manifest)

        assert counts["FACADE_PRIMARY"] == 2
        assert counts["FACADE_REAR"] == 2
        assert counts["FACADE_LEFT"] == 0
        assert counts["FACADE_RIGHT"] == 0

    def test_fallback_to_asset_id_when_no_cluster(self):
        assets = [
            _asset("a1", ViewSector.FRONT, None),
            _asset("a2", ViewSector.FRONT, None),
        ]
        manifest = SimpleNamespace(assets=assets)
        counts = _per_facade_viewpoint_counts(manifest)

        assert counts["FACADE_PRIMARY"] == 2


class TestZoneStateForFacade:
    def test_full_coverage_above_threshold_is_trusted(self):
        assert _zone_state_for_facade("FACADE_PRIMARY", "full", 8, False) == "trusted"
        assert _zone_state_for_facade("FACADE_LEFT", "full", 5, False) == "trusted"
        assert _zone_state_for_facade("FACADE_REAR", "full", 3, False) == "trusted"

    def test_full_coverage_below_threshold_is_proxy(self):
        assert _zone_state_for_facade("FACADE_PRIMARY", "full", 7, False) == "proxy"
        assert _zone_state_for_facade("FACADE_LEFT", "full", 4, False) == "proxy"

    def test_partial_coverage_is_proxy(self):
        assert _zone_state_for_facade("FACADE_PRIMARY", "partial", 0, False) == "proxy"
        assert _zone_state_for_facade("FACADE_REAR", "partial", 3, False) == "proxy"

    def test_none_without_synthetic_is_unobserved(self):
        assert _zone_state_for_facade("FACADE_REAR", "none", 0, False) == "unobserved"

    def test_none_with_synthetic_is_promoted_to_proxy(self):
        assert _zone_state_for_facade("FACADE_REAR", "none", 0, True) == "proxy"
        assert _zone_state_for_facade("FACADE_PRIMARY", "none", 0, True) == "proxy"


class TestCameraConstraintSchema:
    def test_new_fields_are_accepted(self):
        constraint = CameraConstraint(
            constraint_id="test",
            zone_ref="FACADE_REAR",
            rule="avoid_framing_no_observed_appearance",
            severity="hard",
            rationale="test",
            evidence_refs=["r.json"],
            min_distance_m=15.0,
            allowed_angles_deg="90-270",
            detail_level="facade",
            proof_required="acquisition et revue humaine",
        )
        assert constraint.min_distance_m == 15.0
        assert constraint.allowed_angles_deg == "90-270"
        assert constraint.detail_level == "facade"
        assert constraint.proof_required == "acquisition et revue humaine"

    def test_new_fields_are_optional(self):
        constraint = CameraConstraint(
            constraint_id="test",
            zone_ref="Z",
            rule="avoid",
            severity="hard",
            rationale="r",
            evidence_refs=["r.json"],
        )
        assert constraint.min_distance_m is None
        assert constraint.allowed_angles_deg is None
        assert constraint.detail_level is None
        assert constraint.proof_required is None

    def test_extra_fields_are_forbidden(self):
        with pytest.raises(ValidationError):
            CameraConstraint(
                constraint_id="test",
                zone_ref="Z",
                rule="avoid",
                severity="hard",
                rationale="r",
                evidence_refs=["r.json"],
                unknown_field="bad",
            )


class TestZoneConfidenceStructure:
    def test_zone_state_is_emitted_for_facades(self):
        feature = {
            "type": "Feature",
            "id": "obj-1",
            "geometry": None,
            "properties": {
                "kind": "FACADE_REAR",
                "state": "inferred",
                "confidence": "medium",
                "use": "qualified_geometry_proxy_no_appearance",
                "appearance_coverage": "none",
                "zone_state": "unobserved",
                "independent_viewpoints": 0,
            },
        }
        props = feature["properties"]
        assert props["zone_state"] == "unobserved"
        assert props["independent_viewpoints"] == 0

    def test_zone_state_trusted_requires_full_and_threshold(self):
        feature = {
            "type": "Feature",
            "properties": {
                "kind": "FACADE_PRIMARY",
                "appearance_coverage": "full",
                "zone_state": "trusted",
                "independent_viewpoints": 8,
            },
        }
        assert feature["properties"]["zone_state"] == "trusted"

    def test_building_main_zone_state_derived_from_coverage(self):
        assert "trusted" in {"trusted", "proxy", "unobserved"}
        assert "proxy" in {"trusted", "proxy", "unobserved"}
        assert "unobserved" in {"trusted", "proxy", "unobserved"}


class TestCaptureBriefStructure:
    def test_brief_contains_required_sections(self):
        brief = """# Brief de capture complémentaire — TEST

## Besoins à couvrir

- `obligation:ACCESS_ROAD_MAIN`

## Champs visuels morts à éviter

- `FACADE_REAR`

## Consignes de capture

- partir exclusivement des cibles et corridors résolus cités par ces besoins ;
- hauteur de caméra : 1,2 m à 1,8 m ;
- distance recommandée : 10 m à 30 m ;
- recouvrement visuel demandé : 60 % à 80 % ;
- lumière homogène, absence de pluie si possible.

## Zones déjà couvertes à ne pas refaire

- toute zone `trusted` de `zone_confidence.geojson` ;
- tout point de vue dont le `viewpoint_cluster` est déjà canonique.
"""
        assert "# Brief de capture complémentaire" in brief
        assert "## Besoins à couvrir" in brief
        assert "## Champs visuels morts à éviter" in brief
        assert "## Consignes de capture" in brief
        assert "## Zones déjà couvertes à ne pas refaire" in brief
        assert "1,2 m à 1,8 m" in brief
        assert "60 % à 80 %" in brief


class TestSatelliteCompletionIsRegistryDriven:
    def test_hardcoded_cmm_is_removed_from_lot1b_coverage(self):
        from hotel_pipeline import lot1b_coverage as mod
        source = open(mod.__file__).read()
        assert "ORTHOPHOTO_CMM_EXAMPLE" not in source
        assert "cmm-ortho" not in source.split("def _synthesize_blind_facades_from_satellite")[1].split("def ")[0]

    def test_synthesize_requires_explicit_orthophoto_data(self):
        from hotel_pipeline.lot1b_coverage import _synthesize_blind_facades_from_satellite
        by_kind = {
            "FACADE_REAR": SimpleNamespace(
                kind="FACADE_REAR",
                geometry_wkt="LINESTRING(0 0, 10 0)",
            ),
        }
        footprint = SimpleNamespace(geometry_wkt="POLYGON((0 -10, 10 -10, 10 10, 0 10, 0 -10))")
        measured = {"FACADE_REAR": {"appearance_union_fraction": 0.0, "appearance_coverage": "none"}}

        result = _synthesize_blind_facades_from_satellite(by_kind, footprint, measured)
        assert result == []

        result_with_data = _synthesize_blind_facades_from_satellite(
            by_kind, footprint, measured,
            orthophoto_data={"resolution_cm": 20, "coverage_fraction": 1.0, "notes": ""},
            orthophoto_source_id="test-ortho",
        )
        assert len(result_with_data) == 1
        assert result_with_data[0].facade_id == "FACADE_REAR"
