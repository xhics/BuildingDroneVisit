"""Contrat des livrables finaux de couverture du Lot 1B."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pydantic
import pytest
from shapely.geometry import LineString
from typer.testing import CliRunner

from hotel_pipeline.cli import app
from hotel_pipeline.geo.catalog import CoverageState, SOURCES
from hotel_pipeline.lot1b_coverage import (
    _blind_field_kinds,
    _positionless_kinds,
    AcquisitionState,
    CameraConstraint,
    CameraConstraintsManifest,
    ContextManifest,
    DedupRobustnessEvidence,
    ObjectRecheck,
    _as_wgs84,
    _completion_findings,
    _no_claim_kinds,
    _robust_dedup_is_current,
    _router_decision,
    _sha256,
    _source_result,
)
from hotel_pipeline.dedup_levels import robust_input_digest
from hotel_pipeline.schemas import Asset, AssetCategory, Rights
from hotel_pipeline.schemas.policy import DEFAULT_POLICY
from hotel_pipeline.policy_facets import Facet, facet_digest
from hotel_pipeline.schemas.enums import ObjectState
from hotel_pipeline.schemas.geometry import GeometryResolutionStatus
from hotel_pipeline.workspace import Workspace


def _source(source_id):
    return next(source for source in SOURCES if source.source_id == source_id)


def test_cmm_is_context_coverage_not_parcel_evidence() -> None:
    result = _source_result(_source("cmm-ortho"), None)

    assert result.coverage_state is CoverageState.COVERED
    assert result.acquisition_state is AcquisitionState.AVAILABLE_NOT_ACQUIRED
    assert "PROPERTY_PARCEL" in result.cannot_establish
    assert result.establishes == []


def test_cadastre_requires_manual_acquisition_and_stays_unacquired() -> None:
    result = _source_result(_source("cadastre-quebec"), None)

    assert result.coverage_state is CoverageState.MANUAL_ACQUISITION_REQUIRED
    assert result.acquisition_state is AcquisitionState.MANUAL_ACQUISITION_REQUIRED
    assert "PROPERTY_PARCEL" in result.establishes
    assert any("aucune géométrie" in limit for limit in result.limitations)


def test_a_manual_acquisition_cannot_claim_another_coverage_state() -> None:
    payload = _source_result(_source("cadastre-quebec"), None).model_dump()
    payload["coverage_state"] = "covered"
    with pytest.raises(pydantic.ValidationError, match="acquisition manuelle"):
        type(_source_result(_source("cadastre-quebec"), None)).model_validate(payload)


def test_rechecking_without_new_evidence_cannot_promote_an_object() -> None:
    with pytest.raises(pydantic.ValidationError, match="ne promeut pas"):
        ObjectRecheck(
            kind="PROPERTY_PARCEL",
            state_before=ObjectState.UNRESOLVED,
            state_after_review=ObjectState.INFERRED,
            geometry_available=True,
            evidence_checked=["ancienne preuve"],
            finding="aucune donnée nouvelle",
            next_action="acquérir le cadastre",
        )


def test_projected_facade_is_serialised_as_wgs84_not_metres() -> None:
    projected = LineString([(309_190.0, 5_048_275.0), (309_205.0, 5_048_288.0)])
    geographic = _as_wgs84(projected, "EPSG:2950")

    assert -74 < geographic.bounds[0] < -72
    assert 45 < geographic.bounds[1] < 46


def _standing_fixture():
    router = {
        "site": {"by_standing": {
            "known_not_targetable": ["ACCESS_ROAD_MAIN", "PROPERTY_SIGN", "ROOFLINE_MAIN"],
            "unresolved": ["PROPERTY_PARCEL"],
        }},
        "geometric_proxies": [{
            "qualified": True, "covered_objects": ["ROOFLINE_MAIN"],
        }],
    }
    geometry = SimpleNamespace(geometries=[SimpleNamespace(
        feature_id="ACCESS_ROAD_MAIN",
        resolution_status=GeometryResolutionStatus.RESOLVED,
    )])
    return router, geometry


def _site_with(states: dict):
    from hotel_pipeline.schemas.enums import ObjectState

    return SimpleNamespace(objects=[
        SimpleNamespace(kind=kind, state=ObjectState(state))
        for kind, state in states.items()
    ])


def test_positionless_kinds_exclude_qualified_proxies_and_resolved_geometry() -> None:
    router, geometry = _standing_fixture()

    assert _positionless_kinds(router, geometry) == ["PROPERTY_PARCEL", "PROPERTY_SIGN"]


def test_no_claim_keeps_only_what_is_not_established() -> None:
    """Un objet établi sans contour n'est pas un objet dont on ne peut rien dire."""
    router, geometry = _standing_fixture()
    site = _site_with({"PROPERTY_SIGN": "confirmed"})

    assert _no_claim_kinds(router, geometry, site) == ["PROPERTY_PARCEL"]


def test_a_blind_visual_field_is_established_but_never_photographed() -> None:
    """L'enseigne existe et se contourne ; la parcelle ne s'affirme pas."""
    router, geometry = _standing_fixture()
    site = _site_with({"PROPERTY_SIGN": "confirmed"})

    assert _blind_field_kinds(router, geometry, site) == ["PROPERTY_SIGN"]


def test_an_unestablished_object_is_never_a_blind_field() -> None:
    """On ne contourne pas ce dont l'existence même manque : on le tait."""
    router, geometry = _standing_fixture()
    site = _site_with({})

    assert _blind_field_kinds(router, geometry, site) == []
    assert _no_claim_kinds(router, geometry, site) == [
        "PROPERTY_PARCEL", "PROPERTY_SIGN",
    ]


def test_camera_constraint_is_closed_and_carries_evidence() -> None:
    constraint = CameraConstraint(
        constraint_id="c", zone_ref="ROOFLINE_MAIN", rule="avoid",
        severity="hard", rationale="lacune", evidence_refs=["rapport.json"],
    )
    assert constraint.evidence_refs
    with pytest.raises(pydantic.ValidationError):
        CameraConstraint.model_validate({**constraint.model_dump(), "threshold": 0.9})


def test_context_requires_the_closed_set_of_input_digests() -> None:
    with pytest.raises(pydantic.ValidationError, match="empreintes de contexte"):
        ContextManifest(
            hotel_id="h", generated_at="now", input_digests={"site_manifest": "a"},
            provenance={"policy_digest": "p"},
            territories=["FR"], working_crs="EPSG:2154",
            context_anchor_counts={}, source_coverage=[], object_rechecks=[],
            preservation_rules=["préserver"],
        )


def test_camera_constraint_ids_are_unique() -> None:
    constraint = CameraConstraint(
        constraint_id="same", zone_ref="z", rule="avoid", severity="hard",
        rationale="preuve", evidence_refs=["r.json"],
    )
    with pytest.raises(pydantic.ValidationError, match="dupliqué"):
        CameraConstraintsManifest(
            hotel_id="h", generated_at="now", router_decision_digest="d",
            provenance={"policy_digest": "p"},
            constraints=[constraint, constraint],
        )


def test_router_selector_ignores_append_only_invalidations(tmp_path) -> None:
    workspace = Workspace("hotel-test", root=tmp_path)
    workspace.create()
    folder = workspace.path("10_validation")
    old = folder / "router_decision_old.json"
    current = folder / "router_decision_current.json"
    old.write_text(json.dumps({"path": "reject"}))
    current.write_text(json.dumps({"path": "path_d_hybrid"}))
    (folder / "router_invalidations.json").write_text(json.dumps({
        "invalidations": [{"decision_file": old.name}],
    }))

    path, payload = _router_decision(workspace)

    assert path == current
    assert payload["path"] == "path_d_hybrid"


def test_coverage_build_is_exposed_by_the_real_cli() -> None:
    result = CliRunner().invoke(app, ["coverage", "--help"])

    assert result.exit_code == 0
    assert "build" in result.output


def test_completion_separates_rights_limitations_from_done_blockers() -> None:
    cmm = _source_result(_source("cmm-ortho"), None)

    blockers, limitations = _completion_findings(
        uncovered_capture_demands=["obligation:ACCESS_ROAD_MAIN"],
        unresolved_objects=["PROPERTY_PARCEL", "ENTRANCE_MAIN_CURRENT"],
        source_coverage=[cmm],
        asset_count=335,
        duplicate_report_files=329,
        dedup_robustness_current=False,
        source_registry_complete=False,
        uncleared_rights=146,
    )

    assert any("ACCESS_ROAD_MAIN" in reason for reason in blockers)
    assert any("orthophoto" in reason for reason in blockers)
    assert any("329/335" in reason for reason in blockers)
    assert any("registre canonique" in reason for reason in blockers)
    assert all("146 assets" not in reason for reason in blockers)
    assert limitations == [
        "146 assets restent public_uncleared ou unknown ; "
        "ils ne deviennent ni textures ni preuves de production"
    ]


def test_completion_accepts_acquired_ortho_current_dedup_and_source_registry() -> None:
    source_type = type(_source_result(_source("cmm-ortho"), None))
    cmm_payload = _source_result(_source("cmm-ortho"), None).model_dump()
    cmm_payload["acquisition_state"] = AcquisitionState.ACQUIRED
    cmm = source_type.model_validate(cmm_payload)

    blockers, _limitations = _completion_findings(
        uncovered_capture_demands=[],
        unresolved_objects=[],
        source_coverage=[cmm],
        asset_count=335,
        duplicate_report_files=335,
        dedup_robustness_current=True,
        source_registry_complete=True,
        uncleared_rights=0,
    )

    assert blockers == []


def test_robust_dedup_evidence_requires_the_production_algorithm() -> None:
    with pytest.raises(pydantic.ValidationError, match="algorithme de production"):
        DedupRobustnessEvidence(
            asset_manifest_sha256="a" * 64,
            robust_input_digest="inputs",
            algorithm="test-only",
            policy_digest="p",
            dedup_policy_digest="dedup",
            production_used=False,
            candidate_pairs=1,
            matched_pairs=0,
            crop_regression_passed=True,
            watermark_regression_passed=True,
            distinct_regression_passed=True,
        )


def test_robust_dedup_evidence_is_bound_to_dedup_inputs_not_unrelated_fields(
    tmp_path,
) -> None:
    workspace = Workspace("hotel-test", root=tmp_path)
    workspace.create()
    asset = Asset(
        id="asset-1", source="mapillary", source_url_or_id="1",
        rights=Rights.PUBLIC_UNCLEARED, checksum="a" * 64,
        crop_resistant_hash="0" * 64,
        ai_eligible=False, confidence=0.8, category=AssetCategory.OTHER,
    )
    from hotel_pipeline.schemas import AssetManifest
    manifest = AssetManifest(hotel_id="hotel-test", assets=[asset])
    workspace.write_assets(manifest)
    evidence = DedupRobustnessEvidence(
        asset_manifest_sha256=_sha256(workspace.assets_path),
        robust_input_digest=robust_input_digest(manifest.assets),
        algorithm="production-v1",
        policy_digest="p",
        dedup_policy_digest=facet_digest(DEFAULT_POLICY, Facet.DEDUPLICATION),
        production_used=True,
        candidate_pairs=1,
        matched_pairs=0,
        crop_regression_passed=True,
        watermark_regression_passed=True,
        distinct_regression_passed=True,
    )
    workspace.write_json(
        "01_sources/dedup_robustness_report.json", evidence.model_dump(mode="json")
    )
    assert _robust_dedup_is_current(workspace, DEFAULT_POLICY) is True

    # Un champ produit après la déduplication ne périme pas la preuve.
    changed = manifest.model_copy(deep=True)
    changed.assets[0] = changed.assets[0].model_copy(
        update={"visibility_run_id": "later-run"}
    )
    workspace.write_assets(changed)
    assert _robust_dedup_is_current(workspace, DEFAULT_POLICY) is True

    # Le contenu et la famille sont en revanche des entrées réelles.
    changed.assets[0] = changed.assets[0].model_copy(update={"checksum": "b" * 64})
    workspace.write_assets(changed)
    assert _robust_dedup_is_current(workspace, DEFAULT_POLICY) is False
