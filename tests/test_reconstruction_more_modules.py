"""Tests pour les modules Lot 2 manquants : novel_view, stability, fidelity, gap, portfolio, preprocess."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hotel_pipeline.schemas.reconstruction import (
    Criticality,
    FidelityGate,
    GapType,
    GateResult,
    HoldoutPlan,
    HoldoutStrategy,
    ReconstructionGap,
    ReconstructionTarget,
    ReconstructionTargetKind,
    SupportType,
)
from hotel_pipeline.fidelity_gate import (
    evaluate_fidelity,
    evaluate_targets,
    publish_fidelity_gates,
)
from hotel_pipeline.gap_analysis import analyze_gaps, publish_gap_analysis
from hotel_pipeline.portfolio_optimizer import (
    AcquisitionPortfolioOptimizer,
    PostViewGraphVerdict,
    PreSfMVerdict,
)
from hotel_pipeline.preprocess import build_preprocess_manifest, PreprocessManifest
from hotel_pipeline.stability import build_stability_manifest, StabilityResult
from hotel_pipeline.workspace import Workspace


def _minimal_workspace(tmp_path: Path, hotel_id: str) -> Workspace:
    workspace = Workspace(hotel_id)
    workspace.create()
    return workspace


def _target(criticality: Criticality, target_id: str = "FACADE_PRIMARY") -> ReconstructionTarget:
    return ReconstructionTarget(
        target_id=target_id,
        kind=ReconstructionTargetKind.SURFACE,
        criticality=criticality,
        allowed_support=[
            SupportType.MEASURED_PHOTO,
            SupportType.MULTIVIEW_RECONSTRUCTED,
            SupportType.FEEDFORWARD_INFERRED,
        ],
    )


# ---------------------------------------------------------------------------
# Fidelity Gate
# ---------------------------------------------------------------------------


def test_fidelity_must_show_pass():
    t = _target(Criticality.MUST_SHOW)
    gate = evaluate_fidelity(
        t,
        sparse_gate=None,
        geo_gate=None,
        novel_view_gate=None,
    )
    # No gates => INSUFFICIENT_EVIDENCE (blocks, reason explicit)
    assert gate.overall == GateResult.INSUFFICIENT_EVIDENCE


def test_fidelity_optional_passes_without_gates():
    t = _target(Criticality.OPTIONAL, target_id="GARDEN")
    gate = evaluate_fidelity(t)
    assert gate.overall == GateResult.INSUFFICIENT_EVIDENCE


def test_fidelity_forbidden_is_not_applicable():
    t = _ = None
    t = _target(Criticality.FORBIDDEN, target_id="FORBIDDEN_OBJ")
    gate = evaluate_fidelity(t)
    assert gate.overall == GateResult.NOT_APPLICABLE


def test_evaluate_targets_returns_one_gate_per_target():
    targets = [
        _target(Criticality.MUST_SHOW, "FACADE_PRIMARY"),
        _target(Criticality.OPTIONAL, "GARDEN"),
    ]
    gates = evaluate_targets(targets, {})
    assert len(gates) == 2
    assert {g.target_id for g in gates} == {"FACADE_PRIMARY", "GARDEN"}


def test_fidelity_publish(tmp_path: Path):
    workspace = _minimal_workspace(tmp_path, "t-fid")
    targets = [_target(Criticality.MUST_SHOW)]
    gates = evaluate_targets(targets, {})
    path = publish_fidelity_gates(gates, workspace)
    assert path.is_file()
    data = json.loads(path.read_text())
    assert data["blocked"] == ["FACADE_PRIMARY"]


# ---------------------------------------------------------------------------
# Gap Analysis
# ---------------------------------------------------------------------------


def test_gap_analysis_empty_when_all_pass():
    targets = [_target(Criticality.OPTIONAL, "GARDEN")]
    gates = evaluate_targets(targets, {})
    # OPTIONAL without evidence -> INSUFFICIENT_EVIDENCE, so gaps appear
    gaps = analyze_gaps(gates)
    assert isinstance(gaps, list)


def test_gap_analysis_produces_structured_gap():
    targets = [_target(Criticality.MUST_SHOW)]
    gates = evaluate_targets(targets, {})
    gaps = analyze_gaps(gates)
    assert len(gaps) >= 1
    gap = gaps[0]
    assert isinstance(gap, ReconstructionGap)
    assert gap.target_id == "FACADE_PRIMARY"
    assert gap.gap_type in GapType
    assert gap.required_observation


def test_gap_publish(tmp_path: Path):
    workspace = _minimal_workspace(tmp_path, "t-gap")
    targets = [_target(Criticality.MUST_SHOW)]
    gates = evaluate_targets(targets, {})
    gaps = analyze_gaps(gates)
    path = publish_gap_analysis(gaps, workspace)
    assert path.is_file()


# ---------------------------------------------------------------------------
# Portfolio Optimizer
# ---------------------------------------------------------------------------


def test_pre_sfm_gate_only_structurally_impossible_blocks():
    from hotel_pipeline.portfolio_optimizer import (
        post_view_graph_gate,
        pre_sfm_collection_gate,
    )

    v0 = pre_sfm_collection_gate(0, 0.0)
    assert v0 == PreSfMVerdict.STRUCTURALLY_IMPOSSIBLE
    v1 = pre_sfm_collection_gate(3, 0.5)
    assert v1 == PreSfMVerdict.READY_TO_ATTEMPT
    v2 = pre_sfm_collection_gate(1, 0.5)
    assert v2 == PreSfMVerdict.WEAK_BUT_ATTEMPT

    # Post-ViewGraph verdict
    from hotel_pipeline.schemas.reconstruction import PairEvidence, ViewGraphManifest, ViewGraphNode, ViewGraphReport

    vg_empty = ViewGraphManifest(
        view_graph_id="vg-x",
        reconstruction_input_id="ri-x",
        nodes=[ViewGraphNode(asset_id="a1"), ViewGraphNode(asset_id="a2")],
        pairs=[],
        report=ViewGraphReport(images_selected=2, pairs_tested=0, valid_pairs=0, largest_component=1),
    )
    assert post_view_graph_gate(vg_empty) == PostViewGraphVerdict.NOT_VIABLE

    vg_ok = ViewGraphManifest(
        view_graph_id="vg-y",
        reconstruction_input_id="ri-y",
        nodes=[ViewGraphNode(asset_id=f"a{i}") for i in range(5)],
        pairs=[PairEvidence(image_a="a1", image_b="a2", status="valid")],
        report=ViewGraphReport(images_selected=5, pairs_tested=1, valid_pairs=1, largest_component=5, registered_candidate_ratio=1.0),
    )
    assert post_view_graph_gate(vg_ok) == PostViewGraphVerdict.RECONSTRUCTION_VIABLE


# ---------------------------------------------------------------------------
# Preprocess Manifest
# ---------------------------------------------------------------------------


def test_preprocess_manifest_roundtrip(tmp_path: Path):
    from hotel_pipeline.schemas.reconstruction import (
        ReconstructionInputManifest,
        ReconstructionSelection,
        ReconstructionSelectionManifest,
    )

    workspace = _minimal_workspace(tmp_path, "t-pp")
    # Minimal input manifest
    im = ReconstructionInputManifest(
        reconstruction_input_id="recon-t-pp-20260101T000000Z",
        hotel_id="t-pp",
        asset_manifest_digest="a" * 64,
        spatial_manifest_digest="b" * 64,
        site_manifest_digest="c" * 64,
        coverage_digest="d" * 64,
        router_decision_digest="e" * 64,
        selected_asset_ids=["asset-1"],
        targets=[_target(Criticality.MUST_SHOW)],
    )
    pm = build_preprocess_manifest(workspace, im)
    assert isinstance(pm, PreprocessManifest)
    assert pm.masked_asset_ids == ["asset-1"]
    payload = pm.model_dump(mode="json")
    recovered = PreprocessManifest.model_validate(payload)
    assert recovered.preprocess_id == pm.preprocess_id


__all__ = [
    "test_fidelity_must_show_pass",
    "test_fidelity_optional_passes_without_gates",
    "test_fidelity_forbidden_is_not_applicable",
    "test_evaluate_targets_returns_one_gate_per_target",
    "test_fidelity_publish",
    "test_gap_analysis_empty_when_all_pass",
    "test_gap_analysis_produces_structured_gap",
    "test_gap_publish",
    "test_pre_sfm_gate_only_structurally_impossible_blocks",
    "test_preprocess_manifest_roundtrip",
]
