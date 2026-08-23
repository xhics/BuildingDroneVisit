from __future__ import annotations

import math
from pathlib import Path

from hotel_pipeline.anchor_localization import (
    LocalizationHypothesis,
    evaluate_localization_hypothesis,
    run_anchor_localization,
    select_anchor_core,
)
from hotel_pipeline.schemas import (
    AnchorLocalizationPolicy,
    Asset,
    AssetManifest,
    LocalizationDecision,
    PoseEvidenceClass,
)
from hotel_pipeline.workspace import Workspace
from hotel_pipeline.fidelity_gate import _sparse_gate_passes
from hotel_pipeline.schemas.reconstruction import SparseConsensusGate


def _asset(asset_id: str, *, source: str = "camera", heading_is_measured: bool = False) -> Asset:
    return Asset(
        id=asset_id,
        source=source,
        source_url_or_id=asset_id,
        rights="owned",
        ai_eligible=False,
        confidence=1.0,
        category="facade",
        checksum=(asset_id.encode().hex() + "0" * 64)[:64],
        camera_lat=45.0,
        camera_lon=-73.0,
        heading_deg=0.0,
        heading_is_measured=heading_is_measured,
    )


def _valid_hypothesis() -> LocalizationHypothesis:
    return LocalizationHypothesis(
        pose_world_from_camera={"rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "translation": [0, 0, 0]},
        matches=100,
        inliers=60,
        reference_asset_ids=("a", "b", "c"),
        reprojection_errors_px=tuple([1.0] * 60),
        positive_depth_ratio=1.0,
        gps_residual_m=1.0,
        gps_threshold_m=10.0,
        stability_translation_m=0.1,
        stability_rotation_deg=0.1,
    )


def test_virtual_hypothesis_never_becomes_measured() -> None:
    decision, evidence, reasons, _ = evaluate_localization_hypothesis(
        asset=_asset("query"),
        hypothesis=_valid_hypothesis(),
        policy=AnchorLocalizationPolicy(),
        correction_level="virtual",
    )
    assert decision is LocalizationDecision.INFERRED_ONLY
    assert evidence is PoseEvidenceClass.VIEW_INFERRED
    assert "virtual_pose_is_non_probative" in reasons


def test_missing_stability_refuses_otherwise_good_pnp() -> None:
    hypothesis = _valid_hypothesis()
    hypothesis = LocalizationHypothesis(
        **{**hypothesis.__dict__, "stability_translation_m": None, "stability_rotation_deg": None}
    )
    decision, evidence, reasons, _ = evaluate_localization_hypothesis(
        asset=_asset("query"),
        hypothesis=hypothesis,
        policy=AnchorLocalizationPolicy(),
        correction_level="original",
    )
    assert decision is LocalizationDecision.INSUFFICIENT_EVIDENCE
    assert evidence is PoseEvidenceClass.REJECTED
    assert "pose_stability_missing" in reasons


def test_anchor_selection_uses_geo_consensus(tmp_path: Path) -> None:
    workspace = Workspace("anchor-test", root=tmp_path)
    workspace.create()
    radius = 6_378_137.0
    assets = []
    lines = ["# synthetic COLMAP model"]
    for index in range(8):
        east = float((index % 4) * 4)
        north = float((index // 4) * 5)
        lat = 45.0 + math.degrees(north / radius)
        lon = -73.0 + math.degrees(east / (radius * math.cos(math.radians(45.0))))
        asset = _asset(f"asset-{index}", source="source-a" if index < 4 else "source-b")
        asset.camera_lat = lat
        asset.camera_lon = lon
        asset.local_path = f"asset-{index}.jpg"
        assets.append(asset)
        # Identité COLMAP : t = -centre.
        lines.extend(
            [
                f"{index + 1} 1 0 0 0 {-east} {-north} 0 1 asset-{index}.jpg",
                "0 0 -1",
            ]
        )
    workspace.write_assets(AssetManifest(hotel_id="anchor-test", assets=assets))
    model = tmp_path / "model"
    model.mkdir()
    (model / "images.txt").write_text("\n".join(lines), "utf-8")
    (model / "cameras.txt").write_text("# cameras\n", "utf-8")
    (model / "points3D.txt").write_text("# points\n", "utf-8")

    result = select_anchor_core(
        workspace=workspace,
        reconstruction_input_id="input-1",
        source_model_path=model,
    )
    assert result.status == "ready"
    assert len(result.anchor_asset_ids) == 8
    assert result.metrics["source_count"] == 2
    assert result.metrics["geo_rmse_m"] < 0.01


class _Backend:
    def localize(
        self,
        asset: Asset,
        reference_asset_ids,
        *,
        round_index: int,
        hop: int,
        retry_index: int,
        correction_level: str,
    ) -> LocalizationHypothesis | None:
        if asset.id == "localized" and correction_level == "original":
            return _valid_hypothesis()
        if asset.id == "virtual" and correction_level == "virtual":
            return _valid_hypothesis()
        return None


def test_pipeline_counts_only_anchor_and_validated_pnp() -> None:
    assets = [_asset("anchor"), _asset("localized"), _asset("virtual"), _asset("failed")]
    result = run_anchor_localization(
        reconstruction_input_id="input-1",
        anchor_model_id="anchor-model-1",
        selected_assets=assets,
        anchor_poses={"anchor": {"rotation": [], "translation": [0, 0, 0]}},
        backend=_Backend(),
        policy=AnchorLocalizationPolicy(max_rounds=1, max_hop=1),
        raw_registered_images=4,
        allow_virtual_suggestions=True,
    )
    assert result.raw_registered_images == 4
    assert result.measured_anchor_images == 1
    assert result.measured_localized_images == 1
    assert result.inferred_images == 1
    assert result.rejected_images == 1
    assert result.validated_registration_rate == 0.5


def test_raw_colmap_registration_cannot_pass_g5() -> None:
    assert not _sparse_gate_passes(
        SparseConsensusGate(
            raw_registration_rate=0.95,
            raw_registered_images=52,
            registration_rate=0.0,
            validated_registration_rate=0.0,
            largest_component_size=52,
            median_reprojection_px=0.5,
        )
    )


def test_validated_localization_can_pass_sparse_gate() -> None:
    assert _sparse_gate_passes(
        SparseConsensusGate(
            raw_registration_rate=0.95,
            registration_rate=0.75,
            validated_registration_rate=0.75,
            validated_main_component_ratio=0.9,
            external_pose_consistency=True,
            largest_component_size=40,
            median_reprojection_px=1.0,
        )
    )
