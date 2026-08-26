from __future__ import annotations

import pytest

from hotel_pipeline.hybrid_reality import (
    HybridPolicy,
    RealitySource,
    SourceEvidence,
    choose_hybrid_sources,
)


def test_canonical_geometry_beats_google_when_both_exist() -> None:
    decision = choose_hybrid_sources(
        "facade-east",
        [
            SourceEvidence(
                "canonical-mesh",
                RealitySource.CANONICAL,
                geometry_confidence=0.96,
                measured=True,
                supports_geometry=True,
            ),
            SourceEvidence(
                "google-tiles",
                RealitySource.GOOGLE_3D,
                geometry_confidence=0.88,
                measured=True,
                supports_geometry=True,
            ),
            SourceEvidence(
                "photo-east",
                RealitySource.REAL_PHOTO,
                appearance_confidence=0.98,
                effective_gsd_m=0.025,
                coverage_fraction=0.96,
                sharpness=0.95,
                incidence_deg=8.0,
                temporal_confidence=0.98,
                measured=True,
                supports_appearance=True,
            ),
        ],
        distance_m=50.0,
    )
    assert decision.geometry_source_id == "canonical-mesh"
    assert decision.appearance_source_id == "photo-east"
    assert decision.safe


def test_close_shot_rejects_coarse_real_photo_and_does_not_promote_ai() -> None:
    policy = HybridPolicy(minimum_appearance_score=0.5)
    decision = choose_hybrid_sources(
        "facade-west",
        [
            SourceEvidence(
                "canonical-mesh",
                RealitySource.CANONICAL,
                geometry_confidence=0.95,
                measured=True,
                supports_geometry=True,
            ),
            SourceEvidence(
                "coarse-photo",
                RealitySource.REAL_PHOTO,
                appearance_confidence=0.98,
                effective_gsd_m=0.12,
                coverage_fraction=0.95,
                sharpness=0.9,
                measured=True,
                supports_appearance=True,
            ),
            SourceEvidence(
                "ai-fill",
                RealitySource.GENERATIVE,
                appearance_confidence=1.0,
                effective_gsd_m=0.001,
                supports_appearance=True,
            ),
        ],
        distance_m=20.0,
        policy=policy,
    )
    assert decision.geometry_source_id == "canonical-mesh"
    assert not decision.safe
    assert not decision.allow_ai_microtexture
    assert any("appearance" in reason or "generative" in reason for reason in decision.reasons)


def test_distant_context_can_use_google_appearance() -> None:
    decision = choose_hybrid_sources(
        "neighbour-building",
        [
            SourceEvidence(
                "google-context",
                RealitySource.GOOGLE_3D,
                geometry_confidence=0.9,
                appearance_confidence=0.9,
                effective_gsd_m=0.18,
                coverage_fraction=1.0,
                sharpness=0.9,
                temporal_confidence=0.9,
                measured=True,
                supports_geometry=True,
                supports_appearance=True,
            )
        ],
        distance_m=180.0,
        policy=HybridPolicy(minimum_appearance_score=0.25),
    )
    assert decision.geometry_source_id == "google-context"
    assert decision.appearance_source_id == "google-context"
    assert decision.safe


def test_generative_source_can_never_claim_geometry() -> None:
    with pytest.raises(ValueError, match="never support geometry"):
        SourceEvidence(
            "ai",
            RealitySource.GENERATIVE,
            geometry_confidence=1.0,
            supports_geometry=True,
        )
