from __future__ import annotations

import math

import pytest

from hotel_pipeline.texture_reality import (
    CameraTextureDemand,
    TextureEvidence,
    TextureRealityLevel,
    evaluate_texture_reality,
    evaluate_visible_tiles,
    required_gsd,
)


def _good(**changes) -> TextureEvidence:
    values = {
        "effective_gsd_m": 0.02, "coverage": 0.96, "sharpness": 0.95,
        "view_count": 4, "pose_confidence": 0.96, "incidence_deg": 10.0,
        "photometric_consistency": 0.95, "unknown_fraction": 0.02,
    }
    values.update(changes)
    return TextureEvidence(**values)


def test_required_gsd_matches_analytic_pinhole_case():
    demand = CameraTextureDemand(20.0, 60.0, 1920)
    expected = 40.0 * math.tan(math.radians(30.0)) / 1920
    assert required_gsd(demand) == pytest.approx(expected)
    result = evaluate_texture_reality(_good(effective_gsd_m=0.05), demand)
    assert result.upscale_factor == pytest.approx(0.05 / expected)


def test_closeup_is_rejected_while_distant_shot_is_accepted():
    evidence = _good(effective_gsd_m=0.05)
    near = evaluate_texture_reality(evidence, CameraTextureDemand(15, 60, 1920))
    far = evaluate_texture_reality(evidence, CameraTextureDemand(60, 60, 1920))
    assert not near.safe
    assert far.safe


def test_good_resolution_cannot_hide_bad_coverage():
    result = evaluate_texture_reality(
        _good(effective_gsd_m=0.015, coverage=0.55, unknown_fraction=0.45),
        CameraTextureDemand(30, 60, 1920),
        required_level=TextureRealityLevel.SAFE_FOR_CLOSEUP,
    )
    assert result.level is not TextureRealityLevel.SAFE_FOR_CLOSEUP


def test_four_agreeing_views_score_above_one_view():
    demand = CameraTextureDemand(40, 60, 1920)
    assert evaluate_texture_reality(_good(view_count=4), demand).score > evaluate_texture_reality(_good(view_count=1), demand).score


def test_grazing_angle_and_pose_uncertainty_increase_safe_distance():
    demand = CameraTextureDemand(40, 60, 1920)
    clean = evaluate_texture_reality(_good(), demand)
    grazing = evaluate_texture_reality(_good(incidence_deg=75), demand)
    uncertain = evaluate_texture_reality(_good(pose_confidence=0.25), demand)
    assert grazing.min_safe_distance_m > clean.min_safe_distance_m
    assert uncertain.min_safe_distance_m > clean.min_safe_distance_m


def test_visible_unknown_tile_hard_rejects_an_otherwise_good_surface():
    result = evaluate_visible_tiles(
        [_good(), _good(effective_gsd_m=None, coverage=0, unknown_fraction=1)],
        CameraTextureDemand(50, 60, 1920),
    )
    assert not result.safe
    assert result.level is TextureRealityLevel.UNSUPPORTED


def test_4k_and_telephoto_are_more_demanding():
    evidence = _good()
    full_hd = evaluate_texture_reality(evidence, CameraTextureDemand(40, 60, 1920))
    four_k = evaluate_texture_reality(evidence, CameraTextureDemand(40, 60, 3840))
    tele = evaluate_texture_reality(evidence, CameraTextureDemand(40, 25, 1920))
    assert four_k.min_safe_distance_m > full_hd.min_safe_distance_m
    assert tele.min_safe_distance_m > full_hd.min_safe_distance_m


def test_small_frame_occupancy_requests_less_surface_detail():
    evidence = _good(effective_gsd_m=0.04)
    full = evaluate_texture_reality(evidence, CameraTextureDemand(30, 60, 1920, 1.0))
    small = evaluate_texture_reality(evidence, CameraTextureDemand(30, 60, 1920, 0.3))
    assert small.upscale_factor < full.upscale_factor
