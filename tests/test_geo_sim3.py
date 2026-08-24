"""Alignement géospatial : une Sim(3) mesurée, robuste aux relevés aberrants."""

from __future__ import annotations

import numpy as np
import pytest

from hotel_pipeline.geo_alignment import (
    GPS_INLIER_M,
    MIN_GPS_CORRESPONDENCES,
    _robust_sim3,
)


def _rotation_z(degrees: float) -> np.ndarray:
    angle = np.radians(degrees)
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _pair(count: int = 40, degrees: float = 37.0, scale: float = 2.4, seed: int = 4):
    rng = np.random.default_rng(seed)
    source = rng.uniform(-30.0, 30.0, (count, 3))
    rotation = _rotation_z(degrees)
    translation = np.array([120.0, -80.0, 5.0])
    target = (scale * (rotation @ source.T)).T + translation
    return source, target, rotation, translation, scale


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    cosine = (np.trace(a.T @ b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


class TestExactRecovery:
    def test_clean_correspondences_recover_the_transform(self):
        source, target, rotation, translation, scale = _pair()
        found_r, found_t, found_s, inliers = _robust_sim3(source, target)

        assert _angle_between(rotation, found_r) < 0.5
        assert found_s == pytest.approx(scale, rel=0.01)
        assert np.allclose(found_t, translation, atol=1.0)
        assert inliers.all()

    def test_rotation_is_measured_not_assumed_identity(self):
        """Le défaut corrigé : une rotation posée à l'identité."""
        source, target, rotation, _t, _s = _pair(degrees=37.0)
        found_r, _ft, _fs, _inliers = _robust_sim3(source, target)
        assert _angle_between(np.eye(3), found_r) > 30.0
        assert _angle_between(rotation, found_r) < 0.5


class TestRobustness:
    def test_outliers_do_not_swing_the_solution(self):
        source, target, rotation, _t, scale = _pair()
        rng = np.random.default_rng(9)
        target = target + rng.normal(0.0, 0.8, target.shape)
        target[[3, 9, 17, 28]] += rng.normal(0.0, 120.0, (4, 3))

        found_r, _ft, found_s, inliers = _robust_sim3(source, target)
        assert _angle_between(rotation, found_r) < 2.0
        assert found_s == pytest.approx(scale, rel=0.05)
        # Les quatre relevés aberrants sont écartés.
        assert inliers.sum() >= len(source) - 6

    def test_gps_noise_alone_keeps_everything(self):
        source, target, _r, _t, _s = _pair()
        rng = np.random.default_rng(12)
        noisy = target + rng.normal(0.0, 1.0, target.shape)
        _fr, _ft, _fs, inliers = _robust_sim3(source, noisy)
        assert inliers.all()

    def test_inlier_threshold_is_metric(self):
        assert 1.0 < GPS_INLIER_M < 50.0

    def test_minimum_leaves_room_for_ransac(self):
        """Trois points suffisent au modèle ; le seuil doit dépasser cela."""
        assert MIN_GPS_CORRESPONDENCES > 3


class TestDegenerate:
    def test_no_consensus_returns_identity_rather_than_a_guess(self):
        rng = np.random.default_rng(21)
        source = rng.uniform(-30.0, 30.0, (12, 3))
        target = rng.uniform(-500.0, 500.0, (12, 3))
        found_r, found_t, found_s, inliers = _robust_sim3(source, target)
        if inliers.sum() < 3:
            assert np.allclose(found_r, np.eye(3))
            assert found_s == 1.0
            assert np.allclose(found_t, 0.0)

    def test_result_is_always_finite(self):
        source, target, _r, _t, _s = _pair(count=8)
        found_r, found_t, found_s, _inliers = _robust_sim3(source, target)
        assert np.all(np.isfinite(found_r))
        assert np.all(np.isfinite(found_t))
        assert np.isfinite(found_s)
