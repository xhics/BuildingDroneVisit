"""Localiser une caméra contre le LiDAR, sans autre image."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hotel_pipeline.geo.lidar_localize import (
    DECISIVE_MARGIN,
    MIN_SUPPORTED_FRACTION,
    SEARCH_RADIUS_M,
    SUPPORT_RADIUS_PX,
    edge_distance_field,
    localize,
    score_pose,
)


class _Ridge:
    def __init__(self, start, end):
        self.start = np.asarray(start, dtype=float)
        self.end = np.asarray(end, dtype=float)

    @property
    def length_m(self) -> float:
        return float(np.linalg.norm(self.end - self.start))


class _Camera:
    """Caméra pinhole regardant vers +y."""

    def __init__(self, position=(0.0, -40.0, 2.5), heading_deg=0.0, size=(640, 480)):
        self.position = np.asarray(position, dtype=float)
        self.width, self.height = size
        look = math.radians(90.0 - heading_deg)
        self.fwd = np.array([math.cos(look), math.sin(look), 0.0])
        self.right = np.array([self.fwd[1], -self.fwd[0], 0.0])
        self.up = np.cross(self.right, self.fwd)
        self.f = 500.0

    def project(self, points):
        d = np.asarray(points, dtype=float) - self.position
        z = d @ self.fwd
        safe = np.where(z > 1e-6, z, 1e-6)
        return (
            np.c_[
                self.width / 2 + self.f * (d @ self.right) / safe,
                self.height / 2 - self.f * (d @ self.up) / safe,
            ],
            z,
        )


def _image_with_line(row=200):
    """Image noire avec un trait horizontal clair : un contour franc."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[row : row + 3, :] = 255
    return image


class TestDistanceField:
    def test_the_field_is_zero_on_a_contour(self):
        field_px = edge_distance_field(_image_with_line())
        assert float(field_px[200, 320]) < 3.0

    def test_the_field_is_capped(self):
        field_px = edge_distance_field(_image_with_line())
        assert float(field_px.max()) <= SUPPORT_RADIUS_PX

    def test_a_blank_image_yields_the_cap_everywhere(self):
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        field_px = edge_distance_field(blank)
        assert float(field_px.min()) == pytest.approx(SUPPORT_RADIUS_PX)


class TestScorePose:
    def _ridge_on_the_line(self):
        # Une arête horizontale à 90° du champ, projetée près de la rangée 200.
        return [_Ridge((-10.0, 0.0, 10.0), (10.0, 0.0, 10.0))]

    def test_a_wrong_pose_scores_zero(self):
        """Le test décisif : une pose fausse ne doit rien obtenir."""
        ridges = self._ridge_on_the_line()
        field_px = edge_distance_field(_image_with_line())
        far = _Camera(position=(500.0, -40.0, 2.5))
        score, supported, _projected = score_pose(ridges, far, field_px)
        assert score == pytest.approx(0.0, abs=0.01)
        assert supported == 0

    def test_a_ridge_behind_the_camera_is_not_projected(self):
        ridges = [_Ridge((-10.0, -200.0, 10.0), (10.0, -200.0, 10.0))]
        field_px = edge_distance_field(_image_with_line())
        _score, _supported, projected = score_pose(ridges, _Camera(), field_px)
        assert projected == 0

    def test_no_ridges_yields_nothing(self):
        field_px = edge_distance_field(_image_with_line())
        score, supported, projected = score_pose([], _Camera(), field_px)
        assert (score, supported, projected) == (0.0, 0, 0)


class TestLocalize:
    def _make(self, image):
        def make_camera(position, heading):
            return _Camera(
                position=position,
                heading_deg=heading,
                size=(image.shape[1], image.shape[0]),
            )

        return make_camera

    def test_no_ridges_is_a_stated_refusal(self):
        image = _image_with_line()
        found = localize([], image, self._make(image), np.array([0.0, -40.0, 2.5]), 0.0)
        assert not found.localized
        assert "aucune arête" in found.reason

    def test_a_building_out_of_frame_is_reported(self):
        """Rien à aligner n'est pas un échec : c'est un constat."""
        image = _image_with_line()
        behind = [_Ridge((-10.0, -500.0, 10.0), (10.0, -500.0, 10.0))]
        found = localize(
            behind, image, self._make(image), np.array([0.0, 0.0, 2.5]), 0.0
        )
        assert not found.localized
        assert "hors cadre" in found.reason or "aucune arête" in found.reason

    def test_a_weak_alignment_is_refused(self):
        """Sous le seuil d'appui, l'alignement décrit du hasard."""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        ridges = [_Ridge((-10.0, 0.0, 10.0), (10.0, 0.0, 10.0))]
        found = localize(
            ridges, blank, self._make(blank), np.array([0.0, -40.0, 2.5]), 0.0
        )
        assert not found.localized
        assert "appui" in found.reason or "hasard" in found.reason

    def test_holdout_ridges_are_kept_out_of_the_fit(self):
        image = _image_with_line()
        ridges = [
            _Ridge((-10.0, 0.0, 10.0), (10.0, 0.0, 10.0)),
            _Ridge((-8.0, 0.0, 9.0), (8.0, 0.0, 9.0)),
            _Ridge((-6.0, 0.0, 8.0), (6.0, 0.0, 8.0)),
        ]
        found = localize(
            ridges, image, self._make(image), np.array([0.0, -40.0, 2.5]), 0.0,
            holdout=1,
        )
        assert found.provenance.get("ridges_holdout") == 1
        assert found.provenance.get("ridges_fitting") == 2

    def test_everything_held_out_leaves_nothing_to_fit(self):
        image = _image_with_line()
        ridges = [_Ridge((-10.0, 0.0, 10.0), (10.0, 0.0, 10.0))]
        found = localize(
            ridges, image, self._make(image), np.array([0.0, -40.0, 2.5]), 0.0,
            holdout=5,
        )
        assert "contrôle" in found.reason


class TestReport:
    def test_report_carries_its_caveats(self):
        image = _image_with_line()
        payload = localize(
            [], image, lambda p, h: _Camera(), np.array([0.0, 0.0, 2.5]), 0.0
        ).as_dict()
        assert payload["caveats"]
        # La limite qui compte : elle place une caméra, elle n'invente rien.
        assert any("ni texture ni parallaxe" in c for c in payload["caveats"])

    def test_thresholds_are_coherent(self):
        assert SEARCH_RADIUS_M > 0.0
        assert SUPPORT_RADIUS_PX > 0.0
        assert 0.0 < MIN_SUPPORTED_FRACTION <= 1.0
        assert 0.0 < DECISIVE_MARGIN < 1.0
