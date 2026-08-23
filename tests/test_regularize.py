"""Régularisation des emprises : redresser sans déformer."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hotel_pipeline.conditioning.regularize import (
    apply_to_scene,
    dominant_orientation,
    regularize,
)


def _rotate(points: np.ndarray, degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    matrix = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    return points @ matrix.T


RECTANGLE = np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [0.0, 10.0]])


class TestDominantOrientation:
    def test_axis_aligned_rectangle_reads_zero(self):
        assert dominant_orientation(RECTANGLE) == pytest.approx(0.0, abs=0.01)

    def test_rotated_rectangle_reads_its_rotation(self):
        assert dominant_orientation(_rotate(RECTANGLE, 30.0)) == pytest.approx(
            30.0, abs=0.01
        )

    def test_orientation_folds_modulo_ninety(self):
        """Un mur et sa perpendiculaire décrivent le même équerrage."""
        assert dominant_orientation(_rotate(RECTANGLE, 95.0)) == pytest.approx(
            5.0, abs=0.01
        )

    def test_short_edges_do_not_vote(self):
        """Un raccord d'un mètre ne dit rien de l'orientation du bâtiment."""
        noisy = np.array([[0.0, 0.0], [20.0, 0.0], [20.5, 0.8], [20.0, 10.0], [0.0, 10.0]])
        assert dominant_orientation(noisy) == pytest.approx(0.0, abs=1.0)

    def test_degenerate_footprint_returns_zero(self):
        assert dominant_orientation(np.array([[0.0, 0.0], [1.0, 1.0]])) == 0.0


class TestRegularize:
    def test_perfect_rectangle_is_left_untouched(self):
        adjusted, report = regularize(RECTANGLE, "clean")
        assert np.allclose(adjusted, RECTANGLE, atol=1e-6)
        assert report.max_shift_m == pytest.approx(0.0, abs=1e-6)

    def test_slightly_skewed_corner_is_squared(self):
        skewed = np.array([[0.0, 0.0], [20.0, 0.4], [19.6, 10.0], [0.0, 10.0]])
        adjusted, report = regularize(skewed, "skewed")

        assert report.edges_aligned > 0
        angles = []
        for index in range(len(adjusted)):
            delta = adjusted[(index + 1) % len(adjusted)] - adjusted[index]
            angles.append(math.degrees(math.atan2(delta[1], delta[0])) % 90.0)
        # Chaque arête tombe désormais sur l'un des deux axes dominants.
        for angle in angles:
            assert min(angle, 90.0 - angle) < 3.0

    def test_oblique_edge_is_deliberate_and_survives(self):
        """Un pan coupé à 45° n'est pas une erreur de saisie."""
        cut = np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [10.0, 20.0], [0.0, 20.0]])
        adjusted, _report = regularize(cut, "cut")
        assert np.allclose(adjusted[3], cut[3], atol=1e-6)

    def test_shift_stays_bounded(self):
        rng = np.random.default_rng(7)
        noisy = RECTANGLE + rng.normal(0.0, 0.15, RECTANGLE.shape)
        _adjusted, report = regularize(noisy, "noisy")
        assert report.max_shift_m <= 1.2

    def test_triangle_is_returned_unchanged(self):
        triangle = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 8.0]])
        adjusted, report = regularize(triangle, "triangle")
        assert np.allclose(adjusted, triangle)
        assert report.edges_total == 0

    def test_contour_stays_closed(self):
        """Les sommets partagés reçoivent une position unique, sans déchirure."""
        skewed = np.array([[0.0, 0.0], [20.0, 0.5], [19.5, 10.0], [-0.3, 9.6]])
        adjusted, _report = regularize(skewed, "closed")
        assert len(adjusted) == len(skewed)
        assert np.all(np.isfinite(adjusted))

    def test_report_serialises(self):
        _adjusted, report = regularize(RECTANGLE, "target")
        payload = report.as_dict()
        assert payload["feature_id"] == "target"
        assert set(payload) == {
            "feature_id",
            "dominant_deg",
            "edges_aligned",
            "edges_total",
            "max_shift_m",
        }


class _Prism:
    def __init__(self, footprint, feature_id):
        self.footprint = footprint
        self.feature_id = feature_id
        self.is_target = False


class _Scene:
    def __init__(self, prisms):
        self.prisms = prisms


class TestApplyToScene:
    def test_reports_only_the_volumes_it_changed(self):
        skewed = np.array([[0.0, 0.0], [20.0, 0.4], [19.6, 10.0], [0.0, 10.0]])
        scene = _Scene([_Prism(skewed.copy(), "a"), _Prism(RECTANGLE.copy(), "b")])
        result = apply_to_scene(scene)

        assert result["total"] == 2
        assert result["regularized"] >= 1
        assert result["max_shift_m"] <= 1.2

    def test_footprints_are_replaced_in_place(self):
        skewed = np.array([[0.0, 0.0], [20.0, 0.4], [19.6, 10.0], [0.0, 10.0]])
        prism = _Prism(skewed.copy(), "a")
        apply_to_scene(_Scene([prism]))
        assert not np.allclose(prism.footprint, skewed)
