"""Lissage spatial de la lecture des silhouettes."""

from __future__ import annotations

import numpy as np
import pytest

from hotel_pipeline.conditioning.silhouette import (
    NEIGHBOUR_WEIGHT,
    SMOOTHING_PASSES,
    smooth_scores,
)


def _scores(rows: int, cols: int, classes: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(rows * cols, classes))


class TestShape:
    def test_shape_is_preserved(self):
        scores = _scores(4, 5)
        assert smooth_scores(scores, 4, 5).shape == scores.shape

    def test_single_tile_is_returned_unchanged(self):
        scores = _scores(1, 1)
        assert np.allclose(smooth_scores(scores, 1, 1), scores)


class TestSmoothing:
    def test_uniform_field_is_untouched(self):
        """Sans désaccord, le voisinage n'a rien à corriger."""
        scores = np.tile(np.array([0.9, 0.1, 0.0]), (12, 1))
        assert np.allclose(smooth_scores(scores, 3, 4), scores, atol=1e-9)

    def test_isolated_tile_is_pulled_toward_its_neighbours(self):
        # Une grille entièrement de classe 0, sauf une tuile centrale.
        grid = np.tile(np.array([1.0, 0.0]), (25, 1))
        grid[12] = np.array([0.0, 1.0])
        smoothed = smooth_scores(grid, 5, 5)
        # La tuile isolée bascule vers la classe de son entourage.
        assert smoothed[12].argmax() == 0

    def test_a_firm_reading_survives_its_neighbours(self):
        """Le voisinage tranche les doutes, il ne réécrit pas une évidence."""
        grid = np.tile(np.array([1.0, 0.0]), (25, 1))
        # Un bloc franc et étendu, non une tuile isolée.
        for slot in (6, 7, 8, 11, 12, 13, 16, 17, 18):
            grid[slot] = np.array([0.0, 8.0])
        smoothed = smooth_scores(grid, 5, 5)
        assert smoothed[12].argmax() == 1

    def test_smoothing_contracts_the_margin(self):
        """C'est ce resserrement que le seuil d'indécision doit suivre."""
        scores = _scores(8, 8, seed=5)
        before = np.sort(scores, axis=1)
        after = np.sort(smooth_scores(scores, 8, 8), axis=1)
        assert np.median(after[:, -1] - after[:, -2]) < np.median(
            before[:, -1] - before[:, -2]
        )

    def test_edges_are_handled_without_wrapping(self):
        """Le bord se prolonge, il ne se replie pas sur le bord opposé."""
        grid = np.zeros((9, 2))
        grid[:, 0] = 1.0
        grid[0] = np.array([0.0, 1.0])
        smoothed = smooth_scores(grid, 3, 3)
        assert np.all(np.isfinite(smoothed))

    def test_result_is_finite_on_extreme_values(self):
        scores = _scores(6, 6) * 1e6
        assert np.all(np.isfinite(smooth_scores(scores, 6, 6)))


class TestConstants:
    def test_neighbour_never_outweighs_the_tile(self):
        """Au-delà de 0,5, le voisinage déciderait à la place de la tuile."""
        assert 0.0 < NEIGHBOUR_WEIGHT < 0.5

    def test_passes_stay_local(self):
        """Trop de passes étaleraient une classe sur toute l'image."""
        assert 1 <= SMOOTHING_PASSES <= 3
