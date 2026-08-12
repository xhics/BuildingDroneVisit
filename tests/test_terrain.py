"""Terrain interpolé et mesure de sa fiabilité (Lot 1B §9).

Sous un bâtiment il n'y a pas de sol : la surface est interpolée depuis son
pourtour, et l'enjeu est de dire à quel point elle est fiable. Trois mesures
s'y emploient, dont aucune ne suffit seule.
"""

from __future__ import annotations

import numpy as np
import pytest

from hotel_pipeline.geo.terrain import (
    aggregate_median,
    aligned_origin,
    block_cross_validation,
    interpolate_idw,
    interpolate_tin,
    pseudo_building_validation,
    support_distance,
)


def sloped_ground(n=1200, slope=0.02, seed=7):
    """Sol en pente douce, avec une légère ondulation et du bruit."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 60, n)
    y = rng.uniform(0, 60, n)
    z = 30.0 + slope * x + 0.6 * np.sin(x / 25.0) + rng.normal(0, 0.03, n)
    return x, y, z


class TestGridAlignment:
    def test_origin_snaps_to_the_cell_grid(self):
        assert aligned_origin(309226.27, 5048247.14, 0.5) == (309226.0, 5048247.0)

    def test_same_origin_for_nearby_extents(self):
        """Terrain et toiture doivent partager la grille, sinon la soustraction
        cellule à cellule fabrique des hauteurs."""
        assert aligned_origin(309226.27, 5048247.14, 0.5) == aligned_origin(
            309226.49, 5048247.44, 0.5
        )

    def test_finer_cells_give_a_finer_origin(self):
        assert aligned_origin(10.7, 10.7, 0.25) == (10.5, 10.5)


class TestMedianAggregation:
    def test_median_resists_a_dense_biased_cluster(self):
        """Une moyenne suivrait la zone sur-échantillonnée ; la médiane non."""
        x = np.concatenate([np.full(100, 0.25), np.array([0.75])])
        y = np.concatenate([np.full(100, 0.25), np.array([0.25])])
        z = np.concatenate([np.full(100, 10.0), np.array([50.0])])

        cx, cy, values = aggregate_median(x, y, z, (0.0, 0.0), (2, 2), 0.5)
        assert len(values) == 2
        assert 10.0 in values

    def test_one_value_per_occupied_cell(self):
        x, y, z = sloped_ground(n=500)
        _, _, values = aggregate_median(x, y, z, (0.0, 0.0), (120, 120), 0.5)
        assert 0 < len(values) <= 500


class TestExtrapolationIsRefused:
    def test_tin_returns_nan_outside_the_convex_hull(self):
        """Hors enveloppe, ce n'est plus de l'interpolation."""
        px = np.array([0.0, 10.0, 10.0, 0.0])
        py = np.array([0.0, 0.0, 10.0, 10.0])
        pz = np.array([1.0, 2.0, 3.0, 2.0])

        inside = interpolate_tin(px, py, pz, np.array([5.0]), np.array([5.0]))
        outside = interpolate_tin(px, py, pz, np.array([50.0]), np.array([50.0]))

        assert not np.isnan(inside[0])
        assert np.isnan(outside[0])

    def test_idw_extrapolates_where_tin_refuses(self):
        """Justement pourquoi l'IDW ne sert qu'à mesurer un désaccord."""
        px = np.array([0.0, 10.0, 10.0, 0.0])
        py = np.array([0.0, 0.0, 10.0, 10.0])
        pz = np.array([1.0, 2.0, 3.0, 2.0])
        assert not np.isnan(
            interpolate_idw(px, py, pz, np.array([50.0]), np.array([50.0]))[0]
        )


class TestSupportDistance:
    def test_distance_grows_towards_the_centre_of_a_gap(self):
        """Une cellule au centre d'un bâtiment de 40 m est à 20 m d'un appui."""
        angles = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        px, py = 20 * np.cos(angles), 20 * np.sin(angles)

        centre = support_distance(px, py, np.array([0.0]), np.array([0.0]))
        edge = support_distance(px, py, np.array([18.0]), np.array([0.0]))

        assert centre[0] == pytest.approx(20, abs=0.5)
        assert edge[0] < 3


class TestBlockCrossValidation:
    def test_smooth_ground_is_reconstructed_accurately(self):
        x, y, z = sloped_ground()
        scores = block_cross_validation(x, y, z, block_m=15.0)
        assert scores.n > 0
        assert scores.rmse < 0.35

    def test_scores_expose_bias_separately_from_dispersion(self):
        x, y, z = sloped_ground()
        scores = block_cross_validation(x, y, z, block_m=15.0)
        assert abs(scores.bias) < scores.rmse
        assert scores.p95 >= scores.mae

    def test_too_few_points_yields_no_score_rather_than_a_wrong_one(self):
        tiny = np.array([0.0, 1.0])
        scores = block_cross_validation(tiny, tiny, tiny, block_m=10.0)
        assert scores.n == 0
        assert scores.rmse is None


class TestPseudoBuildingValidation:
    def test_it_produces_trials_with_their_support_distances(self):
        x, y, z = sloped_ground(n=2500)
        trials = pseudo_building_validation(
            x, y, z, width_m=16, height_m=10, ring_m=8.0, trials=3
        )

        assert trials
        for trial in trials:
            assert trial["masked_points"] > 0
            assert trial["support_distance_max_m"] > 2

    def test_a_continuous_gap_is_harder_than_scattered_blocks(self):
        """La validation par blocs reste optimiste : ses trous sont petits.

        Un vide continu de dimensions réelles éloigne bien davantage les
        cellules de leur appui, ce que cette comparaison rend visible.
        """
        x, y, z = sloped_ground(n=2500)
        blocks = block_cross_validation(x, y, z, block_m=12.0)
        trials = pseudo_building_validation(
            x, y, z, width_m=16, height_m=10, ring_m=8.0, trials=2
        )

        assert trials, "aucun essai exploitable — la comparaison n'aurait aucun sens"
        worst = max(t["support_distance_max_m"] for t in trials)
        assert worst > blocks.rmse

    def test_no_trial_is_invented_when_the_ground_is_too_sparse(self):
        x, y, z = sloped_ground(n=40)
        assert pseudo_building_validation(x, y, z, width_m=16, height_m=10) == []

    def test_a_gap_larger_than_the_known_ground_is_refused(self):
        """Sans pourtour, un essai serait dégénéré plutôt qu'informatif."""
        x, y, z = sloped_ground(n=1200)
        assert pseudo_building_validation(x, y, z, width_m=200, height_m=200) == []
