"""Toitures triangulées depuis le nuage LiDAR."""

from __future__ import annotations

import numpy as np
import pytest

from hotel_pipeline.conditioning.heights import build_roof_surface_from_cloud

FOOTPRINT = np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 12.0], [0.0, 12.0]])


def _flat_cloud(height: float, spacing: float = 0.4) -> np.ndarray:
    xs, ys = np.meshgrid(
        np.arange(0.5, 19.5, spacing), np.arange(0.5, 11.5, spacing)
    )
    return np.c_[xs.ravel(), ys.ravel(), np.full(xs.size, height)]


class TestBuildRoofSurface:
    def test_flat_roof_is_recovered_at_its_height(self):
        vertices, faces = build_roof_surface_from_cloud(_flat_cloud(9.0), FOOTPRINT)
        assert len(faces) > 0
        assert vertices[:, 2] == pytest.approx(9.0, abs=0.01)

    def test_pitched_roof_keeps_its_slope(self):
        cloud = _flat_cloud(0.0)
        # Deux versants se rejoignant sur un faîtage central.
        cloud[:, 2] = 6.0 + 0.5 * (6.0 - np.abs(cloud[:, 1] - 6.0))
        vertices, _faces = build_roof_surface_from_cloud(cloud, FOOTPRINT)
        assert vertices[:, 2].max() - vertices[:, 2].min() > 2.0

    def test_points_outside_the_footprint_are_ignored(self):
        inside = _flat_cloud(9.0)
        outside = inside.copy()
        outside[:, 0] += 40.0
        outside[:, 2] = 30.0
        vertices, _faces = build_roof_surface_from_cloud(
            np.vstack([inside, outside]), FOOTPRINT
        )
        assert vertices[:, 2].max() < 10.0

    def test_maximum_wins_over_returns_passing_underneath(self):
        """Un retour sous l'auvent ne doit pas creuser la toiture."""
        cloud = _flat_cloud(9.0)
        low = cloud.copy()
        low[:, 2] = 3.0
        vertices, _faces = build_roof_surface_from_cloud(
            np.vstack([cloud, low]), FOOTPRINT
        )
        assert vertices[:, 2].min() > 8.0

    def test_empty_cloud_yields_no_surface(self):
        """Sans retour, la fonction ne rend rien : le toit reste à fermer."""
        assert build_roof_surface_from_cloud(np.empty((0, 3)), FOOTPRINT) is None

    def test_cloud_entirely_outside_yields_no_surface(self):
        away = _flat_cloud(9.0)
        away[:, 0] += 500.0
        assert build_roof_surface_from_cloud(away, FOOTPRINT) is None

    def test_faces_index_existing_vertices(self):
        vertices, faces = build_roof_surface_from_cloud(_flat_cloud(9.0), FOOTPRINT)
        assert faces.min() >= 0
        assert faces.max() < len(vertices)

    def test_surface_covers_most_of_the_footprint(self):
        """Une toiture trouée laisserait voir l'intérieur du volume."""
        vertices, faces = build_roof_surface_from_cloud(_flat_cloud(9.0), FOOTPRINT)
        area = 0.0
        for a, b, c in faces:
            pa, pb, pc = vertices[a], vertices[b], vertices[c]
            area += 0.5 * abs(
                (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pc[0] - pa[0]) * (pb[1] - pa[1])
            )
        assert area > 0.8 * 20.0 * 12.0

    def test_sparse_cloud_still_produces_a_surface(self):
        vertices, faces = build_roof_surface_from_cloud(
            _flat_cloud(9.0, spacing=2.0), FOOTPRINT
        )
        assert len(faces) > 0
        assert np.all(np.isfinite(vertices))


class TestRoofDetailFollowsDistance:
    """La finesse de toiture suit ce que la caméra peut en voir."""

    def _prism(self, offset: float, target: bool = False):
        class _P:
            pass

        prism = _P()
        prism.is_target = target
        prism.footprint = FOOTPRINT + np.array([offset, 0.0])
        return prism

    def test_target_always_keeps_the_fine_mesh(self):
        from hotel_pipeline.conditioning.heights import (
            ROOF_CELL_NEAR_M,
            _roof_cell_for,
        )

        # Même repoussée au loin, la cible reste le sujet du plan.
        assert _roof_cell_for(self._prism(800.0, target=True), (0.0, 0.0)) == (
            ROOF_CELL_NEAR_M
        )

    def test_close_neighbour_keeps_the_fine_mesh(self):
        from hotel_pipeline.conditioning.heights import (
            ROOF_CELL_NEAR_M,
            _roof_cell_for,
        )

        assert _roof_cell_for(self._prism(10.0), (0.0, 0.0)) == ROOF_CELL_NEAR_M

    def test_distant_neighbour_is_coarsened(self):
        from hotel_pipeline.conditioning.heights import (
            ROOF_CELL_NEAR_M,
            _roof_cell_for,
        )

        assert _roof_cell_for(self._prism(300.0), (0.0, 0.0)) > ROOF_CELL_NEAR_M

    def test_coarsening_is_capped(self):
        from hotel_pipeline.conditioning.heights import (
            ROOF_CELL_FAR_M,
            _roof_cell_for,
        )

        assert _roof_cell_for(self._prism(5000.0), (0.0, 0.0)) == ROOF_CELL_FAR_M

    def test_mesh_grows_monotonically_with_distance(self):
        from hotel_pipeline.conditioning.heights import _roof_cell_for

        cells = [
            _roof_cell_for(self._prism(offset), (0.0, 0.0))
            for offset in (50.0, 150.0, 300.0, 600.0)
        ]
        assert cells == sorted(cells)
