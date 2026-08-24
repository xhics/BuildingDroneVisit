"""Géométrie épipolaire : calibrée quand on peut, honnête sinon."""

from __future__ import annotations

import numpy as np
import pytest

from hotel_pipeline.view_graph import (
    FOCAL_AGREEMENT,
    PLANARITY_RATIO,
    _camera_matrix,
    _planarity,
)


def _intrinsics(fx: float, width: int = 1280, height: int = 720) -> dict:
    return {"fx": fx, "fy": fx, "cx": width / 2, "cy": height / 2}


SHAPE = (720, 1280)


class TestCameraMatrix:
    def test_two_agreeing_cameras_give_a_matrix(self):
        matrix = _camera_matrix(_intrinsics(900.0), _intrinsics(920.0), SHAPE)
        assert matrix is not None
        assert matrix.shape == (3, 3)
        assert matrix[2, 2] == 1.0

    def test_focal_is_the_average_of_the_pair(self):
        matrix = _camera_matrix(_intrinsics(800.0), _intrinsics(900.0), SHAPE)
        assert matrix[0, 0] == pytest.approx(850.0)

    def test_missing_intrinsics_yield_nothing(self):
        assert _camera_matrix(None, _intrinsics(900.0), SHAPE) is None
        assert _camera_matrix(_intrinsics(900.0), None, SHAPE) is None

    def test_disagreeing_focals_yield_nothing(self):
        """Deux appareils trop différents ne partagent pas de calibration."""
        assert _camera_matrix(_intrinsics(500.0), _intrinsics(2000.0), SHAPE) is None

    def test_agreement_threshold_is_respected(self):
        base = 1000.0
        just_inside = base * (1.0 + FOCAL_AGREEMENT * 0.5)
        assert _camera_matrix(_intrinsics(base), _intrinsics(just_inside), SHAPE) is not None

    def test_absent_focal_yields_nothing(self):
        assert _camera_matrix({"fx": 0.0}, _intrinsics(900.0), SHAPE) is None


def _project(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray,
             matrix: np.ndarray) -> np.ndarray:
    projected = (matrix @ (rotation @ points.T + translation)).T
    flat = projected[:, :2] / projected[:, 2:3]
    return flat.reshape(-1, 1, 2).astype(np.float32)


K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
IDENTITY = np.eye(3)
SHIFT = np.array([[1.2], [0.0], [0.0]])


class TestPlanarity:
    def test_a_flat_wall_is_reported_planar(self):
        rng = np.random.default_rng(2)
        # Tous les points sur un même plan : la scène est dégénérée.
        flat = np.c_[rng.uniform(-3, 3, 120), rng.uniform(-2, 2, 120), np.full(120, 11.0)]
        src = _project(flat, IDENTITY, np.zeros((3, 1)), K)
        dst = _project(flat, IDENTITY, SHIFT, K)
        assert _planarity(src, dst, np.ones(120, dtype=bool), 120) == "planar"

    def test_a_volumetric_scene_is_not_planar(self):
        rng = np.random.default_rng(3)
        volume = np.c_[
            rng.uniform(-3, 3, 120), rng.uniform(-2, 2, 120), rng.uniform(8, 16, 120)
        ]
        src = _project(volume, IDENTITY, np.zeros((3, 1)), K)
        dst = _project(volume, IDENTITY, SHIFT, K)
        assert _planarity(src, dst, np.ones(120, dtype=bool), 120) == "none"

    def test_too_few_inliers_is_not_a_verdict(self):
        """Sous huit points, l'homographie ne conclut rien."""
        points = np.zeros((4, 1, 2), dtype=np.float32)
        assert _planarity(points, points, np.ones(4, dtype=bool), 4) == "none"

    def test_threshold_is_a_ratio_of_support(self):
        assert 0.5 < PLANARITY_RATIO <= 1.0


class _Asset:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height


class TestIntrinsicsFromDeclaredFov:
    """Les vues de rue n'ont pas d'EXIF : le champ déclaré prend le relais."""

    def test_declared_fov_yields_a_focal(self):
        from hotel_pipeline.view_graph import _intrinsics_from_fov

        found = _intrinsics_from_fov(_Asset(640, 640), 80.0)
        assert found is not None
        assert found["fx"] > 0

    def test_focal_matches_the_geometry_of_the_field(self):
        from hotel_pipeline.view_graph import _intrinsics_from_fov

        # Un champ de 90° place la focale à la demi-largeur exactement.
        found = _intrinsics_from_fov(_Asset(1000, 800), 90.0)
        assert found["fx"] == pytest.approx(500.0, rel=1e-6)

    def test_narrower_field_gives_a_longer_focal(self):
        from hotel_pipeline.view_graph import _intrinsics_from_fov

        wide = _intrinsics_from_fov(_Asset(640, 640), 100.0)
        narrow = _intrinsics_from_fov(_Asset(640, 640), 40.0)
        assert narrow["fx"] > wide["fx"]

    def test_principal_point_is_the_centre(self):
        from hotel_pipeline.view_graph import _intrinsics_from_fov

        found = _intrinsics_from_fov(_Asset(800, 600), 80.0)
        assert found["cx"] == 400.0
        assert found["cy"] == 300.0

    def test_a_declared_focal_says_it_is_declared(self):
        """Un calibrage déclaré ne doit jamais se lire comme un relevé."""
        from hotel_pipeline.view_graph import _intrinsics_from_fov

        assert _intrinsics_from_fov(_Asset(640, 640), 80.0)["source"] == "declared_fov"

    def test_thumbnail_is_refused(self):
        """Le corpus porte des marqueurs 1×1 : ce ne sont pas des vues."""
        from hotel_pipeline.view_graph import _intrinsics_from_fov

        assert _intrinsics_from_fov(_Asset(1, 1), 80.0) is None

    def test_missing_dimensions_are_refused(self):
        from hotel_pipeline.view_graph import _intrinsics_from_fov

        assert _intrinsics_from_fov(_Asset(0, 0), 80.0) is None

    def test_absurd_field_is_refused(self):
        from hotel_pipeline.view_graph import _intrinsics_from_fov

        assert _intrinsics_from_fov(_Asset(640, 640), 0.0) is None
        assert _intrinsics_from_fov(_Asset(640, 640), 200.0) is None
        assert _intrinsics_from_fov(_Asset(640, 640), None) is None
