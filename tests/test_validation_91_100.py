from datetime import date

import numpy as np
import pytest

from hotel_pipeline.reality_gate import (
    CanonicalSurfaceIdentity,
    GeometryUncertainty,
    PathSurfaceObservation,
    PoseUncertainty,
    RealityMetrics,
    SiteStructure,
    TemporalSource,
    assess_reality,
    resolve_material_appearance,
    temporal_conflicts,
    validate_camera_path_reality,
    validate_surface_assignments,
    visible_fraction,
)
from hotel_pipeline.schemas.canonical_states import MaterialClass, RealityLevel


def test_91_dense_lidar_is_less_uncertain_than_two_photo_inference() -> None:
    lidar = GeometryUncertainty("lidar_dense", (0.015, 0.015, 0.025))
    photos = GeometryUncertainty("sfm_two_views", (0.18, 0.18, 0.35))
    assert lidar.sigma_m < photos.sigma_m
    assert lidar.geometry_confidence > photos.geometry_confidence


def test_92_perturbed_camera_loses_texture_weight() -> None:
    stable = PoseUncertainty((0.005, 0.005, 0.01), (0.02, 0.02, 0.03))
    perturbed = PoseUncertainty((0.2, 0.2, 0.3), (1.0, 1.0, 1.0))
    assert stable.texture_weight(focal_px=1600, depth_m=15) > perturbed.texture_weight(focal_px=1600, depth_m=15)


def test_93_new_extension_creates_explicit_temporal_conflict() -> None:
    lidar = TemporalSource("lidar-2019", date(2019, 6, 1), None, "main-only")
    photo = TemporalSource("photo-2025", date(2025, 6, 1), None, "main-plus-extension")
    assert temporal_conflicts([lidar, photo])[0].source_ids == ("lidar-2019", "photo-2025")


def test_94_reflective_window_is_not_average_of_sky_and_trees() -> None:
    result = resolve_material_appearance(MaterialClass.GLASS, [
        ("sky", (0.1, 0.4, 1.0), 5.0), ("trees", (0.0, 0.7, 0.1), 40.0)
    ])
    assert result.base_color[:3] == (0.35, 0.42, 0.48)
    assert result.selected_view_id == "sky"


def test_95_appearance_resolution_does_not_mutate_geometry() -> None:
    vertices = np.array([[0.0, 0.0, 0.0], [1, 0, 0], [0, 1, 0]])
    before = vertices.copy()
    resolve_material_appearance(MaterialClass.DIFFUSE, [("a", (1, 0, 0), 0), ("b", (0, 1, 0), 0)])
    np.testing.assert_array_equal(vertices, before)


def test_96_surface_cannot_be_shared_by_two_hotels() -> None:
    with pytest.raises(ValueError, match="shared"):
        validate_surface_assignments([
            CanonicalSurfaceIdentity("hotel-a", "main", "facade-1"),
            CanonicalSurfaceIdentity("hotel-b", "main", "facade-1"),
        ])


def test_97_retaining_wall_blocks_low_path() -> None:
    wall = SiteStructure("wall-1", "retaining_wall", np.array([
        [[0, -2, 0], [0, 2, 0], [0, 2, 2]], [[0, -2, 0], [0, 2, 2], [0, -2, 2]],
    ], dtype=float))
    assert wall.blocks_segment((-1, 0, 0.5), (1, 0, 0.5))


def test_98_partial_occlusion_reports_coverage_fraction() -> None:
    samples = [True] * 60 + [False] * 40
    assert visible_fraction(samples) == pytest.approx(0.6)


def test_99_watertight_cube_fails_multievidence_reality_gate() -> None:
    cube = RealityMetrics(0.01, 18.0, 0.35, 0.8, False, True, False, True, True, 1.0, 0.0, 0.2)
    result = assess_reality(cube)
    assert result.level is RealityLevel.NO_FLY_NO_RENDER
    assert {"reprojection", "silhouette", "lidar", "roof_topology"} <= set(result.failed_evidence)


def test_100_unknown_facade_rejects_close_camera_path() -> None:
    accepted, reasons = validate_camera_path_reality([
        PathSurfaceObservation("facade-unknown", RealityLevel.INFERRED, 3.0, 0.2)
    ])
    assert not accepted
    assert any("SAFE_FOR_CLOSEUP" in reason for reason in reasons)

