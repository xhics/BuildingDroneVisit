import numpy as np
import pytest
from shapely.geometry import Polygon

from hotel_pipeline.conditioning.build_canonical import build_canonical_building_mesh
from hotel_pipeline.conditioning.canonical import _building
from hotel_pipeline.conditioning.roof_planes import RoofDecomposition, RoofPlane
from hotel_pipeline.conditioning.roof_reconstruct import reconstruct_roof
from hotel_pipeline.schemas.canonical_states import MeasurementState


def _grid(x0, x1, z_fn):
    x, y = np.meshgrid(np.linspace(x0, x1, 12), np.linspace(0, 10, 12))
    return np.column_stack([x.ravel(), y.ravel(), z_fn(x.ravel())])


def _gable():
    left = _grid(0, 10, lambda x: 6 + 0.4 * x)
    right = _grid(10, 20, lambda x: 14 - 0.4 * x)
    return RoofDecomposition("gable", [
        RoofPlane(left, [-0.4, 0, 1], [5, 5, 8], plane_id="plane_00", source_ids=["lidar"]),
        RoofPlane(right, [0.4, 0, 1], [15, 5, 8], plane_id="plane_01", source_ids=["lidar"]),
    ], total=len(left) + len(right))


def test_two_plane_roof_has_one_exact_finite_ridge():
    decomposition = _gable()
    roof = reconstruct_roof(decomposition, Polygon([(0, 0), (20, 0), (20, 10), (0, 10)]))
    assert roof is not None and roof.topology is not None
    assert len(roof.topology.ridges) == 1
    ridge = roof.topology.ridges[0]
    assert ridge.start[0] == pytest.approx(10.0, abs=0.003)
    assert ridge.end[0] == pytest.approx(10.0, abs=0.003)
    assert ridge.start[2] == pytest.approx(10.0, abs=0.003)
    assert ridge.end[2] == pytest.approx(10.0, abs=0.003)


def test_roof_planes_are_canonical_surfaces_and_watertight():
    mesh = build_canonical_building_mesh(
        np.array([[0, 0], [20, 0], [20, 10], [0, 10]], float),
        top_heights=6.0,
        roof_decomposition=_gable(),
    )
    mesh.assign_surface_ids("hotel", "main")
    assert any(value.startswith("hotel/main/roof/plane-") for value in mesh.surface_ids)
    assert len({value for value in mesh.surface_ids if "/roof/" in value}) == 2
    assert mesh.audit()["non_manifold_edges"] == 0
    assert mesh.audit()["boundary_edges"] == 0
    assert mesh.roof_topology is not None


def test_dense_measured_roof_is_built_once_without_overlay():
    vertices = np.array([
        [0, 0, 8], [10, 0, 8], [10, 6, 8], [0, 6, 8],
    ], float)
    faces = np.array([[0, 1, 2], [0, 2, 3]], int)
    mesh = build_canonical_building_mesh(
        vertices[:, :2], top_heights=99.0,
        measured_roof_vertices=vertices, measured_roof_faces=faces,
    )
    roof_faces = [i for i, kind in enumerate(mesh.face_kind) if kind == "roof"]
    assert len(roof_faces) == len(faces)
    assert max(mesh.vertices[mesh.faces[roof_faces]].ravel()) < 99.0
    assert mesh.roof_topology is not None
    assert mesh.audit()["boundary_edges"] == 0


def test_unsupported_logical_cap_is_never_measured():
    mesh = build_canonical_building_mesh(
        np.array([[0, 0], [4, 0], [4, 3], [0, 3]], float), top_heights=7.0
    )
    roof = [i for i, kind in enumerate(mesh.face_kind) if kind == "roof"]
    assert roof
    assert all(mesh.measurement_states[i] is MeasurementState.UNKNOWN for i in roof)


def test_level_change_creates_vertical_step_not_a_slope():
    low = _grid(0, 9.8, lambda x: np.full_like(x, 8.0))
    high = _grid(10.2, 20, lambda x: np.full_like(x, 9.5))
    decomposition = RoofDecomposition("step", [
        RoofPlane(low, [0, 0, 1], [5, 5, 8], plane_id="plane_00"),
        RoofPlane(high, [0, 0, 1], [15, 5, 9.5], plane_id="plane_01"),
    ], total=len(low) + len(high))
    roof = reconstruct_roof(
        decomposition, Polygon([(0, 0), (20, 0), (20, 10), (0, 10)]),
        min_area_m2=0.1,
    )
    assert roof is not None and roof.topology is not None
    assert len(roof.step_faces) == 2
    assert len(roof.topology.steps) == 1
    step_face = roof.vertices[roof.faces[roof.step_faces[0]]]
    assert np.ptp(step_face[:, 2]) == pytest.approx(1.5)


def test_bad_point_to_plane_rmse_cannot_be_high_confidence():
    rng = np.random.default_rng(4)
    points = _grid(0, 10, lambda x: np.full_like(x, 8.0))
    points[:, 2] += rng.normal(0, 0.65, len(points))
    plane = RoofPlane(points, [0, 0, 1], [5, 5, 8], plane_id="plane_00")
    assert plane.rmse > 0.25
    assert plane.confidence < 0.3


def test_published_building_has_no_parallel_roof_overlay():
    vertices = [[0, 0, 8], [10, 0, 8], [10, 6, 8], [0, 6, 8]]
    building = _building({
        "id": "hotel", "target": True, "fp": [row[:2] for row in vertices],
        "wh": [8, 8, 8, 8], "h": 8, "rv": vertices,
        "rf": [[0, 1, 2], [0, 2, 3]], "conf": 0.95,
        "source_ids": ["lidar"],
    })
    assert building["roof_surface"] is None
    assert building["roof_overlay"] is None
    assert building["roof_geometry_audit"]["roof_overlays"] == 0
    assert sum(
        kind == "roof" for kind in building["solid_mesh"]["face_kind"]
    ) == 2
