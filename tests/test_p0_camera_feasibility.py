from __future__ import annotations

import numpy as np

from hotel_pipeline.camera_feasibility import (
    CameraFeasibilityEvaluator,
    evaluate_canonical_camera,
)
from hotel_pipeline.canonical_camera import CanonicalCamera
from hotel_pipeline.conditioning.canonical_mesh import CanonicalSceneMesh
from hotel_pipeline.schemas.canonical_states import MeasurementState
from hotel_pipeline.workspace import Workspace


def _camera(focal=60.0, width=80, height=80, near=0.1):
    return CanonicalCamera(
        "PINHOLE", width, height, [focal, focal, width / 2, height / 2],
        near_m=near,
    )


def _planes(front=True, unknown=False):
    vertices = [[-3, -3, 10], [3, -3, 10], [3, 3, 10], [-3, 3, 10]]
    faces = [[0, 1, 2], [0, 2, 3]]
    if front:
        vertices += [[-1.5, -1.5, 8], [1.5, -1.5, 8], [1.5, 1.5, 8], [-1.5, 1.5, 8]]
        faces += [[4, 5, 6], [4, 6, 7]]
    states = [MeasurementState.UNKNOWN if unknown else MeasurementState.MEASURED] * len(faces)
    return CanonicalSceneMesh(
        np.asarray(vertices, float), np.asarray(faces),
        face_kind=["wall"] * len(faces), measurement_states=states,
    )


def test_exact_visibility_and_occluder_identity_come_from_surface_buffer():
    mesh = _planes(front=True)
    result = evaluate_canonical_camera(mesh, _camera(), unknown_threshold=1.0)
    surfaces = sorted(result["visible_surfaces"], key=lambda row: row["mean_distance_m"])
    near, far = surfaces
    assert near["visible_fraction"] > 0.95
    assert far["visible_fraction"] < 0.8
    assert near["surface_id"] in far["occluders"]
    assert far["occlusion_fraction"] > 0.15


def test_surface_pixel_fraction_is_distinct_from_visible_fraction():
    result = evaluate_canonical_camera(_planes(front=False), _camera(), unknown_threshold=1.0)
    surface = result["visible_surfaces"][0]
    assert surface["visible_fraction"] > 0.95
    assert surface["pixel_fraction"] < 0.5


def test_incidence_and_local_required_gsd_are_measured_from_visible_pixels():
    surface = evaluate_canonical_camera(
        _planes(front=False), _camera(), unknown_threshold=1.0,
    )["visible_surfaces"][0]
    assert surface["incidence_median_deg"] < 20.0
    assert 0 < surface["effective_required_gsd_m"] < 1.0
    assert surface["depth_min_m"] == surface["depth_max_m"] == 10.0


def test_unknown_visible_fraction_hard_rejects_pose():
    result = evaluate_canonical_camera(
        _planes(front=False, unknown=True), _camera(), unknown_threshold=0.05,
    )
    assert result["unknown_visible_fraction"] > 0.99
    assert not result["accepted"]
    assert any("unknown visible" in reason for reason in result["rejection_reasons"])


def test_texture_minimum_distance_is_a_hard_constraint():
    mesh = _planes(front=False)
    surface_id = next(iter(mesh.surface_catalog))
    result = evaluate_canonical_camera(
        mesh, _camera(), texture_min_distances={surface_id: 25.0},
        unknown_threshold=1.0,
    )
    assert not result["accepted"]
    assert any("texture requires" in reason for reason in result["rejection_reasons"])


def test_fov_changes_true_frame_occupancy_and_framing_score():
    mesh = _planes(front=False)
    tele = evaluate_canonical_camera(mesh, _camera(focal=110), unknown_threshold=1.0)
    wide = evaluate_canonical_camera(mesh, _camera(focal=30), unknown_threshold=1.0)
    assert tele["target_pixel_fraction"] > wide["target_pixel_fraction"]
    assert tele["subject_score"] > wide["subject_score"]


def test_feasibility_is_deterministic_and_uses_no_proxy():
    mesh, camera = _planes(front=True), _camera()
    first = evaluate_canonical_camera(mesh, camera, unknown_threshold=1.0)
    second = evaluate_canonical_camera(mesh, camera, unknown_threshold=1.0)
    assert first == second
    assert first["proxy_usage"] == 0
    assert first["input_mesh_digest"] == mesh.mesh_digest()


def test_camera_planner_consumes_exact_canonical_feasibility(tmp_path):
    mesh, camera = _planes(front=False), _camera()
    evaluator = CameraFeasibilityEvaluator(Workspace("hotel", root=tmp_path))
    field = evaluator.evaluate_pose(
        pose_id="exact", position_local_m=(0, 0, 0), yaw_deg=0,
        pitch_deg=0, fov_deg=60, canonical_mesh=mesh,
        canonical_camera=camera,
    )
    assert field.feasibility_mesh_digest == mesh.mesh_digest()
    assert field.visible_surfaces
    assert field.target_pixel_fraction > 0
    assert field.proxy_fraction == 0
    assert field.accepted
