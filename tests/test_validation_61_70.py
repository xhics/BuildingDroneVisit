from __future__ import annotations

import json
import numpy as np

from hotel_pipeline.architectural_geometry import Opening, primitive_mesh, wall_hit_is_solid
from hotel_pipeline.camera_feasibility import distance_to_mesh, segment_intersects_mesh, yaw_pitch_quaternion
from hotel_pipeline.canonical_gltf import canonical_mesh_arrays, export_canonical_gltf, mesh_digest
from hotel_pipeline.viewer import _webgl_html


def test_free_endpoints_do_not_hide_wall_crossing():
    wall = np.array([[[0, -3, -3], [0, 3, -3], [0, 0, 3]]], float)
    assert distance_to_mesh(np.array([-2, 0, 0]), wall) > 1
    assert distance_to_mesh(np.array([2, 0, 0]), wall) > 1
    assert segment_intersects_mesh(np.array([-2, 0, 0]), np.array([2, 0, 0]), wall)


def test_altitude_changes_mesh_clearance():
    roof = np.array([[[0, 0, 10], [20, 0, 10], [0, 20, 10]]], float)
    assert distance_to_mesh(np.array([2, 2, 40]), roof) > distance_to_mesh(np.array([2, 2, 10.2]), roof)


def test_pose_orientation_is_a_normalized_quaternion():
    quaternion = np.asarray(yaw_pitch_quaternion(33.0, -12.0))
    np.testing.assert_allclose(np.linalg.norm(quaternion), 1.0, atol=1e-8)


def test_gltf_exports_exact_canonical_counts_and_digest(tmp_path):
    payload = {"volumes": [{"solid": {
        "vertices": [[0,0,0], [1,0,0], [0,1,0], [0,0,1]],
        "faces": [[0,1,2], [0,3,1], [0,2,3], [1,3,2]],
    }}]}
    vertices, triangles = canonical_mesh_arrays(payload)
    path = tmp_path / "scene.gltf"
    metadata = export_canonical_gltf(payload, path)
    stored = json.loads(path.read_text())
    assert metadata["vertex_count"] == 4
    assert metadata["triangle_count"] == 4
    assert metadata["mesh_digest"] == mesh_digest(vertices, triangles)
    assert stored["extras"]["reextruded"] is False


def test_opening_removes_wall_hit_instead_of_adding_a_plane():
    door = Opening("door", ((1,0), (2,0), (2,2.2), (1,2.2)), 0.15, "glass", "MEASURED")
    assert not wall_hit_is_solid(1.5, 1.0, [door])
    assert wall_hit_is_solid(3.0, 1.0, [door])


def test_column_and_canopy_are_volumes():
    column = primitive_mesh("column", center=(0,0,1.5), radius_m=0.25, height_m=3.0)
    canopy = primitive_mesh("canopy", center=(0,0,3), size=(4,2,0.3))
    assert len(column["faces"]) >= 16 and len(column["vertices"]) >= 32
    assert len(canopy["faces"]) == 6 and len(canopy["vertices"]) == 8


def test_viewer_uses_gpu_depth_and_perspective_uv(tmp_path):
    payload = {"volumes": [{"solid": {"vertices": [[0,0,0],[1,0,0],[0,1,0]], "faces": [[0,1,2]]}}]}
    path = tmp_path / "scene.gltf"
    export_canonical_gltf(payload, path)
    html = _webgl_html(payload, {}, json.loads(path.read_text()))
    assert "getContext('webgl2'" in html
    assert "gl.enable(gl.DEPTH_TEST)" in html
    assert "in vec2 U" in html
    assert "getContext('2d')" not in html
