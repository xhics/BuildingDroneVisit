from __future__ import annotations

import numpy as np

from hotel_pipeline.camera_feasibility import capsule_intersects_mesh, distance_to_mesh, segment_intersects_mesh
from hotel_pipeline.dense_holdout_renderer import rasterize_canonical_mesh, silhouette_iou
from hotel_pipeline.geo.facade_visibility import local_facade_reprojection_gate
from hotel_pipeline.sparse_reprojection import load_points_by_id


def test_colmap_point_ids_are_not_array_indexes(tmp_path):
    normalized = tmp_path / "normalized"
    normalized.mkdir()
    (normalized / "points3D").write_text(
        "3 0 0 1 0 0 0 0\n57 2 0 1 0 0 0 0\n9821 4 0 1 0 0 0 0\n"
    )
    points = load_points_by_id(tmp_path)
    assert set(points) == {3, 57, 9821}
    assert points[9821].tolist() == [4.0, 0.0, 1.0]


def test_silhouette_iou_is_not_bbox_iou():
    a = np.zeros((8, 8), bool); a[1:7, 1] = True; a[1, 1:7] = True
    b = np.zeros((8, 8), bool); b[1:7, 6] = True; b[6, 1:7] = True
    # Their bounding boxes are identical (bbox IoU=1), but actual masks barely overlap.
    assert silhouette_iou(a, b) < 0.2


class _Camera:
    position = np.array([0.0, 0.0, 0.0])
    def project(self, points):
        points = np.asarray(points, float)
        return np.c_[8 + 8 * points[:, 0] / points[:, 2], 8 - 8 * points[:, 1] / points[:, 2]], points[:, 2]


def test_dense_rasterizer_outputs_surface_depth_normal_and_mask():
    tri = np.array([[-1, -1, 4], [1, -1, 4], [0, 1, 4]], float)
    render = rasterize_canonical_mesh(_Camera(), [tri], [57], 16, 16)
    assert render.silhouette.any()
    assert np.all(render.surface_id[render.silhouette] == 57)
    assert np.isfinite(render.depth[render.silhouette]).all()
    assert np.linalg.norm(render.normal[render.silhouette], axis=1).mean() > 0.9


def test_mesh_clearance_is_translation_invariant():
    tri = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], float)
    p = np.array([0.5, 0.5, 3.0])
    shift = np.array([10_000.0, -4_000.0, 20.0])
    assert distance_to_mesh(p, np.array([tri])) == distance_to_mesh(p + shift, np.array([tri + shift]))


def test_path_crossing_a_wall_collides():
    wall = np.array([[0, -2, -2], [0, 2, -2], [0, 0, 2]], float)
    assert segment_intersects_mesh(np.array([-1, 0, 0.0]), np.array([1, 0, 0.0]), np.array([wall]))


def test_camera_capsule_rejects_near_miss_without_centreline_intersection():
    wall = np.array([[0, -2, -2], [0, 2, -2], [0, 0, 2]], float)
    start, end = np.array([-1, 0.2, 2.2]), np.array([1, 0.2, 2.2])
    assert not segment_intersects_mesh(start, end, np.array([wall]))
    assert capsule_intersects_mesh(start, end, np.array([wall]), 0.35)


def test_local_facade_failure_cannot_hide_in_global_score():
    result = local_facade_reprojection_gate([
        ("front", "a", 1.0, 0.05, 20),
        ("side", "a", 30.0, 1.0, 20),
    ])
    assert not result["passed"]
