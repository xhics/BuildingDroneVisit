from __future__ import annotations

import numpy as np
import pytest

from hotel_pipeline.canonical_camera import CanonicalCamera
from hotel_pipeline.conditioning.canonical_mesh import CanonicalSceneMesh
from hotel_pipeline.dense_holdout_renderer import (
    camera_digest,
    compare_holdout,
    render_digest,
    render_holdout,
    validate_dense_holdouts,
)


def _camera(width=64, height=64, distortion=0.0):
    return CanonicalCamera(
        "SIMPLE_RADIAL", width, height,
        [50.0, width / 2, height / 2, distortion], near_m=0.1,
        camera_id="holdout-camera",
    )


def _square(split=False, z=10.0, shift=0.0):
    if split:
        vertices = np.array([
            [-2 + shift, -2, z], [0 + shift, -2, z], [2 + shift, -2, z],
            [-2 + shift, 0, z], [0 + shift, 0, z], [2 + shift, 0, z],
            [-2 + shift, 2, z], [0 + shift, 2, z], [2 + shift, 2, z],
        ])
        faces = np.array([[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4],
                          [3, 4, 7], [3, 7, 6], [4, 5, 8], [4, 8, 7]])
    else:
        vertices = np.array([[-2 + shift, -2, z], [2 + shift, -2, z],
                             [2 + shift, 2, z], [-2 + shift, 2, z]])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
    return CanonicalSceneMesh(vertices, faces, face_kind=["wall"] * len(faces))


def test_exact_camera_zbuffer_ids_normals_and_determinism():
    camera, mesh = _camera(), _square()
    render = render_holdout(mesh, camera)
    assert render.camera_digest == camera_digest(camera)
    assert render.input_mesh_digest == mesh.mesh_digest()
    assert render.proxy_renderer_usage == 0
    assert render.target_building_mask[32, 32]
    assert render.depth[32, 32] == pytest.approx(10.0, abs=1e-9)
    assert render.triangle_id[32, 32] in set(mesh.triangle_ids)
    assert np.linalg.norm(render.normal[32, 32]) == pytest.approx(1.0)
    assert render_digest(render) == render_digest(render_holdout(mesh, camera))


def test_near_plane_triangle_is_clipped_without_exploding():
    mesh = CanonicalSceneMesh(
        np.array([[-0.02, -0.02, 0.05], [1, -1, 1], [-1, 1, 1.0]]),
        np.array([[0, 1, 2]]), face_kind=["wall"],
    )
    render = render_holdout(mesh, _camera())
    assert np.isfinite(render.depth[render.silhouette]).all()
    assert render.silhouette.sum() <= render.silhouette.size


def test_retriangulation_does_not_change_depth_or_silhouette():
    camera = _camera()
    coarse, fine = render_holdout(_square(), camera), render_holdout(_square(True), camera)
    assert np.array_equal(coarse.silhouette, fine.silhouette)
    common = coarse.silhouette & fine.silhouette
    assert np.max(np.abs(coarse.depth[common] - fine.depth[common])) < 1e-9


def test_distorted_camera_uses_the_same_canonical_camera_contract():
    camera = _camera(distortion=0.15)
    render = render_holdout(_square(), camera)
    assert render.camera_digest == camera_digest(camera)
    assert render.silhouette.any()


def test_geometry_failure_and_appearance_failure_remain_separate():
    camera = _camera()
    truth = render_holdout(_square(), camera)
    wrong_geometry = render_holdout(_square(shift=1.0), camera)
    bad = compare_holdout(wrong_geometry, truth.rgb, truth.target_building_mask)
    wrong_rgb = np.full_like(truth.rgb, 255)
    appearance = compare_holdout(
        truth, wrong_rgb, truth.target_building_mask,
        supported_appearance_mask=truth.target_building_mask,
    )
    assert bad.silhouette_iou < 1.0
    assert appearance.silhouette_iou == 1.0
    assert appearance.appearance_score < 1.0


def test_dense_evidence_is_published_per_image_and_surface(tmp_path):
    camera, mesh = _camera(), _square()
    truth = render_holdout(mesh, camera)
    output = tmp_path / "run-dense-holdout.json"
    payload = validate_dense_holdouts(
        mesh, [{"image_id": "h1", "camera": camera,
                "camera_digest": camera_digest(camera), "rgb": truth.rgb,
                "building_mask": truth.target_building_mask}], output,
        train_asset_ids=["t1", "t2", "t3"], holdout_asset_ids=["h1"],
        frozen_model_digest="a" * 64,
    )
    assert output.is_file()
    assert payload["proxy_renderer_usage"] == 0
    assert payload["canonical_mesh_usage_fraction"] == 1.0
    assert payload["silhouette_iou"] == 1.0
    assert payload["holdout_results"][0]["image_id"] == "h1"
    assert payload["surface_scores"]

