from __future__ import annotations

import json

import numpy as np
import pytest

from hotel_pipeline.architectural_geometry import primitive_mesh
from hotel_pipeline.canonical_gltf import canonical_mesh_arrays, export_canonical_gltf
from hotel_pipeline.reality_gate import (
    PathSurfaceObservation,
    assess_texture_resolution,
    minimum_texture_distance_m,
    validate_camera_path_reality,
)
from hotel_pipeline.schemas.canonical_states import RealityLevel


def _tetrahedron() -> dict:
    return {
        "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "faces": [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]],
        "face_kind": ["base", "wall", "wall", "roof"],
    }


def test_conditioned_scene_buildings_are_exported_without_volumes_alias(tmp_path) -> None:
    payload = {
        "buildings": [
            {"feature_id": "hotel", "solid_mesh": _tetrahedron()}
        ]
    }
    vertices, triangles = canonical_mesh_arrays(payload)
    assert len(vertices) == 4
    assert len(triangles) == 4

    path = tmp_path / "scene.gltf"
    metadata = export_canonical_gltf(payload, path)
    stored = json.loads(path.read_text("utf-8"))
    assert metadata["building_count"] == 1
    assert metadata["triangle_count_by_category"]["building"] == 4
    assert stored["extras"]["triangle_count"] == 4


def test_environment_geometry_is_not_dropped_from_canonical_export() -> None:
    payload = {
        "buildings": [{"feature_id": "hotel", "solid_mesh": _tetrahedron()}],
        "vegetation": [
            {
                "rings": [
                    [[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]],
                    [[-1, -1, 5], [1, -1, 5], [1, 1, 5], [-1, 1, 5]],
                ],
                "provenance_class": "LIDAR_MEASURED",
            }
        ],
        "furniture": [
            {"centre": [4, 4], "radius_m": 0.1, "height_m": 5.0}
        ],
    }
    _vertices, triangles = canonical_mesh_arrays(payload)
    assert len(triangles) > 4


def test_architectural_detail_uses_its_real_faces_not_vertex_fan() -> None:
    canopy = primitive_mesh("canopy", center=(0, 0, 3), size=(4, 2, 0.3))
    payload = {
        "volumes": [
            {
                "solid": {
                    "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                    "faces": [[0, 1, 2]],
                }
            }
        ],
        "facade_features": [{"kind": "canopy", **canopy}],
    }
    _vertices, triangles = canonical_mesh_arrays(payload)
    # One building triangle + six quad faces => twelve canopy triangles.
    assert len(triangles) == 13


def test_12cm_texture_cannot_support_1080p_closeup_without_hallucination() -> None:
    minimum = minimum_texture_distance_m(
        0.12, output_width_px=1920, horizontal_fov_deg=60.0, max_upscale=2.0
    )
    assert minimum == pytest.approx(99.77, abs=0.2)

    close = assess_texture_resolution(
        "facade-east",
        distance_m=30.0,
        effective_gsd_m=0.12,
        output_width_px=1920,
        horizontal_fov_deg=60.0,
        max_upscale=2.0,
    )
    assert not close.safe
    assert close.upscale_factor > 6.0


def test_camera_path_is_rejected_when_texture_resolution_is_too_coarse() -> None:
    accepted, reasons = validate_camera_path_reality(
        [
            PathSurfaceObservation(
                "facade-east",
                RealityLevel.SAFE_FOR_CLOSEUP,
                30.0,
                0.0,
                texture_effective_gsd_m=0.12,
                texture_coverage=0.95,
                texture_sharpness=0.9,
                texture_consensus_views=3,
            )
        ],
        output_width_px=1920,
        horizontal_fov_deg=60.0,
        max_texture_upscale=2.0,
    )
    assert not accepted
    assert any("upscale" in reason for reason in reasons)


def test_v2v_structure_score_detects_large_geometry_drift() -> None:
    cv2 = pytest.importorskip("cv2")
    from simple_mode.reality_qa import edge_structure_score

    source = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.rectangle(source, (50, 50), (270, 190), (255, 255, 255), 4)
    cv2.line(source, (80, 80), (240, 80), (255, 255, 255), 3)
    cv2.line(source, (80, 120), (240, 120), (255, 255, 255), 3)

    same_structure = source.copy()
    same_structure[source > 0] = 180
    drifted = np.zeros_like(source)
    cv2.rectangle(drifted, (85, 50), (305, 190), (255, 255, 255), 4)
    cv2.line(drifted, (115, 80), (275, 80), (255, 255, 255), 3)
    cv2.line(drifted, (115, 120), (275, 120), (255, 255, 255), 3)

    stable_score = edge_structure_score(source, same_structure)
    drift_score = edge_structure_score(source, drifted)
    assert stable_score is not None and stable_score > 0.9
    assert drift_score is not None and drift_score < 0.5
