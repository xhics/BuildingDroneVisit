from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hotel_pipeline.conditioning.canonical import build, viewer_payload
from hotel_pipeline.conditioning.solid import audit, closed_solid
from hotel_pipeline.workspace import Workspace


def _workspace(tmp_path: Path) -> Workspace:
    workspace = Workspace("hotel-test", root=tmp_path)
    workspace.create()
    workspace.write_json(
        "11_conditioning/viewer_payload.json",
        {
            "hotel": "hotel-test",
            "centre": [100.0, 200.0],
            "volumes": [
                {
                    "id": "TARGET",
                    "target": True,
                    "h": 8.0,
                    "assumed": False,
                    "conf": 0.95,
                    "fp": [[0, 0], [10, 0], [10, 6], [0, 6]],
                    "wh": [8, 8, 9, 9],
                }
            ],
            "vegetation": [
                {"c": [14, 4], "r": 3, "h": 9, "shape": "etale"}
            ],
            "furniture": [],
            "ground": [],
            "ridges": [],
            "observation": {"triangulable_fraction": 0.0},
        },
    )
    return workspace


def test_solide_concave_est_ferme_et_supporte() -> None:
    footprint = np.array(
        [[0, 0], [8, 0], [8, 3], [3, 3], [3, 8], [0, 8]], dtype=float
    )
    mesh = closed_solid(footprint, np.array([7, 7, 8, 8, 6, 6], dtype=float))

    report = audit(mesh)

    assert report["watertight"] is True
    assert report["supported"] is True
    assert report["boundary_edges"] == 0
    assert report["non_manifold_edges"] == 0
    assert report["connected_components"] == 1
    assert report["volume_m3"] > 0
    assert report["self_intersection"] is False


def test_scene_canonique_ne_promeut_aucune_observation_en_3d(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    workspace.write_json(
        "11_conditioning/semantic_observations.json",
        {
            "run_id": "semantic-test",
            "model_id": "grounding-dino-test",
            "segmentation_backend": "sam2-test",
            "summary": {
                "images": 2,
                "observations": 5,
                "segmented": 5,
                "geometry_3d_created": 0,
            },
        },
    )
    workspace.write_json(
        "11_conditioning/semantic_correspondences.json",
        {
            "run_id": "semantic-links-test",
            "coordinate_frame": "anchor_colmap_model",
            "summary": {
                "multiview_instances": 2,
                "multiview_observations": 4,
                "shared_measured_tracks": 12,
                "geometry_3d_created": 0,
            },
        },
    )
    workspace.write_json(
        "11_conditioning/vertical_registration.json",
        {
            "run_id": "vertical-test",
            "status": "accepted",
            "scene_geometry_applied": False,
            "hypothesis": {"translation_projected_m": [0, 0, 28]},
            "metrics": {"holdout": {"support_fraction_1m": 0.3}},
            "refusal_reasons": [],
        },
    )
    workspace.write_json(
        "11_conditioning/registered_semantic_support.json",
        {
            "run_id": "registered-support-test",
            "status": "registered_support_ready",
            "coordinate_frame": {"kind": "conditioned_scene_local_ground"},
            "summary": {
                "registered_instances": 1,
                "registered_point_assignments": 2,
                "unique_registered_points": 2,
                "surface_geometry_created": 0,
            },
            "instances": [
                {
                    "instance_id": "tree-test",
                    "class": "tree_evergreen",
                    "provenance_class": "COLMAP_MEASURED",
                    "points": [
                        {"point3d_id": 1, "xyz": [1, 2, 3]},
                        {"point3d_id": 2, "xyz": [2, 3, 4]},
                    ],
                }
            ],
            "single_view_candidates": [
                {
                    "instance_id": "beam-test",
                    "class": "beam",
                    "provenance_class": "COLMAP_MEASURED",
                    "semantic_evidence_class": "single_view_candidate",
                    "points": [{"point3d_id": 3, "xyz": [3, 4, 5]}],
                }
            ],
        },
    )
    workspace.write_json(
        "11_conditioning/semantic_surfaces.json",
        {
            "run_id": "semantic-surfaces-test",
            "status": "completed",
            "summary": {
                "audited_instances": 1,
                "accepted_surfaces": 1,
                "refused_instances": 0,
                "geometry_3d_created": 1,
            },
            "surfaces": [
                {
                    "surface_id": "surface-sign",
                    "class": "road_sign",
                    "surface": {
                        "vertices": [[0, 0, 1], [1, 0, 1], [0, 1, 1]],
                        "faces": [[0, 1, 2]],
                    },
                }
            ],
        },
    )

    outputs = build(workspace)
    scene = json.loads(outputs["scene"].read_text("utf-8"))
    topology = json.loads(outputs["topology"].read_text("utf-8"))
    benchmark = json.loads(outputs["benchmark"].read_text("utf-8"))
    observations = json.loads(outputs["observations"].read_text("utf-8"))

    assert scene["summary"]["watertight_buildings"] == 1
    assert scene["summary"]["supported_buildings"] == 1
    assert topology["all_watertight"] is True
    assert benchmark["acceptance"]["vegetation_has_rings"] is True
    assert observations["summary"]["geometry_3d_created"] == 0
    assert scene["vegetation"][0]["provenance_class"] == "OCCLUDED_INFERRED"
    assert scene["summary"]["semantic_image_observations"]["segmented"] == 5
    assert scene["summary"]["semantic_image_observations"]["geometry_3d_created"] == 0
    assert scene["summary"]["semantic_multiview"]["multiview_instances"] == 2
    assert scene["summary"]["semantic_multiview"]["geometry_3d_created"] == 0
    assert scene["summary"]["colmap_lidar_registration"]["status"] == "accepted"
    assert scene["summary"]["colmap_lidar_registration"]["scene_geometry_applied"] is False
    assert scene["summary"]["registered_semantic_support"]["unique_registered_points"] == 2
    assert scene["summary"]["registered_semantic_support"]["surface_geometry_created"] == 0
    assert scene["summary"]["semantic_surfaces"]["geometry_3d_created"] == 1

    payload = viewer_payload(scene)
    assert payload["counts"]["watertight_buildings"] == 1
    assert len(payload["vegetation"][0]["rings"]) == 6
    assert payload["semantic"]["run_id"] == "semantic-test"
    assert payload["semantic_multiview"]["run_id"] == "semantic-links-test"
    assert payload["registration"]["status"] == "accepted"
    assert len(payload["semantic_support_points"]) == 3
    assert payload["semantic_support_points"][-1]["semantic_evidence_class"] == "single_view_candidate"
    assert len(payload["semantic_surfaces"]) == 1
    assert payload["semantic_surface_summary"]["accepted_surfaces"] == 1
