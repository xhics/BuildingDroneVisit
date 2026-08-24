from __future__ import annotations

from hotel_pipeline.conditioning.semantic_correspondence import (
    build_tracks,
    point_in_polygon,
)


def _observation(observation_id: str, asset_id: str, object_class: str) -> dict:
    return {
        "observation_id": observation_id,
        "asset_id": asset_id,
        "class": object_class,
    }


def test_point_in_polygon_includes_interior_and_boundary() -> None:
    polygon = [[0, 0], [10, 0], [10, 8], [0, 8]]

    assert point_in_polygon((5, 4), polygon) is True
    assert point_in_polygon((0, 4), polygon) is True
    assert point_in_polygon((12, 4), polygon) is False


def test_tracks_require_shared_measured_points_and_same_class() -> None:
    observations = [
        _observation("tree-a", "view-a", "tree_evergreen"),
        _observation("tree-b", "view-b", "tree_evergreen"),
        _observation("sign-c", "view-c", "road_sign"),
    ]
    support = {
        "tree-a": {1, 2, 3, 8},
        "tree-b": {1, 2, 3, 9},
        "sign-c": {1, 2, 3},
    }
    xyz = {point_id: (float(point_id), 0.0, 1.0) for point_id in range(1, 10)}

    pairs, instances = build_tracks(observations, support, xyz)

    assert len(pairs) == 1
    assert pairs[0]["decision"] == "accepted"
    assert len(instances) == 1
    assert instances[0]["class"] == "tree_evergreen"
    assert instances[0]["shared_point3d_ids"] == [1, 2, 3]
    assert instances[0]["geometry_3d"] is None
    assert instances[0]["scene_integration_status"] == "blocked_vertical_registration"


def test_tracks_reject_transitive_merge_of_two_objects_in_same_view() -> None:
    observations = [
        _observation("tree-a-strong", "view-a", "tree_evergreen"),
        _observation("tree-a-other", "view-a", "tree_evergreen"),
        _observation("tree-b", "view-b", "tree_evergreen"),
    ]
    support = {
        "tree-a-strong": {1, 2, 3, 4},
        "tree-a-other": {1, 2, 3},
        "tree-b": {1, 2, 3, 4},
    }
    xyz = {point_id: (float(point_id), 0.0, 1.0) for point_id in range(1, 5)}

    pairs, instances = build_tracks(observations, support, xyz)

    assert len(instances) == 1
    assert instances[0]["observation_ids"] == ["tree-a-strong", "tree-b"]
    assert any(item["decision"] == "rejected_same_view_conflict" for item in pairs)
