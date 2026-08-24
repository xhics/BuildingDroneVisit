from __future__ import annotations

import numpy as np

from hotel_pipeline.conditioning.semantic_surface import fit_planar_candidate


def _vertical_points() -> tuple[list[int], np.ndarray]:
    point_ids = list(range(1, 16))
    points = np.asarray(
        [
            [0.01 * ((index % 3) - 1), float(index % 5), float(index // 5)]
            for index in point_ids
        ],
        dtype=float,
    )
    return point_ids, points


def test_planar_surface_uses_measured_convex_hull_after_holdout() -> None:
    point_ids, points = _vertical_points()

    result = fit_planar_candidate(point_ids, points)

    assert result["status"] == "accepted"
    assert result["metrics"]["holdout_points"] == 3
    assert result["surface"]["type"] == "planar_measured_convex_hull"
    assert result["surface"]["thickness_m"] is None
    assert len(result["surface"]["faces"]) >= 2


def test_planar_surface_refuses_when_holdout_leaves_the_plane() -> None:
    point_ids, points = _vertical_points()
    for index, point_id in enumerate(point_ids):
        if point_id % 5 == 0:
            points[index, 0] = 1.0

    result = fit_planar_candidate(point_ids, points)

    assert result["status"] == "refused"
    assert any("holdout p90 residual" in reason for reason in result["refusal_reasons"])
