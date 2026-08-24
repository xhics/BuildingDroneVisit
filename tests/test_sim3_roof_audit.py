from __future__ import annotations

import numpy as np

from hotel_pipeline.conditioning.sim3_roof_audit import (
    compose_sim3,
    roof_edge_metrics,
)
from hotel_pipeline.geometry_align import apply_sim3


def test_compose_sim3_matches_two_successive_transforms() -> None:
    points = np.asarray([[1.0, 2.0, 3.0], [-2.0, 0.5, 4.0]])
    first_rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    second_rotation = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    first_translation = np.asarray([2.0, 3.0, 4.0])
    second_translation = np.asarray([-1.0, 5.0, 2.0])

    rotation, translation, scale = compose_sim3(
        first_rotation,
        first_translation,
        2.0,
        second_rotation,
        second_translation,
        0.5,
    )
    successive = apply_sim3(
        apply_sim3(points, first_rotation, first_translation, 2.0),
        second_rotation,
        second_translation,
        0.5,
    )

    assert np.allclose(apply_sim3(points, rotation, translation, scale), successive)


def test_roof_edge_metrics_reports_coverage_instead_of_inventing_edges() -> None:
    ridges = [
        (np.asarray([0.0, 0.0, 5.0]), np.asarray([4.0, 0.0, 5.0])),
        (np.asarray([0.0, 10.0, 5.0]), np.asarray([4.0, 10.0, 5.0])),
    ]
    points = np.asarray([[1.0, 0.2, 5.0], [3.0, -0.1, 5.0]])

    metrics = roof_edge_metrics(points, ridges)

    assert metrics["lidar_edges"] == 2
    assert metrics["edges_within_1m"] == 1
    assert metrics["edge_coverage_fraction_1m"] == 0.5
    assert metrics["support_to_edge_p90_m"] < 1.0
