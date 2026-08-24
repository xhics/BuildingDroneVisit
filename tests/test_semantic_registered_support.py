from __future__ import annotations

import numpy as np

from hotel_pipeline.conditioning.semantic_registered_support import (
    assess_single_view_linear_candidate,
    transform_points,
)


def test_transform_points_applies_sim3_registration_and_local_origin() -> None:
    points = np.asarray([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])

    transformed = transform_points(
        points,
        sim3_rotation=np.eye(3),
        sim3_translation=np.asarray([10.0, 20.0, 30.0]),
        sim3_scale=2.0,
        projected_origin_xy=(100.0, 200.0),
        registration_translation=np.asarray([1.0, -2.0, 5.0]),
        scene_origin_xyz=np.asarray([100.0, 200.0, 25.0]),
    )

    assert np.allclose(transformed[0], [13.0, 22.0, 16.0])
    assert np.allclose(transformed[1], [15.0, 26.0, 22.0])


def test_single_view_beam_accepts_only_directional_measured_support() -> None:
    accepted = assess_single_view_linear_candidate(
        "beam",
        np.asarray(
            [[0.0, 0.0, 4.0], [1.5, 0.1, 4.2], [3.0, 0.0, 4.1], [4.5, 0.1, 4.3]]
        ),
    )
    refused = assess_single_view_linear_candidate(
        "beam",
        np.asarray(
            [[0.0, 0.0, 1.0], [0.2, 0.1, 2.0], [0.1, 0.0, 3.0], [0.3, 0.2, 4.0]]
        ),
    )

    assert accepted["status"] == "accepted_support_only"
    assert refused["status"] == "refused"


def test_single_view_column_rejects_a_wide_cluster() -> None:
    result = assess_single_view_linear_candidate(
        "column",
        np.asarray([[0.0, 0.0, 1.0], [3.0, 0.2, 2.0], [6.0, 0.4, 3.0]]),
    )

    assert result["status"] == "refused"
    assert any("horizontal extent" in reason for reason in result["refusal_reasons"])
