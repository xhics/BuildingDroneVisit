"""Tests de la fusion photographique multi-vues pour les façades."""

from __future__ import annotations

import numpy as np
import pytest

from hotel_pipeline.conditioning import facade_texture as ft
from hotel_pipeline.conditioning.facade_texture import (
    _build_triangles_from_payload,
    _canonical_mesh_pose_error,
    _pose_reprojection_allowed,
    _texture_registration_allowed,
)
from hotel_pipeline.geo.orthofacade import plane_from_edge


def test_texture_registration_refuses_a_rejected_registration() -> None:
    allowed, reason = _texture_registration_allowed(
        {"status": "refused", "metrics": {"fit": {"p90_m": 1.0}}}
    )
    assert not allowed
    assert "refus" in reason.lower()


def test_texture_registration_refuses_an_imprecise_accept() -> None:
    allowed, reason = _texture_registration_allowed(
        {"status": "accepted", "metrics": {"holdout": {"p90_m": 3.1}}}
    )
    assert not allowed
    assert "imprécise" in reason.lower()


def test_texture_registration_allows_a_precise_accept() -> None:
    allowed, reason = _texture_registration_allowed(
        {"status": "accepted", "metrics": {"holdout": {"p90_m": 2.15}}}
    )
    assert allowed
    assert reason == ""


def test_build_triangles_from_payload_refuses_noncanonical_geometry():
    class _Cam:
        position = np.array([0.0, -30.0, 2.5], dtype=float)
        f = 400.0

        def project(self, points):
            d = np.asarray(points, float) - self.position
            z = d[:, 1]
            safe = np.maximum(z, 1e-6)
            x = 320 + self.f * d[:, 0] / safe
            y = 240 - self.f * d[:, 2] / safe
            return np.stack([x, y], axis=1), z

    cam = _Cam()
    plane = plane_from_edge(
        np.array([-5.0, 0.0, 0.0]), np.array([5.0, 0.0, 0.0]), 6.0, "EDGE"
    )
    mask = ft._facade_polygon_mask(cam, plane, 640, 480)
    assert mask is not None
    assert mask.any()
    assert not mask.all()
    assert not bool(mask[10, 10])
    assert bool(mask[240, 320])


def test_build_triangles_from_payload():
    payload = {
        "volumes": [
            {
                "fp": [[0, 0], [10, 0], [10, 10], [0, 10]],
                "wh": [8.0, 10.0, 10.0, 8.0],
                "h": 10.0,
            }
        ]
    }
    with pytest.raises(ValueError, match="canonical solid mesh"):
        _build_triangles_from_payload(payload)


def test_pose_error_is_symmetric_and_reports_tail_metrics():
    observed = np.zeros((40, 60), dtype=bool)
    observed[10:30, 15:45] = True
    rendered = np.zeros_like(observed)
    rendered[10:30, 17:47] = True

    class Frame:
        triangle_id = np.where(rendered, 4, -1)
        depth_z = np.where(rendered, 20.0, np.inf)

    metrics = _canonical_mesh_pose_error(Frame(), observed, 4, 1, 1000.0)
    assert metrics["median_px"] >= 0.0
    assert metrics["p90_px"] > 0.0
    assert metrics["p90_px"] >= metrics["median_px"]
    assert metrics["robust_max_px"] >= metrics["p90_px"]
    assert metrics["rendered_edge_pixels"] > 0
    assert metrics["observed_edge_pixels"] > 0


def test_pose_pixel_gate_rejects_bad_median_or_tail():
    allowed, reason = _pose_reprojection_allowed({"median_px": 2.0, "p90_px": 5.0})
    assert allowed and reason == ""
    assert not _pose_reprojection_allowed({"median_px": 3.1, "p90_px": 5.0})[0]
    allowed, reason = _pose_reprojection_allowed({"median_px": 2.0, "p90_px": 6.1})
    assert not allowed
    assert "p90" in reason
