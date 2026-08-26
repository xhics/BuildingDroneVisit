"""Tests de la gate de pose par (façade, vue)."""

from __future__ import annotations

import numpy as np
import pytest

from hotel_pipeline.geo.facade_visibility import measure_facade_alignment
from hotel_pipeline.geo.orthofacade import plane_from_edge, rectify


class _Camera:
    width, height = 640, 480

    def __init__(self, position=(0.0, -30.0, 2.5), focal=400.0):
        self.position = np.asarray(position, dtype=float)
        self.f = focal
        self.fwd = np.array([0.0, 1.0, 0.0])
        self.right = np.array([1.0, 0.0, 0.0])
        self.up = np.array([0.0, 0.0, 1.0])

    def project(self, points):
        d = np.asarray(points, dtype=float) - self.position
        z = d @ self.fwd
        if np.all(z <= 0.5):
            return None, z
        safe = np.where(z > 1e-6, z, 1e-6)
        return (
            np.c_[
                self.width / 2 + self.f * (d @ self.right) / safe,
                self.height / 2 - self.f * (d @ self.up) / safe,
            ],
            z,
        )


def _wall_triangles():
    a = np.array([-5.0, 0.0, 0.0])
    b = np.array([5.0, 0.0, 0.0])
    c = np.array([5.0, 0.0, 6.0])
    d = np.array([-5.0, 0.0, 6.0])
    return [np.array([a, b, c]), np.array([a, c, d])], [0, 0]


def test_pose_error_measured_in_pixels_and_meters():
    cam = _Camera()
    plane = plane_from_edge(
        np.array([-5.0, 0.0, 0.0]), np.array([5.0, 0.0, 0.0]), 6.0, "E"
    )
    triangles, fids = _wall_triangles()
    from hotel_pipeline.geo.facade_visibility import ProxyDepth
    proxy = ProxyDepth.render(cam, triangles, fids, 640, 480)
    mask = np.ones((480, 640), dtype=bool)
    err_px, err_m, cols = measure_facade_alignment(cam, plane, proxy, mask)
    assert cols > 0
    assert err_px >= 0
    assert err_m >= 0
    assert err_m == pytest.approx(err_px * 0.12, rel=0.5)


def test_pose_gate_refuses_beyond_threshold():
    from hotel_pipeline.geo.facade_visibility import LidarOcclusion, ProxyDepth, admit, FacadeTexelCandidate

    cam = _Camera()
    plane = plane_from_edge(
        np.array([-5.0, 0.0, 0.0]), np.array([5.0, 0.0, 0.0]), 6.0, "E"
    )
    triangles, fids = _wall_triangles()
    proxy = ProxyDepth.render(cam, triangles, fids, 640, 480)
    candidate = FacadeTexelCandidate(
        source_view="A",
        col=0, row=0,
        u_m=0.0, v_norm=0.5, pixel_xy=(320, 240),
        wall_depth_m=30.0,
        proxy_first_hit_depth_m=50.0,
        proxy_first_hit_face_id=None,
        lidar_depth_m=None,
        semantic_visible=True,
        semantic_source="building",
        pose_error_px=50.0,
        pose_error_m=6.0,
        local_gsd_m=0.05,
        incidence_deg=30.0,
        sharpness=1.0,
        colour_rgb=(100.0, 100.0, 100.0),
    )
    ok, code, _ = admit(candidate)
    assert not ok
    assert code == "pose_error"


def test_orthofacade_statuses_are_exclusive():
    from hotel_pipeline.geo.orthofacade import TexelSupport

    support = TexelSupport()
    assert support.status == "non_observe"
    assert not support.is_observed

    support.contributing = 1
    assert support.status == "vue_unique"
    assert support.is_observed

    support.contributing = 2
    support.rejection_reason = "REJECTED_DISAGREEMENT"
    assert support.status == "REJECTED_DISAGREEMENT"
    assert not support.is_observed

    support.rejection_reason = None
    assert support.status == "accorde"
    assert support.is_observed
