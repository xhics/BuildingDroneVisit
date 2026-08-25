"""Tests de la preuve de visibilité pixel par pixel pour les texels de façade."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hotel_pipeline.geo.facade_visibility import (
    LidarOcclusion,
    ProxyDepth,
    TexelStatus,
    admit,
    measure_facade_alignment,
)


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


class TestProxyDepth:
    def test_no_hit_outside_bounds(self):
        triangles, fids = _wall_triangles()
        cam = _Camera()
        depth = ProxyDepth.render(cam, triangles, fids, 640, 480)
        d, fid = depth.hit(0, 0)
        assert d == float("inf")
        assert fid == -1

    def test_hit_returns_depth_and_face(self):
        triangles, fids = _wall_triangles()
        cam = _Camera()
        depth = ProxyDepth.render(cam, triangles, fids, 640, 480)
        d, fid = depth.hit(320, 240)
        assert math.isfinite(d)
        assert d > 0
        assert fid == 0


class TestLidarOcclusion:
    def test_no_window_returns_none(self):
        cam = _Camera()
        result = LidarOcclusion.from_window(None, cam, np.zeros(3), 400.0, 640, 480)
        assert result is None

    def test_occluding_points_splatted(self):
        class _Window:
            x = np.array([0.0, 1.0])
            y = np.array([0.0, 1.0])
            z = np.array([2.0, 2.0])
            classification = np.array([3, 4])

            def __len__(self):
                return len(self.x)

        cam = _Camera()
        result = LidarOcclusion.from_window(
            _Window(), cam, np.zeros(3), 400.0, 640, 480
        )
        assert result is not None
        assert result.valid.any()


class TestAdmit:
    def test_semantic_absent_rejects(self):
        candidate = type("C", (), {})()
        candidate.semantic_visible = False
        candidate.semantic_source = None
        candidate.proxy_first_hit_face_id = None
        candidate.wall_depth_m = 10.0
        candidate.lidar_depth_m = None
        candidate.pose_error_m = None
        candidate.local_gsd_m = None
        candidate.incidence_deg = None
        ok, code, _ = admit(candidate)
        assert not ok
        assert code == "semantic_absent"

    def test_occluded_by_proxy_rejects(self):
        candidate = type("C", (), {})()
        candidate.semantic_visible = True
        candidate.semantic_source = "building"
        candidate.proxy_first_hit_face_id = 1
        candidate.wall_depth_m = 10.0
        candidate.proxy_first_hit_depth_m = 5.0
        candidate.lidar_depth_m = None
        candidate.pose_error_m = None
        candidate.local_gsd_m = None
        candidate.incidence_deg = None
        ok, code, _ = admit(candidate)
        assert not ok
        assert code == "occluded_by_proxy"

    def test_good_candidate_admitted(self):
        candidate = type("C", (), {})()
        candidate.semantic_visible = True
        candidate.semantic_source = "building"
        candidate.proxy_first_hit_face_id = None
        candidate.wall_depth_m = 10.0
        candidate.proxy_first_hit_depth_m = 20.0
        candidate.lidar_depth_m = None
        candidate.pose_error_m = 0.1
        candidate.local_gsd_m = 0.05
        candidate.incidence_deg = 30.0
        ok, code, _ = admit(candidate)
        assert ok
        assert code is None


class TestMeasureFacadeAlignment:
    def test_aligned_wall_has_small_error(self):
        from hotel_pipeline.geo.orthofacade import plane_from_edge

        cam = _Camera()
        plane = plane_from_edge(
            np.array([-5.0, 0.0, 0.0]), np.array([5.0, 0.0, 0.0]), 6.0, "E"
        )
        triangles, fids = _wall_triangles()
        proxy = ProxyDepth.render(cam, triangles, fids, 640, 480)
        mask = np.zeros((480, 640), dtype=bool)
        mask[200:400, 200:440] = True
        err_px, err_m, cols = measure_facade_alignment(cam, plane, proxy_depth=proxy, building_mask=mask)
        assert cols > 0
        assert err_px < 50
