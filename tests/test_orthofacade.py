"""Orthofaçade : rectifier un mur et dire ce qui l'atteste."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hotel_pipeline.geo.orthofacade import (
    DISAGREEMENT_LEVEL,
    MAX_INCIDENCE_DEG,
    MIN_PIXELS_PER_M,
    TEXEL_M,
    TEXEL_M_FACADE,
    TexelSupport,
    plane_from_edge,
    rectify,
)
from hotel_pipeline.geo.facade_visibility import LidarOcclusion, ProxyDepth


class _Camera:
    """Caméra regardant vers +y, à hauteur d'œil."""

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


def _wall(length=10.0, height=6.0, top_z_start=None, top_z_end=None):
    return plane_from_edge(
        np.array([-length / 2, 0.0, 0.0]),
        np.array([length / 2, 0.0, 0.0]),
        height,
        "MUR",
        top_z_start_m=top_z_start,
        top_z_end_m=top_z_end,
    )


def _image(colour=(120, 130, 140)):
    return np.full((480, 640, 3), colour, dtype=np.uint8)


def _full_mask():
    return np.ones((480, 640), dtype=bool)


class TestPlane:
    def test_the_plane_spans_the_edge(self):
        plane = _wall(length=20.0, height=8.0)
        assert plane.length_m == pytest.approx(20.0)
        assert plane.height_m == 8.0

    def test_a_point_lands_where_it_should(self):
        plane = _wall(length=10.0)
        found = plane.point(5.0, 0.5)
        assert found[0] == pytest.approx(0.0)
        assert found[2] == pytest.approx(3.0)

    def test_the_normal_is_horizontal_and_unit(self):
        plane = _wall()
        assert plane.normal[2] == 0.0
        assert float(np.linalg.norm(plane.normal)) == pytest.approx(1.0)

    def test_top_z_interpolates(self):
        plane = _wall(top_z_start=8.0, top_z_end=12.0)
        assert plane.top_z(0.0) == pytest.approx(8.0)
        assert plane.top_z(plane.length_m) == pytest.approx(12.0)
        assert plane.top_z(plane.length_m * 0.5) == pytest.approx(10.0)

    def test_v_norm_one_lands_at_top(self):
        plane = _wall(top_z_start=8.0, top_z_end=12.0)
        p0 = plane.point(0.0, 1.0)
        p1 = plane.point(plane.length_m, 1.0)
        assert p0[2] == pytest.approx(8.0)
        assert p1[2] == pytest.approx(12.0)


class TestRectify:
    def test_a_facing_view_covers_the_wall(self):
        found = rectify(_wall(), [("A", _image(), _Camera(), _full_mask())])
        assert found.observed_fraction > 0.5
        assert found.image is not None

    def test_no_view_yields_no_image(self):
        found = rectify(_wall(), [])
        assert found.image is None
        assert found.observed_fraction == 0.0
        assert found.provenance["views_supplied"] == 0

    def test_a_grazing_view_is_refused(self):
        camera = _Camera(position=(200.0, -2.0, 2.5))
        found = rectify(_wall(), [("A", _image(), camera)])
        assert found.provenance["views_used"] == 0

    def test_a_view_too_far_to_resolve_is_refused(self):
        far = _Camera(position=(0.0, -5000.0, 2.5), focal=400.0)
        found = rectify(_wall(), [("A", _image(), far, _full_mask())])
        assert found.provenance["views_too_far"] >= 1
        assert found.provenance["views_used"] == 0

    def test_the_texel_grid_follows_the_wall(self):
        plane = _wall(length=10.0, height=5.0)
        found = rectify(plane, [])
        assert found.width_px == int(round(10.0 / TEXEL_M_FACADE))
        assert found.height_px == int(round(5.0 / TEXEL_M_FACADE))


class TestSupport:
    def test_an_unseen_texel_says_so(self):
        assert TexelSupport().status == "non_observe"

    def test_a_single_view_is_not_corroborated(self):
        assert TexelSupport(contributing=1).status == "vue_unique"

    def test_agreeing_views_are_credited(self):
        assert TexelSupport(contributing=3, disagreement=5.0).status == "accorde"

    def test_disagreement_is_reported_not_smoothed(self):
        texel = TexelSupport(contributing=3, rejection_reason="REJECTED_DISAGREEMENT")
        assert texel.status == "REJECTED_DISAGREEMENT"
        assert not texel.is_observed

    def test_two_views_of_the_same_wall_agree(self):
        plane = _wall()
        views = [
            ("A", _image((100, 110, 120)), _Camera(position=(-3.0, -30.0, 2.5)), _full_mask()),
            ("B", _image((100, 110, 120)), _Camera(position=(3.0, -30.0, 2.5)), _full_mask()),
        ]
        found = rectify(plane, views)
        assert found.by_status().get("REJECTED_DISAGREEMENT", 0) == 0

    def test_contradicting_views_are_flagged(self):
        plane = _wall()
        views = [
            ("A", _image((0, 0, 0)), _Camera(position=(-3.0, -30.0, 2.5)), _full_mask()),
            ("B", _image((255, 255, 255)), _Camera(position=(3.0, -30.0, 2.5)), _full_mask()),
        ]
        found = rectify(plane, views)
        assert found.by_status().get("REJECTED_DISAGREEMENT", 0) > 0

    def test_disagreeing_views_are_never_fabricated(self):
        plane = _wall()
        views = [
            ("A", _image((0, 0, 0)), _Camera(position=(-3.0, -30.0, 2.5)), _full_mask()),
            ("B", _image((255, 255, 255)), _Camera(position=(3.0, -30.0, 2.5)), _full_mask()),
        ]
        found = rectify(plane, views)
        for texel in found.support:
            if texel.contributing >= 2:
                assert not texel.is_observed
        assert found.observed_fraction == 0.0

    def test_agreeing_views_stay_observed(self):
        plane = _wall()
        views = [
            ("A", _image((100, 110, 120)), _Camera(position=(-3.0, -30.0, 2.5)), _full_mask()),
            ("B", _image((100, 110, 120)), _Camera(position=(3.0, -30.0, 2.5)), _full_mask()),
        ]
        found = rectify(plane, views)
        assert found.observed_fraction > 0.5
        assert all(t.is_observed for t in found.support if t.contributing >= 2)

    def test_a_visibility_mask_excludes_non_building_pixels(self):
        mask = np.zeros((480, 640), dtype=bool)
        found = rectify(_wall(), [("A", _image(), _Camera(), mask)])
        assert found.observed_fraction == 0.0

    def test_alpha_is_zero_on_rejected_texels(self):
        plane = _wall()
        views = [
            ("A", _image((0, 0, 0)), _Camera(position=(-3.0, -30.0, 2.5)), _full_mask()),
            ("B", _image((255, 255, 255)), _Camera(position=(3.0, -30.0, 2.5)), _full_mask()),
        ]
        found = rectify(plane, views)
        assert found.image is not None
        observed = np.asarray([t.is_observed for t in found.support]).reshape(
            found.height_px, found.width_px
        )
        alpha = np.where(observed, 255, 0).astype(np.uint8)
        assert alpha.max() == 0
        assert alpha.min() == 0


class TestReport:
    def test_report_carries_its_caveats(self):
        payload = rectify(_wall(), [("A", _image(), _Camera(), _full_mask())]).as_dict()
        assert payload["caveats"]
        assert any("extrusion" in c for c in payload["caveats"])

    def test_thresholds_are_coherent(self):
        assert 0.0 < TEXEL_M < 1.0
        assert 0.0 < TEXEL_M_FACADE < 1.0
        assert 0.0 < MAX_INCIDENCE_DEG < 90.0
        assert MIN_PIXELS_PER_M > 0.0
        assert DISAGREEMENT_LEVEL > 0.0

    def test_as_dict_texel_m_is_facade(self):
        payload = rectify(_wall(), [("A", _image(), _Camera(), _full_mask())]).as_dict()
        assert payload["texel_m"] == TEXEL_M_FACADE


def _proxy_with_occluder(width=640, height=480, occ_x=320, occ_y=240, occ_radius=20):
    depth = np.full((height, width), np.inf)
    face_map = np.full((height, width), -1, dtype=np.int32)
    for dy in range(-occ_radius, occ_radius + 1):
        for dx in range(-occ_radius, occ_radius + 1):
            if dx * dx + dy * dy <= occ_radius * occ_radius:
                py, px = occ_y + dy, occ_x + dx
                if 0 <= py < height and 0 <= px < width:
                    depth[py, px] = 5.0
                    face_map[py, px] = 99
    return ProxyDepth(width=width, height=height, depth=depth, face_id_map=face_map)


class TestProxyPerView:
    def test_proxy_occludes_pixels_behind_it(self):
        proxy = _proxy_with_occluder()
        mask = np.ones((480, 640), dtype=bool)
        found = rectify(_wall(), [("A", _image(), _Camera(), mask, proxy, None)])
        occ_col = int(320 / 640 * found.width_px)
        occ_row = int(240 / 480 * found.height_px)
        occ_slot = occ_row * found.width_px + occ_col
        assert found.support[occ_slot].contributing == 0
        assert found.support[occ_slot].rejection_reason == "REJECTED_OCCLUDED"

    def test_proxy_per_view_not_global(self):
        proxy_occ = _proxy_with_occluder(occ_x=100)
        mask = np.ones((480, 640), dtype=bool)
        found = rectify(
            _wall(),
            [
                ("A", _image(), _Camera(), mask, proxy_occ, None),
                ("B", _image(), _Camera(position=(3.0, -30.0, 2.5)), mask, None, None),
            ],
        )
        assert found.observed_fraction > 0.0


class TestLidarPerView:
    def test_lidar_occludes_pixels(self):
        proxy = _proxy_with_occluder()
        lidar_depth = np.full((480, 640), np.inf)
        lidar_valid = np.zeros((480, 640), dtype=bool)
        lidar_depth[240, 320] = 5.0
        lidar_valid[240, 320] = True
        lidar = LidarOcclusion(depth=lidar_depth, valid=lidar_valid)
        mask = np.ones((480, 640), dtype=bool)
        found = rectify(_wall(), [("A", _image(), _Camera(), mask, proxy, lidar)])
        occ_slot = int(240 / 480 * found.height_px) * found.width_px + int(320 / 640 * found.width_px)
        assert found.support[occ_slot].rejection_reason == "REJECTED_OCCLUDED"


class TestMadBeforeStd:
    def test_outlier_rejected_by_mad_keeps_inliers(self):
        plane = _wall()
        views = [
            ("A", _image((100, 110, 120)), _Camera(position=(-3.0, -30.0, 2.5)), _full_mask()),
            ("B", _image((102, 108, 118)), _Camera(position=(3.0, -30.0, 2.5)), _full_mask()),
            ("C", _image((0, 200, 0)), _Camera(position=(0.0, -30.0, 2.5)), _full_mask()),
        ]
        found = rectify(plane, views)
        assert found.observed_fraction > 0.3

    def test_consensus_colour_written_to_canvas(self):
        plane = _wall()
        views = [
            ("A", _image((100, 110, 120)), _Camera(position=(-3.0, -30.0, 2.5)), _full_mask()),
            ("B", _image((102, 108, 118)), _Camera(position=(3.0, -30.0, 2.5)), _full_mask()),
        ]
        found = rectify(plane, views)
        assert found.image is not None
        observed = [t for t in found.support if t.is_observed]
        assert len(observed) > 0


class TestRejectionPerCandidate:
    def test_late_good_view_not_blocked_by_early_bad(self):
        plane = _wall()
        mask_a = np.zeros((480, 640), dtype=bool)
        mask_b = np.ones((480, 640), dtype=bool)
        views = [
            ("A", _image((100, 110, 120)), _Camera(position=(-3.0, -30.0, 2.5)), mask_a),
            ("B", _image((100, 110, 120)), _Camera(position=(3.0, -30.0, 2.5)), mask_b),
        ]
        found = rectify(plane, views)
        assert found.observed_fraction > 0.3
