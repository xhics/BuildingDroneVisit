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
    TexelSupport,
    plane_from_edge,
    rectify,
)


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


def _wall(length=10.0, height=6.0):
    return plane_from_edge(
        np.array([-length / 2, 0.0, 0.0]),
        np.array([length / 2, 0.0, 0.0]),
        height,
        "MUR",
    )


def _image(colour=(120, 130, 140)):
    return np.full((480, 640, 3), colour, dtype=np.uint8)


class TestPlane:
    def test_the_plane_spans_the_edge(self):
        plane = _wall(length=20.0, height=8.0)
        assert plane.length_m == pytest.approx(20.0)
        assert plane.height_m == 8.0

    def test_a_point_lands_where_it_should(self):
        plane = _wall(length=10.0)
        found = plane.point(5.0, 3.0)
        assert found[0] == pytest.approx(0.0)
        assert found[2] == pytest.approx(3.0)

    def test_the_normal_is_horizontal_and_unit(self):
        plane = _wall()
        assert plane.normal[2] == 0.0
        assert float(np.linalg.norm(plane.normal)) == pytest.approx(1.0)


class TestRectify:
    def test_a_facing_view_covers_the_wall(self):
        found = rectify(_wall(), [("A", _image(), _Camera())])
        assert found.observed_fraction > 0.5
        assert found.image is not None

    def test_no_view_yields_no_image(self):
        found = rectify(_wall(), [])
        assert found.image is None
        assert found.observed_fraction == 0.0
        assert found.provenance["views"] == 0

    def test_a_grazing_view_is_refused(self):
        """Quelques pixels y décrivent plusieurs mètres."""
        # Caméra presque dans le plan du mur.
        camera = _Camera(position=(200.0, -2.0, 2.5))
        found = rectify(_wall(), [("A", _image(), camera)])
        assert found.provenance["views_used"] == 0

    def test_a_view_too_far_to_resolve_is_refused(self):
        """Le défaut mesuré : un mur de 35 px agrandi trente fois."""
        far = _Camera(position=(0.0, -5000.0, 2.5), focal=400.0)
        found = rectify(_wall(), [("A", _image(), far)])
        assert found.provenance["views_too_far"] >= 1
        assert found.provenance["views_used"] == 0

    def test_the_texel_grid_follows_the_wall(self):
        plane = _wall(length=10.0, height=5.0)
        found = rectify(plane, [])
        assert found.width_px == int(round(10.0 / TEXEL_M))
        assert found.height_px == int(round(5.0 / TEXEL_M))


class TestSupport:
    def test_an_unseen_texel_says_so(self):
        assert TexelSupport().status == "non_observe"

    def test_a_single_view_is_not_corroborated(self):
        assert TexelSupport(contributing=1).status == "vue_unique"

    def test_agreeing_views_are_credited(self):
        assert TexelSupport(contributing=3, disagreement=5.0).status == "accorde"

    def test_disagreement_is_reported_not_smoothed(self):
        """Un désaccord signale une pose fausse ou un décrochement."""
        texel = TexelSupport(contributing=3, disagreement=DISAGREEMENT_LEVEL + 10)
        assert texel.status == "desaccord"

    def test_two_views_of_the_same_wall_agree(self):
        plane = _wall()
        views = [
            ("A", _image((100, 110, 120)), _Camera(position=(-3.0, -30.0, 2.5))),
            ("B", _image((100, 110, 120)), _Camera(position=(3.0, -30.0, 2.5))),
        ]
        found = rectify(plane, views)
        assert found.by_status().get("desaccord", 0) == 0

    def test_contradicting_views_are_flagged(self):
        plane = _wall()
        views = [
            ("A", _image((0, 0, 0)), _Camera(position=(-3.0, -30.0, 2.5))),
            ("B", _image((255, 255, 255)), _Camera(position=(3.0, -30.0, 2.5))),
        ]
        found = rectify(plane, views)
        assert found.by_status().get("desaccord", 0) > 0

    def test_disagreeing_views_are_never_fabricated(self):
        """Une vue plus directe mais traversée par un objet ne doit pas l'emporter.

        Deux vues en désaccord franc (ici noir vs blanc) signalent une pose
        fausse ou un objet mobile : aucune tuile n'est écrite, le proxy reste.
        """
        plane = _wall()
        views = [
            ("A", _image((0, 0, 0)), _Camera(position=(-3.0, -30.0, 2.5))),
            ("B", _image((255, 255, 255)), _Camera(position=(3.0, -30.0, 2.5))),
        ]
        found = rectify(plane, views)
        for texel in found.support:
            if texel.contributing >= 2:
                assert not texel.is_observed
        assert found.observed_fraction == 0.0

    def test_agreeing_views_stay_observed(self):
        plane = _wall()
        views = [
            ("A", _image((100, 110, 120)), _Camera(position=(-3.0, -30.0, 2.5))),
            ("B", _image((100, 110, 120)), _Camera(position=(3.0, -30.0, 2.5))),
        ]
        found = rectify(plane, views)
        assert found.observed_fraction > 0.5
        assert all(t.is_observed for t in found.support if t.contributing >= 2)

    def test_a_visibility_mask_excludes_non_building_pixels(self):
        mask = np.zeros((480, 640), dtype=bool)
        found = rectify(_wall(), [("A", _image(), _Camera(), mask)])

        assert found.observed_fraction == 0.0


class TestReport:
    def test_report_carries_its_caveats(self):
        payload = rectify(_wall(), [("A", _image(), _Camera())]).as_dict()
        assert payload["caveats"]
        # La réserve principale : le plan est un proxy, pas un mur relevé.
        assert any("extrusion" in c for c in payload["caveats"])

    def test_thresholds_are_coherent(self):
        assert 0.0 < TEXEL_M < 1.0
        assert 0.0 < MAX_INCIDENCE_DEG < 90.0
        assert MIN_PIXELS_PER_M > 0.0
        assert DISAGREEMENT_LEVEL > 0.0
