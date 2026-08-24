"""Sélection des volumes par capacité réelle d'occultation."""

from __future__ import annotations

import numpy as np
import pytest

from hotel_pipeline.conditioning.occlusion import (
    OcclusionReport,
    _angular_span,
    select,
)


class _Prism:
    def __init__(self, feature_id, footprint, target=False):
        self.feature_id = feature_id
        self.footprint = np.asarray(footprint, dtype=np.float64)
        self.is_target = target


class _Scene:
    def __init__(self, prisms, centre=(0.0, 0.0), radius=20.0):
        self.prisms = prisms
        self.centre = centre
        self._radius = radius

    def radius_m(self):
        return self._radius

    @property
    def target(self):
        return next((p for p in self.prisms if p.is_target), None)


def _box(cx, cy, size=10.0):
    h = size / 2
    return [[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h]]


TARGET = _Prism("TARGET", _box(0.0, 0.0, 20.0), target=True)


class TestAngularSpan:
    def test_span_is_centred_on_the_footprint(self):
        middle, half = _angular_span(np.array(_box(50.0, 0.0)), (0.0, 0.0))
        assert middle == pytest.approx(0.0, abs=1.0)
        assert half > 0

    def test_span_survives_the_north_wrap(self):
        """Une emprise à cheval sur 0° ne doit pas voir ses angles s'annuler."""
        middle, half = _angular_span(np.array(_box(-50.0, 0.0)), (0.0, 0.0))
        assert middle == pytest.approx(180.0, abs=1.0)
        assert half < 45.0


class TestSelection:
    def test_target_is_always_kept(self):
        scene = _Scene([TARGET])
        report = select(scene)
        assert report.kept == ["TARGET"]

    def test_distant_volume_is_dropped(self):
        far = _Prism("FAR", _box(600.0, 0.0))
        scene = _Scene([TARGET, far])
        select(scene)
        assert [p.feature_id for p in scene.prisms] == ["TARGET"]

    def test_dropped_volume_says_why(self):
        scene = _Scene([TARGET, _Prism("FAR", _box(600.0, 0.0))])
        report = select(scene)
        verdict = next(v for v in report.verdicts if v.feature_id == "FAR")
        assert "au-delà de l'orbite" in verdict.reason

    def test_near_volume_in_axis_is_kept(self):
        near = _Prism("NEAR", _box(28.0, 0.0))
        scene = _Scene([TARGET, near])
        select(scene)
        assert "NEAR" in [p.feature_id for p in scene.prisms]

    def test_scene_without_target_is_left_alone(self):
        scene = _Scene([_Prism("A", _box(10.0, 0.0))])
        report = select(scene)
        assert report.verdicts == []
        assert len(scene.prisms) == 1

    def test_selection_mutates_the_scene(self):
        scene = _Scene([TARGET, _Prism("FAR", _box(900.0, 0.0))])
        before = len(scene.prisms)
        select(scene)
        assert len(scene.prisms) < before


class TestMargins:
    def test_reach_is_what_decides(self):
        """La caméra couvre tous les azimuts : seule la distance tranche."""
        near = _Prism("NEAR", _box(28.0, 0.0))
        far = _Prism("FAR", _box(300.0, 0.0))
        report = select(_Scene([TARGET, near, far]))
        assert "NEAR" in report.kept
        assert "FAR" in report.dropped

    def test_a_volume_behind_the_target_still_counts(self):
        """Il peut dépasser derrière elle et compter dans la silhouette."""
        behind = _Prism("BEHIND", _box(-30.0, 0.0))
        assert "BEHIND" in select(_Scene([TARGET, behind])).kept

    def test_depth_margin_extends_the_reach(self):
        prisms = [TARGET, _Prism("EDGE", _box(60.0, 0.0))]
        short = select(_Scene(list(prisms)), depth_margin_m=0.0)
        long = select(_Scene(list(prisms)), depth_margin_m=200.0)
        assert len(long.kept) >= len(short.kept)


class TestReport:
    def test_report_serialises(self):
        scene = _Scene([TARGET, _Prism("FAR", _box(600.0, 0.0))])
        payload = select(scene).as_dict()
        assert payload["kept_count"] == 1
        assert payload["dropped_count"] == 1
        assert payload["caveats"]

    def test_kept_and_dropped_partition_the_scene(self):
        prisms = [TARGET] + [_Prism(f"N{i}", _box(100.0 * i, 0.0)) for i in range(1, 5)]
        report = select(_Scene(prisms))
        assert set(report.kept) & set(report.dropped) == set()
        assert len(report.kept) + len(report.dropped) == len(report.verdicts)
