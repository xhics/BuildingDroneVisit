"""Préflight LAZ — indicateurs mesurés, pas annoncés (Lot 1B §9).

L'indicateur décisif est une **proportion de cellules occupées**, non un nombre
de points : un toit en amas discontinus produit un compte flatteur et une
enveloppe fausse.

Le dénominateur de cette proportion est le piège. Mesuré contre la boîte
englobante, il a fait passer une toiture à 25 points par mètre carré pour
« fragmentaire à 34 % » — la fraction exacte que le polygone occupe de sa boîte.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

from hotel_pipeline.geo.preflight import (
    BUILDING,
    GROUND,
    UNCLASSIFIED,
    ClassStats,
    PreflightReport,
    _add_warnings,
    _cell_coverage,
    _stats,
)


def filled(polygon: Polygon, step: float = 0.3):
    """Nuage couvrant régulièrement un polygone."""
    minx, miny, maxx, maxy = polygon.bounds
    xs = np.arange(minx, maxx, step)
    ys = np.arange(miny, maxy, step)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    from shapely.vectorized import contains

    inside = contains(polygon, gx.ravel(), gy.ravel())
    return gx.ravel()[inside], gy.ravel()[inside]


#: Rectangle oblique : sa boîte englobante vaut environ le double de son aire.
OBLIQUE = Polygon([(0, 0), (40, 20), (36, 28), (-4, 8)])
SQUARE = Polygon([(0, 0), (30, 0), (30, 30), (0, 30)])


class TestCoverageDenominator:
    def test_a_fully_covered_oblique_building_reads_near_complete(self):
        """Le défaut mesuré : 34 % annoncés sur un toit intégralement couvert."""
        x, y = filled(OBLIQUE)
        assert _cell_coverage(x, y, OBLIQUE, 1.0) > 0.95

    def test_an_oblique_polygon_occupies_far_less_than_its_box(self):
        """Ce que le dénominateur fautif mesurait en réalité."""
        minx, miny, maxx, maxy = OBLIQUE.bounds
        box_area = (maxx - minx) * (maxy - miny)
        assert OBLIQUE.area / box_area < 0.55

    def test_a_square_building_is_unaffected_by_the_bug(self):
        """Le défaut restait invisible sur une empreinte alignée."""
        x, y = filled(SQUARE)
        assert _cell_coverage(x, y, SQUARE, 1.0) > 0.95

    def test_half_covered_roof_reads_about_half(self):
        x, y = filled(SQUARE)
        keep = x < 15
        assert 0.4 < _cell_coverage(x[keep], y[keep], SQUARE, 1.0) < 0.6

    def test_no_points_reads_zero(self):
        empty = np.array([])
        assert _cell_coverage(empty, empty, SQUARE, 1.0) == 0.0

    def test_scattered_points_expose_fragmentation(self):
        """Beaucoup de points sur peu de cellules : le compte tromperait."""
        x = np.repeat(np.array([2.5, 3.5, 4.5]), 500)
        y = np.repeat(np.array([2.5, 3.5, 4.5]), 500)
        assert _cell_coverage(x, y, SQUARE, 1.0) < 0.01


class TestClassStatistics:
    def test_counts_and_elevations_per_class(self):
        codes = np.array([BUILDING] * 4 + [GROUND] * 2)
        z = np.array([37.0, 38.0, 39.0, 40.0, 27.0, 28.0])

        roof = _stats(codes, z, BUILDING)
        assert roof.count == 4
        assert roof.z_median == pytest.approx(38.5)
        assert _stats(codes, z, GROUND).count == 2

    def test_absent_class_yields_no_elevation(self):
        stats = _stats(np.array([GROUND]), np.array([27.0]), BUILDING)
        assert stats.count == 0
        assert stats.z_median is None


class TestWarnings:
    def _report(self, **classes) -> PreflightReport:
        report = PreflightReport(roof_cell_coverage=classes.pop("coverage", 0.98))
        for code, stats in classes.items():
            report.footprint_classes[int(code[1:])] = stats
        return report

    def test_missing_ground_is_flagged(self):
        report = self._report(
            c2=ClassStats(GROUND, "sol", 0), c6=ClassStats(BUILDING, "bâtiment", 5000)
        )
        _add_warnings(report)
        assert any("classe 2" in w for w in report.warnings)

    def test_missing_building_blocks_the_roofline(self):
        report = self._report(
            c2=ClassStats(GROUND, "sol", 900), c6=ClassStats(BUILDING, "bâtiment", 0)
        )
        _add_warnings(report)
        assert any("ROOFLINE_MAIN non dérivable" in w for w in report.warnings)

    def test_fragmentary_roof_is_flagged_as_sketched_not_measured(self):
        report = self._report(
            coverage=0.3,
            c2=ClassStats(GROUND, "sol", 900),
            c6=ClassStats(BUILDING, "bâtiment", 5000),
        )
        _add_warnings(report)
        assert any("esquissée, non mesurée" in w for w in report.warnings)

    def test_dense_and_continuous_roof_raises_nothing(self):
        report = self._report(
            coverage=0.98,
            c2=ClassStats(GROUND, "sol", 900),
            c6=ClassStats(BUILDING, "bâtiment", 46829),
        )
        _add_warnings(report)
        assert report.warnings == []

    def test_unclassified_reaching_roof_height_is_flagged(self):
        """La classe 1 peut porter des superstructures : ne pas l'exclure à l'aveugle."""
        report = self._report(
            c2=ClassStats(GROUND, "sol", 900),
            c6=ClassStats(BUILDING, "bâtiment", 5000, z_median=37.8),
            c1=ClassStats(UNCLASSIFIED, "non classé", 400, z_median=38.5),
        )
        _add_warnings(report)
        assert any("superstructures" in w for w in report.warnings)

    def test_low_unclassified_is_not_flagged(self):
        """Sur ce site, la classe 1 médiane à 29,3 m contre 37,8 m au toit."""
        report = self._report(
            c2=ClassStats(GROUND, "sol", 900),
            c6=ClassStats(BUILDING, "bâtiment", 5000, z_median=37.8),
            c1=ClassStats(UNCLASSIFIED, "non classé", 1626, z_median=29.3),
        )
        _add_warnings(report)
        assert report.warnings == []
