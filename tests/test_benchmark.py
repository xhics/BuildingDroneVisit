"""Comparer des sites : ce que le corpus autorise, et ce qu'il interdit."""

from __future__ import annotations

import pytest

from hotel_pipeline.benchmark import (
    MIN_SITES_FOR_CALIBRATION,
    MIN_SITES_FOR_SPREAD,
    Benchmark,
    SiteRecord,
)


def _site(name, tri=0.5, conf=0.1, comp=0.9, total=100):
    return SiteRecord(
        hotel_id=name,
        assets_total=total,
        views_confirming=int(conf * total),
        cells_total=100,
        cells_triangulable=int(tri * 100),
        registered_images=100,
        largest_component=int(comp * 100),
    )


class TestFractions:
    def test_fractions_are_derived_not_stored(self):
        site = _site("a", tri=0.25, conf=0.1, comp=0.8)
        assert site.triangulable_fraction == pytest.approx(0.25)
        assert site.confirming_fraction == pytest.approx(0.1)
        assert site.connected_fraction == pytest.approx(0.8)

    def test_an_empty_site_does_not_divide_by_zero(self):
        site = SiteRecord(hotel_id="vide")
        assert site.triangulable_fraction == 0.0
        assert site.connected_fraction == 0.0


class TestSpread:
    def test_one_site_yields_no_spread(self):
        """Le cas réel : un seul pilote ne calibre rien."""
        found = Benchmark(sites=[_site("a")]).spread("triangulable_fraction")
        assert not found["computable"]
        assert "au moins" in found["reason"]

    def test_enough_sites_yield_a_spread(self):
        sites = [_site(f"s{i}", tri=0.2 + 0.1 * i) for i in range(MIN_SITES_FOR_SPREAD)]
        found = Benchmark(sites=sites).spread("triangulable_fraction")
        assert found["computable"]
        assert found["min"] < found["max"]
        assert "stdev" in found

    def test_a_missing_measure_is_not_a_zero(self):
        """Un champ absent est une mesure qui n'a pas eu lieu."""
        sites = [_site(f"s{i}") for i in range(MIN_SITES_FOR_SPREAD)]
        for site in sites:
            site.facade_coverage = None
        assert not Benchmark(sites=sites).spread("facade_coverage")["computable"]


class TestCalibrationStatus:
    def test_a_single_site_calibrates_nothing(self):
        status = Benchmark(sites=[_site("a")]).calibration_status()
        assert "aucun seuil n'est calibré" in status

    def test_a_handful_compares_without_dispersing(self):
        status = Benchmark(sites=[_site("a"), _site("b")]).calibration_status()
        assert "dispersion non" in status

    def test_enough_sites_make_calibration_defensible(self):
        sites = [_site(f"s{i}") for i in range(MIN_SITES_FOR_CALIBRATION)]
        assert "défendable" in Benchmark(sites=sites).calibration_status()

    def test_an_empty_benchmark_says_so(self):
        assert "un seul site" in Benchmark().calibration_status()


class TestOutliers:
    def test_a_deviating_site_is_named(self):
        sites = [_site(f"s{i}", tri=0.5) for i in range(5)]
        sites.append(_site("aberrant", tri=0.02))
        found = Benchmark(sites=sites).outliers("triangulable_fraction")
        assert "aberrant" in found

    def test_uniform_sites_have_no_outlier(self):
        sites = [_site(f"s{i}", tri=0.5) for i in range(4)]
        assert Benchmark(sites=sites).outliers("triangulable_fraction") == []

    def test_too_few_sites_yield_no_outlier(self):
        assert Benchmark(sites=[_site("a")]).outliers("triangulable_fraction") == []


class TestReport:
    def test_report_carries_its_caveats(self):
        payload = Benchmark(sites=[_site("a")]).as_dict()
        assert payload["site_count"] == 1
        assert payload["caveats"]
        # La réserve qui compte : un chiffre ne se compare pas seul.
        assert any("mitoyen" in c for c in payload["caveats"])

    def test_characteristics_travel_with_the_measures(self):
        site = _site("a")
        site.attached = True
        site.height_m = 38.0
        payload = Benchmark(sites=[site]).as_dict()
        assert payload["sites"][0]["attached"] is True
        assert payload["sites"][0]["height_m"] == 38.0
