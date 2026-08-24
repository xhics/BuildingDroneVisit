"""Admission au dense : refuser ce dont on sait qu'il ne donnera rien."""

from __future__ import annotations

import pytest

from hotel_pipeline.dense_admission import (
    MAX_REPROJECTION_PX,
    MIN_COMPONENT_IMAGES,
    MIN_CONNECTED_FRACTION,
    MIN_TRIANGULABLE_FRACTION,
    evaluate,
)


def _check(verdict, name):
    return next(c for c in verdict.checks if c.name == name)


class TestConnectivity:
    def test_a_connected_solve_passes(self):
        verdict = evaluate(40, 38, triangulable_fraction=0.7)
        assert _check(verdict, "connexite").passed
        assert verdict.admitted

    def test_a_fragmented_solve_is_refused(self):
        """Le cas mesuré : cinq images connexes sur neuf."""
        verdict = evaluate(9, 5, triangulable_fraction=0.7)
        assert not _check(verdict, "connexite").passed
        assert not verdict.admitted

    def test_a_tiny_component_is_refused_even_if_it_is_everything(self):
        """Deux images entièrement connexes ne contraignent pas une géométrie."""
        verdict = evaluate(2, 2, triangulable_fraction=0.9)
        assert not _check(verdict, "connexite").passed

    def test_the_refusal_says_what_would_lift_it(self):
        verdict = evaluate(9, 5, triangulable_fraction=0.7)
        assert "liaison" in _check(verdict, "connexite").remedy

    def test_a_passing_check_carries_no_remedy(self):
        verdict = evaluate(40, 38, triangulable_fraction=0.7)
        assert _check(verdict, "connexite").remedy == ""


class TestCoverage:
    def test_poor_coverage_is_refused(self):
        """Le cas mesuré : 14 % des cellules triangulables."""
        verdict = evaluate(40, 38, triangulable_fraction=0.14)
        assert not _check(verdict, "couverture").passed
        assert not verdict.admitted

    def test_sufficient_coverage_passes(self):
        verdict = evaluate(40, 38, triangulable_fraction=0.72)
        assert _check(verdict, "couverture").passed

    def test_a_missing_map_does_not_block(self):
        """L'ignorance n'est pas une preuve : le contrôle est non concluant."""
        verdict = evaluate(40, 38, triangulable_fraction=None)
        check = _check(verdict, "couverture")
        assert check.passed
        assert "non concluant" in check.reason
        assert "observation-map" in check.remedy


class TestReprojection:
    def test_precise_poses_pass(self):
        verdict = evaluate(40, 38, 0.7, reprojection_px=1.1)
        assert _check(verdict, "reprojection").passed

    def test_imprecise_poses_are_refused(self):
        verdict = evaluate(40, 38, 0.7, reprojection_px=9.0)
        check = _check(verdict, "reprojection")
        assert not check.passed
        assert "intrinsèques" in check.remedy

    def test_an_unreported_error_does_not_block(self):
        verdict = evaluate(40, 38, 0.7, reprojection_px=None)
        assert _check(verdict, "reprojection").passed


class TestVerdict:
    def test_every_check_must_pass_to_admit(self):
        verdict = evaluate(40, 38, triangulable_fraction=0.14)
        assert not verdict.admitted
        assert [c.name for c in verdict.blocking] == ["couverture"]

    def test_several_failures_are_all_reported(self):
        verdict = evaluate(9, 5, triangulable_fraction=0.14)
        assert len(verdict.blocking) == 2

    def test_report_serialises_with_remedies(self):
        payload = evaluate(9, 5, triangulable_fraction=0.14).as_dict()
        assert payload["admitted"] is False
        assert payload["blocking"]
        assert payload["remedies"]
        assert payload["caveats"]

    def test_admitting_is_not_guaranteeing(self):
        payload = evaluate(40, 38, triangulable_fraction=0.7).as_dict()
        assert any("n'est pas garantir" in c for c in payload["caveats"])

    def test_no_registered_images_is_refused(self):
        verdict = evaluate(0, 0, triangulable_fraction=0.9)
        assert not verdict.admitted


class TestThresholds:
    def test_thresholds_are_coherent(self):
        assert 0.0 < MIN_CONNECTED_FRACTION <= 1.0
        assert MIN_COMPONENT_IMAGES >= 3
        assert 0.0 < MIN_TRIANGULABLE_FRACTION <= 1.0
        assert MAX_REPROJECTION_PX > 0.0
