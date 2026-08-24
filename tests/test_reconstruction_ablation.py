"""Comparer des variantes de masquage plutôt que d'en supposer une meilleure."""

from __future__ import annotations

import numpy as np
import pytest

from hotel_pipeline.reconstruction_ablation import (
    MAX_MASKED_FRACTION,
    MIN_INLIERS,
    AblationReport,
    VariantResult,
    apply_mask,
    compare,
    largest_component,
)


class TestLargestComponent:
    def test_a_chain_is_one_component(self):
        assert largest_component(["a", "b", "c"], [("a", "b"), ("b", "c")]) == 3

    def test_two_islands_are_counted_separately(self):
        found = largest_component(
            ["a", "b", "c", "d"], [("a", "b"), ("c", "d")]
        )
        assert found == 2

    def test_isolated_nodes_count_as_one(self):
        assert largest_component(["a", "b", "c"], []) == 1

    def test_no_nodes_yields_zero(self):
        assert largest_component([], []) == 0

    def test_unknown_edges_are_ignored(self):
        """Une arête vers un nœud absent ne doit pas faire tomber le calcul."""
        assert largest_component(["a", "b"], [("a", "z"), ("a", "b")]) == 2


class TestApplyMask:
    def test_no_mask_leaves_the_image_untouched(self):
        image = np.full((10, 10, 3), 200, dtype=np.uint8)
        masked, share = apply_mask(image, None)
        assert share == 0.0
        assert np.array_equal(masked, image)

    def test_masked_pixels_are_blanked(self):
        image = np.full((10, 10, 3), 200, dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[:5] = 255
        masked, share = apply_mask(image, mask)
        assert share == pytest.approx(0.5)
        assert masked[:5].sum() == 0
        assert masked[5:].sum() > 0

    def test_dilation_covers_more(self):
        image = np.full((40, 40, 3), 200, dtype=np.uint8)
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[18:22, 18:22] = 255
        _tight, small = apply_mask(image, mask, dilate_px=0)
        _loose, large = apply_mask(image, mask, dilate_px=5)
        assert large > small

    def test_the_original_image_is_not_modified(self):
        image = np.full((10, 10, 3), 200, dtype=np.uint8)
        mask = np.ones((10, 10), dtype=np.uint8) * 255
        apply_mask(image, mask)
        assert image.sum() > 0


def _matcher(pairs_by_variant):
    def build(produce, result):
        result.pairs_attempted = len(pairs_by_variant[produce])
        return pairs_by_variant[produce]

    return build


class TestCompare:
    def test_the_variant_with_the_largest_component_wins(self):
        def poor(_image):
            return None

        def rich(_image):
            return None

        pairs = {
            poor: [("a", "b", 40)],
            rich: [("a", "b", 30), ("b", "c", 30)],
        }
        report = compare({"poor": poor, "rich": rich}, _matcher(pairs), ["a", "b", "c"])
        assert report.best().name == "rich"

    def test_inliers_break_a_tie_on_connectivity(self):
        def thin(_image):
            return None

        def thick(_image):
            return None

        pairs = {thin: [("a", "b", 20)], thick: [("a", "b", 90)]}
        report = compare({"thin": thin, "thick": thick}, _matcher(pairs), ["a", "b"])
        assert report.best().name == "thick"

    def test_weak_pairs_are_not_counted_as_valid(self):
        def weak(_image):
            return None

        pairs = {weak: [("a", "b", MIN_INLIERS - 1)]}
        report = compare({"weak": weak}, _matcher(pairs), ["a", "b"])
        assert report.variants[0].pairs_valid == 0


class TestVerdict:
    def _report(self, baseline_component, other_component):
        report = AblationReport()
        report.variants = [
            VariantResult("sans_masque", 10, 10, 500, baseline_component, 10),
            VariantResult("masque", 10, 10, 400, other_component, 10),
        ]
        return report

    def test_a_masking_that_gains_nothing_is_said_so(self):
        """Le résultat mesuré sur ce pilote : le masque coûte sans gagner."""
        payload = self._report(8, 6).as_dict()
        assert payload["chosen"] == "sans_masque"
        assert "ne compense pas" in payload["verdict"]

    def test_a_masking_that_connects_more_is_credited(self):
        payload = self._report(5, 9).as_dict()
        assert payload["chosen"] == "masque"
        assert "élargit" in payload["verdict"]

    def test_report_carries_its_caveats(self):
        payload = AblationReport().as_dict()
        assert payload["caveats"]
        assert any("trop tard" in c for c in payload["caveats"])

    def test_an_empty_report_has_no_winner(self):
        assert AblationReport().best() is None


class TestThresholds:
    def test_masking_everything_is_refused(self):
        assert 0.0 < MAX_MASKED_FRACTION < 1.0

    def test_inlier_floor_is_the_epipolar_minimum(self):
        assert MIN_INLIERS >= 8
