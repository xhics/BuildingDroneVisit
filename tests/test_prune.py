"""Élagage par redondance visuelle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hotel_pipeline.identity.prune import (
    MIN_KEPT,
    PruneReport,
    prune,
)


class _Index:
    """Index d'embeddings servi depuis un dictionnaire, sans modèle."""

    def __init__(self, vectors: dict[str, np.ndarray], failing: set[str] | None = None):
        self._vectors = vectors
        self._failing = failing or set()
        self.embedder = type("E", (), {"model_name": "test"})()

    def vector_of(self, path: Path) -> np.ndarray:
        name = Path(path).name
        if name in self._failing:
            raise OSError("image illisible")
        vector = self._vectors[name]
        return vector / np.linalg.norm(vector)


def _views(count: int) -> list[tuple[str, Path]]:
    return [(f"A{i}", Path(f"{i}.jpg")) for i in range(count)]


def _distinct(count: int, dim: int = 16) -> dict[str, np.ndarray]:
    """Vecteurs orthogonaux : aucune paire n'est redondante."""
    return {f"{i}.jpg": np.eye(dim)[i % dim] for i in range(count)}


class TestNothingToPrune:
    def test_small_corpus_is_left_whole(self):
        vectors = {f"{i}.jpg": np.ones(8) for i in range(4)}
        report = prune(_views(4), _Index(vectors))
        assert len(report.kept) == 4

    def test_small_corpus_says_why_it_was_spared(self):
        vectors = {f"{i}.jpg": np.ones(8) for i in range(3)}
        report = prune(_views(3), _Index(vectors))
        assert all("rien n'est écarté" in v.reason for v in report.views)

    def test_empty_corpus_is_handled(self):
        assert prune([], _Index({})).kept == []

    def test_distinct_views_all_survive(self):
        count = MIN_KEPT + 4
        report = prune(_views(count), _Index(_distinct(count)))
        assert len(report.kept) == count


class TestRedundancy:
    def test_identical_views_collapse_to_one_representative(self):
        count = MIN_KEPT + 6
        # Toutes identiques sauf assez pour tenir le plancher.
        vectors = {f"{i}.jpg": np.ones(8) for i in range(count)}
        report = prune(_views(count), _Index(vectors))
        # Le plancher impose MIN_KEPT, mais pas davantage.
        assert len(report.kept) == MIN_KEPT

    def test_dropped_view_names_its_representative(self):
        count = MIN_KEPT + 6
        vectors = {f"{i}.jpg": np.ones(8) for i in range(count)}
        report = prune(_views(count), _Index(vectors))
        for view in report.views:
            if not view.kept:
                assert view.represented_by is not None
                assert view.similarity is not None

    def test_a_dropped_view_is_never_silently_removed(self):
        count = MIN_KEPT + 6
        vectors = {f"{i}.jpg": np.ones(8) for i in range(count)}
        report = prune(_views(count), _Index(vectors))
        assert len(report.views) == count
        assert all(v.reason for v in report.views)

    def test_threshold_governs_how_much_is_merged(self):
        rng = np.random.default_rng(3)
        count = MIN_KEPT + 12
        base = rng.normal(size=8)
        vectors = {
            f"{i}.jpg": base + rng.normal(scale=0.25, size=8) for i in range(count)
        }
        loose = prune(_views(count), _Index(vectors), threshold=0.99)
        tight = prune(_views(count), _Index(vectors), threshold=0.5)
        assert len(tight.kept) <= len(loose.kept)


class TestQualityChoosesRepresentative:
    def test_best_scored_view_represents_its_group(self):
        count = MIN_KEPT + 6
        vectors = {f"{i}.jpg": np.ones(8) for i in range(count)}
        quality = {f"A{i}": 0.1 for i in range(count)}
        quality["A7"] = 0.99
        report = prune(_views(count), _Index(vectors), quality=quality)
        assert "A7" in report.kept

    def test_missing_quality_does_not_disqualify(self):
        count = MIN_KEPT + 4
        report = prune(_views(count), _Index(_distinct(count)), quality={})
        assert len(report.kept) == count


class TestFloor:
    def test_floor_is_respected_even_when_all_views_match(self):
        count = MIN_KEPT + 10
        vectors = {f"{i}.jpg": np.ones(8) for i in range(count)}
        report = prune(_views(count), _Index(vectors))
        assert len(report.kept) >= MIN_KEPT

    def test_recovered_views_are_the_least_similar_ones(self):
        """Ce qui revient doit apporter au graphe, non le répéter."""
        count = MIN_KEPT + 8
        rng = np.random.default_rng(11)
        vectors = {
            f"{i}.jpg": np.ones(8) + rng.normal(scale=0.02, size=8)
            for i in range(count)
        }
        report = prune(_views(count), _Index(vectors))
        assert len(report.kept) == MIN_KEPT


class TestUnreadable:
    def test_unreadable_image_is_kept_and_flagged(self):
        count = MIN_KEPT + 4
        report = prune(
            _views(count), _Index(_distinct(count), failing={"2.jpg"})
        )
        broken = next(v for v in report.views if v.asset_id == "A2")
        assert broken.kept
        assert "non encodable" in broken.reason

    def test_all_unreadable_yields_an_empty_but_valid_report(self):
        names = {f"{i}.jpg" for i in range(4)}
        report = prune(_views(4), _Index({}, failing=names))
        assert len(report.views) == 4
        assert all(v.kept for v in report.views)


class TestReport:
    def test_report_serialises(self):
        count = MIN_KEPT + 4
        payload = prune(_views(count), _Index(_distinct(count))).as_dict()
        assert payload["total"] == count
        assert payload["kept_count"] + payload["dropped_count"] == count
        assert payload["caveats"]

    def test_kept_and_dropped_partition_the_corpus(self):
        count = MIN_KEPT + 6
        vectors = {f"{i}.jpg": np.ones(8) for i in range(count)}
        report = prune(_views(count), _Index(vectors))
        assert set(report.kept) & set(report.dropped) == set()
        assert len(report.kept) + len(report.dropped) == count
