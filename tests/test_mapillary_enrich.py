"""Enrichissement Mapillary : compléter sans écraser."""

from __future__ import annotations

import pytest

from hotel_pipeline.mapillary_enrich import (
    BATCH_SIZE,
    DIVERGENCE_NOTICE_DEG,
    EnrichedAsset,
    EnrichmentReport,
    apply,
    enrich,
)


class _Asset:
    def __init__(self, identifier, heading=None):
        self.id = identifier
        self.heading_deg = heading
        self.computed_heading_deg = None
        self.sequence_id = None


def _fetch(payloads):
    def fetch(identifiers):
        return {i: payloads[i] for i in identifiers if i in payloads}

    return fetch


class TestEnrich:
    def test_a_computed_heading_is_recovered(self):
        assets = [_Asset("mapillary-1", 341.9)]
        report = enrich(
            assets, _fetch({"1": {"computed_compass_angle": 350.7, "sequence": "S"}})
        )
        entry = report.enriched[0]
        assert entry.computed_heading_deg == pytest.approx(350.7)
        assert entry.sequence_id == "S"

    def test_negative_angles_are_normalised(self):
        """Mapillary publie parfois des caps négatifs."""
        report = enrich(
            [_Asset("mapillary-1")], _fetch({"1": {"computed_compass_angle": -14.07}})
        )
        assert report.enriched[0].computed_heading_deg == pytest.approx(345.93, abs=0.01)

    def test_non_mapillary_assets_are_left_alone(self):
        """Les autres sources n'exposent pas ces champs : rien à inventer."""
        report = enrich([_Asset("street_view-abc", 10.0)], _fetch({}))
        assert report.enriched == []

    def test_an_image_the_source_ignores_is_reported(self):
        report = enrich([_Asset("mapillary-9")], _fetch({}))
        entry = report.enriched[0]
        assert entry.computed_heading_deg is None
        assert "ne rend rien" in entry.reason

    def test_a_missing_computed_angle_is_said_so(self):
        report = enrich([_Asset("mapillary-1")], _fetch({"1": {"sequence": "S"}}))
        assert report.enriched[0].reason == "cap calculé absent"
        assert report.enriched[0].sequence_id == "S"

    def test_a_failing_batch_does_not_stop_the_rest(self):
        def broken(_identifiers):
            raise RuntimeError("réseau indisponible")

        report = enrich([_Asset("mapillary-1")], broken)
        assert len(report.enriched) == 1
        assert report.enriched[0].computed_heading_deg is None

    def test_batching_covers_every_asset(self):
        assets = [_Asset(f"mapillary-{i}") for i in range(BATCH_SIZE * 2 + 5)]
        seen: list[int] = []

        def counting(identifiers):
            seen.append(len(identifiers))
            return {}

        enrich(assets, counting)
        assert sum(seen) == len(assets)


class TestDivergence:
    def test_divergence_is_the_shorter_arc(self):
        entry = EnrichedAsset("a", computed_heading_deg=350.0, declared_heading_deg=10.0)
        assert entry.divergence_deg == pytest.approx(20.0)

    def test_divergence_is_none_without_both_values(self):
        assert EnrichedAsset("a", computed_heading_deg=10.0).divergence_deg is None

    def test_a_marked_divergence_is_flagged(self):
        report = EnrichmentReport(
            enriched=[
                EnrichedAsset("a", 10.0, declared_heading_deg=10.0),
                EnrichedAsset("b", 200.0, declared_heading_deg=10.0),
            ]
        )
        assert len(report.diverging()) == 1

    def test_notice_threshold_is_meaningful(self):
        assert 0.0 < DIVERGENCE_NOTICE_DEG < 180.0


class TestApply:
    def test_fields_are_posed_on_the_asset(self):
        asset = _Asset("mapillary-1", 341.9)
        report = EnrichmentReport(
            enriched=[EnrichedAsset("mapillary-1", 350.7, "S", 341.9)]
        )
        assert apply([asset], report) == 1
        assert asset.computed_heading_deg == 350.7
        assert asset.sequence_id == "S"

    def test_the_declared_heading_is_never_overwritten(self):
        """Le point de la conception : les deux caps coexistent."""
        asset = _Asset("mapillary-1", 341.9)
        apply([asset], EnrichmentReport(enriched=[EnrichedAsset("mapillary-1", 350.7)]))
        assert asset.heading_deg == 341.9

    def test_an_unknown_asset_is_skipped(self):
        report = EnrichmentReport(enriched=[EnrichedAsset("mapillary-absent", 10.0)])
        assert apply([], report) == 0

    def test_an_empty_enrichment_touches_nothing(self):
        asset = _Asset("mapillary-1", 10.0)
        assert apply([asset], EnrichmentReport(enriched=[EnrichedAsset("mapillary-1")])) == 0
        assert asset.computed_heading_deg is None


class TestReport:
    def test_sequences_are_grouped(self):
        report = EnrichmentReport(
            enriched=[
                EnrichedAsset("a", 1.0, "S1"),
                EnrichedAsset("b", 2.0, "S1"),
                EnrichedAsset("c", 3.0, "S2"),
            ]
        )
        assert report.sequences() == {"S1": ["a", "b"], "S2": ["c"]}

    def test_report_serialises_with_caveats(self):
        payload = EnrichmentReport(
            enriched=[EnrichedAsset("a", 10.0, "S", 12.0)]
        ).as_dict()
        assert payload["with_computed_heading"] == 1
        assert payload["caveats"]
        assert any("ne remplace pas" in c for c in payload["caveats"])
