"""Contrat du registre factuel des familles photographiques."""

from __future__ import annotations

import json

import pydantic
import pytest
from typer.testing import CliRunner

from hotel_pipeline.cli import app
from hotel_pipeline.source_registry import (
    SourceFamilyRecord,
    SourceFamilyState,
    SourceRegistry,
)


def _record(state: SourceFamilyState, *, closed: bool) -> SourceFamilyRecord:
    return SourceFamilyRecord(
        family_id="source", priority="A", collector_id="source",
        state=state, asset_count=0, candidate_count=0,
        evidence=["preuve.json"], reason="raison", campaign_closed=closed,
    )


def test_family_closure_requires_a_terminal_state() -> None:
    with pytest.raises(pydantic.ValidationError, match="campaign_closed"):
        _record(SourceFamilyState.EVIDENCE_PRESENT, closed=True)


def test_registry_counts_only_required_terminal_families() -> None:
    closed = _record(SourceFamilyState.QUERIED_CURRENT, closed=True)
    pending = closed.model_copy(update={
        "family_id": "pending", "state": SourceFamilyState.NOT_EVIDENCED,
        "campaign_closed": False,
    })
    registry = SourceRegistry(
        hotel_id="h", generated_at="now",
        input_digests={
            "asset_manifest": "a", "candidate_manifest": "c", "lot_1b_plan": "p",
        },
        families=[closed, pending], required_families=2, closed_families=1,
        closure_complete=False,
    )
    assert not registry.closure_complete


def test_registry_refuses_a_false_complete_claim() -> None:
    pending = _record(SourceFamilyState.NOT_IMPLEMENTED, closed=False)
    with pytest.raises(pydantic.ValidationError, match="closure_complete"):
        SourceRegistry(
            hotel_id="h", generated_at="now",
            input_digests={
                "asset_manifest": "a", "candidate_manifest": "c", "lot_1b_plan": "p",
            },
            families=[pending], required_families=1, closed_families=0,
            closure_complete=True,
        )


def test_sources_registry_is_exposed_by_the_real_cli() -> None:
    result = CliRunner().invoke(app, ["sources", "--help"])
    assert result.exit_code == 0
    assert "registry" in result.output


# --- reçus d'indisponibilité ------------------------------------------------


def _workspace(tmp_path):
    from hotel_pipeline.workspace import Workspace

    workspace = Workspace("hotel-test", root=tmp_path)
    workspace.create()
    return workspace


def test_a_receipt_refuses_an_unknown_family(tmp_path) -> None:
    from hotel_pipeline.source_registry import record_unavailable

    with pytest.raises(ValueError, match="famille inconnue"):
        record_unavailable(_workspace(tmp_path), "source_imaginaire", "motif", "op")


def test_receipts_accumulate_without_rewriting_the_previous_ones(tmp_path) -> None:
    from hotel_pipeline.source_registry import record_unavailable

    workspace = _workspace(tmp_path)
    record_unavailable(workspace, "booking", "API fermée", "op")
    path = record_unavailable(workspace, "iceportal", "aucun accès public", "op")

    rows = json.loads(path.read_text("utf-8"))
    assert [row["family_id"] for row in rows] == ["booking", "iceportal"]
    assert all(not row["withdrawn"] for row in rows)


def test_withdrawing_keeps_the_history_and_reopens_the_family(tmp_path) -> None:
    from hotel_pipeline.source_registry import (
        record_unavailable,
        withdraw_unavailable,
    )

    workspace = _workspace(tmp_path)
    record_unavailable(workspace, "booking", "API fermée", "op")
    path = withdraw_unavailable(workspace, "booking", "op", "API rouverte")

    rows = json.loads(path.read_text("utf-8"))
    # Le constat initial demeure : il explique pourquoi la famille fut close.
    assert len(rows) == 2
    assert all(row["withdrawn"] for row in rows)


def test_withdrawing_an_absent_receipt_is_refused(tmp_path) -> None:
    from hotel_pipeline.source_registry import withdraw_unavailable

    with pytest.raises(ValueError, match="aucun reçu actif"):
        withdraw_unavailable(_workspace(tmp_path), "booking", "op", "motif")


def _registry_workspace(tmp_path):
    """Espace minimal permettant à `build` de produire un registre réel."""
    from hotel_pipeline.schemas import (
        Asset,
        AssetCategory,
        AssetManifest,
        Rights,
    )

    workspace = _workspace(tmp_path)
    asset = Asset(
        id="asset-1", source="mapillary", source_url_or_id="1",
        rights=Rights.PUBLIC_UNCLEARED, checksum="a" * 64,
        ai_eligible=False, confidence=0.8, category=AssetCategory.OTHER,
    )
    workspace.write_assets(AssetManifest(hotel_id="hotel-test", assets=[asset]))
    workspace.write_json(
        "01_sources/candidates_20260101T000000000000Z.json",
        {"candidates": [{"source": "mapillary"}]},
    )
    return workspace


def _family(registry, family_id):
    return next(row for row in registry.families if row.family_id == family_id)


def test_a_receipt_closes_the_family_in_the_published_registry(tmp_path) -> None:
    from hotel_pipeline.source_registry import build, record_unavailable

    workspace = _registry_workspace(tmp_path)
    before = SourceRegistry.model_validate_json(build(workspace).read_text("utf-8"))
    assert _family(before, "booking").state is SourceFamilyState.NOT_IMPLEMENTED
    assert not _family(before, "booking").campaign_closed

    record_unavailable(workspace, "booking", "API fermée au public", "op")
    after = SourceRegistry.model_validate_json(build(workspace).read_text("utf-8"))

    booking = _family(after, "booking")
    assert booking.state is SourceFamilyState.UNAVAILABLE_DOCUMENTED
    assert booking.campaign_closed
    assert booking.reason == "API fermée au public"
    assert after.closed_families == before.closed_families + 1


def test_a_stale_receipt_never_hides_a_real_current_query(tmp_path) -> None:
    """Une famille réellement interrogée prime sur son reçu d'indisponibilité."""
    from hotel_pipeline.source_registry import build, record_unavailable

    workspace = _registry_workspace(tmp_path)
    record_unavailable(workspace, "mapillary", "prétendue indisponible", "op")
    registry = SourceRegistry.model_validate_json(build(workspace).read_text("utf-8"))

    mapillary = _family(registry, "mapillary")
    assert mapillary.state is SourceFamilyState.QUERIED_CURRENT
    assert "manifeste canonique courant" in mapillary.reason


def test_withdrawing_a_receipt_reopens_the_campaign(tmp_path) -> None:
    from hotel_pipeline.source_registry import (
        build,
        record_unavailable,
        withdraw_unavailable,
    )

    workspace = _registry_workspace(tmp_path)
    record_unavailable(workspace, "booking", "API fermée", "op")
    closed = SourceRegistry.model_validate_json(build(workspace).read_text("utf-8"))
    withdraw_unavailable(workspace, "booking", "op", "API rouverte")
    reopened = SourceRegistry.model_validate_json(build(workspace).read_text("utf-8"))

    assert not _family(reopened, "booking").campaign_closed
    assert reopened.closed_families == closed.closed_families - 1


# --- reçus de campagne ------------------------------------------------------


def test_a_campaign_receipt_closes_a_family_queried_outside_discovery(
    tmp_path,
) -> None:
    """Un collecteur exécuté directement laisse enfin une trace lisible."""
    from hotel_pipeline.source_registry import build, record_campaign

    workspace = _registry_workspace(tmp_path)
    before = SourceRegistry.model_validate_json(build(workspace).read_text("utf-8"))
    assert not _family(before, "wikimedia_commons").campaign_closed

    record_campaign(
        workspace, "wikimedia_commons",
        query="geosearch 45.57,-73.44", returned=3,
        evidence="3 images CC BY-SA", by="op",
    )
    after = SourceRegistry.model_validate_json(build(workspace).read_text("utf-8"))

    commons = _family(after, "wikimedia_commons")
    assert commons.state is SourceFamilyState.QUERIED_CURRENT
    assert commons.campaign_closed
    assert "3 résultat(s)" in commons.reason


def test_an_empty_campaign_still_closes_the_family(tmp_path) -> None:
    """C'est l'interrogation qui ferme, pas la moisson."""
    from hotel_pipeline.source_registry import build, record_campaign

    workspace = _registry_workspace(tmp_path)
    record_campaign(
        workspace, "flickr", query="CC autour du site", returned=0,
        evidence="aucune image sous licence CC dans le rayon", by="op",
    )
    registry = SourceRegistry.model_validate_json(build(workspace).read_text("utf-8"))

    flickr = _family(registry, "flickr")
    assert flickr.campaign_closed
    assert "0 résultat(s)" in flickr.reason


def test_a_campaign_receipt_refuses_an_unknown_family(tmp_path) -> None:
    from hotel_pipeline.source_registry import record_campaign

    with pytest.raises(ValueError, match="famille inconnue"):
        record_campaign(
            _workspace(tmp_path), "source_imaginaire",
            query="q", returned=1, evidence="e", by="op",
        )


def test_a_current_discovery_outranks_a_campaign_receipt(tmp_path) -> None:
    """Le manifeste courant reste la preuve la plus forte."""
    from hotel_pipeline.source_registry import build, record_campaign

    workspace = _registry_workspace(tmp_path)
    record_campaign(
        workspace, "mapillary", query="ancienne campagne", returned=1,
        evidence="trace plus ancienne", by="op",
    )
    registry = SourceRegistry.model_validate_json(build(workspace).read_text("utf-8"))

    mapillary = _family(registry, "mapillary")
    assert mapillary.state is SourceFamilyState.QUERIED_CURRENT
    assert "manifeste canonique courant" in mapillary.reason
