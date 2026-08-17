"""Preuve tirée du corpus déjà collecté, et résolution d'un objet de site.

Deux mécanismes manquaient au pilote : constater sur un asset qu'aucun plan
ciblé n'avait commandé, et **établir** un objet de site — `site unresolve`
savait démentir, rien ne savait confirmer.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from hotel_pipeline.cli import app
from hotel_pipeline.schemas import Asset, AssetCategory, AssetManifest, Rights
from hotel_pipeline.workspace import Workspace

runner = CliRunner()


def _workspace(tmp_path, monkeypatch, *, checksum: str = "a" * 64):
    monkeypatch.setenv("HOTEL_PIPELINE_WORK", str(tmp_path / "work"))
    workspace = Workspace("hotel-test")
    workspace.create()
    asset = Asset(
        id="mapillary-1", source="mapillary", source_url_or_id="1",
        rights=Rights.OPEN_DATA, checksum=checksum,
        ai_eligible=False, confidence=0.9, category=AssetCategory.FACADE,
    )
    workspace.write_assets(AssetManifest(hotel_id="hotel-test", assets=[asset]))
    return workspace


def test_corpus_assessment_records_an_honest_provenance(tmp_path, monkeypatch) -> None:
    """La filiation dit `corpus:`, jamais un plan ciblé inventé."""
    workspace = _workspace(tmp_path, monkeypatch)
    result = runner.invoke(app, [
        "assets", "preview", "assess-corpus", "hotel-test",
        "--asset", "mapillary-1", "--demand", "obligation:PROPERTY_SIGN",
        "--verdict", "established", "--rationale", "enseigne lisible",
        "--by", "op",
    ])
    assert result.exit_code == 0, result.stdout

    log = workspace.read_previews()
    entry = log.entries[-1]
    assert entry.plan_id == "corpus:mapillary"
    assert entry.request_digest.startswith("corpus:")
    # L'empreinte reste celle du fichier réellement examiné.
    assert entry.checksum == "a" * 64


def test_corpus_assessment_refuses_an_asset_from_a_targeted_plan(
    tmp_path, monkeypatch
) -> None:
    """Un fichier commandé relève de `preview assess`, qui vérifie le besoin."""
    from hotel_pipeline.schemas.acquisition import (
        AcquisitionProvenance,
        CaptureIntent,
    )

    workspace = _workspace(tmp_path, monkeypatch)
    manifest = workspace.read_assets()
    manifest.assets[0].acquisition = AcquisitionProvenance(
        provider_id="mapillary",
        plan_id="20260101T000000000000Z",
        plan_digest="abcd",
        candidate_id="mapillary-1",
        intents=[CaptureIntent.BUILDING_CAPTURE],
        serves_demands=["obligation:FACADE_PRIMARY"],
    )
    workspace.write_assets(manifest)

    result = runner.invoke(app, [
        "assets", "preview", "assess-corpus", "hotel-test",
        "--asset", "mapillary-1", "--demand", "obligation:PROPERTY_SIGN",
        "--verdict", "established", "--rationale", "x", "--by", "op",
    ])
    assert result.exit_code == 2
    assert "plan ciblé" in result.output


def test_corpus_assessment_refuses_an_asset_without_a_checksum(
    tmp_path, monkeypatch
) -> None:
    """Sans empreinte, le constat ne désignerait aucun fichier vérifiable."""
    workspace = _workspace(tmp_path, monkeypatch)
    manifest = workspace.read_assets()
    manifest.assets[0].checksum = ""
    workspace.write_assets(manifest)

    result = runner.invoke(app, [
        "assets", "preview", "assess-corpus", "hotel-test",
        "--asset", "mapillary-1", "--demand", "obligation:PROPERTY_SIGN",
        "--verdict", "established", "--rationale", "x", "--by", "op",
    ])
    assert result.exit_code == 2
    assert "empreinte" in result.output


# --- résolution d'un objet de site -----------------------------------------


def _site(workspace, kind: str = "PROPERTY_SIGN"):
    from hotel_pipeline.schemas.enums import ObjectState
    from hotel_pipeline.schemas.site import SiteManifest, SiteObject

    site = SiteManifest(
        hotel_id="hotel-test",
        objects=[SiteObject(
            object_id=f"hotel-test:{kind}", kind=kind,
            state=ObjectState.UNRESOLVED, unresolved_reason="non établi",
        )],
    )
    workspace.write_site(site)
    return site


def test_confirming_without_an_established_preview_is_refused(
    tmp_path, monkeypatch
) -> None:
    """« Confirmé » sans constat serait une conviction, pas un fait."""
    workspace = _workspace(tmp_path, monkeypatch)
    _site(workspace)

    result = runner.invoke(app, [
        "site", "resolve", "hotel-test", "--kind", "PROPERTY_SIGN",
        "--state", "confirmed", "--rationale", "je l'ai vue", "--by", "op",
    ])
    assert result.exit_code == 2
    assert "aucun constat établi" in result.output


def test_confirming_reads_the_established_previews(tmp_path, monkeypatch) -> None:
    workspace = _workspace(tmp_path, monkeypatch)
    _site(workspace)
    runner.invoke(app, [
        "assets", "preview", "assess-corpus", "hotel-test",
        "--asset", "mapillary-1", "--demand", "obligation:PROPERTY_SIGN",
        "--verdict", "established", "--rationale", "enseigne lisible",
        "--by", "op",
    ])

    result = runner.invoke(app, [
        "site", "resolve", "hotel-test", "--kind", "PROPERTY_SIGN",
        "--state", "confirmed", "--rationale", "deux vues concordantes",
        "--by", "op",
    ])
    assert result.exit_code == 0, result.stdout

    from hotel_pipeline.schemas.enums import ObjectState

    obj = workspace.read_site().objects[0]
    assert obj.state is ObjectState.CONFIRMED
    assert obj.unresolved_reason is None
    assert "preview:mapillary-1" in obj.evidence
    # Aucune géométrie n'est fabriquée par la résolution.
    assert obj.geometry_wkt is None
    assert obj.centroid_lat is None


def test_inferring_needs_no_preview_but_still_a_rationale(
    tmp_path, monkeypatch
) -> None:
    """`inferred` dit ce qu'il est : observé sans être établi."""
    workspace = _workspace(tmp_path, monkeypatch)
    _site(workspace, "PARK_AND_RIDE")

    result = runner.invoke(app, [
        "site", "resolve", "hotel-test", "--kind", "PARK_AND_RIDE",
        "--state", "inferred", "--rationale", "terminus observé sur une vue",
        "--by", "op",
    ])
    assert result.exit_code == 0, result.stdout

    from hotel_pipeline.schemas.enums import ObjectState

    assert workspace.read_site().objects[0].state is ObjectState.INFERRED


def test_resolve_refuses_to_unresolve(tmp_path, monkeypatch) -> None:
    """Dé-résoudre reste le travail de `site unresolve`, qui périme ce qui dérive."""
    workspace = _workspace(tmp_path, monkeypatch)
    _site(workspace)

    result = runner.invoke(app, [
        "site", "resolve", "hotel-test", "--kind", "PROPERTY_SIGN",
        "--state", "unresolved", "--rationale", "x", "--by", "op",
    ])
    assert result.exit_code == 2
    assert "site unresolve" in result.output


def test_a_resolution_leaves_an_append_only_trace(tmp_path, monkeypatch) -> None:
    workspace = _workspace(tmp_path, monkeypatch)
    _site(workspace, "PARK_AND_RIDE")
    runner.invoke(app, [
        "site", "resolve", "hotel-test", "--kind", "PARK_AND_RIDE",
        "--state", "inferred", "--rationale", "terminus observé", "--by", "op",
    ])

    traces = list(workspace.path("00_manifest").glob("site_resolve_PARK_AND_RIDE_*.json"))
    assert len(traces) == 1
    payload = json.loads(traces[0].read_text("utf-8"))
    assert payload["state"] == "inferred"
    assert payload["decided_by"] == "op"
