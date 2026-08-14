"""Droits d'usage : acquisition factuelle, décision séparée (collecte V2).

Ce qui est éprouvé : accepter un risque n'améliore aucun droit, une
autorisation sans preuve est refusée, et rien ne s'écrase — une décision se
corrige en ajoutant.
"""

from __future__ import annotations

import pytest

from hotel_pipeline.rights import acquisition_rights, apply, effect
from hotel_pipeline.schemas import Asset, Rights
from hotel_pipeline.schemas.rights import CLEARABLE, RightsAction, RightsDecision

DIGEST = "a" * 64


def asset(**overrides) -> Asset:
    fields = dict(
        id="mapillary-1", source="mapillary", source_url_or_id="1",
        rights=Rights.PUBLIC_UNCLEARED, ai_eligible=False, confidence=0.5,
        category="facade", checksum=DIGEST,
    )
    fields.update(overrides)
    return Asset(**fields)


def decision(action=RightsAction.CLEAR, **overrides) -> RightsDecision:
    fields = dict(
        action=action, decided_by="Hicham", rationale="autorisation écrite du gérant",
        scope="reconstruction interne", reviewed_checksum=DIGEST,
        evidence=["courriel du 2026-08-14"],
    )
    if action is RightsAction.CLEAR:
        fields["granted_rights"] = Rights.LICENSED
    fields.update(overrides)
    return RightsDecision(**fields)


# --- l'acquisition ne tranche rien --------------------------------------------


def test_acquisition_writes_public_uncleared_and_nothing_else() -> None:
    """« --rights owned » permettait d'écrire un statut sans la moindre preuve."""
    written = acquisition_rights()

    assert written["rights"] is Rights.PUBLIC_UNCLEARED
    assert written["rights_encumbered"] is False
    assert written["rights_note"] is None


def test_a_source_licence_is_recorded_as_a_claim() -> None:
    """Un fournisseur affichant « CC BY » ne prouve pas qu'il pouvait l'accorder."""
    written = acquisition_rights("CC BY-SA 4.0")

    assert written["rights"] is Rights.PUBLIC_UNCLEARED
    assert "revendiquée par la source" in written["rights_note"]


# --- accepter un risque n'accorde rien ----------------------------------------


def test_assuming_a_risk_never_improves_the_rights() -> None:
    """C'est tout l'objet de la séparation.

    Falsifier l'état juridique pour se donner le droit de continuer rendrait
    le manifeste inutilisable comme preuve de diligence.
    """
    before = asset()

    after = apply(before, decision(RightsAction.ASSUME_RISK, granted_rights=None))

    assert after.rights is Rights.PUBLIC_UNCLEARED
    assert after.rights_encumbered is True
    assert "risque assumé" in after.rights_note


def test_an_assumed_risk_carries_no_granted_rights() -> None:
    with pytest.raises(ValueError, match="n'établit aucun droit"):
        decision(RightsAction.ASSUME_RISK, granted_rights=Rights.OWNED)


def test_an_assumed_risk_must_say_what_was_examined() -> None:
    with pytest.raises(ValueError, match="sans preuve de ce qui a été examiné"):
        decision(RightsAction.ASSUME_RISK, granted_rights=None, evidence=[])


def test_an_encumbered_asset_may_still_be_used_and_stays_traced() -> None:
    """Votre décision : assumer le risque sans falsifier l'état juridique."""
    encumbered = apply(asset(), decision(RightsAction.ASSUME_RISK, granted_rights=None))

    assert encumbered.usable_in_production is True
    assert encumbered.rights_encumbered is True
    assert encumbered.rights is Rights.PUBLIC_UNCLEARED


# --- une autorisation se prouve -----------------------------------------------


def test_a_clearance_without_evidence_is_refused() -> None:
    """Une autorisation qu'on ne peut pas produire est une affirmation."""
    with pytest.raises(ValueError, match="sans preuve"):
        decision(evidence=[])


def test_a_clearance_must_grant_something() -> None:
    with pytest.raises(ValueError, match="sans droits établis"):
        decision(granted_rights=None)


def test_a_clearance_cannot_establish_an_absence() -> None:
    """`public_uncleared` décrit une absence : on ne décide pas d'une absence."""
    with pytest.raises(ValueError, match="décrit une absence"):
        decision(granted_rights=Rights.PUBLIC_UNCLEARED)

    with pytest.raises(ValueError, match="décrit une absence"):
        decision(granted_rights=Rights.UNKNOWN)


def test_a_proven_clearance_lifts_the_encumbrance() -> None:
    encumbered = apply(asset(), decision(RightsAction.ASSUME_RISK, granted_rights=None))

    cleared = apply(encumbered, decision(granted_rights=Rights.LICENSED))

    assert cleared.rights is Rights.LICENSED
    assert cleared.rights_encumbered is False
    assert len(cleared.rights_history) == 2


def test_the_scope_survives_in_the_note() -> None:
    """« usage interne » et « diffusion publique » ne sont pas la même permission."""
    cleared = apply(asset(), decision(scope="diffusion publique"))

    assert "diffusion publique" in cleared.rights_note


# --- rien ne s'écrase ----------------------------------------------------------


def test_decisions_are_appended_never_replaced() -> None:
    subject = apply(asset(), decision(granted_rights=Rights.LICENSED))
    revoked = apply(
        subject,
        decision(RightsAction.REVOKE, granted_rights=None, rationale="licence caduque",
                 supersedes_index=0),
    )

    assert len(revoked.rights_history) == 2
    assert revoked.rights_history[0].action is RightsAction.CLEAR
    assert revoked.rights is Rights.PUBLIC_UNCLEARED


def test_a_decision_taken_on_another_file_is_refused() -> None:
    """Les droits examinés étaient ceux d'une image qui n'est plus celle-ci."""
    with pytest.raises(ValueError, match="ne portait pas sur ce fichier"):
        apply(asset(checksum="b" * 64), decision())


def test_the_clearable_set_excludes_absences() -> None:
    assert CLEARABLE == {Rights.OWNED, Rights.LICENSED, Rights.OPEN_DATA}
    assert Rights.PUBLIC_UNCLEARED not in CLEARABLE
    assert Rights.UNKNOWN not in CLEARABLE


# --- la commande, de bout en bout ---------------------------------------------


@pytest.fixture
def project(tmp_path, monkeypatch):
    import json

    from typer.testing import CliRunner

    from hotel_pipeline.cli import app
    from hotel_pipeline.schemas import AssetManifest
    from hotel_pipeline.workspace import Workspace

    monkeypatch.setenv("HOTEL_PIPELINE_WORK", str(tmp_path / "work"))
    monkeypatch.setenv("HOTEL_PIPELINE_PROFILES", str(tmp_path / "profiles"))
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    runner.invoke(app, [
        "init", "hotel-test", "--address", "1 rue Test", "--name", "Hôtel Test",
        "--country", "CA", "--timezone", "America/Toronto", "--ocr-language", "fr",
    ])
    workspace = Workspace("hotel-test")
    workspace.write_assets(AssetManifest(hotel_id="hotel-test", assets=[asset()]))
    return runner, workspace


def test_the_cli_refuses_a_clearance_without_evidence(project) -> None:
    from hotel_pipeline.cli import app

    runner, workspace = project

    result = runner.invoke(app, [
        "assets", "rights", "clear", "hotel-test", "mapillary-1",
        "--granted", "owned", "--scope", "tout", "--by", "moi",
        "--rationale", "parce que",
    ])

    assert result.exit_code == 2
    assert "sans preuve" in result.output
    # Rien n'a été écrit.
    assert workspace.read_assets().assets[0].rights is Rights.PUBLIC_UNCLEARED


def test_the_cli_assume_risk_says_it_grants_nothing(project) -> None:
    from hotel_pipeline.cli import app

    runner, workspace = project

    result = runner.invoke(app, [
        "assets", "rights", "assume-risk", "hotel-test", "mapillary-1",
        "--scope", "reconstruction interne", "--by", "Hicham",
        "--rationale", "droits d'usage assumés", "--evidence", "note du 2026-08-14",
    ])

    assert result.exit_code == 0, result.output
    assert "n'accorde rien" in result.output

    updated = workspace.read_assets().assets[0]
    assert updated.rights is Rights.PUBLIC_UNCLEARED
    assert updated.rights_encumbered is True
    assert len(updated.rights_history) == 1


def test_the_acquisition_command_no_longer_takes_rights() -> None:
    """L'option permettait d'écrire `owned` sans aucune preuve."""
    import inspect

    from hotel_pipeline.cli import assets_acquire

    assert "rights" not in inspect.signature(assets_acquire).parameters
