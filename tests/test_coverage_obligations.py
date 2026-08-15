"""Obligations de couverture : aucun objet n'est oublié en silence.

Sans ce chaînon, un manifeste de besoins pouvait omettre la façade arrière et
paraître complet : le Router aurait compté des besoins tous satisfaits, sans
savoir qu'un objet n'en avait jamais eu.
"""

from __future__ import annotations

import pytest

from hotel_pipeline.coverage_obligations import (
    NO_PHOTOGRAPHIC_OBLIGATION,
    OBLIGATIONS,
    CoverageObligation,
    ObligationStatus,
    ObligationWaiver,
    assess,
    missing_demands,
)
from hotel_pipeline.schemas.acquisition import (
    CaptureDemand,
    CaptureIntent,
    TargetKind,
)
from hotel_pipeline.schemas.critical_objects import REQUIRED_OBJECTS


def demand_for(obligation: CoverageObligation, demand_id: str | None = None) -> CaptureDemand:
    return CaptureDemand(
        demand_id=demand_id or f"d-{obligation.object_id.lower()}",
        intent=obligation.intent,
        target_kind=obligation.target_kind,
        target_ref=obligation.expected_target_ref,
    )


def mandatory() -> list[CoverageObligation]:
    return [obligation for obligation in OBLIGATIONS if obligation.mandatory]


def waiver(object_id: str, **overrides) -> ObligationWaiver:
    fields = dict(
        object_id=object_id, status=ObligationStatus.NOT_APPLICABLE,
        decided_by="Hicham", rationale="l'établissement n'en a pas",
        evidence=["visite du 2026-08-14"],
    )
    fields.update(overrides)
    return ObligationWaiver(**fields)


# --- rien ne disparaît en silence ---------------------------------------------


def test_an_empty_manifest_leaves_every_mandatory_obligation_unmet() -> None:
    report = assess([])

    assert set(report.unmet) == {o.object_id for o in mandatory()}
    assert report.complete is False


def test_the_rear_facade_cannot_be_quietly_omitted() -> None:
    """La face la plus souvent absente, et celle qu'un oubli emporte."""
    complete = [demand_for(o) for o in mandatory()]
    without_rear = [d for d in complete if d.target_ref != "rear"]

    assert assess(complete).complete is True

    partial = assess(without_rear)
    assert partial.complete is False
    assert partial.unmet == ["FACADE_REAR"]


def test_a_covered_obligation_names_the_demands_that_cover_it() -> None:
    report = assess([demand_for(o) for o in mandatory()])

    assert report.demands_by_object["FACADE_REAR"] == ["d-facade_rear"]
    assert ObligationStatus.UNMET.value not in report.by_status


def test_two_demands_may_cover_one_obligation() -> None:
    rear = next(o for o in OBLIGATIONS if o.object_id == "FACADE_REAR")

    report = assess([demand_for(rear, "d1"), demand_for(rear, "d2")])

    assert report.demands_by_object["FACADE_REAR"] == ["d1", "d2"]


# --- dispenser est une décision, jamais une omission --------------------------


def test_a_waived_obligation_is_not_unmet() -> None:
    complete = [demand_for(o) for o in mandatory() if o.object_id != "PROPERTY_SIGN"]

    report = assess(complete, waivers=[waiver("PROPERTY_SIGN")])

    assert report.complete is True
    assert "PROPERTY_SIGN" in report.by_status[ObligationStatus.NOT_APPLICABLE.value]


def test_a_dispense_without_evidence_is_refused() -> None:
    """Renoncer sans dire sur quoi on se fonde interdit d'y revenir."""
    with pytest.raises(ValueError, match="dispense sans preuve"):
        waiver("PROPERTY_SIGN", evidence=[])


def test_a_dispense_cannot_declare_an_obligation_met() -> None:
    """`demanded` et `unmet` se constatent ; ils ne se décident pas."""
    with pytest.raises(ValueError, match="une dispense déclare"):
        waiver("PROPERTY_SIGN", status=ObligationStatus.DEMANDED)

    with pytest.raises(ValueError, match="une dispense déclare"):
        waiver("PROPERTY_SIGN", status=ObligationStatus.UNMET)


def test_waiving_and_not_applicable_are_distinct() -> None:
    """« L'objet n'existe pas » et « on renonce » ne disent pas la même chose."""
    absent = waiver("PARKING_HOTEL", status=ObligationStatus.NOT_APPLICABLE)
    renounced = waiver(
        "FACADE_REAR", status=ObligationStatus.WAIVED,
        rationale="mitoyenne, aucun recul possible",
    )

    assert absent.status is not renounced.status


# --- ce que le gabarit n'exige pas en photo -----------------------------------


def test_every_template_object_is_either_obliged_or_explicitly_exempt() -> None:
    """Un objet ajouté au gabarit et oublié n'exigerait rien, en silence."""
    obliged = {obligation.object_id for obligation in OBLIGATIONS}
    exempt = set(NO_PHOTOGRAPHIC_OBLIGATION)

    unaccounted = sorted(set(REQUIRED_OBJECTS) - obliged - exempt)

    assert unaccounted == []


def test_the_roofline_is_never_demanded_from_the_street() -> None:
    """L'exiger en photo produirait une obligation intenable."""
    assert "ROOFLINE_MAIN" in NO_PHOTOGRAPHIC_OBLIGATION
    assert "invisible depuis la voirie" in NO_PHOTOGRAPHIC_OBLIGATION["ROOFLINE_MAIN"]
    assert all(o.object_id != "ROOFLINE_MAIN" for o in OBLIGATIONS)


def test_the_parcel_is_never_proven_by_a_photograph() -> None:
    assert "PROPERTY_PARCEL" in NO_PHOTOGRAPHIC_OBLIGATION
    assert all(o.object_id != "PROPERTY_PARCEL" for o in OBLIGATIONS)


# --- un besoin hors gabarit se signale, sans être fautif ----------------------


def test_a_demand_serving_no_obligation_is_reported() -> None:
    """Une cible mal orthographiée ne doit pas passer pour une exigence de plus."""
    extra = CaptureDemand(
        demand_id="d-typo", intent=CaptureIntent.BUILDING_CAPTURE,
        target_kind=TargetKind.VIEW_SECTOR, target_ref="roof",
    )

    report = assess([*(demand_for(o) for o in mandatory()), extra])

    assert report.orphan_demands == ["d-typo"]
    # Elle ne rend pas le manifeste incomplet pour autant : on peut vouloir
    # davantage que le minimum.
    assert report.complete is True


# --- ce que ce rapport ne dit pas ---------------------------------------------


def test_a_covered_obligation_is_not_a_satisfied_demand() -> None:
    """Ce module dit qu'aucun objet n'a été oublié, pas qu'on possède les vues."""
    report = assess([demand_for(o) for o in mandatory()])

    assert report.complete is True
    assert "pas qu'on possède les vues" in report.as_dict()["note"]


def test_the_missing_obligations_are_actionable() -> None:
    """Signaler un oubli sans dire quoi écrire n'aide personne."""
    report = assess([])

    missing = missing_demands(report)

    assert {o.object_id for o in missing} == set(report.unmet)
    assert all(o.expected_target_ref for o in missing)
    assert all(o.rationale for o in missing)


def test_every_obligation_says_why_it_exists() -> None:
    assert all(obligation.rationale for obligation in OBLIGATIONS)


def test_no_obligation_names_a_property() -> None:
    """Le gabarit est générique : aucun établissement n'y figure."""
    blob = " ".join(
        f"{o.object_id} {o.expected_target_ref} {o.rationale}" for o in OBLIGATIONS
    ).lower()

    for name in ("welcominns", "boucherville", "mortagne", "ampère"):
        assert name not in blob


# --- la commande nomme l'oubli ------------------------------------------------


@pytest.fixture
def project(tmp_path, monkeypatch):
    import json

    from typer.testing import CliRunner

    from hotel_pipeline.cli import app
    from hotel_pipeline.schemas.acquisition import (
        CandidateManifest, CaptureCandidate, CaptureDemandManifest,
    )
    from hotel_pipeline.workspace import Workspace

    monkeypatch.setenv("HOTEL_PIPELINE_WORK", str(tmp_path / "work"))
    monkeypatch.setenv("HOTEL_PIPELINE_PROFILES", str(tmp_path / "profiles"))
    monkeypatch.setenv("HOTEL_PIPELINE_OFFLINE", "1")
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    runner.invoke(app, [
        "init", "hotel-test", "--address", "1 rue Test", "--name", "Hôtel Test",
        "--country", "CA", "--timezone", "America/Toronto", "--ocr-language", "fr",
        "--lat", "45.573", "--lon", "-73.443",
    ])

    workspace = Workspace("hotel-test")

    def write_demands(demands):
        workspace.write_json(
            "01_sources/capture_demands.json",
            json.loads(
                CaptureDemandManifest(hotel_id="hotel-test", demands=demands)
                .model_dump_json()
            ),
        )

    workspace.write_json(
        "01_sources/candidates_20260814T000000000000Z.json",
        json.loads(
            CandidateManifest(
                hotel_id="hotel-test",
                candidates=[CaptureCandidate(
                    candidate_id="c1", source="mapillary", provider_id="1",
                    camera_lat=45.573, camera_lon=-73.443,
                )],
            ).model_dump_json()
        ),
    )
    return runner, workspace, write_demands


def test_the_cli_names_the_forgotten_rear_facade(project) -> None:
    """Le cas que la validation existe pour empêcher."""
    from hotel_pipeline.cli import app

    runner, _, write_demands = project
    write_demands([
        demand_for(o) for o in mandatory() if o.expected_target_ref != "rear"
    ])

    result = runner.invoke(app, ["assets", "plan", "hotel-test"])

    assert result.exit_code == 0, result.output
    assert "obligation(s) sans demande ni dispense" in result.output
    assert "FACADE_REAR" in result.output
    assert "coverage_waivers.json" in result.output


def test_a_complete_manifest_says_so(project) -> None:
    from hotel_pipeline.cli import app

    runner, _, write_demands = project
    write_demands([demand_for(o) for o in mandatory()])

    result = runner.invoke(app, ["assets", "plan", "hotel-test"])

    assert result.exit_code == 0, result.output
    assert "obligations couvertes" in result.output
    assert "sans demande ni dispense" not in result.output


def test_a_declared_waiver_silences_the_warning(project) -> None:
    import json

    from hotel_pipeline.cli import app

    runner, workspace, write_demands = project
    write_demands([
        demand_for(o) for o in mandatory() if o.object_id != "PROPERTY_SIGN"
    ])
    workspace.write_json(
        "01_sources/coverage_waivers.json",
        {"waivers": [json.loads(waiver("PROPERTY_SIGN").model_dump_json())]},
    )

    result = runner.invoke(app, ["assets", "plan", "hotel-test"])

    assert result.exit_code == 0, result.output
    assert "obligations couvertes" in result.output
