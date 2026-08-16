"""Mesurer un plan existant, sans jamais le refaire (collecte V2).

`assets plan --measure-volumes` reconstruit la sélection : les appels autorisés
pourraient alors porter sur d'autres candidats que ceux qu'un relecteur a
examinés. Un budget approuvé sur six requêtes précises ne vaut plus rien si la
sélection change entre l'examen et la mesure.

Le test décisif de ce fichier rend la sélection **différente** et vérifie que
`measure-plan` ne l'appelle pas.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from hotel_pipeline.cli import app
from hotel_pipeline.workspace import Workspace


def _project(tmp_path, monkeypatch):
    """Un espace de travail avec un plan, ses candidats et ses besoins."""
    monkeypatch.setenv("HOTEL_PIPELINE_WORK", str(tmp_path / "travail"))
    monkeypatch.setenv("HOTEL_PIPELINE_OFFLINE", "1")

    runner = CliRunner()
    runner.invoke(app, [
        "init", "essai", "--address", "1 rue Test", "--name", "Essai",
        "--country", "CA", "--timezone", "America/Toronto",
        "--ocr-language", "fr", "--lat", "45.5", "--lon", "-73.4",
    ])

    workspace = Workspace("essai")
    assert workspace.root.resolve().is_relative_to(tmp_path.resolve()), (
        "l'espace de travail doit rester dans le répertoire temporaire"
    )
    return runner, workspace


def _candidate(candidate_id="c1", **overrides):
    from hotel_pipeline.schemas.acquisition import CaptureCandidate

    fields = dict(
        candidate_id=candidate_id, source="mapillary", provider_id=candidate_id,
        camera_lat=45.5, camera_lon=-73.4,
        request_spec={"provider_id": candidate_id, "resolution": "thumb_2048"},
        available_resolutions=["thumb_256", "thumb_2048"],
    )
    fields.update(overrides)
    return CaptureCandidate(**fields)


def _write_plan(workspace, candidates, status="draft", plan_id="P1"):
    """Un plan cohérent avec ses candidats, empreintes comprises."""
    from hotel_pipeline.acquisition_request import resolve
    from hotel_pipeline.schemas.acquisition import (
        AcquisitionPlan,
        CandidateManifest,
        CaptureIntent,
        PlannedAcquisition,
        PlanStatus,
    )

    acquisitions = []
    for candidate in candidates:
        acquisition = PlannedAcquisition(
            candidate_id=candidate.candidate_id,
            intents=[CaptureIntent.BUILDING_CAPTURE],
            serves_demands=["obligation:front"],
            selection_rationale="aperçu",
            resolution="256",
        )
        request = resolve(candidate, acquisition)
        acquisitions.append(acquisition.model_copy(update={
            "provider_resolution": request.provider_resolution,
            "request_digest": request.digest,
        }))

    from hotel_pipeline.schemas.acquisition import REQUIRED_PLAN_DIGESTS

    # Un plan exécutable exige ses empreintes : la garde est légitime, et la
    # contourner ferait porter le test sur un plan qui n'existerait pas.
    digests = (
        {name: "d" * 16 for name in REQUIRED_PLAN_DIGESTS}
        if status != "draft" else {}
    )
    plan = AcquisitionPlan(
        plan_id=plan_id, hotel_id="essai", status=PlanStatus(status),
        acquisitions=acquisitions, **digests,
    )
    workspace.write_json(
        f"01_sources/acquisition_plan_{plan_id}.json",
        json.loads(plan.model_dump_json()),
    )
    workspace.write_json(
        "01_sources/candidates_20260101T000000Z.json",
        json.loads(
            CandidateManifest(hotel_id="essai", candidates=list(candidates))
            .model_dump_json()
        ),
    )
    return workspace.path("01_sources", f"acquisition_plan_{plan_id}.json")


# --- le test décisif ----------------------------------------------------------


def test_measure_never_reruns_the_selection(tmp_path, monkeypatch) -> None:
    """La sélection est rendue différente : elle ne doit pas être appelée.

    Si `measure-plan` la rejouait, il mesurerait d'autres candidats que ceux
    du plan — et le budget approuvé porterait sur des requêtes que personne
    n'a examinées.
    """
    runner, workspace = _project(tmp_path, monkeypatch)
    path = _write_plan(workspace, [_candidate("c1"), _candidate("c2")])

    from hotel_pipeline import plan as plan_module

    called: list[str] = []

    def exploding_select(*_args, **_kwargs):
        called.append("select")
        raise AssertionError("la sélection ne doit jamais être rejouée")

    def exploding_build(*_args, **_kwargs):
        called.append("build")
        raise AssertionError("le plan ne doit jamais être reconstruit")

    monkeypatch.setattr(plan_module, "select", exploding_select)
    monkeypatch.setattr(plan_module, "build", exploding_build)

    probed: list[str] = []
    monkeypatch.setattr(
        "hotel_pipeline.volumes.content_length",
        lambda request: probed.append(request.candidate_id) or 4096,
    )

    result = runner.invoke(app, ["assets", "measure-plan", "essai", "--plan", str(path)])

    assert called == [], "ni select ni build n'ont été appelés"
    assert result.exit_code == 0, result.output
    assert sorted(probed) == ["c1", "c2"], (
        "les mesures portent exactement sur les acquisitions du plan"
    )


def test_only_the_planned_acquisitions_are_probed(tmp_path, monkeypatch) -> None:
    """Le manifeste contient bien plus de candidats que le plan n'en retient."""
    runner, workspace = _project(tmp_path, monkeypatch)
    retenus = [_candidate("c1"), _candidate("c2")]
    path = _write_plan(workspace, retenus)

    # Un manifeste plus large : mesurer tout dépenserait sur ce qui ne sera
    # jamais acquis.
    from hotel_pipeline.schemas.acquisition import CandidateManifest

    workspace.write_json(
        "01_sources/candidates_20260101T000000Z.json",
        json.loads(
            CandidateManifest(
                hotel_id="essai",
                candidates=[*retenus, *[_candidate(f"x{i}") for i in range(20)]],
            ).model_dump_json()
        ),
    )

    probed: list[str] = []
    monkeypatch.setattr(
        "hotel_pipeline.volumes.content_length",
        lambda request: probed.append(request.candidate_id) or 2048,
    )

    result = runner.invoke(app, ["assets", "measure-plan", "essai", "--plan", str(path)])

    assert result.exit_code == 0, result.output
    assert len(probed) == 2, f"22 candidats au manifeste, 2 mesurés — obtenu {probed}"


# --- ce qui est refusé avant tout appel ---------------------------------------


def test_an_invalidated_plan_is_not_measured(tmp_path, monkeypatch) -> None:
    """Dépenser des appels sur un plan retiré serait payer pour rien."""
    runner, workspace = _project(tmp_path, monkeypatch)
    path = _write_plan(workspace, [_candidate("c1")])

    from hotel_pipeline.plan_invalidation import InvalidationReason, build

    event = build([path], InvalidationReason.OPERATOR_DECISION, "essai")
    workspace.write_json(
        f"01_sources/plan_invalidation_{event.invalidation_id}_committed.json",
        event.as_dict(state="committed"),
    )

    probed: list[str] = []
    monkeypatch.setattr(
        "hotel_pipeline.volumes.content_length",
        lambda request: probed.append(request.candidate_id) or 1,
    )

    result = runner.invoke(app, ["assets", "measure-plan", "essai", "--plan", str(path)])

    assert result.exit_code == 2
    assert "invalidé" in result.output
    assert probed == [], "aucun appel émis"


def test_a_non_draft_plan_is_not_measured(tmp_path, monkeypatch) -> None:
    """Un plan consenti porte déjà son volume : le remesurer le changerait."""
    runner, workspace = _project(tmp_path, monkeypatch)
    path = _write_plan(workspace, [_candidate("c1")], status="executable")

    probed: list[str] = []
    monkeypatch.setattr(
        "hotel_pipeline.volumes.content_length",
        lambda request: probed.append(request.candidate_id) or 1,
    )

    result = runner.invoke(app, ["assets", "measure-plan", "essai", "--plan", str(path)])

    assert result.exit_code == 2
    assert "seul un brouillon se mesure" in result.output
    assert probed == []


def test_a_diverging_digest_stops_before_the_first_call(tmp_path, monkeypatch) -> None:
    """Le plan décrit autre chose que ce qu'on s'apprête à mesurer.

    Mesurer d'abord pour s'en apercevoir ensuite serait payer pour rien.
    """
    runner, workspace = _project(tmp_path, monkeypatch)
    path = _write_plan(workspace, [_candidate("c1")])

    payload = json.loads(path.read_text("utf-8"))
    payload["acquisitions"][0]["request_digest"] = "0" * 16
    path.write_text(json.dumps(payload), "utf-8")

    probed: list[str] = []
    monkeypatch.setattr(
        "hotel_pipeline.volumes.content_length",
        lambda request: probed.append(request.candidate_id) or 1,
    )

    result = runner.invoke(app, ["assets", "measure-plan", "essai", "--plan", str(path)])

    assert result.exit_code == 2
    assert "différentes de celles du plan" in result.output
    assert probed == [], "aucun appel émis"


# --- ce qui est publié ---------------------------------------------------------


def test_the_original_draft_is_never_modified(tmp_path, monkeypatch) -> None:
    """Un plan qui change après examen ne se relit plus comme celui qui a été
    approuvé."""
    runner, workspace = _project(tmp_path, monkeypatch)
    path = _write_plan(workspace, [_candidate("c1")])
    before = path.read_bytes()

    monkeypatch.setattr("hotel_pipeline.volumes.content_length", lambda _r: 4096)
    result = runner.invoke(app, ["assets", "measure-plan", "essai", "--plan", str(path)])

    assert result.exit_code == 0, result.output
    assert path.read_bytes() == before, "pas un octet modifié"

    measured = sorted(
        workspace.path("01_sources").glob("acquisition_plan_P1-measured-*.json")
    )
    assert len(measured) == 1, "le plan mesuré est publié à côté"
    payload = json.loads(measured[0].read_text("utf-8"))
    assert payload["acquisitions"][0]["expected_bytes"] == 4096


def test_a_failed_measure_still_publishes_the_ledger(tmp_path, monkeypatch) -> None:
    """Une mesure interrompue a coûté des appels : les taire donnerait à croire
    qu'elle n'a rien consommé."""
    runner, workspace = _project(tmp_path, monkeypatch)
    path = _write_plan(workspace, [_candidate("c1")])

    def exploding(_request):
        raise KeyboardInterrupt("interruption au milieu")

    monkeypatch.setattr("hotel_pipeline.volumes.content_length", exploding)

    runner.invoke(app, ["assets", "measure-plan", "essai", "--plan", str(path)])

    receipts = sorted(workspace.path("01_sources").glob("volume_measure_P1_*.json"))
    assert len(receipts) == 1, "le registre est publié même sur échec"
    payload = json.loads(receipts[0].read_text("utf-8"))
    assert "transport" in payload
    assert payload["volumes"] is None, "aucune mesure aboutie"
    assert path.read_bytes(), "le brouillon reste intact"


def test_logical_operations_and_http_exchanges_are_published(
    tmp_path, monkeypatch
) -> None:
    """Les deux comptes séparément : une redirection les fait diverger."""
    runner, workspace = _project(tmp_path, monkeypatch)
    path = _write_plan(workspace, [_candidate("c1"), _candidate("c2")])

    monkeypatch.setattr("hotel_pipeline.volumes.content_length", lambda _r: 1024)
    runner.invoke(app, ["assets", "measure-plan", "essai", "--plan", str(path)])

    receipt = sorted(workspace.path("01_sources").glob("volume_measure_P1_*.json"))[0]
    transport = json.loads(receipt.read_text("utf-8"))["transport"]

    assert "actual_logical_operations" in transport
    assert "actual_http_exchanges" in transport
