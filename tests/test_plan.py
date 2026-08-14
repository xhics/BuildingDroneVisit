"""Sélection des acquisitions (collecte V2, étape 2).

Ce qui est éprouvé : un candidat s'évalue **par besoin**, un volume inconnu ne
vaut pas zéro, et rien ne devient exécutable sans consentement ni empreintes.
"""

from __future__ import annotations

import pytest

from hotel_pipeline.plan import PlanRefused, build, consent, evaluate, select
from hotel_pipeline.schemas.acquisition import (
    REQUIRED_PLAN_DIGESTS,
    CandidateGeometry,
    CaptureCandidate,
    CaptureDemand,
    CaptureIntent,
    Eligibility,
    PlanStatus,
    TargetKind,
    VolumeStatus,
)

DIGESTS = {name: f"{name[:4]}0" for name in REQUIRED_PLAN_DIGESTS}


def candidate(candidate_id: str = "c1", **overrides) -> CaptureCandidate:
    fields = dict(
        candidate_id=candidate_id, source="mapillary", provider_id=candidate_id,
        camera_lat=45.573, camera_lon=-73.443,
    )
    fields.update(overrides)
    return CaptureCandidate(**fields)


def demand(demand_id: str = "d1", **overrides) -> CaptureDemand:
    fields = dict(
        demand_id=demand_id, intent=CaptureIntent.BUILDING_CAPTURE,
        target_kind=TargetKind.VIEW_SECTOR, target_ref="front",
    )
    fields.update(overrides)
    return CaptureDemand(**fields)


# --- un candidat s'évalue par besoin ------------------------------------------


def test_one_candidate_gets_one_verdict_per_demand() -> None:
    """Une vue cadre mal la façade et documente bien la voie d'accès."""
    facade = demand("facade", min_projected_width_fraction=0.5)
    access = demand(
        "acces", intent=CaptureIntent.CONTEXT_CAPTURE,
        target_kind=TargetKind.CONTEXT_CORRIDOR, target_ref="way/1",
    )
    geometry = CandidateGeometry(unclipped_width_fraction=0.1)

    plan, evaluations, _ = build(
        "h", [candidate()], [facade, access], DIGESTS,
        geometries={("c1", "facade"): geometry, ("c1", "acces"): geometry},
    )

    verdicts = {e.demand_id: e.eligibility for e in evaluations}
    assert verdicts["facade"] is Eligibility.REJECTED
    assert verdicts["acces"] is Eligibility.ELIGIBLE
    # L'image est retenue, mais seulement pour ce qu'elle sert.
    assert plan.acquisitions[0].serves_demands == ["acces"]


def test_a_rejection_always_carries_its_reason() -> None:
    """Un candidat écarté sans motif n'apprend rien à la recherche suivante."""
    result = evaluate(
        candidate(), demand(min_projected_width_fraction=0.5),
        CandidateGeometry(unclipped_width_fraction=0.05),
    )

    assert result.eligibility is Eligibility.REJECTED
    assert "taille projetée espérée" in result.rejection_reason


def test_a_candidate_without_a_position_is_rejected_not_guessed() -> None:
    result = evaluate(candidate(camera_lat=None, camera_lon=None), demand())

    assert result.eligibility is Eligibility.REJECTED
    assert "position de caméra inconnue" in result.rejection_reason


def test_the_unmeasured_asks_for_a_preview_rather_than_a_bet() -> None:
    """Sans mesure de cadrage, engager la pleine résolution serait parier."""
    result = evaluate(candidate(), demand(min_projected_width_fraction=0.3))

    assert result.eligibility is Eligibility.PREVIEW_REQUIRED
    assert result.rejection_reason is None


def test_a_demand_that_asks_nothing_accepts_the_unmeasured() -> None:
    assert evaluate(candidate(), demand()).eligibility is Eligibility.ELIGIBLE


def test_an_established_candidate_outranks_one_still_to_be_seen() -> None:
    measured = CandidateGeometry(unclipped_width_fraction=0.9)
    wanted = demand(min_projected_width_fraction=0.3)

    evaluations = [
        evaluate(candidate("c-inconnu"), wanted),
        evaluate(candidate("c-mesure"), wanted, measured),
    ]
    planned = select(evaluations, [wanted])

    assert [a.candidate_id for a in planned] == ["c-mesure"]


# --- le volume connu et le volume inconnu -------------------------------------


def test_an_unknown_size_is_never_counted_as_zero() -> None:
    """Additionner comme nulles les tailles ignorées annoncerait un faux total."""
    plan, _, report = build(
        "h",
        [candidate("c1"), candidate("c2", camera_lat=45.5745, camera_lon=-73.4445)],
        [demand(viewpoints_required=2)], DIGESTS, sizes={"c1": 1000},
    )

    assert plan.known_bytes == 1000
    assert plan.unknown_size_items == ["c2"]
    assert plan.volume_status is VolumeStatus.PARTIAL
    assert report.as_dict()["volume"]["unknown_size_items"] == 1


def test_a_fully_announced_plan_states_an_exact_volume() -> None:
    plan, _, _ = build(
        "h",
        [candidate("c1"), candidate("c2", camera_lat=45.5745, camera_lon=-73.4445)],
        [demand(viewpoints_required=2)], DIGESTS, sizes={"c1": 1000, "c2": 2000},
    )

    assert plan.volume_status is VolumeStatus.EXACT
    assert plan.known_bytes == 3000


def test_a_plan_that_knows_nothing_says_unknown_not_estimated() -> None:
    """« Estimé » laisserait croire qu'un calcul a eu lieu."""
    plan, _, report = build("h", [candidate()], [demand()], DIGESTS)

    assert plan.volume_status is VolumeStatus.UNKNOWN
    assert report.volume_status == "unknown"
    assert "estim" not in report.as_dict()["volume"]["note"]


def test_planning_downloads_nothing() -> None:
    _, _, report = build("h", [candidate()], [demand()], DIGESTS)

    assert report.as_dict()["bytes_downloaded"] == 0


# --- le besoin précède, et son absence se lit ---------------------------------


def test_a_plan_without_a_demand_is_refused() -> None:
    with pytest.raises(PlanRefused, match="aucun besoin"):
        build("h", [candidate()], [], DIGESTS)


def test_a_plan_without_a_candidate_names_the_missing_step() -> None:
    with pytest.raises(PlanRefused, match="assets discover"):
        build("h", [], [demand()], DIGESTS)


def test_an_unplanned_demand_is_reported_rather_than_hidden() -> None:
    """« Aucune vue de l'arrière » doit se distinguer de « pas cherché »."""
    front = demand("front")
    rear = demand("rear", target_ref="rear", min_projected_width_fraction=0.9)

    _, _, report = build(
        "h", [candidate()], [front, rear], DIGESTS,
        geometries={("c1", "rear"): CandidateGeometry(unclipped_width_fraction=0.1)},
    )

    assert report.demands_planned == {"front": 1}
    assert report.demands_unplanned == ["rear"]


def test_the_count_is_of_viewpoints_not_of_files() -> None:
    """Cinq cadrages d'une même position ne font pas cinq observations.

    Les compter séparément ferait croire un besoin servi par plusieurs vues
    indépendantes alors qu'il n'y en a qu'une, et un SfM n'en tirerait aucune
    parallaxe.
    """
    wanted = demand(viewpoints_required=2)
    same_place = [candidate(f"c{i}") for i in range(5)]

    plan, _, _ = build("h", same_place, [wanted], DIGESTS)

    assert len(plan.acquisitions) == 1


def test_distinct_positions_do_count_separately() -> None:
    wanted = demand(viewpoints_required=2)
    spread = [
        candidate("c0", camera_lat=45.5730, camera_lon=-73.4430),
        candidate("c1", camera_lat=45.5740, camera_lon=-73.4440),
        candidate("c2", camera_lat=45.5750, camera_lon=-73.4450),
    ]

    plan, _, _ = build("h", spread, [wanted], DIGESTS)

    assert len(plan.acquisitions) == 2


def test_two_framings_of_one_panorama_are_one_viewpoint() -> None:
    """Le cas d'acceptation de Street View, avant même son résolveur."""
    from hotel_pipeline.plan import viewpoint_of

    first = candidate("sv-1", panorama_id="pano-A", requested_heading_deg=0.0)
    second = candidate("sv-2", panorama_id="pano-A", requested_heading_deg=180.0)
    elsewhere = candidate("sv-3", panorama_id="pano-B")

    assert viewpoint_of(first) == viewpoint_of(second)
    assert viewpoint_of(first) != viewpoint_of(elsewhere)


# --- rien ne s'acquiert sans consentement ni empreintes -----------------------


def test_a_plan_is_born_a_draft() -> None:
    """Un brouillon existe pour être discuté ; il ne s'acquiert jamais."""
    plan, _, _ = build("h", [candidate()], [demand()], DIGESTS)

    assert plan.status is PlanStatus.DRAFT


def test_consent_turns_a_draft_into_an_executable_plan() -> None:
    plan, _, _ = build("h", [candidate()], [demand()], DIGESTS)

    executable = consent(plan, DIGESTS)

    assert executable.status is PlanStatus.EXECUTABLE
    assert executable.missing_digests() == []


def test_consent_is_refused_when_an_imprint_is_missing() -> None:
    """Un plan qu'on ne peut pas rattacher à un état aurait choisi ses images
    pour un autre."""
    partial = dict(DIGESTS)
    partial["corpus_digest"] = None
    plan, _, _ = build("h", [candidate()], [demand()], partial)

    with pytest.raises(PlanRefused, match="corpus_digest"):
        consent(plan, partial)


def test_consent_revalidates_rather_than_relabelling() -> None:
    """`model_copy` ne rejoue pas les validateurs : un plan vide passerait."""
    plan, _, _ = build("h", [candidate()], [demand()], DIGESTS)
    emptied = plan.model_copy(update={"acquisitions": []})

    with pytest.raises(ValueError, match="exécutable et vide"):
        consent(emptied, DIGESTS)


def test_the_module_carries_every_imprint_the_schema_requires() -> None:
    """Un champ ajouté au schéma ne doit pas rester non transmis."""
    from hotel_pipeline.plan import _PLAN_DIGEST_FIELDS

    assert set(REQUIRED_PLAN_DIGESTS) == set(_PLAN_DIGEST_FIELDS)


# --- une acquisition peut servir plusieurs besoins ----------------------------


def test_one_acquisition_may_serve_several_demands() -> None:
    """Ne porter qu'une intention obligeait à taire l'autre."""
    facade = demand("facade")
    access = demand(
        "acces", intent=CaptureIntent.CONTEXT_CAPTURE,
        target_kind=TargetKind.CONTEXT_CORRIDOR, target_ref="way/1",
    )

    plan, _, _ = build("h", [candidate()], [facade, access], DIGESTS)

    acquisition = plan.acquisitions[0]
    assert acquisition.serves_demands == ["acces", "facade"]
    assert set(acquisition.intents) == {
        CaptureIntent.BUILDING_CAPTURE, CaptureIntent.CONTEXT_CAPTURE
    }
    assert acquisition.primary_intent in acquisition.intents


def test_every_acquisition_names_the_demands_it_serves() -> None:
    """Télécharger sans savoir quel besoin on sert, c'est justifier après."""
    plan, _, _ = build("h", [candidate()], [demand()], DIGESTS)

    assert all(a.serves_demands for a in plan.acquisitions)
    assert all(a.selection_rationale.strip() for a in plan.acquisitions)


# --- le consentement, de bout en bout -----------------------------------------


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Un projet portant besoins et candidats, sans aucune image."""
    import json

    from typer.testing import CliRunner

    from hotel_pipeline.cli import app
    from hotel_pipeline.schemas.acquisition import CandidateManifest, CaptureDemandManifest
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
    workspace.write_json(
        "01_sources/capture_demands.json",
        json.loads(
            CaptureDemandManifest(hotel_id="hotel-test", demands=[demand()])
            .model_dump_json()
        ),
    )
    workspace.write_json(
        "01_sources/candidates_20260814T000000000000Z.json",
        json.loads(
            CandidateManifest(hotel_id="hotel-test", candidates=[candidate()])
            .model_dump_json()
        ),
    )
    return runner, workspace, tmp_path


def test_the_cli_plan_downloads_nothing_and_stays_a_draft(project) -> None:
    import json

    from hotel_pipeline.cli import app

    runner, workspace, _ = project

    result = runner.invoke(app, ["assets", "plan", "hotel-test"])

    assert result.exit_code == 0, result.output
    assert "brouillon" in result.output
    written = sorted(workspace.path("01_sources").glob("acquisition_plan_*.json"))
    assert len(written) == 1
    assert json.loads(written[0].read_text("utf-8"))["status"] == "draft"

    images = workspace.path("02_images")
    assert not images.exists() or not any(images.rglob("*.jpg"))


def test_consenting_to_an_unknown_volume_is_refused(project) -> None:
    """Le cas dangereux : consentir à un total dont une part est invisible."""
    from hotel_pipeline.cli import app

    runner, _, _ = project

    result = runner.invoke(
        app, ["assets", "plan", "hotel-test", "--consent-bytes", "0"]
    )

    assert result.exit_code == 2
    assert "consentement refusé" in result.output
    assert "n'a pas été montré" in result.output


def test_a_draft_names_the_exact_command_that_would_execute_it(project) -> None:
    from hotel_pipeline.cli import app

    runner, _, _ = project
    result = runner.invoke(app, ["assets", "plan", "hotel-test"])

    assert "--consent-bytes" in result.output
