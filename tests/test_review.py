"""Revue humaine de visibilité (Lot 1B §6).

Ce qui est éprouvé ici tient en une phrase : une décision prise par une
personne ne doit pouvoir être ni effacée, ni recalculée, ni prise à la légère.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hotel_pipeline import review
from hotel_pipeline.classify_cascade import classify
from hotel_pipeline.roles import role_for
from hotel_pipeline.schemas import (
    Asset,
    ClusterRole,
    PipelinePolicy,
    ReviewDecision,
    ReviewStatus,
    Subject,
    TemporalStatus,
    ViewSector,
)


def policy() -> PipelinePolicy:
    return PipelinePolicy()


def image(tmp_path, name: str = "vue.jpg", content: bytes = b"jpeg-donnees"):
    path = tmp_path / name
    path.write_bytes(content)
    return path, hashlib.sha256(content).hexdigest()


def asset(tmp_path, **overrides) -> Asset:
    """Un asset Mapillary porteur : cap mesuré, cible visible, non masqué."""
    path, digest = image(tmp_path, overrides.pop("name", "vue.jpg"))
    fields = dict(
        id="mapillary-1",
        source="mapillary",
        source_url_or_id="1",
        rights="open_data",
        ai_eligible=False,
        confidence=0.9,
        category="facade",
        checksum=digest,
        local_path=str(path),
        camera_lat=45.5730,
        camera_lon=-73.4433,
        heading_deg=45.0,
        heading_is_measured=True,
        sees_building=True,
        contains_building=True,
        target_building_visible=True,
        subjects=[Subject.BUILDING],
        view_sector=ViewSector.FRONT,
        cluster_role=ClusterRole.CANONICAL,
        temporal_status=TemporalStatus.AFTER_EVENT,
        review_status=ReviewStatus.NEEDS_REVIEW,
        subject_scores={"building": 0.5537, "parking": 0.9997, "sign": 0.1273},
    )
    fields.update(overrides)
    return Asset(**fields)


def decide(assets, asset_id="mapillary-1", decision=ReviewDecision.CONFIRMED, **kwargs):
    fields = dict(
        by="Hicham",
        rationale="façade du WelcomINNS reconnue, enseigne lisible",
        evidence=["enseigne visible en haut à gauche"],
    )
    fields.update(kwargs)
    return review.decide(assets, asset_id, decision, **fields)


# --- la décision survit à la reclassification -------------------------------


def test_a_decision_survives_reclassification(tmp_path) -> None:
    """Le défaut : `review_status` était recalculé après coup.

    Le verdict de visibilité survivait bien, mais `human_accepted` retombait en
    `automatic_accepted` — une acceptation automatique se substituait donc à un
    verdict humain, sans que rien ne le dise.
    """
    assets = [asset(tmp_path)]
    decide(assets)
    assert assets[0].review_status is ReviewStatus.HUMAN_ACCEPTED

    classify(assets, classifier=None, policy=policy())

    assert assets[0].target_visibility_decision is ReviewDecision.CONFIRMED
    assert assets[0].review_status is ReviewStatus.HUMAN_ACCEPTED
    assert assets[0].target_building_visible is True
    assert "revue humaine" in assets[0].target_evidence


def test_a_rejection_survives_reclassification(tmp_path) -> None:
    assets = [asset(tmp_path)]
    decide(assets, decision=ReviewDecision.REJECTED, rationale="c'est le Toyota voisin")

    classify(assets, classifier=None, policy=policy())

    assert assets[0].review_status is ReviewStatus.REJECTED
    assert assets[0].target_building_visible is False


def test_a_reviewed_unresolved_is_not_promoted_automatically(tmp_path) -> None:
    """Examiné sans conclure n'est pas la même chose que jamais examiné."""
    assets = [asset(tmp_path)]
    decide(assets, decision=ReviewDecision.UNRESOLVED, rationale="trop sombre pour trancher")

    classify(assets, classifier=None, policy=policy())

    assert assets[0].review_status is ReviewStatus.NEEDS_REVIEW
    assert assets[0].has_been_reviewed


def test_a_never_reviewed_asset_can_still_be_accepted_automatically(tmp_path) -> None:
    """`target_visibility_decision` vaut `unresolved` par défaut.

    S'y fier seul ferait passer chaque image jamais examinée pour une revue
    sans conclusion, et plus aucune acceptation automatique n'existerait.
    """
    assets = [asset(tmp_path)]
    assert not assets[0].has_been_reviewed

    classify(assets, classifier=None, policy=policy())

    assert assets[0].review_status is ReviewStatus.AUTOMATIC_ACCEPTED


# --- effet sur les rôles ----------------------------------------------------


def test_a_rejection_removes_a_geometry_carrier(tmp_path) -> None:
    assets = [asset(tmp_path)]
    decide(assets)  # confirmé : l'asset devient porteur
    assert role_for(assets[0], policy())[0].value == "photo_geometry"

    decide(assets, decision=ReviewDecision.REJECTED, rationale="bâtiment voisin")

    role, reason = role_for(assets[0], policy())
    assert role.value == "context_lock"
    # La cible n'est plus tenue pour visible : le motif le dit, et l'asset
    # n'est pas jeté pour autant — il reste un verrou de contexte utile.
    assert reason == "bâtiment cible non établi"


def test_a_confirmation_alone_does_not_create_a_carrier(tmp_path) -> None:
    """Confirmer la visibilité ne dispense d'aucun autre prédicat.

    Une vue confirmée mais non arbitrée par la déduplication reste un verrou de
    contexte : la revue tranche le contenu, pas la position ni la redondance.
    """
    assets = [asset(tmp_path, cluster_role=ClusterRole.INACTIVE)]

    decide(assets)

    role, reason = role_for(assets[0], policy())
    assert role.value == "context_lock"
    assert reason == "point de vue non arbitré ou déjà couvert"


def test_a_confirmation_creates_a_carrier_when_all_else_passes(tmp_path) -> None:
    assets = [asset(tmp_path, target_building_visible=None)]
    assert role_for(assets[0], policy())[0].value == "context_lock"

    decide(assets)

    role, reason = role_for(assets[0], policy())
    assert role.value == "photo_geometry"
    assert reason == "cible visible, située et arbitrée"


def test_an_occluded_view_stays_a_context_lock_even_if_confirmed(tmp_path) -> None:
    """L'occlusion est géométrique : aucun verdict humain ne la lève ici."""
    assets = [asset(tmp_path, occluded_by="way/999")]

    decide(assets)

    assert role_for(assets[0], policy())[0].value == "context_lock"


# --- contrôles avant mutation -----------------------------------------------


def test_an_altered_image_is_refused_without_mutation(tmp_path) -> None:
    """La décision porte sur ce qui a été vu, pas sur ce que le fichier est devenu."""
    assets = [asset(tmp_path)]
    before = assets[0].model_dump_json()
    (tmp_path / "vue.jpg").write_bytes(b"une-autre-image")

    with pytest.raises(review.ReviewRefused, match="l'image a changé"):
        decide(assets)

    assert assets[0].model_dump_json() == before


def test_a_missing_file_is_refused(tmp_path) -> None:
    assets = [asset(tmp_path)]
    (tmp_path / "vue.jpg").unlink()

    with pytest.raises(review.ReviewRefused, match="introuvable"):
        decide(assets)


def test_a_missing_rationale_is_refused(tmp_path) -> None:
    assets = [asset(tmp_path)]

    with pytest.raises(review.ReviewRefused, match="sans justification"):
        decide(assets, rationale="   ")

    assert not assets[0].has_been_reviewed


def test_a_missing_author_is_refused(tmp_path) -> None:
    assets = [asset(tmp_path)]

    with pytest.raises(review.ReviewRefused, match="sans auteur"):
        decide(assets, by="")


def test_an_unknown_asset_is_refused(tmp_path) -> None:
    with pytest.raises(review.ReviewRefused, match="asset inconnu"):
        decide([asset(tmp_path)], asset_id="absent")


# --- historique append-only -------------------------------------------------


def test_a_correction_keeps_the_previous_decision(tmp_path) -> None:
    """Les champs plats ne gardent que la dernière décision.

    Sans historique, corriger une revue effacerait ce qu'elle corrige : on ne
    saurait plus ni ce qui avait été conclu, ni par qui, ni sur quelle preuve.
    """
    assets = [asset(tmp_path)]
    decide(assets, by="Hicham", rationale="façade reconnue")
    decide(
        assets,
        decision=ReviewDecision.REJECTED,
        by="Hicham",
        rationale="erreur : c'est l'immeuble voisin",
    )

    history = assets[0].review_history
    assert len(history) == 2
    assert history[0].decision is ReviewDecision.CONFIRMED
    assert history[0].rationale == "façade reconnue"
    assert history[1].decision is ReviewDecision.REJECTED
    assert history[1].supersedes_index == 0

    # Les champs plats et la dernière entrée disent la même chose.
    assert assets[0].target_visibility_decision is ReviewDecision.REJECTED
    assert assets[0].review_status is ReviewStatus.REJECTED
    assert assets[0].review_rationale == history[1].rationale


def test_the_history_records_what_was_actually_seen(tmp_path) -> None:
    assets = [asset(tmp_path)]
    decide(assets)

    entry = assets[0].review_history[0]
    assert entry.reviewed_checksum == assets[0].checksum
    assert entry.decided_by == "Hicham"
    assert entry.evidence


# --- les trois populations --------------------------------------------------


def test_the_three_populations_are_counted_separately(tmp_path) -> None:
    """Un bloquant est en attente ; un membre de la cohorte peut ne pas l'être.

    Les additionner ferait de « revue terminée » une phrase sans contenu.
    """
    blocking_asset = asset(tmp_path, name="a.jpg")
    # Mapillary, bâtiment présent, mais déjà accepté automatiquement : dans la
    # cohorte de validation, hors de la file d'attente.
    accepted = asset(
        tmp_path, name="b.jpg", id="mapillary-2",
        review_status=ReviewStatus.AUTOMATIC_ACCEPTED,
    )
    # Un second verrou : la grappe n'est pas arbitrée. L'asset attend une revue
    # sans être bloquant — la confirmer ne le rendrait pas porteur.
    second_lock = asset(
        tmp_path, name="e.jpg", id="mapillary-4", cluster_role=ClusterRole.INACTIVE,
    )
    # En attente et bloquant lui aussi : la cible n'est pas établie, mais c'est
    # précisément ce qu'une revue établit — tout le reste est déjà satisfait.
    waiting = asset(
        tmp_path, name="c.jpg", id="street_view-1", source="street_view",
        target_building_visible=None, contains_building=False, subjects=[],
    )
    # Mapillary sans bâtiment : hors cohorte.
    scenery = asset(
        tmp_path, name="d.jpg", id="mapillary-3",
        contains_building=False, subjects=[],
        review_status=ReviewStatus.AUTOMATIC_ACCEPTED,
    )

    numbers = review.counts(
        [blocking_asset, accepted, waiting, scenery, second_lock], policy()
    )

    assert numbers.pending == 3
    # Deux des trois en attente sont débloquables ; le troisième est retenu par
    # un second verrou — sa grappe n'est pas arbitrée — qu'aucune revue ne lève.
    assert numbers.blocking == 2
    assert numbers.cohort == 3
    assert numbers.pending_by_source == {"mapillary": 2, "street_view": 1}


def test_the_queue_carries_what_is_needed_to_judge(tmp_path) -> None:
    assets = [asset(tmp_path)]
    queue = review.build_queue(assets, "blocking", policy())
    item = queue.items[0].as_dict()

    assert item["checksum"] == assets[0].checksum
    assert item["local_path"]
    assert item["role_reason"] == "en attente de revue humaine"
    assert item["heading_is_measured"] is True
    assert item["previous_reviews"] == 0
    # Les trois nombres accompagnent toujours la file.
    assert set(queue.as_dict()["counts"]) >= {"pending", "blocking", "cohort"}


def test_an_unknown_queue_is_refused(tmp_path) -> None:
    with pytest.raises(review.ReviewRefused, match="file inconnue"):
        review.build_queue([asset(tmp_path)], "toutes", policy())


def test_the_board_is_self_contained_html(tmp_path) -> None:
    queue = review.build_queue([asset(tmp_path)], "blocking", policy())
    html = review.to_html(queue)

    assert html.startswith("<!doctype html>")
    assert "mapillary-1" in html
    assert "Ne pas additionner" in html
    # Les scores du modèle figurent à côté de l'image, pour juger sur pièce.
    assert "building" in html


# --- pas d'acceptation en masse ---------------------------------------------


def test_no_bulk_acceptance_exists() -> None:
    """Une décision non regardée n'est pas une décision.

    Le module n'expose aucun point d'entrée acceptant plusieurs assets : la
    seule voie est `decide`, un asset à la fois, avec auteur et motif.
    """
    from typer.testing import CliRunner

    from hotel_pipeline.cli import app

    exported = [name for name in dir(review) if name.startswith(("accept", "bulk", "approve"))]
    assert exported == []

    output = CliRunner().invoke(app, ["assets", "review", "--help"]).output
    assert "queue" in output and "set" in output
    assert "all" not in output.split("Commands")[-1]


# --- câblage de la commande -------------------------------------------------


def cli_workspace(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from hotel_pipeline.cli import app
    from hotel_pipeline.schemas import AssetManifest
    from hotel_pipeline.workspace import Workspace

    monkeypatch.setenv("HOTEL_PIPELINE_WORK", str(tmp_path / "work"))
    monkeypatch.setenv("HOTEL_PIPELINE_PROFILES", str(tmp_path / "profiles"))
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    runner.invoke(app, ["init", "hotel-test", "--address", "1 rue Test"])

    workspace = Workspace("hotel-test")
    subject = asset(tmp_path, target_building_visible=None)
    workspace.write_assets(AssetManifest(hotel_id="hotel-test", assets=[subject]))
    return runner, workspace, subject


def test_cli_records_a_decision_and_reports_the_impact(tmp_path, monkeypatch) -> None:
    from hotel_pipeline.cli import app

    runner, workspace, _ = cli_workspace(tmp_path, monkeypatch)

    result = runner.invoke(app, [
        "assets", "review", "set", "hotel-test", "mapillary-1",
        "--decision", "confirmed", "--by", "Hicham",
        "--rationale", "façade et enseigne du WelcomINNS",
        "--evidence", "enseigne lisible",
    ])

    assert result.exit_code == 0, result.output
    assert "context_lock" in result.output and "photo_geometry" in result.output

    stored = workspace.read_assets().assets[0]
    assert stored.review_status is ReviewStatus.HUMAN_ACCEPTED
    assert len(stored.review_history) == 1

    reports = list(workspace.path("01_sources").glob("review_decision_*.json"))
    assert len(reports) == 1
    impact = json.loads(reports[0].read_text("utf-8"))
    assert impact["corpus_roles"]["before"] != impact["corpus_roles"]["after"]
    assert impact["asset_role"]["after"].startswith("photo_geometry")
    # Le rapport porte sa provenance, comme tout rapport du pipeline.
    assert impact["provenance"]["policy_version"]


def test_cli_refuses_an_altered_image_without_touching_the_manifest(
    tmp_path, monkeypatch
) -> None:
    from hotel_pipeline.cli import app

    runner, workspace, subject = cli_workspace(tmp_path, monkeypatch)
    before = workspace.assets_path.read_text("utf-8")
    Path(subject.local_path).write_bytes(b"image-remplacee")

    result = runner.invoke(app, [
        "assets", "review", "set", "hotel-test", "mapillary-1",
        "--decision", "confirmed", "--by", "Hicham", "--rationale", "peu importe",
        "--evidence", "capture annotée",
    ])

    assert result.exit_code == 1
    assert workspace.assets_path.read_text("utf-8") == before
    assert not list(workspace.path("01_sources").glob("review_decision_*.json"))


def test_cli_queue_writes_both_the_json_and_the_board(tmp_path, monkeypatch) -> None:
    from hotel_pipeline.cli import app

    runner, workspace, _ = cli_workspace(tmp_path, monkeypatch)

    result = runner.invoke(app, ["assets", "review", "queue", "hotel-test",
                                 "--queue", "pending"])

    assert result.exit_code == 0, result.output
    assert "trois populations distinctes" in result.output
    assert len(list(workspace.path("01_sources").glob("review_queue_pending_*.json"))) == 1
    assert len(list(workspace.path("01_sources").glob("review_board_pending_*.html"))) == 1


# --- l'historique est append-only par le schéma -----------------------------


def entry(**overrides):
    from hotel_pipeline.schemas import ReviewEntry

    fields = dict(
        decision=ReviewDecision.CONFIRMED,
        decided_by="Hicham",
        rationale="façade reconnue",
        evidence=["enseigne lisible"],
        reviewed_checksum="a" * 64,
    )
    fields.update(overrides)
    return ReviewEntry(**fields)


def reviewed_asset(tmp_path, history, **overrides):
    """Un asset dont les champs plats suivent la dernière entrée."""
    from hotel_pipeline.schemas.assets import DECISION_STATUS, VISIBILITY_OF

    last = history[-1]
    fields = dict(
        review_history=history,
        target_visibility_decision=last.decision,
        review_status=DECISION_STATUS[last.decision],
        target_building_visible=VISIBILITY_OF[last.decision],
        reviewer=last.decided_by,
        review_rationale=last.rationale,
        review_evidence=last.evidence,
    )
    fields.update(overrides)
    return asset(tmp_path, **fields)


def test_an_entry_without_evidence_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 1 item"):
        entry(evidence=[])
    with pytest.raises(ValueError, match="preuve"):
        entry(evidence=["   "])


def test_an_entry_without_reviewed_checksum_is_refused() -> None:
    """Une décision qui ne dit pas sur quoi elle portait ne s'oppose à rien."""
    from hotel_pipeline.schemas import ReviewEntry

    with pytest.raises(ValueError):
        ReviewEntry(
            decision=ReviewDecision.CONFIRMED, decided_by="H",
            rationale="r", evidence=["e"],
        )


def test_the_first_entry_cannot_supersede_anything(tmp_path) -> None:
    with pytest.raises(ValueError, match="première revue ne corrige rien"):
        reviewed_asset(tmp_path, [entry(supersedes_index=0)])


def test_a_later_entry_must_say_what_it_corrects(tmp_path) -> None:
    with pytest.raises(ValueError, match="ne dit pas laquelle elle corrige"):
        reviewed_asset(tmp_path, [entry(), entry(decision=ReviewDecision.REJECTED,
                                                 rationale="erreur")])


def test_a_correction_cannot_point_forward(tmp_path) -> None:
    with pytest.raises(ValueError, match="ne lui est pas antérieure"):
        reviewed_asset(
            tmp_path,
            [entry(), entry(decision=ReviewDecision.REJECTED, rationale="erreur",
                            supersedes_index=1)],
        )


def test_flat_fields_must_follow_the_last_entry(tmp_path) -> None:
    from hotel_pipeline.schemas.assets import DECISION_STATUS

    with pytest.raises(ValueError, match="décision courante"):
        reviewed_asset(tmp_path, [entry()],
                       target_visibility_decision=ReviewDecision.REJECTED,
                       review_status=DECISION_STATUS[ReviewDecision.REJECTED],
                       target_building_visible=False)

    with pytest.raises(ValueError, match="statut .* incompatible"):
        reviewed_asset(tmp_path, [entry()], review_status=ReviewStatus.NEEDS_REVIEW)

    with pytest.raises(ValueError, match="visibilité .* incompatible"):
        reviewed_asset(tmp_path, [entry()], target_building_visible=None)

    with pytest.raises(ValueError, match="auteur ou motif courant divergent"):
        reviewed_asset(tmp_path, [entry()], reviewer="quelqu'un d'autre")


def test_a_human_status_without_history_is_refused(tmp_path) -> None:
    """Le cas que la revue doit rendre impossible : un verdict sans trace."""
    with pytest.raises(ValueError, match="sans aucune revue à l'appui"):
        asset(tmp_path, review_status=ReviewStatus.HUMAN_ACCEPTED)

    with pytest.raises(ValueError, match="sans aucune revue"):
        asset(tmp_path, target_visibility_decision=ReviewDecision.REJECTED,
              target_building_visible=False, review_status=ReviewStatus.REJECTED)


def test_decide_revalidates_before_mutating(tmp_path, monkeypatch) -> None:
    """`model_copy(update=...)` ne revalide pas : la revue doit le faire.

    Sans cette revalidation, la seule voie qui met les invariants en jeu serait
    aussi la seule à pouvoir les contourner.
    """
    assets = [asset(tmp_path)]
    before = assets[0].model_dump_json()

    # Une correspondance décision → statut volontairement fausse, posée dans le
    # seul espace de noms de la revue : le schéma, lui, garde la bonne.
    monkeypatch.setattr(
        review, "DECISION_STATUS",
        {**review.DECISION_STATUS, ReviewDecision.CONFIRMED: ReviewStatus.AUTOMATIC_ACCEPTED},
    )

    with pytest.raises(review.ReviewRefused, match="incohérente avec le manifeste"):
        decide(assets)

    assert assets[0].model_dump_json() == before


def test_a_decision_without_evidence_is_refused(tmp_path) -> None:
    assets = [asset(tmp_path)]

    with pytest.raises(review.ReviewRefused, match="sans preuve"):
        decide(assets, evidence=[])
    with pytest.raises(review.ReviewRefused, match="sans preuve"):
        decide(assets, evidence=["  ", ""])

    assert not assets[0].has_been_reviewed


# --- identité d'une file ----------------------------------------------------


def test_two_queues_of_the_same_second_do_not_collide(tmp_path) -> None:
    """L'horodatage à la seconde ne suffit pas à distinguer deux exécutions."""
    first = review.build_queue([asset(tmp_path, name="a.jpg")], "pending", policy())
    second = review.build_queue(
        [asset(tmp_path, name="b.jpg", id="mapillary-2")], "pending", policy()
    )

    assert first.slug != second.slug
    assert first.manifest_digest != second.manifest_digest


def test_the_queue_records_the_state_it_describes(tmp_path) -> None:
    """Une file périmée doit pouvoir être reconnue comme telle.

    La première file réelle annonçait une cohorte de 189 quand le code en
    calculait 25, et rien sur le fichier ne permettait de le voir.
    """
    assets = [asset(tmp_path)]
    queue = review.build_queue(assets, "blocking", policy())

    assert queue.manifest_digest == review.manifest_digest(assets)
    assert queue.as_dict()["manifest_digest"] == queue.manifest_digest

    decide(assets)
    assert review.manifest_digest(assets) != queue.manifest_digest
