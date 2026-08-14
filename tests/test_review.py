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
    """Un asset Mapillary porteur : cap mesuré, cible visible, non masqué.

    Son aptitude géométrique est établie, sauf mention contraire : sans elle,
    aucun rôle porteur n'est possible, et ces cas-là portent sur la visibilité.
    """
    path, digest = image(tmp_path, overrides.pop("name", "vue.jpg"))
    suitability = overrides.pop("suitability", "primary")
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
    if suitability:
        from hotel_pipeline.review import assessment_fields
        from hotel_pipeline.schemas import GeometrySuitability

        fields.update(
            assessment_fields(
                GeometrySuitability(suitability), "hm",
                "façade cadrée, lignes exploitables",
                ["contrôle du cadrage et de la netteté sur la cible"], digest,
            )
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
    # Le nom porte le mode : une planche d'analyse et une planche aveugle du
    # même état ne doivent pas se confondre.
    assert len(list(workspace.path("01_sources").glob("review_queue_analysis_*.json"))) == 1
    assert len(list(workspace.path("01_sources").glob("review_board_analysis_*.html"))) == 1


def test_cli_blind_mode_writes_a_separate_redacted_board(tmp_path, monkeypatch) -> None:
    from hotel_pipeline.cli import app

    runner, workspace, _ = cli_workspace(tmp_path, monkeypatch)

    result = runner.invoke(app, ["assets", "review", "queue", "hotel-test",
                                 "--queue", "pending", "--mode", "blind"])

    assert result.exit_code == 0, result.output
    files = list(workspace.path("01_sources").glob("review_queue_blind_*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text("utf-8"))
    assert payload["blinding"] == "blind"
    assert set(payload["items"][0]) == {"asset_id", "checksum", "source"}


def test_cli_rejects_an_unknown_mode(tmp_path, monkeypatch) -> None:
    from hotel_pipeline.cli import app

    runner, _, _ = cli_workspace(tmp_path, monkeypatch)
    result = runner.invoke(app, ["assets", "review", "queue", "hotel-test",
                                 "--mode", "semi-aveugle"])
    assert result.exit_code == 1


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
    with pytest.raises(ValueError, match="première décision de visibilité ne corrige"):
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


# --- identité, aptitude et point de vue sont trois questions ----------------


def test_two_near_identical_confirmations_make_one_auxiliary_viewpoint(tmp_path) -> None:
    """Le cas réel : deux vues du même point, à deux degrés près.

    Le WelcomINNS y est identifiable à 117 m, sur 40 % de la largeur du cadre.
    Confirmer leur identité les promouvait toutes deux en porteuses, si bien
    que le corpus paraissait compter deux observations géométriques
    indépendantes là où il n'y a qu'un point de vue auxiliaire, photographié
    deux fois.
    """
    from hotel_pipeline.schemas import GeometrySuitability

    first = asset(tmp_path, name="v1.jpg", id="mapillary-7688979294475178",
                  suitability=None, viewpoint_cluster="vp-1",
                  target_building_visible=None)
    second = asset(tmp_path, name="v2.jpg", id="mapillary-949396083308163",
                   suitability=None, viewpoint_cluster="vp-1",
                   target_building_visible=None, cluster_role=ClusterRole.OVERLAP)
    assets = [first, second]

    for asset_id in (first.id, second.id):
        decide(assets, asset_id=asset_id, rationale="pylône « HW HÔTEL WELCOMINNS » lisible")
        review.assess(
            assets, asset_id, GeometrySuitability.AUXILIARY, "hm",
            "cible reconnaissable mais façade trop reculée pour la structure",
            ["distance ≈ 117 m, arbres et clôture devant la façade"],
            {"frame_width_fraction": 0.40, "frame_height_fraction": 0.20},
        )

    # Les deux restent admises — `auxiliary` autorise l'usage géométrique…
    assert all(role_for(a, policy())[0].value == "photo_geometry" for a in assets)
    # …mais elles ne comptent que pour **un** point de vue, et auxiliaire.
    assert review.viewpoints_by_suitability(assets) == {"auxiliary": 1}


def test_a_primary_and_an_auxiliary_viewpoint_are_counted_apart(tmp_path) -> None:
    from hotel_pipeline.schemas import GeometrySuitability

    close = asset(tmp_path, name="a.jpg", id="mapillary-1", suitability="primary",
                  viewpoint_cluster="vp-1")
    far = asset(tmp_path, name="b.jpg", id="mapillary-2", suitability="auxiliary",
                viewpoint_cluster="vp-2")

    assert review.viewpoints_by_suitability([close, far]) == {
        "auxiliary": 1, "primary": 1
    }


def test_the_best_member_names_the_viewpoint(tmp_path) -> None:
    """Une grappe vaut ce que vaut son meilleur membre, pas sa moyenne."""
    strong = asset(tmp_path, name="a.jpg", id="m-1", suitability="primary",
                   viewpoint_cluster="vp-1")
    weak = asset(tmp_path, name="b.jpg", id="m-2", suitability="auxiliary",
                 viewpoint_cluster="vp-1")

    assert review.viewpoints_by_suitability([strong, weak]) == {"primary": 1}


def test_an_unverified_target_is_not_counted_as_a_viewpoint(tmp_path) -> None:
    unseen = asset(tmp_path, name="a.jpg", id="m-1", suitability="primary",
                   viewpoint_cluster="vp-1", target_building_visible=None)

    assert review.viewpoints_by_suitability([unseen]) == {}


def test_the_cluster_canonical_follows_the_target(tmp_path) -> None:
    """Le défaut distinct que l'arbitrage laissait passer.

    Le représentant d'un point de vue était choisi sur la résolution seule :
    une image plus lourde tournée ailleurs l'emportait sur celle qui montre
    réellement le bâtiment.
    """
    from hotel_pipeline import dedup_levels
    from hotel_pipeline.schemas import ClusterRole as CR

    elsewhere = asset(tmp_path, name="a.jpg", id="m-large", suitability=None,
                      target_building_visible=None, viewpoint_cluster="vp-1",
                      width=4000, height=3000, file_size_bytes=5_000_000)
    on_target = asset(tmp_path, name="b.jpg", id="m-small", suitability=None,
                      viewpoint_cluster="vp-1",
                      width=1024, height=768, file_size_bytes=200_000)
    assets = [elsewhere, on_target]

    dedup_levels.assign_roles(assets)

    canonical = next(a for a in assets if a.cluster_role is CR.CANONICAL)
    assert canonical.id == "m-small"


def test_an_identity_confirmed_view_awaiting_aptitude_is_blocking(tmp_path) -> None:
    """La revue porte sur deux questions ; une seule tranchée bloque encore."""
    assets = [asset(tmp_path, suitability=None)]
    decide(assets)

    assert review.counts(assets, policy()).blocking == 1
    assert role_for(assets[0], policy())[1] == "aptitude géométrique non évaluée"


def test_scenery_is_never_blocking_even_though_a_review_could_say_yes(tmp_path) -> None:
    """La file doit mesurer le travail restant, pas le corpus.

    Simuler une revue favorable sur tout ce qui n'est pas apprécié faisait
    entrer les 247 vues d'environnement, dont la revue dirait non.
    """
    scenery = asset(
        tmp_path, suitability=None, id="street_view-1", source="street_view",
        target_building_visible=False, contains_building=False, subjects=[],
        review_status=ReviewStatus.AUTOMATIC_ACCEPTED,
    )

    assert review.counts([scenery], policy()).blocking == 0


def test_an_assessed_view_is_no_longer_blocking(tmp_path) -> None:
    assets = [asset(tmp_path, suitability=None)]
    decide(assets)
    assert review.counts(assets, policy()).blocking == 1

    from hotel_pipeline.schemas import GeometrySuitability

    review.assess(
        assets, assets[0].id, GeometrySuitability.PRIMARY, "hm",
        "façade franche sur la moitié du cadre", ["mesures de cadrage"],
    )

    assert review.counts(assets, policy()).blocking == 0


# --- registre versionnable --------------------------------------------------


def test_the_register_carries_decisions_and_no_images(tmp_path) -> None:
    """`work/` est ignoré par Git : sans registre, rien n'est versionnable.

    Les décisions sont la seule chose que le pipeline ne sait pas régénérer.
    """
    from hotel_pipeline import decisions

    assets = [asset(tmp_path, suitability=None), asset(tmp_path, name="b.jpg", id="m-2")]
    decide(assets)

    register = decisions.export(assets, "hotel-test").as_dict()

    ids = [d["asset_id"] for d in register["decisions"]]
    assert ids == ["m-2", "mapillary-1"]  # trié, et l'asset sans décision exclu
    first = next(d for d in register["decisions"] if d["asset_id"] == "mapillary-1")
    assert first["checksum"] == assets[0].checksum
    assert first["review_history"][0]["decided_by"] == "Hicham"
    assert "local_path" not in json.dumps(register)


def test_applying_the_register_reproduces_the_decisions(tmp_path) -> None:
    from hotel_pipeline import decisions
    from hotel_pipeline.schemas import GeometrySuitability

    decided = [asset(tmp_path, suitability=None)]
    decide(decided)
    review.assess(decided, "mapillary-1", GeometrySuitability.AUXILIARY, "hm",
                  "cible reconnaissable, façade reculée", ["mesures de cadrage"])
    register = decisions.export(decided, "hotel-test").as_dict()

    # Un manifeste neuf, sans aucune décision.
    fresh = [asset(tmp_path, suitability=None)]
    assert not fresh[0].has_been_reviewed

    decisions.apply(fresh, register)

    assert fresh[0].review_status is ReviewStatus.HUMAN_ACCEPTED
    assert fresh[0].target_building_visible is True
    assert fresh[0].geometry_suitability is GeometrySuitability.AUXILIARY
    assert len(fresh[0].review_history) == 1
    assert fresh[0].reviewer == "Hicham"


def test_a_register_pointing_at_another_image_is_refused(tmp_path) -> None:
    """Une décision porte sur ce qui a été vu, jamais sur ce qui l'a remplacé."""
    from hotel_pipeline import decisions

    decided = [asset(tmp_path, suitability=None)]
    decide(decided)
    register = decisions.export(decided, "hotel-test").as_dict()
    register["decisions"][0]["checksum"] = "f" * 64

    fresh = [asset(tmp_path, suitability=None)]
    before = fresh[0].model_dump_json()

    with pytest.raises(decisions.RegisterRefused, match="l'image a changé"):
        decisions.apply(fresh, register)

    assert fresh[0].model_dump_json() == before


def test_a_register_entry_for_an_unknown_asset_is_refused(tmp_path) -> None:
    from hotel_pipeline import decisions

    decided = [asset(tmp_path, suitability=None)]
    decide(decided)
    register = decisions.export(decided, "hotel-test").as_dict()

    with pytest.raises(decisions.RegisterRefused, match="sans asset correspondant"):
        decisions.apply([asset(tmp_path, name="b.jpg", id="autre")], register)
def test_the_register_reproduces_the_welcominns_state() -> None:
    """Le corpus de base plus le registre donnent l'état après revue.

    Le corpus reste sans décision : c'est le registre, versionné à part, qui
    prouve ce qui a été jugé. Rejouer l'un sur l'autre doit rendre exactement
    les mêmes nombres, sans quoi « l'état après revue » ne serait qu'un
    souvenir.
    """
    from pathlib import Path as _Path

    from hotel_pipeline import decisions, dedup_levels, roles
    from hotel_pipeline.schemas import Asset as _Asset, ClusterRole as _CR

    snapshot = json.loads(_Path("tests/fixtures/corpus_snapshot.json").read_text("utf-8"))
    assets = [_Asset.model_validate(a) for a in snapshot["assets"]]
    register = json.loads(
        _Path("decisions/welcominns-boucherville/asset_reviews.json").read_text("utf-8")
    )

    assert all(not a.review_history and not a.geometry_history for a in assets)

    decisions.apply(assets, register)
    building = snapshot["building"]
    dedup_levels.run(assets, building["lat"], building["lon"])
    report = roles.assign(assets)

    assert report.counts.get("photo_geometry") == 2
    viewpoints = review.viewpoints_by_suitability(assets)
    # Deux fichiers porteurs, mais **un** point de vue : ils sont pris du même
    # endroit à deux degrés près. Aucun point de vue `primary` à ce jour.
    assert viewpoints["auxiliary"] == 1
    assert "primary" not in viewpoints

    cluster = next(
        a.viewpoint_cluster for a in assets if a.id == "mapillary-7688979294475178"
    )
    canonical = [
        a.id for a in assets
        if a.viewpoint_cluster == cluster and a.cluster_role is _CR.CANONICAL
    ]
    assert canonical == ["mapillary-7688979294475178"]

    numbers = review.counts(assets, policy())
    assert (numbers.pending, numbers.blocking, numbers.cohort) == (33, 11, 25)


# --- atomicité du registre --------------------------------------------------


def test_a_corrupt_second_record_leaves_the_first_asset_untouched(tmp_path) -> None:
    """Le défaut : la revalidation tombait pendant l'écriture, pas avant.

    Premier enregistrement valide, second dont la filiation est fausse : le
    premier asset était déjà muté quand l'exception partait. Un appel annoncé
    comme atomique laissait le manifeste à demi modifié.
    """
    from hotel_pipeline import decisions

    first = asset(tmp_path, name="a.jpg", id="m-1", suitability=None)
    second = asset(tmp_path, name="b.jpg", id="m-2", suitability=None)
    decided = [first, second]
    decide(decided, asset_id="m-1")
    decide(decided, asset_id="m-2")
    register = decisions.export(decided, "hotel-test").as_dict()

    # La seconde entrée prétend corriger une décision qui n'existe pas.
    register["decisions"][1]["review_history"][0]["supersedes_index"] = 0

    fresh = [asset(tmp_path, name="a.jpg", id="m-1", suitability=None),
             asset(tmp_path, name="b.jpg", id="m-2", suitability=None)]
    before = [a.model_dump_json() for a in fresh]

    with pytest.raises(decisions.RegisterRefused):
        decisions.apply(fresh, register)

    # Toute la liste, octet pour octet.
    assert [a.model_dump_json() for a in fresh] == before


def test_a_register_of_another_hotel_is_refused(tmp_path) -> None:
    from hotel_pipeline import decisions

    decided = [asset(tmp_path, suitability=None)]
    decide(decided)
    register = decisions.export(decided, "un-autre-hotel").as_dict()

    fresh = [asset(tmp_path, suitability=None)]
    with pytest.raises(decisions.RegisterRefused, match="ne portent pas sur ce corpus"):
        decisions.apply(fresh, register, hotel_id="hotel-test")
    assert not fresh[0].has_been_reviewed


def test_duplicate_records_are_refused(tmp_path) -> None:
    """Deux entrées pour un même asset : l'ordre du fichier déciderait."""
    from hotel_pipeline import decisions

    decided = [asset(tmp_path, suitability=None)]
    decide(decided)
    register = decisions.export(decided, "hotel-test").as_dict()
    register["decisions"].append(dict(register["decisions"][0]))

    fresh = [asset(tmp_path, suitability=None)]
    with pytest.raises(decisions.RegisterRefused, match="entrée dupliquée"):
        decisions.apply(fresh, register)
    assert not fresh[0].has_been_reviewed


def test_a_malformed_register_is_refused_as_data_not_as_a_crash(tmp_path) -> None:
    from hotel_pipeline import decisions

    fresh = [asset(tmp_path, suitability=None)]

    with pytest.raises(decisions.RegisterRefused, match="objet JSON"):
        decisions.apply(fresh, ["pas un objet"])
    with pytest.raises(decisions.RegisterRefused, match="sans liste 'decisions'"):
        decisions.apply(fresh, {"hotel_id": "hotel-test"})
    with pytest.raises(decisions.RegisterRefused, match="manquant"):
        decisions.apply(fresh, {"decisions": [{"asset_id": "m-1"}]})

    decided = [asset(tmp_path, suitability=None)]
    decide(decided)
    register = decisions.export(decided, "hotel-test").as_dict()
    del register["decisions"][0]["review_history"][0]["evidence"]
    with pytest.raises(decisions.RegisterRefused, match="entrée de registre invalide"):
        decisions.apply(fresh, register)

    assert not fresh[0].has_been_reviewed


def test_a_file_altered_since_the_review_is_detected(tmp_path) -> None:
    """Registre et manifeste peuvent s'accorder sur une image qui a changé."""
    from hotel_pipeline import decisions

    assets = [asset(tmp_path, suitability=None)]
    decide(assets)
    assert decisions.verify_files(assets) == []

    Path(assets[0].local_path).write_bytes(b"image-remplacee")
    problems = decisions.verify_files(assets)

    assert len(problems) == 1
    assert "l'image a changé depuis sa revue" in problems[0]


def test_cli_import_refuses_an_altered_image_and_keeps_the_manifest(
    tmp_path, monkeypatch
) -> None:
    from hotel_pipeline import decisions
    from hotel_pipeline.cli import app

    runner, workspace, subject = cli_workspace(tmp_path, monkeypatch)
    manifest = workspace.read_assets()
    decide(manifest.assets, asset_id=subject.id)
    workspace.write_assets(manifest)

    register_root = tmp_path / "registre"
    assert runner.invoke(app, ["assets", "review", "export", "hotel-test",
                               "--root", str(register_root)]).exit_code == 0

    # Le manifeste est remis à zéro, puis l'image est réécrite.
    fresh = workspace.read_assets()
    fresh.assets = [asset(tmp_path, suitability=None, id=subject.id)]
    workspace.write_assets(fresh)
    Path(subject.local_path).write_bytes(b"image-remplacee")
    before = workspace.assets_path.read_text("utf-8")

    result = runner.invoke(app, ["assets", "review", "import", "hotel-test",
                                 "--root", str(register_root)])

    assert result.exit_code == 4
    assert workspace.assets_path.read_text("utf-8") == before
    assert not list(workspace.path("01_sources").glob("review_import_*.json"))


def test_cli_import_publishes_a_report_and_the_roles(tmp_path, monkeypatch) -> None:
    from hotel_pipeline.cli import app

    runner, workspace, subject = cli_workspace(tmp_path, monkeypatch)
    manifest = workspace.read_assets()
    decide(manifest.assets, asset_id=subject.id)
    workspace.write_assets(manifest)

    register_root = tmp_path / "registre"
    runner.invoke(app, ["assets", "review", "export", "hotel-test",
                        "--root", str(register_root)])
    result = runner.invoke(app, ["assets", "review", "import", "hotel-test",
                                 "--root", str(register_root)])

    assert result.exit_code == 0, result.output
    reports = list(workspace.path("01_sources").glob("review_import_*.json"))
    assert len(reports) == 1
    payload = json.loads(reports[0].read_text("utf-8"))
    assert payload["applied"] == 1
    assert payload["register_digest"] in reports[0].name
    assert set(payload["roles"]) == {"before", "after"}
    assert set(payload["viewpoints_by_suitability"]) == {"before", "after"}
    assert (workspace.path("01_sources") / "roles_report.json").is_file()


# --- vérité terrain : aveuglement, cohorte, séquences -------------------------


def test_the_blind_queue_shows_nothing_the_system_concluded(tmp_path) -> None:
    """Étiqueter en voyant la réponse produirait des étiquettes qui en héritent."""
    assets = [asset(tmp_path)]
    queue = review.build_queue(assets, "mapillary-candidates", policy())

    blind = queue.as_dict(blind=True)
    analysis = queue.as_dict()

    assert blind["blinding"] == "blind"
    assert set(blind["items"][0]) == {"asset_id", "checksum", "source"}
    # La planche d'analyse, elle, montre tout : les deux vues coexistent.
    assert "role" in analysis["items"][0]
    assert "subject_scores" in analysis["items"][0]


def test_the_blind_board_carries_no_verdict(tmp_path) -> None:
    assets = [asset(tmp_path)]
    queue = review.build_queue(assets, "mapillary-candidates", policy())
    html = review.to_blind_html(queue)

    body = html.split("</header>", 1)[1]
    for leak in ("photo_geometry", "context_lock", "0.55", "front", "clear",
                 "automatic_accepted"):
        assert leak not in body, leak
    assert "mapillary-1" in body


def test_the_blind_board_offers_an_approved_reference(tmp_path) -> None:
    """Reconnaître le bâtiment sans rien apprendre des vues à étiqueter."""
    queue = review.build_queue([asset(tmp_path)], "mapillary-candidates", policy())
    html = review.to_blind_html(
        queue,
        reference={
            "asset_id": "mapillary-7688979294475178",
            "local_path": "reference.jpg",
            "checksum": "c" * 64,
            "rationale": "pylône HW HÔTEL WELCOMINNS lisible",
        },
    )

    assert "Référence approuvée" in html
    assert "mapillary-7688979294475178" in html


def test_the_blind_order_is_deterministic_and_not_the_manifest_order(tmp_path) -> None:
    """Les vues voisines d'une séquence ne doivent pas se suivre."""
    from hotel_pipeline import cohort

    assets = [
        asset(tmp_path, name=f"v{i}.jpg", id=f"mapillary-{i:03d}") for i in range(12)
    ]
    digest = cohort.cohort_digest(assets)

    first = [a.id for a in cohort.blind_order(assets, digest)]
    second = [a.id for a in cohort.blind_order(list(reversed(assets)), digest)]

    assert first == second
    assert first != [a.id for a in assets]


def protocol_for(assets, hotel_id="hotel-test"):
    from hotel_pipeline import cohort

    return cohort.build_protocol(assets, hotel_id)


def test_a_blind_decision_must_be_bound_to_a_protocol(tmp_path) -> None:
    """`blind` est une déclaration ; le protocole en est la preuve."""
    from hotel_pipeline.schemas import Blinding

    assets = [asset(tmp_path)]

    with pytest.raises(review.ReviewRefused, match="sans protocole"):
        decide(assets, blinding="blind")
    with pytest.raises(review.ReviewRefused, match="aveuglement inconnu"):
        decide(assets, blinding="semi-aveugle")

    decide(assets, blinding="blind", protocol=protocol_for(assets), protocol_digest="p1")
    entry = assets[0].review_history[0]

    assert entry.blinding is Blinding.BLIND
    assert entry.review_protocol_id.startswith("blind-")
    assert entry.blind_queue_digest == "p1"
    # Le défaut reste `unblinded` : les sept décisions déjà prises l'ont été en
    # voyant le diagnostic.
    assert review.ReviewEntry.model_fields["blinding"].default is Blinding.UNBLINDED


def test_a_protocol_that_does_not_cover_the_asset_is_refused(tmp_path) -> None:
    other = [asset(tmp_path, name="autre.jpg", id="mapillary-999")]
    assets = [asset(tmp_path)]

    with pytest.raises(review.ReviewRefused, match="ne figure pas au protocole"):
        decide(assets, blinding="blind", protocol=protocol_for(other),
               protocol_digest="p1")


def test_a_protocol_with_another_checksum_is_refused(tmp_path) -> None:
    """L'image a changé depuis la constitution de la file."""
    assets = [asset(tmp_path)]
    stale = protocol_for(assets)
    stale.members[0]["checksum"] = "f" * 64

    with pytest.raises(review.ReviewRefused, match="l'image a changé"):
        decide(assets, blinding="blind", protocol=stale, protocol_digest="p1")


def test_a_non_blind_protocol_cannot_carry_a_blind_decision(tmp_path) -> None:
    assets = [asset(tmp_path)]
    analysis = protocol_for(assets)
    analysis.blinding = "unblinded"

    with pytest.raises(review.ReviewRefused, match="n'est pas aveugle"):
        decide(assets, blinding="blind", protocol=analysis, protocol_digest="p1")


def test_the_geometry_decision_records_the_same_protocol(tmp_path) -> None:
    """L'aptitude s'étiquette avec les mêmes précautions que l'identité."""
    from hotel_pipeline.schemas import Blinding, GeometrySuitability

    assets = [asset(tmp_path, suitability=None)]
    protocol = protocol_for(assets)
    decide(assets, blinding="blind", protocol=protocol, protocol_digest="p1")

    review.assess(
        assets, "mapillary-1", GeometrySuitability.AUXILIARY, "Claude (Opus 5)",
        "cible reconnaissable, façade reculée", ["mesures de cadrage"],
        blinding="blind", protocol=protocol, protocol_digest="p1",
    )
    entry = assets[0].geometry_history[-1]

    assert entry.blinding is Blinding.BLIND
    assert entry.review_protocol_id == protocol.protocol_id

    with pytest.raises(review.ReviewRefused, match="sans protocole"):
        review.assess(
            assets, "mapillary-1", GeometrySuitability.PRIMARY, "Claude (Opus 5)",
            "correction", ["mesures"], blinding="blind",
        )


def test_a_protocol_survives_an_unrelated_manifest_change(tmp_path) -> None:
    """Chaque décision modifie le manifeste : s'y rattacher le périmerait aussitôt."""
    subject = asset(tmp_path)
    neighbour = asset(tmp_path, name="voisine.jpg", id="mapillary-2")
    assets = [subject, neighbour]
    protocol = protocol_for(assets)

    # Une première décision change le manifeste…
    decide(assets, asset_id="mapillary-2", blinding="blind",
           protocol=protocol, protocol_digest="p1")

    # …et la seconde reste couverte par le même protocole.
    decide(assets, asset_id="mapillary-1", blinding="blind",
           protocol=protocol, protocol_digest="p1")

    assert assets[0].review_history[0].review_protocol_id == protocol.protocol_id


def test_the_protocol_binds_the_cohort_and_its_evidence(tmp_path) -> None:
    from hotel_pipeline import cohort

    assets = [asset(tmp_path), asset(tmp_path, name="b.jpg", id="mapillary-2")]
    protocol = cohort.build_protocol(
        assets, "hotel-test",
        reference={"asset_id": "mapillary-9", "checksum": "c" * 64,
                   "local_path": "r.jpg", "rationale": "pylône lisible",
                   "in_cohort": False},
        predictions_digest="pred123", sequence_register_digest="seq456",
    )
    payload = protocol.as_dict()

    assert payload["cohort_digest"] == cohort.cohort_digest(assets)
    assert payload["predictions_digest"] == "pred123"
    assert payload["sequence_register_digest"] == "seq456"
    assert payload["reference"]["in_cohort"] is False
    # L'ordre de présentation est celui de l'empreinte de cohorte.
    assert payload["presentation_order"] == [
        a.id for a in cohort.blind_order(assets, payload["cohort_digest"])
    ]


def test_the_cohort_states_what_it_cannot_measure(tmp_path) -> None:
    """Sélectionnée par le modèle, elle exclut les faux négatifs."""
    from hotel_pipeline import cohort

    snapshot = cohort.predictions([asset(tmp_path)], policy())

    assert "rappel" in " ".join(snapshot["scope"]["cannot_measure"])
    assert "précision parmi les candidats détectés" in snapshot["scope"]["measures"]
    assert "contains_building" in snapshot["cohort_definition"]


def test_the_predictions_are_captured_before_labelling(tmp_path) -> None:
    from hotel_pipeline import cohort

    subject = asset(tmp_path)
    snapshot = cohort.predictions([subject], policy())
    row = snapshot["predictions"][0]

    assert row["already_reviewed"] is False
    assert row["subject_scores"] == subject.subject_scores
    assert row["role"] and row["role_reason"]
    assert snapshot["cohort_digest"] == cohort.cohort_digest([subject])


def test_an_unreachable_sequence_is_declared_unknown(tmp_path) -> None:
    """Une proximité géographique n'est pas une séquence."""
    from hotel_pipeline import cohort

    def failing(_ids):
        raise RuntimeError("502 depuis Mapillary")

    register = cohort.build_register([asset(tmp_path)], "h", failing)

    assert register.correlation == "unknown"
    assert "502" in register.error
    assert register.entries == []


def test_a_partial_sequence_lookup_is_not_declared_known(tmp_path) -> None:
    from hotel_pipeline import cohort

    assets = [asset(tmp_path, name="a.jpg", id="mapillary-111"),
              asset(tmp_path, name="b.jpg", id="mapillary-222")]

    register = cohort.build_register(
        assets, "h", lambda ids: {"111": {"sequence": "SEQ-A", "captured_at": 1}}
    )

    assert register.correlation == "partial"
    assert register.by_sequence()["SEQ-A"] == ["mapillary-111"]
    assert register.by_sequence()["sans-séquence"] == ["mapillary-222"]


def test_a_complete_lookup_is_known(tmp_path) -> None:
    from hotel_pipeline import cohort

    assets = [asset(tmp_path, name="a.jpg", id="mapillary-111"),
              asset(tmp_path, name="b.jpg", id="mapillary-222")]

    register = cohort.build_register(
        assets, "h",
        lambda ids: {
            "111": {"sequence": "SEQ-A", "captured_at": 1},
            "222": {"sequence": "SEQ-A", "captured_at": 2},
        },
    )

    assert register.correlation == "known"
    assert register.by_sequence() == {"SEQ-A": ["mapillary-111", "mapillary-222"]}
    assert register.response_digest


def test_a_command_copied_from_the_blind_board_records_the_protocol(
    tmp_path, monkeypatch
) -> None:
    """La commande imprimée doit produire une décision réellement aveugle.

    Elle omettait `--blinding blind`, tandis que la CLI vaut `unblinded` par
    défaut : les 18 décisions auraient été mal qualifiées.
    """
    import re
    import shlex

    from hotel_pipeline.cli import app
    from hotel_pipeline.schemas import Blinding

    runner, workspace, subject = cli_workspace(tmp_path, monkeypatch)
    manifest = workspace.read_assets()
    manifest.assets[0] = manifest.assets[0].model_copy(
        update={"contains_building": True}
    )
    workspace.write_assets(manifest)

    assert runner.invoke(app, [
        "assets", "review", "queue", "hotel-test",
        "--queue", "mapillary-candidates", "--mode", "blind",
    ]).exit_code == 0

    board = next(workspace.path("01_sources").glob("review_board_blind_*.html"))
    html = board.read_text("utf-8")
    printed = re.search(r"<pre>(.*?)</pre>", html, re.S).group(1)
    printed = printed.replace("&lt;", "<").replace("&gt;", ">").replace("\\\n", " ")

    # La commande est exécutée telle qu'elle est imprimée, aux valeurs près.
    arguments = shlex.split(printed)[1:]
    arguments = [
        "hotel-test" if part == "<hôtel>" else part for part in arguments
    ]
    arguments = [
        "confirmed" if part == "confirmed|rejected|unresolved" else part
        for part in arguments
    ]
    arguments = ["Claude (Opus 5)" if part == "<vous>" else part for part in arguments]
    arguments = ["façade reconnue" if part == "…" else part for part in arguments]

    result = runner.invoke(app, arguments)

    assert result.exit_code == 0, result.output
    entry = workspace.read_assets().assets[0].review_history[0]
    assert entry.blinding is Blinding.BLIND
    assert entry.review_protocol_id.startswith("blind-")
    assert entry.blind_queue_digest


def test_the_blind_order_uses_the_cohort_digest(tmp_path, monkeypatch) -> None:
    """La planche annonce l'empreinte de cohorte : elle doit l'employer."""
    import json as json_module

    from hotel_pipeline import cohort
    from hotel_pipeline.cli import app

    runner, workspace, _ = cli_workspace(tmp_path, monkeypatch)
    manifest = workspace.read_assets()
    manifest.assets = [
        manifest.assets[0].model_copy(update={"contains_building": True}),
        manifest.assets[0].model_copy(
            update={"id": "mapillary-2", "checksum": "b" * 64,
                    "contains_building": True}
        ),
    ]
    workspace.write_assets(manifest)

    runner.invoke(app, ["assets", "review", "queue", "hotel-test",
                        "--queue", "mapillary-candidates", "--mode", "blind"])

    protocol = json_module.loads(
        next(workspace.path("01_sources").glob("review_protocol_*.json")).read_text("utf-8")
    )
    expected = cohort.cohort_digest(cohort.members(manifest.assets))

    assert protocol["cohort_digest"] == expected
    assert protocol["protocol_id"] == f"blind-{expected}"
    assert protocol["presentation_order"] == [
        a.id for a in cohort.blind_order(cohort.members(manifest.assets), expected)
    ]
