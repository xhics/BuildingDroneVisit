"""Statut terminal `human_unresolved` et migration sans perte.

Le défaut corrigé : une revue close sans conclusion restait indéfiniment « en
attente ». Le décompte réclamait donc un travail déjà fait, et confondait deux
situations opposées — personne n'a jugé, personne ne peut trancher.
"""

from __future__ import annotations

import json

import pytest

from hotel_pipeline import review
from hotel_pipeline.migrate_review_status import (
    migrate_file,
    migrate_payload,
    needs_migration,
)
from hotel_pipeline.schemas import (
    Asset,
    AssetManifest,
    Blinding,
    ReviewDecision,
    ReviewEntry,
    ReviewStatus,
)
from hotel_pipeline.schemas.assets import DECISION_STATUS, VISIBILITY_OF
from hotel_pipeline.schemas.policy import PipelinePolicy


@pytest.fixture
def policy() -> PipelinePolicy:
    return PipelinePolicy()


def entry(decision: ReviewDecision, **overrides) -> ReviewEntry:
    return ReviewEntry(
        decision=decision, decided_by="Claude (Opus 5)", rationale="motif",
        evidence=["preuve"], reviewed_checksum="a" * 64,
        blinding=Blinding.UNBLINDED, **overrides,
    )


def asset(asset_id="mapillary-1", history=(), **overrides) -> Asset:
    history = list(history)
    fields = dict(
        id=asset_id, source="mapillary", source_url_or_id="1", rights="open_data",
        ai_eligible=False, confidence=0.5, category="facade", checksum="a" * 64,
    )
    if history:
        last = history[-1]
        fields.update(
            review_history=history,
            target_visibility_decision=last.decision,
            review_status=DECISION_STATUS[last.decision],
            target_building_visible=VISIBILITY_OF[last.decision],
            reviewer=last.decided_by, review_rationale=last.rationale,
            review_evidence=last.evidence,
        )
    fields.update(overrides)
    return Asset(**fields)


# --- deux situations opposées, deux statuts ---------------------------------


def test_never_judged_and_judged_without_conclusion_are_distinct() -> None:
    untouched = asset("m-1")
    examined = asset("m-2", [entry(ReviewDecision.UNRESOLVED)])

    assert untouched.review_status is ReviewStatus.NEEDS_REVIEW
    assert examined.review_status is ReviewStatus.HUMAN_UNRESOLVED
    assert untouched.review_status is not examined.review_status


def test_an_examined_view_leaves_the_pending_queue(policy) -> None:
    corpus = [asset("m-1"), asset("m-2", [entry(ReviewDecision.UNRESOLVED)])]

    numbers = review.counts(corpus, policy)

    assert [a.id for a in review.pending(corpus)] == ["m-1"]
    assert [a.id for a in review.reviewed_unresolved(corpus)] == ["m-2"]
    assert (numbers.pending, numbers.reviewed_unresolved) == (1, 1)


def test_an_examined_view_no_longer_blocks(policy) -> None:
    """La revue a eu lieu : la redemander ne produirait aucune preuve nouvelle."""
    corpus = [asset("m-1", [entry(ReviewDecision.UNRESOLVED)])]

    assert review.blocking(corpus, policy) == []


def test_the_queue_is_addressable_from_the_command_line() -> None:
    assert "reviewed-unresolved" in review.QUEUES
    description, selector = review.QUEUES["reviewed-unresolved"]
    corpus = [asset("m-1", [entry(ReviewDecision.UNRESOLVED)]), asset("m-2")]

    assert [a.id for a in selector(corpus, None)] == ["m-1"]
    assert "indécidable" in description


def test_the_undecided_view_reads_as_closed_not_pending(policy) -> None:
    """Le rôle reste un verrou de contexte, mais le motif change."""
    from hotel_pipeline import roles

    corpus = [
        asset(
            "m-1", [entry(ReviewDecision.UNRESOLVED)], contains_building=True,
            camera_lat=45.573, camera_lon=-73.443, heading_deg=90.0,
            heading_is_measured=True,
        )
    ]
    _, reason = roles.role_for(corpus[0], policy)

    assert "revue close sans conclusion" in reason


# --- réouverture : preuve nouvelle seulement ---------------------------------


def test_only_an_added_entry_can_reopen_an_undecided_review() -> None:
    """Aucun recalcul ne rouvre : il faut une décision ajoutée.

    Reposer `needs_review` sur une revue non conclusive reviendrait à effacer
    le fait qu'une personne a regardé.
    """
    history = [entry(ReviewDecision.UNRESOLVED)]

    with pytest.raises(ValueError, match="incompatible avec la décision"):
        asset("m-1", history, review_status=ReviewStatus.NEEDS_REVIEW)

    # La voie légitime : une entrée supersédante, appuyée sur une preuve neuve.
    reopened = asset(
        "m-1",
        [*history, entry(ReviewDecision.CONFIRMED, supersedes_index=0)],
    )
    assert reopened.review_status is ReviewStatus.HUMAN_ACCEPTED
    assert len(reopened.review_history) == 2


def test_the_terminal_status_still_requires_a_review() -> None:
    """Constater qu'on ne peut pas trancher suppose d'avoir regardé."""
    with pytest.raises(ValueError, match="sans aucune revue"):
        asset("m-1", review_status=ReviewStatus.HUMAN_UNRESOLVED)


# --- migration ---------------------------------------------------------------


def legacy_payload() -> dict:
    """Un manifeste tel qu'il était écrit avant le statut terminal."""
    payload = json.loads(
        AssetManifest(
            hotel_id="h",
            assets=[
                asset("m-1", [entry(ReviewDecision.UNRESOLVED)]),
                asset("m-2", [entry(ReviewDecision.CONFIRMED)]),
                asset("m-3"),
            ],
        ).model_dump_json()
    )
    payload["assets"][0]["review_status"] = ReviewStatus.NEEDS_REVIEW.value
    return payload


def test_the_migration_converts_only_what_the_history_proves() -> None:
    payload = legacy_payload()
    assert needs_migration(payload)

    migrated, report = migrate_payload(payload)

    assert (report.converted, report.converted_ids) == (1, ["m-1"])
    # `m-3` n'a jamais été examiné : son `needs_review` est le bon statut, et
    # sa décision par défaut vaut aussi `unresolved`. S'y fier l'aurait converti.
    assert report.never_reviewed == 1
    statuses = {a["id"]: a["review_status"] for a in migrated["assets"]}
    assert statuses == {
        "m-1": "human_unresolved",
        "m-2": "human_accepted",
        "m-3": "needs_review",
    }


def test_a_never_reviewed_corpus_needs_no_migration() -> None:
    """Le piège : `target_visibility_decision` vaut `unresolved` par défaut.

    Un corpus jamais examiné porte donc partout la décision « indécise » et le
    statut « en attente ». S'y fier convertirait tout le corpus, et déclarerait
    close une revue qui n'a jamais eu lieu. Seul l'historique fait foi.
    """
    payload = json.loads(
        AssetManifest(hotel_id="h", assets=[asset("m-1"), asset("m-2")]).model_dump_json()
    )
    assert all(
        a["target_visibility_decision"] == "unresolved"
        and a["review_status"] == "needs_review"
        for a in payload["assets"]
    )

    assert not needs_migration(payload)
    _, report = migrate_payload(payload)
    assert (report.converted, report.never_reviewed) == (0, 2)


def test_the_migration_is_idempotent() -> None:
    once, _ = migrate_payload(legacy_payload())
    twice, report = migrate_payload(json.loads(json.dumps(once)))

    assert report.converted == 0
    assert report.already_terminal == 1
    assert twice == once


def test_the_migration_touches_nothing_but_the_status(tmp_path) -> None:
    path = tmp_path / "assets.json"
    payload = legacy_payload()
    path.write_text(json.dumps(payload), "utf-8")

    manifest, report = migrate_file(path)

    assert report.untouched_decisions
    migrated = json.loads(manifest.model_dump_json())
    for before, after in zip(payload["assets"], migrated["assets"]):
        assert {k: v for k, v in before.items() if k != "review_status"} == {
            k: v for k, v in after.items() if k != "review_status"
        }
    # Le verdict lui-même survit intact.
    assert manifest.assets[0].target_visibility_decision is ReviewDecision.UNRESOLVED
    assert len(manifest.assets[0].review_history) == 1


def test_an_unmigrated_manifest_says_what_to_run() -> None:
    """Un manifeste antérieur ne doit pas ressembler à une donnée corrompue."""
    with pytest.raises(ValueError, match="migrate-review-status"):
        AssetManifest.model_validate(legacy_payload())
