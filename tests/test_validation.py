"""Validation conditionnelle et non-régression des confusions (Lot 1B V2).

Le cas décisif y est générique : un **concurrent de même nature** ne se confond
pas comme un immeuble de bureaux voisin. Il partage la fonction, l'enseigne et
souvent l'architecture ; un modèle qui répond « un hôtel est visible » a raison
et se trompe en même temps.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hotel_pipeline import validation
from hotel_pipeline.schemas import (
    Asset,
    Blinding,
    GeometryEntry,
    GeometrySuitability,
    PropertyMatchStatus,
    ReviewDecision,
    ReviewEntry,
)
from hotel_pipeline.schemas.assets import DECISION_STATUS, VISIBILITY_OF


def entry(decision=ReviewDecision.REJECTED, blinding=Blinding.BLIND, **overrides):
    fields = dict(
        decision=decision,
        decided_by="Claude (Opus 5)",
        rationale="motif",
        evidence=["preuve"],
        reviewed_checksum="a" * 64,
        blinding=blinding,
    )
    if blinding is Blinding.BLIND:
        fields.update(review_protocol_id="blind-x-y", review_protocol_digest="d")
    fields.update(overrides)
    return ReviewEntry(**fields)


def asset(asset_id="mapillary-1", history=(), **overrides) -> Asset:
    history = list(history)
    fields = dict(
        id=asset_id, source="mapillary", source_url_or_id="1", rights="open_data",
        ai_eligible=False, confidence=0.5, category="facade", checksum="a" * 64,
        subject_scores={"building": 0.99},
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


# --- le concurrent de même nature ---------------------------------------------


def test_a_same_kind_competitor_is_its_own_confusion_class() -> None:
    """« Hôtel Mortagne » n'est pas un faux positif comme les autres.

    Un immeuble de bureaux voisin ne partage que le quartier. Un hôtel
    concurrent partage la fonction, l'enseigne et l'architecture : un modèle
    qui répond « bâtiment » — voire « hôtel » — a raison sur la classe et tort
    sur l'objet.
    """
    corpus = [
        asset("mapillary-mortagne", [entry(rationale="enseigne « HÔTEL MORTAGNE » lisible au fond")]),
        asset("mapillary-tetra", [entry(rationale="immeuble de bureaux du 1205 rue Ampère")]),
        asset("mapillary-isomed", [entry(rationale="numéro 1201 et enseigne Isomed")]),
        asset("mapillary-vide", [entry(rationale="stationnement enneigé, aucun bâtiment")]),
    ]

    kinds = {c["asset_id"]: c["kind"] for c in validation.confusions(corpus)}

    assert kinds["mapillary-mortagne"] == "competitor_same_kind"
    assert kinds["mapillary-tetra"] == "neighbouring_office"
    assert kinds["mapillary-isomed"] == "neighbouring_office"
    assert kinds["mapillary-vide"] == "no_building"


def test_the_competitor_case_survives_the_real_corpus() -> None:
    """Non-régression sur les motifs réellement écrits pendant la passe aveugle.

    Le corpus vit hors dépôt (`work/` est ignoré) : les huit rejets sont figés
    en fixture pour que ce cas survive à un clone neuf.
    """
    fixture = json.loads(
        (Path(__file__).parent / "data/blind_rejects_welcominns.json").read_text("utf-8")
    )
    corpus = [
        asset(row["asset_id"], [entry(rationale=row["rationale"])])
        for row in fixture["rejects"]
    ]
    expected = {row["asset_id"]: row["expected_kind"] for row in fixture["rejects"]}

    kinds = {c["asset_id"]: c["kind"] for c in validation.confusions(corpus)}

    assert kinds == expected
    assert kinds["mapillary-1338281626865323"] == "competitor_same_kind"
    assert sum(1 for k in kinds.values() if k == "competitor_same_kind") == 1
    assert sum(1 for k in kinds.values() if k == "neighbouring_office") == 3


def test_a_competitor_sign_disqualifies_by_property_match() -> None:
    """Le verrou existant : lire une enseigne étrangère écarte l'image.

    C'est la seule barrière automatique contre ce cas ; le modèle de sujets,
    lui, dirait « bâtiment » avec raison.
    """
    from hotel_pipeline.classify_cascade import property_status

    status = property_status(
        asset("mapillary-1", sign_text="HÔTEL MORTAGNE"),
        expected=["welcominns", "welcome inns"],
        excluded=["mortagne"],
    )
    assert status is PropertyMatchStatus.MISMATCH


# --- les populations restent séparées -------------------------------------------


def test_the_blind_pass_ignores_later_supersessions() -> None:
    """Une correction de seconde passe ne doit pas améliorer le score aveugle."""
    blind = entry(ReviewDecision.UNRESOLVED)
    corrected = entry(
        ReviewDecision.CONFIRMED, blinding=Blinding.UNBLINDED, supersedes_index=0,
        rationale="continuité de séquence",
    )
    subject = asset("mapillary-1", [blind, corrected])

    predictions = {"predictions": [
        {"asset_id": "mapillary-1", "target_building_visible": True,
         "role": "context_lock", "subject_scores": {"building": 0.9}}
    ]}
    result = validation.blind_pass([subject], predictions)

    # La première passe reste indécise, malgré la confirmation ultérieure.
    assert result["three_class"] == {"unresolved": 1}
    assert result["resolved_only"]["count"] == 0
    # L'état opérationnel, lui, suit la dernière décision.
    assert validation.operational([subject]) == {"labels": 1, "confirmed": 1}


def test_binary_metrics_only_cover_resolved_labels() -> None:
    """Compter une indécise reviendrait à lui prêter une vérité qu'elle nie."""
    corpus = [
        asset("m-1", [entry(ReviewDecision.CONFIRMED)]),
        asset("m-2", [entry(ReviewDecision.REJECTED)]),
        asset("m-3", [entry(ReviewDecision.UNRESOLVED)]),
    ]
    predictions = {"predictions": [
        {"asset_id": "m-1", "target_building_visible": True},
        {"asset_id": "m-2", "target_building_visible": False},
        {"asset_id": "m-3", "target_building_visible": True},
    ]}

    result = validation.blind_pass(corpus, predictions)

    assert result["labels"] == 3
    assert result["resolved_only"]["count"] == 2
    assert result["undecided_rate"] == pytest.approx(1 / 3, abs=1e-4)


def test_the_report_states_what_it_cannot_measure() -> None:
    limits = " ".join(validation.LIMITS)

    assert "rappel" in limits
    assert "faux négatifs" in limits
    assert "deux séquences" in limits
    assert "aucun seuil" in limits


def test_results_are_broken_down_by_sequence() -> None:
    """Deux vues d'une même séquence ne sont pas deux observations indépendantes."""
    corpus = [
        asset("m-1", [entry(ReviewDecision.CONFIRMED)]),
        asset("m-2", [entry(ReviewDecision.CONFIRMED)]),
        asset("m-3", [entry(ReviewDecision.REJECTED)]),
    ]
    register = {"entries": [
        {"asset_id": "m-1", "sequence_id": "SEQ-A"},
        {"asset_id": "m-2", "sequence_id": "SEQ-A"},
        {"asset_id": "m-3", "sequence_id": "SEQ-B"},
    ]}

    grouped = validation.by_sequence(corpus, register)

    assert grouped["SEQ-A"] == {"labels": 2, "confirmed": 2}
    assert grouped["SEQ-B"] == {"labels": 1, "rejected": 1}


def test_the_geometry_summary_separates_blind_from_operational() -> None:
    blind = GeometryEntry(
        suitability=GeometrySuitability.AUXILIARY, decided_by="C", rationale="r",
        evidence=["e"], reviewed_checksum="a" * 64, blinding=Blinding.BLIND,
        review_protocol_id="p", review_protocol_digest="d",
    )
    revised = GeometryEntry(
        suitability=GeometrySuitability.PRIMARY, decided_by="C", rationale="mesures",
        evidence=["e"], reviewed_checksum="a" * 64, supersedes_index=0,
    )
    subject = asset(
        "m-1", [entry(ReviewDecision.CONFIRMED)],
        geometry_history=[blind, revised],
        geometry_suitability=GeometrySuitability.PRIMARY,
    )

    assert validation.geometry_summary([subject])["counts"] == {"auxiliary": 1}
    assert validation.geometry_summary([subject], blind_only=False)["counts"] == {
        "primary": 1
    }


# --- une revue non conclusive n'est pas rétablie par le système ------------------


def test_an_undecided_review_is_never_overturned_into_a_confirmation() -> None:
    """Le défaut trouvé en rejouant la cascade.

    Une personne avait regardé sans conclure ; la cascade a écrit un verdict à
    sa place. Elle peut constater qu'elle ne voit rien, jamais établir la cible.
    """
    from hotel_pipeline.classify_cascade import classify

    reviewed = asset(
        "m-1", [entry(ReviewDecision.UNRESOLVED)],
        camera_lat=45.573, camera_lon=-73.443, heading_is_measured=True,
        target_in_frame_fraction=0.6, sees_building=True,
        property_match_status=PropertyMatchStatus.MATCH,
        local_path=None,
    )
    assets = [reviewed]

    classify(assets, classifier=None)

    assert assets[0].target_building_visible is not True
    assert "non conclusive" in assets[0].target_evidence


def test_the_manifest_refuses_a_confirmation_under_an_undecided_review() -> None:
    with pytest.raises(ValueError, match="incompatible avec la décision"):
        asset("m-1", [entry(ReviewDecision.UNRESOLVED)], target_building_visible=True)

    # `False` reste permis : le système peut constater qu'il ne voit rien.
    assert asset(
        "m-1", [entry(ReviewDecision.UNRESOLVED)], target_building_visible=False
    ).target_building_visible is False
