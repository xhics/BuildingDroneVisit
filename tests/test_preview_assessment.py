"""L'aperçu devient preuve ou rejet, pour un besoin précis (collecte V2).

La boucle ne se refermait pas : `review geometry --measure` gardait ses mesures
dans `GeometryEntry`, `demands assess` lisait les champs plats de l'asset, et
`AcquisitionProvenance` ne conservait pas `serves_demands`. Après
téléchargement, rien ne permettait de transformer une preview en constat pour
le besoin qui l'avait motivée.
"""

from __future__ import annotations

import pytest

from hotel_pipeline.schemas.acquisition import (
    CaptureDemand,
    CaptureIntent,
    TargetKind,
)
from hotel_pipeline.schemas.preview import (
    PreviewAssessment,
    PreviewAssessmentLog,
    PreviewVerdict,
    promotable,
)


def _assessment(**overrides):
    fields = dict(
        asset_id="a1", demand_id="obligation:front",
        plan_id="p1", request_digest="d" * 16, checksum="c" * 16,
        rationale="mesuré sur l'aperçu", assessed_by="operateur",
        verdict=PreviewVerdict.ESTABLISHED,
    )
    fields.update(overrides)
    return PreviewAssessment(**fields)


def _demand(demand_id="obligation:front", **overrides):
    fields = dict(
        demand_id=demand_id, intent=CaptureIntent.BUILDING_CAPTURE,
        target_kind=TargetKind.VIEW_SECTOR, target_ref="front",
    )
    fields.update(overrides)
    return CaptureDemand(**fields)


# --- un constat porte sur un couple, non sur un fichier -----------------------


def test_a_measure_on_one_demand_does_not_credit_another() -> None:
    """La contamination interdite : une même acquisition sert souvent deux
    besoins, et le verdict n'est pas le même.

    Conclure de l'enseigne à la façade ferait porter à celle-ci une mesure
    prise sur autre chose.
    """
    log = PreviewAssessmentLog(hotel_id="pilote", entries=[
        _assessment(
            asset_id="a1", demand_id="obligation:PROPERTY_SIGN",
            verdict=PreviewVerdict.ESTABLISHED,
            rationale="l'enseigne est lisible",
        ),
        _assessment(
            asset_id="a1", demand_id="obligation:front",
            verdict=PreviewVerdict.REFUTED,
            in_frame_fraction=0.42,
            rationale="part dans le cadre 0,42 sous le minimum 0,50",
        ),
    ])

    assert log.established_for("obligation:PROPERTY_SIGN") == {"a1"}
    assert log.established_for("obligation:front") == set(), (
        "la façade n'est pas créditée par une mesure prise sur l'enseigne"
    )


def test_the_latest_assessment_prevails_without_erasing_the_first() -> None:
    """Un couple peut être réexaminé ; l'écraser effacerait ce qui a été vu."""
    from datetime import datetime, timedelta, timezone

    ancien = datetime.now(timezone.utc) - timedelta(hours=2)
    log = PreviewAssessmentLog(hotel_id="pilote", entries=[
        _assessment(
            verdict=PreviewVerdict.INCONCLUSIVE,
            unmeasured=["fraction visible"],
            rationale="occultation partielle, mesure impossible",
            assessed_at=ancien,
        ),
        _assessment(
            verdict=PreviewVerdict.ESTABLISHED,
            rationale="seconde mesure, cible entière visible",
        ),
    ])

    assert len(log.entries) == 2, "les deux constats subsistent"
    assert log.latest_for("a1", "obligation:front").verdict is (
        PreviewVerdict.ESTABLISHED
    )
    assert log.established_for("obligation:front") == {"a1"}


# --- ce qu'un constat ne peut pas dire ----------------------------------------


def test_an_established_verdict_cannot_leave_things_unmeasured() -> None:
    """Ce qu'on ignore ne peut pas fonder ce qu'on affirme."""
    with pytest.raises(ValueError, match="ne peut pas fonder"):
        _assessment(
            verdict=PreviewVerdict.ESTABLISHED,
            unmeasured=["largeur projetée"],
        )


def test_an_inconclusive_verdict_must_say_what_is_missing() -> None:
    """Sinon il ne se distingue pas d'un refus."""
    with pytest.raises(ValueError, match="sans rien d'inconnu"):
        _assessment(verdict=PreviewVerdict.INCONCLUSIVE, unmeasured=[])


def test_an_assessment_without_a_rationale_is_refused() -> None:
    """Un constat sans motif ne se conteste pas."""
    with pytest.raises(ValueError):
        _assessment(rationale="")


def test_an_assessment_names_the_file_it_looked_at() -> None:
    """Deux résolutions du même candidat sont deux fichiers : un constat pris
    sur l'un ne vaut pas pour l'autre."""
    with pytest.raises(ValueError):
        _assessment(request_digest="")
    with pytest.raises(ValueError):
        _assessment(checksum="")


# --- promotion vers la pleine résolution --------------------------------------


def test_a_demand_metric_left_unmeasured_blocks_promotion() -> None:
    """Payer la pleine résolution pour découvrir ce qu'un aperçu disait déjà."""
    demand = _demand(min_projected_width_fraction=0.15, min_visible_fraction=0.5)
    assessment = _assessment(projected_width_fraction=0.20)

    allowed, missing = promotable(assessment, demand)

    assert allowed is False
    assert missing == ["fraction visible non mesurée"]


def test_all_required_metrics_established_allows_promotion() -> None:
    demand = _demand(min_projected_width_fraction=0.15, min_visible_fraction=0.5)
    assessment = _assessment(
        projected_width_fraction=0.20, visible_fraction=0.80,
    )

    allowed, missing = promotable(assessment, demand)

    assert allowed is True and missing == []


def test_a_metric_below_the_threshold_blocks_promotion() -> None:
    demand = _demand(min_projected_width_fraction=0.15)
    assessment = _assessment(projected_width_fraction=0.09)

    allowed, missing = promotable(assessment, demand)

    assert allowed is False
    assert "sous le minimum" in missing[0]


def test_without_any_preview_nothing_is_promotable() -> None:
    allowed, missing = promotable(None, _demand())

    assert allowed is False
    assert missing == ["aucun aperçu examiné pour ce besoin"]


def test_a_refuted_preview_is_not_promotable() -> None:
    assessment = _assessment(
        verdict=PreviewVerdict.REFUTED,
        rationale="la façade est masquée par un véhicule",
    )

    allowed, missing = promotable(assessment, _demand())

    assert allowed is False
    assert "véhicule" in missing[0]


# --- ce que la provenance conserve --------------------------------------------


def test_the_provenance_keeps_what_the_file_was_for() -> None:
    """Sans `serves_demands`, la preview arrivait sans rattachement : on
    ignorait ce qu'elle venait vérifier."""
    from hotel_pipeline.schemas.acquisition import AcquisitionProvenance

    provenance = AcquisitionProvenance(
        provider_id="p1", candidate_id="c1", plan_id="plan-1",
        plan_digest="d" * 16, intents=[CaptureIntent.BUILDING_CAPTURE],
        serves_demands=["obligation:front", "obligation:PROPERTY_SIGN"],
        demand_levels={
            "obligation:front": "recommended_for_preview",
            "obligation:PROPERTY_SIGN": "eligible_for_full_acquisition",
        },
    )

    assert provenance.serves_demands == [
        "obligation:front", "obligation:PROPERTY_SIGN"
    ]
    assert provenance.demand_levels["obligation:front"] == (
        "recommended_for_preview"
    ), "le niveau est par besoin : un fichier peut servir deux exigences"


def test_refuting_every_preview_leaves_the_demand_open() -> None:
    """Rejeter une vue ne rend pas la façade inatteignable.

    Les confondre ferait d'un mauvais candidat une impossibilité, et le
    pipeline cesserait de chercher ce qui existe peut-être ailleurs.
    """
    log = PreviewAssessmentLog(hotel_id="pilote", entries=[
        _assessment(asset_id="a1", verdict=PreviewVerdict.REFUTED,
                    rationale="autoroute, aucun bâtiment d'hôtel"),
        _assessment(asset_id="a2", verdict=PreviewVerdict.REFUTED,
                    rationale="intérieur de concession automobile"),
    ])

    assert log.refuted_for("obligation:front") == {"a1", "a2"}
    assert log.established_for("obligation:front") == set()

    # Rien dans le journal ne dit que le besoin est clos : il n'a simplement
    # aucune preuve établie.
    assert not hasattr(log, "unreachable")
    assert not any(
        getattr(entry, "closes_demand", False) for entry in log.entries
    )
