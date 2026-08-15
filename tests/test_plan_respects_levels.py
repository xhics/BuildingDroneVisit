"""Le plan est contraint par les niveaux de la recherche (collecte V2).

Les trois listes étaient publiées et ignorées : `assets plan` transmettait les
2 357 candidats à `build()`, `select()` ne filtrait que les rejets, et
`PlannedAcquisition` retenait sa résolution par défaut. Une vue bornée à
l'aperçu entrait donc au plan en pleine résolution.

Ce que ces tests protègent : un niveau publié contraint, il n'informe pas.
"""

from __future__ import annotations

import pytest

from hotel_pipeline.plan import select
from hotel_pipeline.schemas.acquisition import (
    CandidateEvaluation,
    CaptureCandidate,
    CaptureDemand,
    CaptureIntent,
    Eligibility,
    TargetKind,
)

PREVIEW = "recommended_for_preview"
FULL = "eligible_for_full_acquisition"
ENRICH = "recommended_for_enrichment"


def _demand(demand_id="obligation:front", required=1, **overrides):
    fields = dict(
        demand_id=demand_id, intent=CaptureIntent.BUILDING_CAPTURE,
        target_kind=TargetKind.VIEW_SECTOR, target_ref="front",
        viewpoints_required=required,
    )
    fields.update(overrides)
    return CaptureDemand(**fields)


def _candidate(candidate_id, **overrides):
    fields = dict(
        candidate_id=candidate_id, source="street_view", provider_id=candidate_id,
        camera_lat=45.5, camera_lon=-73.4, heading_is_measured=True,
    )
    fields.update(overrides)
    return CaptureCandidate(**fields)


def _evaluation(candidate_id, demand_id="obligation:front",
                eligibility=Eligibility.PREVIEW_REQUIRED):
    fields = dict(
        candidate_id=candidate_id, demand_id=demand_id,
        intent=CaptureIntent.BUILDING_CAPTURE, eligibility=eligibility,
    )
    if eligibility is Eligibility.REJECTED:
        # Le schéma exige le motif : un rejet muet ne s'explique pas.
        fields["rejection_reason"] = "hors secteur"
    return CandidateEvaluation(**fields)


def test_a_preview_is_planned_as_a_thumbnail() -> None:
    """Vérifier ce qu'une vue montre ne demande pas la pleine résolution."""
    planned = select(
        [_evaluation("c-apercu")], [_demand()],
        candidates={"c-apercu": _candidate("c-apercu")},
        levels={("c-apercu", "obligation:front"): PREVIEW},
        preview_resolution="256", full_resolution="2048",
    )

    assert len(planned) == 1
    assert planned[0].resolution == "256", (
        "une preview en 2048 dépenserait le volume avant de savoir s'il le valait"
    )
    assert PREVIEW in planned[0].selection_rationale


def test_a_fully_eligible_candidate_is_planned_at_full_resolution() -> None:
    planned = select(
        [_evaluation("c-complet", eligibility=Eligibility.ELIGIBLE)], [_demand()],
        candidates={"c-complet": _candidate("c-complet")},
        levels={("c-complet", "obligation:front"): FULL},
        preview_resolution="256", full_resolution="2048",
    )

    assert planned[0].resolution == "2048"


def test_an_enrichment_recommendation_is_also_a_thumbnail() -> None:
    """Un appel de métadonnées ne justifie pas une image entière."""
    planned = select(
        [_evaluation("c-enrichir")], [_demand()],
        candidates={"c-enrichir": _candidate("c-enrichir")},
        levels={("c-enrichir", "obligation:front"): ENRICH},
        preview_resolution="256", full_resolution="2048",
    )

    assert planned[0].resolution == "256"


def test_an_unrecommended_candidate_is_never_selected() -> None:
    """Le plan ne repêche pas ce que la recherche n'a recommandé à rien.

    Sans cette règle, les 2 357 candidats restaient sélectionnables et les
    trois listes ne contraignaient rien.
    """
    planned = select(
        [_evaluation("c-recommande"), _evaluation("c-jamais-vu")],
        [_demand(required=2)],
        candidates={
            "c-recommande": _candidate("c-recommande"),
            "c-jamais-vu": _candidate("c-jamais-vu"),
        },
        levels={("c-recommande", "obligation:front"): PREVIEW},
    )

    retained = {item.candidate_id for item in planned}
    assert retained == {"c-recommande"}
    assert "c-jamais-vu" not in retained


def test_without_any_levels_the_plan_keeps_its_former_behaviour() -> None:
    """Aucun niveau **du tout** signifie « aucune recherche », non « rien
    d'autorisé ».

    Refuser tout dans ce cas casserait les espaces de travail antérieurs à la
    recherche adaptative, sans rien protéger.
    """
    planned = select(
        [_evaluation("c-1")], [_demand()],
        candidates={"c-1": _candidate("c-1")},
        levels=None,
    )

    assert len(planned) == 1
    assert planned[0].resolution == "2048"


def test_a_rejected_candidate_stays_out_whatever_its_level() -> None:
    planned = select(
        [_evaluation("c-refuse", eligibility=Eligibility.REJECTED)], [_demand()],
        candidates={"c-refuse": _candidate("c-refuse")},
        levels={("c-refuse", "obligation:front"): FULL},
    )

    assert planned == []


# --- regroupement des cadrages quasi identiques -------------------------------


def test_near_identical_framings_merge_with_an_audit_trail() -> None:
    """Deux cadrages à 1,5° montrent la même chose et coûtent deux requêtes."""
    from hotel_pipeline.discover import merge_near_identical_framings

    framings = [
        _candidate("pano-A-1", panorama_id="A", requested_heading_deg=131.8),
        _candidate("pano-A-2", panorama_id="A", requested_heading_deg=133.3),
        _candidate("pano-A-3", panorama_id="A", requested_heading_deg=199.7),
    ]

    kept, merged = merge_near_identical_framings(framings, 15.0)

    assert len(kept) == 2, "les deux cadrages voisins n'en font qu'un"
    assert merged == {"pano-A-2": "pano-A-1"}, (
        "l'écarté est nommé avec son remplaçant : un compteur ne dirait pas lequel"
    )


def test_distinct_framings_of_one_panorama_are_preserved() -> None:
    """Fusionner par panorama effacerait des cadrages réellement différents."""
    from hotel_pipeline.discover import merge_near_identical_framings

    framings = [
        _candidate("pano-A-1", panorama_id="A", requested_heading_deg=0.0),
        _candidate("pano-A-2", panorama_id="A", requested_heading_deg=90.0),
    ]

    kept, merged = merge_near_identical_framings(framings, 15.0)

    assert len(kept) == 2
    assert merged == {}


def test_a_candidate_without_a_framing_is_left_alone() -> None:
    """Sans cap déclaré, rien ne dit que deux vues se confondent."""
    from hotel_pipeline.discover import merge_near_identical_framings

    framings = [
        _candidate("mly-1", source="mapillary", panorama_id=None),
        _candidate("mly-2", source="mapillary", panorama_id=None),
    ]

    kept, merged = merge_near_identical_framings(framings, 15.0)

    assert len(kept) == 2
    assert merged == {}


# --- l'autorisation vaut pour un besoin, jamais pour le candidat --------------


def test_a_full_level_for_one_demand_does_not_serve_another() -> None:
    """Le cas réel du pilote : le seul candidat pleinement éligible l'est pour
    le stationnement, et possède des évaluations non rejetées pour l'entrée et
    l'enseigne.

    Un niveau porté par le seul `candidate_id` laissait cette autorisation
    couvrir des besoins qui ne l'avaient jamais recommandé.
    """
    parking, facade = "obligation:PARKING_HOTEL", "obligation:front"
    planned = select(
        [
            _evaluation("c-park", demand_id=parking, eligibility=Eligibility.ELIGIBLE),
            _evaluation("c-park", demand_id=facade),
        ],
        [_demand(parking), _demand(facade)],
        candidates={"c-park": _candidate("c-park")},
        levels={("c-park", parking): FULL},
    )

    assert len(planned) == 1
    assert planned[0].serves_demands == [parking], (
        "l'autorisation obtenue pour le stationnement ne couvre pas la façade"
    )


def test_preview_for_one_demand_and_full_for_another_never_promotes() -> None:
    """Le plus prudent l'emporte : un fichier, une résolution."""
    parking, facade = "obligation:PARKING_HOTEL", "obligation:front"
    planned = select(
        [
            _evaluation("c-double", demand_id=parking,
                        eligibility=Eligibility.ELIGIBLE),
            _evaluation("c-double", demand_id=facade),
        ],
        [_demand(parking), _demand(facade)],
        candidates={"c-double": _candidate("c-double")},
        levels={("c-double", parking): FULL, ("c-double", facade): PREVIEW},
        preview_resolution="256", full_resolution="2048",
    )

    assert sorted(planned[0].serves_demands) == sorted([parking, facade])
    assert planned[0].resolution == "256", (
        "servir aussi un besoin borné à l'aperçu ne promeut pas le fichier"
    )


def test_a_search_that_recommended_nothing_plans_nothing() -> None:
    """`{}` signifie « cherché, rien recommandé » — non « aucune contrainte ».

    Confondre les deux réactivait tout : une évaluation preview était
    sélectionnée alors que la recherche n'avait rien retenu.
    """
    planned = select(
        [_evaluation("c-1")], [_demand()],
        candidates={"c-1": _candidate("c-1")},
        levels={},
    )

    assert planned == [], "un registre vide n'autorise aucun candidat"


def test_a_legacy_manifest_without_a_search_run_stays_planable() -> None:
    """`None` reste la compatibilité explicite, reconnue à l'absence de run."""
    planned = select(
        [_evaluation("c-1")], [_demand()],
        candidates={"c-1": _candidate("c-1")},
        levels=None,
    )

    assert len(planned) == 1


def test_the_legacy_mode_follows_the_search_run_not_the_registry() -> None:
    """Un manifeste issu d'une recherche n'est jamais traité en legacy."""
    from hotel_pipeline.cli import _recommendation_levels

    class Searched:
        adaptive_search_run_id = "20260815T000000Z"
        recommendations: list = []

    class Legacy:
        adaptive_search_run_id = None
        recommendations: list = []

    assert _recommendation_levels(Searched()) == {}, (
        "cherché sans rien recommander : registre vide, pas legacy"
    )
    assert _recommendation_levels(Legacy()) is None


def test_framings_with_different_optics_are_never_merged() -> None:
    """Même panorama, même cap, mais pas la même image.

    Un gros plan et un grand angle pris dans la même direction ne montrent pas
    la même chose : les réunir sur le seul cap perdrait l'un des deux.
    """
    from hotel_pipeline.discover import merge_near_identical_framings

    for field, other in (
        ("requested_fov_deg", 30.0),
        ("requested_pitch_deg", 15.0),
        ("advertised_width", 1024),
    ):
        base = dict(
            panorama_id="A", requested_heading_deg=131.8, requested_fov_deg=80.0,
            requested_pitch_deg=0.0, advertised_width=2048, advertised_height=2048,
        )
        twin = dict(base, requested_heading_deg=133.3, **{field: other})
        kept, merged = merge_near_identical_framings(
            [_candidate("f-1", **base), _candidate("f-2", **twin)], 15.0
        )

        assert len(kept) == 2, f"{field} différent : ce ne sont pas deux fois la même vue"
        assert merged == {}


def test_identical_optics_and_close_bearings_still_merge() -> None:
    """Sans quoi le durcissement supprimerait le regroupement lui-même."""
    from hotel_pipeline.discover import merge_near_identical_framings

    base = dict(
        panorama_id="A", requested_fov_deg=80.0, requested_pitch_deg=0.0,
        advertised_width=2048, advertised_height=2048,
    )
    kept, merged = merge_near_identical_framings(
        [
            _candidate("f-1", requested_heading_deg=131.8, **base),
            _candidate("f-2", requested_heading_deg=133.3, **base),
        ],
        15.0,
    )

    assert len(kept) == 1
    assert merged == {"f-2": "f-1"}
