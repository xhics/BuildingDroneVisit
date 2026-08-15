"""Câblage recherche adaptative → découverte (collecte V2).

Ce qui est éprouvé ici n'est pas que les fonctions existent — les tests de
`test_adaptive_search.py` s'en chargent — mais que la découverte les **appelle**
et publie ce qu'elles produisent. Sans ces tests, neutraliser entièrement la
passe adaptative laissait passer toute la suite : le socle était juste, et
personne ne s'en servait.

La frontière que ces tests protègent : la recherche présélectionne pour
enrichir, seul `assets plan` décide ce qui sera acquis. Un manifeste qui ne
retiendrait que les candidats recommandés effacerait la trace de ce qui a été
vu puis écarté.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from hotel_pipeline.discover import discover
from hotel_pipeline.schemas import DEFAULT_POLICY
from hotel_pipeline.schemas.acquisition import CaptureDemandManifest, Eligibility

from test_adaptive_search import TARGET, candidate, demand

HOTEL = "pilote-test"


@dataclass
class FakeContext:
    """Le contexte que le CLI construit, réduit à ce que `discover` consomme."""

    outstanding: list = field(default_factory=list)
    skipped: dict = field(default_factory=dict)
    anchors: dict = field(default_factory=dict)
    target: tuple | None = TARGET
    policy: object = field(
        default_factory=lambda: DEFAULT_POLICY.adaptive_search
    )
    lineage: dict = field(default_factory=dict)


def _demands(*demands):
    return CaptureDemandManifest(hotel_id=HOTEL, demands=list(demands))


def _south_of(target, metres: float, east: float = 0.0):
    """Une position décalée de la cible, en mètres."""
    lat, lon = target
    return (lat - metres / 111_320.0, lon + east / 78_000.0)


@pytest.fixture
def rear_demand():
    return demand("obligation:FACADE_REAR", "rear")


@pytest.fixture
def three_candidates():
    near = _south_of(TARGET, 40)
    side = _south_of(TARGET, 45, east=30)
    far = _south_of(TARGET, 55, east=-35)
    return [
        candidate("mly-near", *near),
        candidate("mly-side", *side),
        candidate("mly-far", *far),
    ]


def test_search_is_actually_invoked(rear_demand, three_candidates):
    """Contrôle négatif : sans passe adaptative, ce test tombe.

    C'est précisément ce qui manquait — la suite entière passait alors que la
    recherche pouvait être court-circuitée.
    """
    manifest, report = discover(
        HOTEL, _demands(rear_demand), {"mapillary": three_candidates},
        search=FakeContext(outstanding=[rear_demand]),
    )

    assert manifest.evaluations, "aucune évaluation : la passe n'a pas eu lieu"
    assert report.search is not None
    assert report.search.demands_searched == ["obligation:FACADE_REAR"]
    assert manifest.adaptive_search_run_id == report.run_id
    assert manifest.adaptive_search_report_digest


def test_every_candidate_stays_in_the_manifest(rear_demand, three_candidates):
    """Les non-recommandés restent : « écarté » n'est pas « jamais vu »."""
    manifest, _ = discover(
        HOTEL, _demands(rear_demand), {"mapillary": three_candidates},
        search=FakeContext(outstanding=[rear_demand]),
    )

    assert len(manifest.candidates) == 3
    assert len(manifest.recommended_for_plan) < 3, (
        "si tout est recommandé, ce test ne prouve plus la conservation"
    )
    written_off = set(c.candidate_id for c in manifest.candidates) - set(
        manifest.recommended_for_plan
    )
    evaluated = {e.candidate_id for e in manifest.evaluations}
    assert written_off <= evaluated, (
        "un candidat écarté sans évaluation ne s'explique pas"
    )


def test_recommendation_is_not_acquisition(rear_demand, three_candidates):
    """Recommander sert à enrichir ; le plan décide seul d'acquérir."""
    manifest, _ = discover(
        HOTEL, _demands(rear_demand), {"mapillary": three_candidates},
        search=FakeContext(outstanding=[rear_demand]),
    )

    assert all(
        e.eligibility in (Eligibility.PREVIEW_REQUIRED, Eligibility.REJECTED)
        for e in manifest.evaluations
    ), "aucune évaluation ne doit valoir feu vert d'acquisition"


def test_recommendations_must_exist_in_the_manifest(rear_demand, three_candidates):
    """Un identifiant recommandé absent du manifeste est refusé."""
    manifest, _ = discover(
        HOTEL, _demands(rear_demand), {"mapillary": three_candidates},
        search=FakeContext(outstanding=[rear_demand]),
    )

    with pytest.raises(ValueError, match="recommandés absents"):
        manifest.model_copy(
            update={"recommended_for_plan": ["mly-fantome"]}
        ).model_validate(manifest.model_dump() | {
            "recommended_for_plan": ["mly-fantome"]
        })


def test_counts_separate_requests_from_candidates(rear_demand, three_candidates):
    """Une source prolixe n'est pas une source souvent interrogée."""
    manifest, _ = discover(
        HOTEL, _demands(rear_demand), {"mapillary": three_candidates},
        search=FakeContext(outstanding=[rear_demand]),
    )

    counts = manifest.candidates_by_source["mapillary"]
    assert counts.returned == 3
    assert counts.unique == 3
    assert counts.recommended + counts.rejected == counts.unique


def test_skipped_demands_keep_their_reason(rear_demand, three_candidates):
    """Une cible non résolue est dite, pas remplacée en silence."""
    reason = "cible non résolue : ENTRANCE_MAIN_CURRENT"
    _, report = discover(
        HOTEL, _demands(rear_demand), {"mapillary": three_candidates},
        search=FakeContext(
            outstanding=[rear_demand],
            skipped={"obligation:ENTRANCE_MAIN_CURRENT": reason},
        ),
    )

    assert report.search.demands_skipped["obligation:ENTRANCE_MAIN_CURRENT"] == reason


def test_total_rejection_is_recorded_not_silent(rear_demand):
    """Tout écarter est un résultat ; un zéro nu ne le dirait pas."""
    # Un candidat à 8 km : hors de portée de n'importe quel seuil de distance.
    remote = candidate("mly-loin", TARGET[0] + 0.072, TARGET[1])

    manifest, report = discover(
        HOTEL, _demands(rear_demand), {"mapillary": [remote]},
        search=FakeContext(outstanding=[rear_demand]),
    )

    assert not manifest.recommended_for_plan
    assert report.search.all_rejected.get("obligation:FACADE_REAR") == 1
    assert manifest.candidates, "le candidat écarté reste au manifeste"


def test_lineage_is_carried_into_the_report(rear_demand, three_candidates):
    """Un rapport sans filiation ne se rattache à aucun état."""
    lineage = {
        "demand_digest": "d" * 16,
        "demand_assessment_digest": "a" * 16,
        "asset_manifest_digest": "m" * 16,
        "capture_geometry_digest": "g" * 16,
        "policy_dependency_digests": {"search_preference": "p" * 16},
    }
    _, report = discover(
        HOTEL, _demands(rear_demand), {"mapillary": three_candidates},
        search=FakeContext(outstanding=[rear_demand], lineage=lineage),
    )

    assert report.search.demand_digest == "d" * 16
    assert report.search.capture_geometry_digest == "g" * 16
    assert report.search.policy_dependency_digests["search_preference"] == "p" * 16
    assert report.search.hotel_id == HOTEL


def test_no_search_context_leaves_no_orphan_report(rear_demand, three_candidates):
    """Sans besoins ouverts, aucun rapport de recherche n'est fabriqué."""
    manifest, report = discover(
        HOTEL, _demands(rear_demand), {"mapillary": three_candidates},
        search=FakeContext(outstanding=[]),
    )

    assert report.search is None
    assert manifest.adaptive_search_run_id is None
    assert manifest.adaptive_search_report_digest is None
    assert manifest.candidates, "les candidats restent collectés"


def test_report_publishes_the_search(rear_demand, three_candidates):
    """Ce qui n'est pas au rapport publié n'a pas eu lieu, pour un lecteur."""
    _, report = discover(
        HOTEL, _demands(rear_demand), {"mapillary": three_candidates},
        search=FakeContext(outstanding=[rear_demand]),
    )

    published = report.as_dict()["adaptive_search"]
    assert published is not None
    assert published["run_id"] == report.run_id
    assert report.as_dict()["bytes_downloaded"] == 0
