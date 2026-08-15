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
from test_adaptive_sector import FlatProjection, at_bearing, sector_context

HOTEL = "pilote-test"


@dataclass
class FakeContext:
    """Le contexte que le CLI construit, réduit à ce que `discover` consomme."""

    outstanding: list = field(default_factory=list)
    skipped: dict = field(default_factory=dict)
    anchors: dict = field(default_factory=dict)
    target: tuple | None = TARGET
    sector: object = None
    viewpoint_separation_m: float | None = None
    framing_merge_bearing_deg: float | None = None
    viewpoints: dict = field(default_factory=dict)
    policy: object = field(
        default_factory=lambda: DEFAULT_POLICY.adaptive_search
    )
    lineage: dict = field(default_factory=dict)


def _recommended(manifest) -> set:
    """Les trois niveaux réunis — utile quand seul l'ensemble importe."""
    return (
        set(manifest.recommended_for_enrichment)
        | set(manifest.recommended_for_preview)
        | set(manifest.eligible_for_full_acquisition)
    )


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
    recommended = _recommended(manifest)
    assert len(recommended) < 3, (
        "si tout est recommandé, ce test ne prouve plus la conservation"
    )
    written_off = set(c.candidate_id for c in manifest.candidates) - recommended
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
        manifest.model_validate(manifest.model_dump() | {
            "recommended_for_preview": ["mly-fantome"]
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


def test_distance_is_a_reprieve_not_a_verdict(rear_demand):
    """Hors portée automatique n'est pas inutilisable.

    Sans intrinsèques de caméra, la distance seule ne prouve pas qu'une cible
    serait trop petite. Faute de candidat plus proche, le lointain redevient
    donc examinable — plutôt que de laisser le besoin sans rien.
    """
    remote = candidate("mly-loin", TARGET[0] + 0.072, TARGET[1])

    manifest, report = discover(
        HOTEL, _demands(rear_demand), {"mapillary": [remote]},
        search=FakeContext(outstanding=[rear_demand]),
    )

    assert manifest.candidates, "le candidat lointain reste au manifeste"
    measure = report.search.measures[0]
    assert measure.outside_automatic_range
    assert "non écarté" in measure.preview_only_reason
    assert measure.rejection_reason is None, (
        "la distance met à l'écart, elle ne condamne pas"
    )


def test_a_closer_candidate_wins_over_a_distant_one(rear_demand):
    """Le repli ne doit pas mettre lointains et proches sur le même rang."""
    near = candidate("mly-proche", *_south_of(TARGET, 40))
    remote = candidate("mly-loin", TARGET[0] + 0.072, TARGET[1])

    manifest, _ = discover(
        HOTEL, _demands(rear_demand), {"mapillary": [near, remote]},
        search=FakeContext(outstanding=[rear_demand]),
    )

    assert "mly-loin" not in _recommended(manifest)
    assert "mly-proche" in _recommended(manifest)


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


def test_framings_panoramas_and_viewpoints_are_counted_separately(rear_demand):
    """1442 cadrages pour 721 panoramas n'est pas un doublon.

    Un seul chiffre les confondrait, et le rapport se lirait comme si la
    déduplication avait échoué alors que ce sont deux acquisitions légitimes.
    """
    framings = [
        candidate("pano-A-1", *_south_of(TARGET, 40), panorama_id="A"),
        candidate("pano-A-2", *_south_of(TARGET, 40), panorama_id="A"),
        candidate("pano-B-1", *_south_of(TARGET, 45, east=30), panorama_id="B"),
    ]

    manifest, report = discover(
        HOTEL, _demands(rear_demand), {"street_view": framings},
        search=FakeContext(outstanding=[rear_demand]),
    )

    counts = report.viewpoint_counts
    assert counts["framing_candidates"] == 3
    assert counts["distinct_panoramas"] == 2
    assert counts["viewpoints"] == 2
    assert report.duplicates_dropped == 0, (
        "deux cadrages ne sont pas des doublons d'identité"
    )
    assert len(manifest.candidates) == 3


def test_a_fallback_recommendation_says_so(rear_demand):
    """Retenu faute de mieux n'est pas retenu ordinairement."""
    remote = candidate("mly-loin", TARGET[0] + 0.072, TARGET[1])

    _, report = discover(
        HOTEL, _demands(rear_demand), {"mapillary": [remote]},
        search=FakeContext(outstanding=[rear_demand]),
    )

    measure = report.search.measures[0]
    assert measure.recommended_by_fallback, (
        "sans cette marque, le plan prendrait un repli pour un choix"
    )


# --- la découverte transmet-elle ce qu'elle a reçu ? --------------------------
#
# Ces deux tests comblent un angle mort : neutraliser la transmission de
# `sector` ou de `viewpoints` entre `discover` et la mesure ne faisait échouer
# aucun test. Le contexte était construit, jamais vérifié comme transmis.


def test_discover_passes_the_sector_context_through():
    """Sans transmission, les besoins sectoriels cessent de se distinguer."""
    need = demand("obligation:front", "front")
    front = candidate("c-face", *at_bearing(5.0))
    behind = candidate("c-dos", *at_bearing(180.0))

    _, report = discover(
        HOTEL, _demands(need), {"mapillary": [front, behind]},
        search=FakeContext(
            outstanding=[need], target=(0.0, 0.0),
            sector=sector_context(need.demand_id, 0.0),
        ),
    )

    fits = {m.candidate_id: m.sector_fit.value for m in report.search.measures}
    assert fits["c-face"] == "exact"
    assert fits["c-dos"] == "wrong_sector", (
        "le contexte sectoriel n'a pas atteint la mesure"
    )


def test_discover_passes_the_viewpoint_grouping_through():
    """Sans transmission, deux cadrages rempliraient un quota de deux vues."""
    need = demand("obligation:front", "front", viewpoints_required=2)
    framings = [
        candidate("pano-A-1", *at_bearing(5.0), panorama_id="A"),
        candidate("pano-A-2", *at_bearing(5.0), panorama_id="A"),
    ]

    manifest, _ = discover(
        HOTEL, _demands(need), {"street_view": framings},
        search=FakeContext(
            outstanding=[need], target=(0.0, 0.0),
            sector=sector_context(need.demand_id, 0.0),
        ),
    )

    assert len(_recommended(manifest)) == 1, (
        "deux cadrages d'un même panorama ne remplissent pas un quota de deux "
        "points de vue"
    )


def test_a_stage_that_did_not_run_says_so(rear_demand, three_candidates):
    """Zéro appel et zéro résultat produisent le même chiffre.

    Sans déclaration explicite, la seconde passe Mapillary pouvait rester non
    câblée en présentant les compteurs d'une recherche complète.
    """
    _, report = discover(
        HOTEL, _demands(rear_demand), {"mapillary": three_candidates},
        search=FakeContext(outstanding=[rear_demand]),
    )

    skipped = report.search.stages_skipped
    assert "metadata_enrichment" in skipped
    assert "sequence_expansion" in skipped
    assert "request_provenance" in skipped
    assert report.search.enrichment_calls == 0
    assert all(reason for reason in skipped.values()), (
        "une étape sautée sans motif ne vaut pas mieux qu'un zéro"
    )


def test_discover_actually_merges_near_identical_framings(rear_demand):
    """La fonction de regroupement doit être **appelée**, pas seulement juste.

    Le pilote produisait trois cadrages par panorama, dont deux séparés de
    1,5° : deux requêtes pour une même image.
    """
    lat, lon = _south_of(TARGET, 40)
    framings = [
        candidate("sv-1", lat, lon, panorama_id="A", requested_heading_deg=131.8),
        candidate("sv-2", lat, lon, panorama_id="A", requested_heading_deg=133.3),
        candidate("sv-3", lat, lon, panorama_id="A", requested_heading_deg=199.7),
    ]

    manifest, report = discover(
        HOTEL, _demands(rear_demand), {"street_view": framings},
        search=FakeContext(outstanding=[rear_demand], framing_merge_bearing_deg=15.0),
    )

    kept = {c.candidate_id for c in manifest.candidates}
    assert len(kept) == 2, "les cadrages voisins n'ont pas été regroupés"
    assert report.framings_merged == {"sv-2": "sv-1"}, (
        "l'écarté doit rester nommé dans la trace d'audit"
    )


def test_sequences_reach_the_continuity_measure(rear_demand):
    """La séquence traverse collecteur → candidat → mesure.

    `sequence` vient dans la même requête Mapillary : l'enrichissement n'exige
    aucun appel supplémentaire. Sans ce câblage, la continuité restait
    `not_queried` et tout besoin l'exigeant demeurait borné à l'aperçu.
    """
    near, side = _south_of(TARGET, 40), _south_of(TARGET, 45, east=30)
    pair = [
        candidate("mly-1", *near, sequence_id="seq-A"),
        candidate("mly-2", *side, sequence_id="seq-A"),
    ]

    _, report = discover(
        HOTEL, _demands(rear_demand), {"mapillary": pair},
        search=FakeContext(outstanding=[rear_demand]),
    )

    statuses = {m.sequence_status.value for m in report.search.measures}
    assert statuses == {"known"}, "la séquence n'a pas atteint la mesure"
    assert {m.sequence_id for m in report.search.measures} == {"seq-A"}


def test_a_source_without_sequences_says_not_returned(rear_demand, three_candidates):
    """« Le fournisseur n'en a pas rendu » n'est pas « nous n'avons pas
    demandé »."""
    _, report = discover(
        HOTEL, _demands(rear_demand), {"mapillary": three_candidates},
        search=FakeContext(outstanding=[rear_demand]),
    )

    statuses = {m.sequence_status.value for m in report.search.measures}
    assert statuses == {"not_returned"}
    assert all(m.sequence_id is None for m in report.search.measures)


def test_the_collector_sequence_reaches_the_candidate():
    """Le premier maillon : `CollectedImage` → `CaptureCandidate`.

    Les tests suivants construisent des candidats directement ; sans celui-ci,
    couper la recopie dans `candidates_from` ne faisait rien échouer.
    """
    from hotel_pipeline.collectors.base import CollectedImage
    from hotel_pipeline.discover import candidates_from

    image = CollectedImage(
        source="mapillary", source_id="1", url="http://exemple",
        lat=45.0, lon=-73.0, sequence_id="seq-A",
    )
    built = candidates_from("mapillary", [image])[0]

    assert built.sequence_id == "seq-A"
    assert "thumb_256" in built.available_resolutions, (
        "un plan demandant un aperçu doit pouvoir l'obtenir"
    )
