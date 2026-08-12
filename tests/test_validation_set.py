"""Jeu de validation humaine du classifieur (Lot 1B §6).

Le §6 conditionne l'acceptation du classifieur à une matrice de confusion
lisible. Ce jeu de 36 images du corpus réel a été étiqueté à la main sur
planches-contacts ; les scores sont figés pour que la régression se détecte
sans charger de modèle.

Il a déjà attrapé deux défauts que les tests unitaires ne voyaient pas :
CLIP ignorant la négation, puis un opposé unique laissant les photographies
d'intérieur scorer 0,98 en « façade vue de l'extérieur ».
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hotel_pipeline.triage.classify import SUBJECT_ACCEPT, SUBJECT_REJECT

FIXTURE = Path(__file__).parent / "fixtures" / "validation_set.json"


@pytest.fixture(scope="module")
def validation() -> dict:
    return json.loads(FIXTURE.read_text("utf-8"))


def scores_for(validation: dict, group: str, subject: str) -> list[float]:
    return [
        validation["scores"][str(n)][subject]
        for n in validation["labels"][group]
        if str(n) in validation["scores"]
    ]


class TestBuildingDiscrimination:
    def test_hotel_views_are_accepted(self, validation):
        values = scores_for(validation, "hotel", "building")
        assert values, "aucune vue d'hôtel dans le jeu"
        assert min(values) >= SUBJECT_ACCEPT

    def test_interiors_are_not_mistaken_for_facades(self, validation):
        """Le défaut le plus grave rencontré : 0,98 sur une piscine intérieure."""
        assert max(scores_for(validation, "interior", "building")) < SUBJECT_REJECT

    def test_roads_alone_yield_no_building(self, validation):
        assert max(scores_for(validation, "road_only", "building")) < SUBJECT_REJECT

    def test_promotional_graphics_yield_no_building(self, validation):
        assert max(scores_for(validation, "stock_or_logo", "building")) < SUBJECT_REJECT

    def test_suburban_houses_are_not_hotels(self, validation):
        """Le voisinage est pavillonnaire : confondre serait fatal à la couverture."""
        assert max(scores_for(validation, "house", "building")) < SUBJECT_REJECT

    def test_separation_gap_is_wide(self, validation):
        """Les seuils doivent tomber dans un intervalle vide, pas sur un bord."""
        positives = min(scores_for(validation, "hotel", "building"))
        negatives = max(
            scores_for(validation, "interior", "building")
            + scores_for(validation, "road_only", "building")
            + scores_for(validation, "house", "building")
        )
        assert positives - negatives > 0.4


class TestInteriorDiscrimination:
    def test_interiors_are_recognised(self, validation):
        assert sorted(scores_for(validation, "interior", "interior"))[1] >= SUBJECT_ACCEPT

    def test_exterior_views_are_not_interiors(self, validation):
        assert max(scores_for(validation, "hotel", "interior")) < SUBJECT_REJECT
        assert max(scores_for(validation, "road_only", "interior")) < SUBJECT_REJECT


class TestCorpusComposition:
    """Ce que le jeu dit du corpus, et qui compte autant que le classifieur."""

    def test_no_promotional_source_shows_the_building(self, validation):
        """Places et site officiel ne fournissent aucune vue extérieure."""
        promotional = {
            n
            for n, source in validation["sources"].items()
            if source in {"places", "website"}
        }
        hotel_views = {str(n) for n in validation["labels"]["hotel"]}
        assert promotional & hotel_views == set()

    def test_street_view_supplies_the_only_building_views(self, validation):
        sources = {validation["sources"][str(n)] for n in validation["labels"]["hotel"]}
        assert sources == {"street_view"}

    def test_thresholds_stay_ordered(self):
        assert 0.0 < SUBJECT_REJECT < SUBJECT_ACCEPT < 1.0
