"""Non-régression sur le corpus réel (Lot 1B, câblage).

Instantané des 329 assets du WelcomINNS, figé après le câblage de la politique
et du profil. Avec la politique par défaut, le pipeline doit reproduire
**exactement** les mêmes comptes : la généralisation ne devait rien changer au
résultat, seulement à la façon dont il est paramétré.

Aucun appel réseau ni modèle : les scores du classifieur sont figés dans
l'instantané, seules les étapes déterministes sont rejouées.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hotel_pipeline import dedup_levels, roles
from hotel_pipeline.coverage import street_view_coverage
from hotel_pipeline.schemas import DEFAULT_POLICY, Asset

FIXTURE = Path(__file__).parent / "fixtures" / "corpus_snapshot.json"


@pytest.fixture(scope="module")
def snapshot() -> dict:
    return json.loads(FIXTURE.read_text("utf-8"))


@pytest.fixture
def assets(snapshot) -> list[Asset]:
    return [Asset.model_validate(a) for a in snapshot["assets"]]


class TestCorpusIsStable:
    def test_snapshot_covers_the_whole_corpus(self, assets):
        assert len(assets) == 329

    def test_deduplication_reproduces_the_snapshot(self, assets, snapshot):
        building = snapshot["building"]
        report = dedup_levels.run(
            assets, building["lat"], building["lon"], policy=DEFAULT_POLICY
        )
        assert report.as_dict() == snapshot["expected"]["dedup"]

    def test_roles_reproduce_the_snapshot(self, assets, snapshot):
        building = snapshot["building"]
        dedup_levels.run(assets, building["lat"], building["lon"], policy=DEFAULT_POLICY)
        report = roles.assign(assets, policy=DEFAULT_POLICY)
        assert report.as_dict() == snapshot["expected"]["roles"]

    def test_street_view_coverage_reproduces_the_snapshot(self, assets, snapshot):
        report = street_view_coverage(assets)
        assert report.as_dict() == snapshot["expected"]["coverage"]


class TestKeyFindingsHold:
    """Les conclusions tirées du corpus, protégées contre une régression."""

    def test_no_asset_carries_geometry_before_its_aptitude_is_assessed(
        self, assets, snapshot
    ):
        """Deux vues franchissent tous les prédicats d'identité et de position.

        Elles ne portent pourtant aucune géométrie tant que personne n'a dit ce
        qu'elles montrent réellement de la structure : l'aptitude n'est pas
        déduite de la reconnaissance.
        """
        building = snapshot["building"]
        dedup_levels.run(assets, building["lat"], building["lon"])
        report = roles.assign(assets)
        carriers = [a for a in assets if a.reconstruction_role.value == "photo_geometry"]

        assert carriers == []
        assert report.reasons["aptitude géométrique non évaluée"] == 3

    def test_geometry_candidates_are_all_measured_headings(self, assets, snapshot):
        """Aucune vue à cap choisi ne doit reparaître comme candidate."""
        building = snapshot["building"]
        dedup_levels.run(assets, building["lat"], building["lon"])
        report = roles.assign(assets)
        candidates = [
            a for a in assets
            if roles.role_for(a)[1] == "aptitude géométrique non évaluée"
        ]
        assert len(candidates) == report.reasons["aptitude géométrique non évaluée"]
        assert all(a.heading_is_measured for a in candidates)

    def test_no_street_view_position_is_confirmed_visible(self, assets):
        assert street_view_coverage(assets).visible == 0

    def test_occluded_positions_are_counted(self, assets):
        assert street_view_coverage(assets).occluded == 29


class TestPolicyChangesTheOutcome:
    """L'instantané vaut pour la politique par défaut, et pour elle seule."""

    def test_a_stricter_policy_changes_the_counts(self, assets, snapshot):
        building = snapshot["building"]
        strict = DEFAULT_POLICY.model_copy(deep=True)
        strict.dedup.position_tolerance_m = 1.0

        report = dedup_levels.run(assets, building["lat"], building["lon"], policy=strict)
        assert report.viewpoints != snapshot["expected"]["dedup"]["independent_viewpoints"]
