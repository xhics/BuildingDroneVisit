"""Séparation stricte entre présence d'un bâtiment et visibilité de la cible.

Défaut mesuré : `contains_building` était recalculé depuis la liste `subjects`,
qui fusionne modèle, géométrie et OCR. La géométrie forçait donc le drapeau à
vrai, et 11 vues sur 13 portaient la mention « bâtiment confirmé » avec des
scores modèle descendant jusqu'à 0,0006.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hotel_pipeline.classify_cascade import classify
from hotel_pipeline.coverage import street_view_coverage
from hotel_pipeline.schemas import (
    Asset,
    AssetCategory,
    PropertyMatchStatus,
    ReconstructionRole,
    ReviewDecision,
    ReviewEntry,
    ReviewStatus,
    Rights,
    Subject,
)
from hotel_pipeline.schemas.assets import DECISION_STATUS, VISIBILITY_OF
from hotel_pipeline.triage.classify import MultiLabelResult


def make(asset_id="a", **overrides) -> Asset:
    fields = dict(
        id=asset_id,
        source="mapillary",
        source_url_or_id="https://x.invalid/1.jpg",
        rights=Rights.OPEN_DATA,
        ai_eligible=False,
        confidence=0.5,
        category=AssetCategory.OTHER,
        checksum="a" * 64,
    )
    fields.update(overrides)
    return Asset(**fields)


class FakeClassifier:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def multi_label(self, _path) -> MultiLabelResult:  # noqa: ANN001
        return MultiLabelResult(scores=self.scores)


@pytest.fixture
def image(tmp_path) -> str:
    path = tmp_path / "i.jpg"
    path.write_bytes(b"x")
    return str(path)


class TestNoGeometryLeak:
    """Le test que l'audit réclamait explicitement."""

    def test_measured_fov_with_weak_model_score_is_not_the_target(self, image):
        assets = [
            make(
                sees_building=True,
                heading_is_measured=True,
                local_path=image,
            )
        ]
        classify(assets, classifier=FakeClassifier({"building": 0.01}))
        asset = assets[0]

        assert asset.contains_building is False
        assert asset.target_building_visible is not True
        assert "aucun bâtiment détecté" in asset.target_evidence

    def test_measured_fov_with_strong_model_score_is_the_target(self, image):
        assets = [make(sees_building=True, heading_is_measured=True, local_path=image)]
        classify(assets, classifier=FakeClassifier({"building": 0.95}))

        assert assets[0].contains_building is True
        assert assets[0].target_building_visible is True

    def test_contains_building_ignores_geometry_added_subject(self, image):
        """`subjects` contient BUILDING par la géométrie, pas `contains_building`."""
        assets = [make(sees_building=True, heading_is_measured=True, local_path=image)]
        classify(assets, classifier=FakeClassifier({"building": 0.02}))

        assert Subject.BUILDING in assets[0].subjects  # posé par la géométrie
        assert assets[0].contains_building is False    # mais le modèle dit non

    def test_without_a_classifier_content_stays_unevaluated(self):
        assets = [make(sees_building=True, heading_is_measured=True)]
        classify(assets)
        assert assets[0].contains_building is None
        assert assets[0].target_building_visible is None

    def test_weak_score_never_reaches_photo_geometry(self, image):
        from hotel_pipeline.roles import role_for

        assets = [
            make(
                camera_lat=45.573,
                camera_lon=-73.443,
                sees_building=True,
                heading_is_measured=True,
                local_path=image,
            )
        ]
        classify(assets, classifier=FakeClassifier({"building": 0.0006}))
        assert role_for(assets[0])[0] is not ReconstructionRole.PHOTO_GEOMETRY


class TestIdentityAndVisibilityAreSeparateAxes:
    def test_sign_alone_does_not_prove_the_building_is_visible(self, image):
        """Une photo d'un panneau isolé ne porte pas de géométrie."""
        assets = [
            make(
                property_match_status=PropertyMatchStatus.MATCH,
                sign_text="HOTEL WELCOMINNS",
                local_path=image,
            )
        ]
        classify(assets, classifier=FakeClassifier({"building": 0.01, "sign": 0.99}))
        assert assets[0].target_building_visible is not True

    def test_sign_plus_building_confirms_the_target(self, image):
        assets = [
            make(
                property_match_status=PropertyMatchStatus.MATCH,
                sign_text="HOTEL WELCOMINNS",
                local_path=image,
            )
        ]
        classify(assets, classifier=FakeClassifier({"building": 0.95, "sign": 0.99}))
        assert assets[0].target_building_visible is True

    def test_competitor_sign_rejects_regardless_of_building(self, image):
        assets = [
            make(property_match_status=PropertyMatchStatus.MISMATCH, local_path=image)
        ]
        classify(assets, classifier=FakeClassifier({"building": 0.99}))
        assert assets[0].target_building_visible is False


def reviewed(decision: ReviewDecision, rationale: str, **overrides) -> dict:
    """Champs d'un asset réellement passé en revue.

    Le manifeste refuse désormais une décision humaine sans historique : elle
    serait invérifiable, et rien ne dirait qui l'a prise ni sur quoi.
    """
    return dict(
        target_visibility_decision=decision,
        target_building_visible=VISIBILITY_OF[decision],
        review_status=DECISION_STATUS[decision],
        reviewer="hm",
        review_rationale=rationale,
        review_evidence=["capture annotée"],
        review_history=[
            ReviewEntry(
                decision=decision,
                decided_by="hm",
                rationale=rationale,
                evidence=["capture annotée"],
                reviewed_checksum="a" * 64,
            )
        ],
        **overrides,
    )


class TestHumanDecisionWins:
    def test_confirmed_decision_survives_a_reclassification(self, image):
        assets = [
            make(
                local_path=image,
                **reviewed(ReviewDecision.CONFIRMED, "façade visible à gauche du cadre"),
            )
        ]
        classify(assets, classifier=FakeClassifier({"building": 0.001}))
        assert assets[0].target_building_visible is True
        assert "revue humaine" in assets[0].target_evidence

    def test_rejected_decision_survives_a_strong_model_score(self, image):
        assets = [
            make(
                local_path=image,
                sees_building=True,
                **reviewed(ReviewDecision.REJECTED, "c'est le concessionnaire voisin"),
            )
        ]
        classify(assets, classifier=FakeClassifier({"building": 0.99}))
        assert assets[0].target_building_visible is False

    def test_unresolved_leaves_the_automatic_path_intact(self, image):
        assets = [make(local_path=image, sees_building=True, heading_is_measured=True)]
        classify(assets, classifier=FakeClassifier({"building": 0.99}))
        assert assets[0].target_building_visible is True


class TestCoverageReport:
    def test_categories_partition_the_positions(self):
        """La somme des quatre catégories égale le nombre de positions."""
        assets = [
            make("v", source="street_view", target_building_visible=True),
            make("o", source="street_view", occluded_by="way/9"),
            make("w", source="street_view", property_match_status=PropertyMatchStatus.MISMATCH),
            make("c", source="street_view", target_building_visible=False),
            make("u", source="street_view", target_building_visible=None),
        ]
        coverage = street_view_coverage(assets)
        assert coverage.positions == 5
        assert coverage.partition_total == 5

    def test_undetermined_is_a_subset_of_context_only(self):
        """Il ne s'ajoute pas au total, sous peine de double comptage."""
        assets = [
            make("u1", source="street_view", target_building_visible=None),
            make("c1", source="street_view", target_building_visible=False),
        ]
        coverage = street_view_coverage(assets)
        assert coverage.context_only == 2
        assert coverage.undetermined == 1
        assert coverage.partition_total == coverage.positions

    def test_other_sources_are_excluded(self):
        assets = [make("m", source="mapillary", target_building_visible=True)]
        assert street_view_coverage(assets).positions == 0

    def test_visible_views_are_broken_down_by_sector(self):
        from hotel_pipeline.schemas import ViewSector

        assets = [
            make("a", source="street_view", target_building_visible=True,
                 view_sector=ViewSector.FRONT),
            make("b", source="street_view", target_building_visible=True,
                 view_sector=ViewSector.FRONT),
            make("c", source="street_view", target_building_visible=True,
                 view_sector=ViewSector.REAR),
        ]
        coverage = street_view_coverage(assets)
        assert coverage.by_sector == {"front": 2, "rear": 1}
