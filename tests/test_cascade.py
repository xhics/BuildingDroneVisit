"""Cascade de catégorisation et secteurs (Lot 1B §6, §13 étape 3, §14).

Les deux exigences centrales : une confiance insuffisante produit `unknown`
plutôt qu'un choix par défaut, et l'appartenance ne se déduit jamais de la
catégorie.
"""

from __future__ import annotations

import pytest

from hotel_pipeline.classify_cascade import classify, property_status
from hotel_pipeline.schemas import (
    Asset,
    AssetCategory,
    CaptureType,
    PropertyMatchStatus,
    ReviewStatus,
    Rights,
    Subject,
    ViewSector,
)
from hotel_pipeline.sectors import sector_for
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
    """Classifieur factice : les seuils se testent sans modèle ni poids."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def multi_label(self, _path) -> MultiLabelResult:  # noqa: ANN001
        return MultiLabelResult(scores=self.scores)


class TestSectorGeometry:
    """L'azimut d'observation ne devient un secteur qu'avec une façade avant."""

    def test_standing_in_front_direction_sees_the_front(self):
        assert sector_for(observer_bearing_deg=180, front_azimuth_deg=180) is ViewSector.FRONT

    def test_opposite_side_is_the_rear(self):
        assert sector_for(observer_bearing_deg=0, front_azimuth_deg=180) is ViewSector.REAR

    def test_quarter_turn_is_a_side(self):
        assert sector_for(270, 180) is ViewSector.RIGHT
        assert sector_for(90, 180) is ViewSector.LEFT

    def test_diagonal_is_a_corner(self):
        assert sector_for(225, 180) is ViewSector.FRONT_RIGHT_CORNER

    def test_every_bearing_maps_to_a_sector(self):
        """Les huit zones couvrent 360° : aucun azimut ne reste sans secteur."""
        assert all(
            sector_for(bearing, 180.0) is not ViewSector.UNKNOWN
            for bearing in range(0, 360, 7)
        )

    def test_transition_is_never_produced_by_geometry(self):
        """`transition` est sémantique — une vue route-entrée-parking —,
        pas une plage angulaire."""
        assert all(
            sector_for(bearing, 180.0) is not ViewSector.TRANSITION
            for bearing in range(0, 360, 7)
        )


class TestCascadeOrder:
    def test_geometry_establishes_the_building_without_any_model(self):
        """Aucune probabilité ne doit être requise pour ce qui est mesuré."""
        assets = [make(sees_building=True)]
        classify(assets)
        assert Subject.BUILDING in assets[0].subjects
        assert "geometry" in assets[0].classification_method

    def test_street_imagery_always_shows_the_road(self):
        assets = [make(capture_type=CaptureType.STREET_IMAGERY)]
        classify(assets)
        assert Subject.ROAD in assets[0].subjects

    def test_ocr_establishes_a_sign_not_an_identity(self):
        """Lire une enseigne prouve qu'il y en a une, pas que c'est la bonne."""
        assets = [make(sign_text="HOTEL MORTAGNE")]
        classify(assets)
        assert Subject.SIGN in assets[0].subjects
        assert assets[0].property_match_status is PropertyMatchStatus.UNCERTAIN

    def test_sector_stays_unknown_without_a_front_azimuth(self):
        assets = [make(sees_building=True, bearing_from_building_deg=180.0)]
        classify(assets, front_azimuth=None)
        assert assets[0].view_sector is ViewSector.UNKNOWN

    def test_sector_derived_when_front_is_known(self):
        assets = [make(sees_building=True, bearing_from_building_deg=180.0)]
        classify(assets, front_azimuth=180.0)
        assert assets[0].view_sector is ViewSector.FRONT


class TestThresholds:
    def test_confident_subject_is_accepted(self, tmp_path):
        path = tmp_path / "i.jpg"
        path.write_bytes(b"x")
        assets = [make(local_path=str(path))]
        classify(assets, classifier=FakeClassifier({"parking": 0.95, "building": 0.05}))
        assert Subject.PARKING in assets[0].subjects

    def test_weak_scores_produce_no_subject(self, tmp_path):
        """Le défaut du classifieur précédent : désigner un vainqueur quand même."""
        path = tmp_path / "i.jpg"
        path.write_bytes(b"x")
        assets = [make(local_path=str(path))]
        classify(assets, classifier=FakeClassifier({"parking": 0.30, "building": 0.10}))
        assert assets[0].subjects == []

    def test_ambiguous_decisive_subject_forces_review(self, tmp_path):
        """Un doute sur le bâtiment décide de la couverture : il va en revue."""
        path = tmp_path / "i.jpg"
        path.write_bytes(b"x")
        assets = [make(local_path=str(path))]
        classify(assets, classifier=FakeClassifier({"building": 0.55}))
        assert assets[0].review_status is ReviewStatus.NEEDS_REVIEW

    def test_flat_scores_yield_low_confidence(self):
        result = MultiLabelResult(scores={"building": 0.5, "parking": 0.5})
        assert result.confidence() == pytest.approx(0.0)

    def test_decisive_scores_yield_high_confidence(self):
        result = MultiLabelResult(scores={"building": 0.99, "parking": 0.02})
        assert result.confidence() > 0.9

    def test_model_never_overrides_geometry(self, tmp_path):
        """Le modèle complète la mesure, il ne la contredit pas."""
        path = tmp_path / "i.jpg"
        path.write_bytes(b"x")
        assets = [make(sees_building=True, local_path=str(path))]
        classify(assets, classifier=FakeClassifier({"building": 0.01}))
        assert Subject.BUILDING in assets[0].subjects


class TestIndependenceOfDecisions:
    """Appartenance et catégorie sont deux décisions distinctes (§6)."""

    def test_geometry_confirms_belonging(self):
        assert property_status(make(sees_building=True), ["WelcomINNS"], []) is (
            PropertyMatchStatus.MATCH
        )

    def test_competitor_sign_disqualifies(self):
        asset = make(sign_text="HÔTEL MORTAGNE")
        assert property_status(asset, ["WelcomINNS"], ["Hôtel Mortagne"]) is (
            PropertyMatchStatus.MISMATCH
        )

    def test_a_beautiful_facade_proves_nothing_about_identity(self):
        """Le classifieur ne doit jamais fonder l'appartenance."""
        asset = make(category=AssetCategory.FACADE)
        assert property_status(asset, ["WelcomINNS"], ["Hôtel Mortagne"]) is (
            PropertyMatchStatus.UNCERTAIN
        )


class TestReport:
    def test_report_counts_review_and_unknown_sectors(self):
        assets = [make("a"), make("b", sees_building=True)]
        report = classify(assets)
        assert report.total == 2
        assert report.unknown_sector == 2
        assert "unknown" in report.sectors_assigned


class TestPromptConstruction:
    """CLIP ne traite pas la négation (régression mesurée sur le corpus réel).

    Le prompt opposé « a photo with no building visible » contenait le mot
    « building » et l'emportait sur une image de bâtiment : 0 hôtel détecté sur
    118 vues Street View dont la première montrait clairement le WelcomINNS.
    """

    def test_no_opposite_prompt_uses_negation(self):
        from hotel_pipeline.triage.classify import SUBJECT_PROMPTS

        forbidden = (" no ", "without", "not ", "n't")
        offenders = [
            f"{subject}:{alt}"
            for subject, (_, alternatives) in SUBJECT_PROMPTS.items()
            for alt in alternatives
            if any(token in f" {alt.lower()} " for token in forbidden)
        ]
        assert offenders == [], f"négation dans les alternatives : {offenders}"

    def test_alternatives_cover_the_real_corpus(self):
        """Un opposé unique laissait les intérieurs scorer 0,98 en façade.

        Les alternatives doivent représenter le monde du corpus — intérieurs,
        routes, pavillons, visuels promotionnels — et non un seul contre-exemple.
        """
        from hotel_pipeline.triage.classify import SUBJECT_PROMPTS

        for subject, (_, alternatives) in SUBJECT_PROMPTS.items():
            assert len(alternatives) >= 3, f"{subject} : trop peu d'alternatives"

    def test_every_subject_declares_a_positive_and_alternatives(self):
        from hotel_pipeline.triage.classify import SUBJECT_PROMPTS

        for subject, (positive, alternatives) in SUBJECT_PROMPTS.items():
            assert positive.strip(), subject
            assert all(a.strip() for a in alternatives), subject


class TestHeadingProvenance:
    """Un cap observé et un cap choisi ne valent pas la même preuve.

    Défaut mesuré : la géométrie attribuait le sujet « bâtiment » à 105 vues
    Street View, dont 20 seulement étaient confirmées par le modèle. Le cap
    étant dirigé par nous vers l'empreinte, la visibilité s'y déduisait de
    notre propre intention.
    """

    def test_measured_heading_establishes_the_building(self):
        """Imagerie de roulage : le cap est celui qu'un conducteur a adopté."""
        assets = [make(sees_building=True, heading_is_measured=True)]
        classify(assets)
        assert Subject.BUILDING in assets[0].subjects

    def test_chosen_heading_establishes_only_the_aim(self):
        """Street View : viser l'empreinte ne prouve rien sur le contenu."""
        assets = [make(sees_building=True, heading_is_measured=False)]
        classify(assets)
        assert Subject.BUILDING not in assets[0].subjects
        assert "aim_only" in assets[0].classification_method

    def test_model_alone_can_still_confirm_a_chosen_heading(self, tmp_path):
        path = tmp_path / "i.jpg"
        path.write_bytes(b"x")
        assets = [make(sees_building=True, heading_is_measured=False, local_path=str(path))]
        classify(assets, classifier=FakeClassifier({"building": 0.95}))
        assert Subject.BUILDING in assets[0].subjects

    def test_sector_is_still_derived_from_a_chosen_heading(self):
        """La position reste une mesure, même quand la visée ne l'est pas."""
        assets = [
            make(sees_building=True, heading_is_measured=False, bearing_from_building_deg=180.0)
        ]
        classify(assets, front_azimuth=180.0)
        assert assets[0].view_sector is ViewSector.FRONT

    def test_street_view_collector_declares_a_chosen_heading(self):
        from hotel_pipeline.collectors.streetview import Panorama, _image_for

        image = _image_for(Panorama("P", 45.57, -73.44, "2025-05", "©"), 90.0, 50.0)
        assert image.heading_is_measured is False

    def test_mapillary_collector_keeps_a_measured_heading(self):
        from hotel_pipeline.collectors.base import CollectedImage

        assert CollectedImage(source="mapillary", source_id="1", url="u").heading_is_measured
