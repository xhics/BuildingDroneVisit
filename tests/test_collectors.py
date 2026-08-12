"""Collecteurs, droits à la source et tri (§9, §11)."""

from __future__ import annotations

import pytest
from PIL import Image
from pydantic import ValidationError

from hotel_pipeline.collectors import POLICIES, CollectedImage, to_asset
from hotel_pipeline.schemas import Asset, AssetCategory, PropertyMatchStatus, Rights
from hotel_pipeline.triage import evaluate, group_duplicates, normalise, phash
from hotel_pipeline.triage.quality import audit_basic, basic_scores, normalised_quality


def image_of(source: str, source_id: str = "1") -> CollectedImage:
    return CollectedImage(source=source, source_id=source_id, url="https://x.invalid/1.jpg")


class TestRightsAtSource:
    def test_mapillary_is_open_data(self):
        asset = to_asset(image_of("mapillary"))
        assert asset.rights is Rights.OPEN_DATA
        assert not asset.rights_encumbered
        assert "CC BY-SA" in asset.attribution

    def test_street_view_is_uncleared_by_default(self):
        asset = to_asset(image_of("street_view"))
        assert asset.rights is Rights.PUBLIC_UNCLEARED
        assert not asset.rights_encumbered
        assert not asset.usable_in_production

    def test_assuming_rights_marks_the_asset(self):
        """La décision de l'opérateur est inscrite, pas dissoute (§9)."""
        asset = to_asset(image_of("street_view"), assume_rights=True)
        assert asset.rights_encumbered
        assert asset.usable_in_production
        assert asset.rights_note

    def test_assumption_does_not_touch_clean_sources(self):
        """Assumer des droits ne doit rien changer là où ils sont déjà bons."""
        asset = to_asset(image_of("mapillary"), assume_rights=True)
        assert not asset.rights_encumbered

    def test_every_policy_declares_attribution_or_note(self):
        for policy in POLICIES.values():
            assert policy.attribution or policy.note, policy.name


class TestEncumberedPromotion:
    def test_encumbered_asset_may_be_promoted(self):
        asset = to_asset(image_of("street_view"), assume_rights=True)
        promoted = asset.model_copy(update={"production_eligible": True})
        Asset.model_validate(promoted.model_dump())

    def test_uncleared_without_assumption_still_refused(self):
        asset = to_asset(image_of("street_view"))
        with pytest.raises(ValidationError, match="production_eligible"):
            Asset.model_validate(
                asset.model_copy(update={"production_eligible": True}).model_dump()
            )

    def test_encumbered_flag_is_meaningless_on_clean_rights(self):
        with pytest.raises(ValidationError, match="n'a pas de sens"):
            Asset.model_validate(
                to_asset(image_of("mapillary")).model_copy(
                    update={"rights_encumbered": True}
                ).model_dump()
            )


class TestSignOcr:
    """Le risque nº1 du §3 devient mesurable."""

    def test_expected_name_matches_despite_accents_and_case(self):
        reading = evaluate("HÔTEL WELCOMINNS\n1195 rue Ampère", ["WelcomINNS"], ["Mortagne"])
        assert reading.status is PropertyMatchStatus.MATCH

    def test_excluded_name_wins_over_expected(self):
        """Une enseigne du voisin disqualifie l'image, même si l'hôtel est cité."""
        reading = evaluate("Hôtel Mortagne — voisin du WelcomINNS", ["WelcomINNS"], ["Mortagne"])
        assert reading.status is PropertyMatchStatus.MISMATCH
        assert reading.matched_term == "Mortagne"

    def test_no_readable_sign_is_uncertain(self):
        assert evaluate("PARKING", ["WelcomINNS"], ["Mortagne"]).status is (
            PropertyMatchStatus.UNCERTAIN
        )

    def test_normalisation_strips_accents_and_punctuation(self):
        assert normalise("Hôtel  WelcomINNS!") == "hotel welcominns"


class TestDedup:
    @pytest.fixture
    def images(self, tmp_path):
        paths = {}
        base = Image.new("RGB", (128, 128), (30, 90, 160))
        for x in range(0, 128, 8):
            for y in range(0, 128, 16):
                base.putpixel((x, y), (240, 240, 10))

        paths["a"] = tmp_path / "a.jpg"
        base.save(paths["a"])

        # Même cliché, réenregistré : doit tomber dans le même groupe.
        paths["a_copy"] = tmp_path / "a_copy.jpg"
        base.save(paths["a_copy"], quality=60)

        different = Image.new("RGB", (128, 128), (10, 10, 10))
        for x in range(0, 128, 3):
            for y in range(0, 128, 3):
                different.putpixel((x, y), (255, 0, 0))
        paths["b"] = tmp_path / "b.jpg"
        different.save(paths["b"])
        return paths

    def test_near_duplicates_share_a_group(self, images):
        hashes = {name: phash(path) for name, path in images.items()}
        groups = group_duplicates(hashes)
        assert groups["a"] == groups["a_copy"]

    def test_distinct_images_are_separated(self, images):
        hashes = {name: phash(path) for name, path in images.items()}
        groups = group_duplicates(hashes)
        assert groups["b"] != groups["a"]

    def test_empty_hashes_yield_no_groups(self):
        assert group_duplicates({}) == {}


class TestQuality:
    def test_flat_image_is_detected_as_blurry(self, tmp_path):
        path = tmp_path / "flat.jpg"
        Image.new("RGB", (64, 64), (128, 128, 128)).save(path)
        assert path.name in audit_basic([path]).blurry

    def test_black_image_is_dark(self, tmp_path):
        path = tmp_path / "black.jpg"
        Image.new("RGB", (64, 64), (0, 0, 0)).save(path)
        assert path.name in audit_basic([path]).dark

    def test_white_image_is_light(self, tmp_path):
        path = tmp_path / "white.jpg"
        Image.new("RGB", (64, 64), (255, 255, 255)).save(path)
        assert path.name in audit_basic([path]).light

    def test_quality_score_is_bounded(self, tmp_path):
        path = tmp_path / "flat.jpg"
        Image.new("RGB", (64, 64), (128, 128, 128)).save(path)
        assert 0.0 <= normalised_quality(basic_scores(path)) <= 1.0


class TestTinyImages:
    """Les pixels de suivi ne doivent pas produire de NaN (§11, G3)."""

    def test_one_pixel_image_scores_without_nan(self, tmp_path):
        import math

        path = tmp_path / "pixel.png"
        Image.new("RGB", (1, 1), (255, 255, 255)).save(path)
        scores = basic_scores(path)
        assert not math.isnan(scores["sharpness"])
        assert scores["sharpness"] == 0.0

    def test_tiny_image_is_never_considered_sharp(self, tmp_path):
        path = tmp_path / "tiny.png"
        Image.new("RGB", (2, 2), (10, 200, 10)).save(path)
        assert normalised_quality(basic_scores(path)) == 0.0
