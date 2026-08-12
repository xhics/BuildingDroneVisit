"""Migration vers la structure du Lot 1B (§13 étape 1, §14).

Critère d'acceptation : les anciens manifestes se chargent, et **aucune
catégorie ambiguë n'est transformée silencieusement en certitude**.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hotel_pipeline.migration import migrate
from hotel_pipeline.schemas import (
    Asset,
    AssetCategory,
    AssetManifest,
    CaptureType,
    EntranceVersion,
    ExteriorInterior,
    ReconstructionRole,
    ReviewStatus,
    Rights,
    Subject,
    TemporalStatus,
    ViewSector,
)


def legacy_asset(**overrides) -> Asset:
    """Un asset tel que produit avant le Lot 1B, sans les nouveaux champs."""
    fields = dict(
        id="mapillary-1",
        source="mapillary",
        source_url_or_id="https://x.invalid/1.jpg",
        rights=Rights.OPEN_DATA,
        ai_eligible=False,
        confidence=0.9,
        category=AssetCategory.FACADE,
        checksum="a" * 64,
    )
    fields.update(overrides)
    return Asset(**fields)


class TestBackwardCompatibility:
    def test_legacy_asset_still_loads(self):
        """Un manifeste antérieur doit se charger sans migration préalable."""
        asset = legacy_asset()
        assert asset.source_family is None
        assert asset.view_sector is ViewSector.UNKNOWN
        assert asset.review_status is ReviewStatus.NEEDS_REVIEW

    def test_default_role_is_never_geometry(self):
        """Un asset ne devient source de géométrie que sur décision explicite."""
        assert legacy_asset().reconstruction_role is ReconstructionRole.REFERENCE_ONLY

    def test_invalid_enum_is_rejected(self):
        with pytest.raises(ValidationError):
            legacy_asset(view_sector="derrière-à-gauche-ish")

    def test_unknown_subject_is_rejected(self):
        with pytest.raises(ValidationError):
            legacy_asset(subjects=["batiment"])


class TestDeterministicDerivation:
    def test_source_family_derived(self):
        manifest, report = migrate(AssetManifest(hotel_id="h", assets=[legacy_asset()]))
        assert manifest.assets[0].source_family == "mapillary"
        assert report.source_family_set == 1

    def test_capture_type_derived_from_source(self):
        manifest, _ = migrate(
            AssetManifest(
                hotel_id="h",
                assets=[
                    legacy_asset(id="a", source="mapillary"),
                    legacy_asset(id="b", source="website"),
                    legacy_asset(id="c", source="tripadvisor", rights=Rights.PUBLIC_UNCLEARED),
                ],
            )
        )
        by_id = {a.id: a for a in manifest.assets}
        assert by_id["a"].capture_type is CaptureType.STREET_IMAGERY
        assert by_id["b"].capture_type is CaptureType.PROMOTIONAL
        assert by_id["c"].capture_type is CaptureType.TRAVELER

    def test_temporal_status_follows_human_decision(self):
        manifest, _ = migrate(
            AssetManifest(
                hotel_id="h",
                assets=[legacy_asset(entrance_version=EntranceVersion.POST_2024)],
            )
        )
        assert manifest.assets[0].temporal_status is TemporalStatus.POST_2024

    def test_identical_checksums_share_an_exact_group(self):
        manifest, _ = migrate(
            AssetManifest(
                hotel_id="h",
                assets=[legacy_asset(id="a"), legacy_asset(id="b"), legacy_asset(id="c", checksum="b" * 64)],
            )
        )
        by_id = {a.id: a for a in manifest.assets}
        assert by_id["a"].exact_duplicate_group == by_id["b"].exact_duplicate_group
        assert by_id["c"].exact_duplicate_group != by_id["a"].exact_duplicate_group

    def test_placeholder_checksum_creates_no_group(self):
        """Un checksum non calculé ne doit pas fusionner des images distinctes."""
        manifest, _ = migrate(
            AssetManifest(
                hotel_id="h",
                assets=[legacy_asset(id="a", checksum="0" * 64), legacy_asset(id="b", checksum="0" * 64)],
            )
        )
        assert all(a.exact_duplicate_group is None for a in manifest.assets)


class TestNoSilentCertainty:
    """Le cœur du critère d'acceptation de l'étape 1."""

    def test_view_sector_stays_unknown(self):
        """La catégorie « façade » ne dit pas depuis quel secteur on regarde."""
        manifest, _ = migrate(
            AssetManifest(hotel_id="h", assets=[legacy_asset(category=AssetCategory.FACADE)])
        )
        assert manifest.assets[0].view_sector is ViewSector.UNKNOWN

    def test_legacy_classification_is_flagged_for_review(self):
        manifest, _ = migrate(
            AssetManifest(
                hotel_id="h",
                assets=[legacy_asset(exterior_or_interior=ExteriorInterior.EXTERIOR)],
            )
        )
        asset = manifest.assets[0]
        assert asset.review_status is ReviewStatus.NEEDS_REVIEW
        assert asset.classification_method == "legacy_openclip_single_label"

    def test_subjects_come_from_evidence_not_from_category(self):
        """Seules la géométrie et l'OCR peuvent poser un sujet à la migration."""
        manifest, _ = migrate(
            AssetManifest(
                hotel_id="h",
                assets=[
                    legacy_asset(id="devine", category=AssetCategory.PARKING),
                    legacy_asset(id="mesure", sees_building=True),
                    legacy_asset(id="lu", sign_text="HOTEL WELCOMINNS"),
                ],
            )
        )
        by_id = {a.id: a for a in manifest.assets}
        assert by_id["devine"].subjects == []
        assert Subject.BUILDING in by_id["mesure"].subjects
        assert Subject.SIGN in by_id["lu"].subjects

    def test_role_is_never_upgraded_by_migration(self):
        manifest, _ = migrate(
            AssetManifest(hotel_id="h", assets=[legacy_asset(sees_building=True)])
        )
        assert manifest.assets[0].reconstruction_role is ReconstructionRole.REFERENCE_ONLY


class TestIdempotence:
    def test_second_migration_changes_nothing(self):
        manifest = AssetManifest(hotel_id="h", assets=[legacy_asset()])
        once, _ = migrate(manifest)
        snapshot = once.model_dump_json()

        twice, report = migrate(once)
        assert twice.model_dump_json() == snapshot
        assert report.already_migrated == 1

    def test_report_lists_unmapped_sources(self):
        _, report = migrate(
            AssetManifest(hotel_id="h", assets=[legacy_asset(source="source-inconnue")])
        )
        assert "source-inconnue" in report.unmapped_sources


class TestCounting:
    def test_viewpoints_counted_not_files(self):
        """Les Gates comptent des points de vue, jamais des fichiers (§5)."""
        manifest = AssetManifest(
            hotel_id="h",
            assets=[
                legacy_asset(id="a", viewpoint_cluster="vp-1"),
                legacy_asset(id="b", viewpoint_cluster="vp-1"),
                legacy_asset(id="c", viewpoint_cluster="vp-2"),
            ],
        )
        assert manifest.viewpoints() == 2

    def test_republications_count_once(self):
        manifest = AssetManifest(
            hotel_id="h",
            assets=[
                legacy_asset(id="expedia", perceptual_duplicate_group="p-1"),
                legacy_asset(id="kayak", perceptual_duplicate_group="p-1"),
                legacy_asset(id="autre", perceptual_duplicate_group="p-2"),
            ],
        )
        assert manifest.unique_photographs() == 2
