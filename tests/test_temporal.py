"""Datation par portée (Lot 1B, audit du câblage).

Un statut global ne suffisait pas : une photographie peut montrer une entrée
rénovée et une façade inchangée. Et remplacer le verrou humain par une
dérivation automatique aurait perdu la seule information qu'aucune date de
fichier ne porte.
"""

from __future__ import annotations

from datetime import date

import pytest

from hotel_pipeline.schemas import (
    DEFAULT_POLICY,
    Asset,
    AssetCategory,
    PropertyProfile,
    Rights,
    Subject,
    TemporalDecision,
    TemporalStatus,
)
from hotel_pipeline.schemas.profile import RenovationEvent
from hotel_pipeline.temporal import (
    appearance_allowed,
    assess,
    derive_scope,
    undetermined_sensitive_scopes,
)


def make(asset_id="a", **overrides) -> Asset:
    fields = dict(
        id=asset_id, source="mapillary", source_url_or_id="u", rights=Rights.OPEN_DATA,
        ai_eligible=False, confidence=0.5, category=AssetCategory.OTHER, checksum="a" * 64,
    )
    fields.update(overrides)
    return Asset(**fields)


@pytest.fixture
def profile() -> PropertyProfile:
    return PropertyProfile(
        country_code="CA", timezone="America/Toronto", ocr_languages=["fr", "en"],
        property_id="p", address="a", official_name="X",
        renovation_events=[
            RenovationEvent(
                event_id="entree", scope="entrance",
                started_on=date(2024, 10, 1),
                completed_on=date(2025, 3, 1), completion_confirmed=True,
            ),
            RenovationEvent(
                event_id="approbation-facade", scope="facade",
                approved_on=date(2024, 9, 16),
            ),
        ],
    )


class TestDerivationRules:
    def test_capture_after_confirmed_completion_is_current(self, profile):
        status, method = derive_scope(make(capture_year=2026), profile, "entrance")
        assert status is TemporalStatus.CURRENT_CONFIRMED
        assert "achèvement confirmé" in method

    def test_capture_before_attested_start_is_earlier(self, profile):
        status, _ = derive_scope(make(capture_year=2023), profile, "entrance")
        assert status is TemporalStatus.BEFORE_EVENT

    def test_capture_during_the_works_year_stays_unknown(self, profile):
        """Une année ne situe pas une image à l'intérieur de cette année."""
        assert derive_scope(make(capture_year=2025), profile, "entrance")[0] is (
            TemporalStatus.UNKNOWN
        )

    def test_approval_only_never_resolves(self, profile):
        """Une approbation ne prouve ni le début ni la fin des travaux."""
        status, method = derive_scope(make(capture_year=2026), profile, "facade")
        assert status is TemporalStatus.UNKNOWN
        assert "insuffisant" in method

    def test_missing_capture_date_stays_unknown(self, profile):
        status, method = derive_scope(make(), profile, "entrance")
        assert status is TemporalStatus.UNKNOWN
        assert "date de capture inconnue" in method

    def test_scope_without_declared_works_stays_unknown(self, profile):
        assert derive_scope(make(capture_year=2026), profile, "roof")[0] is TemporalStatus.UNKNOWN

    def test_no_profile_yields_unknown(self):
        assert derive_scope(make(capture_year=2026), None, "entrance")[0] is (
            TemporalStatus.UNKNOWN
        )


class TestHumanDecisionWins:
    def test_decision_overrides_derivation(self, profile):
        asset = make(
            capture_year=2023,
            temporal_decisions=[
                TemporalDecision(
                    scope="entrance", status=TemporalStatus.CURRENT_CONFIRMED,
                    decided_by="hm", rationale="fournie par l'hôtel après travaux",
                )
            ],
        )
        status, method = derive_scope(asset, profile, "entrance")
        assert status is TemporalStatus.CURRENT_CONFIRMED
        assert "revue humaine" in method

    def test_decision_survives_a_reassessment(self, profile):
        assets = [
            make(
                capture_year=2023,
                temporal_decisions=[
                    TemporalDecision(
                        scope="entrance", status=TemporalStatus.CURRENT_CONFIRMED,
                        decided_by="hm", rationale="preuve datée",
                    )
                ],
            )
        ]
        assess(assets, profile)
        assert assets[0].temporal_by_scope["entrance"] is TemporalStatus.CURRENT_CONFIRMED

    def test_decision_on_one_scope_leaves_others_derived(self, profile):
        assets = [
            make(
                capture_year=2026,
                temporal_decisions=[
                    TemporalDecision(
                        scope="facade", status=TemporalStatus.HISTORICAL,
                        decided_by="hm", rationale="cliché ancien republié",
                    )
                ],
            )
        ]
        assess(assets, profile)
        assert assets[0].temporal_by_scope["facade"] is TemporalStatus.HISTORICAL
        assert assets[0].temporal_by_scope["entrance"] is TemporalStatus.CURRENT_CONFIRMED


class TestScopesAreIndependent:
    def test_one_photo_can_be_current_on_one_scope_and_not_another(self, profile):
        assets = [make(capture_year=2026)]
        assess(assets, profile)
        scopes = assets[0].temporal_by_scope
        assert scopes["entrance"] is TemporalStatus.CURRENT_CONFIRMED
        assert scopes["facade"] is TemporalStatus.UNKNOWN

    def test_aggregate_takes_the_most_restrictive(self, profile):
        assets = [make(capture_year=2023)]
        assess(assets, profile)
        assert assets[0].temporal_status is TemporalStatus.BEFORE_EVENT


class TestSensitiveScopesOnly:
    def test_an_asset_not_showing_the_scope_is_not_blocking(self, profile):
        """Bloquer une image qui ne montre pas l'entrée serait absurde."""
        asset = make(subjects=[Subject.ROAD], temporal_by_scope={"entrance": TemporalStatus.UNKNOWN})
        assert undetermined_sensitive_scopes(asset) == []

    def test_an_asset_showing_an_undated_entrance_is_blocking(self):
        asset = make(
            subjects=[Subject.ENTRANCE], temporal_by_scope={"entrance": TemporalStatus.UNKNOWN}
        )
        assert undetermined_sensitive_scopes(asset) == ["entrance"]

    def test_a_dated_entrance_is_not_blocking(self):
        asset = make(
            subjects=[Subject.ENTRANCE],
            temporal_by_scope={"entrance": TemporalStatus.CURRENT_CONFIRMED},
        )
        assert undetermined_sensitive_scopes(asset) == []


class TestAppearanceVersusGeometry:
    def test_undated_appearance_is_refused_by_default(self):
        asset = make(temporal_by_scope={"entrance": TemporalStatus.UNKNOWN})
        assert appearance_allowed(asset, "entrance") is False

    def test_current_appearance_is_allowed(self):
        asset = make(temporal_by_scope={"entrance": TemporalStatus.CURRENT_CONFIRMED})
        assert appearance_allowed(asset, "entrance") is True

    def test_policy_can_permit_undated_appearance(self):
        lenient = DEFAULT_POLICY.model_copy(deep=True)
        lenient.temporal.allow_unknown_for_appearance = True
        asset = make(temporal_by_scope={"entrance": TemporalStatus.UNKNOWN})
        assert appearance_allowed(asset, "entrance", lenient) is True

    def test_geometry_remains_allowed_when_appearance_is_not(self):
        """La séparation des deux usages est le cœur de la correction."""
        from hotel_pipeline.roles import role_for
        from hotel_pipeline.schemas import ClusterRole, ReconstructionRole, ReviewStatus

        asset = make(
            camera_lat=45.573, camera_lon=-73.443, subjects=[Subject.BUILDING],
            target_building_visible=True, review_status=ReviewStatus.AUTOMATIC_ACCEPTED,
            cluster_role=ClusterRole.CANONICAL, temporal_status=TemporalStatus.UNKNOWN,
            temporal_by_scope={"entrance": TemporalStatus.UNKNOWN},
            **usable(),
        )
        assert role_for(asset)[0] is ReconstructionRole.PHOTO_GEOMETRY
        assert appearance_allowed(asset, "entrance") is False


def usable(suitability="primary", by="hm", rationale="façade franche, lignes raccordables"):
    """Champs d'une aptitude géométrique établie.

    Une vue n'est plus porteuse du seul fait qu'on y reconnaît l'hôtel :
    l'aptitude est une décision distincte, et elle exige son historique.
    """
    from hotel_pipeline.review import assessment_fields
    from hotel_pipeline.schemas import GeometrySuitability

    return assessment_fields(
        GeometrySuitability(suitability), by, rationale,
        ["cadrage et netteté vérifiés sur la façade"], "a" * 64,
    )
