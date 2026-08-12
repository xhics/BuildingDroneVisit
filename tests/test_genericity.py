"""Généricité — le code ne connaît aucun établissement (Lot 1B).

Trois spécificités avaient été gravées dans le code : des identifiants
nominatifs dans le registre d'objets critiques, une date de rénovation promue
au rang de type, et une emprise calibrée sur un hôtel de 116 chambres.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from hotel_pipeline.schemas import DEFAULT_POLICY, PipelinePolicy, PropertyProfile
from hotel_pipeline.schemas.critical_objects import EXCLUDED_KINDS, REQUIRED_OBJECTS
from hotel_pipeline.schemas.profile import RenovationEvent


class TestNoPropertyLeaksIntoCode:
    def test_no_named_property_in_the_object_template(self):
        """`PROPERTY_WELCOMINNS` et `RUE_AMPERE` étaient des constantes."""
        blob = " ".join(REQUIRED_OBJECTS + EXCLUDED_KINDS).lower()
        for token in ("welcominns", "ampere", "ampère", "mortagne", "boucherville"):
            assert token not in blob

    def test_excluded_entries_are_kinds_not_names(self):
        """Le parc-o-bus De Mortagne est une instance, pas un type."""
        assert "PARK_AND_RIDE" in EXCLUDED_KINDS
        assert not any("DE_" in kind for kind in EXCLUDED_KINDS)

    def test_temporal_vocabulary_carries_no_date(self):
        from hotel_pipeline.schemas import EntranceVersion, TemporalStatus

        values = [m.value for m in EntranceVersion] + [m.value for m in TemporalStatus]
        assert not any(any(c.isdigit() for c in value) for value in values)


class TestPropertyProfile:
    @pytest.fixture
    def profile(self) -> PropertyProfile:
        return PropertyProfile.model_validate_json(
            Path("profiles/welcominns-boucherville.json").read_text("utf-8")
        )

    def test_real_profile_loads(self, profile):
        assert profile.official_name == "Hôtel WelcomINNS"
        assert profile.room_count == 116

    def test_identity_terms_include_aliases(self, profile):
        terms = profile.identity_terms()
        assert "Hôtel WelcomINNS" in terms
        assert "Welcom Inns" in terms

    def test_competitor_terms_are_full_names(self, profile):
        """Exclure le jeton « Mortagne » disqualifiait l'hôtel lui-même."""
        assert profile.excluded_terms() == ["Hôtel Mortagne"]
        assert all(" " in term for term in profile.excluded_terms())

    def test_footprint_derived_from_rooms_and_levels(self, profile):
        low, high = profile.footprint_range_m2()
        assert low < 1823 < high  # l'empreinte réelle du WelcomINNS

    def test_small_motel_is_not_excluded_by_a_hardcoded_range(self):
        """Une plage figée à 1 500 m² écartait tout établissement modeste."""
        motel = PropertyProfile(
            property_id="m", address="a", official_name="Motel", room_count=20
        )
        low, high = motel.footprint_range_m2()
        assert low < 400 < high

    def test_tall_tower_keeps_a_small_footprint_plausible(self):
        """600 chambres sur 20 niveaux occupent au sol moins qu'un motel.

        Une borne calculée sans les étages plaçait le minimum à 4 800 m² et
        éliminait l'empreinte réelle.
        """
        tower = PropertyProfile(
            property_id="t", address="a", official_name="Tour",
            room_count=600, expected_levels=20,
        )
        low, high = tower.footprint_range_m2()
        assert low <= 750
        assert high >= 2100

    def test_unknown_levels_widen_rather_than_exclude(self):
        tower = PropertyProfile(
            property_id="t", address="a", official_name="Tour", room_count=600
        )
        low, high = tower.footprint_range_m2()
        assert low <= 1250
        assert high >= 30_000

    def test_profile_without_size_falls_back_to_wide_bounds(self):
        unknown = PropertyProfile(property_id="u", address="a", official_name="X")
        low, high = unknown.footprint_range_m2()
        assert low < 1000 < high

    def test_incoherent_bounds_rejected(self):
        with pytest.raises(ValueError, match="footprint_min_m2"):
            PropertyProfile(
                property_id="x", address="a", official_name="X",
                footprint_min_m2=5000, footprint_max_m2=1000,
            )


class TestRenovationEvents:
    @pytest.fixture
    def profile(self) -> PropertyProfile:
        return PropertyProfile(
            property_id="p", address="a", official_name="X",
            renovation_events=[
                RenovationEvent(
                    event_id="e1", scope="facade",
                    completed_on=date(2019, 5, 1), completion_confirmed=True,
                ),
                RenovationEvent(
                    event_id="e2", scope="entrance",
                    approved_on=date(2024, 9, 16),
                    started_on=date(2024, 10, 1),
                    completed_on=date(2025, 3, 1), completion_confirmed=True,
                ),
            ],
        )

    def test_latest_event_by_scope(self, profile):
        assert profile.latest_event("entrance").event_id == "e2"
        assert profile.latest_event("facade").event_id == "e1"

    def test_photo_after_confirmed_completion_is_current(self, profile):
        assert profile.shows_current_appearance(date(2025, 5, 1), "entrance") is True

    def test_photo_before_works_started_is_not_current(self, profile):
        assert profile.shows_current_appearance(date(2023, 1, 1), "entrance") is False

    def test_photo_during_works_is_undecided(self, profile):
        """Entre le début et la fin, l'image peut montrer un chantier."""
        assert profile.shows_current_appearance(date(2024, 12, 1), "entrance") is None

    def test_approval_alone_never_establishes_current_appearance(self):
        """Approuver n'est ni commencer ni achever.

        Le dossier municipal du WelcomINNS ne porte qu'une approbation : une
        photographie postérieure peut montrer l'ancienne entrée.
        """
        profile = PropertyProfile(
            property_id="p", address="a", official_name="X",
            renovation_events=[
                RenovationEvent(
                    event_id="e", scope="entrance", approved_on=date(2024, 9, 16)
                )
            ],
        )
        assert profile.shows_current_appearance(date(2025, 6, 1), "entrance") is None

    def test_unconfirmed_completion_is_not_enough(self):
        profile = PropertyProfile(
            property_id="p", address="a", official_name="X",
            renovation_events=[
                RenovationEvent(
                    event_id="e", scope="entrance",
                    completed_on=date(2024, 12, 1), completion_confirmed=False,
                )
            ],
        )
        assert profile.shows_current_appearance(date(2025, 6, 1), "entrance") is None

    def test_event_without_any_date_is_rejected(self):
        with pytest.raises(ValueError, match="sans aucune date"):
            RenovationEvent(event_id="e", scope="entrance")

    def test_incoherent_dates_rejected(self):
        with pytest.raises(ValueError, match="incohérentes"):
            RenovationEvent(
                event_id="e", scope="entrance",
                started_on=date(2025, 1, 1), completed_on=date(2024, 1, 1),
            )

    def test_no_declared_event_yields_no_answer(self):
        """Sans travaux connus, supposer « à jour » serait une invention."""
        profile = PropertyProfile(property_id="p", address="a", official_name="X")
        assert profile.shows_current_appearance(date(2025, 1, 1)) is None

    def test_multiple_renovations_supported(self, profile):
        assert len(profile.renovation_events) == 2


class TestPipelinePolicy:
    def test_thresholds_live_in_the_policy_not_the_profile(self):
        assert DEFAULT_POLICY.model.subject_accept == 0.50
        assert not hasattr(PropertyProfile, "subject_accept")

    def test_policy_is_versioned(self):
        assert DEFAULT_POLICY.version

    def test_calibration_provenance_is_recorded(self):
        """Un seuil sans trace de calibration est un nombre sans autorité."""
        assert DEFAULT_POLICY.model.calibration_id
        assert DEFAULT_POLICY.model.calibrated_on_sites >= 1

    def test_policy_is_serialisable_and_reloadable(self):
        reloaded = PipelinePolicy.model_validate_json(DEFAULT_POLICY.model_dump_json())
        assert reloaded == DEFAULT_POLICY

    def test_incoherent_threshold_rejected(self):
        with pytest.raises(ValueError):
            PipelinePolicy.model_validate({"model": {"subject_accept": 1.7}})


class TestRenovationInvariants:
    """Trois failles relevées à l'audit du profil."""

    def test_confirmed_completion_requires_a_date(self):
        from datetime import date as _date

        with pytest.raises(ValueError, match="sans completed_on"):
            RenovationEvent(
                event_id="e", scope="entrance",
                approved_on=_date(2024, 1, 1), completion_confirmed=True,
            )

    def test_approval_after_completion_is_rejected(self):
        """Le contrôle sautait quand started_on manquait."""
        from datetime import date as _date

        with pytest.raises(ValueError, match="approved_on/completed_on"):
            RenovationEvent(
                event_id="e", scope="entrance",
                approved_on=_date(2025, 1, 1), completed_on=_date(2024, 1, 1),
            )

    def test_photo_before_an_approval_stays_undecided(self):
        """Approuver ne prouve pas que le chantier a commencé.

        Une image antérieure à l'approbation était déclarée « pas actuelle »,
        alors que les travaux pouvaient n'avoir jamais démarré — auquel cas
        elle montre bien l'état courant.
        """
        from datetime import date as _date

        profile = PropertyProfile(
            property_id="p", address="a", official_name="X",
            renovation_events=[
                RenovationEvent(event_id="e", scope="entrance", approved_on=_date(2024, 9, 16))
            ],
        )
        assert profile.shows_current_appearance(_date(2023, 1, 1), "entrance") is None

    def test_photo_before_a_confirmed_start_is_not_current(self):
        from datetime import date as _date

        profile = PropertyProfile(
            property_id="p", address="a", official_name="X",
            renovation_events=[
                RenovationEvent(
                    event_id="e", scope="entrance",
                    started_on=_date(2024, 10, 1),
                    completed_on=_date(2025, 3, 1), completion_confirmed=True,
                )
            ],
        )
        assert profile.shows_current_appearance(_date(2023, 1, 1), "entrance") is False
