"""Registre de calibration et péremption sélective (portabilité).

Deux dettes fermées ensemble parce qu'elles disent la même chose : une trace
n'est utile que si elle nomme ce dont elle dépend. `calibrated_on_sites=1`
donnait un nombre sans site ; une empreinte de profil périmait tout, y compris
ce qui ne la lit jamais.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from hotel_pipeline import calibration_registry as registry
from hotel_pipeline import staleness
from hotel_pipeline.calibration_registry import CalibrationEntry, CalibrationRegistry
from hotel_pipeline.schemas import PipelinePolicy
from hotel_pipeline.staleness import Facet


def entry(**overrides) -> CalibrationEntry:
    fields = dict(
        calibration_id="campagne-x", site_ids=["site-a"],
        corpus_digests={"site-a": "d0"}, method="lecture manuelle",
        calibrated_on=date(2026, 8, 12), version="1.0.0",
    )
    fields.update(overrides)
    return CalibrationEntry(**fields)


# --- le registre rend l'attachement vérifiable --------------------------------


def test_the_policy_gains_no_field_so_its_digest_cannot_move() -> None:
    """Le registre vit dehors précisément pour cela."""
    from hotel_pipeline.provenance import policy_digest

    fields = set(PipelinePolicy.model_fields)
    before = policy_digest(PipelinePolicy())

    assert "calibration_registry" not in fields
    from hotel_pipeline.schemas.policy import ModelPolicy

    assert "site_ids" not in set(ModelPolicy.model_fields)
    assert policy_digest(PipelinePolicy()) == before


def test_a_site_cited_without_a_corpus_proves_nothing() -> None:
    with pytest.raises(ValueError, match="sans empreinte de corpus"):
        entry(site_ids=["site-a", "site-b"])


def test_a_corpus_for_an_absent_site_is_refused() -> None:
    with pytest.raises(ValueError, match="sites absents"):
        entry(corpus_digests={"site-a": "d0", "site-z": "d1"})


def test_a_cited_campaign_missing_from_the_registry_is_reported() -> None:
    policy = PipelinePolicy.model_validate(
        {"model": {"calibration_id": "campagne-absente", "calibrated_on_sites": 2}}
    )

    problems = registry.check(policy, CalibrationRegistry())

    assert any("absente du registre" in problem for problem in problems)


def test_a_site_count_that_disagrees_with_the_registry_is_reported() -> None:
    """Deux comptes qui divergent, c'est un compte faux quelque part."""
    policy = PipelinePolicy.model_validate(
        {"model": {"calibration_id": "campagne-x", "calibrated_on_sites": 3}}
    )
    known = CalibrationRegistry(entries=[entry()])

    problems = registry.check(policy, known)

    assert any("le registre en cite 1" in problem for problem in problems)


def test_an_uncalibrated_policy_needs_no_registry_entry() -> None:
    assert registry.check(PipelinePolicy(), CalibrationRegistry()) == []


def test_the_pilot_campaigns_are_declared_and_agree() -> None:
    """Non-régression sur le registre réel du dépôt."""
    path = Path("calibrations/registry.json")
    if not path.is_file():  # pragma: no cover — dépend du dépôt
        pytest.skip("registre absent")

    known = registry.load(path)
    pilot = known.get("welcominns-2026-08-36-images")

    assert pilot is not None
    assert pilot.site_ids == ["welcominns-boucherville"]
    assert pilot.corpus_digests["welcominns-boucherville"]
    assert "36 vues" in pilot.method

    policy = PipelinePolicy.model_validate_json(
        Path(
            "work/welcominns-boucherville/00_manifest/pipeline_policy.json"
        ).read_text("utf-8")
    ) if Path(
        "work/welcominns-boucherville/00_manifest/pipeline_policy.json"
    ).is_file() else None
    if policy is not None:
        assert registry.check(policy, known) == []


# --- la péremption ne déborde pas ---------------------------------------------


def profile(**overrides) -> dict:
    base = {
        "official_name": "Hôtel Test", "aliases": [], "competitor_names": [],
        "ocr_languages": ["fr"], "timezone": "America/Toronto",
        "lat": 45.57, "lon": -73.44, "address": "1 rue Test",
        "country_code": "CA", "subdivision_code": "QC",
        "room_count": 116, "expected_levels": 3,
        "renovation_events": [], "website_url": None,
    }
    base.update(overrides)
    return base


def test_a_rename_never_invalidates_the_terrain_or_the_lidar() -> None:
    """Le critère explicite : un changement identitaire n'atteint pas le sol."""
    report = staleness.assess(profile(), profile(official_name="Hôtel Renommé"))

    assert report.changed_facets == ["identity"]
    preserved = {row["production"] for row in report.preserved}
    assert {"elevation_derivation", "lidar_acquisition", "geospatial_qualification"} <= preserved
    assert set(report.invalidated) == {"identity_classification", "asset_review"}


def test_changing_ocr_languages_invalidates_identity_only() -> None:
    report = staleness.assess(profile(), profile(ocr_languages=["fr", "en"]))

    assert report.changed_facets == ["identity"]
    assert "elevation_derivation" not in report.invalidated


def test_changing_the_timezone_invalidates_the_dating() -> None:
    report = staleness.assess(profile(), profile(timezone="Europe/Paris"))

    assert report.invalidated == ["temporal_assessment"]


def test_moving_the_site_invalidates_its_spatial_reference() -> None:
    report = staleness.assess(profile(), profile(lat=45.60))

    assert "spatial_reference" in report.invalidated
    assert "building_candidates" in report.invalidated


def test_a_declarative_territory_field_invalidates_nothing() -> None:
    """Le territoire se résout depuis la position, jamais depuis ce champ."""
    report = staleness.assess(profile(subdivision_code=None), profile())

    assert report.changed_facets == ["territory_declaration"]
    assert report.invalidated == []


def test_explicit_nulls_are_not_a_change() -> None:
    """Le faux positif observé : le modèle sérialisait `started_on: null`.

    La datation du pilote se serait périmée au seul motif d'un changement
    d'écriture, sans qu'aucune valeur ne bouge.
    """
    terse = profile(renovation_events=[{"event_id": "e", "scope": "entrance",
                                        "approved_on": "2024-09-16"}])
    verbose = profile(renovation_events=[{"event_id": "e", "scope": "entrance",
                                          "approved_on": "2024-09-16",
                                          "started_on": None, "completed_on": None}])

    report = staleness.assess(terse, verbose)

    assert report.changed_facets == []
    assert report.invalidated == []


def test_a_field_nobody_reads_invalidates_nothing() -> None:
    report = staleness.assess(profile(), profile(website_url="https://exemple.test"))

    assert report.changed_facets == []
    assert report.invalidated == []


def test_the_pilot_profile_migration_spared_the_geospatial() -> None:
    """Le cas réel : le profil a gagné pays, fuseau et langues.

    Seule la datation devait en pâtir. Périmer le MNT, la toiture ou le nuage
    aurait obligé à tout recalculer pour une information déclarative.
    """
    before = profile(country_code=None, subdivision_code=None, timezone=None)
    after = profile()

    report = staleness.assess(before, after)

    assert set(report.invalidated) == {"temporal_assessment"}
    preserved = {row["production"] for row in report.preserved}
    assert "elevation_derivation" in preserved
    assert "geospatial_qualification" in preserved


def test_every_facet_is_wired_to_the_profile() -> None:
    """Une facette sans champ ne périmerait jamais rien, silencieusement."""
    assert set(staleness.FACET_FIELDS) == set(Facet)
    assert all(names for names in staleness.FACET_FIELDS.values())


def test_the_profile_fields_that_matter_all_belong_to_a_facet() -> None:
    """Un champ décisionnel oublié ne périmerait rien.

    Les champs purement descriptifs en sont exclus délibérément, et la liste
    des exclusions est explicite pour qu'un ajout futur soit un choix.
    """
    from hotel_pipeline.schemas import PropertyProfile

    covered = {name for names in staleness.FACET_FIELDS.values() for name in names}
    descriptive = {"property_id", "website_url", "place_query"}

    assert set(PropertyProfile.model_fields) - covered == descriptive
