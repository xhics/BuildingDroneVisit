"""Découverte de candidats, sur métadonnées seulement (collecte V2, étape 1).

Ce qui est éprouvé : aucun octet d'image, aucune URL conservée, aucune
dimension recopiée comme mesurée, et un besoin énoncé **avant** la collecte —
sans quoi le corpus définirait après coup ce qu'on cherchait.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from hotel_pipeline.collectors.base import CollectedImage
from hotel_pipeline.discover import (
    LIMITS,
    DiscoveryRefused,
    candidates_from,
    deduplicate,
    discover,
)
from hotel_pipeline.schemas.acquisition import (
    CaptureCandidate,
    CaptureDemand,
    CaptureDemandManifest,
    CaptureIntent,
    TargetKind,
)


def demand(demand_id: str = "d1", **overrides) -> CaptureDemand:
    fields = dict(
        demand_id=demand_id, intent=CaptureIntent.BUILDING_CAPTURE,
        target_kind=TargetKind.VIEW_SECTOR, target_ref="front",
    )
    fields.update(overrides)
    return CaptureDemand(**fields)


def demands(*items: CaptureDemand) -> CaptureDemandManifest:
    return CaptureDemandManifest(
        hotel_id="h", demands=list(items) or [demand()]
    )


def image(source_id: str = "1", **overrides) -> CollectedImage:
    fields = dict(
        source="mapillary", source_id=source_id,
        url=f"https://cdn.example/{source_id}.jpg?token=secret",
        lat=45.57, lon=-73.44, heading_deg=90.0, captured_year=2024,
    )
    fields.update(overrides)
    return CollectedImage(**fields)


# --- le besoin précède la collecte -------------------------------------------


def test_discovery_without_a_declared_demand_is_refused() -> None:
    """Sans objectif, la collecte définirait après coup ce qu'on cherchait."""
    with pytest.raises(DiscoveryRefused, match="aucun besoin déclaré"):
        discover("h", CaptureDemandManifest(hotel_id="h"), {"mapillary": []})


def test_a_declared_demand_is_enough_to_discover() -> None:
    manifest, report = discover("h", demands(), {"mapillary": []})

    assert manifest.candidates == []
    assert report.candidates_by_source == {"mapillary": 0}


# --- aucune URL, aucun secret -------------------------------------------------


def test_no_url_survives_the_conversion() -> None:
    """Une URL de CDN expire, une URL signée porte la clé d'API."""
    candidates = candidates_from("mapillary", [image()])

    serialised = json.dumps([json.loads(c.model_dump_json()) for c in candidates])
    assert "://" not in serialised
    assert "token" not in serialised
    assert "secret" not in serialised


def test_the_schema_refuses_a_url_smuggled_into_the_request_spec() -> None:
    with pytest.raises(ValueError, match="contient une URL"):
        CaptureCandidate(
            candidate_id="c", source="mapillary", provider_id="1",
            request_spec={"thumb": "https://cdn.example/1.jpg"},
        )


def test_the_schema_refuses_anything_that_looks_like_a_secret() -> None:
    with pytest.raises(ValueError, match="ressemble à un secret"):
        CaptureCandidate(
            candidate_id="c", source="mapillary", provider_id="1",
            request_spec={"api_key": "abc"},
        )


def test_what_remains_is_enough_to_rebuild_the_address() -> None:
    candidate = candidates_from("mapillary", [image("42")])[0]

    assert candidate.request_spec["provider_id"] == "42"
    assert candidate.request_spec["resolution"] == "thumb_2048"
    assert candidate.available_resolutions == ["thumb_2048"]


# --- annoncé n'est pas mesuré -------------------------------------------------


def test_no_dimension_is_copied_as_if_it_were_measured() -> None:
    """Les dimensions n'existeront qu'après acquisition d'un fichier."""
    candidate = candidates_from("mapillary", [image()])[0]

    assert candidate.advertised_width is None
    assert candidate.advertised_height is None


def test_a_capture_year_never_becomes_a_precise_instant() -> None:
    """La source publie une année ; en faire une date exacte l'inventerait."""
    candidate = candidates_from("mapillary", [image()])[0]

    assert candidate.captured_at == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert candidates_from("mapillary", [image(captured_year=None)])[0].captured_at is None


def test_the_report_states_what_discovery_cannot_establish() -> None:
    _, report = discover("h", demands(), {"mapillary": []})
    published = report.as_dict()

    assert published["bytes_downloaded"] == 0
    assert any("aucune image n'a été téléchargée" in limit for limit in published["limits"])
    assert any("aucun cadrage n'est calculé" in limit for limit in LIMITS)


def test_the_limits_name_no_property_and_no_corpus_size() -> None:
    text = " ".join(LIMITS).lower()

    for name in ("welcominns", "mortagne", "boucherville", "189"):
        assert name not in text


# --- une source en panne n'est pas une source vide ---------------------------


def test_a_failed_source_is_recorded_not_counted_as_empty() -> None:
    """Sinon le plan jugerait un corpus qu'il croit complet."""
    manifest, report = discover(
        "h", demands(),
        {"mapillary": [], "places": "hors ligne : aucun réseau autorisé"},
    )

    assert report.sources_queried == ["mapillary"]
    assert "places" in report.sources_skipped
    # `queries` ne compte que ce qui a été réellement interrogé.
    assert manifest.queries == {"mapillary": 0}
    assert "places" not in manifest.queries


# --- identité et doublons -----------------------------------------------------


def test_identical_views_are_collapsed_once() -> None:
    """Empiler deux index gonflerait le volume annoncé, donc le consentement."""
    twice = candidates_from("mapillary", [image("7"), image("7"), image("8")])

    unique, dropped = deduplicate(twice)

    assert len(unique) == 2
    assert dropped == 1


def test_a_panoramic_source_identifies_by_its_framing() -> None:
    """Deux cadrages d'un panorama sont deux prises de vue."""
    from hotel_pipeline.schemas.acquisition import capture_identity

    framing = dict(fov_deg=80.0, pitch_deg=0.0, size="640x640")
    first = capture_identity("street_view", "pano-1", heading_deg=10.0, **framing)
    second = capture_identity("street_view", "pano-1", heading_deg=200.0, **framing)

    assert first != second
    # Une source publiant des images distinctes n'en a pas besoin.
    assert capture_identity("mapillary", "42") == capture_identity("mapillary", "42")


def test_an_unknown_source_must_declare_its_identity_strategy() -> None:
    from hotel_pipeline.schemas.acquisition import capture_identity

    with pytest.raises(ValueError):
        capture_identity("source-inconnue", "1")


# --- le manifeste est lié à ce qui l'a produit -------------------------------


def test_the_manifest_cites_the_demands_it_was_built_against() -> None:
    manifest, _ = discover(
        "h", demands(), {"mapillary": []},
        demand_digest="dem0", policy_digest="pol0",
    )

    assert manifest.demand_digest == "dem0"
    assert manifest.policy_digest == "pol0"


def test_candidates_are_ordered_so_two_runs_compare() -> None:
    manifest, _ = discover(
        "h", demands(),
        {"mapillary": candidates_from("mapillary", [image("9"), image("2")])},
    )

    identifiers = [c.candidate_id for c in manifest.candidates]
    assert identifiers == sorted(identifiers)
