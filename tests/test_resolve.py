"""Résolution de propriété et séparations géométriques (Lot 1).

Le corpus de test reproduit délibérément le piège du §3 du plan directeur :
l'empreinte de l'hôtel n'est pas étiquetée « hôtel », un hôtel voisin l'est,
et un parc-o-bus voisine le stationnement de l'établissement.
"""

from __future__ import annotations

import pytest

from hotel_pipeline.resolve import (
    build_candidates,
    check_separations,
    looks_like_park_and_ride,
    parking_features,
)
from hotel_pipeline.schemas.enums import ObjectState
from hotel_pipeline.schemas.spatial import GeocodeResult, SpatialManifest

TRUE_BUILDING = "way/1001"
NEIGHBOUR_HOTEL = "way/1002"
HOTEL_PARKING = "way/2001"
PARK_AND_RIDE = "way/2002"


@pytest.fixture
def geocode() -> GeocodeResult:
    return GeocodeResult(lat=45.5896, lon=-73.4372, provider="fixture")


@pytest.fixture
def candidates(overpass_elements, geocode):
    return build_candidates(overpass_elements, geocode)


@pytest.fixture
def manifest(candidates, geocode) -> SpatialManifest:
    return SpatialManifest(
        hotel_id="welcominns-boucherville",
        address="1195 rue Ampère",
        geocode=geocode,
        candidates=candidates,
    )


class TestCandidates:
    def test_only_buildings_become_candidates(self, candidates):
        ids = {c.feature_id for c in candidates}
        assert HOTEL_PARKING not in ids
        assert PARK_AND_RIDE not in ids
        assert TRUE_BUILDING in ids

    def test_geometry_measured_in_metres(self, candidates):
        building = next(c for c in candidates if c.feature_id == TRUE_BUILDING)
        assert 1800 < building.area_m2 < 2300
        assert building.distance_to_geocode_m < 30

    def test_scoring_favours_the_wrong_building(self, candidates):
        """Le piège du §3, reproduit et mesuré.

        L'hôtel voisin porte l'étiquette `tourism=hotel` que la vraie empreinte
        n'a pas. Le classement automatique le place donc en tête : c'est
        exactement pourquoi la décision revient à l'humain (§12).
        """
        top = candidates[0]
        assert top.feature_id == NEIGHBOUR_HOTEL

        truth = next(c for c in candidates if c.feature_id == TRUE_BUILDING)
        assert truth.score < top.score

    def test_score_reasons_are_recorded(self, candidates):
        truth = next(c for c in candidates if c.feature_id == TRUE_BUILDING)
        assert any("emprise plausible" in r for r in truth.score_reasons)


class TestManifestState:
    def test_multiple_candidates_is_conflicted(self, manifest):
        assert manifest.state is ObjectState.CONFLICTED

    def test_confirmation_makes_it_confirmed(self, manifest):
        manifest.confirmed_building_id = TRUE_BUILDING
        manifest.confirmed_by = "hm"
        assert manifest.state is ObjectState.CONFIRMED

    def test_cannot_confirm_an_unknown_feature(self, candidates, geocode):
        with pytest.raises(ValueError, match="absent des candidats"):
            SpatialManifest(
                hotel_id="h",
                address="a",
                geocode=geocode,
                candidates=candidates,
                confirmed_building_id="way/999999",
                confirmed_by="hm",
            )

    def test_confirmation_requires_an_author(self, candidates, geocode):
        """Une confirmation sans auteur n'est pas une preuve."""
        with pytest.raises(ValueError, match="auteur"):
            SpatialManifest(
                hotel_id="h",
                address="a",
                geocode=geocode,
                candidates=candidates,
                confirmed_building_id=TRUE_BUILDING,
            )


class TestParkAndRideDetection:
    def test_park_ride_tag_detected(self):
        assert looks_like_park_and_ride({"park_ride": "yes"})

    def test_french_label_detected(self):
        assert looks_like_park_and_ride({"name": "Stationnement incitatif De Mortagne"})

    def test_customer_parking_not_flagged(self):
        assert not looks_like_park_and_ride({"amenity": "parking", "access": "customers"})

    def test_both_parkings_present_in_fixture(self, overpass_elements):
        assert len(parking_features(overpass_elements)) == 2


class TestSeparations:
    def test_not_evaluable_without_confirmation(self, manifest, overpass_elements):
        assertions = check_separations(manifest, overpass_elements)
        assert [a.name for a in assertions] == ["building_confirmed"]
        assert not assertions[0].passed

    def test_correct_building_passes_all_separations(self, manifest, overpass_elements):
        manifest.confirmed_building_id = TRUE_BUILDING
        manifest.confirmed_by = "hm"
        assertions = check_separations(manifest, overpass_elements)

        assert all(a.passed for a in assertions), [a.detail for a in assertions if not a.passed]
        assert manifest.parking_feature_id == HOTEL_PARKING
        assert manifest.park_and_ride_feature_id == PARK_AND_RIDE

    def test_park_and_ride_never_becomes_hotel_parking(self, manifest, overpass_elements):
        """Le risque nommé au §3 : confondre les deux stationnements."""
        manifest.confirmed_building_id = TRUE_BUILDING
        manifest.confirmed_by = "hm"
        check_separations(manifest, overpass_elements)
        assert manifest.parking_feature_id != manifest.park_and_ride_feature_id

    def test_wrong_building_fails_adjacency(self, manifest, overpass_elements):
        """Confirmer l'hôtel voisin doit échouer : aucun stationnement contigu."""
        manifest.confirmed_building_id = NEIGHBOUR_HOTEL
        manifest.confirmed_by = "hm"
        assertions = check_separations(manifest, overpass_elements)

        failed = {a.name for a in assertions if not a.passed}
        assert "parking_adjacent_to_building" in failed
