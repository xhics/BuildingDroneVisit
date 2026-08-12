"""Manifeste de site — instances réelles issues du gabarit (Lot 1B §4).

Deux invariants gouvernent ce modèle : rien n'est inventé, rien n'est éliminé.
Un objet que les données ne permettent pas d'établir existe quand même, à
l'état `unresolved`, avec son motif — le supprimer ferait croire qu'il n'a pas
été cherché, le remplir ferait croire qu'il est connu.
"""

from __future__ import annotations

import pytest

from hotel_pipeline.schemas import ObjectState, Subject
from hotel_pipeline.schemas.critical_objects import EXCLUDED_KINDS, REQUIRED_OBJECTS
from hotel_pipeline.schemas.site import SiteManifest, SiteObject, SiteRelation
from hotel_pipeline.schemas.spatial import (
    BuildingCandidate,
    GeocodeResult,
    GeometricAssertion,
    SpatialManifest,
)
from hotel_pipeline.site import PARK_AND_RIDE, build, object_id

BUILDING_WKT = (
    "POLYGON ((-73.44380 45.57355, -73.44280 45.57355, "
    "-73.44280 45.57445, -73.44380 45.57445, -73.44380 45.57355))"
)

HOTEL = "hotel-test"


def polygon_element(element_id: int, tags: dict, lat0=45.5730, lon0=-73.4438):
    return {
        "type": "way",
        "id": element_id,
        "tags": tags,
        "geometry": [
            {"lat": lat0, "lon": lon0},
            {"lat": lat0, "lon": lon0 + 0.0008},
            {"lat": lat0 + 0.0002, "lon": lon0 + 0.0008},
            {"lat": lat0 + 0.0002, "lon": lon0},
            {"lat": lat0, "lon": lon0},
        ],
    }


@pytest.fixture
def spatial() -> SpatialManifest:
    return SpatialManifest(
        hotel_id=HOTEL,
        address="1 rue Test",
        geocode=GeocodeResult(lat=45.5741, lon=-73.4433, provider="test"),
        candidates=[
            BuildingCandidate(
                feature_id="way/1", source="overpass",
                centroid_lat=45.5740, centroid_lon=-73.4433,
                area_m2=1800, distance_to_geocode_m=5, wkt=BUILDING_WKT,
            )
        ],
        confirmed_building_id="way/1",
        confirmed_by="hm",
        confirmation_rationale="vérifié sur aérien",
        parking_feature_id="way/2",
        park_and_ride_feature_id="way/3",
        assertions=[
            GeometricAssertion(name="parking_adjacent_to_building", passed=True, detail="à 5 m")
        ],
    )


@pytest.fixture
def elements() -> list[dict]:
    return [
        polygon_element(2, {"amenity": "parking"}),
        polygon_element(3, {"amenity": "parking", "park_ride": "yes"}, lat0=45.5760),
    ]


@pytest.fixture
def roads() -> list[dict]:
    return [
        {
            "type": "way", "id": 9, "tags": {"highway": "residential", "name": "rue Test"},
            "geometry": [{"lat": 45.5735, "lon": -73.4433}, {"lat": 45.5735, "lon": -73.4420}],
        }
    ]


class TestTemplateIsFullyInstantiated:
    def test_every_required_kind_gets_an_instance(self, spatial, elements, roads):
        site, _ = build(HOTEL, spatial, elements, roads)
        assert site.missing_required() == []

    def test_object_ids_are_stable_and_source_independent(self, spatial, elements, roads):
        """Un changement d'identifiant OSM ne doit pas renommer l'objet."""
        site, _ = build(HOTEL, spatial, elements, roads)
        assert site.by_id(object_id(HOTEL, "BUILDING_MAIN")) is not None
        assert "way/" not in object_id(HOTEL, "BUILDING_MAIN")

    def test_confirmed_building_carries_its_evidence(self, spatial, elements, roads):
        site, _ = build(HOTEL, spatial, elements, roads)
        building = site.by_id(object_id(HOTEL, "BUILDING_MAIN"))
        assert building.state is ObjectState.CONFIRMED
        assert building.source_ref == "way/1"
        assert building.confirmed_by == "hm"


class TestNothingIsInventedNorRemoved:
    def test_unavailable_sources_yield_unresolved_objects_with_reasons(
        self, spatial, elements, roads
    ):
        site, _ = build(HOTEL, spatial, elements, roads)
        roofline = site.by_id(object_id(HOTEL, "ROOFLINE_MAIN"))
        assert roofline.state is ObjectState.UNRESOLVED
        assert "LiDAR" in roofline.unresolved_reason

    def test_unresolved_objects_never_carry_geometry(self, spatial, elements, roads):
        site, _ = build(HOTEL, spatial, elements, roads)
        assert all(o.geometry_wkt is None for o in site.unresolved())

    def test_every_unresolved_object_states_why(self, spatial, elements, roads):
        site, _ = build(HOTEL, spatial, elements, roads)
        assert all(o.unresolved_reason for o in site.unresolved())

    def test_confirmed_without_evidence_is_rejected(self):
        with pytest.raises(ValueError, match="sans preuve"):
            SiteObject(object_id="x", kind="BUILDING_MAIN", state=ObjectState.CONFIRMED)

    def test_unresolved_with_geometry_is_rejected(self):
        """Porter une géométrie et se dire inconnu est contradictoire."""
        with pytest.raises(ValueError, match="géométrie"):
            SiteObject(
                object_id="x", kind="BUILDING_MAIN",
                state=ObjectState.UNRESOLVED, geometry_wkt=BUILDING_WKT,
            )


class TestExclusionsAreRealInstances:
    def test_park_and_ride_is_an_instance_not_a_word(self, spatial, elements, roads):
        site, _ = build(HOTEL, spatial, elements, roads)
        park_ride = site.by_id(object_id(HOTEL, PARK_AND_RIDE))
        assert park_ride.state is ObjectState.INFERRED
        assert park_ride.source_ref == "way/3"
        assert park_ride.geometry_wkt is not None

    def test_park_and_ride_is_related_to_what_it_must_not_be(self, spatial, elements, roads):
        site, _ = build(HOTEL, spatial, elements, roads)
        park_ride = site.by_id(object_id(HOTEL, PARK_AND_RIDE))
        targets = {r.target_id.split(":")[-1] for r in park_ride.relations}
        assert targets == {"BUILDING_MAIN", "PARKING_HOTEL"}
        assert all(r.predicate == "distinct_from" for r in park_ride.relations)

    def test_absent_park_and_ride_is_still_instantiated(self, spatial, roads):
        """Aucun parc-o-bus étiqueté n'est pas la preuve qu'il n'y en a pas."""
        spatial.park_and_ride_feature_id = None
        site, _ = build(HOTEL, spatial, [], roads)
        park_ride = site.by_id(object_id(HOTEL, PARK_AND_RIDE))
        assert park_ride.state is ObjectState.UNRESOLVED
        assert "absence non vérifiée" in park_ride.unresolved_reason

    def test_excluded_kinds_are_listed_apart(self, spatial, elements, roads):
        site, _ = build(HOTEL, spatial, elements, roads)
        assert [o.kind for o in site.excluded_instances()] == [PARK_AND_RIDE]
        assert PARK_AND_RIDE in EXCLUDED_KINDS


class TestRelationsAreExplicit:
    def test_parking_is_linked_to_the_building(self, spatial, elements, roads):
        site, _ = build(HOTEL, spatial, elements, roads)
        parking = site.by_id(object_id(HOTEL, "PARKING_HOTEL"))
        relation = parking.relation_to(object_id(HOTEL, "BUILDING_MAIN"))
        assert relation.predicate == "adjacent_to"

    def test_parking_is_declared_distinct_from_the_park_and_ride(
        self, spatial, elements, roads
    ):
        site, _ = build(HOTEL, spatial, elements, roads)
        parking = site.by_id(object_id(HOTEL, "PARKING_HOTEL"))
        assert parking.relation_to(object_id(HOTEL, PARK_AND_RIDE)).predicate == "distinct_from"

    def test_access_road_serves_the_building(self, spatial, elements, roads):
        site, _ = build(HOTEL, spatial, elements, roads)
        road = site.by_id(object_id(HOTEL, "ACCESS_ROAD_MAIN"))
        assert road.relation_to(object_id(HOTEL, "BUILDING_MAIN")).predicate == "serves"
        assert road.source_ref == "way/9"

    def test_facades_are_parts_of_the_building(self, spatial, elements, roads):
        site, _ = build(HOTEL, spatial, elements, roads)
        facade = site.by_id(object_id(HOTEL, "FACADE_PRIMARY"))
        assert facade.relation_to(object_id(HOTEL, "BUILDING_MAIN")).predicate == "part_of"

    def test_a_relation_to_an_absent_object_is_rejected(self):
        with pytest.raises(ValueError, match="objet absent"):
            SiteManifest(
                hotel_id=HOTEL,
                objects=[
                    SiteObject(
                        object_id="a", kind="BUILDING_MAIN",
                        relations=[SiteRelation(predicate="serves", target_id="fantome")],
                    )
                ],
            )


class TestSignFromEvidence:
    def test_sign_is_inferred_from_a_matching_read(self, spatial, elements, roads):
        from hotel_pipeline.schemas import Asset, AssetCategory, PropertyMatchStatus, Rights

        asset = Asset(
            id="img-1", source="website", source_url_or_id="u", rights=Rights.UNKNOWN,
            ai_eligible=False, confidence=0.5, category=AssetCategory.SIGN,
            checksum="a" * 64, subjects=[Subject.SIGN],
            property_match_status=PropertyMatchStatus.MATCH,
        )
        site, _ = build(HOTEL, spatial, elements, roads, [asset])
        sign = site.by_id(object_id(HOTEL, "PROPERTY_SIGN"))
        assert sign.state is ObjectState.INFERRED
        assert "img-1" in sign.evidence

    def test_sign_without_a_matching_read_stays_unresolved(self, spatial, elements, roads):
        site, _ = build(HOTEL, spatial, elements, roads, [])
        assert site.by_id(object_id(HOTEL, "PROPERTY_SIGN")).state is ObjectState.UNRESOLVED


class TestNoBuildingConfirmed:
    def test_site_is_still_built_without_a_confirmed_building(self, spatial, elements, roads):
        """Le manifeste doit exister même quand rien n'est tranché."""
        spatial.confirmed_building_id = None
        spatial.confirmed_by = None
        site, report = build(HOTEL, spatial, elements, roads)
        assert site.missing_required() == []
        assert site.by_id(object_id(HOTEL, "BUILDING_MAIN")).state is ObjectState.UNRESOLVED
        assert report.confirmed == 0


class TestGeoProvenance:
    """Un objet dérivé sans provenance n'est pas vérifiable (audit LiDAR)."""

    def _source(self, **overrides):
        from hotel_pipeline.schemas import GeoSourceProvenance

        fields = dict(
            source_id="lidar-2023-31H05NE",
            dataset="Données LiDAR du Québec",
            vintage="2023",
            tile_id="31H05NE",
            crs_horizontal="EPSG:2950",
            crs_vertical="CGVD2013",
            point_density_per_m2=8.0,
        )
        fields.update(overrides)
        return GeoSourceProvenance(**fields)

    def test_derived_object_must_reference_a_declared_source(self):
        with pytest.raises(ValueError, match="sources non déclarées"):
            SiteManifest(
                hotel_id=HOTEL,
                objects=[
                    SiteObject(
                        object_id="a", kind="TERRAIN_MAIN", state=ObjectState.INFERRED,
                        derived_from_sources=["source-fantome"],
                        derivation_method="MNT",
                    )
                ],
            )

    def test_derived_object_must_state_its_method(self):
        with pytest.raises(ValueError, match="sans méthode"):
            SiteManifest(
                hotel_id=HOTEL,
                geo_sources=[self._source()],
                objects=[
                    SiteObject(
                        object_id="a", kind="TERRAIN_MAIN", state=ObjectState.INFERRED,
                        derived_from_sources=["lidar-2023-31H05NE"],
                    )
                ],
            )

    def test_valid_derivation_is_accepted_and_listed(self):
        manifest = SiteManifest(
            hotel_id=HOTEL,
            geo_sources=[self._source()],
            objects=[
                SiteObject(
                    object_id="a", kind="ROOFLINE_MAIN", state=ObjectState.INFERRED,
                    derived_from_sources=["lidar-2023-31H05NE"],
                    derivation_method="MNS − MNT, seuil 2 m",
                )
            ],
        )
        assert [o.kind for o in manifest.derived()] == ["ROOFLINE_MAIN"]
        assert manifest.source("lidar-2023-31H05NE").crs_vertical == "CGVD2013"

    def test_vertical_datum_without_horizontal_is_rejected(self):
        """Une altitude sans référentiel horizontal ne situe rien."""
        with pytest.raises(ValueError, match="ne situe rien"):
            self._source(crs_horizontal=None)

    def test_summary_counts_sources_and_derived_objects(self, spatial, elements, roads):
        site, _ = build(HOTEL, spatial, elements, roads)
        summary = site.summary()
        assert summary["geo_sources"] == 0
        assert summary["derived_objects"] == 0
