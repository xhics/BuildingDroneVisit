"""Géométries de capture (Lot 1B V2, étape 2).

Les cas s'appuient sur des réponses Overpass **authentiques** du WelcomINNS,
réduites en nombre d'éléments mais non retouchées. La voie d'accès du site,
`way/938806358`, y figure telle que la source la rend : une allée de
stationnement fermée, en `access=customers`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from shapely import wkt as shapely_wkt
from shapely.geometry import LineString, Polygon

from hotel_pipeline.geo import capture_geometry as cg
from hotel_pipeline.geo.resolve_geometry import resolve
from hotel_pipeline.schemas import (
    AccessStatus,
    CaptureGeometryManifest,
    CorridorClass,
    GeometryResolutionStatus,
    GeometryRole,
    GeometrySourceSnapshot,
    ResolvedGeometry,
    RoadCorridor,
    SourceQueryStatus,
)

FIXTURE = json.loads(
    Path("tests/fixtures/overpass_roads_boucherville.json").read_text("utf-8")
)
CORPUS = json.loads(Path("tests/fixtures/corpus_snapshot.json").read_text("utf-8"))
BUILDING_WKT = (
    "POLYGON ((-73.4437522 45.5738006, -73.4432099 45.5735268, -73.4428367 45.5738944, "
    "-73.4433793 45.5741684, -73.4437522 45.5738006))"
)


def access_elements() -> list[dict]:
    return FIXTURE["access_road_query"]["elements"]


def road_elements() -> list[dict]:
    return FIXTURE["roads_query"]["elements"]


def elements_from_corpus() -> list[dict]:
    """Le cache de collecte réel : 28 bâtiments, 28 stationnements, zéro route."""
    return json.loads(Path("work-cache-unused.json").read_text()) if False else []


#: Service du pilote, construit une fois pour toute la suite. Explicite : le
#: référentiel de travail ne se suppose plus nulle part.
SERVICE = None  # renseigné à l'import, après définition du constructeur


def manifest_with(**overrides):
    """Manifeste déclarant ses référentiels, comme tout manifeste produit."""
    from hotel_pipeline.geo.geometry_loader import CURRENT_SCHEMA_VERSION

    reference = boucherville_service().reference
    fields = dict(
        schema_version=CURRENT_SCHEMA_VERSION,
        hotel_id="h",
        snapshots=[snapshot()],
        working_crs=reference.working_crs,
        spatial_context_digest=reference.context_digest(),
    )
    fields.update(overrides)
    return CaptureGeometryManifest(**fields)


def boucherville_service():
    """Service de projection réel du pilote, construit explicitement.

    Aucune fixture ne rétablit un défaut EPSG:2950 : le référentiel se résout
    depuis la position du site, comme en production. Le rétablir « pour sauver
    les tests » réintroduirait précisément le défaut qu'ils doivent surveiller.
    """
    from hotel_pipeline.geo import territory
    from hotel_pipeline.geo.projection import ProjectionService

    return ProjectionService(
        territory.resolve("welcominns-boucherville", 45.574128, -73.443289)
    )


def run(**overrides):
    """Résolution complète sur les réponses figées."""
    fields = dict(
        projection_service=boucherville_service(),
        hotel_id="welcominns-boucherville",
        building_wkt=BUILDING_WKT,
        access_road_ref="way/938806358",
        elements=[],
        elements_digest="cache0000000000",
        roads=road_elements(),
        roads_error=None,
        access_element=access_elements(),
        access_error=None,
        radius_m=350.0,
        parking_ref=None,
        policy_digest="pol0000000000000",
    )
    fields.update(overrides)
    return resolve(**fields)


SERVICE = boucherville_service()


# --- la voie d'accès existe réellement ---------------------------------------


def test_the_access_road_is_really_there_and_really_a_line() -> None:
    """`way/938806358` est absent du cache de collecte, présent chez la source.

    C'est une boucle fermée : l'interpréter comme une surface la rendait
    incompatible avec son rôle de voie, alors qu'OSM la dit linéaire tant
    qu'elle ne porte pas `area=yes`.
    """
    manifest, report = run()
    road = next(g for g in manifest.geometries if g.feature_id == "ACCESS_ROAD_MAIN")

    assert road.resolution_status is GeometryResolutionStatus.RESOLVED
    assert road.geometry_type == "LineString"
    assert road.source_ref == "way/938806358"
    assert shapely_wkt.loads(road.wgs84_wkt).is_closed


def test_the_access_road_is_restricted_not_forbidden() -> None:
    """`access=customers` est un accès conditionnel, non une interdiction.

    Le confondre avec `private` fermait par avance l'allée qui longe l'hôtel,
    alors qu'une capture autorisée par l'établissement y reste envisageable.
    """
    manifest, _ = run()
    road = next(g for g in manifest.geometries if g.feature_id == "ACCESS_ROAD_MAIN")
    corridor = next(
        c for c in manifest.corridors if c.corridor_id == "CORRIDOR_ACCESS_ROAD_MAIN"
    )

    assert corridor.access_status is AccessStatus.RESTRICTED
    assert any("accès conditionnel" in c for c in road.caveats)
    assert road.resolution_status is GeometryResolutionStatus.RESOLVED


def test_access_values_are_graded_not_binary() -> None:
    for value, expected in [
        ("private", AccessStatus.PRIVATE),
        ("no", AccessStatus.PRIVATE),
        ("customers", AccessStatus.RESTRICTED),
        ("permit", AccessStatus.RESTRICTED),
        ("delivery", AccessStatus.RESTRICTED),
        ("destination", AccessStatus.RESTRICTED),
        ("yes", AccessStatus.PUBLIC_CONFIRMED),
    ]:
        status, _ = cg.access_status_of({"highway": "service", "access": value})
        assert status is expected, value


def test_the_access_road_has_its_own_corridor_without_duplicating_its_shape() -> None:
    """Écartée du réseau candidat pour ne pas être résolue deux fois, elle
    disparaissait du classement : la voie la plus importante du site n'y
    figurait pas."""
    manifest, report = run()

    corridor = next(
        c for c in manifest.corridors if c.corridor_id == "CORRIDOR_ACCESS_ROAD_MAIN"
    )
    assert corridor.corridor_class is CorridorClass.ACCESS_MAIN
    assert corridor.feature_id == "ACCESS_ROAD_MAIN"
    assert corridor.admissible_for_building is False
    assert corridor.distance_to_building_m is not None

    # Une seule géométrie pour cette voie, malgré son corridor.
    shapes = [g for g in manifest.geometries if g.source_ref == "way/938806358"]
    assert len(shapes) == 1
    assert len(manifest.corridors) == len(
        [g for g in manifest.geometries
         if g.role in (GeometryRole.ACCESS_ROAD, GeometryRole.ROAD_CANDIDATE)]
    )


def test_an_unresolved_access_road_gets_no_corridor() -> None:
    """Un corridor sans géométrie ne décrirait rien."""
    manifest, _ = run(access_element=[])
    assert not any(
        c.corridor_id == "CORRIDOR_ACCESS_ROAD_MAIN" for c in manifest.corridors
    )


def test_the_adjacency_threshold_comes_from_the_policy() -> None:
    """Une politique posée doit changer le classement, non décorer un rapport."""
    element = {"id": 1, "tags": {"highway": "residential"}}

    strict, _ = cg.classify(element, 12.0, None, None, adjacency_max_m=5.0)
    lenient, _ = cg.classify(element, 12.0, None, None, adjacency_max_m=30.0)

    assert strict is CorridorClass.NON_ADJACENT_POTENTIAL
    assert lenient is CorridorClass.ADJACENT_ROAD


def test_the_policy_threshold_reaches_the_whole_resolution() -> None:
    wide, _ = run(adjacency_max_m=200.0)
    narrow, _ = run(adjacency_max_m=1.0)

    def adjacent(manifest):
        return len([c for c in manifest.corridors
                    if c.corridor_class is CorridorClass.ADJACENT_ROAD])

    assert adjacent(wide) > adjacent(narrow)


def test_every_target_exclusion_is_traced() -> None:
    """Une empreinte voisine retirée par erreur serait autrement invisible."""
    target = shapely_wkt.loads(BUILDING_WKT)
    coords = [{"lat": y, "lon": x} for x, y in target.exterior.coords]
    elements = [
        {"id": 54581348, "type": "way", "tags": {"building": "hotel"}, "geometry": coords},
        # La même empreinte sous une autre référence : recouvrement total.
        {"id": 777, "type": "way", "tags": {"building": "yes"}, "geometry": coords},
    ]

    _, report = run(elements=elements)

    methods = {e["method"]: e for e in report.target_exclusions}
    assert methods["exact_source_ref"]["source_ref"] == "way/54581348"
    doubled = methods.get("equals") or methods.get("overlap_fallback")
    assert doubled["source_ref"] == "way/777"
    assert all(e["target_ref"] == "way/54581348" for e in report.target_exclusions)


def test_a_network_failure_is_never_an_absence() -> None:
    """Une panne ne prouve rien ; l'objet reste inféré au manifeste de site."""
    manifest, report = run(access_element=None, access_error="504 sur tous les miroirs")

    snapshot = next(s for s in manifest.snapshots if s.snapshot_id == "overpass-access-road")
    road = next(g for g in manifest.geometries if g.feature_id == "ACCESS_ROAD_MAIN")

    assert snapshot.status is SourceQueryStatus.DISCOVERY_ERROR
    assert road.resolution_status is GeometryResolutionStatus.UNRESOLVED
    assert "interrogation en échec" in road.unresolved_reason
    assert "reste inféré" in road.unresolved_reason


def test_a_real_absence_is_stated_as_such() -> None:
    manifest, _ = run(access_element=[])

    snapshot = next(s for s in manifest.snapshots if s.snapshot_id == "overpass-access-road")
    road = next(g for g in manifest.geometries if g.feature_id == "ACCESS_ROAD_MAIN")

    assert snapshot.status is SourceQueryStatus.NOT_FOUND
    assert "inconnu de la source" in road.unresolved_reason


def test_a_way_without_geometry_is_unresolved() -> None:
    stripped = [{**access_elements()[0], "geometry": []}]
    manifest, _ = run(access_element=stripped)

    road = next(g for g in manifest.geometries if g.feature_id == "ACCESS_ROAD_MAIN")
    assert road.resolution_status is GeometryResolutionStatus.UNRESOLVED
    assert "sans géométrie" in road.unresolved_reason


# --- classement des voies -----------------------------------------------------


def test_roads_are_classified_and_none_is_admissible_yet() -> None:
    """Aucune route n'est admissible avant la visibilité multi-rayons."""
    manifest, report = run()

    assert manifest.corridors
    assert all(not c.admissible_for_building for c in manifest.corridors)
    assert set(report.corridors) <= {c.value for c in CorridorClass}


def test_a_parking_aisle_is_classified_by_its_service_tag() -> None:
    """Une allée de stationnement n'est ni une voie publique ni un accès."""
    corridor_class, why = cg.classify(
        {"id": 1, "tags": {"highway": "service", "service": "parking_aisle",
                           "access": "customers"}},
        distance_to_building_m=5.0, distance_to_parking_m=0.0, access_ref=None,
    )
    status, _ = cg.access_status_of({"highway": "service", "access": "customers"})

    assert corridor_class is CorridorClass.PARKING_AISLE
    assert "parking_aisle" in why
    assert status is AccessStatus.RESTRICTED


def test_a_private_access_is_read_from_the_tag() -> None:
    status, why = cg.access_status_of({"highway": "service", "access": "private"})
    assert status is AccessStatus.PRIVATE
    assert "private" in why


def test_a_missing_access_tag_is_never_public_confirmed() -> None:
    """La plupart des voies publiques n'en portent aucun ; bien des allées non plus."""
    inferred, _ = cg.access_status_of({"highway": "residential"})
    unknown, _ = cg.access_status_of({"highway": "service"})

    assert inferred is AccessStatus.PUBLIC_INFERRED
    assert unknown is AccessStatus.UNKNOWN


def test_adjacency_is_measured_on_the_shape_not_on_the_box() -> None:
    """Le WelcomINNS est oblique : sa boîte déborde de vingt-cinq mètres.

    Un segment logé dans le coin de la boîte est à zéro mètre d'elle et à
    24,7 m de l'empreinte. Mesurer sur la boîte ferait donc entrer dans
    l'adjacence des voies qui n'y sont pas — et l'écart croît avec l'obliquité.
    """
    target = shapely_wkt.loads(BUILDING_WKT)
    projected = cg.project(target, SERVICE)
    minx, miny, _, _ = target.bounds

    corner = LineString([(minx, miny), (minx + 0.00002, miny + 0.00001)])
    assert corner.within(target.envelope)

    to_box = cg.project(corner, SERVICE).distance(cg.project(target.envelope, SERVICE))
    to_shape = cg.project(corner, SERVICE).distance(projected)

    assert to_box == 0.0
    assert to_shape > 20.0


def test_a_road_beyond_the_threshold_stays_potential() -> None:
    corridor_class, why = cg.classify(
        {"id": 1, "tags": {"highway": "residential"}},
        distance_to_building_m=cg.DEFAULT_ADJACENCY_M + 1, distance_to_parking_m=None,
        access_ref=None,
    )
    assert corridor_class is CorridorClass.NON_ADJACENT_POTENTIAL
    assert "hors adjacence" in why

    adjacent, _ = cg.classify(
        {"id": 2, "tags": {"highway": "residential"}},
        distance_to_building_m=cg.DEFAULT_ADJACENCY_M - 1, distance_to_parking_m=None,
        access_ref=None,
    )
    assert adjacent is CorridorClass.ADJACENT_ROAD


def test_a_non_adjacent_road_cannot_be_declared_admissible() -> None:
    with pytest.raises(ValueError, match="reste potentielle"):
        RoadCorridor(
            corridor_id="c", feature_id="f",
            corridor_class=CorridorClass.NON_ADJACENT_POTENTIAL,
            rationale="à 120 m", admissible_for_building=True,
        )


# --- obstacles ----------------------------------------------------------------


def test_the_target_building_is_never_an_obstacle() -> None:
    """L'oublier ferait rejeter toutes les vues pour occlusion par la cible."""
    target = shapely_wkt.loads(BUILDING_WKT)
    coords = [(x, y) for x, y in target.exterior.coords]
    elements = [
        {"id": 54581348, "type": "way", "tags": {"building": "hotel"},
         "geometry": [{"lat": y, "lon": x} for x, y in coords]},
        {"id": 999, "type": "way", "tags": {"building": "yes"},
         "geometry": [{"lat": 45.5750 + dy, "lon": -73.4450 + dx}
                      for dx, dy in [(0, 0), (0.0004, 0), (0.0004, 0.0003), (0, 0.0003), (0, 0)]]},
    ]

    manifest, _ = run(elements=elements)

    obstacles = manifest.by_role(GeometryRole.OBSTACLE_BUILDING)
    assert [o.source_ref for o in obstacles] == ["way/999"]
    target_geometry = manifest.by_role(GeometryRole.TARGET_BUILDING)[0]
    assert target_geometry.source_ref == "way/54581348"


def test_an_obstacle_without_height_says_so() -> None:
    elements = [
        {"id": 999, "type": "way", "tags": {"building": "yes"},
         "geometry": [{"lat": 45.5750 + dy, "lon": -73.4450 + dx}
                      for dx, dy in [(0, 0), (0.0004, 0), (0.0004, 0.0003), (0, 0.0003), (0, 0)]]},
    ]
    manifest, _ = run(elements=elements)
    obstacle = manifest.by_role(GeometryRole.OBSTACLE_BUILDING)[0]

    assert obstacle.height_known is False
    assert obstacle.height_m is None
    assert any("hauteur inconnue" in c for c in obstacle.caveats)


def test_a_declared_height_comes_from_the_source() -> None:
    elements = [
        {"id": 999, "type": "way", "tags": {"building": "yes", "height": "12.5 m"},
         "geometry": [{"lat": 45.5750 + dy, "lon": -73.4450 + dx}
                      for dx, dy in [(0, 0), (0.0004, 0), (0.0004, 0.0003), (0, 0.0003), (0, 0)]]},
    ]
    manifest, _ = run(elements=elements)
    obstacle = manifest.by_role(GeometryRole.OBSTACLE_BUILDING)[0]

    assert obstacle.height_known is True
    assert obstacle.height_m == 12.5
    assert obstacle.height_source == "tag OSM height"


def test_levels_are_never_converted_into_metres() -> None:
    """Trois étages ne font pas neuf mètres : ce serait une hauteur inventée."""
    from hotel_pipeline.geo.resolve_geometry import _height_of

    assert _height_of({"building:levels": "3"}) == (None, None)


# --- cohérence des deux référentiels ------------------------------------------


def test_the_two_crs_are_verified_by_recomputation() -> None:
    manifest, report = run()
    assert report.crs_problems == []
    assert cg.verify(manifest, SERVICE) == []


def test_swapped_axes_are_caught() -> None:
    """`always_xy=False` échangerait latitude et longitude sans rien dire."""
    swapped = Polygon([(45.57, -73.44), (45.58, -73.44), (45.58, -73.43), (45.57, -73.43)])
    # La forme projetée est fabriquée hors du service : c'est l'entrée fautive
    # que le contrôle doit reconnaître, et le service refuserait de la produire.
    from pyproj import Transformer
    from shapely.ops import transform as shapely_transform

    forward = Transformer.from_crs("EPSG:4326", "EPSG:2950", always_xy=True)
    bad = shapely_transform(
        lambda xs, ys, zs=None: forward.transform(xs, ys), swapped
    )

    problems = cg.check_crs_pair(swapped.wkt, bad.wkt, SERVICE)

    assert any("inversées" in p for p in problems)


def test_a_shifted_projection_is_caught() -> None:
    target = shapely_wkt.loads(BUILDING_WKT)
    from shapely.affinity import translate

    drifted = translate(cg.project(target, SERVICE), xoff=3.0)
    problems = cg.check_crs_pair(target.wkt, drifted.wkt, SERVICE)

    assert any("divergent" in p for p in problems)


def test_an_incompatible_type_is_caught() -> None:
    target = shapely_wkt.loads(BUILDING_WKT)
    line = LineString(list(target.exterior.coords))
    assert any("types incompatibles" in p
               for p in cg.check_crs_pair(target.wkt, cg.project(line, SERVICE).wkt, SERVICE))


def test_a_round_trip_returns_to_the_source() -> None:
    from pyproj import Transformer
    from shapely.ops import transform

    target = shapely_wkt.loads(BUILDING_WKT)
    back = Transformer.from_crs("EPSG:2950", "EPSG:4326", always_xy=True)
    returned = transform(lambda xs, ys, zs=None: back.transform(xs, ys), cg.project(target, SERVICE))

    assert returned.hausdorff_distance(target) < 1e-7


def test_the_digest_ignores_the_writing_not_the_shape() -> None:
    target = shapely_wkt.loads(BUILDING_WKT)
    rewritten = shapely_wkt.loads(target.wkt)
    moved = shapely_wkt.loads(
        BUILDING_WKT.replace("45.5738006", "45.5738106")
    )

    assert cg.canonical_digest(target) == cg.canonical_digest(rewritten)
    assert cg.canonical_digest(target) != cg.canonical_digest(moved)


# --- instantanés et péremption -------------------------------------------------


def test_a_successful_snapshot_must_carry_a_digest() -> None:
    with pytest.raises(ValueError, match="succès sans empreinte"):
        GeometrySourceSnapshot(
            snapshot_id="s", source="overpass", endpoint="e", query="q",
            status=SourceQueryStatus.SUCCESS, element_count=3,
        )


def test_an_error_reports_nothing_and_says_why() -> None:
    with pytest.raises(ValueError, match="panne sans description"):
        GeometrySourceSnapshot(
            snapshot_id="s", source="overpass", endpoint="e", query="q",
            status=SourceQueryStatus.DISCOVERY_ERROR,
        )
    with pytest.raises(ValueError, match="une erreur ne rapporte rien"):
        GeometrySourceSnapshot(
            snapshot_id="s", source="overpass", endpoint="e", query="q",
            status=SourceQueryStatus.DISCOVERY_ERROR, error="504", element_count=2,
        )


def test_an_empty_response_is_not_a_success() -> None:
    with pytest.raises(ValueError, match="réponse vide"):
        GeometrySourceSnapshot(
            snapshot_id="s", source="overpass", endpoint="e", query="q",
            status=SourceQueryStatus.SUCCESS, element_count=0,
            response_digest="abc",
        )


def test_a_changed_source_makes_the_geometry_stale() -> None:
    """Une géométrie périmée n'est pas fausse : elle a perdu son support."""
    manifest, _ = run()
    before = len([g for g in manifest.geometries
                  if g.resolution_status is GeometryResolutionStatus.RESOLVED])

    changed = cg.mark_stale(manifest, {"overpass-access-road": "empreinte-differente"})

    assert "ACCESS_ROAD_MAIN" in changed
    road = next(g for g in manifest.geometries if g.feature_id == "ACCESS_ROAD_MAIN")
    assert road.resolution_status is GeometryResolutionStatus.STALE
    assert road.wgs84_wkt is None
    assert "la source a changé" in road.unresolved_reason
    assert len([g for g in manifest.geometries
                if g.resolution_status is GeometryResolutionStatus.RESOLVED]) == before - 1


# --- invariants du manifeste ---------------------------------------------------


def resolved(feature_id: str, role: GeometryRole, geometry, **extra) -> ResolvedGeometry:
    return cg.resolved_from(
        feature_id, role, f"way/{feature_id}", "snap", geometry,
        "essai", ["preuve"], SERVICE, **extra
    )


def square(x: float = -73.44, y: float = 45.57) -> Polygon:
    return Polygon([(x, y), (x + 0.0004, y), (x + 0.0004, y + 0.0003), (x, y + 0.0003)])


def snapshot() -> GeometrySourceSnapshot:
    return GeometrySourceSnapshot(
        snapshot_id="snap", source="overpass", endpoint="e", query="q",
        status=SourceQueryStatus.SUCCESS, element_count=1, response_digest="d",
    )


def test_a_resolved_geometry_needs_its_provenance() -> None:
    with pytest.raises(ValueError, match="résolue sans provenance"):
        ResolvedGeometry(
            feature_id="f", role=GeometryRole.TARGET_BUILDING,
            resolution_status=GeometryResolutionStatus.RESOLVED,
            wgs84_wkt=square().wkt,
        )


def test_an_unresolved_geometry_carries_no_shape() -> None:
    with pytest.raises(ValueError, match="forme présente sur un état"):
        ResolvedGeometry(
            feature_id="f", role=GeometryRole.ACCESS_ROAD,
            resolution_status=GeometryResolutionStatus.UNRESOLVED,
            unresolved_reason="absente", wgs84_wkt=square().wkt,
        )


def test_an_unresolved_geometry_needs_a_reason() -> None:
    with pytest.raises(ValueError, match="non résolue sans motif"):
        ResolvedGeometry(
            feature_id="f", role=GeometryRole.ACCESS_ROAD,
            resolution_status=GeometryResolutionStatus.UNRESOLVED,
        )


def test_a_role_constrains_the_geometry_type() -> None:
    with pytest.raises(ValueError, match="type 'Polygon'"):
        resolved("f", GeometryRole.ACCESS_ROAD, square())


def test_always_xy_must_be_explicit() -> None:
    geometry = resolved("f", GeometryRole.TARGET_BUILDING, square())
    with pytest.raises(ValueError, match="always_xy"):
        geometry.model_copy(update={"always_xy": None}).model_validate(
            geometry.model_copy(update={"always_xy": None}).model_dump()
        )


def test_a_corridor_geometry_must_cite_its_roads() -> None:
    corridor = resolved("CORRIDOR", GeometryRole.CONTEXT_CORRIDOR, square())
    with pytest.raises(ValueError, match="sans filiation"):
        manifest_with(geometries=[corridor])


def test_a_relation_to_an_absent_geometry_is_refused() -> None:
    corridor = resolved("CORRIDOR", GeometryRole.CONTEXT_CORRIDOR, square(),
                        derived_from=["ROAD_1"])
    with pytest.raises(ValueError, match="dérive de formes absentes"):
        manifest_with(geometries=[corridor])


def test_a_corridor_pointing_at_an_absent_feature_is_refused() -> None:
    with pytest.raises(ValueError, match="absente du manifeste"):
        manifest_with(
            snapshots=[snapshot()],
            corridors=[RoadCorridor(corridor_id="c", feature_id="fantome",
                                    corridor_class=CorridorClass.EXCLUDED,
                                    rationale="essai")],
        )


def test_the_target_cannot_also_be_an_obstacle() -> None:
    shape = square()
    with pytest.raises(ValueError, match="figure parmi les obstacles"):
        manifest_with(
            snapshots=[snapshot()],
            geometries=[
                resolved("T", GeometryRole.TARGET_BUILDING, shape),
                resolved("T", GeometryRole.OBSTACLE_BUILDING, shape).model_copy(
                    update={"feature_id": "O"}
                ),
            ],
        )


def test_duplicate_feature_ids_are_refused() -> None:
    with pytest.raises(ValueError, match="dupliqués"):
        manifest_with(
            snapshots=[snapshot()],
            geometries=[
                resolved("F", GeometryRole.TARGET_BUILDING, square()),
                resolved("F", GeometryRole.OBSTACLE_BUILDING, square(-73.45)),
            ],
        )


def test_an_unknown_snapshot_is_refused() -> None:
    with pytest.raises(ValueError, match="instantané 'snap' absent"):
        # Sans instantané déclaré : une géométrie ne peut pas citer une réponse
        # source que le manifeste ne contient pas.
        manifest_with(
            snapshots=[],
            geometries=[resolved("F", GeometryRole.TARGET_BUILDING, square())],
        )


# --- le manifeste géométrique ne touche pas aux objets -------------------------


def test_the_manifest_never_states_anything_about_the_site_objects() -> None:
    """`ACCESS_ROAD_MAIN` peut rester inféré avec une géométrie non résolue."""
    fields = set(CaptureGeometryManifest.model_fields)
    assert "objects" not in fields
    assert "object_states" not in fields

    manifest, _ = run(access_element=[])
    road = next(g for g in manifest.geometries if g.feature_id == "ACCESS_ROAD_MAIN")
    assert road.resolution_status is GeometryResolutionStatus.UNRESOLVED
