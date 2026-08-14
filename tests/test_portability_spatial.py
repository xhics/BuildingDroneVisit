"""Territoire et référentiels dynamiques (portabilité, commit 2).

Le défaut mesuré avant ce lot : Lyon recevait le territoire `QC`, se voyait
proposer le LiDAR québécois, et se projetait en EPSG:2950 — à 5 637 km à l'est,
sans qu'une seule erreur ne soit levée. Distances, azimuts et occlusions se
calculaient là-dessus, et le rapport avait l'air normal.
"""

from __future__ import annotations

import pytest

from hotel_pipeline.geo import territory
from hotel_pipeline.geo.catalog import route
from hotel_pipeline.geo.projection import ProjectionRefused, ProjectionService
from hotel_pipeline.geo.visibility_engine import (
    CameraVertical,
    Obstacle,
    TargetVertical,
    vertical_verdict,
)
from hotel_pipeline.schemas.spatial_reference import (
    HeightType,
    SpatialReferenceContext,
    TerritoryState,
    VerticalReference,
    VerticalTransform,
)
from hotel_pipeline.schemas.visibility import VerticalVisibilityStatus

BOUCHERVILLE = (45.574128, -73.443289)
LYON = (45.7640, 4.8357)
OPEN_SEA = (0.0, -30.0)


# --- territoire --------------------------------------------------------------


def test_a_point_outside_known_jurisdictions_is_not_quebec() -> None:
    """Le défaut central : `territories = {"QC"}` était inconditionnel."""
    assert territory.jurisdictions_for(*OPEN_SEA) == []
    assert "QC" not in territory.jurisdictions_for(*LYON)


def test_lyon_resolves_to_france_and_boucherville_to_quebec() -> None:
    lyon = territory.resolve("lyon", *LYON)
    pilot = territory.resolve("pilote", *BOUCHERVILLE)

    assert lyon.jurisdictions == ["FR"]
    assert pilot.jurisdictions == ["CA", "QC", "QC-CMM", "QC-MONTEREGIE"]
    assert lyon.territory_state is pilot.territory_state is TerritoryState.RESOLVED


def test_an_unknown_territory_is_a_state_not_an_absence() -> None:
    nowhere = territory.resolve("nulle-part", *OPEN_SEA)

    assert nowhere.territory_state is TerritoryState.UNKNOWN
    assert nowhere.working_crs is None
    assert not nowhere.is_resolved


def test_no_source_is_proposed_on_an_unresolved_territory() -> None:
    """« Aucune source » et « je ne sais pas où je suis » diffèrent."""
    routing = route(*OPEN_SEA)

    assert routing.territories == set()
    assert routing.territorial_candidates == []
    assert all("non résolu" in reason for reason in routing.rejected.values())


def test_lyon_is_offered_no_quebec_source() -> None:
    routing = route(*LYON)

    assert routing.territories == {"FR"}
    assert routing.territorial_candidates == []
    assert "lidar-quebec" in routing.rejected


def test_boucherville_keeps_the_sources_it_had() -> None:
    """Rendre le routage portable ne doit rien retirer au pilote."""
    routing = route(*BOUCHERVILLE)
    offered = {source.source_id for source in routing.territorial_candidates}

    assert "lidar-quebec" in offered
    assert "cadastre-quebec" in offered
    # L'exclusion GéoMont tient toujours : la CMM est hors de son emprise.
    assert "geomont-ortho-2023" in routing.rejected


# --- référentiel de travail ---------------------------------------------------


def test_the_pilot_keeps_epsg_2950() -> None:
    resolved = territory.resolve("pilote", *BOUCHERVILLE)

    assert resolved.working_crs == "EPSG:2950"
    assert resolved.working_unit == "m"
    assert resolved.contains(*BOUCHERVILLE)


def test_lyon_gets_lambert_93_never_2950() -> None:
    resolved = territory.resolve("lyon", *LYON)

    assert resolved.working_crs == "EPSG:2154"
    assert resolved.working_crs != "EPSG:2950"


def test_a_working_crs_declares_unit_axes_extent_and_reason() -> None:
    """Un CRS dont on ne peut rien dire n'est pas opposable."""
    with pytest.raises(ValueError, match="working_unit"):
        SpatialReferenceContext(
            hotel_id="h", reference_lat=45.5, reference_lon=-73.4,
            territory_state=TerritoryState.RESOLVED,
            jurisdictions=["QC"], working_crs="EPSG:2950",
        )


def test_no_working_crs_is_chosen_on_an_unknown_territory() -> None:
    with pytest.raises(ValueError, match="territoire inconnu"):
        SpatialReferenceContext(
            hotel_id="h", reference_lat=0.0, reference_lon=-30.0,
            territory_state=TerritoryState.UNKNOWN,
            working_crs="EPSG:2950", working_unit="m",
            working_axes="easting,northing",
            working_area_of_use=[-75.0, 44.98, -72.0, 62.53],
            selection_method="au hasard",
        )


# --- service de projection ----------------------------------------------------


def pilot_service() -> ProjectionService:
    return ProjectionService(territory.resolve("pilote", *BOUCHERVILLE))


def test_the_pilot_projects_where_it_always_did() -> None:
    x, y = pilot_service().point(*BOUCHERVILLE)

    assert round(x, 1) == 309226.3
    assert round(y, 1) == 5048247.1


def test_a_position_outside_the_extent_is_refused_before_any_calculation() -> None:
    """Le défaut le plus dangereux : pyproj rendait des mètres finis et faux."""
    with pytest.raises(ProjectionRefused, match="hors de l'emprise"):
        pilot_service().point(*LYON)


def test_every_camera_is_checked_not_only_the_centre() -> None:
    """Un obstacle ou une caméra hors fuseau déforme le calcul sans le dire."""
    service = pilot_service()
    positions = [BOUCHERVILLE, (45.575, -73.444), LYON]

    with pytest.raises(ProjectionRefused, match="1 position"):
        service.check_within_area(positions, "positions caméra")

    # Sans l'intrus, les trois contrôles passent.
    report = service.verify(positions[:2], "positions caméra")
    assert report["positions"] == 2
    assert report["max_roundtrip_deviation_deg"] < 1e-7


def test_an_unresolved_context_cannot_build_a_projection() -> None:
    with pytest.raises(ProjectionRefused, match="non résolu"):
        ProjectionService(territory.resolve("nulle-part", *OPEN_SEA))


def test_geometry_and_visibility_share_one_service() -> None:
    """La même opération était écrite deux fois, une seule protégée."""
    import pathlib

    source = pathlib.Path("src/hotel_pipeline/geo/visibility_run.py").read_text("utf-8")
    code = "\n".join(line.split("#", 1)[0] for line in source.splitlines())

    assert 'Transformer.from_crs("EPSG:4326", "EPSG:2950"' not in code
    assert "ProjectionService" in source


# --- référentiel vertical -----------------------------------------------------


def test_an_unknown_vertical_reference_forbids_qualification_not_measurement() -> None:
    resolved = territory.resolve("pilote", *BOUCHERVILLE)

    assert resolved.vertical.crs is None
    assert not resolved.vertical_is_usable
    # Mesurer reste possible : c'est qualifier une hauteur qui ne l'est pas.
    assert resolved.is_resolved


def test_the_vertical_reference_comes_from_the_source_never_from_the_country() -> None:
    declared = territory.vertical_from_acquisition(
        {"sources": [{"crs_vertical": "CGVD 1928"}]}
    )
    silent = territory.vertical_from_acquisition({"sources": [{"tile_id": "x"}]})

    assert declared.crs == "CGVD 1928"
    assert declared.height_type is HeightType.ORTHOMETRIC
    assert silent.crs is None
    assert silent.height_type is HeightType.UNKNOWN


def test_two_declared_vertical_references_establish_nothing() -> None:
    """Choisir le plus fréquent reviendrait à trancher au jugé."""
    mixed = territory.vertical_from_acquisition(
        {"sources": [{"crs_vertical": "CGVD 1928"}, {"crs_vertical": "CGVD2013"}]}
    )

    assert mixed.crs is None
    assert "aucun ne fait autorité" in mixed.provenance


def test_an_unnamed_vertical_crs_keeps_an_unknown_height_type() -> None:
    """Le nom d'un référentiel ne dit pas ce qu'il mesure."""
    exotic = territory.vertical_from_acquisition(
        {"sources": [{"crs_vertical": "référentiel local du chantier"}]}
    )

    assert exotic.crs == "référentiel local du chantier"
    assert exotic.height_type is HeightType.UNKNOWN


# --- comparaison verticale ----------------------------------------------------


def known_vertical(**overrides) -> VerticalReference:
    fields = dict(crs="CGVD2013", height_type=HeightType.ORTHOMETRIC)
    fields.update(overrides)
    return VerticalReference(**fields)


def blocking_case(camera_crs=None, obstacle_crs=None, target_crs=None):
    """Une géométrie où l'obstacle masque prouvablement la cible."""
    obstacle = Obstacle(
        feature_id="way/1", shape=None, height_m=30.0, ground_m=10.0,
        vertical_crs=obstacle_crs,
    )
    camera = CameraVertical(
        ground_m=10.0, height_above_ground_m=2.0, vertical_crs=camera_crs
    )
    target = TargetVertical(ground_m=10.0, height_m=12.0, vertical_crs=target_crs)
    return obstacle, camera, target


def test_matching_references_still_prove_a_block() -> None:
    obstacle, camera, target = blocking_case("CGVD2013", "CGVD2013", "CGVD2013")

    proven, status, missing = vertical_verdict(
        None, obstacle, 20.0, 60.0, camera, target, known_vertical()
    )

    assert proven is True
    assert status is VerticalVisibilityStatus.FULLY_KNOWN
    assert missing == []


def test_mixed_references_never_prove_a_block() -> None:
    """Orthométrique et ellipsoïdal diffèrent de dizaines de mètres ici."""
    obstacle, camera, target = blocking_case("CGVD2013", "NAD83-ellipsoidal", "CGVD2013")

    proven, status, missing = vertical_verdict(
        None, obstacle, 20.0, 60.0, camera, target, known_vertical()
    )

    assert proven is False
    assert status is VerticalVisibilityStatus.UNKNOWN
    assert any("sans transformation déclarée" in item for item in missing)


def test_an_undeclared_reference_is_not_assumed_to_match() -> None:
    obstacle, camera, target = blocking_case("CGVD2013", None, "CGVD2013")

    proven, status, missing = vertical_verdict(
        None, obstacle, 20.0, 60.0, camera, target, known_vertical()
    )

    assert proven is False
    assert status is VerticalVisibilityStatus.UNKNOWN
    assert any("non déclaré" in item for item in missing)


def test_a_declared_transform_restores_comparability() -> None:
    """Deux référentiels différents ne sont pas toujours incompatibles."""
    reference = known_vertical(
        transforms=[
            VerticalTransform(
                source_crs="NAD83-ellipsoidal", target_crs="CGVD2013",
                source_height_type=HeightType.ELLIPSOIDAL,
                target_height_type=HeightType.ORTHOMETRIC,
                operation="EPSG:9985 — CGG2013a", geoid_model="CGG2013a",
                accuracy_m=0.03, provenance="registre EPSG",
            )
        ]
    )
    obstacle, camera, target = blocking_case(
        "CGVD2013", "NAD83-ellipsoidal", "CGVD2013"
    )

    proven, status, missing = vertical_verdict(
        None, obstacle, 20.0, 60.0, camera, target, reference
    )

    assert proven is True
    assert status is VerticalVisibilityStatus.FULLY_KNOWN
    assert missing == []


def test_a_transform_crossing_height_types_must_name_its_geoid() -> None:
    """Sans géoïde, l'écart serait appliqué au jugé."""
    with pytest.raises(ValueError, match="modèle de géoïde"):
        VerticalTransform(
            source_crs="A", target_crs="B",
            source_height_type=HeightType.ELLIPSOIDAL,
            target_height_type=HeightType.ORTHOMETRIC,
            operation="au jugé", provenance="nulle part",
        )


def test_a_transform_is_never_declared_on_an_unknown_height_type() -> None:
    with pytest.raises(ValueError, match="type de hauteur inconnu"):
        VerticalTransform(
            source_crs="A", target_crs="B",
            source_height_type=HeightType.UNKNOWN,
            target_height_type=HeightType.ORTHOMETRIC,
            operation="x", provenance="y",
        )


def test_runs_without_a_declared_reference_keep_their_behaviour() -> None:
    """Les runs déjà produits sur une source verticale unique restent rejouables."""
    obstacle, camera, target = blocking_case()

    proven, status, _ = vertical_verdict(
        None, obstacle, 20.0, 60.0, camera, target, None
    )

    assert proven is True
    assert status is VerticalVisibilityStatus.FULLY_KNOWN
