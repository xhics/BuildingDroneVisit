"""Raccord du socle portable au pipeline réel (portabilité, commit 2b).

Deux intégrations, et non des contrôles unitaires : c'est la chaîne qui avait
laissé passer le défaut. Le socle résolvait bien Lyon en EPSG:2154, mais
`geo resolve` ne transmettait pas le service, `capture_geometry` gardait son
repli vers EPSG:2950, et `check_crs_pair` contrôlait une emprise puis
recalculait avec l'autre référentiel.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from shapely.geometry import Polygon

from hotel_pipeline.geo import capture_geometry as cg
from hotel_pipeline.geo import territory
from hotel_pipeline.geo.geometry_loader import (
    CURRENT_SCHEMA_VERSION,
    LegacyManifestRefused,
    bind_legacy,
    is_legacy,
    load_capture_geometry,
)
from hotel_pipeline.geo.projection import ProjectionService
from hotel_pipeline.geo.resolve_geometry import resolve
from hotel_pipeline.geo.visibility_run import check_spatial_agreement

LYON = (45.7640, 4.8357)
BOUCHERVILLE = (45.574128, -73.443289)

#: Empreinte lyonnaise fictive, autour de la position de référence.
LYON_BUILDING = Polygon(
    [(4.8355, 45.7638), (4.8359, 45.7638), (4.8359, 45.7642), (4.8355, 45.7642)]
)


def service_for(hotel_id: str, lat: float, lon: float) -> ProjectionService:
    return ProjectionService(territory.resolve(hotel_id, lat, lon))


# --- Lyon, de bout en bout ----------------------------------------------------


def test_lyon_resolves_geometry_entirely_in_lambert_93() -> None:
    """`geo reference → geo resolve`, sans une trace d'EPSG:2950."""
    service = service_for("lyon", *LYON)
    assert service.working_crs == "EPSG:2154"

    manifest, report = resolve(
        hotel_id="lyon",
        building_wkt=LYON_BUILDING.wkt,
        access_road_ref=None,
        elements=[],
        elements_digest="cache-lyon",
        roads=[],
        roads_error=None,
        access_element=None,
        access_error=None,
        radius_m=350.0,
        parking_ref=None,
        policy_digest="pol-lyon",
        projection_service=service,
    )

    assert manifest.working_crs == "EPSG:2154"
    assert manifest.schema_version == CURRENT_SCHEMA_VERSION
    assert manifest.spatial_context_digest == service.reference.context_digest()

    resolved_geometries = [
        geometry for geometry in manifest.geometries if geometry.projected_crs
    ]
    assert resolved_geometries
    assert {geometry.projected_crs for geometry in resolved_geometries} == {"EPSG:2154"}

    # Le contrôle des deux référentiels reprojette avec **le même** service :
    # contrôler l'emprise de 2154 puis recalculer en 2950 déclarait divergente
    # toute géométrie lyonnaise valide.
    assert report.crs_problems == []

    serialised = manifest.model_dump_json()
    assert "EPSG:2950" not in serialised


def test_lyon_visibility_agrees_with_its_own_context() -> None:
    """Le manifeste et le contexte doivent décrire le même espace."""
    service = service_for("lyon", *LYON)
    manifest, _ = resolve(
        hotel_id="lyon", building_wkt=LYON_BUILDING.wkt, access_road_ref=None,
        elements=[], elements_digest="c", roads=[], roads_error=None,
        access_element=None, access_error=None, radius_m=350.0, parking_ref=None,
        policy_digest="p", projection_service=service,
    )

    assert check_spatial_agreement(manifest, service.reference) == []


def test_a_quebec_context_is_refused_on_a_lyon_manifest() -> None:
    """Le cas que rien n'attrapait : caméras en 2154, cible en 2950."""
    service = service_for("lyon", *LYON)
    manifest, _ = resolve(
        hotel_id="lyon", building_wkt=LYON_BUILDING.wkt, access_road_ref=None,
        elements=[], elements_digest="c", roads=[], roads_error=None,
        access_element=None, access_error=None, radius_m=350.0, parking_ref=None,
        policy_digest="p", projection_service=service,
    )

    problems = check_spatial_agreement(
        manifest, territory.resolve("pilote", *BOUCHERVILLE)
    )

    assert problems
    assert any("EPSG:2154" in problem and "EPSG:2950" in problem for problem in problems)
    assert any("contexte spatial" in problem for problem in problems)


def test_lyon_never_projects_through_the_quebec_zone() -> None:
    """Le défaut le plus dangereux : des mètres finis et faux."""
    from hotel_pipeline.geo.projection import ProjectionRefused

    quebec = service_for("pilote", *BOUCHERVILLE)

    with pytest.raises(ProjectionRefused, match="hors de l'emprise"):
        cg.project(LYON_BUILDING, quebec)

    # Avec son propre service, la même forme passe sans réserve.
    assert cg.project(LYON_BUILDING, service_for("lyon", *LYON)).is_valid


def test_no_projection_happens_without_a_service() -> None:
    with pytest.raises(ValueError, match="ne se suppose pas"):
        cg.project(LYON_BUILDING, None)


def test_the_engine_refuses_to_invent_a_reference_frame() -> None:
    """`crs` n'a pas de valeur par défaut, et ne doit jamais en reprendre une.

    Le défaut retiré était littéral — `if crs is None: crs = "EPSG:2950"`. Un
    argument obligatoire ne se vérifie pas en le passant : il se vérifie en
    l'omettant.
    """
    import inspect

    from hotel_pipeline.geo import visibility_engine as engine

    signature = inspect.signature(engine.assess)
    parameter = signature.parameters["crs"]

    assert parameter.default is inspect.Parameter.empty
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY

    with pytest.raises(TypeError, match="crs"):
        engine.assess(
            "a", "s", "t", (0.0, 0.0), LYON_BUILDING, [],
            __import__(
                "hotel_pipeline.schemas", fromlist=["DEFAULT_POLICY"]
            ).DEFAULT_POLICY.visibility,
        )


def test_no_module_hardcodes_the_pilot_reference_frame() -> None:
    """Le littéral EPSG:2950 n'a plus sa place dans un chemin de calcul.

    Il reste légitime dans le catalogue de référentiels, l'adaptateur legacy et
    les commentaires qui expliquent la correction — pas ailleurs.
    """
    import ast
    import pathlib

    #: `territory` porte le catalogue des référentiels, `geometry_loader` le
    #: référentiel implicite des fichiers antérieurs, `geometry` les constantes
    #: du schéma. Ailleurs, un littéral serait un choix codé en dur.
    allowed = {"territory.py", "geometry_loader.py", "geometry.py"}

    offenders = []
    for path in pathlib.Path("src/hotel_pipeline").rglob("*.py"):
        if path.name in allowed:
            continue
        tree = ast.parse(path.read_text("utf-8"))
        # Les docstrings expliquent la correction et doivent rester ; seules
        # les chaînes qui *agissent* comptent. On les distingue en retirant
        # d'abord toutes les docstrings de l'arbre.
        docstrings = {
            id(ast.get_docstring(node, clean=False))
            for node in ast.walk(tree)
            if isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            )
        }
        literals = {
            id(node.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if "EPSG:2950" not in node.value:
                continue
            if node.value in docstrings or id(node) in literals:
                continue
            offenders.append(f"{path.name}:{node.lineno}")

    assert offenders == []


def test_run_assessment_refuses_a_manifest_from_another_space() -> None:
    """Le contrôle doit être **appelé**, pas seulement exister.

    Sans cela, les caméras se projetaient dans le référentiel du contexte
    pendant que la cible et les obstacles restaient dans celui du manifeste.
    """
    from hotel_pipeline.geo.visibility_run import run_assessment
    from hotel_pipeline.schemas import DEFAULT_POLICY

    service = service_for("lyon", *LYON)
    manifest, _ = resolve(
        hotel_id="lyon", building_wkt=LYON_BUILDING.wkt, access_road_ref=None,
        elements=[], elements_digest="c", roads=[], roads_error=None,
        access_element=None, access_error=None, radius_m=350.0, parking_ref=None,
        policy_digest="p", projection_service=service,
    )
    digests = {
        "capture_geometry": "a", "policy": "b", "site_manifest": "c",
        "asset_files": "d", "asset_manifest": "e", "obstacles": "f", "roads": "g",
    }

    with pytest.raises(ValueError, match="EPSG:2950"):
        run_assessment(
            "run-1", "lyon", [], manifest, DEFAULT_POLICY, digests,
            spatial_reference=territory.resolve("pilote", *BOUCHERVILLE),
        )


def test_run_assessment_refuses_to_run_without_any_context() -> None:
    from hotel_pipeline.geo.visibility_run import run_assessment
    from hotel_pipeline.schemas import DEFAULT_POLICY

    with pytest.raises(ValueError, match="aucun contexte spatial"):
        run_assessment("run-1", "lyon", [], None, DEFAULT_POLICY, {})


def test_visibility_apply_compares_the_reference_frame_and_its_context() -> None:
    """Projeter les mesures d'un fuseau sur les assets d'un autre.

    Les nombres restaient plausibles, et rien ne les démentait : le contrôle
    doit figurer dans `verify()`, et pas seulement dans le moteur.
    """
    from hotel_pipeline.geo import visibility_apply
    from hotel_pipeline.schemas import AssetManifest
    from hotel_pipeline.schemas.visibility import VisibilityRun

    pilot = territory.resolve("h", *BOUCHERVILLE)
    manifest = AssetManifest(hotel_id="h", assets=[])
    from hotel_pipeline.geo.visibility_run import base_manifest_digest

    run = VisibilityRun(
        run_id="r", hotel_id="h", engine_version="multiray-1.0.0",
        method="uniform_angular_cells", parameters={"max_angular_step_deg": "0.25"},
        capture_geometry_digest="a", policy_digest="b", site_manifest_digest="c",
        asset_files_digest="d", asset_manifest_digest=base_manifest_digest(manifest),
        target_digest="t", obstacles_digest="o", road_geometry_digest="rd",
        # Mesuré ailleurs : autre référentiel, autre contexte.
        spatial_context_digest=territory.resolve("lyon", *LYON).context_digest(),
        crs="EPSG:2154",
    )
    current = {
        "policy": "b", "capture_geometry": "a", "site_manifest": "c",
        "asset_files": "d", "obstacles": "o", "roads": "rd", "target": "t",
    }

    problems = visibility_apply.verify(run, manifest, "h", current, pilot)

    assert any("référentiel" in problem for problem in problems)
    assert any("contexte spatial" in problem for problem in problems)

    # Sans contexte courant, l'absence est dite plutôt que passée sous silence.
    blind = visibility_apply.verify(run, manifest, "h", current, None)
    assert any("aucun contexte spatial courant" in problem for problem in blind)


# --- Boucherville : le manifeste publié reste lisible, et intact ---------------

PILOT_MANIFEST = Path(
    "work/welcominns-boucherville/06_geo/capture_geometry.json"
)


def pilot_payload() -> dict:
    if not PILOT_MANIFEST.is_file():  # pragma: no cover — dépend du corpus local
        pytest.skip("manifeste du pilote absent")
    return json.loads(PILOT_MANIFEST.read_text("utf-8"))


def test_the_published_manifest_is_recognised_as_legacy() -> None:
    """Il a été écrit avant que le référentiel soit une donnée."""
    payload = pilot_payload()

    assert is_legacy(payload)
    assert "schema_version" not in payload
    assert "working_crs" not in payload


def test_a_missing_schema_version_is_never_filled_silently() -> None:
    """Lui donner « 1.0.0 » lui prêterait des garanties qu'il n'a pas eues."""
    from hotel_pipeline.schemas.geometry import CaptureGeometryManifest

    with pytest.raises(ValueError, match="schema_version"):
        CaptureGeometryManifest.model_validate(pilot_payload())


def test_the_legacy_manifest_loads_against_the_current_context() -> None:
    reference = territory.resolve("welcominns-boucherville", *BOUCHERVILLE)

    bound = bind_legacy(pilot_payload(), reference)

    assert bound.working_crs == "EPSG:2950"
    assert bound.schema_version == "1.0.0-legacy"
    assert bound.spatial_context_digest == reference.context_digest()
    assert check_spatial_agreement(bound, reference) == []


def test_binding_never_rewrites_the_file(tmp_path) -> None:
    """Un artefact publié dont l'empreinte est citée ne se réécrit pas."""
    before = PILOT_MANIFEST.read_bytes()

    manifest, was_legacy = load_capture_geometry(
        PILOT_MANIFEST, territory.resolve("welcominns-boucherville", *BOUCHERVILLE)
    )

    assert was_legacy
    assert manifest.working_crs == "EPSG:2950"
    assert PILOT_MANIFEST.read_bytes() == before


def test_a_legacy_manifest_is_refused_under_a_foreign_context() -> None:
    """Un manifeste québécois relu sous un contexte lyonnais n'est pas lu."""
    with pytest.raises(LegacyManifestRefused, match="EPSG:2154"):
        bind_legacy(pilot_payload(), territory.resolve("lyon", *LYON))


def test_a_legacy_manifest_needs_a_context_to_be_bound() -> None:
    with pytest.raises(LegacyManifestRefused, match="geo reference"):
        bind_legacy(pilot_payload(), None)


# --- découverte : un adaptateur, ou rien ---------------------------------------


def test_discovery_is_routed_to_an_adapter_never_assumed() -> None:
    from hotel_pipeline.geo.adapters import elevation_adapter
    from hotel_pipeline.geo.catalog import route

    pilot, reasons = elevation_adapter(route(*BOUCHERVILLE))
    assert pilot is not None
    assert pilot.source_id == "lidar-quebec"
    assert reasons == []


def test_lyon_issues_no_query_at_all() -> None:
    """Interroger le WFS québécois pour Lyon faisait passer son silence pour
    une absence de couverture."""
    from hotel_pipeline.geo.adapters import elevation_adapter
    from hotel_pipeline.geo.catalog import route

    adapter, reasons = elevation_adapter(route(*LYON))

    assert adapter is None
    assert any("aucune source territorialement admissible" in r for r in reasons)
