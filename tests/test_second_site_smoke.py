"""Smoke test d'un second établissement, hors Québec (portabilité, commit 3).

La chaîne complète, sur un hôtel lyonnais, sans charger un seul fichier du
WelcomINNS :

```text
init → profile → site manifest → source routing → capture geometry
     → demands → discover --dry-run
```

Ce qui est éprouvé n'est pas qu'elle « marche » : c'est qu'elle **refuse au bon
endroit**. Un second site doit aller jusqu'au bout de ce qui est possible, puis
s'arrêter en disant ce qui manque — sans repli vers le Québec, sans EPSG:2950,
sans calibration empruntée, et sans télécharger quoi que ce soit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hotel_pipeline.cli import app

runner = CliRunner()

LYON = (45.7640, 4.8357)
HOTEL = "hotel-lyon-part-dieu"

#: Empreinte fictive autour de la position de référence.
BUILDING = (
    "POLYGON ((4.83540 45.76380, 4.83600 45.76380, "
    "4.83600 45.76430, 4.83540 45.76430, 4.83540 45.76380))"
)


@pytest.fixture
def second_site(tmp_path, monkeypatch):
    """Un espace de travail neuf, sans aucun artefact du pilote."""
    monkeypatch.setenv("HOTEL_PIPELINE_WORK", str(tmp_path / "work"))
    monkeypatch.setenv("HOTEL_PIPELINE_PROFILES", str(tmp_path / "profiles"))
    monkeypatch.setenv("HOTEL_PIPELINE_OFFLINE", "1")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, [
        "init", HOTEL, "--address", "5 place Charles Béraudier, 69003 Lyon",
        "--name", "Hôtel Part-Dieu", "--country", "FR", "--timezone",
        "Europe/Paris", "--ocr-language", "fr",
        "--lat", str(LYON[0]), "--lon", str(LYON[1]),
    ])
    assert result.exit_code == 0, result.output
    return tmp_path


def read(tmp_path: Path, relative: str) -> dict:
    return json.loads((tmp_path / "work" / HOTEL / relative).read_text("utf-8"))


# --- init et profil -----------------------------------------------------------


def test_init_creates_a_profile_that_owes_nothing_to_the_pilot(second_site) -> None:
    profile = json.loads(
        (second_site / "profiles" / f"{HOTEL}.json").read_text("utf-8")
    )

    assert profile["country_code"] == "FR"
    assert profile["timezone"] == "Europe/Paris"
    assert profile["ocr_languages"] == ["fr"]
    # Aucune langue héritée, aucun concurrent hérité, aucune emprise héritée.
    assert "en" not in profile["ocr_languages"]
    assert profile["competitor_names"] == []
    assert profile["footprint_min_m2"] is None


def test_the_new_policy_carries_no_borrowed_calibration(second_site) -> None:
    from hotel_pipeline.schemas import PipelinePolicy
    from hotel_pipeline.schemas.policy import UNCALIBRATED

    policy_path = second_site / "work" / HOTEL / "00_manifest" / "pipeline_policy.json"
    if not policy_path.is_file():
        from hotel_pipeline.schemas import DEFAULT_POLICY as policy
    else:  # pragma: no cover — selon que `init` matérialise ou non
        policy = PipelinePolicy.model_validate_json(policy_path.read_text("utf-8"))

    for section in (policy.model, policy.terrain, policy.qualification):
        assert section.calibration_id == UNCALIBRATED
        assert section.calibrated_on_sites == 0

    assert "welcominns" not in policy.model_dump_json().lower()


def test_no_pilot_file_is_ever_read(second_site) -> None:
    """Le critère central : aucune dépendance au projet du pilote."""
    for name in ("00_manifest", "01_sources", "06_geo"):
        directory = second_site / "work" / HOTEL / name
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if path.is_file():
                assert "welcominns" not in path.read_text("utf-8", errors="ignore").lower()


# --- référentiels -------------------------------------------------------------


def test_geo_reference_resolves_france_never_quebec(second_site) -> None:
    result = runner.invoke(app, ["geo", "reference", HOTEL])

    assert result.exit_code == 0, result.output
    reference = read(second_site, "00_manifest/spatial_reference.json")

    assert reference["territory_state"] == "resolved"
    assert reference["jurisdictions"] == ["FR"]
    assert reference["working_crs"] == "EPSG:2154"
    assert "QC" not in reference["jurisdictions"]
    assert "EPSG:2950" not in json.dumps(reference)


def test_the_vertical_reference_stays_unknown_rather_than_invented(
    second_site,
) -> None:
    runner.invoke(app, ["geo", "reference", HOTEL])
    reference = read(second_site, "00_manifest/spatial_reference.json")

    # Aucune donnée n'a été acquise : le référentiel vertical est inconnu, et
    # le rester est la bonne réponse. Le déduire du pays serait une invention.
    assert reference["vertical"]["crs"] is None
    assert reference["vertical"]["height_type"] == "unknown"


# --- routage des sources ------------------------------------------------------


def test_source_routing_offers_no_quebec_source(second_site) -> None:
    from hotel_pipeline.geo.catalog import route

    routing = route(*LYON)

    assert routing.territories == {"FR"}
    assert routing.territorial_candidates == []
    assert "lidar-quebec" in routing.rejected
    assert "cadastre-quebec" in routing.rejected


# --- besoins : ce que le site demande, ce qu'aucune source ne fournit ---------


def test_demands_are_stated_and_none_can_be_met(second_site) -> None:
    """Un besoin non couvert doit se lire, pas se deviner."""
    from hotel_pipeline.geo.catalog import route
    from hotel_pipeline.schemas.critical_objects import REQUIRED_OBJECTS

    routing = route(*LYON)
    unmet = {kind: routing.for_object(kind) for kind in REQUIRED_OBJECTS}

    assert REQUIRED_OBJECTS, "le gabarit doit déclarer des objets requis"
    assert all(candidates == [] for candidates in unmet.values())


# --- géométrie de capture -----------------------------------------------------


def test_capture_geometry_is_resolved_entirely_in_lambert_93(second_site) -> None:
    from hotel_pipeline.geo import territory
    from hotel_pipeline.geo.projection import ProjectionService
    from hotel_pipeline.geo.resolve_geometry import resolve

    service = ProjectionService(territory.resolve(HOTEL, *LYON))
    manifest, report = resolve(
        hotel_id=HOTEL, building_wkt=BUILDING, access_road_ref=None,
        elements=[], elements_digest="cache-lyon", roads=[], roads_error=None,
        access_element=None, access_error=None, radius_m=350.0, parking_ref=None,
        policy_digest="pol-lyon", projection_service=service,
    )

    assert manifest.working_crs == "EPSG:2154"
    assert report.crs_problems == []
    assert "EPSG:2950" not in manifest.model_dump_json()


# --- découverte : aucune requête, et aucun téléchargement ---------------------


def test_discover_is_unsupported_and_issues_no_query(second_site) -> None:
    runner.invoke(app, ["geo", "reference", HOTEL])

    result = runner.invoke(app, ["geo", "discover", HOTEL])

    # Faute de bâtiment confirmé ou d'adaptateur, la commande s'arrête — et
    # dans les deux cas sans avoir rien interrogé ni rien téléchargé.
    assert result.exit_code != 0
    assert "welcominns" not in result.output.lower()

    lidar = second_site / "work" / HOTEL / "06_geo" / "lidar_raw"
    assert not lidar.exists() or list(lidar.iterdir()) == []


def test_the_adapter_refuses_before_any_network_call(second_site) -> None:
    """Le défaut corrigé : le WFS québécois était appelé quoi qu'il arrive."""
    from hotel_pipeline.geo.adapters import elevation_adapter
    from hotel_pipeline.geo.catalog import route

    adapter, reasons = elevation_adapter(route(*LYON))

    assert adapter is None
    assert reasons
    assert any("territorialement admissible" in reason for reason in reasons)


# --- qualification : refusée, et pour les bonnes raisons ----------------------


def test_qualification_stops_on_named_missing_capabilities(second_site) -> None:
    runner.invoke(app, ["geo", "reference", HOTEL])

    result = runner.invoke(app, ["geo", "qualify", HOTEL])

    assert result.exit_code != 0
    output = result.output.lower()
    # L'erreur nomme ce qui manque, plutôt que d'échouer sur un fichier absent.
    assert "geospatial_qualification" in output or "manifeste" in output


# --- rien n'a été téléchargé --------------------------------------------------


def test_nothing_was_downloaded_anywhere(second_site) -> None:
    heavy = [
        path
        for path in (second_site / "work" / HOTEL).rglob("*")
        if path.is_file() and path.stat().st_size > 1_000_000
    ]

    assert heavy == []
