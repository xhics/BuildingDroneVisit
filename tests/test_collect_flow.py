"""Flux `collect` complet et ses verrous humains (complément §4).

Aucun appel réseau : le manifeste spatial est pré-alimenté depuis le corpus de
test, exactement comme il le serait après une résolution réelle.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from hotel_pipeline.cli import app
from hotel_pipeline.resolve import build_candidates
from hotel_pipeline.schemas.spatial import GeocodeResult, SpatialManifest
from hotel_pipeline.steps import ELEMENTS_FILE
from hotel_pipeline.workspace import Workspace

runner = CliRunner()

TRUE_BUILDING = "way/1001"
NEIGHBOUR_HOTEL = "way/1002"

EXIT_BLOCKED = 3
EXIT_NOT_IMPLEMENTED = 2


@pytest.fixture
def hotel(tmp_path, monkeypatch, overpass_elements):
    """Un hôtel initialisé dont la résolution spatiale est déjà faite."""
    monkeypatch.setenv("HOTEL_PIPELINE_WORK", str(tmp_path / "work"))
    hotel_id = "welcominns-boucherville"

    assert runner.invoke(app, ["init", hotel_id, "--name", "Hôtel Test", "--country", "CA", "--timezone", "America/Toronto", "--ocr-language", "fr", "--address", "1195 rue Ampère"]).exit_code == 0

    geocode = GeocodeResult(lat=45.5896, lon=-73.4372, provider="fixture")
    workspace = Workspace(hotel_id)
    workspace.write_spatial(
        SpatialManifest(
            hotel_id=hotel_id,
            address="1195 rue Ampère",
            geocode=geocode,
            candidates=build_candidates(overpass_elements, geocode),
        )
    )
    workspace.write_json(ELEMENTS_FILE, overpass_elements)
    return hotel_id


def csv_at(tmp_path, body: str):
    path = tmp_path / "inv.csv"
    path.write_text(
        "id,source,rights,category,exterior_or_interior,entrance_version\n" + body,
        encoding="utf-8",
    )
    return str(path)


class TestBuildingLock:
    def test_collect_blocks_on_building_confirmation(self, hotel):
        result = runner.invoke(app, ["collect", hotel])
        assert result.exit_code == EXIT_BLOCKED
        assert "BUILDING_MAIN" in result.stdout

    def test_block_is_persisted_and_visible_in_status(self, hotel):
        runner.invoke(app, ["collect", hotel])
        result = runner.invoke(app, ["status", hotel])
        assert "BLOQUÉ" in result.stdout
        assert "confirm-building" in result.stdout

    def test_candidates_are_listed_for_the_human(self, hotel):
        result = runner.invoke(app, ["candidates", hotel])
        assert result.exit_code == 0
        assert TRUE_BUILDING in result.stdout
        assert NEIGHBOUR_HOTEL in result.stdout

    def test_confirming_an_unknown_feature_is_refused(self, hotel):
        result = runner.invoke(
            app,
            ["confirm-building", hotel, "way/999", "--by", "hm", "--rationale", "x"],
        )
        assert result.exit_code == 1

    def test_confirmation_is_persisted_once(self, hotel):
        runner.invoke(
            app,
            [
                "confirm-building", hotel, TRUE_BUILDING,
                "--by", "hm", "--rationale", "aérien + orthophoto",
            ],
        )
        spatial = Workspace(hotel).read_spatial()
        assert spatial.confirmed_building_id == TRUE_BUILDING
        assert spatial.confirmed_by == "hm"
        assert spatial.confirmation_rationale == "aérien + orthophoto"

        # Le verrou ne doit pas être redemandé à l'exécution suivante.
        result = runner.invoke(app, ["collect", hotel])
        assert "BUILDING_MAIN" not in result.stdout


class TestSeparationsGate:
    def test_wrong_building_blocks_on_separations(self, hotel):
        runner.invoke(
            app,
            ["confirm-building", hotel, NEIGHBOUR_HOTEL, "--by", "hm", "--rationale", "erreur"],
        )
        result = runner.invoke(app, ["collect", hotel])
        assert result.exit_code == EXIT_BLOCKED
        assert "parking_adjacent_to_building" in result.stdout

    def test_correct_building_reaches_the_media_step(self, hotel):
        """La collecte est automatique : hors ligne, elle rend un corpus vide.

        Le blocage porte alors sur l'absence d'asset exploitable, et non sur un
        inventaire manuel à fournir — c'est ce qui permet à « une adresse, une
        commande » de tenir (§1).
        """
        runner.invoke(
            app, ["confirm-building", hotel, TRUE_BUILDING, "--by", "hm", "--rationale", "ok"]
        )
        result = runner.invoke(app, ["collect", hotel])
        assert result.exit_code == EXIT_BLOCKED
        assert "aucun asset éligible production" in result.stdout


class TestMediaLocks:
    @pytest.fixture
    def confirmed(self, hotel):
        runner.invoke(
            app, ["confirm-building", hotel, TRUE_BUILDING, "--by", "hm", "--rationale", "ok"]
        )
        return hotel

    def test_blocks_when_nothing_is_production_eligible(self, confirmed, tmp_path):
        path = csv_at(tmp_path, "img-1,tripadvisor,public_uncleared,facade,exterior,unknown\n")
        runner.invoke(app, ["assets", "import", confirmed, path])

        result = runner.invoke(app, ["collect", confirmed])
        assert result.exit_code == EXIT_BLOCKED
        assert "aucun asset éligible production" in result.stdout

    def test_undated_geometry_is_not_blocking(self, confirmed, tmp_path):
        """Une entrée non datée ne doit pas interdire toute la géométrie.

        Le blocage portait auparavant sur `entrance_version` de tout extérieur
        éligible, confondant un problème d'apparence avec un problème de
        structure.
        """
        path = csv_at(tmp_path, "img-1,hotel,owned,facade,exterior,unknown\n")
        runner.invoke(app, ["assets", "import", confirmed, path])
        runner.invoke(app, ["assets", "promote", confirmed, "img-1"])

        result = runner.invoke(app, ["collect", confirmed])
        assert result.exit_code == 0, result.stdout

    def test_collect_completes_once_every_lock_is_released(self, confirmed, tmp_path):
        path = csv_at(tmp_path, "img-1,hotel,owned,facade,exterior,after_renovation\n")
        runner.invoke(app, ["assets", "import", confirmed, path])
        runner.invoke(app, ["assets", "promote", confirmed, "img-1"])

        result = runner.invoke(app, ["collect", confirmed])
        assert result.exit_code == 0, result.stdout

        manifest = Workspace(confirmed).read_manifest()
        assert "collect" in manifest.completed_steps()
        assert manifest.blocked is None

    def test_temporal_decision_is_recorded_with_its_author(self, confirmed, tmp_path):
        path = csv_at(tmp_path, "img-1,hotel,owned,facade,exterior,unknown\n")
        runner.invoke(app, ["assets", "import", confirmed, path])
        result = runner.invoke(
            app,
            ["temporal", "set", confirmed, "img-1", "entrance", "current_confirmed",
             "--by", "hm", "--rationale", "photo fournie par l'hôtel après travaux"],
        )
        assert result.exit_code == 0, result.stdout

        asset = Workspace(confirmed).read_assets().assets[0]
        assert asset.temporal_decisions[0].decided_by == "hm"
        assert asset.temporal_decisions[0].scope == "entrance"
        assert asset.temporal_by_scope["entrance"].value == "current_confirmed"

    def test_phase1_then_stops_on_lot1b(self, confirmed, tmp_path):
        """Une fois collect franchi, l'arrêt suivant est le Lot 1B.

        Sur un corpus minimal, le registre des sources n'a aucun manifeste
        canonique de candidats à lire : c'est un arrêt documenté, pas une
        trace brute, et il précède l'étape non construite.
        """
        path = csv_at(tmp_path, "img-1,hotel,owned,facade,exterior,after_renovation\n")
        runner.invoke(app, ["assets", "import", confirmed, path])
        runner.invoke(app, ["assets", "promote", confirmed, "img-1"])

        result = runner.invoke(app, ["run-phase1", confirmed])
        assert result.exit_code == EXIT_BLOCKED
        assert "corpus incomplet" in result.stdout

    def test_preflight_blocks_on_incomplete_lot1b(self, confirmed):
        """preflight est construit, mais bloque si le Lot 1B est incomplet."""
        result = runner.invoke(app, ["preflight", confirmed])
        assert result.exit_code == EXIT_BLOCKED
        assert "preflight" in result.stdout
        assert "Lot 1B" in result.stdout


class TestOfflineGuard:
    def test_network_calls_are_refused_in_tests(self, tmp_path, monkeypatch):
        """Garde-fou : aucun test ne peut appeler le réseau en silence (§17)."""
        from hotel_pipeline.providers.cache import OfflineError
        from hotel_pipeline.providers.geocode import geocode

        with pytest.raises(OfflineError):
            geocode("1195 rue Ampère, Boucherville")


class TestProvidedCoordinates:
    def test_lat_without_lon_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOTEL_PIPELINE_WORK", str(tmp_path))
        result = runner.invoke(app, ["init", "h", "--name", "Hôtel Test", "--country", "CA", "--timezone", "America/Toronto", "--ocr-language", "fr", "--address", "a", "--lat", "45.57"])
        assert result.exit_code == 1

    def test_coordinates_are_persisted(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOTEL_PIPELINE_WORK", str(tmp_path))
        result = runner.invoke(
            app,
            ["init", "h", "--name", "Hôtel Test", "--country", "CA", "--timezone", "America/Toronto", "--ocr-language", "fr", "--address", "a", "--lat", "45.574128", "--lon", "-73.443289"],
        )
        assert result.exit_code == 0
        assert "géocodage court-circuité" in result.stdout

        manifest = Workspace("h").read_manifest()
        assert manifest.lat == pytest.approx(45.574128)
        assert manifest.lon == pytest.approx(-73.443289)
