"""Câblage prouvé par la CLI (Lot 1B, audit du câblage).

Les tests unitaires montraient que les composants acceptent une politique.
Ils ne montraient pas que **le chemin d'exécution réel** la charge et l'écrit :
aucun rapport du corpus ne portait de provenance, et `stamp()` n'était appelé
que par des tests.

Ces cas modifient une politique sur disque, exécutent une commande, puis
relisent le JSON produit.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from hotel_pipeline.cli import app
from hotel_pipeline.schemas import DEFAULT_POLICY, Asset, AssetManifest, PipelinePolicy
from hotel_pipeline.schemas.spatial import BuildingCandidate, GeocodeResult, SpatialManifest
from hotel_pipeline.workspace import Workspace

runner = CliRunner()

BUILDING_WKT = (
    "POLYGON ((-73.44380 45.57355, -73.44280 45.57355, "
    "-73.44280 45.57445, -73.44380 45.57445, -73.44380 45.57355))"
)


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Un hôtel prêt, avec bâtiment confirmé et deux assets géolocalisés."""
    monkeypatch.setenv("HOTEL_PIPELINE_WORK", str(tmp_path / "work"))
    monkeypatch.setenv("HOTEL_PIPELINE_PROFILES", str(tmp_path / "profiles"))
    monkeypatch.chdir(tmp_path)

    hotel_id = "hotel-test"
    runner.invoke(app, ["init", hotel_id, "--address", "1 rue Test"])

    workspace = Workspace(hotel_id)
    candidate = BuildingCandidate(
        feature_id="way/1", source="overpass", centroid_lat=45.5741, centroid_lon=-73.4433,
        area_m2=1800, distance_to_geocode_m=5, wkt=BUILDING_WKT,
    )
    workspace.write_spatial(
        SpatialManifest(
            hotel_id=hotel_id, address="1 rue Test",
            geocode=GeocodeResult(lat=45.5741, lon=-73.4433, provider="test"),
            candidates=[candidate], confirmed_building_id="way/1", confirmed_by="test",
        )
    )

    def asset(asset_id, lon):
        return Asset(
            id=asset_id, source="mapillary", source_url_or_id="u",
            rights="open_data", ai_eligible=False, confidence=0.5,
            category="other", checksum=asset_id * 8,
            camera_lat=45.5730, camera_lon=lon, heading_deg=0.0,
            target_distance_m=40.0,
        )

    workspace.write_assets(
        AssetManifest(hotel_id=hotel_id, assets=[asset("a", -73.44330), asset("b", -73.44340)])
    )
    return hotel_id


def write_policy(tmp_path, **dedup):
    policy = DEFAULT_POLICY.model_copy(deep=True)
    policy.version = "test-9.9.9"
    policy.model.calibration_id = "calibration-de-test"
    for key, value in dedup.items():
        setattr(policy.dedup, key, value)
    (tmp_path / "pipeline_policy.json").write_text(policy.model_dump_json(), encoding="utf-8")
    return policy


class TestProvenanceReachesDisk:
    def test_report_carries_the_policy_version_from_disk(self, project, tmp_path):
        write_policy(tmp_path)
        result = runner.invoke(app, ["assets", "dedup", project])
        assert result.exit_code == 0, result.stdout

        report = json.loads(
            (Workspace(project).path("01_sources", "duplicate_report.json")).read_text("utf-8")
        )
        assert report["provenance"]["policy_version"] == "test-9.9.9"
        assert report["provenance"]["calibration_id"] == "calibration-de-test"

    def test_report_carries_a_policy_digest(self, project, tmp_path):
        write_policy(tmp_path)
        runner.invoke(app, ["assets", "dedup", project])
        report = json.loads(
            (Workspace(project).path("01_sources", "duplicate_report.json")).read_text("utf-8")
        )
        assert len(report["provenance"]["policy_digest"]) == 16

    def test_report_body_is_preserved_alongside_provenance(self, project, tmp_path):
        write_policy(tmp_path)
        runner.invoke(app, ["assets", "dedup", project])
        report = json.loads(
            (Workspace(project).path("01_sources", "duplicate_report.json")).read_text("utf-8")
        )
        assert report["files"] == 2


class TestPolicyOnDiskChangesTheOutcome:
    """La preuve que la politique n'est pas décorative."""

    def test_loose_tolerance_merges_the_two_viewpoints(self, project, tmp_path):
        write_policy(tmp_path, position_tolerance_m=50.0)
        runner.invoke(app, ["assets", "dedup", project])
        report = json.loads(
            (Workspace(project).path("01_sources", "duplicate_report.json")).read_text("utf-8")
        )
        assert report["independent_viewpoints"] == 1

    def test_tight_tolerance_separates_them(self, project, tmp_path):
        write_policy(tmp_path, position_tolerance_m=1.0)
        runner.invoke(app, ["assets", "dedup", project])
        report = json.loads(
            (Workspace(project).path("01_sources", "duplicate_report.json")).read_text("utf-8")
        )
        assert report["independent_viewpoints"] == 2


class TestProfileLoading:
    def test_missing_profile_is_reported_not_silent(self, project, tmp_path):
        """Tourner sur des valeurs de secours sans le dire était le défaut."""
        write_policy(tmp_path)
        result = runner.invoke(app, ["assets", "dedup", project])
        assert "profil introuvable" in result.stdout

    def test_present_profile_is_stamped_on_the_report(self, project, tmp_path):
        write_policy(tmp_path)
        profiles = tmp_path / "profiles"
        profiles.mkdir(exist_ok=True)
        (profiles / f"{project}.json").write_text(
            json.dumps(
                {
                    "property_id": project,
                    "address": "1 rue Test",
                    "official_name": "Hôtel Test",
                    "room_count": 116,
                    "expected_levels": 3,
                }
            ),
            encoding="utf-8",
        )
        runner.invoke(app, ["assets", "dedup", project])
        report = json.loads(
            (Workspace(project).path("01_sources", "duplicate_report.json")).read_text("utf-8")
        )
        assert report["provenance"]["property_profile_id"] == project
        assert "property_profile_digest" in report["provenance"]


class TestPolicyFileIsOptional:
    def test_default_policy_applies_without_a_file(self, project):
        result = runner.invoke(app, ["assets", "dedup", project])
        assert result.exit_code == 0
        report = json.loads(
            (Workspace(project).path("01_sources", "duplicate_report.json")).read_text("utf-8")
        )
        assert report["provenance"]["policy_version"] == PipelinePolicy().version
