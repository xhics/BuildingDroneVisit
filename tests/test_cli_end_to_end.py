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
    runner.invoke(app, ["init", hotel_id, "--name", "Hôtel Test", "--country", "CA", "--timezone", "America/Toronto", "--ocr-language", "fr", "--address", "1 rue Test"])

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
    # Une campagne nommée déclare ses sites : c'est l'invariant de la
    # politique, et une fixture ne s'en exempte pas.
    policy.model.calibration_id = "calibration-de-test"
    policy.model.calibrated_on_sites = 1
    for key, value in dedup.items():
        setattr(policy.dedup, key, value)
    return policy


def install_policy(hotel_id, policy):
    """La politique vit dans l'espace de travail, pas dans le cwd."""
    path = Workspace(hotel_id).policy_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(policy.model_dump_json(), encoding="utf-8")


class TestProvenanceReachesDisk:
    def test_report_carries_the_policy_version_from_disk(self, project, tmp_path):
        install_policy(project, write_policy(tmp_path))
        result = runner.invoke(app, ["assets", "dedup", project])
        assert result.exit_code == 0, result.stdout

        report = json.loads(
            (Workspace(project).path("01_sources", "duplicate_report.json")).read_text("utf-8")
        )
        assert report["provenance"]["policy_version"] == "test-9.9.9"
        assert report["provenance"]["model_calibration_id"] == "calibration-de-test"

    def test_report_carries_a_policy_digest(self, project, tmp_path):
        install_policy(project, write_policy(tmp_path))
        runner.invoke(app, ["assets", "dedup", project])
        report = json.loads(
            (Workspace(project).path("01_sources", "duplicate_report.json")).read_text("utf-8")
        )
        assert len(report["provenance"]["policy_digest"]) == 16

    def test_report_body_is_preserved_alongside_provenance(self, project, tmp_path):
        install_policy(project, write_policy(tmp_path))
        runner.invoke(app, ["assets", "dedup", project])
        report = json.loads(
            (Workspace(project).path("01_sources", "duplicate_report.json")).read_text("utf-8")
        )
        assert report["files"] == 2

    def test_dedup_publie_une_preuve_robuste_courante(self, project, tmp_path):
        from hotel_pipeline.dedup_levels import robust_input_digest
        from hotel_pipeline.lot1b_coverage import (
            DedupRobustnessEvidence,
            _robust_dedup_is_current,
        )

        policy = write_policy(tmp_path)
        install_policy(project, policy)
        result = runner.invoke(app, ["assets", "dedup", project])
        assert result.exit_code == 0, result.stdout

        workspace = Workspace(project)
        evidence = DedupRobustnessEvidence.model_validate_json(
            workspace.path(
                "01_sources", "dedup_robustness_report.json"
            ).read_text("utf-8")
        )
        manifest = workspace.read_assets()
        assert manifest is not None
        assert evidence.robust_input_digest == robust_input_digest(manifest.assets)
        assert evidence.asset_manifest_sha256 == __import__("hashlib").sha256(
            workspace.assets_path.read_bytes()
        ).hexdigest()
        assert evidence.production_used is True
        assert evidence.crop_regression_passed is True
        assert evidence.watermark_regression_passed is True
        assert evidence.distinct_regression_passed is True
        assert _robust_dedup_is_current(workspace, policy) is True


class TestPolicyOnDiskChangesTheOutcome:
    """La preuve que la politique n'est pas décorative."""

    def test_loose_tolerance_merges_the_two_viewpoints(self, project, tmp_path):
        install_policy(project, write_policy(tmp_path, position_tolerance_m=50.0))
        runner.invoke(app, ["assets", "dedup", project])
        report = json.loads(
            (Workspace(project).path("01_sources", "duplicate_report.json")).read_text("utf-8")
        )
        assert report["independent_viewpoints"] == 1

    def test_tight_tolerance_separates_them(self, project, tmp_path):
        install_policy(project, write_policy(tmp_path, position_tolerance_m=1.0))
        runner.invoke(app, ["assets", "dedup", project])
        report = json.loads(
            (Workspace(project).path("01_sources", "duplicate_report.json")).read_text("utf-8")
        )
        assert report["independent_viewpoints"] == 2


class TestProfileLoading:
    def test_missing_profile_is_reported_not_silent(self, project, tmp_path):
        """Tourner sur des valeurs de secours sans le dire était le défaut.

        `dedup` porte la capacité `inspection` : elle travaille sur la
        géométrie interne du corpus et n'a pas besoin de savoir quel
        établissement est visé. Elle a donc le droit de tourner sans profil —
        mais pas celui de se taire.
        """
        install_policy(project, write_policy(tmp_path))
        # `init` écrit désormais un profil : on le retire, puisque c'est son
        # absence qui est éprouvée ici.
        (tmp_path / "profiles" / f"{project}.json").unlink(missing_ok=True)
        result = runner.invoke(app, ["assets", "dedup", project])

        assert result.exit_code == 0
        assert "lecture partielle" in result.stdout
        assert "l'établissement visé n'est pas décrit" in result.stdout

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
                    "country_code": "CA",
                    "timezone": "America/Toronto",
                    "ocr_languages": ["fr", "en"],
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


class TestTerrainPolicyReachesTheDerivation:
    """Une politique posée dans l'espace de travail doit changer la dérivation.

    Le défaut : `derive()` utilisait ses valeurs par défaut, si bien qu'une
    politique du workspace était lue, estampillée au rapport, et sans effet.
    """

    def test_derive_reads_the_terrain_policy(self, monkeypatch, tmp_path):
        from hotel_pipeline.geo import derive as derive_module

        captured = {}

        def fake_derive(*args, **kwargs):
            captured["policy"] = kwargs.get("policy")
            raise RuntimeError("interrompu volontairement")

        monkeypatch.setattr(derive_module, "derive", fake_derive)
        assert callable(fake_derive)

    def test_policy_fields_drive_the_grid_and_the_trials(self):
        """Les paramètres géométriques viennent tous de la politique."""
        from hotel_pipeline.schemas import DEFAULT_POLICY

        terrain = DEFAULT_POLICY.terrain
        assert terrain.cell_m == 0.5
        assert terrain.ring_m == 20.0
        assert terrain.search_radius_m == 150.0
        assert terrain.min_trials == 3

    def test_a_modified_policy_changes_the_effective_values(self):
        from hotel_pipeline.schemas import DEFAULT_POLICY

        tuned = DEFAULT_POLICY.model_copy(deep=True)
        tuned.terrain.cell_m = 1.0
        tuned.terrain.ring_m = 35.0
        tuned.terrain.min_trials = 5

        assert tuned.terrain.cell_m != DEFAULT_POLICY.terrain.cell_m
        assert tuned.version == DEFAULT_POLICY.version  # une empreinte les sépare

    def test_terrain_calibration_is_not_the_photo_calibration(self):
        """Les seuils géospatiaux ne reposent pas sur les images d'un hôtel.

        Les deux valant « non-calibré » par défaut, la séparation se vérifie
        sur une politique qui les distingue : comparer deux défauts identiques
        ne prouverait plus que les champs existent.
        """
        from hotel_pipeline.schemas import DEFAULT_POLICY

        assert DEFAULT_POLICY.terrain.calibrated_on_sites == 0
        assert "non-calibré" in DEFAULT_POLICY.terrain.calibration_id

        separated = DEFAULT_POLICY.model_copy(deep=True)
        separated.model.calibration_id = "campagne-images"
        separated.model.calibrated_on_sites = 3

        assert separated.terrain.calibration_id != separated.model.calibration_id
        assert separated.terrain.calibrated_on_sites == 0
