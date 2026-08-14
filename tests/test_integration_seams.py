"""Raccords entre étapes — trois défauts qu'aucun test unitaire ne voyait.

Chacun se situe **entre** des composants corrects : un état relu trop tôt, deux
vocabulaires qui ne coïncident pas, et deux points d'entrée qui chargent des
profils différents.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from hotel_pipeline.cli import app
from hotel_pipeline.context import PipelineContext
from hotel_pipeline.schemas import (
    DEFAULT_POLICY,
    Asset,
    AssetCategory,
    AssetManifest,
    ProjectManifest,
    Rights,
    Subject,
    TemporalDecision,
    TemporalStatus,
)
from hotel_pipeline.temporal import (
    SCOPE_SUBJECTS,
    assess,
    subjects_for_scope,
    undetermined_sensitive_scopes,
)
from hotel_pipeline.workspace import Workspace

runner = CliRunner()


def make(asset_id="a", **overrides) -> Asset:
    fields = dict(
        id=asset_id, source="mapillary", source_url_or_id="u", rights=Rights.OWNED,
        ai_eligible=False, confidence=0.5, category=AssetCategory.OTHER, checksum="a" * 64,
    )
    fields.update(overrides)
    return Asset(**fields)


class TestStaleStateAfterAssess:
    """`assess()` remplace les instances : la liste éligible doit être relue."""

    def test_reassessment_updates_the_instances_in_place(self):
        assets = [
            make(
                subjects=[Subject.ENTRANCE],
                temporal_decisions=[
                    TemporalDecision(
                        scope="entrance", status=TemporalStatus.CURRENT_CONFIRMED,
                        decided_by="hm", rationale="preuve datée",
                    )
                ],
            )
        ]
        stale = assets[0]
        assess(assets, None)

        assert undetermined_sensitive_scopes(assets[0]) == []
        # L'ancienne instance ignore la dérivation : la relire est obligatoire.
        assert undetermined_sensitive_scopes(stale) == ["entrance"]

    def test_manifest_returns_the_refreshed_instances(self):
        manifest = AssetManifest(
            hotel_id="h",
            assets=[
                make(
                    production_eligible=True,
                    subjects=[Subject.ENTRANCE],
                    temporal_decisions=[
                        TemporalDecision(
                            scope="entrance", status=TemporalStatus.CURRENT_CONFIRMED,
                            decided_by="hm", rationale="r",
                        )
                    ],
                )
            ],
        )
        before = manifest.production_eligible()
        assess(manifest.assets, None)
        after = manifest.production_eligible()

        assert undetermined_sensitive_scopes(before[0]) == ["entrance"]
        assert undetermined_sensitive_scopes(after[0]) == []


class TestScopeToSubjectMapping:
    """`signage` ne correspond pas au sujet `sign` : sans table, jamais déclenché."""

    def test_signage_scope_maps_to_the_sign_subject(self):
        assert subjects_for_scope("signage") == ("sign",)

    def test_signage_is_detected_on_an_image_showing_a_sign(self):
        asset = make(
            subjects=[Subject.SIGN], temporal_by_scope={"signage": TemporalStatus.UNKNOWN}
        )
        assert "signage" in undetermined_sensitive_scopes(asset)

    def test_facade_scope_maps_to_the_building_subject(self):
        assert subjects_for_scope("facade") == ("building",)

    def test_unknown_scope_falls_back_to_its_own_name(self):
        assert subjects_for_scope("terrasse") == ("terrasse",)

    def test_every_sensitive_scope_has_a_mapping(self):
        for scope in DEFAULT_POLICY.temporal.sensitive_scopes:
            assert scope in SCOPE_SUBJECTS, scope

    def test_an_image_without_the_subject_is_still_not_blocking(self):
        asset = make(
            subjects=[Subject.ROAD], temporal_by_scope={"signage": TemporalStatus.UNKNOWN}
        )
        assert undetermined_sensitive_scopes(asset) == []


class TestSingleProfilePerProject:
    """Commandes autonomes et run-phase1 doivent charger le même profil."""

    @pytest.fixture
    def project(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOTEL_PIPELINE_WORK", str(tmp_path / "work"))
        monkeypatch.setenv("HOTEL_PIPELINE_PROFILES", str(tmp_path / "profiles"))
        profiles = tmp_path / "profiles"
        profiles.mkdir()
        for name in ("hotel-test", "profil-declare"):
            (profiles / f"{name}.json").write_text(
                json.dumps(
                    {
                        "property_id": name,
                        "address": "1 rue Test",
                        "official_name": f"Hôtel {name}",
                        "country_code": "CA",
                        "timezone": "America/Toronto",
                        "ocr_languages": ["fr"],
                    }
                ),
                encoding="utf-8",
            )
        return tmp_path

    def test_declared_profile_wins_over_the_hotel_id(self, project):
        workspace = Workspace("hotel-test")
        workspace.create()
        workspace.write_manifest(
            ProjectManifest(
                hotel_id="hotel-test", address="a", property_profile_id="profil-declare"
            )
        )

        context, warning = PipelineContext.for_workspace(workspace)
        assert warning is None
        assert context.profile.property_id == "profil-declare"

    def test_hotel_id_is_used_when_no_profile_is_declared(self, project):
        workspace = Workspace("hotel-test")
        workspace.create()
        workspace.write_manifest(ProjectManifest(hotel_id="hotel-test", address="a"))

        context, _ = PipelineContext.for_workspace(workspace)
        assert context.profile.property_id == "hotel-test"

    def test_missing_manifest_falls_back_to_the_hotel_id(self, project):
        workspace = Workspace("hotel-test")
        workspace.create()
        context, _ = PipelineContext.for_workspace(workspace)
        assert context.profile.property_id == "hotel-test"

    def test_cli_command_uses_the_declared_profile(self, project):
        """Le raccord vérifié depuis la CLI, pas seulement depuis le contexte."""
        workspace = Workspace("hotel-test")
        workspace.create()
        workspace.write_manifest(
            ProjectManifest(
                hotel_id="hotel-test", address="a", property_profile_id="profil-declare"
            )
        )
        workspace.write_assets(AssetManifest(hotel_id="hotel-test", assets=[make()]))

        result = runner.invoke(app, ["temporal", "assess", "hotel-test"])
        assert result.exit_code == 0, result.stdout

        report = json.loads(
            workspace.path("01_sources", "temporal_report.json").read_text("utf-8")
        )
        assert report["provenance"]["property_profile_id"] == "profil-declare"
