"""Arborescence, idempotence et écriture atomique (plan directeur §18)."""

from __future__ import annotations

import pytest

from hotel_pipeline.schemas import ProjectManifest, StepRecord
from hotel_pipeline.workspace import SUBDIRS, Workspace


@pytest.fixture
def workspace(tmp_path) -> Workspace:
    ws = Workspace("welcominns-boucherville", root=tmp_path)
    ws.create()
    return ws


class TestTree:
    def test_all_subdirs_created(self, workspace):
        for subdir in SUBDIRS:
            assert workspace.path(subdir).is_dir(), subdir

    def test_rights_separation_is_structural(self, workspace):
        """Les deux espaces du §9 sont séparés sur le disque, pas par convention."""
        assert workspace.path("02_images/reference_only").is_dir()
        assert workspace.path("02_images/production_eligible").is_dir()

    def test_create_is_idempotent(self, workspace):
        workspace.path("05_colmap", "keep.txt").write_text("x")
        workspace.create()
        assert workspace.path("05_colmap", "keep.txt").read_text() == "x"


class TestManifest:
    def test_roundtrip(self, workspace):
        workspace.write_manifest(
            ProjectManifest(hotel_id="welcominns-boucherville", address="1195 rue Ampère")
        )
        assert workspace.read_manifest().address == "1195 rue Ampère"

    def test_missing_manifest_gives_actionable_error(self, tmp_path):
        ws = Workspace("absent", root=tmp_path)
        with pytest.raises(FileNotFoundError, match="hotel-pipeline init"):
            ws.read_manifest()

    def test_no_temp_file_left_behind(self, workspace):
        """L'écriture atomique ne doit pas laisser de .tmp derrière elle."""
        workspace.write_manifest(ProjectManifest(hotel_id="h", address="a"))
        assert list(workspace.path("00_manifest").glob("*.tmp")) == []

    def test_text_reports_use_the_same_atomic_writer(self, workspace):
        path = workspace.write_text("coverage/report.md", "preuve\n")

        assert path.read_text() == "preuve\n"
        assert list(path.parent.glob("*.tmp")) == []


class TestStepRecording:
    def test_completed_steps_tracked(self):
        manifest = ProjectManifest(hotel_id="h", address="a")
        assert manifest.completed_steps() == set()
        manifest.record(StepRecord(name="collect"))
        assert manifest.completed_steps() == {"collect"}

    def test_rerunning_a_step_replaces_its_record(self):
        """Rejouer une étape ne doit pas empiler des traces en double."""
        manifest = ProjectManifest(hotel_id="h", address="a")
        manifest.record(StepRecord(name="collect", parameters={"v": "1"}))
        manifest.record(StepRecord(name="collect", parameters={"v": "2"}))
        assert len(manifest.steps) == 1
        assert manifest.steps[0].parameters == {"v": "2"}
