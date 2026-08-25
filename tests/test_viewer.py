from __future__ import annotations

import hashlib
import json
from pathlib import Path

from typer.testing import CliRunner

from hotel_pipeline.cli import app
from hotel_pipeline.demo import assess
from hotel_pipeline.scene_package import _has_reconstruction
from hotel_pipeline.viewer import build
from hotel_pipeline.workspace import Workspace


def _workspace(tmp_path: Path, hotel_id: str = "hotel-test") -> Workspace:
    workspace = Workspace(hotel_id, root=tmp_path)
    workspace.create()
    return workspace


def test_viewer_migre_le_payload_historique_et_devient_reproductible(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    legacy = workspace.path("11_conditioning", "viewer.html")
    legacy.parent.mkdir(parents=True)
    payload = {
        "hotel": workspace.hotel_id,
        "volumes": [{"fp": [[0, 0], [10, 0], [10, 10]], "h": 8, "target": True}],
        "vegetation": [],
        "furniture": [],
        "ground": [],
        "ridges": [],
        "observation": {"cells": [], "missing": []},
        "counts": {"volumes": 1},
    }
    legacy.write_text(
        f"<script>const PAYLOAD = {json.dumps(payload)};\n\nconst cv = 1;</script>",
        encoding="utf-8",
    )

    outputs = build(workspace)
    stored = json.loads(outputs.payload.read_text("utf-8"))
    manifest = json.loads(outputs.manifest.read_text("utf-8"))

    assert stored == payload
    assert "MODE DÉMONSTRATION" in outputs.html.read_text("utf-8")
    assert "facadeHeight(o.c)-.12" in outputs.html.read_text("utf-8")
    assert "obs:false,plan:false,ridge:false" in outputs.html.read_text("utf-8")
    assert "Masques IA 2D" in outputs.html.read_text("utf-8")
    assert "Instances IA multi-vues" in outputs.html.read_text("utf-8")
    assert "Pistes SfM partagées" in outputs.html.read_text("utf-8")
    assert "Alignement COLMAP/LiDAR" in outputs.html.read_text("utf-8")
    assert "Points SfM recalés" in outputs.html.read_text("utf-8")
    assert "Hypothèses linéaires mono-vue" in outputs.html.read_text("utf-8")
    assert "supports IA/SfM" in outputs.html.read_text("utf-8")
    assert "1</kbd> bâtiment" in outputs.html.read_text("utf-8")
    assert "CAMERA.focus||TARGET.focus" in outputs.html.read_text("utf-8")
    assert "focus[0]+Math.cos(az)" in outputs.html.read_text("utf-8")
    assert "ui-hidden" in outputs.html.read_text("utf-8")
    assert "Surfaces 3D contraintes" in outputs.html.read_text("utf-8")
    assert "f.k==='veg'?.28:f.k==='pole'?.58:1" in outputs.html.read_text("utf-8")
    assert "show={roof:true,vol:true,veg:false" in outputs.html.read_text("utf-8")
    assert "Ressemblance structurelle" in outputs.html.read_text("utf-8")
    assert "FAÇADE" in outputs.html.read_text("utf-8")
    assert "Données techniques" in outputs.html.read_text("utf-8")
    assert "sitePlane()" in outputs.html.read_text("utf-8")
    assert ":.82" not in outputs.html.read_text("utf-8")
    assert manifest["mode"] == "demo"
    assert manifest["payload_current"] is True
    assert manifest["formal_phase1_status"] == "not_overridden"
    assert manifest["payload"]["sha256"] == hashlib.sha256(
        outputs.payload.read_bytes()
    ).hexdigest()
    facade_audit = workspace.path(
        "11_conditioning", "facade_similarity_audit.json"
    )
    assert facade_audit.is_file()
    assert manifest["facade_similarity"]["photometric_claim"] is False
    assert manifest["facade_similarity"]["sha256"] == hashlib.sha256(
        facade_audit.read_bytes()
    ).hexdigest()


def test_viewer_peut_partir_du_obj_canonique(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    package = workspace.path("08_composite", "scene_package_abc")
    package.mkdir(parents=True)
    (package / "environment.obj").write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8"
    )
    (package / "scene.json").write_text(
        json.dumps({"phase1_verdict": "phase1_verdict.json"}), encoding="utf-8"
    )
    workspace.write_json(
        "08_composite/scene_package_current.json",
        {"manifest": "08_composite/scene_package_abc/scene.json"},
    )

    outputs = build(workspace)
    payload = json.loads(outputs.payload.read_text("utf-8"))
    assert len(payload["mesh"]["vertices"]) == 3
    assert payload["mesh"]["faces"] == [[0, 1, 2]]


def test_viewer_affiche_un_solveur_gpu_refuse_sans_injecter_sa_geometrie(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    legacy = workspace.path("11_conditioning", "viewer.html")
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        '<script>const PAYLOAD = {"hotel":"hotel-test"};\n\nconst cv = 1;</script>',
        "utf-8",
    )
    workspace.write_json(
        "11_conditioning/feed_forward_shape_audit.json",
        {"status": "rejected", "viewer_integration": False},
    )

    outputs = build(workspace)
    payload = json.loads(outputs.payload.read_text("utf-8"))
    html = outputs.html.read_text("utf-8")

    assert payload["feed_forward_shape"]["status"] == "rejected"
    assert "Forme GPU multi-vues" in html
    assert "refusée (2/2)" in html
    assert "nuages fragmentés" in html
    assert "max-height:calc(100vh - 28px)" in html
    assert "#legend{right:14px;top:14px" in html
    assert "Orthofaçades photographiques sur masse LiDAR" in html


def test_viewer_signale_un_payload_devenu_perime(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    legacy = workspace.path("11_conditioning", "viewer.html")
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        '<script>const PAYLOAD = {"hotel":"hotel-test"};\n\nconst cv = 1;</script>',
        "utf-8",
    )
    source = workspace.path("06_geo", "observation_map.json")
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("{}", "utf-8")
    build(workspace)

    source.write_text('{"changed":true}', "utf-8")
    outputs = build(workspace)
    manifest = json.loads(outputs.manifest.read_text("utf-8"))
    assert manifest["payload_current"] is False
    assert manifest["demo_readiness"] == "stale_payload"


def test_viewer_ignore_une_republication_formelle_sans_changement_geometrique(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    legacy = workspace.path("11_conditioning", "viewer.html")
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        '<script>const PAYLOAD = {"hotel":"hotel-test"};\n\nconst cv = 1;</script>',
        "utf-8",
    )
    pointer = workspace.path("08_composite", "scene_package_current.json")
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text('{"manifest":"first"}', "utf-8")
    build(workspace)

    pointer.write_text('{"manifest":"second"}', "utf-8")
    outputs = build(workspace)
    manifest = json.loads(outputs.manifest.read_text("utf-8"))
    meta = json.loads(
        workspace.path(
            "11_conditioning", "viewer_payload_meta.json"
        ).read_text("utf-8")
    )
    assert manifest["payload_current"] is True
    assert meta["contract_version"] == 3
    assert meta["source_scope"] == "geometry-v7"


def test_un_run_synthetique_ne_passe_jamais_g5(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    workspace.write_json(
        "07_reconstruction/runs/run-synthetic.json",
        {"backend": "synthetic", "status": "completed"},
    )
    passed, evidence = _has_reconstruction(workspace)
    assert passed is False
    assert evidence["synthetic_runs"] == 1
    assert evidence["completed_real_runs"] == 0


def test_decision_g5_negative_est_un_diagnostic_pas_un_succes(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    workspace.write_json(
        "05_colmap/experiment/decision.json",
        {
            "g5_passed": False,
            "geometry_validation": {
                "g5": {
                    "passed": False,
                    "validated_registration_rate": 0.25,
                    "required_registration_rate": 0.60,
                    "refusal_reasons": ["parallaxe insuffisante"],
                }
            },
        },
    )
    passed, evidence = _has_reconstruction(workspace)
    assert passed is False
    assert evidence["type"] == "validated_diagnostic"
    assert evidence["validated_registration_rate"] == 0.25


def test_statut_demo_reste_distinct_du_verdict_formel(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    workspace.path("11_conditioning").mkdir(exist_ok=True)
    workspace.path("11_conditioning", "viewer.html").write_text("ok", "utf-8")
    workspace.write_json(
        "11_conditioning/viewer_manifest.json",
        {"source_digests": {"scene": "abc"}, "payload_current": True},
    )
    workspace.write_json(
        "11_conditioning/orbit/conditioning_report.json",
        {"frame_count": 12, "verdict": "condition_partially"},
    )
    workspace.write_json("09_confidence/fidelity_audit.json", {"score": 0.9})
    package = workspace.path("08_composite", "scene_package_x")
    package.mkdir(parents=True)
    (package / "scene.json").write_text(
        json.dumps({"phase1_verdict": "phase1_verdict.json"}), "utf-8"
    )
    (package / "phase1_verdict.json").write_text(
        json.dumps({"status": "NEEDS_AUTHORIZED_CAPTURE"}), "utf-8"
    )
    workspace.write_json(
        "08_composite/scene_package_current.json",
        {"manifest": "08_composite/scene_package_x/scene.json"},
    )

    report = assess(workspace)
    assert report["status"] == "DEMO_READY"
    assert report["formal_phase1_status"] == "NEEDS_AUTHORIZED_CAPTURE"
    assert report["formal_phase1_overridden"] is False


def test_cli_expose_viewer_et_demo(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOTEL_PIPELINE_WORK", str(tmp_path))
    runner = CliRunner()
    assert runner.invoke(app, ["viewer", "--help"]).exit_code == 0
    assert runner.invoke(app, ["demo", "--help"]).exit_code == 0
