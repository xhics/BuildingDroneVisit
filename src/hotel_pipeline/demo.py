"""Préparation et diagnostic d'une démonstration locale.

La démonstration est une portée produit distincte de la Phase 1. Elle peut être
présentable avant la captation finale et la clearance des droits, mais elle ne
peut ni modifier les Gates ni promouvoir le paquet proxy en résultat accepté.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from . import scene_package, viewer
from .workspace import Workspace


def _json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text("utf-8"))


def _check(check_id: str, label: str, passed: bool, evidence: str) -> dict:
    return {
        "check_id": check_id,
        "label": label,
        "state": "passed" if passed else "missing",
        "evidence": evidence,
    }


def assess(workspace: Workspace) -> dict:
    """Mesure uniquement ce qui est utile à une présentation locale."""
    viewer_manifest_path = workspace.path(
        "11_conditioning", "viewer_manifest.json"
    )
    viewer_html = workspace.path("11_conditioning", "viewer.html")
    conditioning_path = workspace.path(
        "11_conditioning", "orbit", "conditioning_report.json"
    )
    fidelity_path = workspace.path("09_confidence", "fidelity_audit.json")
    scene_pointer_path = workspace.path(
        "08_composite", "scene_package_current.json"
    )

    viewer_manifest = _json(viewer_manifest_path)
    conditioning = _json(conditioning_path)
    fidelity = _json(fidelity_path)
    scene_pointer = _json(scene_pointer_path)

    formal_status = "unknown"
    if scene_pointer:
        scene_path = workspace.root / str(scene_pointer.get("manifest", ""))
        scene = _json(scene_path)
        if scene:
            verdict = _json(scene_path.parent / str(scene.get("phase1_verdict", "")))
            if verdict:
                formal_status = str(verdict.get("status", "unknown"))

    g5_passed, reconstruction = scene_package._has_reconstruction(workspace)
    checks = [
        _check(
            "viewer",
            "viewer autonome ouvrable",
            viewer_html.is_file() and viewer_manifest is not None,
            str(viewer_html),
        ),
        _check(
            "scene",
            "paquet de scène canonique",
            scene_pointer is not None,
            str(scene_pointer_path),
        ),
        _check(
            "conditioning",
            "conditionnement orbital calculé",
            conditioning is not None and int(conditioning.get("frame_count", 0)) > 0,
            (
                f"{conditioning.get('frame_count')} frames, "
                f"verdict {conditioning.get('verdict')}"
                if conditioning
                else str(conditioning_path)
            ),
        ),
        _check(
            "fidelity",
            "audit de fidélité disponible",
            fidelity is not None and isinstance(fidelity.get("score"), (int, float)),
            (
                f"score {float(fidelity['score']):.4f}"
                if fidelity and isinstance(fidelity.get("score"), (int, float))
                else str(fidelity_path)
            ),
        ),
        _check(
            "provenance",
            "provenance du viewer",
            bool(
                viewer_manifest
                and viewer_manifest.get("source_digests")
                and viewer_manifest.get("payload_current") is True
            ),
            str(viewer_manifest_path),
        ),
    ]
    ready = all(item["state"] == "passed" for item in checks)
    return {
        "contract_version": 1,
        "hotel_id": workspace.hotel_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "DEMO_READY" if ready else "DEMO_INCOMPLETE",
        "formal_phase1_status": formal_status,
        "formal_phase1_overridden": False,
        "checks": checks,
        "metrics": {
            "conditioning_verdict": (
                conditioning.get("verdict") if conditioning else None
            ),
            "conditioning_frames": (
                conditioning.get("frame_count") if conditioning else None
            ),
            "strong_fraction": (
                conditioning.get("strong_fraction") if conditioning else None
            ),
            "unreferenced_fraction": (
                conditioning.get("unreferenced_fraction") if conditioning else None
            ),
            "fidelity_score": fidelity.get("score") if fidelity else None,
            "g5_passed": g5_passed,
            "g5_evidence": reconstruction,
        },
        "deferred_until_acceptance": {
            "final_authorized_capture": True,
            "production_rights_clearance": True,
        },
        "presentation_claims": {
            "allowed": [
                "prototype 3D inspectable",
                "géométrie LiDAR conditionnée",
                "diagnostic de couverture et de reconstruction",
            ],
            "forbidden": [
                "ENVIRONMENT_3D_READY",
                "reconstruction photoréaliste complète",
                "droits de production déjà libérés",
            ],
        },
    }


def prepare(workspace: Workspace) -> dict:
    """Republie le paquet, le viewer et le manifeste de démonstration."""
    scene_outputs = scene_package.build(workspace)
    viewer_outputs = viewer.build(workspace)
    report = assess(workspace)
    report["outputs"] = {
        "scene": {name: str(path) for name, path in scene_outputs.items()},
        "viewer": {
            "html": str(viewer_outputs.html),
            "payload": str(viewer_outputs.payload),
            "manifest": str(viewer_outputs.manifest),
        },
    }
    manifest = workspace.write_json("11_conditioning/demo_manifest.json", report)
    report["manifest"] = str(manifest)
    return report
