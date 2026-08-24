"""Registre versionné des observations architecturales issues des images.

Une observation 2D n'est jamais promue en point 3D par simple ressemblance.
Le registre relie chaque segment à son image et au verdict de pose courant ;
il dit seulement s'il est éligible à une triangulation future.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from ..workspace import Workspace


def _read(path: Path) -> dict:
    return json.loads(path.read_text("utf-8"))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _latest_localization(workspace: Workspace) -> Path | None:
    folder = workspace.path("07_reconstruction", "localization")
    if not folder.is_dir():
        return None
    found = sorted(folder.glob("anchor-localization-*.json"))
    return found[-1] if found else None


def build(workspace: Workspace) -> tuple[Path, dict]:
    ridge_path = workspace.path("06_geo", "ridge_match.json")
    identity_path = workspace.path("09_confidence", "identity_screening.json")
    localization_path = _latest_localization(workspace)

    pose_by_asset: dict[str, dict] = {}
    localization = None
    if localization_path is not None:
        localization = _read(localization_path)
        pose_by_asset = {
            str(item.get("asset_id")): item for item in localization.get("poses", [])
        }

    observations: list[dict] = []
    if ridge_path.is_file():
        ridge = _read(ridge_path)
        for index, item in enumerate(ridge.get("matches", [])):
            asset_id = str(item.get("asset_id", "unknown"))
            pose = pose_by_asset.get(asset_id, {})
            pose_accepted = (
                pose.get("decision") == "accepted"
                and pose.get("evidence_class")
                in {"anchor_measured", "measured_localized"}
            )
            matched = bool(item.get("matched")) and not bool(item.get("ambiguous"))
            observations.append(
                {
                    "observation_id": f"roof-edge-{index:05d}",
                    "asset_id": asset_id,
                    "class": "roof_edge",
                    "geometry_2d": (
                        {"type": "segment", "xyxy": item.get("segment")}
                        if item.get("segment") is not None
                        else None
                    ),
                    "detector": "geometric_line_matcher",
                    "detector_status": "accepted" if matched else "rejected",
                    "pose_status": "validated" if pose_accepted else "unvalidated",
                    "pose_evidence_class": pose.get("evidence_class", "unknown"),
                    "ridge_index": item.get("ridge_index"),
                    "match_cost": item.get("cost"),
                    "provenance_class": "UNKNOWN",
                    "geometry_3d": None,
                    "triangulation_status": "blocked",
                    "blockers": [] if matched and pose_accepted else [
                        "2D match rejected or camera pose not validated"
                    ],
                }
            )

    if identity_path.is_file():
        identity = _read(identity_path)
        for index, anchor in enumerate(identity.get("anchors", {}).get("anchors", [])):
            observations.append(
                {
                    "observation_id": f"sign-{index:03d}",
                    "asset_id": anchor.get("asset_id"),
                    "class": "sign",
                    "geometry_2d": None,
                    "detector": "ocr_identity_anchor",
                    "detector_status": "semantic_only",
                    "pose_status": "not_required_for_identity",
                    "provenance_class": "SEMANTICALLY_CONSTRAINED",
                    "geometry_3d": None,
                    "triangulation_status": "blocked",
                    "blockers": ["no validated 2D extent"],
                    "evidence": anchor.get("evidence"),
                }
            )

        # Baseline locale : les formes sont relevées sur les images d'identité
        # positives seulement. Elles restent des candidats, jamais des objets
        # 3D, jusqu'à confirmation sémantique et multivue.
        from .image_objects import detect_cached

        matches = [
            item
            for item in identity.get("assets", [])
            if item.get("status") == "match" and item.get("path")
        ][:20]
        for asset in matches:
            asset_id = str(asset.get("asset_id"))
            image_path = Path(str(asset.get("path")))
            if not image_path.is_file():
                continue
            pose = pose_by_asset.get(asset_id, {})
            pose_accepted = (
                pose.get("decision") == "accepted"
                and pose.get("evidence_class")
                in {"anchor_measured", "measured_localized"}
            )
            for index, candidate in enumerate(
                detect_cached(
                    image_path,
                    asset_id,
                    workspace.path("11_conditioning", "structural_cache"),
                )
            ):
                observations.append(
                    {
                        "observation_id": f"structure-{asset_id}-{index:03d}",
                        "asset_id": asset_id,
                        "class": candidate["class"],
                        "geometry_2d": candidate["geometry_2d"],
                        "detector": "opencv_structural_candidates_v1",
                        "detector_status": "candidate",
                        "detector_score": candidate.get("score"),
                        "pose_status": "validated" if pose_accepted else "unvalidated",
                        "pose_evidence_class": pose.get("evidence_class", "unknown"),
                        "provenance_class": "UNKNOWN",
                        "geometry_3d": None,
                        "triangulation_status": "blocked",
                        "blockers": [
                            "semantic class not validated",
                            "multiview correspondence not established",
                        ],
                    }
                )

    multiview = Counter(
        o.get("ridge_index")
        for o in observations
        if o.get("class") == "roof_edge"
        and o.get("detector_status") == "accepted"
        and o.get("pose_status") == "validated"
    )
    for observation in observations:
        if observation.get("class") != "roof_edge":
            continue
        support = multiview.get(observation.get("ridge_index"), 0)
        observation["validated_multiview_support"] = support
        if (
            observation.get("detector_status") == "accepted"
            and observation.get("pose_status") == "validated"
            and support >= 2
        ):
            observation["provenance_class"] = "SEMANTICALLY_CONSTRAINED"
            observation["triangulation_status"] = "eligible"
            observation["blockers"] = []
        elif not observation["blockers"]:
            observation["blockers"] = ["fewer than two validated camera poses"]

    eligible = sum(o["triangulation_status"] == "eligible" for o in observations)
    payload = {
        "contract_version": 1,
        "hotel_id": workspace.hotel_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "geometry_requires": (
                "at least two validated poses plus accepted multiview consistency, "
                "or an intersection with measured LiDAR"
            ),
            "provenance_classes": [
                "COLMAP_MEASURED",
                "SEMANTICALLY_CONSTRAINED",
                "OCCLUDED_INFERRED",
                "UNKNOWN",
            ],
        },
        "detectors": {
            "geometric_line_matcher": "available",
            "opencv_structural_candidates": "available",
            "grounding_dino": (
                "available" if importlib.util.find_spec("groundingdino") else "not_installed"
            ),
            "sam2": "available" if importlib.util.find_spec("sam2") else "not_installed",
        },
        "pose_source": (
            None if localization_path is None else str(localization_path.relative_to(workspace.root))
        ),
        "source_digests": {
            str(path.relative_to(workspace.root)): _digest(path)
            for path in (ridge_path, identity_path, localization_path)
            if path is not None and path.is_file()
        },
        "summary": {
            "observations": len(observations),
            "eligible_for_triangulation": eligible,
            "geometry_3d_created": 0,
            "by_class": dict(sorted(Counter(o["class"] for o in observations).items())),
            "validated_pose_rate": (
                None if localization is None else localization.get("validated_registration_rate")
            ),
        },
        "observations": observations,
    }
    path = workspace.write_json(
        "11_conditioning/architectural_observations.json", payload
    )
    return path, payload
