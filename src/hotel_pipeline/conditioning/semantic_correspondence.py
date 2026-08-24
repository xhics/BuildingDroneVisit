"""Association multi-vues des observations semantiques par pistes COLMAP.

Le module ne rapproche pas deux objets parce qu'ils se ressemblent. Une
correspondance n'est acceptee que lorsque les masques 2D recouvrent les memes
points 3D deja mesures par le noyau COLMAP. Les points restent exprimes dans
le repere du modele d'ancrage et ne sont pas injectes dans la scene locale tant
que l'enregistrement vertical COLMAP/LiDAR n'est pas valide.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from ..workspace import Workspace


class SemanticCorrespondenceUnavailable(RuntimeError):
    """Les artefacts mesures requis ne permettent pas l'association."""


def _read(path: Path) -> dict:
    return json.loads(path.read_text("utf-8"))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _latest_localization(workspace: Workspace) -> Path | None:
    folder = workspace.path("07_reconstruction", "localization")
    found = sorted(folder.glob("anchor-localization-*.json")) if folder.is_dir() else []
    return found[-1] if found else None


def point_in_polygon(point: tuple[float, float], polygon: list[list[float]]) -> bool:
    """Test pair/impair, bord inclus, sans dependance geometrique lourde."""
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    tolerance = 1e-6
    for index, current in enumerate(polygon):
        previous = polygon[index - 1]
        ax, ay = float(current[0]), float(current[1])
        bx, by = float(previous[0]), float(previous[1])
        cross = (x - bx) * (ay - by) - (y - by) * (ax - bx)
        if abs(cross) <= tolerance:
            dot = (x - bx) * (x - ax) + (y - by) * (y - ay)
            if dot <= tolerance:
                return True
        if (ay > y) != (by > y):
            intersection = (bx - ax) * (y - ay) / (by - ay) + ax
            if x < intersection:
                inside = not inside
    return inside


def _contains(observation: dict, xy: tuple[float, float]) -> bool:
    segmentation = observation.get("segmentation_2d") or {}
    polygon = segmentation.get("points") or []
    if len(polygon) >= 3:
        return point_in_polygon(xy, polygon)
    box = (observation.get("geometry_2d") or {}).get("xyxy")
    if not box or len(box) != 4:
        return False
    return float(box[0]) <= xy[0] <= float(box[2]) and float(box[1]) <= xy[1] <= float(box[3])


def _union_find_tracks(
    observations: list[dict],
    support_by_observation: dict[str, set[int]],
    *,
    min_shared_tracks: int,
    min_overlap: float,
) -> tuple[list[dict], list[list[str]]]:
    """Construit des composantes sans jamais mettre deux objets d'une vue ensemble."""
    by_id = {str(item["observation_id"]): item for item in observations}
    candidates: list[dict] = []
    for index, left in enumerate(observations):
        left_id = str(left["observation_id"])
        left_support = support_by_observation.get(left_id, set())
        if not left_support:
            continue
        for right in observations[index + 1 :]:
            if left.get("asset_id") == right.get("asset_id"):
                continue
            if left.get("class") != right.get("class"):
                continue
            right_id = str(right["observation_id"])
            right_support = support_by_observation.get(right_id, set())
            if not right_support:
                continue
            shared = left_support & right_support
            if not shared:
                continue
            overlap = len(shared) / max(1, min(len(left_support), len(right_support)))
            accepted = len(shared) >= min_shared_tracks and overlap >= min_overlap
            candidates.append(
                {
                    "left_observation_id": left_id,
                    "right_observation_id": right_id,
                    "class": left.get("class"),
                    "shared_point3d_ids": sorted(shared),
                    "shared_tracks": len(shared),
                    "minimum_support_overlap": round(overlap, 5),
                    "decision": "accepted" if accepted else "insufficient_support",
                }
            )

    parent = {observation_id: observation_id for observation_id in by_id}
    members = {observation_id: {observation_id} for observation_id in by_id}
    assets = {
        observation_id: {str(item.get("asset_id"))}
        for observation_id, item in by_id.items()
    }

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    accepted_edges = sorted(
        (item for item in candidates if item["decision"] == "accepted"),
        key=lambda item: (-item["shared_tracks"], -item["minimum_support_overlap"]),
    )
    for edge in accepted_edges:
        left_root = find(edge["left_observation_id"])
        right_root = find(edge["right_observation_id"])
        if left_root == right_root:
            continue
        # Deux detections de la meme classe dans une image representent deux
        # instances potentielles. Les fusionner par transitivite serait ambigu.
        if assets[left_root] & assets[right_root]:
            edge["decision"] = "rejected_same_view_conflict"
            continue
        parent[right_root] = left_root
        members[left_root] |= members[right_root]
        assets[left_root] |= assets[right_root]

    groups: dict[str, list[str]] = defaultdict(list)
    for observation_id in by_id:
        groups[find(observation_id)].append(observation_id)
    tracks = [sorted(group) for group in groups.values() if len(group) >= 2]
    tracks.sort(key=lambda group: (-len(group), group))
    return candidates, tracks


def build_tracks(
    observations: list[dict],
    support_by_observation: dict[str, set[int]],
    point_xyz: dict[int, tuple[float, float, float]],
    *,
    min_shared_tracks: int = 3,
    min_overlap: float = 0.12,
) -> tuple[list[dict], list[dict]]:
    """Produit les paires auditees et les instances multi-vues mesurees."""
    pairs, groups = _union_find_tracks(
        observations,
        support_by_observation,
        min_shared_tracks=min_shared_tracks,
        min_overlap=min_overlap,
    )
    by_id = {str(item["observation_id"]): item for item in observations}
    instances: list[dict] = []
    for index, observation_ids in enumerate(groups):
        point_votes: Counter[int] = Counter()
        for observation_id in observation_ids:
            point_votes.update(support_by_observation.get(observation_id, set()))
        shared_ids = sorted(point_id for point_id, count in point_votes.items() if count >= 2)
        measured = np.asarray(
            [point_xyz[point_id] for point_id in shared_ids if point_id in point_xyz],
            dtype=float,
        )
        if not len(measured):
            continue
        member_observations = [by_id[observation_id] for observation_id in observation_ids]
        object_class = str(member_observations[0].get("class"))
        instances.append(
            {
                "instance_id": f"semantic-track-{index:04d}",
                "class": object_class,
                "observation_ids": observation_ids,
                "asset_ids": sorted({str(item.get("asset_id")) for item in member_observations}),
                "validated_view_count": len(member_observations),
                "shared_point3d_ids": shared_ids,
                "measured_support_point_count": len(shared_ids),
                "measured_support_centroid": np.round(measured.mean(axis=0), 6).tolist(),
                "measured_support_bounds": {
                    "minimum": np.round(measured.min(axis=0), 6).tolist(),
                    "maximum": np.round(measured.max(axis=0), 6).tolist(),
                },
                "coordinate_frame": "anchor_colmap_model",
                "provenance_class": "COLMAP_MEASURED_SUPPORT",
                "geometry_3d": None,
                "triangulation_status": "measured_support_only",
                "scene_integration_status": "blocked_vertical_registration",
                "blockers": [
                    "COLMAP-to-LiDAR vertical registration not validated",
                    "object surface is not reconstructed from sparse support alone",
                ],
            }
        )
    return pairs, instances


def _resolve_model_path(workspace: Workspace, manifest_path: Path, manifest: dict) -> Path:
    raw = Path(str(manifest.get("model_path", "")))
    if raw.is_dir():
        return raw
    model_id = str(manifest.get("anchor_model_id", ""))
    if model_id:
        parts = raw.parts
        if model_id in parts:
            suffix = parts[parts.index(model_id) + 1 :]
            relocated = manifest_path.parent / model_id
            if suffix:
                relocated = relocated.joinpath(*suffix)
            if relocated.is_dir():
                return relocated
        candidates = sorted((manifest_path.parent / model_id).glob("stability/run-*/0"))
        if candidates:
            return candidates[0]
    raise SemanticCorrespondenceUnavailable("modele COLMAP d'ancrage introuvable")


def _resolve_images(reconstruction: object, inputs: list[dict]) -> dict[str, object]:
    images = list(reconstruction.images.values())
    resolved: dict[str, object] = {}
    for item in inputs:
        asset_id = str(item.get("asset_id"))
        token = Path(str(item.get("path", ""))).stem
        matches = [image for image in images if token and token in str(image.name)]
        if len(matches) == 1:
            resolved[asset_id] = matches[0]
    return resolved


def _point_support(
    observations: Iterable[dict],
    image_by_asset: dict[str, object],
    reconstruction: object,
) -> tuple[dict[str, set[int]], dict[int, tuple[float, float, float]]]:
    support: dict[str, set[int]] = {}
    xyz: dict[int, tuple[float, float, float]] = {}
    for observation in observations:
        observation_id = str(observation["observation_id"])
        image = image_by_asset.get(str(observation.get("asset_id")))
        point_ids: set[int] = set()
        if image is not None:
            for point2d in image.points2D:
                if not point2d.has_point3D():
                    continue
                xy = tuple(float(value) for value in point2d.xy)
                if not _contains(observation, xy):
                    continue
                point_id = int(point2d.point3D_id)
                point_ids.add(point_id)
                if point_id in reconstruction.points3D:
                    point = reconstruction.points3D[point_id]
                    xyz[point_id] = tuple(float(value) for value in point.xyz)
        support[observation_id] = point_ids
    return support, xyz


def run(
    workspace: Workspace,
    *,
    min_shared_tracks: int = 3,
    min_overlap: float = 0.12,
) -> tuple[Path, dict]:
    """Associe le dernier run semantique au noyau COLMAP mesure."""
    if min_shared_tracks < 1:
        raise ValueError("min_shared_tracks doit etre positif")
    if not 0.0 <= min_overlap <= 1.0:
        raise ValueError("min_overlap doit etre compris entre 0 et 1")
    semantic_path = workspace.path("11_conditioning", "semantic_observations.json")
    localization_path = _latest_localization(workspace)
    if not semantic_path.is_file():
        raise SemanticCorrespondenceUnavailable("observations semantiques absentes")
    if localization_path is None:
        raise SemanticCorrespondenceUnavailable("localisation ancree absente")
    semantic = _read(semantic_path)
    localization = _read(localization_path)
    anchor_model_id = str(localization.get("anchor_model_id", ""))
    anchor_manifest_path = workspace.path(
        "07_reconstruction", "anchors", f"{anchor_model_id}.json"
    )
    if not anchor_manifest_path.is_file():
        raise SemanticCorrespondenceUnavailable("manifeste du noyau COLMAP absent")
    anchor_manifest = _read(anchor_manifest_path)
    model_path = _resolve_model_path(workspace, anchor_manifest_path, anchor_manifest)
    try:
        import pycolmap
    except ImportError as exc:  # pragma: no cover - extra sfm
        raise SemanticCorrespondenceUnavailable(
            "installez l'extra `sfm` pour relire les pistes COLMAP"
        ) from exc
    reconstruction = pycolmap.Reconstruction(str(model_path))
    image_by_asset = _resolve_images(reconstruction, semantic.get("inputs", []))
    observations = list(semantic.get("observations", []))
    support, point_xyz = _point_support(observations, image_by_asset, reconstruction)
    pairs, instances = build_tracks(
        observations,
        support,
        point_xyz,
        min_shared_tracks=min_shared_tracks,
        min_overlap=min_overlap,
    )
    observation_support = [
        {
            "observation_id": str(observation["observation_id"]),
            "asset_id": observation.get("asset_id"),
            "class": observation.get("class"),
            "colmap_point3d_ids": sorted(support.get(str(observation["observation_id"]), set())),
            "colmap_support_count": len(support.get(str(observation["observation_id"]), set())),
            "coordinate_frame": "anchor_colmap_model",
        }
        for observation in observations
    ]
    generated_at = datetime.now(timezone.utc)
    run_id = f"semantic-links-{generated_at.strftime('%Y%m%dT%H%M%SZ')}"
    multiview_ids = {
        observation_id
        for instance in instances
        for observation_id in instance["observation_ids"]
    }
    payload = {
        "contract_version": 1,
        "run_id": run_id,
        "hotel_id": workspace.hotel_id,
        "generated_at": generated_at.isoformat(),
        "policy": {
            "association_basis": "shared measured COLMAP point tracks inside semantic masks",
            "min_shared_tracks": min_shared_tracks,
            "min_support_overlap": min_overlap,
            "same_view_instance_conflicts": "rejected",
            "appearance_only_matching": "forbidden",
            "scene_geometry_creation": "forbidden until vertical registration",
        },
        "coordinate_frame": "anchor_colmap_model",
        "sources": {
            "semantic_observations": str(semantic_path.relative_to(workspace.root)),
            "localization": str(localization_path.relative_to(workspace.root)),
            "anchor_model_manifest": str(anchor_manifest_path.relative_to(workspace.root)),
            "anchor_model_digest": anchor_manifest.get("model_digest"),
        },
        "source_digests": {
            str(path.relative_to(workspace.root)): _digest(path)
            for path in (semantic_path, localization_path, anchor_manifest_path)
        },
        "summary": {
            "semantic_observations": len(observations),
            "resolved_colmap_images": len(image_by_asset),
            "observations_with_colmap_support": sum(bool(item["colmap_support_count"]) for item in observation_support),
            "candidate_pairs": len(pairs),
            "accepted_pairs": sum(item["decision"] == "accepted" for item in pairs),
            "multiview_instances": len(instances),
            "multiview_observations": len(multiview_ids),
            "shared_measured_tracks": len({point_id for instance in instances for point_id in instance["shared_point3d_ids"]}),
            "geometry_3d_created": 0,
            "by_class": dict(sorted(Counter(item["class"] for item in instances).items())),
        },
        "observation_support": observation_support,
        "pair_decisions": pairs,
        "instances": instances,
    }
    relative = f"11_conditioning/semantic_correspondence_runs/{run_id}.json"
    payload["versioned_artifact"] = relative
    workspace.write_json(relative, payload)
    path = workspace.write_json(
        "11_conditioning/semantic_correspondences.json", payload
    )
    return path, payload


__all__ = [
    "SemanticCorrespondenceUnavailable",
    "build_tracks",
    "point_in_polygon",
    "run",
]
