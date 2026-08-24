"""Projection des supports semantiques mesures dans le repere de la scene.

Cette etape n'extrude aucun masque et ne reconstruit aucune surface. Elle
applique seulement un enregistrement COLMAP/LiDAR accepte aux points COLMAP
deja mesures qui soutiennent les instances multi-vues. Des candidats lineaires
observes dans une seule vue peuvent aussi etre publies comme diagnostics,
jamais comme surfaces.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..geometry_align import apply_sim3
from ..workspace import Workspace
from .semantic_correspondence import (
    SemanticCorrespondenceUnavailable,
    _resolve_model_path,
)


class SemanticSupportRegistrationUnavailable(RuntimeError):
    """L'enregistrement mesure n'est pas disponible ou pas accepte."""


SINGLE_VIEW_LINEAR_CLASSES = frozenset({"beam", "column"})


def assess_single_view_linear_candidate(object_class: str, points: np.ndarray) -> dict:
    """Controle la direction de points mesures sans autoriser de surface."""
    points = np.asarray(points, dtype=float)
    minimum = 4 if object_class == "beam" else 3
    reasons: list[str] = []
    if object_class not in SINGLE_VIEW_LINEAR_CLASSES:
        reasons.append("class is not eligible for single-view linear support")
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < minimum:
        reasons.append(f"at least {minimum} measured points are required")
        return {"status": "refused", "refusal_reasons": reasons}
    extents = np.ptp(points, axis=0)
    horizontal = float(max(extents[0], extents[1]))
    vertical = float(extents[2])
    if object_class == "beam":
        if horizontal < 2.0:
            reasons.append(
                f"horizontal extent too small for beam: {horizontal:.3f} < 2.000 m"
            )
        if vertical > 2.0:
            reasons.append(
                f"vertical extent too large for beam: {vertical:.3f} > 2.000 m"
            )
        if horizontal / max(vertical, 0.1) < 2.0:
            reasons.append("beam directionality ratio below 2.0")
    elif object_class == "column":
        if vertical < 1.5:
            reasons.append(
                f"vertical extent too small for column: {vertical:.3f} < 1.500 m"
            )
        if horizontal > 1.5:
            reasons.append(
                f"horizontal extent too large for column: {horizontal:.3f} > 1.500 m"
            )
        if vertical / max(horizontal, 0.1) < 1.5:
            reasons.append("column directionality ratio below 1.5")
    return {
        "status": "accepted_support_only" if not reasons else "refused",
        "refusal_reasons": reasons,
        "metrics": {
            "measured_points": len(points),
            "extent_x_m": round(float(extents[0]), 5),
            "extent_y_m": round(float(extents[1]), 5),
            "extent_z_m": round(vertical, 5),
        },
    }


def _read(path: Path) -> dict:
    return json.loads(path.read_text("utf-8"))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transform_points(
    points: np.ndarray,
    *,
    sim3_rotation: np.ndarray,
    sim3_translation: np.ndarray,
    sim3_scale: float,
    projected_origin_xy: tuple[float, float],
    registration_translation: np.ndarray,
    scene_origin_xyz: np.ndarray,
) -> np.ndarray:
    """COLMAP brut -> Sim3 geographique -> LiDAR -> scene locale."""
    transformed = apply_sim3(
        np.asarray(points, dtype=float),
        np.asarray(sim3_rotation, dtype=float),
        np.asarray(sim3_translation, dtype=float),
        float(sim3_scale),
    )
    transformed[:, :2] += np.asarray(projected_origin_xy, dtype=float)
    transformed += np.asarray(registration_translation, dtype=float)
    transformed -= np.asarray(scene_origin_xyz, dtype=float)
    return transformed


def run(workspace: Workspace) -> tuple[Path, dict]:
    correspondence_path = workspace.path(
        "11_conditioning", "semantic_correspondences.json"
    )
    registration_path = workspace.path(
        "11_conditioning", "vertical_registration.json"
    )
    if not correspondence_path.is_file() or not registration_path.is_file():
        raise SemanticSupportRegistrationUnavailable(
            "correspondances ou enregistrement COLMAP/LiDAR absents"
        )
    correspondence = _read(correspondence_path)
    registration = _read(registration_path)
    if registration.get("status") != "accepted":
        raise SemanticSupportRegistrationUnavailable(
            "l'enregistrement COLMAP/LiDAR n'est pas accepte"
        )

    anchor_manifest_path = workspace.root / correspondence["sources"][
        "anchor_model_manifest"
    ]
    anchor_manifest = _read(anchor_manifest_path)
    selection_path = workspace.path(
        "07_reconstruction",
        "anchors",
        f"{anchor_manifest['anchor_selection_id']}.json",
    )
    selection = _read(selection_path)
    try:
        import pycolmap
        from pyproj import Transformer
        from shapely import contains_xy
        from shapely.geometry import Polygon
    except ImportError as exc:  # pragma: no cover - extras geo+sfm
        raise SemanticSupportRegistrationUnavailable(
            "les extras `geo` et `sfm` sont requis"
        ) from exc
    try:
        model_path = _resolve_model_path(
            workspace, anchor_manifest_path, anchor_manifest
        )
    except SemanticCorrespondenceUnavailable as exc:
        raise SemanticSupportRegistrationUnavailable(str(exc)) from exc
    reconstruction = pycolmap.Reconstruction(str(model_path))
    sim3 = selection["metrics"]["sim3"]
    origin_xy = Transformer.from_crs(
        "EPSG:4326",
        registration["hypothesis"]["horizontal_crs"],
        always_xy=True,
    ).transform(
        float(selection["metrics"]["enu_origin_lon"]),
        float(selection["metrics"]["enu_origin_lat"]),
    )
    registration_translation = np.asarray(
        registration["hypothesis"]["translation_projected_m"], dtype=float
    )
    scene_origin = np.asarray(
        registration["hypothesis"]["scene_origin_projected_xyz"], dtype=float
    )

    capture_path = workspace.path("06_geo", "capture_geometry.json")
    from .scene import load_scene

    scene = load_scene(capture_path)
    target_polygon = Polygon(scene.target.footprint).buffer(2.0) if scene.target else None
    registered_instances: list[dict] = []
    single_view_candidates: list[dict] = []
    single_view_audits: list[dict] = []
    point_assignments = 0
    unique_point_ids: set[int] = set()
    for instance in correspondence.get("instances", []):
        rows = [
            (int(point_id), reconstruction.points3D[int(point_id)])
            for point_id in instance.get("shared_point3d_ids", [])
            if int(point_id) in reconstruction.points3D
        ]
        if not rows:
            continue
        local = transform_points(
            np.asarray([point.xyz for _point_id, point in rows]),
            sim3_rotation=np.asarray(sim3["rotation"], dtype=float),
            sim3_translation=np.asarray(sim3["translation"], dtype=float),
            sim3_scale=float(sim3["scale"]),
            projected_origin_xy=(float(origin_xy[0]), float(origin_xy[1])),
            registration_translation=registration_translation,
            scene_origin_xyz=scene_origin,
        )
        projected = local + scene_origin
        object_class = str(instance.get("class"))
        inside_fraction = None
        if object_class == "building" and target_polygon is not None:
            inside_fraction = float(
                np.mean(contains_xy(target_polygon, projected[:, 0], projected[:, 1]))
            )
        plausible_z = (local[:, 2] >= -2.0) & (local[:, 2] <= 50.0)
        registered_instances.append(
            {
                "instance_id": instance.get("instance_id"),
                "class": object_class,
                "validated_view_count": instance.get("validated_view_count"),
                "provenance_class": "COLMAP_MEASURED",
                "semantic_evidence_class": "multi_view",
                "coordinate_frame": "conditioned_scene_local_ground",
                "registered_support_point_count": len(rows),
                "registered_support_centroid": np.round(local.mean(axis=0), 5).tolist(),
                "registered_support_bounds": {
                    "minimum": np.round(local.min(axis=0), 5).tolist(),
                    "maximum": np.round(local.max(axis=0), 5).tolist(),
                },
                "plausible_height_fraction": round(float(np.mean(plausible_z)), 5),
                "inside_target_footprint_fraction": (
                    None if inside_fraction is None else round(inside_fraction, 5)
                ),
                "points": [
                    {
                        "point3d_id": point_id,
                        "xyz": np.round(xyz, 5).tolist(),
                    }
                    for (point_id, _point), xyz in zip(rows, local, strict=True)
                ],
                "surface_geometry": None,
                "surface_reconstruction_status": "not_started",
                "blockers": [
                    "sparse points do not define a watertight object surface"
                ],
            }
        )
        point_assignments += len(rows)
        unique_point_ids.update(point_id for point_id, _point in rows)

    multiview_observation_ids = {
        str(observation_id)
        for instance in correspondence.get("instances", [])
        for observation_id in instance.get("observation_ids", [])
    }
    for support in correspondence.get("observation_support", []):
        object_class = str(support.get("class"))
        observation_id = str(support.get("observation_id"))
        if (
            object_class not in SINGLE_VIEW_LINEAR_CLASSES
            or observation_id in multiview_observation_ids
        ):
            continue
        rows = [
            (int(point_id), reconstruction.points3D[int(point_id)])
            for point_id in support.get("colmap_point3d_ids", [])
            if int(point_id) in reconstruction.points3D
        ]
        if rows:
            local = transform_points(
                np.asarray([point.xyz for _point_id, point in rows]),
                sim3_rotation=np.asarray(sim3["rotation"], dtype=float),
                sim3_translation=np.asarray(sim3["translation"], dtype=float),
                sim3_scale=float(sim3["scale"]),
                projected_origin_xy=(float(origin_xy[0]), float(origin_xy[1])),
                registration_translation=registration_translation,
                scene_origin_xyz=scene_origin,
            )
        else:
            local = np.empty((0, 3), dtype=float)
        assessment = assess_single_view_linear_candidate(object_class, local)
        single_view_audits.append(
            {
                "observation_id": observation_id,
                "asset_id": support.get("asset_id"),
                "class": object_class,
                **assessment,
            }
        )
        if assessment["status"] != "accepted_support_only":
            continue
        single_view_candidates.append(
            {
                "instance_id": f"single-view-{observation_id}",
                "observation_id": observation_id,
                "asset_id": support.get("asset_id"),
                "class": object_class,
                "validated_view_count": 1,
                "provenance_class": "COLMAP_MEASURED",
                "semantic_evidence_class": "single_view_candidate",
                "coordinate_frame": "conditioned_scene_local_ground",
                "registered_support_point_count": len(rows),
                "registered_support_centroid": np.round(local.mean(axis=0), 5).tolist(),
                "registered_support_bounds": {
                    "minimum": np.round(local.min(axis=0), 5).tolist(),
                    "maximum": np.round(local.max(axis=0), 5).tolist(),
                },
                "points": [
                    {"point3d_id": point_id, "xyz": np.round(xyz, 5).tolist()}
                    for (point_id, _point), xyz in zip(rows, local, strict=True)
                ],
                "validation": assessment["metrics"],
                "surface_geometry": None,
                "surface_reconstruction_status": "blocked_single_view_semantic",
                "blockers": [
                    "semantic class is supported by one validated image only",
                    "multi-view semantic confirmation is required for a surface",
                ],
            }
        )
        point_assignments += len(rows)
        unique_point_ids.update(point_id for point_id, _point in rows)

    generated_at = datetime.now(timezone.utc)
    run_id = f"registered-semantic-support-{generated_at.strftime('%Y%m%dT%H%M%SZ')}"
    payload = {
        "contract_version": 2,
        "run_id": run_id,
        "hotel_id": workspace.hotel_id,
        "generated_at": generated_at.isoformat(),
        "status": "registered_support_ready",
        "coordinate_frame": {
            "kind": "conditioned_scene_local_ground",
            "origin_projected_xyz": np.round(scene_origin, 6).tolist(),
            "horizontal_crs": registration["hypothesis"]["horizontal_crs"],
        },
        "policy": {
            "input_points": "shared COLMAP tracks from semantic multi-view instances",
            "single_view_linear_support": (
                "measured COLMAP points may be shown as labelled diagnostics after "
                "a directionality gate; surface creation remains forbidden"
            ),
            "registration": "accepted COLMAP-to-LiDAR translation only",
            "surface_creation": "forbidden in this stage",
        },
        "summary": {
            "registered_instances": len(registered_instances),
            "single_view_candidates": len(single_view_candidates),
            "single_view_audited": len(single_view_audits),
            "single_view_refused": sum(
                item["status"] == "refused" for item in single_view_audits
            ),
            "registered_point_assignments": point_assignments,
            "unique_registered_points": len(unique_point_ids),
            "surface_geometry_created": 0,
            "by_class": dict(
                sorted(Counter(item["class"] for item in registered_instances).items())
            ),
            "single_view_by_class": dict(
                sorted(Counter(item["class"] for item in single_view_candidates).items())
            ),
        },
        "sources": {
            "semantic_correspondences": str(
                correspondence_path.relative_to(workspace.root)
            ),
            "vertical_registration": str(
                registration_path.relative_to(workspace.root)
            ),
            "anchor_selection": str(selection_path.relative_to(workspace.root)),
            "anchor_model_manifest": str(
                anchor_manifest_path.relative_to(workspace.root)
            ),
            "capture_geometry": str(capture_path.relative_to(workspace.root)),
        },
        "source_digests": {
            str(path.relative_to(workspace.root)): _digest(path)
            for path in (
                correspondence_path,
                registration_path,
                selection_path,
                anchor_manifest_path,
                capture_path,
            )
        },
        "instances": registered_instances,
        "single_view_candidates": single_view_candidates,
        "single_view_audits": single_view_audits,
    }
    relative = f"11_conditioning/registered_semantic_support_runs/{run_id}.json"
    payload["versioned_artifact"] = relative
    workspace.write_json(relative, payload)
    path = workspace.write_json(
        "11_conditioning/registered_semantic_support.json", payload
    )
    return path, payload


__all__ = [
    "SemanticSupportRegistrationUnavailable",
    "assess_single_view_linear_candidate",
    "run",
    "transform_points",
]
