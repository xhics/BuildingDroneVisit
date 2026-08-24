"""Audit indépendant du repère commun COLMAP/LiDAR.

L'audit ne modifie ni la scène ni l'enregistrement courant. Il vérifie trois
questions séparées :

1. la Sim(3) géographique du modèle source est-elle applicable au noyau COLMAP
   reconstruit ;
2. une correction Sim(3) complète améliore-t-elle réellement le holdout LiDAR ;
3. les 32 arêtes de toiture LiDAR sont-elles couvertes à moins d'un mètre.

Une amélioration du seul résidu d'ajustement n'autorise jamais la géométrie.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..anchor_localization import _enu_one, _robust_sim3, load_model_poses
from ..geometry_align import apply_sim3, umeyama_sim3
from ..workspace import Workspace
from .heights import find_laz
from .laz_cache import read_window
from .scene import load_scene
from .semantic_correspondence import (
    _point_support,
    _resolve_images,
    _resolve_model_path,
    build_tracks,
)
from .semantic_registered_support import transform_points
from .vertical_registration import (
    _initial_z_offset,
    _translation_icp,
    _voxel_downsample,
)


class Sim3RoofAuditUnavailable(RuntimeError):
    """Les artefacts nécessaires à l'audit sont absents."""


def _read(path: Path) -> dict:
    return json.loads(path.read_text("utf-8"))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compose_sim3(
    first_rotation: np.ndarray,
    first_translation: np.ndarray,
    first_scale: float,
    second_rotation: np.ndarray,
    second_translation: np.ndarray,
    second_scale: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compose ``first`` puis ``second`` dans la convention du dépôt."""
    rotation = np.asarray(second_rotation) @ np.asarray(first_rotation)
    scale = float(second_scale) * float(first_scale)
    translation = (
        float(second_scale)
        * (np.asarray(second_rotation) @ np.asarray(first_translation))
        + np.asarray(second_translation)
    )
    return rotation, translation, scale


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _point_segment_distance(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> float:
    direction = end - start
    denominator = float(direction @ direction)
    position = 0.0 if denominator <= 1e-12 else float(
        np.clip(((point - start) @ direction) / denominator, 0.0, 1.0)
    )
    return float(np.linalg.norm(point - (start + position * direction)))


def roof_edge_metrics(
    points: np.ndarray,
    ridges: list[tuple[np.ndarray, np.ndarray]],
) -> dict:
    """Mesure la couverture bidirectionnelle des arêtes métriques."""
    points = np.asarray(points, dtype=float)
    if not len(points) or not ridges:
        return {
            "support_points": len(points),
            "lidar_edges": len(ridges),
            "edges_within_1m": 0,
            "edge_coverage_fraction_1m": 0.0,
            "edge_nearest_median_m": None,
            "edge_nearest_p90_m": None,
            "support_to_edge_median_m": None,
            "support_to_edge_p90_m": None,
        }
    support_distances = np.asarray(
        [
            min(_point_segment_distance(point, start, end) for start, end in ridges)
            for point in points
        ]
    )
    edge_distances = np.asarray(
        [
            min(_point_segment_distance(point, start, end) for point in points)
            for start, end in ridges
        ]
    )
    edges_within = int(np.sum(edge_distances <= 1.0))
    return {
        "support_points": len(points),
        "lidar_edges": len(ridges),
        "edges_within_1m": edges_within,
        "edge_coverage_fraction_1m": round(edges_within / len(ridges), 5),
        "edge_nearest_median_m": round(float(np.median(edge_distances)), 5),
        "edge_nearest_p90_m": round(float(np.percentile(edge_distances, 90)), 5),
        "support_to_edge_median_m": round(float(np.median(support_distances)), 5),
        "support_to_edge_p90_m": round(float(np.percentile(support_distances, 90)), 5),
    }


def _asset_for_image(image_name: str, assets: list[dict]) -> dict | None:
    stem = Path(image_name).stem
    matches = [
        asset
        for asset in assets
        if asset.get("camera_lat") is not None
        and asset.get("camera_lon") is not None
        and str(asset.get("id", "")).split("-", 1)[-1] in stem
    ]
    return matches[0] if len(matches) == 1 else None


def _camera_rows(
    model_path: Path,
    assets: list[dict],
    *,
    origin_lat: float,
    origin_lon: float,
) -> list[dict]:
    rows: list[dict] = []
    for pose in load_model_poses(model_path):
        asset = _asset_for_image(pose.image_name, assets)
        if asset is None:
            continue
        rows.append(
            {
                "asset_id": str(asset["id"]),
                "source": str(asset.get("source_family") or asset.get("source")),
                "camera_center": pose.camera_center,
                "target_enu": _enu_one(
                    float(asset["camera_lat"]),
                    float(asset["camera_lon"]),
                    origin_lat,
                    origin_lon,
                ),
            }
        )
    return rows


def _residual_summary(residuals: np.ndarray, inliers: np.ndarray) -> dict:
    return {
        "inliers": int(inliers.sum()),
        "total": len(residuals),
        "inlier_fraction": round(float(np.mean(inliers)), 5),
        "inlier_rmse_m": (
            round(float(np.sqrt(np.mean(residuals[inliers] ** 2))), 5)
            if inliers.any()
            else None
        ),
        "all_median_m": round(float(np.median(residuals)), 5),
        "all_p90_m": round(float(np.percentile(residuals, 90)), 5),
    }


def _audit_model(
    name: str,
    model_path: Path,
    assets: list[dict],
    *,
    origin_lat: float,
    origin_lon: float,
    seed: int,
) -> dict:
    rows = _camera_rows(
        model_path,
        assets,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
    )
    payload: dict = {
        "model": name,
        "model_path": str(model_path),
        "registered_images": len(load_model_poses(model_path)),
        "gps_correspondences": len(rows),
    }
    if len(rows) < 3:
        return {
            **payload,
            "status": "underdetermined",
            "reason": "fewer than three GPS correspondences",
        }
    source = np.asarray([item["camera_center"] for item in rows])
    target = np.asarray([item["target_enu"] for item in rows])
    rotation, translation, scale, inliers = _robust_sim3(
        source,
        target,
        threshold_m=10.0,
        seed=seed,
    )
    residuals = np.linalg.norm(
        apply_sim3(source, rotation, translation, scale) - target,
        axis=1,
    )
    sources = Counter(
        item["source"] for item, accepted in zip(rows, inliers, strict=True) if accepted
    )
    metrics = _residual_summary(residuals, inliers)
    globally_consistent = (
        metrics["inliers"] >= 8
        and metrics["inlier_fraction"] >= 0.70
        and len(sources) >= 2
        and metrics["inlier_rmse_m"] is not None
        and metrics["inlier_rmse_m"] <= 3.0
    )
    return {
        **payload,
        "status": "consistent" if globally_consistent else "partial_consensus",
        "sim3": {
            "scale": float(scale),
            "rotation": np.asarray(rotation).tolist(),
            "translation": np.asarray(translation).tolist(),
        },
        "metrics": metrics,
        "inlier_sources": dict(sorted(sources.items())),
        "camera_residuals": [
            {
                "asset_id": item["asset_id"],
                "source": item["source"],
                "residual_m": round(float(residual), 5),
                "inlier": bool(accepted),
            }
            for item, residual, accepted in zip(rows, residuals, inliers, strict=True)
        ],
    }


def _nearest_metrics(points: np.ndarray, tree: object) -> dict:
    distances, _indices = tree.query(points, k=1, workers=-1)
    return {
        "points": len(points),
        "median_m": round(float(np.median(distances)), 5),
        "p90_m": round(float(np.percentile(distances, 90)), 5),
        "support_fraction_1m": round(float(np.mean(distances <= 1.0)), 5),
    }


def _similarity_icp_candidates(
    base: np.ndarray,
    point_ids: np.ndarray,
    lidar: np.ndarray,
) -> tuple[list[dict], dict, np.ndarray]:
    from scipy.spatial import cKDTree

    tree = cKDTree(lidar)
    fitting = point_ids % 5 != 0
    holdout = ~fitting
    candidates: list[tuple[tuple[float, float], dict, np.ndarray]] = []
    for trim in (0.5, 0.6, 0.7, 0.8):
        current = base.copy()
        rotation = np.eye(3)
        translation = np.zeros(3)
        scale = 1.0
        for threshold in (3.0, 2.0, 1.5, 1.0, 0.75, 0.5):
            for _iteration in range(4):
                distances, indices = tree.query(current[fitting], k=1, workers=-1)
                selected = distances < threshold
                if int(selected.sum()) < 20:
                    continue
                cap = float(np.quantile(distances[selected], trim))
                selected &= distances <= cap
                trial_rotation, trial_translation, trial_scale = umeyama_sim3(
                    base[fitting][selected], lidar[indices[selected]]
                )
                angle = _rotation_angle_deg(trial_rotation)
                if not 0.80 <= trial_scale <= 1.20 or angle > 10.0:
                    continue
                rotation = trial_rotation
                translation = trial_translation
                scale = float(trial_scale)
                current = apply_sim3(base, rotation, translation, scale)
        fit_metrics = _nearest_metrics(current[fitting], tree)
        holdout_metrics = _nearest_metrics(current[holdout], tree)
        record = {
            "trim_fraction": trim,
            "correction": {
                "scale": scale,
                "rotation": np.asarray(rotation).tolist(),
                "translation_m": np.asarray(translation).tolist(),
                "rotation_deg": round(_rotation_angle_deg(rotation), 5),
            },
            "fit": fit_metrics,
            "holdout": holdout_metrics,
        }
        candidates.append(
            ((fit_metrics["p90_m"], fit_metrics["median_m"]), record, current)
        )
    _score, chosen, transformed = min(candidates, key=lambda item: item[0])
    return [item[1] for item in candidates], chosen, transformed


def run(workspace: Workspace) -> tuple[Path, dict]:
    correspondence_path = workspace.path(
        "11_conditioning", "semantic_correspondences.json"
    )
    registration_path = workspace.path("11_conditioning", "vertical_registration.json")
    support_path = workspace.path(
        "11_conditioning", "registered_semantic_support.json"
    )
    scene_path = workspace.path("11_conditioning", "conditioned_scene.json")
    asset_path = workspace.path("00_manifest", "asset_manifest.json")
    semantic_path = workspace.path("11_conditioning", "semantic_observations.json")
    required = [
        correspondence_path,
        registration_path,
        support_path,
        scene_path,
        asset_path,
        semantic_path,
    ]
    if any(not path.is_file() for path in required):
        raise Sim3RoofAuditUnavailable("artefacts COLMAP/LiDAR incomplets")

    correspondence = _read(correspondence_path)
    registration = _read(registration_path)
    support = _read(support_path)
    conditioned_scene = _read(scene_path)
    assets = _read(asset_path).get("assets", [])
    manifest_path = workspace.root / correspondence["sources"]["anchor_model_manifest"]
    manifest = _read(manifest_path)
    selection_path = workspace.path(
        "07_reconstruction", "anchors", f"{manifest['anchor_selection_id']}.json"
    )
    selection = _read(selection_path)
    try:
        anchor_path = _resolve_model_path(workspace, manifest_path, manifest)
    except Exception as exc:
        raise Sim3RoofAuditUnavailable(str(exc)) from exc

    origin_lat = float(selection["metrics"]["enu_origin_lat"])
    origin_lon = float(selection["metrics"]["enu_origin_lon"])
    direct_anchor = _audit_model(
        "active_anchor",
        anchor_path,
        assets,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        seed=20260824,
    )

    anchor_rows = _camera_rows(
        anchor_path,
        assets,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
    )
    old_sim3 = selection["metrics"]["sim3"]
    anchor_source = np.asarray([item["camera_center"] for item in anchor_rows])
    anchor_target = np.asarray([item["target_enu"] for item in anchor_rows])
    old_residuals = np.linalg.norm(
        apply_sim3(
            anchor_source,
            np.asarray(old_sim3["rotation"]),
            np.asarray(old_sim3["translation"]),
            float(old_sim3["scale"]),
        )
        - anchor_target,
        axis=1,
    )
    inherited_sim3_audit = {
        "source_model_scale": float(old_sim3["scale"]),
        "applied_to": "rebuilt active anchor model",
        "camera_gps_median_m": round(float(np.median(old_residuals)), 5),
        "camera_gps_p90_m": round(float(np.percentile(old_residuals, 90)), 5),
        "camera_gps_min_m": round(float(np.min(old_residuals)), 5),
        "camera_gps_max_m": round(float(np.max(old_residuals)), 5),
        "applicable": bool(np.percentile(old_residuals, 90) <= 10.0),
    }

    components: list[dict] = []
    component_root = workspace.path(
        "05_colmap", "aliked_lightglue_solve", "sparse-combined", "models"
    )
    if component_root.is_dir():
        for component in sorted(path for path in component_root.iterdir() if path.is_dir()):
            components.append(
                _audit_model(
                    f"combined-{component.name}",
                    component,
                    assets,
                    origin_lat=origin_lat,
                    origin_lon=origin_lon,
                    seed=20260824 + int(component.name),
                )
            )

    try:
        import pycolmap
        from pyproj import Transformer
        from scipy.spatial import cKDTree
        from shapely import contains_xy
        from shapely.geometry import Polygon
    except ImportError as exc:
        raise Sim3RoofAuditUnavailable("les extras geo et sfm sont requis") from exc

    reconstruction = pycolmap.Reconstruction(str(anchor_path))
    building = next(
        (
            item
            for item in support.get("instances", [])
            if item.get("class") == "building"
        ),
        None,
    )
    if building is None:
        raise Sim3RoofAuditUnavailable("support bâtiment enregistré absent")
    building_points = np.asarray([item["xyz"] for item in building["points"]])
    building_ids = np.asarray(
        [int(item["point3d_id"]) for item in building["points"]], dtype=np.int64
    )

    scene = load_scene(workspace.path("06_geo", "capture_geometry.json"))
    window = read_window(find_laz(workspace, scene.centre), scene.centre, 110.0)
    if window is None or scene.target is None:
        raise Sim3RoofAuditUnavailable("LiDAR ou bâtiment cible absent")
    target_zone = Polygon(scene.target.footprint).buffer(2.0)
    usable = (window.classification == 6) & contains_xy(
        target_zone, window.x, window.y
    )
    scene_origin = np.asarray(
        registration["hypothesis"]["scene_origin_projected_xyz"], dtype=float
    )
    lidar = np.column_stack((window.x[usable], window.y[usable], window.z[usable]))
    lidar -= scene_origin
    lidar = _voxel_downsample(lidar, 0.10)
    lidar_tree = cKDTree(lidar)
    fitting = building_ids % 5 != 0
    holdout = ~fitting
    baseline = {
        "fit": _nearest_metrics(building_points[fitting], lidar_tree),
        "holdout": _nearest_metrics(building_points[holdout], lidar_tree),
    }
    candidate_rows, candidate, corrected_building = _similarity_icp_candidates(
        building_points, building_ids, lidar
    )

    origin_xy = Transformer.from_crs(
        "EPSG:4326",
        registration["hypothesis"]["horizontal_crs"],
        always_xy=True,
    ).transform(origin_lon, origin_lat)
    all_anchor = transform_points(
        np.asarray([point.xyz for point in reconstruction.points3D.values()]),
        sim3_rotation=np.asarray(old_sim3["rotation"]),
        sim3_translation=np.asarray(old_sim3["translation"]),
        sim3_scale=float(old_sim3["scale"]),
        projected_origin_xy=(float(origin_xy[0]), float(origin_xy[1])),
        registration_translation=np.asarray(
            registration["hypothesis"]["translation_projected_m"]
        ),
        scene_origin_xyz=scene_origin,
    )
    local_target = Polygon(
        [
            (float(x - scene.centre[0]), float(y - scene.centre[1]))
            for x, y in scene.target.footprint
        ]
    ).buffer(2.0)
    in_target = contains_xy(local_target, all_anchor[:, 0], all_anchor[:, 1])
    roof_support = all_anchor[
        in_target & (all_anchor[:, 2] >= 9.0) & (all_anchor[:, 2] <= 14.0)
    ]
    correction = candidate["correction"]
    corrected_all = apply_sim3(
        all_anchor,
        np.asarray(correction["rotation"]),
        np.asarray(correction["translation_m"]),
        float(correction["scale"]),
    )
    corrected_in_target = contains_xy(
        local_target, corrected_all[:, 0], corrected_all[:, 1]
    )
    corrected_roof_support = corrected_all[
        corrected_in_target
        & (corrected_all[:, 2] >= 9.0)
        & (corrected_all[:, 2] <= 14.0)
    ]
    ridges = [
        (np.asarray(item["a"], dtype=float), np.asarray(item["b"], dtype=float))
        for item in conditioned_scene.get("ridges", [])
    ]
    baseline_roof = roof_edge_metrics(roof_support, ridges)
    candidate_roof = roof_edge_metrics(corrected_roof_support, ridges)
    candidate["roof_control"] = candidate_roof
    candidate["effective_geographic_scale"] = round(
        float(old_sim3["scale"]) * float(correction["scale"]), 8
    )

    # Le modèle source est le seul repère auquel la Sim(3) héritée appartient
    # réellement. On répète donc le même audit sur ses seules vues sémantiques,
    # sans importer les cinq autres composants disjoints.
    source_component_probe: dict
    source_model_path = Path(str(selection.get("source_model_path", "")))
    if source_model_path.is_dir():
        semantic = _read(semantic_path)
        source_reconstruction = pycolmap.Reconstruction(str(source_model_path))
        source_images = _resolve_images(
            source_reconstruction, semantic.get("inputs", [])
        )
        source_support, source_xyz = _point_support(
            semantic.get("observations", []),
            source_images,
            source_reconstruction,
        )
        _pairs, source_instances = build_tracks(
            semantic.get("observations", []),
            source_support,
            source_xyz,
        )
        source_buildings = [
            item for item in source_instances if item.get("class") == "building"
        ]
        if len(source_buildings) == 1:
            source_rows = [
                (int(point_id), source_reconstruction.points3D[int(point_id)])
                for point_id in source_buildings[0].get("shared_point3d_ids", [])
                if int(point_id) in source_reconstruction.points3D
                and float(source_reconstruction.points3D[int(point_id)].error) <= 3.0
                and len(source_reconstruction.points3D[int(point_id)].track.elements) >= 2
            ]
            source_raw = np.asarray([point.xyz for _point_id, point in source_rows])
            source_ids = np.asarray(
                [point_id for point_id, _point in source_rows], dtype=np.int64
            )
            source_projected = apply_sim3(
                source_raw,
                np.asarray(old_sim3["rotation"]),
                np.asarray(old_sim3["translation"]),
                float(old_sim3["scale"]),
            )
            source_projected[:, :2] += np.asarray(origin_xy)
            source_fit = source_ids % 5 != 0
            source_holdout = ~source_fit
            source_initial_z = _initial_z_offset(
                source_projected[source_fit], lidar + scene_origin
            )
            source_translation = _translation_icp(
                source_projected[source_fit],
                lidar + scene_origin,
                np.asarray([0.0, 0.0, source_initial_z]),
            )
            source_base = source_projected + source_translation - scene_origin
            source_baseline = {
                "fit": _nearest_metrics(source_base[source_fit], lidar_tree),
                "holdout": _nearest_metrics(source_base[source_holdout], lidar_tree),
            }
            (
                source_candidates,
                source_candidate,
                _source_corrected,
            ) = _similarity_icp_candidates(source_base, source_ids, lidar)

            all_source = apply_sim3(
                np.asarray(
                    [point.xyz for point in source_reconstruction.points3D.values()]
                ),
                np.asarray(old_sim3["rotation"]),
                np.asarray(old_sim3["translation"]),
                float(old_sim3["scale"]),
            )
            all_source[:, :2] += np.asarray(origin_xy)
            all_source += source_translation
            all_source -= scene_origin
            source_inside = contains_xy(
                local_target, all_source[:, 0], all_source[:, 1]
            )
            source_roof = all_source[
                source_inside
                & (all_source[:, 2] >= 9.0)
                & (all_source[:, 2] <= 14.0)
            ]
            source_correction = source_candidate["correction"]
            all_source_corrected = apply_sim3(
                all_source,
                np.asarray(source_correction["rotation"]),
                np.asarray(source_correction["translation_m"]),
                float(source_correction["scale"]),
            )
            source_corrected_inside = contains_xy(
                local_target,
                all_source_corrected[:, 0],
                all_source_corrected[:, 1],
            )
            source_corrected_roof = all_source_corrected[
                source_corrected_inside
                & (all_source_corrected[:, 2] >= 9.0)
                & (all_source_corrected[:, 2] <= 14.0)
            ]
            source_baseline_roof = roof_edge_metrics(source_roof, ridges)
            source_candidate_roof = roof_edge_metrics(source_corrected_roof, ridges)
            source_baseline["roof_control"] = source_baseline_roof
            source_candidate["roof_control"] = source_candidate_roof
            source_reasons: list[str] = []
            if source_baseline["holdout"]["points"] < 30:
                source_reasons.append("fewer than 30 LiDAR holdout points")
            if source_candidate["holdout"]["median_m"] > 1.0:
                source_reasons.append("LiDAR holdout median remains above 1 m")
            if source_candidate["holdout"]["p90_m"] > 2.5:
                source_reasons.append("LiDAR holdout p90 remains above 2.5 m")
            if source_candidate_roof["edge_nearest_p90_m"] > 1.0:
                source_reasons.append("32-edge LiDAR p90 remains above 1 m")
            if source_candidate_roof["edge_coverage_fraction_1m"] < 0.75:
                source_reasons.append("fewer than 24 of 32 roof edges are within 1 m")
            source_component_probe = {
                "status": (
                    "accepted_common_frame"
                    if not source_reasons
                    else "refused_common_frame"
                ),
                "refusal_reasons": source_reasons,
                "model_path": str(source_model_path),
                "resolved_semantic_images": len(source_images),
                "multiview_instances": len(source_instances),
                "shared_measured_tracks": len(
                    {
                        point_id
                        for instance in source_instances
                        for point_id in instance.get("shared_point3d_ids", [])
                    }
                ),
                "building_points": len(source_rows),
                "translation_m": np.asarray(source_translation).tolist(),
                "translation_baseline": source_baseline,
                "full_sim3_candidates": source_candidates,
                "selected_full_sim3_candidate": source_candidate,
            }
        else:
            source_component_probe = {
                "status": "unavailable",
                "reason": "a unique source-model building instance was not found",
            }
    else:
        source_component_probe = {
            "status": "unavailable",
            "reason": "source model path is absent",
        }

    reasons: list[str] = []
    if not inherited_sim3_audit["applicable"]:
        reasons.append("source-model Sim3 is not applicable to the rebuilt anchor gauge")
    if direct_anchor.get("status") != "consistent":
        reasons.append("active anchor has no two-source globally consistent GPS Sim3")
    if candidate["holdout"]["p90_m"] > baseline["holdout"]["p90_m"]:
        reasons.append("full Sim3 correction worsens LiDAR holdout p90")
    if candidate_roof["edge_nearest_p90_m"] is None or candidate_roof[
        "edge_nearest_p90_m"
    ] > 1.0:
        reasons.append("32-edge LiDAR control remains above 1 m")
    if candidate_roof["edge_coverage_fraction_1m"] < 0.75:
        reasons.append("fewer than 24 of 32 LiDAR roof edges have support within 1 m")
    if source_component_probe.get("status") != "accepted_common_frame":
        reasons.append("the isolated source component also fails the common-frame gate")
    status = "accepted_common_frame" if not reasons else "refused_common_frame"

    generated = datetime.now(timezone.utc)
    run_id = f"sim3-roof-audit-{generated.strftime('%Y%m%dT%H%M%SZ')}"
    payload = {
        "contract_version": 1,
        "run_id": run_id,
        "hotel_id": workspace.hotel_id,
        "generated_at": generated.isoformat(),
        "status": status,
        "refusal_reasons": reasons,
        "policy": {
            "gps_sim3_inlier_m": 10.0,
            "component_min_inliers": 8,
            "component_min_inlier_fraction": 0.70,
            "component_min_sources": 2,
            "lidar_holdout": "point3D_id modulo 5",
            "roof_support_height_m": [9.0, 14.0],
            "roof_edge_max_p90_m": 1.0,
            "roof_edge_min_coverage_fraction_1m": 0.75,
            "application": "audit only; never mutates scene geometry",
        },
        "inherited_source_sim3": inherited_sim3_audit,
        "active_anchor_direct_sim3": direct_anchor,
        "components": components,
        "translation_baseline": {**baseline, "roof_control": baseline_roof},
        "full_sim3_candidates": candidate_rows,
        "selected_full_sim3_candidate": candidate,
        "source_component_probe": source_component_probe,
        "dependent_artifacts": [
            {
                "path": "11_conditioning/vertical_registration.json",
                "status": "stale_by_common_frame_refusal",
            },
            {
                "path": "11_conditioning/registered_semantic_support.json",
                "status": "stale_by_common_frame_refusal",
            },
            {
                "path": "11_conditioning/semantic_surfaces.json",
                "status": "stale_by_common_frame_refusal",
            },
            {
                "path": "11_conditioning/viewer_payload.json",
                "status": "stale_by_common_frame_refusal",
                "owner_note": "viewer integration intentionally left to agent 1",
            },
        ],
        "next_action": (
            "rebuild and validate separate source-frame submodels; do not average "
            "combined-0..5 and do not publish surfaces before the 32-edge gate passes"
        ),
        "sources": {
            "semantic_correspondences": str(
                correspondence_path.relative_to(workspace.root)
            ),
            "vertical_registration": str(
                registration_path.relative_to(workspace.root)
            ),
            "registered_semantic_support": str(
                support_path.relative_to(workspace.root)
            ),
            "conditioned_scene": str(scene_path.relative_to(workspace.root)),
            "anchor_model_manifest": str(manifest_path.relative_to(workspace.root)),
            "anchor_selection": str(selection_path.relative_to(workspace.root)),
        },
        "source_digests": {
            str(path.relative_to(workspace.root)): _digest(path)
            for path in required + [manifest_path, selection_path]
        },
        "scene_geometry_applied": False,
    }
    relative = f"11_conditioning/sim3_roof_audit_runs/{run_id}.json"
    payload["versioned_artifact"] = relative
    workspace.write_json(relative, payload)
    path = workspace.write_json("11_conditioning/sim3_roof_audit.json", payload)
    return path, payload


__all__ = [
    "Sim3RoofAuditUnavailable",
    "compose_sim3",
    "roof_edge_metrics",
    "run",
]
