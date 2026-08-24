"""Audit mesure de l'enregistrement du noyau COLMAP sur le LiDAR.

Le GPS fournit l'alignement horizontal initial, mais aucune altitude camera
fiable. Ce module estime une translation 3D, la controle sur des points COLMAP
ecartes de l'ajustement et la compare a des translations negatives. Un
resultat ambigu est publie comme diagnostic refuse, jamais applique a la scene.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..geometry_align import apply_sim3
from ..workspace import Workspace
from .heights import find_laz
from .laz_cache import read_window
from .scene import load_scene
from .semantic_correspondence import (
    SemanticCorrespondenceUnavailable,
    _resolve_model_path,
)


class VerticalRegistrationUnavailable(RuntimeError):
    """Les mesures necessaires a l'audit d'enregistrement sont absentes."""


def _read(path: Path) -> dict:
    return json.loads(path.read_text("utf-8"))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assess_metrics(
    *,
    fit_points: int,
    holdout_points: int,
    holdout_support_fraction_1m: float,
    holdout_median_m: float,
    holdout_p90_m: float,
    best_negative_support_fraction_1m: float,
) -> tuple[str, list[str]]:
    """Applique un gate conservateur, independant de l'optimiseur."""
    reasons: list[str] = []
    if fit_points < 80:
        reasons.append(f"fit points below threshold: {fit_points} < 80")
    if holdout_points < 30:
        reasons.append(f"holdout points below threshold: {holdout_points} < 30")
    if holdout_support_fraction_1m < 0.45:
        reasons.append(
            "holdout support within 1 m below threshold: "
            f"{holdout_support_fraction_1m:.3f} < 0.450"
        )
    if holdout_median_m > 1.0:
        reasons.append(f"holdout median residual above threshold: {holdout_median_m:.3f} > 1.000 m")
    if holdout_p90_m > 2.5:
        reasons.append(f"holdout p90 residual above threshold: {holdout_p90_m:.3f} > 2.500 m")
    margin = holdout_support_fraction_1m - best_negative_support_fraction_1m
    if margin < 0.08:
        reasons.append(f"negative-control margin below threshold: {margin:.3f} < 0.080")
    return ("accepted" if not reasons else "refused", reasons)


def _voxel_downsample(points: np.ndarray, voxel_m: float = 0.15) -> np.ndarray:
    cells = np.floor(points / voxel_m).astype(np.int64)
    _unique, indices = np.unique(cells, axis=0, return_index=True)
    return points[np.sort(indices)]


def _initial_z_offset(
    source: np.ndarray,
    lidar: np.ndarray,
    *,
    radius_m: float = 3.0,
) -> float:
    from scipy.spatial import cKDTree

    tree = cKDTree(lidar[:, :2])
    distances, indices = tree.query(
        source[:, :2], k=32, distance_upper_bound=radius_m, workers=-1
    )
    valid = np.isfinite(distances) & (indices < len(lidar))
    safe = np.minimum(indices, len(lidar) - 1)
    differences = (lidar[safe, 2] - source[:, None, 2])[valid]
    if len(differences) < 100:
        raise VerticalRegistrationUnavailable(
            "moins de 100 couples horizontaux COLMAP/LiDAR"
        )
    edges = np.arange(-20.0, 70.2, 0.2)
    histogram, _ = np.histogram(differences, bins=edges)
    best = int(np.argmax(histogram))
    seed = 0.5 * (edges[best] + edges[best + 1])
    neighbourhood = differences[np.abs(differences - seed) <= 0.5]
    return float(np.median(neighbourhood))


def _translation_icp(
    source: np.ndarray,
    lidar: np.ndarray,
    initial: np.ndarray,
) -> np.ndarray:
    """ICP translation seulement; rotation et echelle restent celles du Sim3."""
    from scipy.spatial import cKDTree

    tree = cKDTree(lidar)
    translation = np.asarray(initial, dtype=float).copy()
    for threshold in (8.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5):
        for _iteration in range(3):
            distances, indices = tree.query(source + translation, k=1, workers=-1)
            selected = distances < threshold
            if int(selected.sum()) < 40:
                break
            deltas = lidar[indices[selected]] - (source[selected] + translation)
            median = np.median(deltas, axis=0)
            deviation = np.linalg.norm(deltas - median, axis=1)
            retained = deltas[deviation <= np.percentile(deviation, 70)]
            translation += np.median(retained, axis=0)
    return translation


def _residual_metrics(source: np.ndarray, lidar_tree: object, translation: np.ndarray) -> dict:
    distances, _indices = lidar_tree.query(source + translation, k=1, workers=-1)
    return {
        "points": len(distances),
        "median_m": round(float(np.median(distances)), 5),
        "p90_m": round(float(np.percentile(distances, 90)), 5),
        "support_fraction_0_5m": round(float(np.mean(distances <= 0.5)), 5),
        "support_fraction_1m": round(float(np.mean(distances <= 1.0)), 5),
        "support_fraction_1_5m": round(float(np.mean(distances <= 1.5)), 5),
    }


def run(workspace: Workspace) -> tuple[Path, dict]:
    """Estime puis audite la translation COLMAP vers le nuage LiDAR."""
    correspondence_path = workspace.path(
        "11_conditioning", "semantic_correspondences.json"
    )
    capture_path = workspace.path("06_geo", "capture_geometry.json")
    if not correspondence_path.is_file() or not capture_path.is_file():
        raise VerticalRegistrationUnavailable(
            "correspondances semantiques ou geometrie de capture absentes"
        )
    correspondence = _read(correspondence_path)
    anchor_manifest_path = workspace.root / correspondence["sources"][
        "anchor_model_manifest"
    ]
    anchor_manifest = _read(anchor_manifest_path)
    selection_id = str(anchor_manifest.get("anchor_selection_id", ""))
    selection_path = workspace.path(
        "07_reconstruction", "anchors", f"{selection_id}.json"
    )
    if not selection_path.is_file():
        raise VerticalRegistrationUnavailable("selection Sim3 du noyau absente")
    selection = _read(selection_path)
    sim3 = selection.get("metrics", {}).get("sim3", {})
    try:
        rotation = np.asarray(sim3["rotation"], dtype=float)
        translation = np.asarray(sim3["translation"], dtype=float)
        scale = float(sim3["scale"])
    except (KeyError, TypeError, ValueError) as exc:
        raise VerticalRegistrationUnavailable("Sim3 geographique incomplet") from exc
    try:
        import pycolmap
        from pyproj import Transformer
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover - extras geo+sfm
        raise VerticalRegistrationUnavailable(
            "les extras `geo` et `sfm` sont requis"
        ) from exc

    try:
        model_path = _resolve_model_path(
            workspace, anchor_manifest_path, anchor_manifest
        )
    except SemanticCorrespondenceUnavailable as exc:
        raise VerticalRegistrationUnavailable(str(exc)) from exc
    reconstruction = pycolmap.Reconstruction(str(model_path))
    building_instances = [
        item
        for item in correspondence.get("instances", [])
        if item.get("class") == "building"
        and int(item.get("validated_view_count", 0)) >= 2
    ]
    if len(building_instances) != 1:
        raise VerticalRegistrationUnavailable(
            "une instance batiment multi-vues unique est requise"
        )
    building_instance = building_instances[0]
    building_point_ids = {
        int(point_id) for point_id in building_instance["shared_point3d_ids"]
    }
    rows = [
        (int(point_id), point)
        for point_id, point in reconstruction.points3D.items()
        if int(point_id) in building_point_ids
        and float(point.error) <= 3.0
        and len(point.track.elements) >= 2
    ]
    if len(rows) < 100:
        raise VerticalRegistrationUnavailable("nuage COLMAP qualifie insuffisant")
    source = apply_sim3(
        np.asarray([point.xyz for _point_id, point in rows]),
        rotation,
        translation,
        scale,
    )
    point_ids = np.asarray([point_id for point_id, _point in rows], dtype=np.int64)
    origin_lat = float(selection["metrics"]["enu_origin_lat"])
    origin_lon = float(selection["metrics"]["enu_origin_lon"])
    scene = load_scene(capture_path)
    projected_crs = scene.crs
    origin_x, origin_y = Transformer.from_crs(
        "EPSG:4326", projected_crs, always_xy=True
    ).transform(origin_lon, origin_lat)
    source[:, 0] += origin_x
    source[:, 1] += origin_y
    in_window = (
        (np.abs(source[:, 0] - scene.centre[0]) <= 100.0)
        & (np.abs(source[:, 1] - scene.centre[1]) <= 100.0)
    )
    source = source[in_window]
    point_ids = point_ids[in_window]

    laz_path = find_laz(workspace, scene.centre)
    if laz_path is None:
        raise VerticalRegistrationUnavailable("tuile LiDAR absente")
    window = read_window(laz_path, scene.centre, 110.0, with_colour=False)
    if window is None:
        raise VerticalRegistrationUnavailable("fenetre LiDAR vide")
    try:
        from shapely import contains_xy
        from shapely.geometry import Polygon
    except ImportError as exc:  # pragma: no cover - dependance principale
        raise VerticalRegistrationUnavailable("Shapely est requis") from exc
    if scene.target is None:
        raise VerticalRegistrationUnavailable("emprise du batiment cible absente")
    target_zone = Polygon(scene.target.footprint).buffer(2.0)
    usable = (window.classification == 6) & contains_xy(
        target_zone, window.x, window.y
    )
    lidar = np.column_stack((window.x[usable], window.y[usable], window.z[usable]))
    lidar = _voxel_downsample(lidar, voxel_m=0.10)
    if len(lidar) < 1_000:
        raise VerticalRegistrationUnavailable(
            "retours LiDAR de classe batiment insuffisants"
        )
    ground = window.classification == 2
    if int(ground.sum()) < 100:
        raise VerticalRegistrationUnavailable(
            "reference altimetrique LiDAR de classe sol insuffisante"
        )
    ground_reference_z = float(np.median(window.z[ground]))

    holdout = point_ids % 5 == 0
    fitting = ~holdout
    initial_z = _initial_z_offset(source[fitting], lidar)
    fitted_translation = _translation_icp(
        source[fitting], lidar, np.asarray([0.0, 0.0, initial_z])
    )
    tree = cKDTree(lidar)
    fit_metrics = _residual_metrics(source[fitting], tree, fitted_translation)
    holdout_metrics = _residual_metrics(source[holdout], tree, fitted_translation)
    controls = {
        "east_plus_10m": np.asarray([10.0, 0.0, 0.0]),
        "east_minus_10m": np.asarray([-10.0, 0.0, 0.0]),
        "north_plus_10m": np.asarray([0.0, 10.0, 0.0]),
        "north_minus_10m": np.asarray([0.0, -10.0, 0.0]),
        "vertical_plus_5m": np.asarray([0.0, 0.0, 5.0]),
        "vertical_minus_5m": np.asarray([0.0, 0.0, -5.0]),
    }
    control_metrics = {
        name: _residual_metrics(source[holdout], tree, fitted_translation + offset)
        for name, offset in controls.items()
    }
    best_negative = max(
        (item["support_fraction_1m"] for item in control_metrics.values()),
        default=0.0,
    )
    status, refusal_reasons = assess_metrics(
        fit_points=fit_metrics["points"],
        holdout_points=holdout_metrics["points"],
        holdout_support_fraction_1m=holdout_metrics["support_fraction_1m"],
        holdout_median_m=holdout_metrics["median_m"],
        holdout_p90_m=holdout_metrics["p90_m"],
        best_negative_support_fraction_1m=best_negative,
    )
    generated_at = datetime.now(timezone.utc)
    run_id = f"colmap-lidar-registration-{generated_at.strftime('%Y%m%dT%H%M%SZ')}"
    payload = {
        "contract_version": 2,
        "run_id": run_id,
        "hotel_id": workspace.hotel_id,
        "generated_at": generated_at.isoformat(),
        "status": status,
        "refusal_reasons": refusal_reasons,
        "policy": {
            "transform_model": "fixed geographic Sim3 plus translation-only ICP",
            "registration_basis": (
                "semantic building tracks shared across validated views "
                "against class-6 LiDAR inside confirmed building footprint"
            ),
            "fit_point_rule": (
                "building-instance COLMAP points, error <= 3 px and track length >= 2"
            ),
            "lidar_rule": (
                "ASPRS class 6 inside confirmed target footprint buffered by 2 m"
            ),
            "holdout_rule": "point3D_id modulo 5 equals zero",
            "minimum_holdout_support_fraction_1m": 0.45,
            "maximum_holdout_median_m": 1.0,
            "maximum_holdout_p90_m": 2.5,
            "minimum_negative_control_margin": 0.08,
            "application_rule": "never apply a refused hypothesis to scene geometry",
        },
        "hypothesis": {
            "translation_projected_m": np.round(fitted_translation, 6).tolist(),
            "initial_vertical_offset_m": round(initial_z, 6),
            "horizontal_crs": projected_crs,
            "source_frame": "anchor_colmap_model transformed by geographic Sim3",
            "target_frame": "qualified LiDAR",
            "semantic_instance_id": building_instance.get("instance_id"),
            "ground_reference_z_m": round(ground_reference_z, 6),
            "scene_origin_projected_xyz": [
                round(float(scene.centre[0]), 6),
                round(float(scene.centre[1]), 6),
                round(ground_reference_z, 6),
            ],
        },
        "metrics": {
            "fit": fit_metrics,
            "holdout": holdout_metrics,
            "negative_controls": control_metrics,
            "best_negative_support_fraction_1m": best_negative,
            "negative_control_margin": round(
                holdout_metrics["support_fraction_1m"] - best_negative, 5
            ),
            "qualified_colmap_points": len(source),
            "voxelized_lidar_points": len(lidar),
        },
        "sources": {
            "semantic_correspondences": str(
                correspondence_path.relative_to(workspace.root)
            ),
            "anchor_selection": str(selection_path.relative_to(workspace.root)),
            "anchor_model_manifest": str(
                anchor_manifest_path.relative_to(workspace.root)
            ),
            "capture_geometry": str(capture_path.relative_to(workspace.root)),
            "lidar": str(laz_path.relative_to(workspace.root)),
        },
        "source_digests": {
            str(path.relative_to(workspace.root)): _digest(path)
            for path in (
                correspondence_path,
                selection_path,
                anchor_manifest_path,
                capture_path,
                laz_path,
            )
        },
        "scene_geometry_applied": False,
        "geometry_3d_created": 0,
    }
    relative = f"11_conditioning/vertical_registration_runs/{run_id}.json"
    payload["versioned_artifact"] = relative
    workspace.write_json(relative, payload)
    path = workspace.write_json("11_conditioning/vertical_registration.json", payload)
    return path, payload


__all__ = ["VerticalRegistrationUnavailable", "assess_metrics", "run"]
