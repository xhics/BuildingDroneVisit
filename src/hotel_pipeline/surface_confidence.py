"""Confiance par surface après reconstruction (Lot 2 — P5).

Ce module construit `surface_confidence.geojson` avec des mesures
post-SfM : observations indépendantes, diversité angulaire, support
de tracks, erreur de reprojection, accord depth, consensus caméras,
alignement géospatial et pénalité d'extrapolation.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from shapely.geometry import Point, Polygon

from .schemas.reconstruction import SurfaceConfidence, SurfaceConfidenceManifest
from .workspace import Workspace


# ---------------------------------------------------------------------------
# Helpers COLMAP
# ---------------------------------------------------------------------------

_FACADE_BY_SECTOR = {
    "front": "FACADE_PRIMARY",
    "front_left_corner": "FACADE_PRIMARY",
    "left": "FACADE_LEFT",
    "rear_left_corner": "FACADE_REAR",
    "rear": "FACADE_REAR",
    "rear_right_corner": "FACADE_REAR",
    "right": "FACADE_RIGHT",
    "front_right_corner": "FACADE_PRIMARY",
    "unknown": "ROOF",
}


def _parse_images_txt(path: Path) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Retourne (centers, headings) par asset_id."""
    if not path.is_file():
        return {}, {}
    centers: dict[str, np.ndarray] = {}
    headings: dict[str, float] = {}
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 9:
            continue
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        name = parts[9] if len(parts) > 9 else parts[8]
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)],
        ])
        t = np.array([tx, ty, tz])
        center = -R.T @ t
        asset_id = Path(name).stem
        centers[asset_id] = center
        # Heading depuis la rotation de la caméra (troisième colonne de R, projetée sur XY)
        heading = math.degrees(math.atan2(R[0, 2], R[1, 2])) % 360.0
        headings[asset_id] = heading
    return centers, headings


def _parse_points3d_txt(path: Path) -> tuple[list[np.ndarray], list[float], dict[int, set[str]]]:
    """Retourne (points, errors, tracks) où tracks est {point3d_id: {image_name}}."""
    if not path.is_file():
        return [], [], {}
    points: list[np.ndarray] = []
    errors: list[float] = []
    tracks: dict[int, set[str]] = {}
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        point_id = int(parts[0])
        x, y, z = map(float, parts[1:4])
        error = float(parts[7])
        # Le reste est la track : paires IMAGE_ID POINT2D_IDX
        track_parts = parts[8:]
        image_ids = set()
        for i in range(0, len(track_parts), 2):
            if i + 1 < len(track_parts):
                image_ids.add(track_parts[i])
        points.append(np.array([x, y, z]))
        errors.append(error)
        tracks[point_id] = image_ids
    return points, errors, tracks


def _asset_to_zone(asset_id: str, workspace: Workspace) -> str:
    try:
        assets = workspace.read_assets()
        if assets is None:
            return "ROOF"
        by_id = {a.id: a for a in assets.assets}
        asset = by_id.get(asset_id)
        if asset is None or asset.view_sector is None:
            return "ROOF"
        sector = asset.view_sector.value
        return _FACADE_BY_SECTOR.get(sector, "ROOF")
    except Exception:
        return "ROOF"


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class SurfaceConfidenceBuilder:
    """Construit un manifeste de confiance par surface."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def build(
        self,
        reconstruction_run_id: str,
        zones: list[str] | None = None,
    ) -> SurfaceConfidenceManifest:
        if zones is None:
            zones = ["FACADE_PRIMARY", "FACADE_LEFT", "FACADE_RIGHT", "FACADE_REAR", "ROOF"]

        run_json_path = self.workspace.path("07_reconstruction", "runs", f"{reconstruction_run_id}.json")
        if not run_json_path.is_file():
            raise FileNotFoundError(f"run introuvable : {reconstruction_run_id}")

        run_data = json.loads(run_json_path.read_text("utf-8"))
        output_path = run_data.get("output_path")
        if not output_path:
            raise ValueError(f"run {reconstruction_run_id} sans output_path")

        run_dir = Path(output_path)
        if not run_dir.is_dir():
            run_dir = run_dir.parent
        norm = run_dir / "normalized"

        centers, headings = _parse_images_txt(norm / "images")
        points, errors, tracks = _parse_points3d_txt(norm / "points3D")

        # Construire la map image_name -> asset_id (COLMAP utilise le nom de fichier)
        image_name_to_asset: dict[str, str] = {}
        if centers:
            for asset_id in centers:
                image_name_to_asset[f"{asset_id}.jpg"] = asset_id
                image_name_to_asset[f"{asset_id}.png"] = asset_id

        # Parcourir les images COLMAP pour récupérer les vrais noms
        images_file = norm / "images"
        if images_file.is_file():
            for line in images_file.read_text().splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 10:
                    img_name = parts[9]
                elif len(parts) >= 9:
                    img_name = parts[8]
                else:
                    continue
                stem = Path(img_name).stem
                if stem in centers:
                    image_name_to_asset[img_name] = stem

        # Assigner les points aux zones
        zone_points: dict[str, list[np.ndarray]] = {z: [] for z in zones}
        zone_errors: dict[str, list[float]] = {z: [] for z in zones}
        zone_track_lengths: dict[str, list[int]] = {z: [] for z in zones}
        zone_images: dict[str, set[str]] = {z: set() for z in zones}

        for pt, err, (pt_id, img_ids) in zip(points, errors, tracks.items()):
            zone_counts: dict[str, int] = {z: 0 for z in zones}
            for img_id in img_ids:
                img_name = f"{img_id}.jpg" if not img_id.endswith((".jpg", ".png")) else img_id
                asset_id = image_name_to_asset.get(img_name)
                if asset_id is None:
                    # essayer par stem
                    asset_id = image_name_to_asset.get(Path(img_name).stem, "")
                zone = _asset_to_zone(asset_id, self.workspace) if asset_id else "ROOF"
                if zone in zone_counts:
                    zone_counts[zone] += 1
            # Assigner le point à la zone majoritaire
            best_zone = max(zone_counts, key=zone_counts.get) if any(zone_counts.values()) else "ROOF"
            if best_zone in zone_points:
                zone_points[best_zone].append(pt)
                zone_errors[best_zone].append(err)
                zone_track_lengths[best_zone].append(max(zone_counts.values()))
                for img_id in img_ids:
                    img_name = f"{img_id}.jpg" if not img_id.endswith((".jpg", ".png")) else img_id
                    asset_id = image_name_to_asset.get(img_name)
                    if asset_id is None:
                        asset_id = image_name_to_asset.get(Path(img_name).stem, "")
                    if asset_id:
                        z = _asset_to_zone(asset_id, self.workspace)
                        if z in zone_images:
                            zone_images[z].add(asset_id)

        # Géométrie du bâtiment pour geo_prior_agreement
        footprint = None
        try:
            from .geo_alignment import _load_building_footprint
            footprint = _load_building_footprint(self.workspace)
        except Exception:
            pass

        surfaces = []
        confidence_id = (
            f"confidence-{reconstruction_run_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )

        for zone_id in zones:
            obs = len(zone_images.get(zone_id, set()))
            obs_score = min(obs / 8.0, 1.0) if obs > 0 else 0.0

            # Diversité angulaire
            zone_headings = [headings[aid] for aid in zone_images.get(zone_id, set()) if aid in headings]
            if len(zone_headings) >= 2:
                ang_std = np.std(np.array(zone_headings))
                ang_score = min(ang_std / 90.0, 1.0)
            else:
                ang_score = 0.0

            # Track support
            tracks_list = zone_track_lengths.get(zone_id, [])
            track_score = min(float(np.mean(tracks_list)) / 5.0, 1.0) if tracks_list else 0.0

            # Reprojection error (inverse : plus bas = mieux)
            errs = zone_errors.get(zone_id, [])
            if errs:
                mean_err = float(np.mean(errs))
                reproj_score = max(0.0, 1.0 - mean_err / 5.0)
            else:
                reproj_score = 0.0

            # Depth agreement : variance des Z des points
            pts_z = [float(p[2]) for p in zone_points.get(zone_id, [])]
            if len(pts_z) >= 2:
                depth_std = float(np.std(pts_z))
                depth_score = max(0.0, 1.0 - depth_std / 10.0)
            else:
                depth_score = 0.0

            # Camera pose confidence
            cam_conf = min(obs_score * 0.5 + track_score * 0.5, 1.0)

            # Cross-method agreement (MVP : neutre)
            cross_method = 0.5

            # Geo prior agreement : distance moyenne des points à l'empreinte
            geo_score = 0.5
            if footprint is not None and zone_points.get(zone_id):
                zone_pts = np.array(zone_points[zone_id])
                if zone_pts.shape[0] > 0 and footprint.is_valid:
                    try:
                        dists = [footprint.exterior.distance(Point(x, y)) for x, y in zone_pts[:, :2]]
                        mean_dist = float(np.mean(dists))
                        geo_score = max(0.0, 1.0 - mean_dist / 20.0)
                    except Exception:
                        pass

            # Extrapolation penalty : les zones avec peu d'observations sont plus extrapolées
            extrap = max(0.0, 1.0 - obs_score) if zone_id != "ROOF" else 0.3

            # Confiance globale
            components = np.array([obs_score, ang_score, track_score, reproj_score, depth_score, cam_conf, geo_score])
            confidence = float(np.mean(components)) * (1.0 - extrap * 0.5)

            surfaces.append(SurfaceConfidence(
                zone_id=zone_id,
                confidence=round(confidence, 3),
                independent_observations=round(obs_score, 3),
                angular_diversity=round(ang_score, 3),
                track_support=round(track_score, 3),
                reprojection_error=round(reproj_score, 3),
                depth_agreement=round(depth_score, 3),
                camera_pose_confidence=round(cam_conf, 3),
                cross_method_agreement=round(cross_method, 3),
                geo_prior_agreement=round(geo_score, 3),
                extrapolation_penalty=round(extrap, 3),
            ))

        return SurfaceConfidenceManifest(
            confidence_id=confidence_id,
            reconstruction_run_id=reconstruction_run_id,
            surfaces=surfaces,
        )


def publish_surface_confidence(
    manifest: SurfaceConfidenceManifest,
    workspace: Workspace,
) -> Path:
    """Publie le manifeste de confiance sous `07_reconstruction/confidence/`."""
    output_dir = workspace.path("07_reconstruction", "confidence")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{manifest.confidence_id}.geojson"
    _write_geojson(output_path, manifest)
    return output_path


def _write_geojson(path: Path, manifest: SurfaceConfidenceManifest) -> None:
    features = []
    for surface in manifest.surfaces:
        features.append({
            "type": "Feature",
            "id": surface.zone_id,
            "properties": {
                "zone_id": surface.zone_id,
                "confidence": surface.confidence,
                "independent_observations": surface.independent_observations,
                "angular_diversity": surface.angular_diversity,
                "track_support": surface.track_support,
                "reprojection_error": surface.reprojection_error,
                "depth_agreement": surface.depth_agreement,
                "camera_pose_confidence": surface.camera_pose_confidence,
                "cross_method_agreement": surface.cross_method_agreement,
                "geo_prior_agreement": surface.geo_prior_agreement,
                "extrapolation_penalty": surface.extrapolation_penalty,
            },
        })
    geojson = {
        "type": "FeatureCollection",
        "name": "surface_confidence",
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": features,
    }
    path.write_text(json.dumps(geojson, indent=2, ensure_ascii=False) + "\n")


__all__ = [
    "SurfaceConfidenceBuilder",
    "publish_surface_confidence",
]
