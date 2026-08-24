"""Alignement géospatial de la reconstruction (Lot 2 — P4).

Ce module aligne la reconstruction sparse sur les données géospatiales
existantes (empreinte bâtiment, toiture LiDAR, DTM/DSM) et produit un
`GeoAlignmentManifest`.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from shapely.geometry import Polygon, shape

from .schemas.reconstruction import AlignmentAnchor, GeoAlignmentManifest
from .reconstruction_consensus import resolve_model_dir
from .workspace import Workspace

from .logging import get_logger

log = get_logger("geo-alignment")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_pose_line(line: str) -> bool:
    """Une ligne de pose finit par `CAMERA_ID NAME` ; une ligne d'observations
    n'est faite que de nombres."""
    parts = line.split()
    if len(parts) < 10:
        return False
    try:
        float(parts[-1])
    except ValueError:
        return True
    return False


#: Nombre minimal de correspondances caméra↔GPS pour estimer une Sim(3).
#: Trois suffisent mathématiquement ; en exiger six laisse à RANSAC de quoi
#: écarter les pires relevés sans tomber sous le minimum.
MIN_GPS_CORRESPONDENCES = 6

#: Résidu, en mètres, au-delà duquel une correspondance est tenue pour
#: aberrante. Les relevés de rue sont bons à quelques mètres ; un écart de
#: quinze mètres ne s'explique plus par l'imprécision du GPS.
GPS_INLIER_M = 15.0

#: Tirages RANSAC. Le modèle a sept degrés de liberté et se calcule vite ;
#: deux cents essais couvrent largement les proportions d'aberrants observées.
RANSAC_TRIALS = 200


def _camera_gps(workspace: Workspace) -> dict[str, tuple[float, float]]:
    """Position relevée de chaque prise de vue, par identifiant d'image.

    Ce sont les coordonnées que la source déclare — pas une mesure du site.
    Elles portent l'imprécision d'un GPS de roulage, ce que la robustesse de
    l'estimation prend en charge.
    """
    manifest = workspace.path("00_manifest", "asset_manifest.json")
    if not manifest.is_file():
        return {}
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}

    found: dict[str, tuple[float, float]] = {}
    for asset in payload.get("assets", []):
        lat, lon = asset.get("camera_lat"), asset.get("camera_lon")
        if lat is None or lon is None:
            continue
        identifier = asset.get("id")
        if not identifier:
            continue
        found[str(identifier)] = (float(lat), float(lon))
        local = asset.get("local_path")
        if local:
            # Les centres COLMAP sont indexés par nom de fichier : la même
            # position doit être trouvable par les deux clés.
            found[Path(str(local)).stem] = (float(lat), float(lon))
    return found


def _robust_sim3(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Sim(3) estimée sous RANSAC, avec le masque des correspondances retenues.

    Un seul relevé GPS aberrant — un panorama mal géocodé, une reprise de
    position en tunnel — suffirait à faire pivoter toute la scène. Le modèle
    est donc tiré sur des triplets, puis réajusté sur ses seuls inliers.
    """
    from .geometry_align import umeyama_sim3

    count = len(source)
    best_inliers = np.zeros(count, dtype=bool)
    rng = np.random.default_rng(0)

    for _trial in range(RANSAC_TRIALS):
        pick = rng.choice(count, size=3, replace=False)
        rotation, translation, scale = umeyama_sim3(source[pick], target[pick])
        residual = np.linalg.norm(
            (scale * (rotation @ source.T)).T + translation - target, axis=1
        )
        inliers = residual < GPS_INLIER_M
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers

    if best_inliers.sum() < 3:
        # Aucun consensus : l'identité, dite comme telle, vaut mieux qu'une
        # rotation tirée d'un tirage malheureux.
        return np.eye(3), np.zeros(3), 1.0, best_inliers

    rotation, translation, scale = umeyama_sim3(
        source[best_inliers], target[best_inliers]
    )
    return rotation, translation, scale, best_inliers


def _load_colmap_camera_centers(run_dir: Path) -> dict[str, np.ndarray]:
    images_file = run_dir / "normalized" / "images"
    if not images_file.is_file():
        return {}
    # COLMAP écrit **deux** lignes par image : la pose, puis les
    # observations « X Y POINT3D_ID … ». Lire toutes les lignes prenait ces
    # observations pour des poses — autant de caméras fantômes.
    _lines = [
        line for line in images_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if any(not _is_pose_line(line) for line in _lines):
        _lines = _lines[::2]

    centers: dict[str, np.ndarray] = {}
    for line in _lines:
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
        centers[Path(name).stem] = center
    return centers


def _load_building_footprint(workspace: Workspace) -> Polygon | None:
    try:
        spatial = workspace.read_spatial()
        if spatial is None or not spatial.confirmed_building_id:
            return None
        building = spatial.candidate(spatial.confirmed_building_id)
        if building is None or not building.wkt:
            return None
        return shape(json.loads(building.wkt))
    except Exception:
        return None


def _footprint_stats(poly: Polygon) -> tuple[np.ndarray, float, float]:
    minx, miny, maxx, maxy = poly.bounds
    centroid = np.array([(minx + maxx) / 2, (miny + maxy) / 2])
    width = maxx - minx
    height = maxy - miny
    size = float(np.linalg.norm([width, height]))
    return centroid, size, max(width, 1e-6), max(height, 1e-6)


def _recon_stats(centers: dict[str, np.ndarray]) -> tuple[np.ndarray, float, float, float]:
    if not centers:
        return np.zeros(3), 0.0, 0.0, 0.0
    pts = np.array(list(centers.values()))
    centroid = pts.mean(axis=0)
    xy_spread = float(np.linalg.norm(pts[:, :2].max(axis=0) - pts[:, :2].min(axis=0)))
    z_median = float(np.median(pts[:, 2]))
    max_z = float(pts[:, 2].max())
    return centroid, xy_spread, z_median, max_z


def _load_lidar_roof_height(workspace: Workspace) -> float | None:
    try:
        lidar_path = workspace.path("06_geo", "lidar_discovery.json")
        if not lidar_path.is_file():
            return None
        data = json.loads(lidar_path.read_text("utf-8"))
        tiles = data.get("tiles", [])
        if not tiles:
            return None
        tile = tiles[0]
        return tile.get("roof_height_m") or tile.get("max_height_m")
    except Exception:
        return None


class GeoAligner:
    """Aligne une reconstruction sur les ancres géospatiales."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def align(
        self,
        reconstruction_run_id: str,
        *,
        anchors: list[AlignmentAnchor] | None = None,
    ) -> GeoAlignmentManifest:
        if anchors is None:
            anchors = [AlignmentAnchor.FOOTPRINT, AlignmentAnchor.LIDAR_ROOF]

        alignment_id = (
            f"align-{reconstruction_run_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )

        # Charger les centres de caméra de la reconstruction
        run_json_path = self.workspace.path("07_reconstruction", "runs", f"{reconstruction_run_id}.json")
        if not run_json_path.is_file():
            raise FileNotFoundError(
                f"run de reconstruction introuvable : {run_json_path}; "
                "alignement géographique refusé"
            )

        run_data = json.loads(run_json_path.read_text("utf-8"))
        output_path = run_data.get("output_path")
        if not output_path:
            raise ValueError(f"run {reconstruction_run_id} sans output_path")

        run_dir = resolve_model_dir(output_path)

        normalized_dir = run_dir / "normalized"
        if not normalized_dir.is_dir():
            normalized_dir = run_dir.parent / "normalized"
        if not normalized_dir.is_dir():
            normalized_dir = run_dir.parent.parent / "normalized"
        if not normalized_dir.is_dir():
            raise ValueError(f"aucun répertoire normalisé dans {run_dir}")

        centers = _load_colmap_camera_centers(normalized_dir.parent)
        if not centers:
            raise ValueError(f"aucun centre de caméra dans {normalized_dir.parent}")

        recon_centroid, recon_xy_spread, recon_z_median, recon_z_max = _recon_stats(centers)

        footprint = _load_building_footprint(self.workspace)
        if footprint is None:
            raise ValueError("empreinte bâtiment introuvable pour l'alignement")

        footprint_centroid, footprint_size, footprint_width, footprint_height = _footprint_stats(footprint)

        # Premier choix : les positions relevées des prises de vue. Elles
        # donnent une correspondance point à point, donc une rotation et une
        # échelle **mesurées**. Comparer une étendue de nuage à une taille
        # d'emprise ne donnait qu'un ordre de grandeur, et supposait les axes
        # déjà alignés — ce que rien n'attestait.
        gps = _camera_gps(self.workspace)
        source_pts, target_pts, matched = [], [], []
        if gps:
            from pyproj import Transformer

            to_projected = Transformer.from_crs(
                "EPSG:4326", self._working_crs(), always_xy=True
            )
            for name, centre in centers.items():
                found = gps.get(name)
                if found is None:
                    continue
                east, north = to_projected.transform(found[1], found[0])
                source_pts.append(centre)
                target_pts.append([east, north, centre[2]])
                matched.append(name)

        alignment_method = "footprint_extent"
        inlier_count = 0
        if len(source_pts) >= MIN_GPS_CORRESPONDENCES:
            rotation, translation, scale_est, inliers = _robust_sim3(
                np.asarray(source_pts, dtype=np.float64),
                np.asarray(target_pts, dtype=np.float64),
            )
            inlier_count = int(inliers.sum())
            if inlier_count >= 3:
                alignment_method = "camera_gps_sim3"
                R = rotation.tolist()
                scale = float(scale_est)
                t_xy = np.asarray(translation[:2], dtype=np.float64)
                t_z = float(translation[2])
                log.info(
                    "alignement Sim(3) sur %d/%d position(s) de caméra, "
                    "échelle %.4f",
                    inlier_count,
                    len(source_pts),
                    scale,
                )

        if alignment_method == "footprint_extent":
            # Repli assumé : sans positions exploitables, on retombe sur la
            # comparaison d'étendues. Elle ne mesure ni rotation ni position,
            # et le rapport doit le dire plutôt que de laisser croire à une
            # géométrie établie.
            log.warning(
                "aucune correspondance GPS exploitable (%d appariée(s)) : "
                "alignement approximatif, rotation supposée identité",
                len(source_pts),
            )
            scale = footprint_size / max(recon_xy_spread, 1e-6)
            scale = max(scale, 1e-6)
            R = np.eye(3).tolist()
            t_xy = footprint_centroid - scale * recon_centroid[:2]
            t_z = 0.0

        # Ancrage Z via LiDAR si disponible. Il ne s'applique qu'au repli :
        # une Sim(3) estimée porte déjà sa composante verticale, et l'écraser
        # reviendrait à défaire ce qui vient d'être mesuré.
        lidar_roof_z = _load_lidar_roof_height(self.workspace)
        if lidar_roof_z is not None and alignment_method == "footprint_extent":
            t_z = lidar_roof_z - recon_z_median

        t = [float(t_xy[0]), float(t_xy[1]), float(t_z)]

        # Évaluation des erreurs après alignement
        aligned_centers = {}
        for aid, c in centers.items():
            aligned = scale * (c @ np.eye(3).T) + np.array(t)
            aligned_centers[aid] = aligned

        aligned_pts = np.array(list(aligned_centers.values()))
        footprint_error = float(np.mean([
            Polygon(aligned_pts[:, :2]).distance(footprint.boundary)
        ])) if len(aligned_pts) >= 3 else footprint_size * 0.1

        roof_height_error = abs(lidar_roof_z - recon_z_median) if lidar_roof_z is not None else footprint_size * 0.05
        alignment_rmse = math.sqrt(footprint_error**2 + roof_height_error**2)

        return GeoAlignmentManifest(
            alignment_id=alignment_id,
            source_reconstruction_id=reconstruction_run_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            scale=round(float(scale), 6),
            rotation=R,
            translation={"x": round(float(t[0]), 4), "y": round(float(t[1]), 4), "z": round(float(t[2]), 4)},
            horizontal_crs=self._working_crs(),
            vertical_reference=self._vertical_reference(),
            footprint_error_m=round(footprint_error, 3),
            roof_height_error_m=round(roof_height_error, 3),
            alignment_rmse_m=round(alignment_rmse, 3),
            anchors=[a.value for a in anchors],
        )

    def _working_crs(self) -> str:
        try:
            spatial = self.workspace.read_spatial()
            return spatial.working_crs if spatial else "EPSG:4326"
        except Exception:
            return "EPSG:4326"

    def _vertical_reference(self) -> str | None:
        return "ellipsoidal"


def publish_alignment(manifest: GeoAlignmentManifest, workspace: Workspace) -> Path:
    """Publie le manifeste d'alignement sous `07_reconstruction/alignment/`."""
    path = workspace.path("07_reconstruction", "alignment", f"{manifest.alignment_id}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n")
    return path


__all__ = [
    "GeoAligner",
    "publish_alignment",
]
