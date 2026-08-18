"""Rapport de support LiDAR pour le Lot 2 — P4.

Ce module analyse la densité et la couverture des points LiDAR
disponibles (toit, sol, façade) pour déterminer si LiDGS / GS-SDF
sont viables.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .schemas.reconstruction import AlignmentAnchor, ReconstructionInputManifest
from .workspace import Workspace


class LiDARSupportReport(BaseModel):
    """Rapport de densité et couverture LiDAR."""

    model_config = ConfigDict(extra="forbid")

    report_id: str
    reconstruction_input_id: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    roof_point_density: float = Field(default=0.0, ge=0.0)
    ground_point_density: float = Field(default=0.0, ge=0.0)
    facade_vertical_point_density: float = Field(default=0.0, ge=0.0)
    coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    classification: str = Field(default="unknown")
    viable_for_lidgs: bool = False

    @property
    def total_point_density(self) -> float:
        return self.roof_point_density + self.ground_point_density + self.facade_vertical_point_density


def _find_lidar_files(workspace: Workspace) -> list[Path]:
    """Trouve les fichiers LiDAR (.las/.laz) dans le workspace."""
    lidar_raw_dir = workspace.path("06_geo", "lidar_raw")
    if not lidar_raw_dir.is_dir():
        return []
    return sorted(lidar_raw_dir.glob("*.la[sz]"))


def _classify_points(
    points: np.ndarray,
    footprint_wkt: str | None = None,
    height_threshold_ratio: float = 0.3,
) -> dict[str, np.ndarray]:
    """Classifie les points LiDAR en roof, ground, facade.

    Args:
        points: array (N, 3) de points 3D
        footprint_wkt: WKT de l'empreinte du bâtiment (optionnel)
        height_threshold_ratio: ratio de la hauteur max pour séparer toit/façade

    Returns:
        dict avec masks booléens pour 'roof', 'ground', 'facade', 'unclassified'
    """
    if len(points) == 0:
        return {
            "roof": np.array([], dtype=bool),
            "ground": np.array([], dtype=bool),
            "facade": np.array([], dtype=bool),
            "unclassified": np.array([], dtype=bool),
        }

    z = points[:, 2]
    z_min = z.min()
    z_max = z.max()
    z_range = z_max - z_min

    if z_range < 1e-6:
        return {
            "roof": np.zeros(len(points), dtype=bool),
            "ground": np.ones(len(points), dtype=bool),
            "facade": np.zeros(len(points), dtype=bool),
            "unclassified": np.zeros(len(points), dtype=bool),
        }

    roof_threshold = z_min + z_range * (1.0 - height_threshold_ratio)
    ground_threshold = z_min + z_range * height_threshold_ratio

    roof = z >= roof_threshold
    ground = z <= ground_threshold
    facade = np.logical_and(z > ground_threshold, z < roof_threshold)

    if footprint_wkt is not None:
        try:
            from shapely.geometry import Polygon, Point
            from shapely import wkt as shapely_wkt
            footprint = shapely_wkt.loads(footprint_wkt)
            if footprint.is_valid and not footprint.is_empty:
                for i, p in enumerate(points):
                    pt = Point(p[0], p[1])
                    if not footprint.contains(pt) and not footprint.touches(pt):
                        if facade[i]:
                            facade[i] = False
                            # Point outside footprint at mid-height → likely noise
        except Exception:
            pass

    unclassified = np.logical_not(np.logical_or(np.logical_or(roof, ground), facade))

    return {"roof": roof, "ground": ground, "facade": facade, "unclassified": unclassified}


def _compute_density(points: np.ndarray, mask: np.ndarray, area_m2: float) -> float:
    """Calcule la densité de points par m²."""
    if area_m2 <= 0 or not np.any(mask):
        return 0.0
    count = int(mask.sum())
    return float(count / area_m2)


def _estimate_facade_area_from_footprint(
    footprint_wkt: str | None,
    height_m: float,
) -> float:
    """Estime la superficie des façades à partir de l'empreinte et la hauteur."""
    if footprint_wkt is None or height_m <= 0:
        return 0.0
    try:
        from shapely.geometry import Polygon
        from shapely import wkt as shapely_wkt
        footprint = shapely_wkt.loads(footprint_wkt)
        if footprint.is_valid and not footprint.is_empty:
            perimeter = footprint.length
            return float(perimeter * height_m)
    except Exception:
        pass
    return 0.0


def _analyze_lidar_tile(laz_path: Path, footprint_wkt: str | None = None) -> dict[str, Any]:
    """Analyse un fichier LiDAR et retourne les métriques de densité."""
    try:
        import laspy

        with laspy.open(str(laz_path)) as reader:
            header = reader.header
            points_data = reader.read()

        xyz = np.column_stack([
            np.asarray(points_data.x, dtype=np.float64),
            np.asarray(points_data.y, dtype=np.float64),
            np.asarray(points_data.z, dtype=np.float64),
        ])

        total_points = len(xyz)
        if total_points == 0:
            return {"point_count": 0, "error": "fichier vide"}

        bounds = {
            "min_x": float(header.mins[0]),
            "min_y": float(header.mins[1]),
            "min_z": float(header.mins[2]),
            "max_x": float(header.maxs[0]),
            "max_y": float(header.maxs[1]),
            "max_z": float(header.maxs[2]),
        }

        footprint_area = (bounds["max_x"] - bounds["min_x"]) * (bounds["max_y"] - bounds["min_y"])
        height_m = bounds["max_z"] - bounds["min_z"]

        classification = _classify_points(xyz, footprint_wkt)
        roof_density = _compute_density(xyz, classification["roof"], footprint_area)
        ground_density = _compute_density(xyz, classification["ground"], footprint_area)
        facade_area = _estimate_facade_area_from_footprint(footprint_wkt, height_m)
        facade_density = _compute_density(xyz, classification["facade"], facade_area)

        roof_count = int(classification["roof"].sum())
        ground_count = int(classification["ground"].sum())
        facade_count = int(classification["facade"].sum())

        return {
            "point_count": total_points,
            "bounds": bounds,
            "footprint_area_m2": footprint_area,
            "height_m": height_m,
            "roof_point_density": roof_density,
            "ground_point_density": ground_density,
            "facade_vertical_point_density": facade_density,
            "roof_point_count": roof_count,
            "ground_point_count": ground_count,
            "facade_point_count": facade_count,
            "unclassified_count": int(classification["unclassified"].sum()),
            "classification_counts": {
                "roof": roof_count,
                "ground": ground_count,
                "facade": facade_count,
                "unclassified": int(classification["unclassified"].sum()),
            },
            "error": None,
        }
    except Exception as exc:
        return {"point_count": 0, "error": str(exc)}


class LiDARSupportAnalyzer:
    """Analyse le support LiDAR pour la reconstruction."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def analyze(
        self,
        input_manifest: ReconstructionInputManifest | None = None,
    ) -> LiDARSupportReport:
        """Analyse la densité LiDAR disponible depuis les fichiers .las/.laz et lidar_discovery.json."""
        report_id = (
            f"lidar-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )

        lidar_files = _find_lidar_files(self.workspace)
        if not lidar_files:
            return LiDARSupportReport(
                report_id=report_id,
                reconstruction_input_id=input_manifest.reconstruction_input_id if input_manifest else None,
                roof_point_density=0.0,
                ground_point_density=0.0,
                facade_vertical_point_density=0.0,
                coverage=0.0,
                classification="no_lidar_files",
                viable_for_lidgs=False,
            )

        footprint_wkt = None
        try:
            spatial = self.workspace.read_spatial()
            if spatial and spatial.confirmed_building_id:
                building = spatial.candidate(spatial.confirmed_building_id)
                if building:
                    footprint_wkt = building.wkt
        except Exception:
            pass

        tile_metrics = []
        for laz_path in lidar_files:
            metric = _analyze_lidar_tile(laz_path, footprint_wkt)
            if metric.get("error") is None:
                tile_metrics.append(metric)

        if not tile_metrics:
            return LiDARSupportReport(
                report_id=report_id,
                reconstruction_input_id=input_manifest.reconstruction_input_id if input_manifest else None,
                roof_point_density=0.0,
                ground_point_density=0.0,
                facade_vertical_point_density=0.0,
                coverage=0.0,
                classification="analysis_failed",
                viable_for_lidgs=False,
            )

        avg_roof = np.mean([m["roof_point_density"] for m in tile_metrics])
        avg_ground = np.mean([m["ground_point_density"] for m in tile_metrics])
        avg_facade = np.mean([m["facade_vertical_point_density"] for m in tile_metrics])
        total_points = sum(m["point_count"] for m in tile_metrics)

        classification = "aerial"
        if avg_facade >= 10.0 and avg_roof >= 5.0:
            classification = "hybrid"
        elif avg_facade >= 10.0:
            classification = "terrestrial"

        coverage = min(1.0, total_points / 100000.0)

        viable = (
            avg_facade >= 10.0
            and coverage >= 0.7
            and classification in ("aerial", "hybrid", "terrestrial")
        )

        return LiDARSupportReport(
            report_id=report_id,
            reconstruction_input_id=input_manifest.reconstruction_input_id if input_manifest else None,
            roof_point_density=float(avg_roof),
            ground_point_density=float(avg_ground),
            facade_vertical_point_density=float(avg_facade),
            coverage=float(coverage),
            classification=classification,
            viable_for_lidgs=viable,
        )


def publish_lidar_report(report: LiDARSupportReport, workspace: Workspace) -> Path:
    """Publie le rapport LiDAR sous `07_reconstruction/lidar/`."""
    output_dir = workspace.path("07_reconstruction", "lidar")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{report.report_id}.json"
    output_path.write_text(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n")
    return output_path


__all__ = [
    "LiDARSupportReport",
    "LiDARSupportAnalyzer",
    "publish_lidar_report",
]
