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


class LiDARSupportAnalyzer:
    """Analyse le support LiDAR pour la reconstruction."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def analyze(
        self,
        input_manifest: ReconstructionInputManifest | None = None,
    ) -> LiDARSupportReport:
        """Analyse la densité LiDAR disponible depuis lidar_discovery.json."""
        report_id = (
            f"lidar-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )

        lidar_path = self.workspace.path("06_geo", "lidar_discovery.json")
        if not lidar_path.is_file():
            return LiDARSupportReport(
                report_id=report_id,
                reconstruction_input_id=input_manifest.reconstruction_input_id if input_manifest else None,
                roof_point_density=0.0,
                ground_point_density=0.0,
                facade_vertical_point_density=0.0,
                coverage=0.0,
                classification="unknown",
                viable_for_lidgs=False,
            )

        data = json.loads(lidar_path.read_text("utf-8"))
        tiles = data.get("tiles", [])
        if not tiles:
            return LiDARSupportReport(
                report_id=report_id,
                reconstruction_input_id=input_manifest.reconstruction_input_id if input_manifest else None,
                roof_point_density=0.0,
                ground_point_density=0.0,
                facade_vertical_point_density=0.0,
                coverage=0.0,
                classification=data.get("coverage", "unknown"),
                viable_for_lidgs=False,
            )

        tile = tiles[0]
        roof_density = tile.get("roi_density_ppm2", tile.get("roof_point_density", 0.0))
        ground_density = tile.get("ground_density_ppm2", tile.get("ground_point_density", 0.0))
        facade_density = tile.get("vertical_density_ppm2", tile.get("facade_vertical_point_density", 0.0))
        coverage = tile.get("coverage_fraction", tile.get("coverage", 0.0))
        classification = tile.get("classification", data.get("coverage", "aerial"))

        if isinstance(coverage, str):
            coverage = 0.0

        viable = (
            facade_density >= 10.0
            and coverage >= 0.7
            and classification in ("aerial", "hybrid")
        )

        return LiDARSupportReport(
            report_id=report_id,
            reconstruction_input_id=input_manifest.reconstruction_input_id if input_manifest else None,
            roof_point_density=float(roof_density or 0.0),
            ground_point_density=float(ground_density or 0.0),
            facade_vertical_point_density=float(facade_density or 0.0),
            coverage=float(coverage or 0.0),
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
