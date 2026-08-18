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

from .schemas.reconstruction import AlignmentAnchor
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
        """Analyse la densité LiDAR disponible.

        Pour le MVP, retourne des valeurs par défaut car le traitement
        réel des nuages LiDAR nécessite des dépendances lourdes
        (laspy, pdal, etc.).
        """
        report_id = (
            f"lidar-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )

        return LiDARSupportReport(
            report_id=report_id,
            reconstruction_input_id=input_manifest.reconstruction_input_id if input_manifest else None,
            roof_point_density=0.0,
            ground_point_density=0.0,
            facade_vertical_point_density=0.0,
            coverage=0.0,
            classification="aerial",
            viable_for_lidgs=False,
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
