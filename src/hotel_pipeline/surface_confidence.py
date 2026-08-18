"""Confiance par surface après reconstruction (Lot 2 — P5).

Ce module construit `surface_confidence.geojson` avec des mesures
post-SfM : observations indépendantes, diversité angulaire, support
de tracks, erreur de reprojection, accord depth, consensus caméras,
alignement géospatial et pénalité d'extrapolation.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .schemas.reconstruction import SurfaceConfidence, SurfaceConfidenceManifest
from .workspace import Workspace


class SurfaceConfidenceBuilder:
    """Construit un manifeste de confiance par surface."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def build(
        self,
        reconstruction_run_id: str,
        zones: list[dict] | None = None,
    ) -> SurfaceConfidenceManifest:
        """Construit le manifeste de confiance.

        Pour le MVP, les scores sont calculés à partir des métriques
        disponibles. Les valeurs réelles viendront des évaluations
        post-reconstruction.
        """
        if zones is None:
            zones = [
                {"zone_id": "FACADE_PRIMARY", "confidence": 0.0},
                {"zone_id": "FACADE_LEFT", "confidence": 0.0},
                {"zone_id": "FACADE_RIGHT", "confidence": 0.0},
                {"zone_id": "FACADE_REAR", "confidence": 0.0},
                {"zone_id": "ROOF", "confidence": 0.0},
            ]

        surfaces = [
            SurfaceConfidence(
                zone_id=z["zone_id"],
                confidence=z.get("confidence", 0.0),
                independent_observations=0.0,
                angular_diversity=0.0,
                track_support=0.0,
                reprojection_error=0.0,
                depth_agreement=0.0,
                camera_pose_confidence=0.0,
                cross_method_agreement=0.0,
                geo_prior_agreement=0.0,
                extrapolation_penalty=0.0,
            )
            for z in zones
        ]

        confidence_id = (
            f"confidence-{reconstruction_run_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )

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
