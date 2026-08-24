"""Reconstruction bornee de surfaces planes depuis les supports enregistres.

Une surface n'est publiee que si un plan ajuste sur une partie des points
explique aussi des points tenus hors ajustement. L'etendue est le convexe des
seuls points COLMAP mesures; aucune epaisseur ni continuation occultee n'est
ajoutee.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..workspace import Workspace


PLANAR_CLASSES = frozenset(
    {
        "road_sign",
        "sign",
        "window",
        "door",
        "beam",
        "column",
        "canopy",
        "gutter",
        "balcony",
    }
)


class SemanticSurfaceUnavailable(RuntimeError):
    """Les supports enregistres requis sont absents."""


def _read(path: Path) -> dict:
    return json.loads(path.read_text("utf-8"))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centre = points.mean(axis=0)
    _u, singular, basis = np.linalg.svd(points - centre, full_matrices=False)
    normal = basis[-1]
    first_nonzero = next((value for value in normal if abs(value) > 1e-9), 1.0)
    if first_nonzero < 0:
        normal = -normal
        basis[-1] = normal
    residuals = np.abs((points - centre) @ normal)
    return centre, basis, singular, residuals


def fit_planar_candidate(point_ids: list[int], points: np.ndarray) -> dict:
    """Ajuste, controle puis reconstruit un convexe plan mesuré."""
    points = np.asarray(points, dtype=float)
    ids = np.asarray(point_ids, dtype=np.int64)
    reasons: list[str] = []
    if len(points) < 8 or points.shape != (len(points), 3):
        return {
            "status": "refused",
            "refusal_reasons": ["at least 8 measured points are required"],
        }
    holdout = ids % 5 == 0
    if int(holdout.sum()) < 2 or int((~holdout).sum()) < 6:
        order = np.argsort(ids)
        holdout = np.zeros(len(points), dtype=bool)
        holdout[order[::5]] = True
    fit = points[~holdout]
    control = points[holdout]
    centre, basis, singular, fit_residuals = _plane(fit)
    normal = basis[-1]
    control_residuals = np.abs((control - centre) @ normal)
    planarity_ratio = float(singular[-1] / max(singular[-2], 1e-9))
    fit_p90 = float(np.percentile(fit_residuals, 90))
    control_p90 = float(np.percentile(control_residuals, 90))
    if planarity_ratio > 0.45:
        reasons.append(f"fit planarity ratio above threshold: {planarity_ratio:.3f} > 0.450")
    if fit_p90 > 0.35:
        reasons.append(f"fit p90 residual above threshold: {fit_p90:.3f} > 0.350 m")
    if control_p90 > 0.45:
        reasons.append(
            f"holdout p90 residual above threshold: {control_p90:.3f} > 0.450 m"
        )
    if abs(float(normal[2])) > 0.35:
        reasons.append(
            f"surface is not sufficiently vertical: |normal_z|={abs(float(normal[2])):.3f}"
        )
    metrics = {
        "measured_points": len(points),
        "fit_points": len(fit),
        "holdout_points": len(control),
        "planarity_ratio": round(planarity_ratio, 5),
        "fit_residual_p90_m": round(fit_p90, 5),
        "holdout_residual_p90_m": round(control_p90, 5),
    }
    if reasons:
        return {
            "status": "refused",
            "refusal_reasons": reasons,
            "metrics": metrics,
        }

    # Le controle est passe; le plan final et son etendue utilisent alors tous
    # les points mesures, sans extrapolation au-dela de leur convexe.
    centre, basis, singular, residuals = _plane(points)
    coordinates = (points - centre) @ basis[:2].T
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(coordinates)
    except (ImportError, ValueError) as exc:
        return {
            "status": "refused",
            "refusal_reasons": [f"planar convex hull unavailable: {exc}"],
            "metrics": metrics,
        }
    boundary = coordinates[hull.vertices]
    vertices = centre + boundary @ basis[:2]
    area = 0.5 * abs(
        float(
            np.dot(boundary[:, 0], np.roll(boundary[:, 1], 1))
            - np.dot(boundary[:, 1], np.roll(boundary[:, 0], 1))
        )
    )
    extents = np.ptp(boundary, axis=0)
    if area < 0.10 or area > 30.0 or float(extents.max()) > 10.0:
        return {
            "status": "refused",
            "refusal_reasons": [
                f"measured planar extent is implausible: area={area:.3f} m2, max={extents.max():.3f} m"
            ],
            "metrics": {**metrics, "area_m2": round(area, 5)},
        }
    faces = [[0, index, index + 1] for index in range(1, len(vertices) - 1)]
    metrics.update(
        {
            "all_point_residual_p90_m": round(float(np.percentile(residuals, 90)), 5),
            "area_m2": round(area, 5),
            "extent_u_m": round(float(extents[0]), 5),
            "extent_v_m": round(float(extents[1]), 5),
        }
    )
    return {
        "status": "accepted",
        "refusal_reasons": [],
        "metrics": metrics,
        "surface": {
            "type": "planar_measured_convex_hull",
            "vertices": np.round(vertices, 5).tolist(),
            "faces": faces,
            "normal": np.round(basis[-1], 6).tolist(),
            "thickness_m": None,
        },
    }


def run(workspace: Workspace) -> tuple[Path, dict]:
    support_path = workspace.path(
        "11_conditioning", "registered_semantic_support.json"
    )
    if not support_path.is_file():
        raise SemanticSurfaceUnavailable("supports semantiques enregistres absents")
    support = _read(support_path)
    if support.get("status") != "registered_support_ready":
        raise SemanticSurfaceUnavailable("supports semantiques non prets")

    audits: list[dict] = []
    surfaces: list[dict] = []
    for instance in support.get("instances", []):
        object_class = str(instance.get("class"))
        if object_class not in PLANAR_CLASSES:
            continue
        points = instance.get("points", [])
        result = fit_planar_candidate(
            [int(item["point3d_id"]) for item in points],
            np.asarray([item["xyz"] for item in points], dtype=float),
        )
        audit = {
            "instance_id": instance.get("instance_id"),
            "class": object_class,
            **result,
        }
        audits.append(audit)
        if result["status"] != "accepted":
            continue
        surfaces.append(
            {
                "surface_id": f"surface-{instance['instance_id']}",
                "instance_id": instance.get("instance_id"),
                "class": object_class,
                "coordinate_frame": "conditioned_scene_local_ground",
                "provenance_class": "SEMANTICALLY_CONSTRAINED",
                "support_provenance_class": "COLMAP_MEASURED",
                "support_point3d_ids": [
                    int(item["point3d_id"]) for item in points
                ],
                "surface": result["surface"],
                "validation": result["metrics"],
                "completion": "measured convex hull only",
            }
        )

    generated_at = datetime.now(timezone.utc)
    run_id = f"semantic-surfaces-{generated_at.strftime('%Y%m%dT%H%M%SZ')}"
    payload = {
        "contract_version": 1,
        "run_id": run_id,
        "hotel_id": workspace.hotel_id,
        "generated_at": generated_at.isoformat(),
        "status": "completed",
        "policy": {
            "eligible_classes": sorted(PLANAR_CLASSES),
            "minimum_points": 8,
            "holdout": "point3D_id modulo 5, with deterministic fallback",
            "maximum_fit_p90_m": 0.35,
            "maximum_holdout_p90_m": 0.45,
            "maximum_planarity_ratio": 0.45,
            "extent": "convex hull of measured points only",
            "occluded_completion": "forbidden",
        },
        "summary": {
            "audited_instances": len(audits),
            "accepted_surfaces": len(surfaces),
            "refused_instances": sum(item["status"] == "refused" for item in audits),
            "geometry_3d_created": len(surfaces),
            "by_class": dict(sorted(Counter(item["class"] for item in surfaces).items())),
        },
        "sources": {
            "registered_semantic_support": str(
                support_path.relative_to(workspace.root)
            )
        },
        "source_digests": {
            str(support_path.relative_to(workspace.root)): _digest(support_path)
        },
        "audits": audits,
        "surfaces": surfaces,
    }
    relative = f"11_conditioning/semantic_surface_runs/{run_id}.json"
    payload["versioned_artifact"] = relative
    workspace.write_json(relative, payload)
    path = workspace.write_json("11_conditioning/semantic_surfaces.json", payload)
    return path, payload


__all__ = ["PLANAR_CLASSES", "SemanticSurfaceUnavailable", "fit_planar_candidate", "run"]
