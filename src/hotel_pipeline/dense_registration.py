"""Admission and uncertainty-aware fusion of dense geometry."""

from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from .schemas.canonical_states import MeasurementState


def dense_registration_gate(dense_points: np.ndarray, canonical_points: np.ndarray, *, reprojection_px: float | None, normal_agreement: float | None, max_median_distance_m: float = .20, min_overlap: float = .7, max_reprojection_px: float = 3.0, min_normal_agreement: float = .8) -> dict:
    dense, canonical = np.asarray(dense_points,float), np.asarray(canonical_points,float)
    if not len(dense) or not len(canonical):
        return {"status":"UNREGISTERED", "admitted":False, "reason":"empty geometry"}
    distances, _ = cKDTree(canonical).query(dense, k=1)
    median = float(np.median(distances)); overlap = float(np.mean(distances <= max_median_distance_m))
    checks = {
        "distance": median <= max_median_distance_m,
        "overlap": overlap >= min_overlap,
        "reprojection": reprojection_px is not None and reprojection_px <= max_reprojection_px,
        "normals": normal_agreement is not None and normal_agreement >= min_normal_agreement,
    }
    return {"status":"REGISTERED" if all(checks.values()) else "UNREGISTERED", "admitted":all(checks.values()), "median_distance_m":median, "overlap":overlap, "reprojection_px":reprojection_px, "normal_agreement":normal_agreement, "checks":checks}


def fuse_surface(primary_xyz: np.ndarray, secondary_xyz: np.ndarray, *, primary_state: MeasurementState, primary_sigma_m: float, secondary_state: MeasurementState, secondary_sigma_m: float) -> dict:
    """Measured LiDAR remains dominant over noisier dense/inferred support."""
    priority = {MeasurementState.MEASURED:3, MeasurementState.INFERRED:1, MeasurementState.UNKNOWN:0}
    keep_primary = priority[primary_state] > priority[secondary_state] or (priority[primary_state] == priority[secondary_state] and primary_sigma_m <= secondary_sigma_m)
    chosen = np.asarray(primary_xyz if keep_primary else secondary_xyz,float)
    return {"xyz":chosen, "dominant_state":(primary_state if keep_primary else secondary_state).value, "dominant_source":"primary" if keep_primary else "secondary", "secondary_support":"secondary" if keep_primary else "primary"}


__all__ = ["dense_registration_gate", "fuse_surface"]
