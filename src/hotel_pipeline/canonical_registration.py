"""Strict COLMAP-to-world Sim(3) contract in canonical ENU coordinates."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .geometry_align import umeyama_sim3


class DegenerateRegistration(ValueError):
    pass


@dataclass(frozen=True)
class CanonicalSim3:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    source_axes: str = "COLMAP_X_RIGHT_Y_DOWN_Z_FORWARD"
    target_axes: str = "X_EAST_Y_NORTH_Z_UP"
    source_unit: str = "arbitrary"
    target_unit: str = "m"

    def __post_init__(self):
        r = np.asarray(self.rotation, float); t = np.asarray(self.translation, float)
        if r.shape != (3,3) or t.shape != (3,) or self.scale <= 0:
            raise ValueError("invalid Sim(3) components")
        if not np.allclose(r.T @ r, np.eye(3), atol=1e-7) or np.linalg.det(r) < .999999:
            raise ValueError("rotation must be right-handed and orthonormal")
        object.__setattr__(self, "rotation", r); object.__setattr__(self, "translation", t)

    def colmap_to_world(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, float)
        return self.scale * (p @ self.rotation.T) + self.translation

    def world_to_colmap(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, float)
        return ((p - self.translation) @ self.rotation) / self.scale

    def as_dict(self) -> dict:
        return {"scale": self.scale, "rotation": self.rotation.tolist(), "translation": self.translation.tolist(), "formula": "X_world = scale * R * X_colmap + translation", "source_axes": self.source_axes, "target_axes": self.target_axes, "source_unit": self.source_unit, "target_unit": self.target_unit}


def fit_canonical_sim3(source: np.ndarray, target: np.ndarray, *, min_points: int = 4, min_singular_ratio: float = 1e-3) -> CanonicalSim3:
    source, target = np.asarray(source,float), np.asarray(target,float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < min_points:
        raise DegenerateRegistration(f"at least {min_points} 3D correspondences required")
    centred = source - source.mean(axis=0)
    singular = np.linalg.svd(centred, compute_uv=False)
    ratio = float(singular[-1] / max(singular[0], 1e-15))
    if ratio < min_singular_ratio:
        raise DegenerateRegistration(f"poor 3D distribution: singular ratio {ratio:.3g}")
    r, t, s = umeyama_sim3(source, target)
    return CanonicalSim3(s, r, t)


COLMAP_CAMERA_TO_CANONICAL = np.array([[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]])
CANONICAL_TO_THREE = np.array([[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]])


def adapt_direction(vector: np.ndarray, adapter: np.ndarray) -> np.ndarray:
    return np.asarray(adapter,float) @ np.asarray(vector,float)


def fit_vertical_rigid(source: np.ndarray, target: np.ndarray) -> CanonicalSim3:
    """Rigid/scale fit corrects tilt globally; never edits Z independently."""
    return fit_canonical_sim3(source, target)


@dataclass(frozen=True)
class VerticalSourceTransform:
    source_id: str
    original_datum: str
    canonical_datum: str
    offset_m: float
    operation: str

    def apply(self, xyz: np.ndarray) -> np.ndarray:
        result = np.asarray(xyz,float).copy(); result[...,2] += self.offset_m; return result


__all__ = ["CANONICAL_TO_THREE", "COLMAP_CAMERA_TO_CANONICAL", "CanonicalSim3", "DegenerateRegistration", "VerticalSourceTransform", "adapt_direction", "fit_canonical_sim3", "fit_vertical_rigid"]
