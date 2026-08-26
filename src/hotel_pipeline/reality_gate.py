"""Evidence contracts and final spatial Reality Gate."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from math import exp, radians, sqrt
from typing import Iterable, Sequence

import numpy as np

from .camera_feasibility import segment_intersects_mesh
from .schemas.canonical_states import MaterialClass, MeasurementState, RealityLevel


@dataclass(frozen=True)
class GeometryUncertainty:
    source_type: str
    sigma_xyz_m: tuple[float, float, float]
    source_ids: tuple[str, ...] = ()

    @property
    def sigma_m(self) -> float:
        return sqrt(sum(value * value for value in self.sigma_xyz_m))

    @property
    def geometry_confidence(self) -> float:
        return float(exp(-self.sigma_m / 0.25))


@dataclass(frozen=True)
class PoseUncertainty:
    translation_sigma_m: tuple[float, float, float]
    rotation_sigma_deg: tuple[float, float, float]

    def expected_pixel_error(self, *, focal_px: float, depth_m: float) -> float:
        if depth_m <= 0 or focal_px <= 0:
            raise ValueError("depth_m and focal_px must be positive")
        lateral = sqrt(self.translation_sigma_m[0] ** 2 + self.translation_sigma_m[1] ** 2)
        angular = sqrt(sum(radians(v) ** 2 for v in self.rotation_sigma_deg))
        return float(focal_px * (lateral / depth_m + angular))

    def texture_weight(self, *, focal_px: float, depth_m: float, tolerance_px: float = 3.0) -> float:
        error = self.expected_pixel_error(focal_px=focal_px, depth_m=depth_m)
        return float(exp(-0.5 * (error / tolerance_px) ** 2))


@dataclass(frozen=True)
class TemporalSource:
    source_id: str
    capture_date: date
    validity_interval: tuple[date, date] | None
    structure_signature: str

    def interval(self) -> tuple[date, date]:
        return self.validity_interval or (self.capture_date, self.capture_date)


@dataclass(frozen=True)
class TemporalConflict:
    source_ids: tuple[str, str]
    reason: str


def temporal_conflicts(sources: Sequence[TemporalSource]) -> list[TemporalConflict]:
    conflicts: list[TemporalConflict] = []
    for index, left in enumerate(sources):
        for right in sources[index + 1:]:
            l0, l1 = left.interval()
            r0, r1 = right.interval()
            disjoint = l1 < r0 or r1 < l0
            if disjoint and left.structure_signature != right.structure_signature:
                conflicts.append(TemporalConflict(
                    (left.source_id, right.source_id),
                    "non-overlapping validity intervals describe different structures",
                ))
    return conflicts


@dataclass(frozen=True)
class MaterialAppearance:
    material_class: MaterialClass
    base_color: tuple[float, float, float, float]
    roughness: float
    metallic: float
    opacity: float = 1.0
    selected_view_id: str | None = None


def resolve_material_appearance(
    material_class: MaterialClass,
    observations: Sequence[tuple[str, tuple[float, float, float], float]],
) -> MaterialAppearance:
    """Resolve appearance without letting illumination deform geometry.

    Observations are ``(view_id, rgb, incidence_degrees)``.
    """
    if material_class in {MaterialClass.GLASS, MaterialClass.REFLECTIVE_METAL}:
        chosen = min(observations, key=lambda row: abs(row[2]))[0] if observations else None
        metallic = 1.0 if material_class is MaterialClass.REFLECTIVE_METAL else 0.0
        return MaterialAppearance(material_class, (0.35, 0.42, 0.48, 0.55), 0.12, metallic, 0.55, chosen)
    if not observations:
        return MaterialAppearance(material_class, (0.5, 0.5, 0.5, 1.0), 0.8, 0.0)
    rgb = np.median(np.asarray([row[1] for row in observations], dtype=float), axis=0)
    return MaterialAppearance(material_class, (*map(float, rgb), 1.0), 0.8, 0.0)


@dataclass(frozen=True)
class CanonicalSurfaceIdentity:
    building_id: str
    part_id: str
    surface_id: str

    def __post_init__(self) -> None:
        if not all((self.building_id, self.part_id, self.surface_id)):
            raise ValueError("building_id, part_id and surface_id are mandatory")


def validate_surface_assignments(surfaces: Iterable[CanonicalSurfaceIdentity]) -> None:
    owners: dict[str, tuple[str, str]] = {}
    for surface in surfaces:
        owner = (surface.building_id, surface.part_id)
        if surface.surface_id in owners and owners[surface.surface_id] != owner:
            raise ValueError(f"surface {surface.surface_id!r} is shared by multiple structures")
        owners[surface.surface_id] = owner


@dataclass(frozen=True)
class SiteStructure:
    structure_id: str
    structure_type: str
    triangles: np.ndarray = field(compare=False, repr=False)

    def blocks_segment(self, start: Sequence[float], end: Sequence[float]) -> bool:
        return segment_intersects_mesh(np.asarray(start, float), np.asarray(end, float), np.asarray(self.triangles, float))


def visible_fraction(samples_visible: Sequence[bool]) -> float:
    if not samples_visible:
        raise ValueError("visibility requires at least one coverage sample")
    return float(np.mean(np.asarray(samples_visible, dtype=bool)))


@dataclass(frozen=True)
class RealityMetrics:
    geometry_rmse_m: float | None
    reprojection_p90_px: float | None
    silhouette_iou: float | None
    lidar_rmse_m: float | None
    roof_topology_valid: bool | None
    terrain_contact_valid: bool | None
    occlusion_ordering_valid: bool | None
    manifold: bool | None
    provenance_complete: bool
    measured_geometry_fraction: float
    inferred_geometry_fraction: float
    texture_coverage: float


@dataclass(frozen=True)
class RealityAssessment:
    score: float
    level: RealityLevel
    failed_evidence: tuple[str, ...]


def assess_reality(metrics: RealityMetrics) -> RealityAssessment:
    evidence = {
        "reprojection": metrics.reprojection_p90_px,
        "silhouette": metrics.silhouette_iou,
        "lidar": metrics.lidar_rmse_m,
        "roof_topology": metrics.roof_topology_valid,
        "terrain_contact": metrics.terrain_contact_valid,
        "occlusion_ordering": metrics.occlusion_ordering_valid,
        "manifoldness": metrics.manifold,
        "provenance": metrics.provenance_complete,
    }
    missing = tuple(name for name, value in evidence.items() if value is None)
    failures = list(missing)
    if metrics.reprojection_p90_px is not None and metrics.reprojection_p90_px > 8: failures.append("reprojection")
    if metrics.silhouette_iou is not None and metrics.silhouette_iou < 0.8: failures.append("silhouette")
    if metrics.lidar_rmse_m is not None and metrics.lidar_rmse_m > 0.25: failures.append("lidar")
    for name in ("roof_topology", "terrain_contact", "occlusion_ordering", "manifoldness", "provenance"):
        if evidence[name] is False: failures.append(name)
    reprojection = 20.0 if metrics.reprojection_p90_px is None else metrics.reprojection_p90_px
    silhouette = 0.0 if metrics.silhouette_iou is None else metrics.silhouette_iou
    lidar_rmse = 1.0 if metrics.lidar_rmse_m is None else metrics.lidar_rmse_m
    quality = [
        max(0.0, 1.0 - reprojection / 12.0),
        silhouette,
        max(0.0, 1.0 - lidar_rmse / 0.5),
        metrics.measured_geometry_fraction,
        metrics.texture_coverage,
    ]
    score = float(np.mean(quality))
    if failures:
        level = RealityLevel.NO_FLY_NO_RENDER if len(set(failures)) >= 3 else RealityLevel.DISTANT_VIEW_ONLY
    elif score >= 0.9 and metrics.measured_geometry_fraction >= 0.9:
        level = RealityLevel.SAFE_FOR_CLOSEUP
    elif score >= 0.72 and metrics.measured_geometry_fraction >= 0.6:
        level = RealityLevel.SAFE_FOR_NOVEL_VIEW
    elif metrics.inferred_geometry_fraction >= 0.5:
        level = RealityLevel.INFERRED
    else:
        level = RealityLevel.DISTANT_VIEW_ONLY
    return RealityAssessment(round(score, 4), level, tuple(sorted(set(failures))))


@dataclass(frozen=True)
class PathSurfaceObservation:
    surface_id: str
    level: RealityLevel
    distance_m: float
    unknown_visible_fraction: float


def validate_camera_path_reality(
    observations: Sequence[PathSurfaceObservation], *, unknown_limit: float = 0.03,
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    for item in observations:
        if item.unknown_visible_fraction > unknown_limit:
            reasons.append(f"{item.surface_id}: unknown visibility exceeds limit")
        if item.level is RealityLevel.NO_FLY_NO_RENDER:
            reasons.append(f"{item.surface_id}: no-fly/no-render")
        if item.distance_m < 5 and item.level not in {RealityLevel.MEASURED, RealityLevel.SAFE_FOR_CLOSEUP}:
            reasons.append(f"{item.surface_id}: close-up requires SAFE_FOR_CLOSEUP")
        elif item.level not in {RealityLevel.MEASURED, RealityLevel.SAFE_FOR_CLOSEUP, RealityLevel.SAFE_FOR_NOVEL_VIEW}:
            reasons.append(f"{item.surface_id}: unsafe for novel view")
    return not reasons, tuple(reasons)


__all__ = [
    "CanonicalSurfaceIdentity", "GeometryUncertainty", "MaterialAppearance",
    "PathSurfaceObservation", "PoseUncertainty", "RealityAssessment", "RealityMetrics",
    "SiteStructure", "TemporalConflict", "TemporalSource", "assess_reality",
    "resolve_material_appearance", "temporal_conflicts", "validate_camera_path_reality",
    "validate_surface_assignments", "visible_fraction",
]
