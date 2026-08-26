"""Evidence-driven source selection for a hybrid reality renderer.

A highly realistic shot does not have one universally best source. LiDAR or the
canonical mesh may be best for geometry, real photos for facade appearance,
Google photogrammetry for distant context, and generative AI only for
non-structural micro-appearance. This module chooses those roles explicitly per
surface and requested camera shot.

The central invariant is strict: generative sources can never become geometric
truth. If measured/inferred geometry cannot support a shot, the shot is rejected
or moved; it is never repaired by hallucinating structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import cos, exp, radians
from typing import Sequence

from .reality_gate import render_pixel_footprint_m


class RealitySource(StrEnum):
    CANONICAL = "CANONICAL"
    REAL_PHOTO = "REAL_PHOTO"
    GOOGLE_3D = "GOOGLE_3D"
    ORTHOPHOTO = "ORTHOPHOTO"
    INFERRED = "INFERRED"
    GENERATIVE = "GENERATIVE"


@dataclass(frozen=True)
class SourceEvidence:
    source_id: str
    source_type: RealitySource
    geometry_confidence: float = 0.0
    appearance_confidence: float = 0.0
    effective_gsd_m: float | None = None
    coverage_fraction: float = 1.0
    incidence_deg: float = 0.0
    temporal_confidence: float = 1.0
    sharpness: float = 1.0
    measured: bool = False
    supports_geometry: bool = False
    supports_appearance: bool = False

    def __post_init__(self) -> None:
        for name in (
            "geometry_confidence",
            "appearance_confidence",
            "coverage_fraction",
            "temporal_confidence",
            "sharpness",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0.0 <= float(self.incidence_deg) < 90.0:
            raise ValueError("incidence_deg must be in [0, 90)")
        if self.effective_gsd_m is not None and self.effective_gsd_m <= 0:
            raise ValueError("effective_gsd_m must be positive")
        if self.source_type is RealitySource.GENERATIVE and self.supports_geometry:
            raise ValueError("generative evidence can never support geometry")


@dataclass(frozen=True)
class HybridRenderDecision:
    surface_id: str
    geometry_source_id: str | None
    appearance_source_id: str | None
    geometry_score: float
    appearance_score: float
    required_gsd_m: float
    appearance_upscale_factor: float | None
    allow_ai_microtexture: bool
    safe: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class HybridPolicy:
    minimum_geometry_score: float = 0.55
    minimum_appearance_score: float = 0.48
    maximum_measured_texture_upscale: float = 2.0
    maximum_context_texture_upscale: float = 4.0
    minimum_photo_coverage: float = 0.65
    allow_generative_microtexture: bool = True


def _incidence_score(angle_deg: float) -> float:
    # Squared cosine strongly penalizes grazing imagery without an arbitrary
    # hard cliff. It mirrors the texture fusion policy used by orthofacades.
    value = max(0.0, cos(radians(float(angle_deg))))
    return value * value


def geometry_score(evidence: SourceEvidence) -> float:
    if not evidence.supports_geometry:
        return 0.0
    source_prior = {
        RealitySource.CANONICAL: 1.0,
        RealitySource.GOOGLE_3D: 0.82,
        RealitySource.ORTHOPHOTO: 0.35,
        RealitySource.INFERRED: 0.45,
        RealitySource.REAL_PHOTO: 0.0,
        RealitySource.GENERATIVE: 0.0,
    }[evidence.source_type]
    measured_bonus = 1.0 if evidence.measured else 0.82
    return float(
        source_prior
        * measured_bonus
        * evidence.geometry_confidence
        * (0.85 + 0.15 * evidence.temporal_confidence)
    )


def appearance_score(
    evidence: SourceEvidence,
    *,
    required_gsd_m: float,
    policy: HybridPolicy,
) -> tuple[float, float | None]:
    if not evidence.supports_appearance:
        return 0.0, None

    source_prior = {
        RealitySource.REAL_PHOTO: 1.0,
        RealitySource.CANONICAL: 0.96,
        RealitySource.GOOGLE_3D: 0.78,
        RealitySource.ORTHOPHOTO: 0.72,
        RealitySource.INFERRED: 0.38,
        RealitySource.GENERATIVE: 0.20,
    }[evidence.source_type]

    if evidence.effective_gsd_m is None:
        # Unknown resolution cannot outrank a measured high-resolution source.
        resolution_score = 0.42
        upscale = None
    else:
        upscale = evidence.effective_gsd_m / max(required_gsd_m, 1e-12)
        allowed = (
            policy.maximum_context_texture_upscale
            if evidence.source_type in {RealitySource.GOOGLE_3D, RealitySource.ORTHOPHOTO}
            else policy.maximum_measured_texture_upscale
        )
        if upscale <= 1.0:
            resolution_score = 1.0
        else:
            # Smooth but aggressive decay: at the policy limit the source is
            # still usable, beyond it it rapidly loses to more distant/context
            # alternatives.
            resolution_score = exp(-1.35 * max(0.0, upscale / allowed - 0.5))
            if upscale > allowed:
                resolution_score *= 0.25

    score = (
        source_prior
        * evidence.appearance_confidence
        * evidence.coverage_fraction
        * evidence.temporal_confidence
        * evidence.sharpness
        * _incidence_score(evidence.incidence_deg)
        * resolution_score
    )
    return float(score), None if upscale is None else float(upscale)


def choose_hybrid_sources(
    surface_id: str,
    evidence: Sequence[SourceEvidence],
    *,
    distance_m: float,
    output_width_px: int = 1920,
    horizontal_fov_deg: float = 60.0,
    policy: HybridPolicy | None = None,
) -> HybridRenderDecision:
    """Choose geometry and appearance independently for one visible surface."""
    policy = policy or HybridPolicy()
    required_gsd = render_pixel_footprint_m(
        distance_m,
        output_width_px=output_width_px,
        horizontal_fov_deg=horizontal_fov_deg,
    )

    geometry_rows = [(geometry_score(item), item) for item in evidence]
    geometry_rows = [row for row in geometry_rows if row[0] > 0.0]
    geometry_rows.sort(key=lambda row: row[0], reverse=True)
    geometry_value, geometry = geometry_rows[0] if geometry_rows else (0.0, None)

    appearance_rows: list[tuple[float, float | None, SourceEvidence]] = []
    for item in evidence:
        score, upscale = appearance_score(
            item, required_gsd_m=required_gsd, policy=policy
        )
        if score > 0.0:
            appearance_rows.append((score, upscale, item))
    appearance_rows.sort(key=lambda row: row[0], reverse=True)
    if appearance_rows:
        appearance_value, appearance_upscale, appearance = appearance_rows[0]
    else:
        appearance_value, appearance_upscale, appearance = 0.0, None, None

    reasons: list[str] = []
    if geometry is None or geometry_value < policy.minimum_geometry_score:
        reasons.append("no geometry source satisfies reality threshold")
    if appearance is None or appearance_value < policy.minimum_appearance_score:
        reasons.append("no appearance source satisfies reality threshold")
    if (
        appearance is not None
        and appearance.source_type is RealitySource.REAL_PHOTO
        and appearance.coverage_fraction < policy.minimum_photo_coverage
    ):
        reasons.append("real-photo appearance coverage is insufficient")

    allow_ai = bool(
        policy.allow_generative_microtexture
        and geometry is not None
        and geometry_value >= policy.minimum_geometry_score
        and appearance is not None
        and appearance.source_type is not RealitySource.GENERATIVE
    )

    # Generative appearance may be proposed as an emergency candidate, but it
    # can never by itself turn an unsupported surface into a safe shot.
    if appearance is not None and appearance.source_type is RealitySource.GENERATIVE:
        reasons.append("generative appearance cannot establish reality")
        allow_ai = False

    return HybridRenderDecision(
        surface_id=surface_id,
        geometry_source_id=None if geometry is None else geometry.source_id,
        appearance_source_id=None if appearance is None else appearance.source_id,
        geometry_score=round(float(geometry_value), 4),
        appearance_score=round(float(appearance_value), 4),
        required_gsd_m=float(required_gsd),
        appearance_upscale_factor=(
            None if appearance_upscale is None else float(appearance_upscale)
        ),
        allow_ai_microtexture=allow_ai,
        safe=not reasons,
        reasons=tuple(reasons),
    )


def choose_for_surfaces(
    surfaces: dict[str, Sequence[SourceEvidence]],
    *,
    distance_by_surface_m: dict[str, float],
    output_width_px: int = 1920,
    horizontal_fov_deg: float = 60.0,
    policy: HybridPolicy | None = None,
) -> dict[str, HybridRenderDecision]:
    """Batch helper used by a future hybrid renderer/path planner."""
    decisions: dict[str, HybridRenderDecision] = {}
    for surface_id, rows in surfaces.items():
        if surface_id not in distance_by_surface_m:
            raise ValueError(f"missing camera distance for surface {surface_id!r}")
        decisions[surface_id] = choose_hybrid_sources(
            surface_id,
            rows,
            distance_m=distance_by_surface_m[surface_id],
            output_width_px=output_width_px,
            horizontal_fov_deg=horizontal_fov_deg,
            policy=policy,
        )
    return decisions


__all__ = [
    "HybridPolicy",
    "HybridRenderDecision",
    "RealitySource",
    "SourceEvidence",
    "appearance_score",
    "choose_for_surfaces",
    "choose_hybrid_sources",
    "geometry_score",
]
