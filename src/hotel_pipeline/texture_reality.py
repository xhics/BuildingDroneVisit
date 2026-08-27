"""Physical texture evidence gate for requested camera output resolution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class TextureRealityLevel(StrEnum):
    UNSUPPORTED = "UNSUPPORTED"
    DISTANT_ONLY = "DISTANT_ONLY"
    SAFE_FOR_NOVEL_VIEW = "SAFE_FOR_NOVEL_VIEW"
    SAFE_FOR_CLOSEUP = "SAFE_FOR_CLOSEUP"


@dataclass(frozen=True)
class TextureEvidence:
    effective_gsd_m: float | None
    coverage: float
    sharpness: float
    view_count: float
    pose_confidence: float
    incidence_deg: float
    photometric_consistency: float
    unknown_fraction: float


@dataclass(frozen=True)
class CameraTextureDemand:
    distance_m: float
    horizontal_fov_deg: float
    output_width_px: int
    frame_occupancy: float = 1.0
    foreshortening: float = 1.0


@dataclass(frozen=True)
class TextureRealityResult:
    level: TextureRealityLevel
    safe: bool
    required_gsd_m: float
    upscale_factor: float
    score: float
    min_safe_distance_m: float
    reasons: tuple[str, ...]


def required_gsd(demand: CameraTextureDemand) -> float:
    """World metres per requested surface pixel for an ideal pinhole camera."""
    if demand.distance_m <= 0 or demand.output_width_px <= 0:
        raise ValueError("distance and output width must be positive")
    if not 0 < demand.horizontal_fov_deg < 180:
        raise ValueError("horizontal FOV must be between 0 and 180 degrees")
    occupancy = float(np.clip(demand.frame_occupancy, 1e-3, 1.0))
    shortening = float(np.clip(demand.foreshortening, 1e-3, 1.0))
    scene_gsd = (
        2.0 * demand.distance_m
        * math.tan(math.radians(demand.horizontal_fov_deg) / 2.0)
        / demand.output_width_px
    )
    return scene_gsd / (occupancy * shortening)


def _quality(evidence: TextureEvidence) -> tuple[float, dict[str, float]]:
    factors = {
        "coverage": float(np.clip((evidence.coverage - 0.35) / 0.6, 0.0, 1.0)),
        "sharpness": float(np.clip(evidence.sharpness, 0.0, 1.0)),
        "view_consensus": float(np.clip(evidence.view_count / 4.0, 0.2, 1.0)),
        "pose": float(np.clip(evidence.pose_confidence, 0.0, 1.0)),
        "incidence": float(np.clip(math.cos(math.radians(evidence.incidence_deg)), 0.0, 1.0)),
        "photometric": float(np.clip(evidence.photometric_consistency, 0.0, 1.0)),
        "known": float(np.clip(1.0 - evidence.unknown_fraction, 0.0, 1.0)),
    }
    return float(np.prod(list(factors.values())) ** (1.0 / len(factors))), factors


def evaluate_texture_reality(
    evidence: TextureEvidence,
    demand: CameraTextureDemand,
    *,
    required_level: TextureRealityLevel = TextureRealityLevel.SAFE_FOR_NOVEL_VIEW,
) -> TextureRealityResult:
    required = required_gsd(demand)
    available = evidence.effective_gsd_m
    if available is None or not math.isfinite(available) or available <= 0:
        return TextureRealityResult(
            TextureRealityLevel.UNSUPPORTED, False, required, math.inf, 0.0,
            math.inf, ("no measured texture GSD",),
        )
    upscale = available / required
    quality, factors = _quality(evidence)
    resolution_score = float(np.clip(2.0 / max(upscale, 1e-9), 0.0, 1.0))
    score = quality * resolution_score
    reasons = [name for name, value in factors.items() if value < 0.55]

    if evidence.coverage < 0.5 or evidence.unknown_fraction > 0.5 or score < 0.35:
        level = TextureRealityLevel.UNSUPPORTED
        max_upscale = 0.0
    elif upscale <= 1.5 and evidence.coverage >= 0.85 and evidence.view_count >= 2 and quality >= 0.72:
        level = TextureRealityLevel.SAFE_FOR_CLOSEUP
        max_upscale = 1.5
    elif upscale <= 2.0 and evidence.coverage >= 0.7 and quality >= 0.55:
        level = TextureRealityLevel.SAFE_FOR_NOVEL_VIEW
        max_upscale = 2.0
    elif upscale <= 3.0 and evidence.coverage >= 0.5:
        level = TextureRealityLevel.DISTANT_ONLY
        max_upscale = 3.0
    else:
        level = TextureRealityLevel.UNSUPPORTED
        max_upscale = 0.0

    rank = {
        TextureRealityLevel.UNSUPPORTED: 0,
        TextureRealityLevel.DISTANT_ONLY: 1,
        TextureRealityLevel.SAFE_FOR_NOVEL_VIEW: 2,
        TextureRealityLevel.SAFE_FOR_CLOSEUP: 3,
    }
    safe = rank[level] >= rank[required_level]
    policy_upscale = {
        TextureRealityLevel.DISTANT_ONLY: 3.0,
        TextureRealityLevel.SAFE_FOR_NOVEL_VIEW: 2.0,
        TextureRealityLevel.SAFE_FOR_CLOSEUP: 1.5,
    }.get(required_level, max_upscale)
    quality_penalty = 1.0 / max(quality, 0.25)
    min_distance = (
        available * demand.output_width_px * demand.frame_occupancy
        * demand.foreshortening * quality_penalty
        / (2.0 * math.tan(math.radians(demand.horizontal_fov_deg) / 2.0) * policy_upscale)
    )
    if not safe:
        reasons.insert(0, f"level {level.value} below {required_level.value}")
    return TextureRealityResult(
        level, safe, required, upscale, score, min_distance, tuple(reasons),
    )


def evaluate_visible_tiles(
    evidences: list[TextureEvidence], demand: CameraTextureDemand,
    *, required_level: TextureRealityLevel = TextureRealityLevel.SAFE_FOR_NOVEL_VIEW,
) -> TextureRealityResult:
    """Hard gate: every visible UV tile must support the requested shot."""
    if not evidences:
        return evaluate_texture_reality(
            TextureEvidence(None, 0, 0, 0, 0, 90, 0, 1), demand,
            required_level=required_level,
        )
    results = [
        evaluate_texture_reality(item, demand, required_level=required_level)
        for item in evidences
    ]
    return min(results, key=lambda item: (item.safe, item.score))


__all__ = [
    "CameraTextureDemand", "TextureEvidence", "TextureRealityLevel",
    "TextureRealityResult", "evaluate_texture_reality", "evaluate_visible_tiles",
    "required_gsd",
]
