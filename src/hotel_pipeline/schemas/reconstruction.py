"""Schémas pour le Lot 2 — Reconstruction 3D.

Le Lot 2 consomme les artefacts du Lot 1B et produit une reconstruction
photogrammétrique ou géométrique alignée. Ces schémas définissent les
contrats d'entrée et de sortie du pipeline de reconstruction.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import ReconstructionRole


class ReconstructionBackend(str, Enum):
    COLMAP_INCREMENTAL = "colmap_incremental"
    COLMAP_GLOBAL = "colmap_global"
    GLUEMAP = "gluemap"
    MP_SFM = "mpsfm"
    MAP_ANYTHING = "mapanything"
    VGGT = "vggt"
    BRUSH = "brush"
    GSPLAT = "gsplat"


class AlignmentAnchor(str, Enum):
    FOOTPRINT = "footprint"
    LIDAR_ROOF = "lidar_roof"
    DTM = "dtm"
    DSM = "dsm"
    CONTEXT_BUILDING = "context_building"


class ReconstructionInputManifest(BaseModel):
    """Snapshot immuable du corpus sélectionné pour une reconstruction.

    Ce manifeste est créé une fois pour toutes avant toute exécution de
    solveur. Il garantit que COLMAP, GLUEMAP, MP-SfM et les
    vérificateurs feed-forward reçoivent exactement les mêmes données.

    Toute modification du corpus nécessite un nouveau `reconstruction_input_id`.
    """

    model_config = ConfigDict(extra="forbid")

    contract_version: int = Field(default=1, ge=1)
    reconstruction_input_id: str = Field(min_length=1)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    #: Empreintes des manifestes sources au moment du snapshot.
    asset_manifest_digest: str = Field(min_length=64, max_length=64)
    spatial_manifest_digest: str = Field(min_length=64, max_length=64)
    site_manifest_digest: str = Field(min_length=64, max_length=64)
    coverage_digest: str = Field(min_length=64, max_length=64)
    router_decision_digest: str = Field(min_length=64, max_length=64)

    #: Assets sélectionnés pour la reconstruction.
    selected_asset_ids: list[str] = Field(min_length=1)
    #: Assets exclus avec motif.
    excluded_asset_ids: list[str] = Field(default_factory=list)
    #: Motif d'exclusion par asset_id.
    selection_reasons: dict[str, str] = Field(default_factory=dict)

    #: Masques SfM appliqués (sky, people, cars, etc.).
    mask_set_digest: str | None = Field(default=None, min_length=64, max_length=64)

    #: Backends autorisés pour cette reconstruction.
    allowed_backends: list[str] = Field(default_factory=lambda: ["colmap_incremental"])

    @model_validator(mode="after")
    def _no_overlap_between_selected_and_excluded(self) -> "ReconstructionInputManifest":
        overlap = set(self.selected_asset_ids) & set(self.excluded_asset_ids)
        if overlap:
            raise ValueError(
                f"assets présents à la fois dans selected et excluded : {sorted(overlap)}"
            )
        return self


class ReconstructionSelection(BaseModel):
    """Décision de sélection d'un asset pour la reconstruction."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str
    decision: Literal["selected", "rejected", "auxiliary", "texture_only"]
    reason: str | None = None
    reconstruction_role: ReconstructionRole | None = None


class ReconstructionSelectionManifest(BaseModel):
    """Sélection détaillée des assets pour la reconstruction."""

    model_config = ConfigDict(extra="forbid")

    contract_version: int = Field(default=1, ge=1)
    reconstruction_input_id: str = Field(min_length=1)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    selections: list[ReconstructionSelection] = Field(min_length=1)

    @model_validator(mode="after")
    def _selected_assets_are_non_empty(self) -> "ReconstructionSelectionManifest":
        selected = [s for s in self.selections if s.decision == "selected"]
        if not selected:
            raise ValueError("au moins un asset doit être sélectionné pour la reconstruction")
        return self


# ---------------------------------------------------------------------------
# P1 — View Graph
# ---------------------------------------------------------------------------


class ViewGraphNode(BaseModel):
    """Un asset sélectionné, prêt pour le matching."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str
    intrinsics: dict | None = None  # {fx, fy, cx, cy, distortion}
    pose_status: Literal["unknown", "estimated", "registered"] = "unknown"
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)


class PairEvidence(BaseModel):
    """Résultat du matching entre deux images."""

    model_config = ConfigDict(extra="forbid")

    image_a: str
    image_b: str
    retrieval_score: float | None = Field(default=None, ge=0.0)
    matches: int = Field(default=0, ge=0)
    inliers: int = Field(default=0, ge=0)
    inlier_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    relative_pose: dict | None = None  # {R, t, inliers}
    overlap_estimate: float = Field(default=0.0, ge=0.0, le=1.0)
    degeneracy: Literal["none", "planar", "rotation", "pure_translation"] = "none"
    status: Literal["valid", "degenerate", "failed"] = "failed"


class ViewGraphReport(BaseModel):
    """Résumé du graphe de vue — le nouveau G5."""

    model_config = ConfigDict(extra="forbid")

    images_selected: int = Field(ge=0)
    pairs_tested: int = Field(ge=0)
    valid_pairs: int = Field(ge=0)
    largest_component: int = Field(ge=0)
    registered_candidate_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    median_inlier_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    continuity_by_demand: dict[str, float] = Field(default_factory=dict)
    repetitive_risk: Literal["none", "low", "medium", "high"] = "none"
    intrinsics_quality: Literal["poor", "fair", "good", "excellent"] = "fair"


class ViewGraphManifest(BaseModel):
    """Graphe de vue complet pour la reconstruction."""

    model_config = ConfigDict(extra="forbid")

    contract_version: int = Field(default=1, ge=1)
    view_graph_id: str = Field(min_length=1)
    reconstruction_input_id: str = Field(min_length=1)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    nodes: list[ViewGraphNode] = Field(min_length=1)
    pairs: list[PairEvidence] = Field(default_factory=list)
    report: ViewGraphReport


# ---------------------------------------------------------------------------
# P2 — Reconstruction Plan & Run
# ---------------------------------------------------------------------------


class ReconstructionPlan(BaseModel):
    """Plan de reconstruction — indépendant du Router Lot 1B."""

    model_config = ConfigDict(extra="forbid")

    contract_version: int = Field(default=1, ge=1)
    plan_id: str = Field(min_length=1)
    reconstruction_input_id: str = Field(min_length=1)
    view_graph_id: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    selected_backends: list[str] = Field(min_length=1)
    fallback_chain: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)


class ReconstructionRunStatus(BaseModel):
    """État d'une exécution de solveur."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    backend: str
    status: Literal["pending", "running", "completed", "failed"]
    started_at: str | None = None
    finished_at: str | None = None
    metrics: dict = Field(default_factory=dict)
    output_path: str | None = None
    error: str | None = None


class ReconstructionRun(BaseModel):
    """Exécution d'un backend sur un snapshot donné."""

    model_config = ConfigDict(extra="forbid")

    contract_version: int = Field(default=1, ge=1)
    run_id: str = Field(min_length=1)
    reconstruction_input_id: str = Field(min_length=1)
    view_graph_id: str | None = None
    plan_id: str | None = None
    backend: str
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    metrics: dict = Field(default_factory=dict)
    output_path: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# P3 — Consensus
# ---------------------------------------------------------------------------


class CameraConsensusEntry(BaseModel):
    """Consensus par image sur plusieurs backends."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str
    backends: list[str] = Field(min_length=1)
    translation_spread_m: float = Field(default=0.0, ge=0.0)
    rotation_spread_deg: float = Field(default=0.0, ge=0.0)
    focal_spread_px: float = Field(default=0.0, ge=0.0)
    confidence: Literal["high", "medium", "low", "none"] = "none"
    aberrants: list[str] = Field(default_factory=list)


class ReconstructionConsensusReport(BaseModel):
    """Comparaison de plusieurs reconstructions."""

    model_config = ConfigDict(extra="forbid")

    contract_version: int = Field(default=1, ge=1)
    consensus_id: str = Field(min_length=1)
    reconstruction_input_id: str = Field(min_length=1)
    run_ids: list[str] = Field(min_length=2)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    pairwise_alignment_errors: dict[str, float] = Field(default_factory=dict)
    camera_consensus: list[CameraConsensusEntry] = Field(default_factory=list)
    selected_run_id: str | None = None
    selection_rationale: str | None = None


# ---------------------------------------------------------------------------
# P4 — Geo Alignment
# ---------------------------------------------------------------------------


class GeoAlignmentManifest(BaseModel):
    """Alignement de la reconstruction sur le géospatial."""

    model_config = ConfigDict(extra="forbid")

    contract_version: int = Field(default=1, ge=1)
    alignment_id: str = Field(min_length=1)
    source_reconstruction_id: str = Field(min_length=1)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    scale: float = Field(gt=0.0)
    rotation: dict | list = Field(min_length=1)  # 3x3 rotation matrix or quaternion
    translation: dict = Field(min_length=1)  # {x, y, z}

    horizontal_crs: str = Field(min_length=1)
    vertical_reference: str | None = None

    footprint_error_m: float = Field(ge=0.0)
    roof_height_error_m: float = Field(ge=0.0)
    alignment_rmse_m: float = Field(ge=0.0)
    anchors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# P5 — Surface Confidence
# ---------------------------------------------------------------------------


class SurfaceConfidence(BaseModel):
    """Confiance par surface après reconstruction."""

    model_config = ConfigDict(extra="forbid")

    zone_id: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    independent_observations: float = Field(default=0.0, ge=0.0, le=1.0)
    angular_diversity: float = Field(default=0.0, ge=0.0, le=1.0)
    track_support: float = Field(default=0.0, ge=0.0, le=1.0)
    reprojection_error: float = Field(default=0.0, ge=0.0)
    depth_agreement: float = Field(default=0.0, ge=0.0, le=1.0)
    camera_pose_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    cross_method_agreement: float = Field(default=0.0, ge=0.0, le=1.0)
    geo_prior_agreement: float = Field(default=0.0, ge=0.0, le=1.0)
    extrapolation_penalty: float = Field(default=0.0, ge=0.0, le=1.0)


class SurfaceConfidenceManifest(BaseModel):
    """Carte de confiance post-reconstruction."""

    model_config = ConfigDict(extra="forbid")

    contract_version: int = Field(default=1, ge=1)
    confidence_id: str = Field(min_length=1)
    reconstruction_run_id: str = Field(min_length=1)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    surfaces: list[SurfaceConfidence] = Field(min_length=1)


# ---------------------------------------------------------------------------
# P6 — Camera Feasibility & Final Gate
# ---------------------------------------------------------------------------


class CameraFeasibilityField(BaseModel):
    """Évalue si une pose caméra est faisable sur la reconstruction."""

    model_config = ConfigDict(extra="forbid")

    pose_id: str
    position_local_m: tuple[float, float, float]
    yaw_deg: float = Field(ge=0.0, lt=360.0)
    pitch_deg: float = Field(ge=-90.0, le=90.0)
    fov_deg: float = Field(gt=0.0)

    visible_surface_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    unknown_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    proxy_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    reconstructed_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    minimum_distance_violation: bool = False
    collision: bool = False
    framing_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)


class ValidatedCameraPath(BaseModel):
    """Trajectoire de caméra validée par la reconstruction."""

    model_config = ConfigDict(extra="forbid")

    contract_version: int = Field(default=1, ge=1)
    path_id: str = Field(min_length=1)
    reconstruction_run_id: str = Field(min_length=1)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    simulation_only: bool = True
    poses: list[CameraFeasibilityField] = Field(min_length=1)
    derivation: str = Field(min_length=1)


class ScenePackageType(str, Enum):
    HYBRID_PROXY = "hybrid_proxy"
    RECONSTRUCTED_PHOTO_FIRST = "reconstructed_photo_first"
    RECONSTRUCTED_HYBRID = "reconstructed_hybrid"


class ReconstructionGateStatus(str, Enum):
    NEEDS_AUTHORIZED_CAPTURE = "NEEDS_AUTHORIZED_CAPTURE"
    ENVIRONMENT_3D_READY = "ENVIRONMENT_3D_READY"
    BLOCKED = "BLOCKED"


__all__ = [
    "ReconstructionInputManifest",
    "ReconstructionSelection",
    "ReconstructionSelectionManifest",
    "ViewGraphNode",
    "PairEvidence",
    "ViewGraphReport",
    "ViewGraphManifest",
    "ReconstructionPlan",
    "ReconstructionRun",
    "ReconstructionConsensusReport",
    "CameraConsensusEntry",
    "GeoAlignmentManifest",
    "SurfaceConfidence",
    "SurfaceConfidenceManifest",
    "CameraFeasibilityField",
    "ValidatedCameraPath",
    "ScenePackageType",
    "ReconstructionGateStatus",
    "ReconstructionBackend",
    "AlignmentAnchor",
]
