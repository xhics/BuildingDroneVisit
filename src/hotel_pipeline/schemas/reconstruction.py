"""Schémas pour le Lot 2 — Reconstruction 3D.

Le Lot 2 consomme les artefacts du Lot 1B et produit une reconstruction
photogrammétrique ou géométrique alignée. Ces schémas définissent les
contrats d'entrée et de sortie du pipeline de reconstruction.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum, StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import ReconstructionRole

# ---------------------------------------------------------------------------
# P0 — Promotion-driven criticality (remplace les littéraux PRIMARY/SECONDARY)
# ---------------------------------------------------------------------------


class Criticality(StrEnum):
    """Niveau d'exposition promotionnel d'une cible.

    La criticité est pilotée par l'intention promotionnelle, pas par une
    catégorie de surface dure codée dans le code. Un `MUST_SHOW` est ce que
    la vidéo doit montrer ; un `CONTEXT_ONLY` n'apparaît que comme décors.
    """

    MUST_SHOW = "must_show"
    SHOULD_SHOW = "should_show"
    OPTIONAL = "optional"
    CONTEXT_ONLY = "context_only"
    FORBIDDEN = "forbidden"


class SupportType(StrEnum):
    """Nature de la preuve apportée par un asset à une cible.

    Cette taxonomie est ce qui permet de distinguer une reconstruction IA
    (GLUEMAP, MapAnything, VGGT, 3DGS — `MULTIVIEW_RECONSTRUCTED` ou
    `FEEDFORWARD_INFERRED`) d'une façade inventée (`GENERATIVE_COMPLETION`).
    Seule la dernière est bloquée sur preuve d'impossibilité de capture.
    """

    MEASURED_PHOTO = "measured_photo"
    MEASURED_LIDAR = "measured_lidar"
    MULTIVIEW_RECONSTRUCTED = "multiview_reconstructed"
    FEEDFORWARD_INFERRED = "feedforward_inferred"
    GEOSPATIAL_PROXY = "geospatial_proxy"
    GENERATIVE_COMPLETION = "generative_completion"


class SupportRole(StrEnum):
    """Rôle de soutien d'un asset pour une cible (remplace `tier_assignment`)."""

    PRIMARY = "primary"
    AUXILIARY = "auxiliary"
    CONTEXT = "context"
    CROSSCHECK = "crosscheck"


class ReconstructionTargetKind(StrEnum):
    """Nature géométrique d'une cible de reconstruction.

    Distinct de `TargetKind` (acquisition), qui décrit la nature d'une cible
    de *besoin* (site_object / view_sector / …). Ici c'est la forme géométrique
    que la reconstruction doit produire.
    """

    SURFACE = "surface"
    OBJECT = "object"
    LINEAR_FEATURE = "linear_feature"
    AREA = "area"
    CONTEXT = "context"


class ReconstructionBackend(str, Enum):
    COLMAP_INCREMENTAL = "colmap_incremental"
    COLMAP_GLOBAL = "colmap_global"
    GLUEMAP = "gluemap"
    MP_SFM = "mpsfm"
    MAP_ANYTHING = "mapanything"
    VGGT = "vggt"
    BRUSH = "brush"
    GSPLAT = "gsplat"
    SYNTHETIC = "synthetic"


class ReconstructionTarget(BaseModel):
    """Une cible de reconstruction, avec son niveau d'exposition promotionnelle.

    Remplace les littéraux `PRIMARY` / `SECONDARY` gravés partout dans le code :
    la criticité n'est plus une catégorie de surface, mais une décision par
    cible, portée par le Router Lot 1B.
    """

    model_config = ConfigDict(extra="forbid")

    target_id: str = Field(min_length=1)
    kind: ReconstructionTargetKind
    criticality: Criticality
    required_fidelity: float = Field(default=0.0, ge=0.0, le=1.0)
    allowed_support: list[SupportType] = Field(default_factory=list)
    maximum_generative_completion: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="0.0 pour MUST_SHOW : on n'invente jamais l'apparence d'une façade",
    )
    maximum_inferred_geometry: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Politique/calibré : MapAnything/VGGT peuvent contribuer comme "
                    "prior ou cross-check géométrique, sans donner le droit d'inventer",
    )
    minimum_camera_distance_m: float = Field(
        default=0.0, ge=0.0,
        description="Distance minimale de la caméra pour la planification vidéo",
    )

    #: État de la géométrie qui **localise** la cible, tel que le manifeste de
    #: site le porte : `confirmed`, `inferred`, `unresolved`…
    #:
    #: Deux questions distinctes, qu'on fusionnait auparavant : « cette surface
    #: est-elle promotionnellement importante ? » et « sait-on où elle est ? ».
    #: Rabattre la première sur la seconde faisait tomber les quatre façades en
    #: CONTEXT_ONLY parce qu'aucune n'avait été confirmée une à une — et la
    #: porte de fidélité déclarait alors NOT_APPLICABLE la surface même que le
    #: produit doit montrer. L'importance est déclarée ; la localisation est
    #: constatée, et voyage ici.
    geometry_state: str | None = None

    #: La géométrie de cette cible est-elle établie, ou seulement déduite ?
    #: Une cible MUST_SHOW dont la géométrie est inférée reste MUST_SHOW : le
    #: manque se dit dans la porte, il n'efface pas l'exigence.
    geometry_confirmed: bool = False


class AssetTargetSupport(BaseModel):
    """Soutien d'un asset à une cible (remplace `tier_assignment`)."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    support_role: SupportRole
    coverage_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    reconstruction_role: ReconstructionRole


class AlignmentAnchor(str, Enum):
    FOOTPRINT = "footprint"
    LIDAR_ROOF = "lidar_roof"
    DTM = "dtm"
    DSM = "dsm"
    CONTEXT_BUILDING = "context_building"


# ---------------------------------------------------------------------------
# P1 — Reconstruction Input Manifest
# ---------------------------------------------------------------------------


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
    hotel_id: str = Field(min_length=1)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    #: Empreintes des manifestes sources au moment du snapshot.
    asset_manifest_digest: str = Field(min_length=64, max_length=64)
    spatial_manifest_digest: str = Field(min_length=64, max_length=64)
    site_manifest_digest: str = Field(min_length=64, max_length=64)
    coverage_digest: str = Field(min_length=64, max_length=64)
    router_decision_digest: str = Field(min_length=64, max_length=64)

    #: Cibles de reconstruction, avec leur criticité promotionnelle (P0.2).
    #: Remplace les littéraux PRIMARY / SECONDARY : plus aucune surface n'est
    #: catégorisée dans le code, la criticité est une décision par cible.
    targets: list[ReconstructionTarget] = Field(default_factory=list)

    #: Soutien d'un asset à une cible (P0.3, remplace `tier_assignment`).
    asset_target_support: list[AssetTargetSupport] = Field(default_factory=list)

    #: Références vers les évaluations de faisabilité de capture (P4).
    #: Un manifeste P1 initial ne doit pas dépendre d'un artefact P4 futur :
    #: `CaptureFeasibilityAssessment` n'est construit qu'après la première
    #: reconstruction. On stocke seulement des références (`target_id ->
    #: assessment_id`), résolues lors d'une boucle P4 -> P1 ultérieure.
    #: Un premier run Lot 2 laisse simplement ce dictionnaire vide.
    capture_feasibility_assessment_refs: dict[str, str] = Field(default_factory=dict)

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

    #: Cohortes temporelles séparées (point 17 du plan).
    #: Clé = nom de cohorte, valeur = asset_ids.
    #: Exemple : {"current_confirmed": [...], "historical": [...], "unknown": [...]}
    temporal_cohorts: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _no_overlap_between_selected_and_excluded(self) -> "ReconstructionInputManifest":
        overlap = set(self.selected_asset_ids) & set(self.excluded_asset_ids)
        if overlap:
            raise ValueError(
                f"assets présents à la fois dans selected et excluded : {sorted(overlap)}"
            )
        return self

    @model_validator(mode="after")
    def _asset_target_support_targets_exist(self) -> "ReconstructionInputManifest":
        target_ids = {t.target_id for t in self.targets}
        for s in self.asset_target_support:
            if s.target_id not in target_ids:
                raise ValueError(
                    f"AssetTargetSupport {s.asset_id} -> {s.target_id} : "
                    f"cible absente de `targets`"
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

    # Champs renforcés (P1.3) : la dégénérescence homographique et la
    # structure répétitive sont des signaux critiques sur les façades quasi
    # planes. On les rend explicites au niveau du manifeste.
    # edge_id = "{image_a}__{image_b}" -> dégénérescence planaire détectée.
    homography_degeneracy_flags: dict[str, bool] = Field(default_factory=dict)
    # Risque de structure répétitive (0.0 = aucun, 1.0 = élevé), continu.
    repetitive_structure_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    # Rejets de type Doppelgangers / doublons structurels (fenêtres, balcons).
    doppelganger_rejections: int = Field(default=0, ge=0)


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
    temporal_strategy: Literal["current_only", "current_plus_unknown"] = Field(default="current_only")


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
# P2.5 — Anchor-guided automatic localization
# ---------------------------------------------------------------------------


class PoseEvidenceClass(StrEnum):
    """Nature probante d'une pose caméra.

    Les deux premières classes seulement sont des mesures utilisables par G5.
    Une vue corrigée ou produite par un modèle feed-forward reste inférée tant
    qu'un PnP sur les pixels originaux ne l'a pas confirmée.
    """

    ANCHOR_MEASURED = "anchor_measured"
    LOCALIZED_MEASURED = "localized_measured"
    VIEW_INFERRED = "view_inferred"
    REJECTED = "rejected"


class LocalizationDecision(StrEnum):
    ACCEPTED = "accepted"
    INFERRED_ONLY = "inferred_only"
    REJECTED = "rejected"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class PosePriorKind(StrEnum):
    GPS_MEASURED = "gps_measured"
    PROVIDER_MEASURED = "provider_measured"
    RENDER_REQUEST = "render_request"
    EXIF = "exif"
    UNKNOWN = "unknown"


class PosePrior(BaseModel):
    """Prior de pose avec provenance et incertitude explicites."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(min_length=1)
    latitude: float | None = Field(default=None, ge=-90.0, le=90.0)
    longitude: float | None = Field(default=None, ge=-180.0, le=180.0)
    altitude_m: float | None = None
    heading_deg: float | None = Field(default=None, ge=0.0, lt=360.0)
    position_sigma_m: float = Field(default=10.0, gt=0.0)
    heading_sigma_deg: float = Field(default=30.0, gt=0.0)
    position_kind: PosePriorKind = PosePriorKind.UNKNOWN
    heading_kind: PosePriorKind = PosePriorKind.UNKNOWN
    heading_is_measured: bool = False
    source_digest: str | None = None


class AnchorLocalizationPolicy(BaseModel):
    """Seuils versionnés de sélection et de localisation.

    Les valeurs sont un profil initial du pilote, pas des constantes cachées.
    Toute modification produit un nouvel identifiant de politique et donc un
    nouvel artefact de décision.
    """

    model_config = ConfigDict(extra="forbid")

    policy_id: str = "anchor-localization-v1"
    random_seed: int = 20260820
    geo_inlier_threshold_m: float = Field(default=10.0, gt=0.0)
    anchor_rmse_max_m: float = Field(default=3.0, gt=0.0)
    anchor_heading_median_max_deg: float = Field(default=5.0, ge=0.0)
    anchor_heading_p90_max_deg: float = Field(default=30.0, ge=0.0)
    min_anchor_images: int = Field(default=8, ge=3)
    min_anchor_sources: int = Field(default=2, ge=1)
    pnp_min_inliers: int = Field(default=30, ge=4)
    pnp_min_inlier_ratio: float = Field(default=0.25, ge=0.0, le=1.0)
    pnp_min_reference_images: int = Field(default=3, ge=1)
    reprojection_median_max_px: float = Field(default=2.0, gt=0.0)
    reprojection_p95_max_px: float = Field(default=4.0, gt=0.0)
    positive_depth_ratio_min: float = Field(default=0.95, ge=0.0, le=1.0)
    gps_residual_floor_m: float = Field(default=10.0, gt=0.0)
    gps_sigma_multiplier: float = Field(default=3.0, gt=0.0)
    measured_heading_residual_max_deg: float = Field(default=15.0, ge=0.0)
    pose_stability_translation_max_m: float = Field(default=2.0, ge=0.0)
    pose_stability_rotation_max_deg: float = Field(default=3.0, ge=0.0)
    max_rounds: int = Field(default=3, ge=1)
    max_hop: int = Field(default=2, ge=0)
    max_attempts_per_level: int = Field(default=3, ge=1)


class AnchorCandidateEvidence(BaseModel):
    """Décision traçable pour une caméra candidate au noyau."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(min_length=1)
    image_name: str = Field(min_length=1)
    source: str = Field(default="unknown")
    reconstructed_center: tuple[float, float, float]
    geographic_center_enu_m: tuple[float, float, float]
    position_residual_m: float = Field(ge=0.0)
    heading_residual_deg: float | None = Field(default=None, ge=0.0, le=180.0)
    accepted: bool = False
    reasons: list[str] = Field(default_factory=list)


class AnchorSelectionManifest(BaseModel):
    """Sélection robuste des ancres depuis un modèle brut."""

    model_config = ConfigDict(extra="forbid")

    contract_version: int = Field(default=1, ge=1)
    anchor_selection_id: str = Field(min_length=1)
    reconstruction_input_id: str = Field(min_length=1)
    source_run_id: str | None = None
    source_model_path: str = Field(min_length=1)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    policy: AnchorLocalizationPolicy
    source_model_digest: str = Field(min_length=64, max_length=64)
    candidates: list[AnchorCandidateEvidence] = Field(default_factory=list)
    anchor_asset_ids: list[str] = Field(default_factory=list)
    rejected_asset_ids: list[str] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)
    status: Literal["ready", "refused"] = "refused"
    refusal_reasons: list[str] = Field(default_factory=list)


class AnchorModelManifest(BaseModel):
    """Noyau reconstruit indépendamment et figé pour la localisation."""

    model_config = ConfigDict(extra="forbid")

    contract_version: int = Field(default=1, ge=1)
    anchor_model_id: str = Field(min_length=1)
    anchor_selection_id: str = Field(min_length=1)
    reconstruction_input_id: str = Field(min_length=1)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_path: str = Field(min_length=1)
    model_digest: str = Field(min_length=64, max_length=64)
    anchor_asset_ids: list[str] = Field(min_length=1)
    camera_parameters: dict[str, list[float]] = Field(default_factory=dict)
    metrics: dict = Field(default_factory=dict)
    stability_runs: list[dict] = Field(default_factory=list)
    status: Literal["ready", "refused"] = "refused"
    refusal_reasons: list[str] = Field(default_factory=list)


class LocalizationAttempt(BaseModel):
    """Une tentative bornée de localisation d'une image originale."""

    model_config = ConfigDict(extra="forbid")

    attempt_id: str = Field(min_length=1)
    asset_id: str = Field(min_length=1)
    round_index: int = Field(default=0, ge=0)
    hop: int = Field(default=0, ge=0)
    correction_level: Literal[
        "original", "photometric", "deterministic_geometry", "virtual", "feedforward"
    ] = "original"
    original_image_digest: str = Field(min_length=64, max_length=64)
    derived_image_digest: str | None = Field(default=None, min_length=64, max_length=64)
    reference_asset_ids: list[str] = Field(default_factory=list)
    matches: int = Field(default=0, ge=0)
    inliers: int = Field(default=0, ge=0)
    inlier_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    reprojection_median_px: float | None = Field(default=None, ge=0.0)
    reprojection_p95_px: float | None = Field(default=None, ge=0.0)
    positive_depth_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    gps_residual_m: float | None = Field(default=None, ge=0.0)
    heading_residual_deg: float | None = Field(default=None, ge=0.0, le=180.0)
    stability_translation_m: float | None = Field(default=None, ge=0.0)
    stability_rotation_deg: float | None = Field(default=None, ge=0.0)
    pose_world_from_camera: dict | None = None
    decision: LocalizationDecision = LocalizationDecision.INSUFFICIENT_EVIDENCE
    evidence_class: PoseEvidenceClass = PoseEvidenceClass.REJECTED
    reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _measured_pose_requires_original_proof(self) -> "LocalizationAttempt":
        if self.evidence_class is PoseEvidenceClass.LOCALIZED_MEASURED:
            if self.decision is not LocalizationDecision.ACCEPTED:
                raise ValueError("une pose mesurée doit être acceptée")
            if self.pose_world_from_camera is None:
                raise ValueError("une pose mesurée doit porter sa transformation")
            if self.correction_level in {"virtual", "feedforward"}:
                raise ValueError(
                    "une tentative virtuelle/feed-forward ne peut pas être une pose mesurée"
                )
        return self


class LocalizedPoseEvidence(BaseModel):
    """Verdict final par image après toutes les tentatives."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(min_length=1)
    evidence_class: PoseEvidenceClass
    decision: LocalizationDecision
    hop: int = Field(default=0, ge=0)
    pose_world_from_camera: dict | None = None
    accepted_attempt_id: str | None = None
    reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _accepted_measurement_has_pose(self) -> "LocalizedPoseEvidence":
        if self.evidence_class in {
            PoseEvidenceClass.ANCHOR_MEASURED,
            PoseEvidenceClass.LOCALIZED_MEASURED,
        } and self.pose_world_from_camera is None:
            raise ValueError("une pose mesurée doit porter une transformation")
        return self


class LocalizationManifest(BaseModel):
    """Sortie canonique et append-only du pipeline de localisation ancrée."""

    model_config = ConfigDict(extra="forbid")

    contract_version: int = Field(default=1, ge=1)
    localization_run_id: str = Field(min_length=1)
    reconstruction_input_id: str = Field(min_length=1)
    anchor_model_id: str = Field(min_length=1)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    policy: AnchorLocalizationPolicy
    selected_asset_ids: list[str] = Field(min_length=1)
    poses: list[LocalizedPoseEvidence] = Field(min_length=1)
    attempts: list[LocalizationAttempt] = Field(default_factory=list)
    raw_registered_images: int = Field(default=0, ge=0)
    measured_anchor_images: int = Field(default=0, ge=0)
    measured_localized_images: int = Field(default=0, ge=0)
    inferred_images: int = Field(default=0, ge=0)
    rejected_images: int = Field(default=0, ge=0)
    validated_registration_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    validated_main_component_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    status: Literal["completed", "refused"] = "refused"
    refusal_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _counts_match_pose_evidence(self) -> "LocalizationManifest":
        counts = {
            PoseEvidenceClass.ANCHOR_MEASURED: self.measured_anchor_images,
            PoseEvidenceClass.LOCALIZED_MEASURED: self.measured_localized_images,
            PoseEvidenceClass.VIEW_INFERRED: self.inferred_images,
            PoseEvidenceClass.REJECTED: self.rejected_images,
        }
        actual = {kind: 0 for kind in counts}
        for pose in self.poses:
            actual[pose.evidence_class] += 1
        if counts != actual:
            raise ValueError(f"compteurs de localisation incohérents : {counts} != {actual}")
        if len({pose.asset_id for pose in self.poses}) != len(self.poses):
            raise ValueError("plusieurs verdicts finaux pour le même asset")
        return self


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
    orientation_xyzw: tuple[float, float, float, float] | None = None
    near_m: float = Field(default=0.05, gt=0.0)
    far_m: float = Field(default=10_000.0, gt=0.0)
    #: Orientation complète : matrice monde→caméra et quaternion associé.
    #: Toute validation consomme la pose entière — le yaw seul ne suffit pas
    #: à reconstruire une direction de visée qui porterait un pitch.
    orientation_matrix: tuple[float, float, float, float, float, float, float, float, float] | None = None
    orientation_quaternion: tuple[float, float, float, float] | None = None

    visible_surface_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    unknown_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    proxy_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    reconstructed_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    minimum_distance_violation: bool = False
    texture_reality_safe: bool | None = None
    texture_reality_level: str | None = None
    texture_reality_violations: list[str] = Field(default_factory=list)
    requested_output_width_px: int | None = Field(default=None, gt=0)
    visible_surfaces: list[dict] = Field(default_factory=list)
    target_pixel_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    unknown_visible_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    minimum_clearance_m: float | None = Field(default=None, ge=0.0)
    subject_score: float = Field(default=0.0, ge=0.0, le=1.0)
    accepted: bool = False
    rejection_reasons: list[str] = Field(default_factory=list)
    feasibility_mesh_digest: str | None = None
    collision: bool = False
    distance_to_scene_m: float | None = Field(default=None, ge=0.0)
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


# ---------------------------------------------------------------------------
# P3 — Hold-out plan, stability, fidelity, gaps
# ---------------------------------------------------------------------------


class HoldoutStrategy(StrEnum):
    """Stratégie de vues cachées pour la validation novel-view.

    Le 20 % fixe n'est qu'un **profil de benchmark**, pas une règle
    architecturale. Sur 200 vues il va bien ; sur une façade à trois vues
    indépendantes, retirer 20 % peut supprimer l'unique observation qui
    rend le problème reconstructible.
    """

    STRATIFIED_BY_TARGET = "stratified_by_target"
    LEAVE_ONE_VIEWPOINT_OUT = "leave_one_viewpoint_out"
    K_FOLD = "k_fold"


class HoldoutPlan(BaseModel):
    """Plan de vues cachées — interdit de rendre l'ensemble d'entraînement non reconstructible."""

    model_config = ConfigDict(extra="forbid")

    strategy: HoldoutStrategy
    benchmark_profile: float = Field(default=0.2, ge=0.0, le=1.0)
    preserve_reconstructibility: bool = True


class NovelViewCriteria(BaseModel):
    """Seuils de passage de la validation novel-view (calibrables, pas magiques)."""

    model_config = ConfigDict(extra="forbid")

    feature_inliers_min: float = Field(default=0.0, ge=0.0, le=1.0)
    edge_alignment_min: float = Field(default=0.0, ge=0.0, le=1.0)
    silhouette_iou_min: float = Field(default=0.0, ge=0.0, le=1.0)
    lpips_max: float = Field(default=1.0, ge=0.0, le=1.0)
    ssim_min: float = Field(default=0.0, ge=0.0, le=1.0)
    reprojection_px_max: float = Field(default=float("inf"), ge=0.0)
    structural_similarity_min: float = Field(default=0.0, ge=0.0, le=1.0)


class NovelViewValidationGate(BaseModel):
    """Validation sur vues cachées — détecte l'hallucination, pas la moyenne."""

    model_config = ConfigDict(extra="forbid")

    holdout_plan: HoldoutPlan
    feature_inliers: float = Field(default=0.0, ge=0.0, le=1.0)
    edge_alignment: float = Field(default=0.0, ge=0.0, le=1.0)
    silhouette_iou: float | None = Field(default=None, ge=0.0, le=1.0)
    lpips: float = Field(default=1.0, ge=0.0, le=1.0)
    ssim: float = Field(default=0.0, ge=0.0, le=1.0)
    reprojection_px: float = Field(default=0.0, ge=0.0)
    structural_similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    normalized_reprojection_score: float | None = Field(default=None, ge=0.0, le=1.0)
    metric_status: dict[str, str] = Field(default_factory=dict)
    pass_criteria: NovelViewCriteria

    #: Les métriques ci-dessus proviennent-elles d'un rendu réellement comparé
    #: aux vues cachées ? Faux tant qu'aucun moteur de rendu n'est disponible :
    #: des valeurs par défaut ne sont pas une mesure et ne doivent jamais
    #: valoir PASS.
    metrics_measured: bool = False
    #: Motif quand `metrics_measured` est faux.
    unmeasured_reason: str | None = None
    #: Identifiants des vues effectivement retenues comme cachées.
    held_out_asset_ids: list[str] = Field(default_factory=list)
    train_asset_ids: list[str] = Field(default_factory=list)
    holdout_leakage_count: int = Field(default=0, ge=0)
    frozen_model_digest: str | None = None
    holdout_results: list[dict] = Field(default_factory=list)
    surface_scores: dict[str, dict] = Field(default_factory=dict)


class StabilityRun(BaseModel):
    """Un run de reconstruction sur un sous-corpus dégradé."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    corpus_fraction: float = Field(ge=0.0, le=1.0)
    n_cameras: int = Field(ge=0)
    status: str
    alignment_rmse_m: float = Field(default=0.0, ge=0.0)


class StabilityResult(StrEnum):
    """Résultat d'une validation de stabilité.

    `INSUFFICIENT_EVIDENCE` préserve l'inconnu : ce n'est pas un échec
    géométrique démontré, c'est une impossibilité de conclure. Pour un
    MUST_SHOW, cela bloque quand même le pipeline, mais avec une raison
    explicite.
    """

    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class StabilityManifest(BaseModel):
    """Stabilité / ablations — artefact canonique, pas un contrôle ad-hoc.

    On relance la reconstruction sur des sous-corpus dégradés (100/90/80 %)
    et on aligne les runs par Sim(3) pour mesurer la dérive.
    """

    model_config = ConfigDict(extra="forbid")

    stability_id: str = Field(min_length=1)
    baseline_run_id: str = Field(min_length=1)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    corpus_100: StabilityRun
    corpus_90: StabilityRun
    corpus_80: StabilityRun
    aligned_camera_drift: float = Field(ge=0.0)
    geometry_drift: float = Field(ge=0.0)
    target_surface_drift: dict[str, float] = Field(default_factory=dict)
    result: StabilityResult


class GateResult(StrEnum):
    """État final d'une porte — états qui préservent l'inconnu."""

    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NOT_APPLICABLE = "not_applicable"


class FidelityGate(BaseModel):
    """Porte finale de fidélité — arrêt dur sur MUST_SHOW.

    Pour un MUST_SHOW, toutes les portes doivent PASSER. Un
    `INSUFFICIENT_EVIDENCE` bloque aussi, mais avec une raison enregistrée :
    ce n'est pas un échec géométrique démontré, c'est une impossibilité de
    conclure.
    """

    model_config = ConfigDict(extra="forbid")

    target_id: str = Field(min_length=1)
    criticality: Criticality
    sparse_gate: "SparseConsensusGate | None" = None
    geo_gate: "GeoAlignmentGate | None" = None
    dense_gate: "DenseFidelityResult | None" = None
    novel_view_gate: NovelViewValidationGate | None = None
    stability_gate: StabilityResult | None = None
    unsupported_geometry_gate: bool = False

    overall: GateResult = GateResult.INSUFFICIENT_EVIDENCE


class GapType(StrEnum):
    """Nature d'une lacune de reconstruction — chaque valeur donne une action ciblée."""

    DISCONNECTED_GRAPH = "disconnected_graph"
    LOW_PARALLAX = "low_parallax"
    LOW_SUPPORT = "low_support"
    POSE_UNCERTAINTY = "pose_uncertainty"
    APPEARANCE_GAP = "appearance_gap"
    GEO_MISMATCH = "geo_mismatch"


class ReconstructionGap(BaseModel):
    """Une lacune structurée — ce qui déclenche la collecte active (P4).

    Cet objet, pas un vague « besoin de plus de photos », génère le prochain
    `CaptureDemand`. Chaque `gap_type` entraîne une observation ciblée :
    secteur, baseline, angle de vue préférés.
    """

    model_config = ConfigDict(extra="forbid")

    target_id: str = Field(min_length=1)
    gap_type: GapType
    affected_assets: list[str] = Field(default_factory=list)
    affected_viewgraph_components: list[str] = Field(default_factory=list)
    required_observation: str = Field(min_length=1)
    preferred_sector: str = Field(default="")
    preferred_baseline: float = Field(default=0.0, ge=0.0)
    preferred_view_angle: float = Field(default=0.0, ge=0.0, lt=360.0)
    priority: int = Field(default=1, ge=1)


class SurfaceConfidenceScore(BaseModel):
    """Score agrégé versionné — interprétation, pas preuve.

    Les composantes de `SurfaceConfidence` sont la preuve ; ce score n'est
    qu'une interprétation, et doit donc porter son modèle, sa calibration
    et l'empreinte des composantes qu'il agrège.
    """

    model_config = ConfigDict(extra="forbid")

    target_id: str = Field(min_length=1)
    score_model_id: str = Field(min_length=1)
    calibration_id: str = Field(min_length=1)
    component_digest: str = Field(min_length=1)
    value: float = Field(default=0.0, ge=0.0, le=1.0)


class SparseConsensusGate(BaseModel):
    """Porte A : consensus des solveurs creux (avant dense)."""

    model_config = ConfigDict(extra="forbid")

    #: Mesures brutes conservées pour diagnostic. Elles ne suffisent jamais à
    #: faire passer G5.
    raw_registration_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    raw_registered_images: int = Field(default=0, ge=0)
    #: Mesure probante issue d'un LocalizationManifest. `registration_rate`
    #: reste l'alias historique et reçoit cette valeur validée lorsqu'elle est
    #: disponible.
    registration_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    validated_registration_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    measured_anchor_images: int = Field(default=0, ge=0)
    measured_localized_images: int = Field(default=0, ge=0)
    inferred_images: int = Field(default=0, ge=0)
    rejected_images: int = Field(default=0, ge=0)
    validated_main_component_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    external_pose_consistency: bool = False
    localization_manifest_id: str | None = None
    largest_component_size: int = Field(default=0, ge=0)
    median_reprojection_px: float = Field(default=0.0, ge=0.0)
    median_reprojection_normalized: float = Field(default=0.0, ge=0.0, le=1.0)
    track_length_median: float = Field(default=0.0, ge=0.0)
    inlier_ratio_median: float = Field(default=0.0, ge=0.0, le=1.0)
    intrinsics_quality: str = Field(default="poor")
    camera_consensus: dict = Field(default_factory=dict)
    solver_families: list[str] = Field(default_factory=list)
    independent_families_agreeing: int = Field(default=0, ge=0)


class GeoGateCriteria(BaseModel):
    """Seuils de passage de la porte d'alignement géospatial (calibrables)."""

    model_config = ConfigDict(extra="forbid")

    footprint_error_max_m: float = Field(default=2.0, ge=0.0)
    roof_height_error_max_m: float = Field(default=2.0, ge=0.0)
    scale_error_max: float = Field(default=0.1, ge=0.0)
    orientation_error_max_deg: float = Field(default=5.0, ge=0.0)
    alignment_rmse_max_m: float = Field(default=2.0, ge=0.0)


class GeoAlignmentGate(BaseModel):
    """Porte B : alignement géospatial (échelle métrique)."""

    model_config = ConfigDict(extra="forbid")

    footprint_error_m: float = Field(default=0.0, ge=0.0)
    roof_height_error_m: float = Field(default=0.0, ge=0.0)
    scale_error: float = Field(default=0.0, ge=0.0)
    orientation_error_deg: float = Field(default=0.0, ge=0.0)
    alignment_rmse_m: float = Field(default=0.0, ge=0.0)
    anchors: list[str] = Field(default_factory=list)
    pass_criteria: GeoGateCriteria


class DenseFidelityResult(BaseModel):
    """Résultat de la reconstruction dense (Brush/gsplat)."""

    model_config = ConfigDict(extra="forbid")

    method: str = Field(min_length=1)
    geometry_rmse_m: float = Field(default=0.0, ge=0.0)
    appearance_lpips: float = Field(default=1.0, ge=0.0, le=1.0)
    coverage_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    status: str = Field(default="pending")


__all__ = [
    "ReconstructionInputManifest",
    "ReconstructionSelection",
    "ReconstructionSelectionManifest",
    "Criticality",
    "SupportType",
    "SupportRole",
    "ReconstructionTargetKind",
    "ReconstructionTarget",
    "AssetTargetSupport",
    "ViewGraphNode",
    "PairEvidence",
    "ViewGraphReport",
    "ViewGraphManifest",
    "ReconstructionPlan",
    "ReconstructionRun",
    "PoseEvidenceClass",
    "LocalizationDecision",
    "PosePriorKind",
    "PosePrior",
    "AnchorLocalizationPolicy",
    "AnchorCandidateEvidence",
    "AnchorSelectionManifest",
    "AnchorModelManifest",
    "LocalizationAttempt",
    "LocalizedPoseEvidence",
    "LocalizationManifest",
    "ReconstructionConsensusReport",
    "CameraConsensusEntry",
    "GeoAlignmentManifest",
    "SurfaceConfidence",
    "SurfaceConfidenceManifest",
    "GeoGateCriteria",
    "GeoAlignmentGate",
    "SurfaceConfidenceScore",
    "CameraFeasibilityField",
    "ValidatedCameraPath",
    "ScenePackageType",
    "ReconstructionGateStatus",
    "ReconstructionBackend",
    "AlignmentAnchor",
    "HoldoutStrategy",
    "HoldoutPlan",
    "NovelViewCriteria",
    "NovelViewValidationGate",
    "StabilityRun",
    "StabilityResult",
    "StabilityManifest",
    "GateResult",
    "FidelityGate",
    "GapType",
    "ReconstructionGap",
    "SparseConsensusGate",
    "DenseFidelityResult",
]
