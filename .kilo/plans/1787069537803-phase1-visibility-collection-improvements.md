# Phase 1 → Lot 2: Promotional Video Pipeline Implementation Plan (Frozen: Lot 2 V1)

**Goal**: Fidelity reconstruction on promotion-critical views + GENERATIVE_COMPLETION ONLY when capture is genuinely impossible → promotional video via validated camera path + dense rendering.

> Note: "AI reconstruction" is not the trigger. GLUEMAP / MapAnything / VGGT / 3DGS may use AI without inventing geometry. Only `GENERATIVE_COMPLETION` (fabricating unseen appearance/geometry) is gated on proven capture impossibility. The `SupportType` taxonomy below (`MEASURED_PHOTO` / `MULTIVIEW_RECONSTRUCTED` / `FEEDFORWARD_INFERRED` / `GENERATIVE_COMPLETION`) is exactly what enforces that distinction.

---

## Core Architectural Principles

1. **No hardcoded surface categories** — Promotion intent drives criticality
2. **Gates yes, thresholds calibrated** — No magic numbers (0.80, 1.5px, 3 solvers, 0.7)
3. **Correct gate order** — Preprocess → ViewGraph → Sparse consensus → GeoAlignment → Dense → SurfaceConfidence + hold-out → Fidelity → (only if insufficient) Active Collection
4. **Reproducible derived artifacts are first-class inputs** — Masks, EXIF/orientation normalization, camera models, and intrinsics calibration are produced as versioned, digest-tracked artifacts *before* matching, not ad-hoc afterthoughts
5. **Test SfM first** — Existing corpus (335 assets, 269 photos, 209 viewpoints, LiDAR, DTM, roof) → ViewGraph → COLMAP/GLUEMAP → first real measurement
6. **AI taxonomy explicit** — GENERATIVE_COMPLETION forbidden on MUST_SHOW, allowed on OPTIONAL with proof

---

## P0: Contract Repairs (Before Any New Code)

### P0.1 Separate appearance_coverage from geometric_support
**File**: `src/hotel_pipeline/coverage.py`, `src/hotel_pipeline/satellite_completion.py`
- `zone_confidence.geojson` currently mixes orthophoto-derived geometry with photographic evidence
- Split into `appearance_coverage` (photographs only) and `geometric_support` (LiDAR/ortho/satellite)
- Satellite completion writes `geometric_support` with `source: orthophoto`, `confidence: low`, `appearance: none`

### P0.2 ReconstructionTarget + Criticality (replaces PRIMARY/SECONDARY)
**File**: `src/hotel_pipeline/schemas/reconstruction.py` (new)
```python
class ReconstructionTarget(BaseModel):
    target_id: str                    # FACADE_PRIMARY, ENTRANCE, POOL, GARDEN, ROOF, etc.
    kind: TargetKind                  # SURFACE | OBJECT | LINEAR_FEATURE | AREA | CONTEXT
    criticality: Criticality          # MUST_SHOW | SHOULD_SHOW | OPTIONAL | CONTEXT_ONLY
    required_fidelity: float          # 0.0-1.0 (calibrated later)
    allowed_support: list[SupportType] # see taxonomy below
    maximum_generative_completion: float  # 0.0 for MUST_SHOW — never invent facade appearance
    maximum_inferred_geometry: float      # policy/calibrated — feed-forward (MapAnything/VGGT) allowed as geometric prior/cross-check
    minimum_camera_distance_m: float      # meters for video planning

class Criticality(StrEnum):
    MUST_SHOW = "must_show"
    SHOULD_SHOW = "should_show"
    OPTIONAL = "optional"
    CONTEXT_ONLY = "context_only"
    FORBIDDEN = "forbidden"

class SupportType(StrEnum):
    MEASURED_PHOTO = "measured_photo"
    MEASURED_LIDAR = "measured_lidar"
    MULTIVIEW_RECONSTRUCTED = "multiview_reconstructed"
    FEEDFORWARD_INFERRED = "feedforward_inferred"
    GEOSPATIAL_PROXY = "geospatial_proxy"
    GENERATIVE_COMPLETION = "generative_completion"
```

### P0.3 AssetTargetSupport (replaces tier_assignment)
**File**: `src/hotel_pipeline/schemas/reconstruction.py`
```python
class AssetTargetSupport(BaseModel):
    asset_id: str
    target_id: str
    support_role: SupportRole
    coverage_fraction: float
    quality_score: float
    reconstruction_role: ReconstructionRole
```

---

## P1: Lot 2 Foundation (Test SfM with Existing Corpus)

Pipeline chain for P1:
```
ReconstructionInputManifest → PreprocessManifest (masks) → ViewGraph → Sparse (intrinsics-calibrated) → Gate A
```

### P1.1 ReconstructionInputManifest
**File**: `src/hotel_pipeline/schemas/reconstruction.py`, `src/hotel_pipeline/reconstruction_input.py`
```python
class ReconstructionInputManifest(BaseModel):
    reconstruction_input_id: str
    hotel_id: str
    asset_manifest_digest: str
    spatial_manifest_digest: str
    site_manifest_digest: str
    coverage_digest: str
    router_decision_digest: str
    targets: list[ReconstructionTarget]
    selected_asset_ids: list[str]
    asset_target_support: list[AssetTargetSupport]
    capture_feasibility_assessment_refs: dict[str, str] = {}
    mask_set_digest: str | None
    created_at: datetime
```
- **Temporal coherence**: an initial P1 manifest must NOT depend on a P4 artifact. `CaptureFeasibilityAssessment` is only built in P4 (after the first reconstruction), so it cannot appear as a populated field here. We store only references (`target_id -> assessment_id`); on a later P4 → P1 loop, the new snapshot can resolve them. An initial Lot 2 run simply leaves this empty.

### P1.2 PreprocessManifest + Masking (NEW — before ViewGraph)
**File**: `src/hotel_pipeline/preprocess.py` (new), `src/hotel_pipeline/schemas/reconstruction.py`
- The manifest already declares `mask_set_digest`, but no step actually fabricates those masks. Before any SIFT/ALIKED/LightGlue, produce reproducible derived artifacts:
  - **Masks**: sky, persons, vehicles, water, specular reflections (pool glare, glass), vegetation optionally
  - **EXIF/orientation normalization**: consistent image orientation, color space, gain
  - **Per-image camera model seed**: sensor + nominal intrinsics from EXIF, flagged for later calibration
  - **Temporal cohorts** (optional): group by capture epoch for change-resilience
- All outputs are versioned and digest-tracked so the ViewGraph is reproducible.
```python
class PreprocessManifest(BaseModel):
    preprocess_id: str
    reconstruction_input_id: str
    mask_set_digest: str
    masked_asset_ids: list[str]
    exif_normalization: NormalizationReport
    camera_model_seeds: dict[str, CameraSeed]   # asset_id -> seed intrinsics
    temporal_cohorts: list[TemporalCohort] | None
    created_at: datetime
```
- Without this, LightGlue risks building edges on cars, vegetation, and pool reflections.

### P1.3 ViewGraphManifest + LightGlue Integration (strengthened)
**File**: `src/hotel_pipeline/schemas/reconstruction.py`, `src/hotel_pipeline/view_graph.py`
- Retrieval: SALAD/hloc global descriptors
- Matching: SIFT + LightGlue (baseline) + ALIKED + LightGlue (R&D)
- Geometric verification: Essential/Fundamental matrix + RANSAC
- **NEW — Homography + degeneracy detection**: critical on near-planar facades; flag planar-degenerate pairs so they are not over-trusted for triangulation
- **NEW — `repetitive_structure_risk`**: explicitly detect repetitive patterns (windows, balconies, columns)
- **NEW — Doppelgangers++ / GLUEMAP-equivalent protection**: keep duplicate-structure rejection as a first-class safeguard for hotel facades
- Output: ViewGraphManifest with nodes, edges, ViewGraphReport
```python
class ViewGraphManifest(BaseModel):
    view_graph_id: str
    nodes: list[ViewNode]
    edges: list[ViewEdge]
    homography_degeneracy_flags: dict[str, bool]   # edge_id -> planar-degenerate
    repetitive_structure_risk: float
    doppelganger_rejections: int
    report: ViewGraphReport
```

### P1.4 Sparse Reconstruction (COLMAP + GLUEMAP) + Intrinsics Calibration
**File**: `src/hotel_pipeline/reconstruction_run.py`, `src/hotel_pipeline/reconstruction_plan.py`
- **NEW — Intrinsics calibration step between ViewGraph and COLMAP Global**:
  ```
  view_graph → intrinsics_quality check → view_graph_calibrator (if needed) → global mapper
  ```
  Good connectivity with wrong focal lengths still yields a globally wrong reconstruction; calibrate or verify intrinsics first.
- COLMAP incremental (baseline)
- COLMAP global (after intrinsics resolved)
- GLUEMAP (hybrid)
- MP-SfM (optional)
- **Gate A: Camera/Pose Consensus** — after sparse, before dense

### P1.5 Gate A: Camera/Pose Consensus (Sparse Gate)
**File**: `src/hotel_pipeline/reconstruction_consensus.py`
```python
class SparseConsensusGate(BaseModel):
    registration_rate: float
    largest_component_size: int
    median_reprojection_px: float
    median_reprojection_normalized: float  # error / image_diagonal
    track_length_median: float
    inlier_ratio_median: float
    intrinsics_quality: IntrinsicsQuality    # NEW: calibrated vs seeded, residual
    camera_consensus: CameraConsensusResult
    solver_families: list[SolverFamily]    # CLASSICAL | HYBRID | FEEDFORWARD | MONOCULAR_PRIOR | LIDAR
    independent_families_agreeing: int

    pass_criteria: SparseGateCriteria
```

---

## P2: GeoAlignment

### P2.1 GeoAlignment with LiDAR/DTM/DSM
**File**: `src/hotel_pipeline/geo_alignment.py` (existing, enhance)
- Inputs: BUILDING_MAIN footprint (XY), LiDAR roof points (Z), DTM/DSM (terrain), nDSM (height prior)
- Output: GeoAlignmentManifest with Sim(3) parameters, alignment_rmse, anchors

### P2.2 Gate B: GeoAlignmentGate
```python
class GeoAlignmentGate(BaseModel):
    footprint_error_m: float
    roof_height_error_m: float
    scale_error: float
    orientation_error_deg: float
    alignment_rmse_m: float
    anchors: list[AlignmentAnchor]
    pass_criteria: GeoGateCriteria
```

---

## P3: Dense Reconstruction + Validation

### P3.1 Dense Reconstruction (Brush default / gsplat R&D-fallback)
**File**: `src/hotel_pipeline/dense_reconstruction.py`
- **Correction**: Brush and gsplat are NOT two fidelity tiers. Brush stays the **default production backend**; gsplat is the **R&D / fallback / custom-loss backend**.
- Method selection comes from `ReconstructionPlan` and measured results — NOT from target criticality.
```python
class DenseReconstructionConfig(BaseModel):
    default_method: str = "brush"        # production default backend
    alternative_method: str = "gsplat"   # R&D / fallback / custom-loss backend
    method_selection_policy: str = "reconstruction_plan"  # NOT criticality-driven
    depth_priors: dict[str, str]         # asset_id -> "mapanything" | "vggt" | "lidar"
    generative_completion_allowed: list[str]  # target_ids where GENERATIVE_COMPLETION permitted
```

### P3.2 SurfaceConfidence (Decomposed, contradiction fixed)
**File**: `src/hotel_pipeline/surface_confidence.py`
- The decomposed components are the **evidence**; the aggregated score is only a versioned interpretation.
- `effective_confidence` must NOT be a plain stored field. Make it a `@computed_field`, or move it to a separate `SurfaceConfidenceScore` that is required to carry `score_model_id`, `calibration_id`, `component_digest`.
```python
class SurfaceConfidence(BaseModel):
    target_id: str
    # Decomposed components (the evidence, NOT a single score):
    geometry_confidence: float
    appearance_confidence: float
    pose_confidence: float
    observation_support: float        # independent views
    angular_diversity: float          # baseline/angle coverage
    cross_solver_agreement: float     # COLMAP vs GLUEMAP vs feed-forward
    geo_alignment_confidence: float
    novel_view_score: float           # hold-out validation
    extrapolation_penalty: float      # 0.0 for MEASURED, >0 for FEEDFORWARD/GENERATIVE
    support_breakdown: dict[SupportType, float]
    component_digest: str             # pins the exact component set

    # Aggregated score is a SEPARATE, versioned object — never stored as ground truth:
    # SurfaceConfidenceScore(score_model_id, calibration_id, component_digest, value)

class SurfaceConfidenceScore(BaseModel):
    target_id: str
    score_model_id: str               # which aggregation model produced it
    calibration_id: str               # which calibration/threshold set
    component_digest: str             # must match SurfaceConfidence.component_digest
    value: float                      # interpretation only
```

### P3.3 Gate C: Held-Out Novel View Validation (strategy, not fixed fraction)
**File**: `src/hotel_pipeline/novel_view_validation.py`
- **Correction**: `hidden_fraction = 0.2` is a benchmark profile, NOT an architectural rule. On 200 views it is fine; on a facade with 3 independent views, removing 20% can delete the single observation that makes the problem reconstructible.
- Replace with a configurable `HoldoutPlan` that is forbidden from making the training set non-reconstructible.
```python
class HoldoutPlan(BaseModel):
    strategy: HoldoutStrategy          # STRATIFIED_BY_TARGET | LEAVE_ONE_VIEWPOINT_OUT | K_FOLD
    benchmark_profile: float = 0.2     # 20% used only as a benchmark profile
    preserve_reconstructibility: bool = True   # training set must remain reconstructible

class NovelViewValidationGate(BaseModel):
    holdout_plan: HoldoutPlan
    feature_inliers: float
    edge_alignment: float
    silhouette_iou: float
    lpips: float
    ssim: float
    reprojection_px: float
    structural_similarity: float
    pass_criteria: NovelViewCriteria
```

### P3.4 StabilityManifest (Stability / Ablations Result)
**File**: `src/hotel_pipeline/stability.py` (new)
- Produced between hold-out novel-view validation and the final fidelity gate.
- Re-runs reconstruction on degraded corpora and aligns runs via Sim(3) to quantify drift. This is a canonical artifact, not an ad-hoc check.
```python
class StabilityManifest(BaseModel):
    stability_id: str
    baseline_run_id: str
    corpus_100: StabilityRun
    corpus_90: StabilityRun
    corpus_80: StabilityRun
    aligned_camera_drift: float                # Sim(3) drift vs baseline
    geometry_drift: float                      # surface geometry drift
    target_surface_drift: dict[str, float]     # target_id -> drift
    result: StabilityResult                    # PASS | FAIL | INSUFFICIENT_EVIDENCE
```

### P3.5 Gate D: FidelityGate (Final Hard Gate — unknown-preserving states)
**File**: `src/hotel_pipeline/fidelity_gate.py`
- **Correction**: replace `PASS | FAIL | CONDITIONAL` with states that preserve the unknown.
  - `PASS` — all required evidence supports fidelity
  - `FAIL` — a demonstrated geometric/appearance failure
  - `INSUFFICIENT_EVIDENCE` — cannot conclude (NOT a proven failure); for MUST_SHOW this still blocks the pipeline, but the reason is explicit
  - `NOT_APPLICABLE` — target out of current scope
```python
class FidelityGate(BaseModel):
    target_id: str
    criticality: Criticality
    sparse_gate: SparseConsensusGate
    geo_gate: GeoAlignmentGate
    dense_gate: DenseFidelityResult
    novel_view_gate: NovelViewValidationGate
    stability_gate: StabilityResult
    unsupported_geometry_gate: bool

    overall: GateResult  # PASS | FAIL | INSUFFICIENT_EVIDENCE | NOT_APPLICABLE

    # MUST_SHOW targets: ALL gates must PASS (INSUFFICIENT_EVIDENCE also blocks)
    # SHOULD_SHOW: dense + novel_view must PASS
    # OPTIONAL: best-effort
    # INSUFFICIENT_EVIDENCE on a MUST_SHOW → pipeline stops, reason recorded
```

---

## P4: Active Collection Enhancements (After First Measurement)

### P4.1 RoadAccessGraph + CaptureFeasibilityAssessment
**File**: `src/hotel_pipeline/schemas/geometry.py`, `src/hotel_pipeline/geo/geometry_loader.py`
```python
class RoadSegment(BaseModel):
    feature_id: str
    geometry: LineString
    access_status: AccessStatus          # PUBLIC_CONFIRMED | PRIVATE_CONFIRMED | RESTRICTED | UNKNOWN
    service: str | None
    reachability_status: ReachabilityStatus  # REACHABLE | UNREACHABLE | UNKNOWN
    camera_candidate: bool
    streetview_candidate: bool
    mapillary_candidate: bool

class CaptureFeasibilityAssessment(BaseModel):
    target_id: str
    status: FeasibilityStatus            # FEASIBLE | INFEASIBLE_PROVEN | NOT_FOUND_REMOTELY | OWNER_CAPTURE_REQUIRED | UNKNOWN
    remote_public: FeasibilityDetail
    owner_assisted: FeasibilityDetail
    professional_onsite: FeasibilityDetail
    physically_impossible: FeasibilityDetail
    evidence: list[str]
    assessed_at: datetime
```

### P4.2 LiDAR Obstacle Heights (for BLOCKED verdicts)
**File**: `src/hotel_pipeline/lidar_support.py`, `src/hotel_pipeline/geo/visibility_run.py`
```python
def extract_obstacle_heights_from_lidar(workspace, obstacle_footprints: dict[str, Polygon]) -> dict:
    """Per-obstacle: ground_z from DTM, top_z from classified surface points, height = top_z - ground_z"""
    # Returns: {feature_id: {'height_m': float, 'ground_m': float, 'quality': str, 'point_count': int}}
```

### P4.3 STRtree Spatial Index
**File**: `src/hotel_pipeline/geo/visibility_engine.py`
```python
from shapely.strtree import STRtree
# In assess(): obstacle_tree = STRtree([obs.shape for obs in obstacles])
# In _assess_cell(): candidate_indices = obstacle_tree.query(ray.bounds)
```

### P4.4 AcquisitionPortfolioOptimizer (Two Gates)
**File**: `src/hotel_pipeline/portfolio_optimizer.py` (new), `src/hotel_pipeline/plan.py`
```python
# Gate 1: Pre-SfM Collection Gate (does NOT block the first ViewGraph)
# - minimum candidate observations
# - angular diversity
# - resolution
# - currentness
# Result: READY_TO_ATTEMPT | WEAK_BUT_ATTEMPT | STRUCTURALLY_IMPOSSIBLE
#   Only STRUCTURALLY_IMPOSSIBLE blocks. For the existing corpus we attempt anyway:
#   the ViewGraph itself is the measurement we seek. Post-ViewGraph Gate is the real verdict.

# Gate 2: Post-ViewGraph Gate
# - actual overlap
# - valid pair graph
# - connectivity
# - triangulation
# Result: RECONSTRUCTION_VIABLE
```

### P4.5 ReconstructionGapAnalysis (contract that triggers P4)
**File**: `src/hotel_pipeline/gap_analysis.py` (new)
- Consumes FidelityGate + ViewGraph + SurfaceConfidence to produce structured gaps.
- This object — not a vague "need more photos" — generates the next `CaptureDemand`.
```python
class ReconstructionGap(BaseModel):
    target_id: str
    gap_type: GapType   # DISCONNECTED_GRAPH | LOW_PARALLAX | LOW_SUPPORT |
                         # POSE_UNCERTAINTY | APPEARANCE_GAP | GEO_MISMATCH
    affected_assets: list[str]
    affected_viewgraph_components: list[str]
    required_observation: str
    preferred_sector: str
    preferred_baseline: float
    preferred_view_angle: float
    priority: int
```

### P4.6 Reconstruction-Guided Active Collection Loop
```python
# ViewGraph → ReconstructionGapAnalysis → ReconstructionGap → Targeted CaptureDemand → Lot1B discovery/acquisition → update ViewGraph
```
- This is the key P4 ordering point: collection is driven by a *demonstrated* ViewGraph/reconstruction gap, not by "looks like we're missing photos".

---

## P5: Camera & Video

### P5.1 CameraFeasibilityField (Weighted)
**File**: `src/hotel_pipeline/camera_feasibility.py`
```python
def evaluate_camera_feasibility(camera_state, surface_confidence, dense_model):
    required_confidence = f(distance, projected_area, detail_level, target_criticality)
    # MUST_SHOW targets in frustum: require confidence ≥ required_confidence
    # Weight by screen_area × criticality / distance
```

### P5.2 ValidatedCameraPath
**File**: `src/hotel_pipeline/camera_feasibility.py`
```python
def plan_validated_camera_path(surface_confidence, dense_model, keyframes) -> ValidatedCameraPath:
    # Interpolate through feasible corridor (CameraFeasibilityField as cost map)
    # Output: time-parameterized SE(3) trajectory
```

### P5.3 Video Rendering — FUTURE OUTPUT INTERFACE (removed from implementation scope)
- **Out of current implementation plan.** Left as a future output interface only.
- The real definition of DONE for this plan is: `ValidatedCameraPath + validated dense scene + FidelityGate PASS`.
- When re-added later: Input = ValidatedCameraPath + dense_model (Brush/gsplat); Output = 4K MP4, 30fps; Quality gate = no visible artifacts on MUST_SHOW surfaces.

---

## Implementation Order (Corrected)

| Phase | Task | Why First |
|-------|------|-----------|
| **P0** | appearance_coverage ≠ geometric_support | Fixes false "coverage" feeding everything downstream |
| **P0** | ReconstructionTarget + Criticality + AssetTargetSupport | Replaces hardcoded PRIMARY/SECONDARY |
| **P1** | ReconstructionInputManifest | Contract for Lot 2 |
| **P1** | PreprocessManifest + Masks | Reproducible masks/EXIF/intrinsics seeds before matching |
| **P1** | ViewGraphManifest + LightGlue (+ H/F degeneracy, repetitive risk, doppelgangers) | First real SfM measurement |
| **P1** | COLMAP + GLUEMAP sparse (+ intrinsics calibration) | **First real measurement** — answers "can we reconstruct?" |
| **P1** | Camera/Pose Consensus (Gate A) | Validates sparse before dense |
| **P2** | GeoAlignment + Gate B | Metric scale |
| **P3** | Dense (Brush default / gsplat R&D) + SurfaceConfidence | Geometry + appearance |
| **P3** | Hold-out Plan + Novel View Validation (Gate C) | Detects hallucination |
| **P3** | FidelityGate (Gate D) | **Hard stop on MUST_SHOW / INSUFFICIENT_EVIDENCE** |
| **P4** | RoadAccessGraph + CaptureFeasibilityAssessment | Only after knowing what's missing |
| **P4** | LiDAR obstacle heights | For BLOCKED verdicts in impossibility proofs |
| **P4** | STRtree index | Speed for iterative loops |
| **P4** | PortfolioOptimizer (Pre-SfM: READY/WEAK/STRUCTURALLY_IMPOSSIBLE; Post-ViewGraph: viability) | Guides collection from reconstruction gaps |
| **P5** | CameraFeasibilityField + ValidatedCameraPath | Video trajectory (DONE = validated path + scene + gate PASS) |
| _future_ | Video rendering | Out of scope for current implementation |

---

## Validation Gates Summary

| Gate | When | Input | On FAIL / BLOCK (MUST_SHOW) |
|------|------|-------|---------------------|
| **Pre-SfM Collection** | End of `collect` | Candidates + angular diversity | Only STRUCTURALLY_IMPOSSIBLE blocks; WEAK_BUT_ATTEMPT still builds ViewGraph |
| **Post-ViewGraph** | After ViewGraph | Pair graph connectivity | StepBlocked — reconstruction not viable |
| **A: Sparse Consensus** | After sparse solvers | Cameras, tracks, reprojection, intrinsics | **STOP** — cannot proceed to dense |
| **B: GeoAlignment** | After GeoAlignment | Footprint, roof, scale, orientation | **STOP** — metric scale broken |
| **C: Novel View** | After dense | Hold-out plan (strategy, not fixed 20%) | **STOP** — hallucination detected |
| **D: Fidelity** | Final | All above + stability | **STOP PIPELINE** — video not producible (PASS \| FAIL \| INSUFFICIENT_EVIDENCE \| NOT_APPLICABLE) |

---

## Corrected End-to-End Chain

```text
P0 CONTRACTS
appearance != geometry
ReconstructionTarget
AssetTargetSupport
        ↓
P1 INPUT + PREPROCESS
ReconstructionInputManifest
PreprocessManifest / Masks
        ↓
VIEW GRAPH
retrieval
SIFT/ALIKED + LightGlue
E/F/H + RANSAC
repetitive-structure checks
        ↓
SPARSE
COLMAP Incremental
COLMAP Global (after intrinsics calibration)
GLUEMAP
MP-SfM if needed
        ↓
GATE A
Camera/Pose Consensus
        ↓
P2 GEO ALIGNMENT
Sim(3)
footprint + LiDAR + DTM/DSM
        ↓
GATE B
        ↓
P3 DENSE
Brush default
gsplat R&D/fallback
        ↓
SURFACE CONFIDENCE
decomposed metrics
        ↓
HOLDOUT PLAN
novel-view validation
        ↓
STABILITY / ABLATIONS
        ↓
GATE C + FINAL FIDELITY GATE
        ↓
if insufficient (INSUFFICIENT_EVIDENCE / FAIL):
    ReconstructionGapAnalysis
           ↓
    P4 Active Collection
           ↓
    rebuild ViewGraph
           └──────────────↺
        ↓
CAMERA FEASIBILITY
        ↓
VALIDATED CAMERA PATH
        ↓
STOP (DONE = ValidatedCameraPath + validated dense scene + FidelityGate PASS)
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| MUST_SHOW impossible (building against neighbor) | CaptureFeasibilityAssessment = PHYSICALLY_IMPOSSIBLE → pipeline stops honestly |
| Insufficient Street View for rear | Owner-assisted capture demanded before GENERATIVE_COMPLETION |
| LiDAR missing | DTM + **nDSM indépendant s'il existe** + OSM building:levels heuristic (flagged lower quality). If the nDSM was itself derived from the absent LiDAR, it is NOT a valid fallback |
| Feed-forward hallucination | Novel View Gate (hold-out strategy) + extrapolation_penalty |
| Camera path through AI regions | CameraFeasibilityField penalizes GENERATIVE_COMPLETION |
| Solver correlation | SolverFamily taxonomy (CLASSICAL/HYBRID/FEEDFORWARD/MONOCULAR/LIDAR) |
| Wrong intrinsics with good connectivity | Intrinsics quality check + view_graph_calibrator before global mapper |
| Planar-degenerate / repetitive facades | Homography degeneracy flags + repetitive_structure_risk + doppelganger rejection |

---

## Out of Scope

- Real-time video generation
- Dynamic lighting/time-of-day rendering
- Interior reconstruction
- Multi-building campus tours
- Probabilistic heights as truth
- Fixed thresholds presented as authoritative
- Video rendering as a current implementation deliverable (future output interface only)

---

## Freeze: Lot 2 V1 — Architecture Approved

With the corrections above, this plan is **frozen as Lot 2 V1** and implementation begins. No further plan revisions before the first LightGlue measurement — else we risk infinite architecture before ever trying SfM.

### First short sprint (measure before building more)

```text
P0.1       appearance_coverage ≠ geometric_support
P0.2/P0.3  ReconstructionTarget + AssetTargetSupport
P1.1       ReconstructionInputManifest
P1.2       PreprocessManifest + masks
P1.3       ViewGraph (SALAD → LightGlue → E/F/H → RANSAC)
============================
STOP AND MEASURE
============================
How many images connected?
How many components?
What inlier ratios?
Which targets have continuity?
Where are the real holes?
```

Then only:

```text
COLMAP incremental
COLMAP global
GLUEMAP
→ real comparison
```

This matches the final chain: after dense validation, if evidence remains `INSUFFICIENT_EVIDENCE` / `FAIL`, `ReconstructionGapAnalysis → Active Collection → rebuild ViewGraph`, looping until a validated environment is obtained.

**DONE for the plan** = `ValidatedCameraPath + validated dense scene + FidelityGate PASS`.
