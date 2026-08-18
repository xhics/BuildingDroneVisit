# Plan: Lot 2 Foundations — Separation of Concerns and Reconstruction Input

## Goal

Prepare BuildingDroneVisit for Lot 2 reconstruction (SfM → dense → aligned mesh) **without rewriting Lot 1B**. The current proxy package, Router, manifest digests, and coverage reports remain the source of truth for evidence. Lot 2 consumes them.

## Status

P0–P6 are **implemented and tested** (1549 tests pass). The remaining work follows the P1–P6 sequence from the architectural target.

---

## P0 — Completed

| # | What | Status |
|---|------|--------|
| 1 | `appearance_coverage` / `geometric_support` separated in `FacadeCoverage`, `SyntheticCompletion`, `lot1b_coverage`, `zone_confidence.geojson` | ✅ |
| 2 | `ReconstructionInputManifest` schema + `prepare_input()` + `reconstruction prepare-input` CLI | ✅ |
| 3 | Immutable snapshot Lot 1B → Lot 2 via digest chain | ✅ |
| 4 | `ReconstructionSelectionManifest` with per-asset decisions and `temporal_cohorts` | ✅ |

**Key files:**
- `src/hotel_pipeline/geo/facade_coverage.py`
- `src/hotel_pipeline/geo/satellite_completion.py`
- `src/hotel_pipeline/lot1b_coverage.py`
- `src/hotel_pipeline/schemas/reconstruction.py`
- `src/hotel_pipeline/reconstruction_input.py`
- `src/hotel_pipeline/cli.py`

---

## P1 — View Graph (the real G5) — Completed

| # | What | Status |
|---|------|--------|
| 1 | `ViewGraphManifest` schema | ✅ |
| 2 | `ViewGraphBuilder` with retrieval + matching + geometric verification | ✅ |
| 3 | `ReconstructionSelectionManifest` per-asset decisions | ✅ |
| 4 | CLI: `reconstruction view-graph`, `reconstruction preprocess` | ✅ |

**Key files:**
- `src/hotel_pipeline/view_graph.py`
- `src/hotel_pipeline/reconstruction_preprocess.py`
- `src/hotel_pipeline/schemas/reconstruction.py`

---

## P2 — Reconstruction Runs — Completed

| # | What | Status |
|---|------|--------|
| 1 | `ReconstructionRun` schema with metrics | ✅ |
| 2 | `ReconstructionRunner` with COLMAP incremental + global | ✅ |
| 3 | `ReconstructionPlan` independent of Router | ✅ |
| 4 | CLI: `reconstruction plan`, `reconstruction run` | ✅ |

**Key files:**
- `src/hotel_pipeline/reconstruction_run.py`
- `src/hotel_pipeline/reconstruction_plan.py`
- `src/hotel_pipeline/schemas/reconstruction.py`

---

## P3 — Consensus & Cross-Check — Completed

| # | What | Status |
|---|------|--------|
| 1 | `ReconstructionConsensusReport` schema | ✅ |
| 2 | Real Sim(3) Umeyama alignment between runs | ✅ |
| 3 | `CameraConsensusEntry` with pose-level spreads | ✅ |
| 4 | CLI: `reconstruction consensus` | ✅ |

**Key files:**
- `src/hotel_pipeline/reconstruction_consensus.py`
- `src/hotel_pipeline/schemas/reconstruction.py`

---

## P4 — Geo Alignment — Completed

| # | What | Status |
|---|------|--------|
| 1 | `GeoAlignmentManifest` schema | ✅ |
| 2 | `GeoAligner` with footprint + LiDAR roof alignment | ✅ |
| 3 | `LiDARSupportReport` placeholder | ✅ |
| 4 | CLI: `reconstruction align`, `reconstruction lidar-report` | ✅ |

**Key files:**
- `src/hotel_pipeline/geo_alignment.py`
- `src/hotel_pipeline/lidar_support.py`

---

## P5 — Surface Confidence & Validation — Completed

| # | What | Status |
|---|------|--------|
| 1 | `surface_confidence.geojson` from COLMAP outputs | ✅ |
| 2 | Held-out validation | ✅ |
| 3 | Stability validation (90%/80% subsets) | ✅ |
| 4 | Cross-solver validation | ✅ |
| 5 | CLI: `reconstruction validate` | ✅ |

**Key files:**
- `src/hotel_pipeline/surface_confidence.py`
- `src/hotel_pipeline/reconstruction_validation.py`

---

## P6 — Camera Feasibility & Final Gate — Completed

| # | What | Status |
|---|------|--------|
| 1 | `CameraFeasibilityField` with real point cloud FOV analysis | ✅ |
| 2 | `ValidatedCameraPath` sized from reconstruction spread | ✅ |
| 3 | `scene build` extension: use reconstruction if available | ✅ |
| 4 | CLI: `reconstruction camera-feasibility`, `reconstruction gate` | ✅ |

**Key files:**
- `src/hotel_pipeline/camera_feasibility.py`
- `src/hotel_pipeline/scene_package.py`

---

## Pipeline Orchestration — Completed

| # | What | Status |
|---|------|--------|
| 1 | `steps.py`: `preflight` → `reconstruct` → `align` → `validate` | ✅ |
| 2 | Prerequisite checks with `StepBlocked` | ✅ |
| 3 | `run-phase1` traverses all steps | ✅ |

**Key files:**
- `src/hotel_pipeline/steps.py`

---

## Remaining Work

| # | What | Priority |
|---|------|----------|
| 1 | GLUEMAP integration | P2 |
| 2 | MP-SfM integration | P2 |
| 3 | MapAnything / VGGT feed-forward integration | P3 |
| 4 | Brush / 3DGS dense reconstruction | P5 |
| 5 | Real LiDAR point cloud processing for LiDGS viability | P4 |

---

## Completed Since Last Update

| # | What | Status |
|---|------|--------|
| 1 | Real mask generation with OpenCV heuristics (sky, water, people, cars, signage, reflections, mobile_furniture) | ✅ |
| 2 | SIFT detector option in ViewGraphBuilder | ✅ |
| 3 | EXIF-based intrinsics estimation | ✅ |
| 4 | Temporal strategy in ReconstructionPlan (current_only vs current_plus_unknown) | ✅ |
| 5 | selected_asset_ids override in all reconstruction runners | ✅ |
| 6 | SYNTHETIC backend for testing and demos | ✅ |
| 7 | Scene package reconstruction integration (reconstructed_photo_first) | ✅ |
| 8 | CLI command `reconstruction run-all` | ✅ |

---

## P0 — Completed

| # | What | Status |
|---|------|--------|
| 1 | `appearance_coverage` / `geometric_support` separated in `FacadeCoverage`, `SyntheticCompletion`, `lot1b_coverage`, `zone_confidence.geojson` | ✅ |
| 2 | `ReconstructionInputManifest` schema + `prepare_input()` + `reconstruction prepare-input` CLI | ✅ |
| 3 | Immutable snapshot Lot 1B → Lot 2 via digest chain | ✅ |

**Key files:**
- `src/hotel_pipeline/geo/facade_coverage.py`
- `src/hotel_pipeline/geo/satellite_completion.py`
- `src/hotel_pipeline/lot1b_coverage.py`
- `src/hotel_pipeline/schemas/reconstruction.py`
- `src/hotel_pipeline/reconstruction_input.py`
- `src/hotel_pipeline/cli.py`

---

## P1 — View Graph (the real G5)

**Goal:** Replace the planned G5 SfM-full with a measured view-graph that proves continuity before any dense reconstruction.

### Deliverables

1. **`ViewGraphManifest` schema** (`src/hotel_pipeline/schemas/reconstruction.py`)
   - `view_graph_id`, `reconstruction_input_id`, `created_at`
   - `images: list[ViewGraphNode]` with asset_id, intrinsics, pose_status
   - `pairs: list[PairEvidence]` with image_a, image_b, retrieval_score, matches, inliers, inlier_ratio, relative_pose, overlap_estimate, degeneracy, status
   - `summary`: images_selected, pairs_tested, valid_pairs, largest_component, registered_candidate_ratio, median_inlier_ratio, continuity_by_demand, repetitive_risk, intrinsics_quality

2. **`ViewGraphBuilder` module** (`src/hotel_pipeline/view_graph.py`)
   - Input: `ReconstructionInputManifest`
   - Step 1: retrieval candidate pruning via `view_sector` + `viewpoint_cluster` + global descriptor
   - Step 2: matching (SIFT+LightGlue baseline, ALIKED+LightGlue candidate)
   - Step 3: geometric verification (essential matrix, RANSAC)
   - Step 4: overlap_estimate from inlier distribution
   - Output: `ViewGraphManifest`

3. **Preprocessing + masks** (`src/hotel_pipeline/reconstruction_preprocess.py`)
   - Mask types: sky, people, cars, water, large reflections, temporary signage, mobile furniture
   - Output: `05_colmap/preprocessed/<asset_id>.jpg` + `mask_set_digest`
   - Masks stored as `DerivedArtifact`, never mutate original images

4. **CLI commands**
   - `hotel-pipeline reconstruction view-graph <hotel_id>` — build view graph
   - `hotel-pipeline reconstruction preprocess <hotel_id>` — generate masks + normalized images

### Out of Scope for P1
- Actual COLMAP/GLUEMAP execution (P2)
- Cross-solver consensus (P3)
- Dense reconstruction (P5)

---

## P2 — Reconstruction Runs

**Goal:** Execute multiple SfM backends on the same frozen input and normalize outputs.

### Deliverables

1. **`ReconstructionRun` schema** (`src/hotel_pipeline/schemas/reconstruction.py`)
   - `run_id`, `backend`, `reconstruction_input_id`, `view_graph_id`
   - `status`: pending / running / completed / failed
   - `metrics`: registered_ratio, largest_component, median_reprojection_error, track_length, camera_stability
   - `output_path`: path to sparse model

2. **`ReconstructionRunner` module** (`src/hotel_pipeline/reconstruction_run.py`)
   - Backend adapters: COLMAP incremental, COLMAP global
   - Each adapter imports into a common `CameraPoseSet` + `PointGeometry` representation
   - No solver-specific logic leaks downstream

3. **`ReconstructionPlan` schema** (separate from Router)
   - `plan_id`, `reconstruction_input_id`
   - `selected_backends[]`, `fallback_chain`
   - `rationale`: why these backends for this corpus

4. **CLI commands**
   - `hotel-pipeline reconstruction plan <hotel_id>` — select backends based on ViewGraphReport
   - `hotel-pipeline reconstruction run <hotel_id> --backend colmap_incremental`

---

## P3 — Consensus & Cross-Check

**Goal:** Compare multiple reconstructions and select the best, or flag disagreement.

### Deliverables

1. **`ReconstructionConsensusReport` schema**
   - Per-backend metrics
   - Pairwise Sim(3) alignment errors
   - Camera pose spread per image
   - Cross-method agreement score

2. **`CameraConsensus` module**
   - For each asset: which backends registered it, translation/rotation/focal spread
   - Aberrant cameras flagged for exclusion or downgrade

3. **Feed-forward cross-check** (optional P3.5)
   - MapAnything / VGGT as independent depth/pose priors
   - Normalized into same `CameraPoseSet` + `PointGeometry` schema

4. **CLI commands**
   - `hotel-pipeline reconstruction consensus <hotel_id>` — compare all runs
   - `hotel-pipeline reconstruction select <hotel_id>` — pick best reconstruction

---

## P4 — Geo Alignment

**Goal:** Align the sparse reconstruction to the geospatial truth already in the pipeline.

### Deliverables

1. **`GeoAlignmentManifest` schema**
   - `source_reconstruction_id`
   - `scale`, `rotation`, `translation` (Sim(3))
   - `horizontal_crs`, `vertical_reference`
   - `footprint_error`, `roof_height_error`, `alignment_rmse`
   - `anchors[]`: which geospatial features drove alignment

2. **`GeoAligner` module** (`src/hotel_pipeline/geo_alignment.py`)
   - Primary anchor: `BUILDING_MAIN` footprint (XY)
   - Secondary anchor: LiDAR roof cloud (Z + macro geometry)
   - DTM/DSM for terrain consistency check
   - Output: aligned sparse cloud + `GeoAlignmentManifest`

3. **`LiDARSupportReport`**
   - roof point density, ground point density, facade/vertical point density
   - Determines whether LiDGS/GS-SDF are viable (P5 gate)

4. **CLI commands**
   - `hotel-pipeline reconstruction align <hotel_id>`
   - `hotel-pipeline reconstruction lidar-report <hotel_id>`

---

## P5 — Dense Representation

**Goal:** Produce a renderable dense model from the aligned sparse reconstruction.

### Deliverables

1. **Brush / 3DGS integration** (`src/hotel_pipeline/dense_reconstruction.py`)
   - Input: selected cameras + images + masks from P2/P3
   - Backend: Brush (preferred) or gsplat (R&D)
   - Output: dense point cloud / Gaussian splat / mesh

2. **Held-out validation**
   - Reserve 20% of images
   - Render from held-out poses, compare with LPIPS/SSIM/reprojection

3. **Stability test**
   - Reconstruct with 100% / 90% / 80% of images
   - Geometry must remain stable after Sim(3)

4. **`surface_confidence.geojson`**
   - Per-facade and per-zone confidence decomposed into:
     - `independent_observations`
     - `angular_diversity`
     - `track_support`
     - `reprojection_error`
     - `depth_agreement`
     - `camera_pose_confidence`
     - `cross_method_agreement`
     - `geo_prior_agreement`
     - `extrapolation_penalty`

5. **CLI commands**
   - `hotel-pipeline reconstruction dense <hotel_id>`
   - `hotel-pipeline reconstruction validate <hotel_id>`

---

## P6 — Camera Feasibility & Final Gate

**Goal:** Validate that the reconstruction supports the intended camera work, then declare `ENVIRONMENT_3D_READY`.

### Deliverables

1. **`CameraFeasibilityField` schema**
   - Input: camera pose (x, y, z, yaw, pitch, fov)
   - Output: visible_surface_confidence, unknown_fraction, proxy_fraction, reconstructed_fraction, minimum_distance_violation, collision, framing_quality, overall_score

2. **`ValidatedCameraPath`**
   - Upgrades `camera_probe_path.json` (current demand-driven orbit) to a path validated against post-reconstruction confidence
   - Not a promotional path — still diagnostic

3. **Final Gate: `ENVIRONMENT_3D_READY`**
   - Consumes: `surface_confidence.geojson`, `GeoAlignmentManifest`, `ValidatedCameraPath`, held-out validation results
   - Still does NOT generate video (video is after Lot 2)

4. **`scene build` extension**
   - If `07_reconstruction/selected` exists, use it
   - Else preserve current `hybrid_proxy_package` behavior
   - New `scene_package_type` field: `hybrid_proxy` | `reconstructed_photo_first` | `reconstructed_hybrid`

5. **CLI commands**
   - `hotel-pipeline reconstruction camera-feasibility <hotel_id>`
   - `hotel-pipeline reconstruction gate <hotel_id>` — final ENVIRONMENT_3D_READY check

---

## Data Flow Summary

```text
Lot 1B (unchanged)
    │
    ▼
ReconstructionInputManifest ........................ P0 ✅
    │
    ▼
Preprocess + Masks ................................. P1
    │
    ▼
ViewGraphManifest .................................. P1
    │
    ▼
ReconstructionPlan ................................. P2
    │
    ▼
ReconstructionRun[] (COLMAP inc/global, GLUEMAP...) P2
    │
    ▼
ReconstructionConsensus ............................ P3
    │
    ▼
GeoAlignmentManifest ............................... P4
    │
    ▼
Dense Reconstruction (Brush/gsplat) ................ P5
    │
    ▼
Surface Confidence + Validation .................... P5
    │
    ▼
CameraFeasibilityField + ValidatedCameraPath ....... P6
    │
    ▼
ENVIRONMENT_3D_READY ............................... P6
    │
    ▼
STOP — video after Lot 2
```

---

## Key Invariants

1. **Lot 1B is never rewritten.** Its manifests, Router, digests, and proxy package remain the source of truth for evidence.
2. **Two routers, not one.** The existing Router decides *what to build from* (photo-first, geo-first, hybrid). The new `ReconstructionPlan` decides *which algorithm to run* given the measured view graph.
3. **Appearance ≠ Geometry.** `appearance_union_fraction` feeds SfM viability and texture viability. `geometric_support_fraction` feeds proxy and alignment confidence. They are never merged.
4. **No silent package overwrite.** `scene_package_type` distinguishes proxy from reconstructed. A reconstructed package is a new revision, not an upgrade of the proxy.
5. **Rights are provenance, not gates.** Rights are tracked but do not exclude images from reconstruction selection.
6. **Consensus before commitment.** Multiple backends run on the same input; the best is selected by metrics, not by convenience.

---

## Out of Scope (for this plan)

- Actual GLUEMAP, MP-SfM, MapAnything, VGGT integration (P2–P3)
- Brush / 3DGS / dense reconstruction execution (P5)
- GeoAlignmentManifest implementation (P4)
- CameraFeasibilityField implementation (P6)
- ValidatedCameraPath implementation (P6)
- Surface confidence post-reconstruction (P5)

These remain future plans. This plan defines the data contracts, module boundaries, and execution order that make them safe to add later.

---

## Validation

- All existing Lot 1B tests pass (152 passed)
- P0 tests verify:
  - `appearance_union_fraction` excludes synthetic satellite completion
  - `geometric_support_fraction` includes it
  - `ReconstructionInputManifest` round-trips through JSON schema validation
  - `reconstruction prepare-input` produces a deterministic digest from identical inputs
- New P1–P6 tests verify each schema round-trips, each CLI command produces expected output, and the digest chain is unbroken from `ReconstructionInputManifest` through to final gate.

---

## Open Questions

**Q1: Mask format for SfM preprocessing**
- Options: per-image PNG masks, COLMAP-compatible masks, or generic binary masks
- **Recommendation:** Generic binary masks in `05_colmap/preprocessed/masks/`, adapter converts to COLMAP format at execution time. Keeps mask logic independent of solver.

**Q2: View graph storage format**
- Options: COLMAP native, custom JSON, or graph database
- **Recommendation:** Custom JSON (`view_graph_manifest.json`) with COLMAP export adapter. The view graph is the primary artifact; solver-specific formats are derived.

**Q3: Backend selection policy**
- Should `ReconstructionPlan` be deterministic (always run COLMAP inc → COLMAP global → GLUEMAP) or adaptive (skip backends based on ViewGraphReport metrics)?
- **Recommendation:** Adaptive with mandatory baselines. COLMAP incremental always runs as baseline. COLMAP global runs if view graph is dense enough. GLUEMAP/MP-SfM run only if metrics indicate need.
