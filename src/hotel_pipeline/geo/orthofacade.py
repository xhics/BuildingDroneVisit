"""Rectifier les images dans le plan d'un mur, et voir ce qu'on a vraiment.

Le pipeline mesure la couverture en cellules et en fractions. Il ne montre pas
que deux images du même mur **se superposent** — or c'est la seule preuve que
les poses sont mutuellement compatibles. Une carte de couverture peut être
excellente sur des poses fausses ; une orthofaçade, non : les structures y
apparaissent doubles.

**Ce que l'orthofaçade est.** Une mosaïque probatoire, où chaque texel porte ce
qui l'atteste : combien d'images le voient, laquelle domine, avec quelle
incidence, et si elles s'accordent. C'est le `SupportType` du dépôt appliqué à
la texture.

**Ce qu'elle n'est pas.** Ni une reconstruction, ni une texture de production.
Le plan de façade vient de l'extrusion d'une emprise — c'est un proxy mesuré au
sol, non un mur relevé. Un décrochement réel s'y projettera de travers, et le
désaccord entre images le dira plutôt que de le lisser.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..logging import get_logger
from .facade_visibility import effective_gsd_m
from .photometric import GainBias, fit_view_normalizations

log = get_logger("geo-orthofacade")

TEXEL_M = 0.05
TEXEL_M_FACADE = 0.12
MIN_PIXELS_PER_M = 2.0
MAX_INCIDENCE_DEG = 65.0
DISAGREEMENT_LEVEL = 42.0
MAX_INLIER_SPREAD_DE = 12.0
MAD_OUTLIER_K = 2.5
MIN_INLIERS_FOR_CONSENSUS = 2
PROXY_DEPTH_TOLERANCE_M = 0.25
LIDAR_OCCLUSION_MARGIN_M = 1.5


@dataclass
class FacadePlane:
    facade_id: str
    origin: np.ndarray
    along: np.ndarray
    normal: np.ndarray
    length_m: float
    height_m: float
    top_z_start_m: float = 0.0
    top_z_end_m: float = 0.0
    profile_uz_m: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        if self.top_z_start_m <= 0:
            self.top_z_start_m = float(self.height_m)
        if self.top_z_end_m <= 0:
            self.top_z_end_m = float(self.height_m)
        if self.profile_uz_m:
            profile = sorted((float(u), float(z)) for u, z in self.profile_uz_m)
            if profile[0][0] > 0.0:
                profile.insert(0, (0.0, profile[0][1]))
            if profile[-1][0] < self.length_m:
                profile.append((self.length_m, profile[-1][1]))
            self.profile_uz_m = tuple(profile)
            self.height_m = max(z for _, z in profile)

    def top_z(self, u: float) -> float:
        if self.profile_uz_m:
            u = max(0.0, min(float(u), self.length_m))
            for (u0, z0), (u1, z1) in zip(self.profile_uz_m, self.profile_uz_m[1:]):
                if u <= u1:
                    t = (u - u0) / max(u1 - u0, 1e-9)
                    return z0 + t * (z1 - z0)
            return self.profile_uz_m[-1][1]
        """Hauteur du mur à l'abscisse `u`, interpolation linéaire."""
        t = max(0.0, min(1.0, u / max(self.length_m, 1e-6)))
        return self.top_z_start_m + t * (self.top_z_end_m - self.top_z_start_m)

    def point_at_z(self, u: float, z_m: float) -> np.ndarray:
        """Point 3D in metric atlas coordinates (u_m, z_m)."""
        return self.origin + self.along * float(u) + np.array([0.0, 0.0, float(z_m)])

    def contains_uz(self, u: float, z_m: float) -> bool:
        return 0.0 <= u <= self.length_m and 0.0 <= z_m <= self.top_z(u) + 1e-9

    def point(self, u: float, v_norm: float) -> np.ndarray:
        z = float(v_norm) * self.top_z(u)
        return self.origin + self.along * u + np.array([0.0, 0.0, z])

    def point_legacy(self, u: float, v: float) -> np.ndarray:
        v_norm = float(v) / max(self.height_m, 1e-6)
        return self.point(u, v_norm)

    @property
    def normal_deg(self) -> float:
        return math.degrees(math.atan2(self.normal[1], self.normal[0])) % 360.0


@dataclass
class TexelCandidate:
    asset_id: str
    colour_rgb: np.ndarray
    incidence_deg: float = 0.0
    gsd_m: float | None = None
    weight: float = 1.0
    sharpness: float | None = None
    pose_confidence: float | None = None
    view_index: int = -1

    def normalised_colour(self) -> np.ndarray:
        return np.asarray(self.colour_rgb, dtype=np.float64)[:3]


@dataclass
class TexelSupport:
    contributing: int = 0
    best_asset: str | None = None
    best_incidence_deg: float | None = None
    best_distance_m: float | None = None
    disagreement: float = 0.0
    rejection_reason: str | None = None
    rejection_cause: str | None = None
    inlier_count: int = 0
    inlier_spread_de: float = 0.0
    consensus_colour: tuple[float, float, float] | None = None
    candidates: list[TexelCandidate] = field(default_factory=list)
    rejected_candidates: list[tuple[str, str]] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.rejection_reason:
            return self.rejection_reason
        if self.contributing == 0:
            return "non_observe"
        if self.contributing == 1:
            return "vue_unique"
        return "accorde"

    @property
    def is_observed(self) -> bool:
        if self.rejection_reason:
            return False
        if self.contributing == 0:
            return False
        return self.disagreement < DISAGREEMENT_LEVEL


@dataclass
class Orthofacade:
    facade_id: str
    width_px: int
    height_px: int
    image: np.ndarray | None = None
    support: list[TexelSupport] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    status_map: np.ndarray | None = None

    def by_status(self) -> dict[str, int]:
        counts = {}
        for texel in self.support:
            counts[texel.status] = counts.get(texel.status, 0) + 1
        return counts

    @property
    def observed_fraction(self) -> float:
        if not self.support:
            return 0.0
        return sum(1 for t in self.support if t.is_observed) / len(self.support)

    def as_dict(self) -> dict:
        legacy_counts = self.by_status()
        aliases = {
            "non_observe": "UNOBSERVED", "vue_unique": "OBSERVED_SINGLE",
            "accorde": "OBSERVED_CONSENSUS", "desaccord": "REJECTED_DISAGREEMENT",
        }
        counts: dict[str, int] = {}
        for name, count in legacy_counts.items():
            canonical = aliases.get(name, name)
            counts[canonical] = counts.get(canonical, 0) + count
        return {
            "facade_id": self.facade_id,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "texel_m": TEXEL_M_FACADE,
            "observed_fraction": round(self.observed_fraction, 3),
            "by_status": counts,
            "disagreement_fraction": round(
                counts.get("REJECTED_DISAGREEMENT", 0) / max(len(self.support), 1), 3
            ),
            "provenance": self.provenance,
            "caveats": [
                "le plan vient de l'extrusion d'une emprise, non d'un mur releve : un decrochement reel s'y projette de travers",
                "un desaccord entre images signale une pose fausse, un objet mobile ou un decrochement — il ne dit pas lequel",
                "une mosaïque probatoire n'est pas une texture de production",
            ],
        }


def plane_from_edge(
    start: np.ndarray,
    end: np.ndarray,
    height_m: float,
    facade_id: str = "FACADE",
    top_z_start_m: float | None = None,
    top_z_end_m: float | None = None,
    profile_uz_m: Sequence[tuple[float, float]] | None = None,
) -> FacadePlane:
    """Construit le plan d'un mur depuis une arête d'emprise et une hauteur."""
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    delta = end[:2] - start[:2]
    length = float(np.hypot(*delta))
    along = np.array([delta[0] / length, delta[1] / length, 0.0])
    normal = np.array([along[1], -along[0], 0.0])
    h = float(height_m)
    return FacadePlane(
        facade_id=facade_id,
        origin=np.array([start[0], start[1], float(start[2]) if len(start) > 2 else 0.0]),
        along=along,
        normal=normal,
        length_m=length,
        height_m=h,
        top_z_start_m=float(top_z_start_m) if top_z_start_m is not None else h,
        top_z_end_m=float(top_z_end_m) if top_z_end_m is not None else h,
        profile_uz_m=tuple(profile_uz_m or ()),
    )


def _lab(colour: tuple[float, float, float]) -> np.ndarray:
    r, g, b = colour
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    r = r if r > 0.04045 else r / 12.92
    g = g if g > 0.04045 else g / 12.92
    b = b if b > 0.04045 else b / 12.92
    x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047
    y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 1.00000
    z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883
    x = x ** (1/3) if x > 0.008856 else 7.787 * x + 16/116
    y = y ** (1/3) if y > 0.008856 else 7.787 * y + 16/116
    z = z ** (1/3) if z > 0.008856 else 7.787 * z + 16/116
    return np.array([116 * y - 16, 500 * (x - y), 200 * (y - z)])


def _ransac_seed_inliers(labs: np.ndarray) -> np.ndarray:
    count = labs.shape[0]
    best_pair = (0, 1)
    best_distance = math.inf
    for i in range(count):
        for j in range(i + 1, count):
            distance = float(np.mean(np.abs(labs[i] - labs[j])))
            if distance < best_distance:
                best_distance = distance
                best_pair = (i, j)
    centre = labs[list(best_pair)].mean(axis=0)
    tolerance = max(MAD_OUTLIER_K * best_distance, 3.0)
    keep = [index for index in range(count) if float(np.mean(np.abs(labs[index] - centre))) <= tolerance]
    return np.asarray(sorted(set(keep + list(best_pair))), dtype=int)


@dataclass(frozen=True)
class FusionVerdict:
    colour: tuple[float, float, float] | None
    accepted: bool
    status: str
    inlier_count: int = 0
    inlier_spread_de: float = 0.0
    reason: str | None = None


def fuse_texel_candidates(colours, weights=None):
    colours = list(colours)
    count = len(colours)
    if count == 0:
        return FusionVerdict(None, False, "REJECTED_DISAGREEMENT", 0, 0.0, "no_candidate")
    if weights is not None and len(weights) != count:
        raise ValueError("autant de poids que de couleurs sont requis")
    if count == 1:
        colour = tuple(float(c) for c in np.asarray(colours[0], dtype=np.float64)[:3])
        return FusionVerdict(colour, True, "OBSERVED_SINGLE", 1, 0.0, None)

    labs = np.stack([_lab(tuple(np.asarray(c, dtype=np.float64)[:3])) for c in colours])
    med = np.median(labs, axis=0)
    mad = np.median(np.abs(labs - med), axis=0) * 1.4826
    inlier_indices = np.where(np.all(np.abs(labs - med) < MAD_OUTLIER_K * (mad + 1e-8), axis=1))[0]
    if len(inlier_indices) < MIN_INLIERS_FOR_CONSENSUS:
        inlier_indices = _ransac_seed_inliers(labs)

    if len(inlier_indices) < MIN_INLIERS_FOR_CONSENSUS:
        return FusionVerdict(None, False, "REJECTED_DISAGREEMENT", len(inlier_indices), float("inf"), "insufficient_inliers")

    inlier_labs = labs[inlier_indices]
    spread = float(np.mean(np.std(inlier_labs, axis=0)))
    if spread > MAX_INLIER_SPREAD_DE:
        return FusionVerdict(None, False, "REJECTED_DISAGREEMENT", len(inlier_indices), spread, "inlier_spread")

    if weights is None:
        consensus = np.mean([colours[i] for i in inlier_indices], axis=0)
    else:
        weight_values = np.asarray([max(float(weights[i]), 1e-9) for i in inlier_indices], dtype=np.float64)
        weight_values /= weight_values.sum()
        stacked = np.asarray([colours[i] for i in inlier_indices], dtype=np.float64)
        consensus = (stacked * weight_values[:, None]).sum(axis=0)

    consensus_tuple = tuple(float(c) for c in np.clip(consensus[:3], 0.0, 255.0))
    return FusionVerdict(consensus_tuple, True, "OBSERVED_CONSENSUS", len(inlier_indices), spread, None)


GSD_REFERENCE_M = TEXEL_M_FACADE / MIN_PIXELS_PER_M


def candidate_weight(*, incidence_deg, gsd_m=None, sharpness=None, pose_confidence=None):
    cosine = math.cos(math.radians(min(max(incidence_deg, 0.0), 89.0)))
    weight = cosine * cosine
    if gsd_m is not None and gsd_m > 0.0:
        weight *= min(GSD_REFERENCE_M / gsd_m, 1.0) ** 2
    if sharpness is not None:
        weight *= min(max(sharpness, 0.0), 1.0)
    if pose_confidence is not None:
        weight *= min(max(pose_confidence, 0.0), 1.0)
    return max(weight, 1e-6)


def _apply_normalizations(samples, normalizations):
    applied = []
    for view_index, model in sorted(normalizations.items()):
        if model.is_identity():
            continue
        for slot_candidates in samples.values():
            for candidate in slot_candidates:
                if candidate.view_index == view_index:
                    candidate.colour_rgb = model.apply(candidate.colour_rgb)
        applied.append(view_index)
    return applied


def rectify(plane, views, texel_m=TEXEL_M_FACADE, policy=None):
    from .facade_visibility import RegisteredView

    cols = max(int(round(plane.length_m / texel_m)), 1)
    rows = max(int(round(plane.height_m / texel_m)), 1)
    found = Orthofacade(facade_id=plane.facade_id, width_px=cols, height_px=rows)
    found.support = [TexelSupport() for _ in range(rows * cols)]

    if not views:
        found.provenance = {"views_supplied": 0, "reason": "aucune vue fournie"}
        return found

    canvas = np.zeros((rows, cols, 3), dtype=np.float64)
    samples = {}
    candidates_by_view = {}
    rejected_by_slot = {}
    used = 0
    skipped_resolution = 0
    skipped_no_mask = 0
    rejection_counts = {}

    v_norms = np.linspace(0.5 / rows, 1.0 - 0.5 / rows, rows)
    us = (np.arange(cols) + 0.5) * texel_m

    for view_index, view in enumerate(views):
        if isinstance(view, RegisteredView):
            asset_id = view.asset_id
            image = view.image
            camera = view.camera
            visibility_mask = view.semantic_mask
            view_proxy = view.proxy_depth
            view_lidar = view.lidar_depth
            registered = view
        else:
            asset_id, image, camera = view[:3]
            visibility_mask = view[3] if len(view) >= 4 else None
            view_proxy = view[4] if len(view) >= 5 else None
            view_lidar = view[5] if len(view) >= 6 else None
            registered = None

        if visibility_mask is None or not visibility_mask.any():
            skipped_no_mask += 1
            continue

        def _camera_centre(cam):
            found_pos = getattr(cam, "position")
            return np.asarray(found_pos() if callable(found_pos) else found_pos, dtype=np.float64)

        centre = plane.point(plane.length_m * 0.5, 0.5)
        to_camera = _camera_centre(camera) - centre
        span = float(np.linalg.norm(to_camera))
        if span < 1e-6:
            continue
        cosine = float(np.dot(plane.normal, to_camera / span))
        incidence = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        if incidence > MAX_INCIDENCE_DEG:
            continue

        focal = getattr(camera, "f", None)
        if focal:
            pixels_per_m = float(focal) / max(span, 1e-6)
            if pixels_per_m < MIN_PIXELS_PER_M:
                skipped_resolution += 1
                continue

        used += 1
        view_samples = {}
        for row, v_norm in enumerate(v_norms):
            points = np.array([plane.point(u, v_norm) for u in us])
            screen, depth = camera.project(points)
            if screen is None:
                continue
            to_cam = _camera_centre(camera) - points
            span_v = np.linalg.norm(to_cam, axis=1)
            span_safe = np.maximum(span_v, 1e-6)
            cosine_v = (to_cam @ plane.normal) / span_safe
            incidence_v = np.degrees(np.arccos(np.clip(cosine_v, -1.0, 1.0)))

            for col in range(cols):
                x, y = screen[col]
                ix, iy = int(round(x)), int(round(y))
                if not (0 <= ix < image.shape[1] and 0 <= iy < image.shape[0]):
                    continue
                if depth is not None and depth[col] <= 0.5:
                    continue
                if incidence_v[col] > MAX_INCIDENCE_DEG:
                    continue

                slot = row * cols + col
                wall_depth = float(depth[col])

                gsd_m = effective_gsd_m(camera, points[col], plane.along)
                if gsd_m is not None and gsd_m > 1.0 / MIN_PIXELS_PER_M:
                    rejected_by_slot.setdefault(slot, []).append((asset_id, "REJECTED_RESOLUTION"))
                    continue

                if not visibility_mask[iy, ix]:
                    rejected_by_slot.setdefault(slot, []).append((asset_id, "REJECTED_SEMANTIC"))
                    continue

                if registered is not None:
                    if registered.occludes((ix, iy), wall_depth):
                        rejected_by_slot.setdefault(slot, []).append((asset_id, "REJECTED_OCCLUDED"))
                        continue
                else:
                    if view_proxy is not None:
                        hit_depth, hit_fid = view_proxy.hit(ix, iy)
                        if hit_fid is not None and hit_fid >= 0 and wall_depth > hit_depth + PROXY_DEPTH_TOLERANCE_M:
                            rejected_by_slot.setdefault(slot, []).append((asset_id, "REJECTED_OCCLUDED"))
                            continue
                    if view_lidar is not None and view_lidar.valid[iy, ix]:
                        lidar_d = float(view_lidar.depth[iy, ix])
                        if wall_depth > lidar_d + LIDAR_OCCLUSION_MARGIN_M:
                            rejected_by_slot.setdefault(slot, []).append((asset_id, "REJECTED_OCCLUDED"))
                            continue

                colour = image[iy, ix].astype(np.float64)
                inc = float(incidence_v[col])
                candidate = TexelCandidate(
                    asset_id=asset_id,
                    colour_rgb=colour,
                    incidence_deg=inc,
                    gsd_m=gsd_m,
                    weight=candidate_weight(
                        incidence_deg=inc,
                        gsd_m=gsd_m,
                        sharpness=getattr(registered, "sharpness", None) if registered else None,
                        pose_confidence=getattr(registered, "pose_confidence", None) if registered else None,
                    ),
                    view_index=view_index,
                )
                samples.setdefault(slot, []).append(candidate)
                view_samples[slot] = colour
                texel = found.support[slot]
                texel.contributing += 1
                if inc < (texel.best_incidence_deg or math.inf):
                    texel.best_incidence_deg = inc
                    texel.best_distance_m = float(span_v[col])
                    texel.best_asset = asset_id
        if view_samples:
            candidates_by_view[view_index] = view_samples

    normalizations = fit_view_normalizations(candidates_by_view)
    normalized_views = _apply_normalizations(samples, normalizations)
    reference_view_index = next((index for index, model in sorted(normalizations.items()) if model.is_identity()), None)

    for slot, slot_candidates in samples.items():
        row, col = divmod(slot, cols)
        texel = found.support[slot]
        verdict = fuse_texel_candidates([candidate.normalised_colour() for candidate in slot_candidates], [candidate.weight for candidate in slot_candidates])
        texel.candidates = slot_candidates
        texel.inlier_count = verdict.inlier_count
        texel.inlier_spread_de = verdict.inlier_spread_de
        texel.disagreement = verdict.inlier_spread_de
        if verdict.accepted:
            texel.consensus_colour = verdict.colour
            canvas[row, col] = verdict.colour
        else:
            texel.rejection_reason = "REJECTED_DISAGREEMENT"
            texel.rejection_cause = verdict.reason
            rejection_counts["REJECTED_DISAGREEMENT"] = rejection_counts.get("REJECTED_DISAGREEMENT", 0) + 1

    for slot in range(rows * cols):
        texel = found.support[slot]
        if texel.contributing > 0:
            continue
        reasons = [reason for _asset, reason in rejected_by_slot.get(slot, [])]
        if not reasons:
            continue
        dominant = max(set(reasons), key=reasons.count)
        texel.rejection_reason = dominant
        texel.rejected_candidates = rejected_by_slot.get(slot, [])
        rejection_counts[dominant] = rejection_counts.get(dominant, 0) + 1

    found.image = canvas.astype(np.uint8) if used else None
    found.provenance = {
        "views_supplied": len(views),
        "views_used": used,
        "views_too_far": skipped_resolution,
        "views_without_building_mask": skipped_no_mask,
        "texel_m": texel_m,
        "min_pixels_per_m": MIN_PIXELS_PER_M,
        "max_incidence_deg": MAX_INCIDENCE_DEG,
        "disagreement_level": DISAGREEMENT_LEVEL,
        "rejection_counts": rejection_counts,
        "photometric_normalization": {
            "algorithm": "gain_bias_median_overlap_v1",
            "reference_view": reference_view_index,
            "normalized_views": sorted(normalized_views),
            "models": {str(view_index): model.as_dict() for view_index, model in sorted(normalizations.items())},
        },
        "fusion": {
            "pipeline": "candidates_filter_normalize_fuse",
            "rejection_scope": "per_candidate",
            "robust_estimator": f"lab_median_mad_k{MAD_OUTLIER_K}_ransac_min_inliers_{MIN_INLIERS_FOR_CONSENSUS}",
            "max_inlier_spread_de": MAX_INLIER_SPREAD_DE,
            "weights": ["incidence", "effective_gsd", "sharpness", "pose_confidence"],
            "consensus_written_to_atlas": True,
        },
    }
    log.info("%s : %d vue(s) rectifiee(s), %.0f%% du mur observe", plane.facade_id, used, 100 * found.observed_fraction)
    return found


def _build_triangles_from_payload(payload: dict) -> tuple[list[np.ndarray], list[int]]:
    triangles = []
    face_ids = []
    fid = 0
    for volume in payload.get("volumes", []):
        solid = volume.get("solid") or {}
        sv = solid.get("vertices") or []
        sf = solid.get("faces") or []
        if sv and sf:
            for face in sf:
                if len(face) >= 3:
                    tri = np.asarray([sv[idx] for idx in face[:3]], dtype=np.float64)
                    if tri.shape == (3, 3):
                        triangles.append(tri)
                        face_ids.append(fid)
                        fid += 1
            continue
        fp = volume.get("fp") or []
        wh = volume.get("wh") or []
        h_default = float(volume.get("h") or 8.0)
        if len(fp) >= 3:
            for i in range(len(fp)):
                j = (i + 1) % len(fp)
                a = np.array([fp[i][0], fp[i][1], 0.0], dtype=np.float64)
                b = np.array([fp[j][0], fp[j][1], 0.0], dtype=np.float64)
                h_i = float(wh[i]) if i < len(wh) else h_default
                h_j = float(wh[j]) if j < len(wh) else h_default
                c = np.array([fp[j][0], fp[j][1], h_j], dtype=np.float64)
                d = np.array([fp[i][0], fp[i][1], h_i], dtype=np.float64)
                triangles.extend([[a, b, c], [a, c, d]])
                face_ids.extend([fid, fid + 1])
                fid += 2
        rv = volume.get("rv") or []
        rf = volume.get("rf") or []
        if rv and rf:
            for face in rf:
                if len(face) >= 3:
                    tri = np.asarray([rv[idx] for idx in face[:3]], dtype=np.float64)
                    if tri.shape == (3, 3):
                        triangles.append(tri)
                        face_ids.append(fid)
                        fid += 1
    return triangles, face_ids


def _rectify_profile_legacy(
    plane: FacadePlane,
    views: Sequence,
    texel_m: float = TEXEL_M_FACADE,
    policy: dict | None = None,
) -> Orthofacade:
    """Projette chaque vue dans le plan du mur et fusionne les contributions.

    Chaque vue est un tuple (asset_id, image, camera, vis, proxy, laz_occ, ...).
    ``proxy`` et ``laz_occ`` sont par vue : la profondeur est testée avec les
    données de la vue courante, pas une globale.
    """
    cols = max(int(round(plane.length_m / texel_m)), 1)
    rows = max(int(math.ceil(plane.height_m / texel_m)), 1)
    found = Orthofacade(facade_id=plane.facade_id, width_px=cols, height_px=rows)
    found.support = [TexelSupport() for _ in range(rows * cols)]

    if not views:
        found.provenance = {"views_supplied": 0, "reason": "aucune vue fournie"}
        return found

    canvas = np.zeros((rows, cols, 3), dtype=np.float64)
    best_incidence = np.full((rows, cols), np.inf)
    samples: dict[int, list[tuple[np.ndarray, str, float]]] = {}
    rejection_log: dict[int, list[str]] = {}
    used = 0
    skipped_resolution = 0
    rejection_counts: dict[str, int] = {}

    zs = (np.arange(rows) + 0.5) * texel_m
    us = (np.arange(cols) + 0.5) * texel_m

    for view in views:
        asset_id, image, camera = view[:3]
        visibility_mask = view[3] if len(view) >= 4 else None
        view_proxy = view[4] if len(view) >= 5 else None
        view_lidar = view[5] if len(view) >= 6 else None

        centre = plane.point(plane.length_m * 0.5, 0.5)
        to_camera = np.asarray(camera.position, dtype=np.float64) - centre
        span = float(np.linalg.norm(to_camera))
        if span < 1e-6:
            continue
        cosine = float(np.dot(plane.normal, to_camera / span))
        incidence = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        if incidence > MAX_INCIDENCE_DEG:
            continue

        focal = getattr(camera, "f", None)
        if focal:
            pixels_per_m = float(focal) / max(span, 1e-6)
            if pixels_per_m < MIN_PIXELS_PER_M:
                skipped_resolution += 1
                continue

        used += 1
        for row, z_m in enumerate(zs):
            inside = np.asarray([plane.contains_uz(u, z_m) for u in us])
            if not inside.any():
                continue
            points = np.array([plane.point_at_z(u, z_m) for u in us])
            screen, depth = camera.project(points)
            if screen is None:
                continue
            to_cam = np.asarray(camera.position, dtype=np.float64) - points
            span_v = np.linalg.norm(to_cam, axis=1)
            span_safe = np.maximum(span_v, 1e-6)
            cosine_v = (to_cam @ plane.normal) / span_safe
            incidence_v = np.degrees(np.arccos(np.clip(cosine_v, -1.0, 1.0)))

            for col in range(cols):
                if not inside[col]:
                    continue
                x, y = screen[col]
                ix, iy = int(round(x)), int(round(y))
                if not (0 <= ix < image.shape[1] and 0 <= iy < image.shape[0]):
                    continue
                if depth is not None and depth[col] <= 0.5:
                    continue
                if incidence_v[col] > MAX_INCIDENCE_DEG:
                    continue

                slot = row * cols + col
                wall_depth = float(depth[col])

                if visibility_mask is not None and not visibility_mask[iy, ix]:
                    rejection_log.setdefault(slot, []).append("REJECTED_SEMANTIC")
                    continue

                if view_proxy is not None:
                    hit_depth, hit_fid = view_proxy.hit(ix, iy)
                    if hit_fid is not None and hit_fid >= 0 and wall_depth > hit_depth + PROXY_DEPTH_TOLERANCE_M:
                        rejection_log.setdefault(slot, []).append("REJECTED_OCCLUDED")
                        continue

                if view_lidar is not None and view_lidar.valid[iy, ix]:
                    lidar_d = float(view_lidar.depth[iy, ix])
                    if wall_depth > lidar_d + LIDAR_OCCLUSION_MARGIN_M:
                        rejection_log.setdefault(slot, []).append("REJECTED_OCCLUDED")
                        continue

                colour = image[iy, ix].astype(np.float64)
                samples.setdefault(slot, []).append((colour, asset_id, float(incidence_v[col])))
                texel = found.support[slot]
                texel.contributing += 1
                inc = float(incidence_v[col])
                if inc < (texel.best_incidence_deg or math.inf):
                    texel.best_incidence_deg = inc
                    texel.best_distance_m = float(span_v[col])
                    texel.best_asset = asset_id

    for slot, data in samples.items():
        texel = found.support[slot]
        colours = [c for c, _, _ in data]

        if len(colours) < 2:
            row, col = divmod(slot, cols)
            canvas[row, col] = colours[0]
            continue

        labs = np.stack([_lab(tuple(c[:3])) for c in colours])
        med = np.median(labs, axis=0)
        mad = np.median(np.abs(labs - med), axis=0) * 1.4826
        inliers = []
        for i, lab in enumerate(labs):
            if np.all(np.abs(lab - med) < MAD_OUTLIER_K * (mad + 1e-8)):
                inliers.append(colours[i])

        if len(inliers) >= MIN_INLIERS_FOR_CONSENSUS:
            texel.inlier_count = len(inliers)
            inlier_labs = np.stack([_lab(tuple(c[:3])) for c in inliers])
            spread = float(np.mean(np.std(inlier_labs, axis=0)))
            texel.inlier_spread_de = spread
            if spread <= MAX_INLIER_SPREAD_DE:
                texel.consensus_colour = tuple(np.mean(inliers, axis=0)[:3])
                row, col = divmod(slot, cols)
                canvas[row, col] = texel.consensus_colour
            else:
                texel.rejection_reason = "REJECTED_DISAGREEMENT"
                texel.rejection_cause = "inlier_spread"
                rejection_counts[texel.rejection_reason] = rejection_counts.get(texel.rejection_reason, 0) + 1
        else:
            texel.rejection_reason = "REJECTED_DISAGREEMENT"
            texel.rejection_cause = "insufficient_inliers"
            rejection_counts[texel.rejection_reason] = rejection_counts.get(texel.rejection_reason, 0) + 1

    for slot in range(rows * cols):
        texel = found.support[slot]
        if texel.contributing > 0:
            continue
        reasons = rejection_log.get(slot)
        if not reasons:
            continue
        dominant = max(set(reasons), key=reasons.count)
        texel.rejection_reason = dominant
        rejection_counts[dominant] = rejection_counts.get(dominant, 0) + 1

    found.image = canvas.astype(np.uint8) if used else None
    found.provenance = {
        "views_supplied": len(views),
        "views_used": used,
        "views_too_far": skipped_resolution,
        "texel_m": texel_m,
        "min_pixels_per_m": MIN_PIXELS_PER_M,
        "max_incidence_deg": MAX_INCIDENCE_DEG,
        "disagreement_level": DISAGREEMENT_LEVEL,
        "rejection_counts": rejection_counts,
    }
    log.info(
        "%s : %d vue(s) rectifiée(s), %.0f%% du mur observé",
        plane.facade_id,
        used,
        100 * found.observed_fraction,
    )
    return found


__all__ = [
    "DISAGREEMENT_LEVEL", "FusionVerdict", "GSD_REFERENCE_M", "MAX_INCIDENCE_DEG",
    "MAX_INLIER_SPREAD_DE", "MAD_OUTLIER_K", "MIN_INLIERS_FOR_CONSENSUS", "MIN_PIXELS_PER_M",
    "TEXEL_M", "TEXEL_M_FACADE", "FacadePlane", "LidarOcclusion", "Orthofacade", "ProxyDepth",
    "TexelCandidate", "TexelSupport", "candidate_weight", "fuse_texel_candidates", "plane_from_edge", "rectify",
]
