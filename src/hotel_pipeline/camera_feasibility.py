"""Faisabilité caméra et trajectoire validée (Lot 2 — P6).

Ce module évalue si la reconstruction supporte les poses de caméra
intentionnelles, puis produit une `ValidatedCameraPath`.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .reconstruction_consensus import resolve_model_dir
from .schemas.reconstruction import (
    CameraFeasibilityField,
    ValidatedCameraPath,
)
from .workspace import Workspace


def _projected_surface_mask(mesh, camera, face_indices: list[int]) -> np.ndarray:  # noqa: ANN001
    """Closed projected support of exact canonical faces, before occlusion."""
    import shapely
    from shapely.geometry import Polygon

    from .render_engine import (
        _project_camera_space,
        clip_polygon_to_image,
        clip_triangle_near,
    )

    mask = np.zeros((camera.height, camera.width), dtype=bool)
    for face_index in face_indices:
        triangle = mesh.vertices[mesh.faces[face_index]] @ camera.R.T + camera.t
        for piece in clip_triangle_near(triangle, camera.near_m):
            screen, _depth = _project_camera_space(piece, camera)
            polygon = clip_polygon_to_image(screen, camera.width, camera.height)
            if len(polygon) < 3:
                continue
            x0 = max(0, int(np.floor(polygon[:, 0].min())))
            x1 = min(camera.width - 1, int(np.ceil(polygon[:, 0].max())))
            y0 = max(0, int(np.floor(polygon[:, 1].min())))
            y1 = min(camera.height - 1, int(np.ceil(polygon[:, 1].max())))
            if x0 > x1 or y0 > y1:
                continue
            xs, ys = np.meshgrid(
                np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5,
            )
            inside = shapely.intersects_xy(Polygon(polygon), xs.ravel(), ys.ravel())
            mask[y0:y1 + 1, x0:x1 + 1] |= inside.reshape(xs.shape)
    return mask


def evaluate_canonical_camera(
    mesh, camera, *, target_building_id: str | None = None,
    texture_min_distances: dict[str, float] | None = None,
    unknown_threshold: float = 0.05, minimum_target_fraction: float = 0.03,
    minimum_clearance_m: float = 0.35,
) -> dict:
    """Exact per-surface feasibility from canonical depth and ID buffers."""
    from .reality_contract import require_canonical_mesh
    from .render_engine import rasterize_mesh
    from .schemas.canonical_states import MeasurementState

    receipt = require_canonical_mesh(mesh, "camera_feasibility")
    frame = rasterize_mesh(mesh, camera)
    foreground = frame.triangle_id >= 0
    frame_pixels = camera.width * camera.height
    position = camera.position()
    visible_surfaces = []
    target_pixels = np.zeros_like(foreground)
    unknown_pixels = np.zeros_like(foreground)
    reasons = []
    texture_min_distances = texture_min_distances or {}
    for encoded, surface_id in sorted(frame.surface_id_lookup.items()):
        face_indices = [i for i, value in enumerate(mesh.surface_ids) if value == surface_id]
        projected = _projected_surface_mask(mesh, camera, face_indices)
        visible = frame.surface_id == encoded
        projected_count, visible_count = int(projected.sum()), int(visible.sum())
        if projected_count == 0:
            continue
        hidden = projected & ~visible
        owners, counts = np.unique(frame.surface_id[hidden & foreground], return_counts=True)
        occluders = {
            frame.surface_id_lookup[int(owner)]: float(count / projected_count)
            for owner, count in zip(owners, counts)
            if int(owner) in frame.surface_id_lookup and int(owner) != encoded
        }
        distances = frame.depth_z[visible]
        rows, cols = np.where(visible)
        if visible_count:
            rays = np.asarray([
                camera.ray_from_pixel(x + 0.5, y + 0.5) for y, x in zip(rows, cols)
            ])
            view_cos = np.abs(np.sum(frame.normal[visible] * -rays, axis=1))
            incidence = np.degrees(np.arccos(np.clip(view_cos, 0.0, 1.0)))
            world_area = sum(
                np.linalg.norm(np.cross(
                    mesh.vertices[mesh.faces[i, 1]] - mesh.vertices[mesh.faces[i, 0]],
                    mesh.vertices[mesh.faces[i, 2]] - mesh.vertices[mesh.faces[i, 0]],
                )) * 0.5 for i in face_indices
            )
            required_gsd = float(np.sqrt(world_area / max(projected_count, 1)))
            mean_distance = float(np.mean(distances))
            incidence_median = float(np.median(incidence))
            incidence_p90 = float(np.percentile(incidence, 90))
        else:
            required_gsd, mean_distance = float("inf"), float("inf")
            incidence_median = incidence_p90 = 90.0
        record = mesh.surface_catalog[surface_id]
        if target_building_id is None or record.building_id == target_building_id:
            target_pixels |= visible
        unknown_faces = [
            index for index in face_indices
            if mesh.measurement_states[index] is MeasurementState.UNKNOWN
        ]
        if unknown_faces:
            unknown_pixels |= visible & np.isin(frame.triangle_id, unknown_faces)
        minimum = texture_min_distances.get(surface_id)
        texture_safe = minimum is None or mean_distance >= minimum
        if visible_count and not texture_safe:
            reasons.append(
                f"{surface_id}: texture requires {minimum:.2f}m, visible at {mean_distance:.2f}m"
            )
        visible_surfaces.append({
            "surface_id": surface_id,
            "visible_fraction": float(visible_count / projected_count),
            "pixel_fraction": float(visible_count / frame_pixels),
            "projected_pixel_area": projected_count,
            "visible_pixel_area": visible_count,
            "mean_distance_m": mean_distance,
            "depth_min_m": float(np.min(distances)) if visible_count else None,
            "depth_max_m": float(np.max(distances)) if visible_count else None,
            "incidence_median_deg": incidence_median,
            "incidence_p90_deg": incidence_p90,
            "effective_required_gsd_m": required_gsd,
            "occlusion_fraction": float(hidden.sum() / projected_count),
            "occluders": occluders,
            "texture_supported": texture_safe,
        })
    target_fraction = float(target_pixels.sum() / frame_pixels)
    unknown_fraction = float(unknown_pixels.sum() / max(target_pixels.sum(), 1))
    if unknown_fraction > unknown_threshold:
        reasons.append(f"unknown visible fraction {unknown_fraction:.3f} > {unknown_threshold:.3f}")
    if target_fraction < minimum_target_fraction:
        reasons.append(f"target occupies only {target_fraction:.3f} of frame")
    if target_pixels.any():
        y, x = np.where(target_pixels)
        centre_error = np.hypot(
            x.mean() / camera.width - 0.5, y.mean() / camera.height - 0.5,
        )
        centrality = float(np.clip(1.0 - centre_error / 0.707, 0.0, 1.0))
    else:
        centrality = 0.0
    triangles = mesh.vertices[mesh.faces]
    clearance = distance_to_mesh(position, triangles)
    if clearance < minimum_clearance_m:
        reasons.append(
            f"clearance {clearance:.3f}m < {minimum_clearance_m:.3f}m"
        )
    incidence_quality = sum(
        row["pixel_fraction"]
        * max(0.0, math.cos(math.radians(row["incidence_median_deg"])))
        for row in visible_surfaces
    ) / max(sum(row["pixel_fraction"] for row in visible_surfaces), 1e-12)
    subject_score = float(
        np.clip(target_fraction / 0.35, 0.0, 1.0)
        * centrality * incidence_quality
    )
    return {
        "visible_surfaces": visible_surfaces,
        "target_pixel_fraction": target_fraction,
        "unknown_visible_fraction": unknown_fraction,
        "minimum_clearance_m": clearance,
        "subject_score": subject_score,
        "accepted": not reasons,
        "rejection_reasons": reasons,
        "input_mesh_digest": receipt.input_mesh_digest,
        "proxy_usage": 0,
    }

# ---------------------------------------------------------------------------
# Helpers géométriques
# ---------------------------------------------------------------------------


def _load_colmap_points3d(run_dir: Path) -> np.ndarray | None:
    points_file = run_dir / "normalized" / "points3D"
    if not points_file.is_file():
        return None
    pts = []
    for line in points_file.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 4:
            pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not pts:
        return None
    return np.array(pts)


def _load_run_points(reconstruction_run_id: str, workspace: Workspace) -> np.ndarray | None:
    run_json_path = workspace.path("07_reconstruction", "runs", f"{reconstruction_run_id}.json")
    if not run_json_path.is_file():
        return None
    run_data = json.loads(run_json_path.read_text("utf-8"))
    output_path = run_data.get("output_path")
    if not output_path:
        return None
    run_dir = resolve_model_dir(output_path)

    normalized_dir = run_dir / "normalized"
    if not normalized_dir.is_dir():
        normalized_dir = run_dir.parent / "normalized"
    if not normalized_dir.is_dir():
        normalized_dir = run_dir.parent.parent / "normalized"
    if not normalized_dir.is_dir():
        return None
    return _load_colmap_points3d(normalized_dir.parent)


def _load_scene_triangles(workspace: Workspace | Path) -> np.ndarray:
    path = (
        workspace.path("11_conditioning", "conditioned_scene.json")
        if hasattr(workspace, "path")
        else Path(workspace) / "11_conditioning" / "conditioned_scene.json"
    )
    if not path.is_file():
        return np.empty((0, 3, 3))
    payload = json.loads(path.read_text("utf-8"))
    from .conditioning.canonical_mesh import CanonicalSceneMesh
    from .reality_contract import require_canonical_mesh
    triangles: list[np.ndarray] = []
    for volume in payload.get("buildings", payload.get("volumes", [])):
        mesh = volume.get("solid_mesh") or volume.get("solid") or {}
        canonical = CanonicalSceneMesh.from_dict(mesh)
        require_canonical_mesh(canonical, "collision")
        triangles.extend(triangle for triangle, _face_id in canonical.triangles())
    terrain = payload.get("terrain") or {}
    vertices = np.asarray(terrain.get("vertices") or [], dtype=float)
    for face in terrain.get("faces") or []:
        if len(face) >= 3 and vertices.size:
            for i in range(1, len(face) - 1):
                triangles.append(vertices[[face[0], face[i], face[i + 1]]])
    return np.asarray(triangles) if triangles else np.empty((0, 3, 3))


class CanonicalCollisionEngine:
    """Collision/raycast bound to one immutable canonical mesh digest."""

    def __init__(self, mesh) -> None:  # noqa: ANN001
        from .reality_contract import require_canonical_mesh

        self.mesh = mesh
        self.receipt = require_canonical_mesh(mesh, "collision")
        self.triangles = np.asarray([triangle for triangle, _ in mesh.triangles()])

    @property
    def input_mesh_digest(self) -> str:
        return self.receipt.input_mesh_digest

    def distance(self, point: np.ndarray) -> float:
        return distance_to_mesh(np.asarray(point, float), self.triangles)

    def capsule_intersects(self, start: np.ndarray, end: np.ndarray, radius_m: float) -> bool:
        return capsule_intersects_mesh(start, end, self.triangles, radius_m)

    def raycast(self, origin: np.ndarray, direction: np.ndarray, max_distance_m: float = 5e3):  # noqa: ANN001
        """Return the canonical triangle and physical surface, not an array slot."""
        return self.mesh.raycast_hit(origin, direction, max_distance_m)


def _point_triangle_distance(p: np.ndarray, tri: np.ndarray) -> float:
    """Exact closest-point distance to one triangle."""
    a, b, c = tri
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = float(ab @ ap), float(ac @ ap)
    if d1 <= 0 and d2 <= 0:
        return float(np.linalg.norm(ap))
    bp = p - b
    d3, d4 = float(ab @ bp), float(ac @ bp)
    if d3 >= 0 and d4 <= d3:
        return float(np.linalg.norm(bp))
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return float(np.linalg.norm(p - (a + d1 / (d1 - d3) * ab)))
    cp = p - c
    d5, d6 = float(ab @ cp), float(ac @ cp)
    if d6 >= 0 and d5 <= d6:
        return float(np.linalg.norm(cp))
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return float(np.linalg.norm(p - (a + d2 / (d2 - d6) * ac)))
    va = d3 * d6 - d5 * d4
    if va <= 0 and d4 - d3 >= 0 and d5 - d6 >= 0:
        return float(np.linalg.norm(p - (b + (d4 - d3) / ((d4 - d3) + (d5 - d6)) * (c - b))))
    normal = np.cross(ab, ac)
    return abs(float((p - a) @ normal)) / max(float(np.linalg.norm(normal)), 1e-12)


def distance_to_mesh(position: np.ndarray, triangles: np.ndarray) -> float:
    return min((_point_triangle_distance(position, tri) for tri in triangles), default=float("inf"))


def segment_intersects_mesh(start: np.ndarray, end: np.ndarray, triangles: np.ndarray) -> bool:
    """Möller–Trumbore segment test used for swept drone collision."""
    direction = end - start
    for tri in triangles:
        a, b, c = tri
        edge1, edge2 = b - a, c - a
        h = np.cross(direction, edge2)
        det = float(edge1 @ h)
        if abs(det) < 1e-10:
            continue
        inv_det = 1.0 / det
        s = start - a
        u = inv_det * float(s @ h)
        if not 0.0 <= u <= 1.0:
            continue
        q = np.cross(s, edge1)
        v = inv_det * float(direction @ q)
        if v < 0.0 or u + v > 1.0:
            continue
        t = inv_det * float(edge2 @ q)
        if 0.0 <= t <= 1.0:
            return True
    return False


def capsule_intersects_mesh(
    start: np.ndarray, end: np.ndarray, triangles: np.ndarray, radius_m: float,
) -> bool:
    """Conservative swept-volume test for a finite-radius camera/drone."""
    if radius_m < 0:
        raise ValueError("radius_m must be non-negative")
    start, end = np.asarray(start, float), np.asarray(end, float)
    if segment_intersects_mesh(start, end, triangles):
        return True
    length = float(np.linalg.norm(end - start))
    step = max(radius_m * 0.5, 0.02)
    count = max(2, int(math.ceil(length / step)) + 1)
    return any(
        distance_to_mesh(start + t * (end - start), triangles) <= radius_m
        for t in np.linspace(0.0, 1.0, count)
    )


def yaw_pitch_quaternion(yaw_deg: float, pitch_deg: float) -> tuple[float, float, float, float]:
    """Camera orientation quaternion in xyzw convention (roll=0)."""
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    return (-sp * sy, sp * cy, cp * sy, cp * cy)


def _visible_fraction_from_points(
    position: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
    points: np.ndarray,
) -> tuple[float, float, float]:
    if points is None or len(points) == 0:
        return 0.0, 1.0, 0.0
    vecs = points - position
    dists = np.linalg.norm(vecs, axis=1)
    if dists.max() < 1e-6:
        return 0.0, 1.0, 0.0
    dirs = vecs / dists[:, None]

    yaw_rad = math.radians(yaw_deg)
    pitch_rad = math.radians(pitch_deg)
    fov_rad = math.radians(fov_deg / 2.0)

    forward = np.array([
        math.cos(pitch_rad) * math.sin(yaw_rad),
        math.cos(pitch_rad) * math.cos(yaw_rad),
        math.sin(pitch_rad),
    ])
    forward = forward / np.linalg.norm(forward)

    cos_angles = dirs @ forward
    in_fov = cos_angles >= math.cos(fov_rad)
    visible_pts = points[in_fov]
    if len(visible_pts) == 0:
        return 0.0, 1.0, 0.0

    reconstructed = len(visible_pts) / len(points)
    unknown = max(0.0, 1.0 - reconstructed)
    return reconstructed, unknown, reconstructed


def _texture_reality_for_pose(
    workspace: Workspace,
    position: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
    output_width_px: int,
) -> tuple[bool | None, str | None, list[str]]:
    """Evaluate every canonical textured surface visible from this pose."""
    audit_path = (
        workspace.path("11_conditioning", "facade_texture_audit.json")
        if hasattr(workspace, "path")
        else Path(workspace) / "11_conditioning" / "facade_texture_audit.json"
    )
    if not audit_path.is_file():
        return None, None, []
    textures = (json.loads(audit_path.read_text("utf-8")).get("textures") or [])
    if not textures:
        return False, "UNSUPPORTED", ["no canonical surface texture evidence"]
    from .texture_reality import (
        CameraTextureDemand,
        TextureEvidence,
        TextureRealityLevel,
        evaluate_texture_reality,
    )

    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    forward = np.array([
        math.cos(pitch) * math.sin(yaw),
        math.cos(pitch) * math.cos(yaw),
        math.sin(pitch),
    ])
    cos_limit = math.cos(math.radians(fov_deg) / 2.0)
    violations: list[str] = []
    levels: list[str] = []
    rank = {"UNSUPPORTED": 0, "DISTANT_ONLY": 1, "SAFE_FOR_NOVEL_VIEW": 2, "SAFE_FOR_CLOSEUP": 3}
    for texture in textures:
        tile_candidates = []
        for tile in texture.get("reality_tiles") or []:
            centre = np.asarray(tile.get("centre_world") or [], dtype=float)
            normal = np.asarray(tile.get("normal") or [], dtype=float)
            if centre.shape != (3,) or normal.shape != (3,):
                continue
            ray = centre - position
            distance = float(np.linalg.norm(ray))
            if distance <= 1e-9 or float((ray / distance) @ forward) < cos_limit:
                continue
            if float(normal @ (position - centre)) <= 0:
                continue
            tile_candidates.append((distance, tile))
        if tile_candidates:
            for distance, tile in tile_candidates:
                tile_evidence = TextureEvidence(
                    tile.get("effective_gsd_m"), float(tile.get("coverage") or 0.0),
                    float(tile.get("sharpness") or 0.0), float(tile.get("view_count") or 0.0),
                    float(tile.get("pose_confidence") or 0.0), float(tile.get("incidence_deg") or 90.0),
                    float(tile.get("photometric_consistency") or 0.0),
                    float(tile.get("unknown_fraction") or 0.0),
                )
                tile_result = evaluate_texture_reality(
                    tile_evidence, CameraTextureDemand(distance, fov_deg, output_width_px),
                    required_level=TextureRealityLevel.SAFE_FOR_NOVEL_VIEW,
                )
                levels.append(tile_result.level.value)
                if not tile_result.safe:
                    violations.append(
                        f"{texture.get('surface_id')} tile {tile.get('tile')}: "
                        f"{tile_result.level.value}; upscale={tile_result.upscale_factor:.2f}"
                    )
            continue
        visible_distances = []
        for triangle in texture.get("render_triangles") or []:
            vertices = np.asarray(triangle.get("vertices") or [], dtype=float)
            if vertices.shape != (3, 3):
                continue
            centre = vertices.mean(axis=0)
            ray = centre - position
            distance = float(np.linalg.norm(ray))
            if distance <= 1e-9 or float((ray / distance) @ forward) < cos_limit:
                continue
            normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
            if float(normal @ (position - centre)) <= 0:
                continue
            visible_distances.append(distance)
        if not visible_distances:
            continue
        evidence = TextureEvidence(
            texture.get("median_effective_gsd_m"),
            float(texture.get("observed_fraction") or 0.0),
            float(texture.get("median_sharpness") or 0.0),
            float(texture.get("median_view_count") or 0.0),
            float(texture.get("pose_confidence") or 0.0),
            float(texture.get("median_incidence_deg") or 90.0),
            float(texture.get("photometric_consistency") or 0.0),
            float(texture.get("unknown_fraction") or 1.0),
        )
        result = evaluate_texture_reality(
            evidence,
            CameraTextureDemand(min(visible_distances), fov_deg, output_width_px),
            required_level=TextureRealityLevel.SAFE_FOR_NOVEL_VIEW,
        )
        levels.append(result.level.value)
        if not result.safe:
            violations.append(
                f"{texture.get('surface_id')}: {result.level.value}; "
                f"upscale={result.upscale_factor:.2f}; min={result.min_safe_distance_m:.2f}m"
            )
    if not levels:
        return None, None, []
    worst = min(levels, key=rank.get)
    return not violations, worst, violations


# ---------------------------------------------------------------------------
# Évaluateur
# ---------------------------------------------------------------------------


class CameraFeasibilityEvaluator:
    """Évalue la faisabilité d'une pose caméra sur la reconstruction."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def evaluate_pose(
        self,
        *,
        pose_id: str,
        position_local_m: tuple[float, float, float],
        yaw_deg: float,
        pitch_deg: float,
        fov_deg: float,
        min_distance_m: float = 0.0,
        reconstructed_fraction: float = 0.0,
        proxy_fraction: float = 0.0,
        unknown_fraction: float = 0.0,
        reconstruction_run_id: str | None = None,
        safety_radius_m: float = 0.35,
        previous_position_local_m: tuple[float, float, float] | None = None,
        near_m: float = 0.05,
        far_m: float = 10_000.0,
        output_width_px: int = 1920,
        canonical_mesh=None,
        canonical_camera=None,
    ) -> CameraFeasibilityField:
        reconstructed = reconstructed_fraction
        unknown = unknown_fraction
        proxy = proxy_fraction

        exact = None
        if (canonical_mesh is None) != (canonical_camera is None):
            raise ValueError("canonical_mesh and canonical_camera must be provided together")
        if canonical_mesh is not None:
            audit_path = (
                self.workspace.path("11_conditioning", "facade_texture_audit.json")
                if hasattr(self.workspace, "path") else
                Path(self.workspace) / "11_conditioning" / "facade_texture_audit.json"
            )
            minimums = {}
            if audit_path.is_file():
                audit = json.loads(audit_path.read_text("utf-8"))
                for texture in audit.get("textures") or []:
                    profile = (texture.get("texture_reality") or {}).get("1080p_60deg") or {}
                    value = profile.get("SAFE_FOR_NOVEL_VIEW")
                    if value is not None and math.isfinite(float(value)):
                        minimums[str(texture["surface_id"])] = float(value)
            exact = evaluate_canonical_camera(
                canonical_mesh, canonical_camera, texture_min_distances=minimums,
            )
            reconstructed = 1.0 - exact["unknown_visible_fraction"]
            unknown = exact["unknown_visible_fraction"]
            proxy = 0.0
        elif reconstruction_run_id is not None:
            points = _load_run_points(reconstruction_run_id, self.workspace)
            recon_frac, unk_frac, vis_frac = _visible_fraction_from_points(
                np.array(position_local_m), yaw_deg, pitch_deg, fov_deg, points
            )
            if recon_frac > 0:
                reconstructed = recon_frac
                unknown = unk_frac
                proxy = max(0.0, 1.0 - reconstructed - unknown)

        visible = max(0.0, min(1.0, reconstructed + proxy * 0.5))
        triangles = (
            canonical_mesh.vertices[canonical_mesh.faces]
            if canonical_mesh is not None else _load_scene_triangles(self.workspace)
        )
        collision_position = (
            canonical_camera.position() if canonical_camera is not None
            else np.asarray(position_local_m, dtype=float)
        )
        scene_distance = distance_to_mesh(collision_position, triangles)
        collision = bool(scene_distance <= safety_radius_m)
        if previous_position_local_m is not None:
            collision = collision or capsule_intersects_mesh(
                np.asarray(previous_position_local_m, dtype=float),
                np.asarray(position_local_m, dtype=float), triangles, safety_radius_m,
            )
        distance_violation = False

        if min_distance_m > 0:
            distance_violation = scene_distance < min_distance_m

        texture_safe, texture_level, texture_violations = (None, None, [])
        if exact is None:
            texture_safe, texture_level, texture_violations = _texture_reality_for_pose(
                self.workspace, np.asarray(position_local_m, dtype=float),
                yaw_deg, pitch_deg, fov_deg, output_width_px,
            )
        if texture_safe is False:
            distance_violation = True
        if exact is not None and not exact["accepted"]:
            distance_violation = True

        framing = visible if not distance_violation else 0.0
        overall = visible * 0.7 + framing * 0.3

        # Orientation complète stockée : matrice ET quaternion. Aucune
        # validation en aval ne reconstruit la visée à partir du yaw seul.
        rotation = pose_rotation_matrix(yaw_deg, pitch_deg)
        quaternion = _rotation_to_quaternion(rotation)

        return CameraFeasibilityField(
            pose_id=pose_id,
            position_local_m=position_local_m,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            fov_deg=fov_deg,
            orientation_xyzw=yaw_pitch_quaternion(yaw_deg, pitch_deg),
            near_m=near_m,
            far_m=far_m,
            orientation_matrix=tuple(float(v) for v in rotation.reshape(-1)),
            orientation_quaternion=quaternion,
            visible_surface_confidence=round(visible, 3),
            unknown_fraction=round(unknown, 3),
            proxy_fraction=round(proxy, 3),
            reconstructed_fraction=round(reconstructed, 3),
            minimum_distance_violation=distance_violation,
            texture_reality_safe=texture_safe,
            texture_reality_level=texture_level,
            texture_reality_violations=texture_violations,
            requested_output_width_px=output_width_px,
            collision=collision,
            distance_to_scene_m=None if not math.isfinite(scene_distance) else round(scene_distance, 3),
            framing_quality=round(framing, 3),
            overall_score=round(overall, 3),
            visible_surfaces=(exact or {}).get("visible_surfaces", []),
            target_pixel_fraction=(exact or {}).get("target_pixel_fraction", 0.0),
            unknown_visible_fraction=(exact or {}).get("unknown_visible_fraction", 0.0),
            minimum_clearance_m=(exact or {}).get("minimum_clearance_m"),
            subject_score=(exact or {}).get("subject_score", 0.0),
            accepted=(
                bool(exact["accepted"] and not collision and not distance_violation)
                if exact is not None else bool(not collision and not distance_violation)
            ),
            rejection_reasons=(exact or {}).get("rejection_reasons", [])
            + (["collision"] if collision else []),
            feasibility_mesh_digest=(exact or {}).get("input_mesh_digest"),
        )


def build_validated_camera_path(
    workspace: Workspace,
    reconstruction_run_id: str,
    *,
    probe_path_path: str | None = None,
) -> ValidatedCameraPath:
    from shapely.geometry import Polygon

    from .scene_package import _camera_path as build_probe_path

    if probe_path_path:
        try:
            probe = ValidatedCameraPath.model_validate_json(
                Path(probe_path_path).read_text("utf-8")
            )
            return probe
        except Exception:
            pass

    points = _load_run_points(reconstruction_run_id, workspace)
    if points is not None and len(points) > 0:
        centroid = points.mean(axis=0)
        spread = float(np.linalg.norm(points[:, :2].max(axis=0) - points[:, :2].min(axis=0)))
        radius = max(spread * 0.8, 5.0)
        height = float(points[:, 2].mean() + points[:, 2].std()) if len(points) > 1 else 10.0
        polygon = Polygon([
            (centroid[0] - radius, centroid[1] - radius),
            (centroid[0] + radius, centroid[1] - radius),
            (centroid[0] + radius, centroid[1] + radius),
            (centroid[0] - radius, centroid[1] + radius),
        ])
    else:
        polygon = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
        height = 10.0

    probe = build_probe_path(polygon, height_m=height, fov_deg=80.0)

    path_id = (
        f"validated-path-{reconstruction_run_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )

    evaluator = CameraFeasibilityEvaluator(workspace)
    poses = []
    previous_position = None
    for idx, pose in enumerate(probe.poses):
        field = evaluator.evaluate_pose(
            pose_id=f"pose-{idx:03d}",
            position_local_m=pose.position_local_m,
            yaw_deg=pose.azimuth_deg,
            pitch_deg=_pitch_of_pose(pose),
            fov_deg=pose.fov_horizontal_deg,
            reconstruction_run_id=reconstruction_run_id,
            previous_position_local_m=previous_position,
        )
        poses.append(field)
        previous_position = pose.position_local_m

    derivation = (
        f"trajectoire probe autour du nuage reconstruit "
        f"({len(points) if points is not None else 0} points) ; validé par faisabilité caméra"
    )

    return ValidatedCameraPath(
        path_id=path_id,
        reconstruction_run_id=reconstruction_run_id,
        simulation_only=True,
        derivation=derivation,
        poses=poses,
    )


def _pitch_of_pose(pose) -> float:  # noqa: ANN001
    """Pitch complet d'une pose, dérivé de sa géométrie position→look_at.

    L'orientation n'est jamais réduite au yaw : le pitch mesuré par la
    géométrie de la pose traverse tout le pipeline, et la matrice de
    rotation complète reste disponible sur le champ validé.
    """
    look_at = getattr(pose, "look_at_local_m", None)
    if look_at is None:
        return 0.0
    position = np.asarray(pose.position_local_m, dtype=np.float64)
    target = np.asarray(look_at, dtype=np.float64)
    delta = target - position
    horizontal = math.hypot(delta[0], delta[1])
    if horizontal < 1e-9:
        return 90.0 if delta[2] > 0 else -90.0
    return math.degrees(math.atan2(delta[2], horizontal))


def pose_rotation_matrix(
    yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0
) -> np.ndarray:
    """Matrice de rotation monde→caméra (convention Z vers l'avant).

    Toute validation consomme cette pose **complète** — jamais une
    reconstruction partielle à partir du seul yaw.
    """
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return ry @ rx @ rz


def _rotation_to_quaternion(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z) d'une matrice de rotation — trace maximale."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    return (w / norm, x / norm, y / norm, z / norm)


def publish_validated_path(path: ValidatedCameraPath, workspace: Workspace) -> Path:
    """Publie la trajectoire validée sous `07_revalidation/`."""
    out = workspace.path("07_reconstruction", "validated_camera_paths")
    out.mkdir(parents=True, exist_ok=True)
    path_file = out / f"{path.path_id}.json"
    path_file.write_text(json.dumps(path.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n")
    return path_file


__all__ = [
    "CameraFeasibilityEvaluator",
    "build_validated_camera_path",
    "publish_validated_path",
]
