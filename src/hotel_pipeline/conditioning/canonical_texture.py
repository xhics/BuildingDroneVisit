"""Direct photographic texturing of CanonicalSceneMesh physical surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..reality_contract import MeshConsumerReceipt, require_canonical_mesh
from ..schemas.canonical_states import MeasurementState


@dataclass(frozen=True)
class SurfaceTextureChart:
    surface_id: str
    triangle_ids: tuple[int, ...]
    vertex_ids: np.ndarray
    uv_vertices: np.ndarray
    uv_triangles: np.ndarray
    world_triangles: np.ndarray
    texel_size_m: float
    width_px: int
    height_px: int
    origin_world: np.ndarray
    basis_u: np.ndarray
    basis_v: np.ndarray
    normal: np.ndarray

    def world_to_uv(self, points: np.ndarray) -> np.ndarray:
        delta = np.asarray(points, dtype=float) - self.origin_world
        return np.column_stack([delta @ self.basis_u, delta @ self.basis_v])


@dataclass
class TextureObservation:
    image_id: str
    image: np.ndarray
    camera: object
    valid_mask: np.ndarray | None = None
    proxy_depth: object | None = None
    lidar_occlusion: object | None = None
    face_id_offset: int = 0
    sharpness: float | None = None
    pose_confidence: float = 1.0
    pose_error_m: float = 0.0


@dataclass
class SurfaceTextureAtlas:
    chart: SurfaceTextureChart
    rgba: np.ndarray
    state: np.ndarray
    best_source: np.ndarray
    effective_gsd: np.ndarray
    incidence_deg: np.ndarray
    sharpness: np.ndarray
    view_count: np.ndarray
    confidence: np.ndarray
    variance: np.ndarray
    source_mask: np.ndarray
    source_image_ids: list[str]
    receipt: MeshConsumerReceipt
    rejection_counts: dict[str, int] = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        return float(np.mean(self.state != MeasurementState.UNKNOWN.value))

    def texel_provenance(self, x: int, y: int) -> dict:
        source_index = int(self.best_source[y, x])
        return {
            "surface_id": self.chart.surface_id,
            "state": str(self.state[y, x]),
            "best_source": (
                self.source_image_ids[source_index] if source_index >= 0 else None
            ),
            "source_image_ids": [
                image_id for index, image_id in enumerate(self.source_image_ids[:64])
                if int(self.source_mask[y, x]) & (1 << index)
            ],
            "effective_gsd_m": (
                float(self.effective_gsd[y, x])
                if np.isfinite(self.effective_gsd[y, x]) else None
            ),
            "incidence_deg": (
                float(self.incidence_deg[y, x])
                if np.isfinite(self.incidence_deg[y, x]) else None
            ),
            "sharpness": float(self.sharpness[y, x]),
            "view_count": int(self.view_count[y, x]),
            "confidence": float(self.confidence[y, x]),
            "variance": float(self.variance[y, x]),
        }


@dataclass(frozen=True)
class RenderedTexelTrace:
    pixel: tuple[int, int]
    triangle_id: int
    surface_id: str
    uv_m: tuple[float, float]
    texel: tuple[int, int]
    provenance: dict


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("zero-length texture basis")
    return vector / norm


def build_surface_chart(mesh, surface_id: str, texel_size_m: float = 0.12):
    """Parameterize one physical surface from its canonical triangles only."""
    require_canonical_mesh(mesh, "surface_uv_chart")
    indices = [index for index, value in enumerate(mesh.surface_ids) if value == surface_id]
    if not indices:
        raise KeyError(f"unknown surface: {surface_id}")
    face_vertices = mesh.faces[indices]
    vertex_ids = np.unique(face_vertices)
    world = mesh.vertices[vertex_ids]
    normal = _unit(np.mean([mesh._triangle_normal(index) for index in indices], axis=0))
    kind = mesh.face_kind[indices[0]]
    if kind in {"wall", "roof_step"} and abs(normal[2]) < 0.5:
        basis_v = np.array([0.0, 0.0, 1.0])
        basis_u = _unit(np.cross(basis_v, normal))
    else:
        seed = np.array([1.0, 0.0, 0.0])
        if abs(float(seed @ normal)) > 0.9:
            seed = np.array([0.0, 1.0, 0.0])
        basis_u = _unit(seed - normal * float(seed @ normal))
        basis_v = _unit(np.cross(normal, basis_u))
    raw_uv = np.column_stack([world @ basis_u, world @ basis_v])
    uv_min = raw_uv.min(axis=0)
    uv = raw_uv - uv_min
    origin_world = (
        world[0]
        + basis_u * (uv_min[0] - raw_uv[0, 0])
        + basis_v * (uv_min[1] - raw_uv[0, 1])
    )
    lookup = {int(vertex): index for index, vertex in enumerate(vertex_ids)}
    uv_faces = np.asarray([[lookup[int(vertex)] for vertex in face] for face in face_vertices])
    span = np.maximum(uv.max(axis=0), texel_size_m)
    return SurfaceTextureChart(
        surface_id=surface_id,
        triangle_ids=tuple(int(mesh.triangle_ids[index]) for index in indices),
        vertex_ids=vertex_ids,
        uv_vertices=uv,
        uv_triangles=uv_faces,
        world_triangles=np.asarray(indices, dtype=np.int64),
        texel_size_m=float(texel_size_m),
        width_px=max(1, int(np.ceil(span[0] / texel_size_m))),
        height_px=max(1, int(np.ceil(span[1] / texel_size_m))),
        origin_world=origin_world,
        basis_u=basis_u,
        basis_v=basis_v,
        normal=normal,
    )


def observation_weight(
    incidence_cosine: float, gsd_m: float, sharpness: float,
    pose_confidence: float, pose_error_m: float, border_score: float,
) -> float:
    incidence = float(np.clip(incidence_cosine, 0.0, 1.0)) ** 2
    gsd_score = float(np.clip(0.03 / max(gsd_m, 1e-6), 0.05, 1.0))
    pose_score = float(np.exp(-((max(0.0, pose_error_m) / 0.22) ** 2)))
    return float(
        incidence * gsd_score * np.clip(sharpness, 0.0, 1.0)
        * np.clip(pose_confidence, 0.0, 1.0) * pose_score
        * np.clip(border_score, 0.0, 1.0)
    )


def _barycentric(point: np.ndarray, triangle: np.ndarray) -> np.ndarray | None:
    a, b, c = triangle
    v0, v1, v2 = b - a, c - a, point - a
    den = float(v0[0] * v1[1] - v1[0] * v0[1])
    if abs(den) <= 1e-12:
        return None
    u = float((v2[0] * v1[1] - v1[0] * v2[1]) / den)
    v = float((v0[0] * v2[1] - v2[0] * v0[1]) / den)
    bary = np.array([1.0 - u - v, u, v])
    return bary if np.min(bary) >= -1e-7 else None


def _texel_world_samples(mesh, chart: SurfaceTextureChart):
    samples = []
    for y in range(chart.height_px):
        for x in range(chart.width_px):
            uv = np.array([(x + 0.5) * chart.texel_size_m, (y + 0.5) * chart.texel_size_m])
            for local_face, mesh_face_index in zip(chart.uv_triangles, chart.world_triangles):
                bary = _barycentric(uv, chart.uv_vertices[local_face])
                if bary is None:
                    continue
                world = bary @ mesh.vertices[mesh.faces[mesh_face_index]]
                samples.append((x, y, world, int(mesh_face_index)))
                break
    return samples


def _sharpness(image: np.ndarray) -> float:
    gray = np.asarray(image, dtype=float).mean(axis=2)
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    energy = float(np.mean(gx * gx) + np.mean(gy * gy))
    return float(np.clip(np.sqrt(energy) / 32.0, 0.05, 1.0))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    result = np.zeros(values.shape[1], dtype=float)
    for channel in range(values.shape[1]):
        order = np.argsort(values[:, channel])
        cumulative = np.cumsum(weights[order])
        result[channel] = values[order[np.searchsorted(cumulative, cumulative[-1] * 0.5)], channel]
    return result


def texture_surface(
    mesh, surface_id: str, observations: list[TextureObservation],
    texel_size_m: float = 0.12, depth_tolerance_m: float = 0.25,
) -> SurfaceTextureAtlas:
    """Project registered photos onto exact canonical triangles and robustly fuse."""
    receipt = require_canonical_mesh(mesh, "canonical_surface_texture")
    chart = build_surface_chart(mesh, surface_id, texel_size_m)
    height, width = chart.height_px, chart.width_px
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    state = np.full((height, width), MeasurementState.UNKNOWN.value, dtype="U10")
    best_source = np.full((height, width), -1, dtype=np.int32)
    effective_gsd = np.full((height, width), np.inf, dtype=float)
    incidence_deg = np.full((height, width), np.inf, dtype=float)
    sharpness_map = np.zeros((height, width), dtype=float)
    view_count = np.zeros((height, width), dtype=np.uint16)
    confidence = np.zeros((height, width), dtype=float)
    variance = np.zeros((height, width), dtype=float)
    source_mask = np.zeros((height, width), dtype=np.uint64)
    rejection_counts: dict[str, int] = {}
    samples_by_texel: dict[tuple[int, int], list[tuple]] = {}
    medians = [float(np.median(obs.image)) for obs in observations if obs.image.size]
    target_median = float(np.median(medians)) if medians else 128.0

    for x, y, world, face_index in _texel_world_samples(mesh, chart):
        normal = mesh._triangle_normal(face_index)
        for source_index, observation in enumerate(observations):
            screen, depths = observation.camera.project(world.reshape(1, 3))
            if screen is None or float(depths[0]) <= 0:
                rejection_counts["behind_camera"] = rejection_counts.get("behind_camera", 0) + 1
                continue
            px, py = (round(float(value)) for value in screen[0])
            image_height, image_width = observation.image.shape[:2]
            if not (0 <= px < image_width and 0 <= py < image_height):
                rejection_counts["outside_image"] = rejection_counts.get("outside_image", 0) + 1
                continue
            if observation.valid_mask is not None and not bool(observation.valid_mask[py, px]):
                rejection_counts["semantic_or_occluder_mask"] = rejection_counts.get("semantic_or_occluder_mask", 0) + 1
                continue
            expected_face = observation.face_id_offset + face_index
            if observation.proxy_depth is not None:
                proxy_z, proxy_face = observation.proxy_depth.hit(px, py)
                if proxy_face != expected_face or abs(proxy_z - float(depths[0])) > depth_tolerance_m:
                    rejection_counts["occluded_by_mesh"] = rejection_counts.get("occluded_by_mesh", 0) + 1
                    continue
            lidar = observation.lidar_occlusion
            if lidar is not None and lidar.valid[py, px] and lidar.depth[py, px] < float(depths[0]) - 1.5:
                rejection_counts["occluded_by_lidar"] = rejection_counts.get("occluded_by_lidar", 0) + 1
                continue
            camera_position = observation.camera.position
            if callable(camera_position):
                camera_position = camera_position()
            to_camera = np.asarray(camera_position, dtype=float) - world
            distance = float(np.linalg.norm(to_camera))
            incidence = abs(float(normal @ _unit(to_camera)))
            if incidence < np.cos(np.radians(65.0)):
                rejection_counts["grazing_angle"] = rejection_counts.get("grazing_angle", 0) + 1
                continue
            focal = float(
                observation.camera.f
                if hasattr(observation.camera, "f")
                else observation.camera.focal[0]
            )
            gsd = distance / max(focal * incidence, 1e-6)
            border = min(px, py, image_width - 1 - px, image_height - 1 - py)
            border_score = float(np.clip(border / 12.0, 0.05, 1.0))
            sharpness = observation.sharpness if observation.sharpness is not None else _sharpness(observation.image)
            weight = observation_weight(
                incidence, gsd, sharpness, observation.pose_confidence,
                observation.pose_error_m, border_score,
            )
            if weight <= 1e-6:
                continue
            colour = observation.image[py, px, :3].astype(float)
            image_median = max(float(np.median(observation.image)), 1.0)
            gain = float(np.clip(target_median / image_median, 0.6, 1.6))
            colour = np.clip(colour * gain, 0, 255)
            samples_by_texel.setdefault((x, y), []).append(
                (
                    colour, weight, source_index, gsd,
                    np.degrees(np.arccos(np.clip(incidence, 0.0, 1.0))), sharpness,
                )
            )

    for (x, y), samples in samples_by_texel.items():
        colours = np.asarray([sample[0] for sample in samples])
        weights = np.asarray([sample[1] for sample in samples], dtype=float)
        centre = np.median(colours, axis=0)
        distances = np.linalg.norm(colours - centre, axis=1)
        mad = float(np.median(np.abs(distances - np.median(distances))))
        keep = distances <= np.median(distances) + max(8.0, 2.5 * mad)
        colours, weights = colours[keep], weights[keep]
        kept = [sample for sample, good in zip(samples, keep) if good]
        if not len(kept):
            continue
        fused = _weighted_median(colours, weights)
        winner = int(np.argmax(weights))
        rgba[y, x, :3] = np.round(fused).astype(np.uint8)
        rgba[y, x, 3] = 255
        state[y, x] = MeasurementState.MEASURED.value
        best_source[y, x] = int(kept[winner][2])
        view_count[y, x] = len(kept)
        effective_gsd[y, x] = float(
            np.average([sample[3] for sample in kept], weights=weights)
        )
        incidence_deg[y, x] = float(kept[winner][4])
        sharpness_map[y, x] = float(kept[winner][5])
        confidence[y, x] = float(np.clip(1.0 - np.exp(-weights.sum()), 0.0, 1.0))
        variance[y, x] = float(np.mean(np.var(colours, axis=0)))
        source_mask[y, x] = np.uint64(sum(
            1 << int(sample[2]) for sample in kept if int(sample[2]) < 64
        ))

    return SurfaceTextureAtlas(
        chart, rgba, state, best_source, effective_gsd, incidence_deg,
        sharpness_map, view_count,
        confidence, variance, source_mask,
        [observation.image_id for observation in observations],
        receipt, rejection_counts,
    )


def trace_rendered_pixel(
    frame, mesh, atlases: dict[str, SurfaceTextureAtlas], x: int, y: int,
) -> RenderedTexelTrace | None:
    """Trace pixel -> canonical triangle -> surface -> UV -> source photos."""
    if not (0 <= x < frame.width and 0 <= y < frame.height):
        return None
    face_index = int(frame.triangle_id[y, x])
    depth = float(frame.depth_z[y, x])
    if face_index < 0 or not np.isfinite(depth):
        return None
    surface_id = mesh.surface_ids[face_index]
    atlas = atlases.get(surface_id)
    if atlas is None:
        return None
    world = frame.camera.unproject(x + 0.5, y + 0.5, depth)
    uv = atlas.chart.world_to_uv(world.reshape(1, 3))[0]
    tx = int(np.floor(uv[0] / atlas.chart.texel_size_m))
    ty = int(np.floor(uv[1] / atlas.chart.texel_size_m))
    if not (0 <= tx < atlas.chart.width_px and 0 <= ty < atlas.chart.height_px):
        return None
    return RenderedTexelTrace(
        pixel=(x, y), triangle_id=int(mesh.triangle_ids[face_index]),
        surface_id=surface_id, uv_m=(float(uv[0]), float(uv[1])),
        texel=(tx, ty), provenance=atlas.texel_provenance(tx, ty),
    )


__all__ = [
    "RenderedTexelTrace", "SurfaceTextureAtlas", "SurfaceTextureChart", "TextureObservation",
    "build_surface_chart", "observation_weight", "texture_surface",
    "trace_rendered_pixel",
]
