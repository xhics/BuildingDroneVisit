"""Direct glTF export of the canonical reality scene.

The exporter is intentionally dumb about *geometry truth*: it never re-extrudes a
building.  Buildings are copied from their canonical mesh.  Environment objects
whose canonical contract is already a primitive (vegetation envelopes and thin
street furniture) are tessellated only for rendering/export.

Both payload shapes used in the repository are accepted:
- ``conditioned_scene.json`` publishes ``buildings``;
- the viewer compatibility payload publishes ``volumes``.

That alias is resolved here once so the production package and the viewer cannot
silently render different worlds.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .schemas.canonical_states import MeasurementState, SurfaceType


@dataclass(frozen=True)
class _MeshPart:
    label: str
    category: str
    owner: dict
    vertices: np.ndarray
    faces: tuple[tuple[int, ...], ...]
    face_kind: tuple[str, ...]


def _buildings(payload: dict) -> list[dict]:
    """Return the authoritative building collection for either scene contract."""
    buildings = payload.get("buildings")
    if buildings is not None:
        return list(buildings)
    return list(payload.get("volumes") or [])


def _evidence(
    owner: dict, *, triangle_id: int, surface_id: str, surface_type: SurfaceType
) -> dict:
    source_ids = list(owner.get("source_ids") or owner.get("source_refs") or [])
    requested = str(
        owner.get("state") or owner.get("measurement_state") or "UNKNOWN"
    ).upper()
    state = (
        MeasurementState(requested)
        if requested in MeasurementState._value2member_map_
        else MeasurementState.UNKNOWN
    )
    if state is MeasurementState.MEASURED and not source_ids:
        state = MeasurementState.UNKNOWN
    return {
        "triangle_id": triangle_id,
        "surface_id": surface_id,
        "surface_type": surface_type.value,
        "state": state.value,
        "confidence": float(owner.get("confidence", 0.0) or 0.0),
        "source_ids": source_ids,
        "source_dates": list(owner.get("source_dates") or []),
        "generation_method": owner.get(
            "generation_method", "canonical_mesh_passthrough"
        ),
        "residual": owner.get("residual"),
        "dependencies": owner.get("dependencies") or {},
    }


def _faces_from_payload(mesh: dict) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(index) for index in face)
        for face in (mesh.get("faces") or [])
        if len(face) >= 3
    )


def _feature_mesh(feature: dict) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Read a fine architectural mesh without inventing triangle fans.

    Newer producers put ``vertices``/``faces`` at the feature root, while some
    experimental producers wrap them in ``mesh``.  If no faces are provided,
    a flat polygon may still be triangulated as a fan for backward
    compatibility; volumetric primitives are expected to provide their faces.
    """
    mesh = feature.get("mesh") if isinstance(feature.get("mesh"), dict) else feature
    vertices = np.asarray(mesh.get("vertices") or [], dtype=np.float64).reshape((-1, 3))
    faces = _faces_from_payload(mesh)
    if len(vertices) >= 3 and not faces:
        faces = tuple(
            (0, index, index + 1) for index in range(1, len(vertices) - 1)
        )
    return vertices, faces


def _vegetation_mesh(
    patch: dict,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Tessellate a canonical LiDAR/fallback canopy envelope.

    ``rings`` are already part of the canonical vegetation contract.  This
    function merely serializes that volume to triangles; it does not infer a
    species or reshape the envelope.
    """
    rings = [
        np.asarray(ring, dtype=np.float64).reshape((-1, 3))
        for ring in (patch.get("rings") or patch.get("envelope") or [])
        if len(ring) >= 3
    ]
    if not rings:
        return np.empty((0, 3), dtype=np.float64), ()

    vertices = np.vstack(rings)
    offsets: list[int] = []
    cursor = 0
    for ring in rings:
        offsets.append(cursor)
        cursor += len(ring)

    faces: list[tuple[int, ...]] = []
    for ring_index in range(len(rings) - 1):
        lower, upper = rings[ring_index], rings[ring_index + 1]
        # Canonical envelopes normally use equal ring cardinality.  When they
        # do not, connect only the common samples rather than creating crossed
        # quads and mark the remaining uncertainty in provenance.
        count = min(len(lower), len(upper))
        if count < 3:
            continue
        lo, hi = offsets[ring_index], offsets[ring_index + 1]
        for index in range(count):
            nxt = (index + 1) % count
            faces.append((lo + index, lo + nxt, hi + nxt, hi + index))

    # Close the occupancy envelope.  Vegetation transparency is an appearance
    # property; collision/occlusion still needs a bounded volume.
    first = offsets[0]
    faces.append(tuple(reversed([first + i for i in range(len(rings[0]))])))
    last = offsets[-1]
    faces.append(tuple(last + i for i in range(len(rings[-1]))))
    return vertices, tuple(faces)


def _furniture_mesh(
    item: dict, *, segments: int = 12
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Serialize measured street furniture as a thin canonical cylinder."""
    centre = item.get("centre") or item.get("c")
    radius = item.get("radius_m", item.get("r"))
    height = item.get("height_m", item.get("h"))
    if centre is None or radius is None or height is None:
        return np.empty((0, 3), dtype=np.float64), ()
    cx, cy = float(centre[0]), float(centre[1])
    base_z = float(item.get("ground_z_m", item.get("ground_z", 0.0)) or 0.0)
    radius = max(float(radius), 1e-3)
    height = max(float(height), 1e-3)

    vertices: list[list[float]] = []
    for z in (base_z, base_z + height):
        vertices.extend(
            [
                [
                    cx + radius * math.cos(2.0 * math.pi * index / segments),
                    cy + radius * math.sin(2.0 * math.pi * index / segments),
                    z,
                ]
                for index in range(segments)
            ]
        )
    faces: list[tuple[int, ...]] = []
    for index in range(segments):
        nxt = (index + 1) % segments
        faces.append((index, nxt, segments + nxt, segments + index))
    faces.append(tuple(reversed(range(segments))))
    faces.append(tuple(range(segments, 2 * segments)))
    return np.asarray(vertices, dtype=np.float64), tuple(faces)


def _mesh_parts(payload: dict) -> list[_MeshPart]:
    parts: list[_MeshPart] = []

    for building_index, building in enumerate(_buildings(payload)):
        mesh = building.get("solid") or building.get("solid_mesh") or {}
        vertices = np.asarray(mesh.get("vertices") or [], dtype=np.float64).reshape((-1, 3))
        faces = _faces_from_payload(mesh)
        if not len(vertices) or not faces:
            continue
        kinds = tuple(mesh.get("face_kind") or ["wall"] * len(faces))
        building_id = str(
            building.get("building_id")
            or building.get("feature_id")
            or building.get("id")
            or f"building-{building_index}"
        )
        part_id = str(building.get("part_id") or "main")
        parts.append(
            _MeshPart(
                label=f"{building_id}/{part_id}",
                category="building",
                owner=building,
                vertices=vertices,
                faces=faces,
                face_kind=kinds,
            )
        )

    terrain = payload.get("terrain") or {}
    terrain_vertices = np.asarray(
        terrain.get("vertices") or [], dtype=np.float64
    ).reshape((-1, 3))
    terrain_faces = _faces_from_payload(terrain)
    if len(terrain_vertices) and terrain_faces:
        parts.append(
            _MeshPart(
                label="terrain",
                category="terrain",
                owner=terrain,
                vertices=terrain_vertices,
                faces=terrain_faces,
                face_kind=tuple(["terrain"] * len(terrain_faces)),
            )
        )

    solid_kinds = {
        "balcony",
        "canopy",
        "column",
        "pier",
        "beam",
        "gutter",
        "sign_post",
        "railing",
    }
    for feature_index, feature in enumerate(payload.get("facade_features") or []):
        if feature.get("kind") not in solid_kinds:
            continue
        vertices, faces = _feature_mesh(feature)
        if not len(vertices) or not faces:
            continue
        parts.append(
            _MeshPart(
                label=f"detail-{feature_index}",
                category="detail",
                owner=feature,
                vertices=vertices,
                faces=faces,
                face_kind=tuple([str(feature.get("kind") or "detail")] * len(faces)),
            )
        )

    for patch_index, patch in enumerate(payload.get("vegetation") or []):
        vertices, faces = _vegetation_mesh(patch)
        if not len(vertices) or not faces:
            continue
        parts.append(
            _MeshPart(
                label=f"vegetation-{patch_index}",
                category="vegetation",
                owner=patch,
                vertices=vertices,
                faces=faces,
                face_kind=tuple(["vegetation"] * len(faces)),
            )
        )

    for furniture_index, item in enumerate(payload.get("furniture") or []):
        vertices, faces = _furniture_mesh(item)
        if not len(vertices) or not faces:
            continue
        parts.append(
            _MeshPart(
                label=f"furniture-{furniture_index}",
                category="furniture",
                owner=item,
                vertices=vertices,
                faces=faces,
                face_kind=tuple(["furniture"] * len(faces)),
            )
        )

    return parts


def _triangulate(face: Iterable[int]) -> Iterable[tuple[int, int, int]]:
    face = tuple(int(index) for index in face)
    for index in range(1, len(face) - 1):
        yield (face[0], face[index], face[index + 1])


def _surface_type(kind: str, category: str) -> SurfaceType:
    if kind == "wall":
        return SurfaceType.FACADE
    if kind in {"roof", "roof_step"}:
        return SurfaceType.ROOF
    if kind in {"base", "terrain"} or category == "terrain":
        return SurfaceType.TERRAIN
    if category in {"detail", "furniture"}:
        return SurfaceType.ARCHITECTURAL_DETAIL
    return SurfaceType.UNKNOWN


def triangle_provenance(payload: dict) -> list[dict]:
    records: list[dict] = []
    for part in _mesh_parts(payload):
        for face_index, face in enumerate(part.faces):
            kind = (
                part.face_kind[face_index]
                if face_index < len(part.face_kind)
                else "unknown"
            )
            surface_type = _surface_type(kind, part.category)
            for _ in _triangulate(face):
                record = _evidence(
                    part.owner,
                    triangle_id=len(records),
                    surface_id=f"{part.label}/{kind}/{face_index}",
                    surface_type=surface_type,
                )
                record["category"] = part.category
                records.append(record)
    return records


def _canonical_scene_arrays(
    payload: dict,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    categories: list[str] = []
    for part in _mesh_parts(payload):
        offset = len(vertices)
        vertices.extend(part.vertices.tolist())
        for face in part.faces:
            for a, b, c in _triangulate(face):
                triangles.append([offset + a, offset + b, offset + c])
                categories.append(part.category)
    return (
        np.asarray(vertices, dtype=np.float64).reshape((-1, 3)),
        np.asarray(triangles, dtype=np.uint32).reshape((-1, 3)),
        tuple(categories),
    )


def canonical_mesh_arrays(payload: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return all geometry that physically occupies the canonical scene."""
    vertices, triangles, _categories = _canonical_scene_arrays(payload)
    return vertices, triangles


def mesh_digest(vertices: np.ndarray, triangles: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(vertices, np.float64).tobytes())
    digest.update(np.ascontiguousarray(triangles, np.uint32).tobytes())
    return digest.hexdigest()


def _category_counts(categories: tuple[str, ...]) -> dict[str, int]:
    return {
        category: sum(1 for value in categories if value == category)
        for category in sorted(set(categories))
    }


def export_canonical_gltf(payload: dict, path: Path) -> dict:
    """Write an embedded-buffer glTF for the complete canonical reality scene."""
    world_vertices, triangles, categories = _canonical_scene_arrays(payload)
    if not len(world_vertices) or not len(triangles):
        raise ValueError("canonical mesh is empty")

    render_origin = world_vertices.astype(np.float64).mean(axis=0)
    vertices = (world_vertices.astype(np.float64) - render_origin).astype(np.float32)
    position_bytes = vertices.tobytes()

    normals = np.zeros_like(vertices)
    for triangle in triangles:
        a, b, c = vertices[triangle]
        normal = np.cross(b - a, c - a)
        normals[triangle] += normal
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], 1e-12)

    # TEXCOORD_0 remains a coarse fallback until per-surface photographic atlas
    # bindings are emitted.  The metadata says so explicitly so this channel is
    # never mistaken for a production orthofacade UV chart.
    span = np.maximum(
        vertices[:, :2].max(axis=0) - vertices[:, :2].min(axis=0), 1e-12
    )
    uvs = (
        (vertices[:, :2] - vertices[:, :2].min(axis=0)) / span
    ).astype(np.float32)

    normal_bytes = normals.astype(np.float32).tobytes()
    uv_bytes = uvs.tobytes()
    index_bytes = triangles.reshape(-1).tobytes()

    normal_offset = len(position_bytes)
    uv_offset = normal_offset + len(normal_bytes)
    index_offset = uv_offset + len(uv_bytes)
    padding = (-index_offset) % 4
    index_offset += padding
    blob = (
        position_bytes
        + normal_bytes
        + uv_bytes
        + b"\0" * padding
        + index_bytes
    )

    provenance = triangle_provenance(payload)
    if len(provenance) != len(triangles):
        raise ValueError(
            "triangle provenance count does not match canonical triangle count"
        )

    metadata = {
        "vertex_count": int(len(vertices)),
        "triangle_count": int(len(triangles)),
        "mesh_digest": mesh_digest(world_vertices, triangles),
        "source": "CanonicalScene",
        "reextruded": False,
        "building_count": len(_buildings(payload)),
        "vegetation_patch_count": len(payload.get("vegetation") or []),
        "furniture_count": len(payload.get("furniture") or []),
        "triangle_count_by_category": _category_counts(categories),
        "opening_count": len(payload.get("openings", [])),
        "triangle_provenance": provenance,
        "vertical_reference": payload.get("vertical_reference")
        or payload.get("spatial_reference", {}).get("vertical"),
        "floating_origin_world": render_origin.tolist(),
        "render_coordinates": (
            "float32 local; add floating_origin_world for canonical world coordinates"
        ),
        "uv_contract": (
            "fallback_global_xy_only; production photographic atlases require "
            "surface-local UV charts"
        ),
    }
    gltf = {
        "asset": {
            "version": "2.0",
            "generator": "BuildingDroneVisit canonical exporter",
        },
        "buffers": [
            {
                "byteLength": len(blob),
                "uri": "data:application/octet-stream;base64,"
                + base64.b64encode(blob).decode("ascii"),
            }
        ],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": len(position_bytes),
                "target": 34962,
            },
            {
                "buffer": 0,
                "byteOffset": normal_offset,
                "byteLength": len(normal_bytes),
                "target": 34962,
            },
            {
                "buffer": 0,
                "byteOffset": uv_offset,
                "byteLength": len(uv_bytes),
                "target": 34962,
            },
            {
                "buffer": 0,
                "byteOffset": index_offset,
                "byteLength": len(index_bytes),
                "target": 34963,
            },
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(vertices),
                "type": "VEC3",
                "min": vertices.min(axis=0).tolist(),
                "max": vertices.max(axis=0).tolist(),
            },
            {
                "bufferView": 1,
                "componentType": 5126,
                "count": len(vertices),
                "type": "VEC3",
            },
            {
                "bufferView": 2,
                "componentType": 5126,
                "count": len(vertices),
                "type": "VEC2",
            },
            {
                "bufferView": 3,
                "componentType": 5125,
                "count": triangles.size,
                "type": "SCALAR",
            },
        ],
        "materials": [
            {
                "name": "canonical-neutral",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.72, 0.72, 0.72, 1.0],
                    "roughnessFactor": 0.9,
                },
            }
        ],
        "meshes": [
            {
                "name": "CanonicalSceneMesh",
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0,
                            "NORMAL": 1,
                            "TEXCOORD_0": 2,
                        },
                        "indices": 3,
                        "material": 0,
                    }
                ],
                "extras": metadata,
            }
        ],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
        "extras": metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gltf, separators=(",", ":")), "utf-8")
    return metadata


__all__ = [
    "canonical_mesh_arrays",
    "export_canonical_gltf",
    "mesh_digest",
    "triangle_provenance",
]
