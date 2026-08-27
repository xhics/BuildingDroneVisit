"""Direct glTF export of canonical meshes; no secondary extrusion."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from .schemas.canonical_states import MeasurementState, SurfaceType


def _evidence(owner: dict, *, triangle_id: int, surface_id: str, surface_type: SurfaceType) -> dict:
    source_ids = list(owner.get("source_ids") or owner.get("source_refs") or [])
    requested = str(owner.get("state") or owner.get("measurement_state") or "UNKNOWN").upper()
    state = MeasurementState(requested) if requested in MeasurementState._value2member_map_ else MeasurementState.UNKNOWN
    if state is MeasurementState.MEASURED and not source_ids:
        state = MeasurementState.UNKNOWN
    return {
        "triangle_id": triangle_id, "surface_id": surface_id,
        "surface_type": surface_type.value, "state": state.value,
        "confidence": float(owner.get("confidence", 0.0)), "source_ids": source_ids,
        "source_dates": list(owner.get("source_dates") or []),
        "generation_method": owner.get("generation_method", "canonical_mesh_passthrough"),
        "residual": owner.get("residual"), "dependencies": owner.get("dependencies") or {},
    }


def triangle_provenance(payload: dict) -> list[dict]:
    records: list[dict] = []
    for vi, volume in enumerate(payload.get("volumes", [])):
        mesh = volume.get("solid") or volume.get("solid_mesh") or {}
        kinds = mesh.get("face_kind") or ["wall"] * len(mesh.get("faces") or [])
        surface_ids = mesh.get("surface_ids") or []
        triangle_ids = mesh.get("triangle_ids") or []
        states = mesh.get("measurement_states") or []
        confidences = mesh.get("confidence") or []
        provenance = mesh.get("provenance") or []
        building_id = str(volume.get("building_id") or volume.get("feature_id") or f"building-{vi}")
        part_id = str(volume.get("part_id") or "main")
        for face_index, face in enumerate(mesh.get("faces") or []):
            kind = kinds[face_index] if face_index < len(kinds) else "unknown"
            surface_type = {
                "wall": SurfaceType.FACADE,
                "roof": SurfaceType.ROOF,
                "roof_step": SurfaceType.ROOF,
                "base": SurfaceType.TERRAIN,
            }.get(kind, SurfaceType.UNKNOWN)
            owner = dict(volume)
            if face_index < len(provenance):
                owner.update(provenance[face_index])
            if face_index < len(states):
                owner["measurement_state"] = states[face_index]
            if face_index < len(confidences):
                owner["confidence"] = confidences[face_index]
            for _ in range(1, len(face)-1):
                records.append(_evidence(
                    owner,
                    triangle_id=(
                        int(triangle_ids[face_index])
                        if face_index < len(triangle_ids) else len(records)
                    ),
                    surface_id=(
                        str(surface_ids[face_index])
                        if face_index < len(surface_ids)
                        else f"{building_id}/{part_id}/unknown-surface"
                    ),
                    surface_type=surface_type,
                ))
    terrain = payload.get("terrain") or {}
    for face_index, face in enumerate(terrain.get("faces") or []):
        for _ in range(1, len(face)-1):
            records.append(_evidence(terrain, triangle_id=len(records), surface_id=f"terrain-{face_index}", surface_type=SurfaceType.TERRAIN))
    solid_kinds = {"balcony", "canopy", "column", "pier", "beam", "gutter", "sign_post"}
    for fi, feature in enumerate(payload.get("facade_features", [])):
        vertices = feature.get("vertices") or []
        if feature.get("kind") in solid_kinds:
            for _ in range(1, len(vertices)-1):
                records.append(_evidence(feature, triangle_id=len(records), surface_id=f"detail-{fi}", surface_type=SurfaceType.ARCHITECTURAL_DETAIL))
    return records


def canonical_mesh_arrays(payload: dict) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    for volume in payload.get("volumes", []):
        mesh = volume.get("solid") or volume.get("solid_mesh") or {}
        source_vertices = mesh.get("vertices") or []
        source_faces = mesh.get("faces") or []
        offset = len(vertices)
        vertices.extend(source_vertices)
        for face in source_faces:
            for i in range(1, len(face) - 1):
                triangles.append([offset + face[0], offset + face[i], offset + face[i + 1]])
    terrain = payload.get("terrain") or {}
    source_vertices = terrain.get("vertices") or []
    source_faces = terrain.get("faces") or []
    offset = len(vertices)
    vertices.extend(source_vertices)
    for face in source_faces:
        for i in range(1, len(face) - 1):
            triangles.append([offset + face[0], offset + face[i], offset + face[i + 1]])
    # Fine architectural solids are appended from their canonical faces. Flat
    # window/door annotations are deliberately excluded: they are openings,
    # not coplanar meshes layered over a wall.
    solid_kinds = {"balcony", "canopy", "column", "pier", "beam", "gutter", "sign_post"}
    for feature in payload.get("facade_features", []):
        if feature.get("kind") not in solid_kinds:
            continue
        feature_vertices = feature.get("vertices") or []
        if len(feature_vertices) < 3:
            continue
        offset = len(vertices)
        vertices.extend(feature_vertices)
        for i in range(1, len(feature_vertices) - 1):
            triangles.append([offset, offset + i, offset + i + 1])
    return np.asarray(vertices, np.float64), np.asarray(triangles, np.uint32)


def mesh_digest(vertices: np.ndarray, triangles: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(vertices, np.float64).tobytes())
    digest.update(np.ascontiguousarray(triangles, np.uint32).tobytes())
    return digest.hexdigest()


def export_canonical_gltf(payload: dict, path: Path) -> dict:
    """Write an embedded-buffer glTF whose counts/digest describe the source mesh."""
    world_vertices, triangles = canonical_mesh_arrays(payload)
    if not len(world_vertices) or not len(triangles):
        raise ValueError("canonical mesh is empty")
    render_origin = world_vertices.astype(np.float64).mean(axis=0)
    vertices = (world_vertices.astype(np.float64) - render_origin).astype(np.float32)
    position_bytes = vertices.tobytes()
    normals = np.zeros_like(vertices)
    for triangle in triangles:
        a, b, c = vertices[triangle]
        n = np.cross(b - a, c - a)
        normals[triangle] += n
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], 1e-12)
    span = np.maximum(vertices[:, :2].max(axis=0) - vertices[:, :2].min(axis=0), 1e-12)
    uvs = ((vertices[:, :2] - vertices[:, :2].min(axis=0)) / span).astype(np.float32)
    normal_bytes, uv_bytes = normals.astype(np.float32).tobytes(), uvs.tobytes()
    index_bytes = triangles.reshape(-1).tobytes()
    normal_offset = len(position_bytes)
    uv_offset = normal_offset + len(normal_bytes)
    index_offset = uv_offset + len(uv_bytes)
    padding = (-index_offset) % 4
    index_offset += padding
    blob = position_bytes + normal_bytes + uv_bytes + b"\0" * padding + index_bytes
    metadata = {
        "vertex_count": int(len(vertices)),
        "triangle_count": int(len(triangles)),
        "mesh_digest": mesh_digest(world_vertices, triangles),
        "source": "CanonicalSceneMesh",
        "reextruded": False,
        "opening_count": len(payload.get("openings", [])),
        "triangle_provenance": triangle_provenance(payload),
        "surface_catalog": {
            surface_id: surface
            for volume in payload.get("volumes", [])
            for surface_id, surface in (
                (volume.get("solid") or volume.get("solid_mesh") or {})
                .get("surface_catalog", {})
                .items()
            )
        },
        "vertical_reference": payload.get("vertical_reference") or payload.get("spatial_reference", {}).get("vertical"),
        "floating_origin_world": render_origin.tolist(),
        "render_coordinates": "float32 local; add floating_origin_world for canonical world coordinates",
    }
    gltf = {
        "asset": {"version": "2.0", "generator": "BuildingDroneVisit canonical exporter"},
        "buffers": [{"byteLength": len(blob), "uri": "data:application/octet-stream;base64," + base64.b64encode(blob).decode("ascii")}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(position_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": normal_offset, "byteLength": len(normal_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": uv_offset, "byteLength": len(uv_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": index_offset, "byteLength": len(index_bytes), "target": 34963},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(vertices), "type": "VEC3", "min": vertices.min(axis=0).tolist(), "max": vertices.max(axis=0).tolist()},
            {"bufferView": 1, "componentType": 5126, "count": len(vertices), "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": len(vertices), "type": "VEC2"},
            {"bufferView": 3, "componentType": 5125, "count": triangles.size, "type": "SCALAR"},
        ],
        "materials": [{"name": "canonical", "pbrMetallicRoughness": {"baseColorFactor": [0.72, 0.72, 0.72, 1.0], "roughnessFactor": 0.9}}],
        "meshes": [{"name": "CanonicalSceneMesh", "primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}, "indices": 3, "material": 0}], "extras": metadata}],
        "nodes": [{"mesh": 0}], "scenes": [{"nodes": [0]}], "scene": 0,
        "extras": metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gltf, separators=(",", ":")), "utf-8")
    return metadata


def export_canonical_mesh_gltf(mesh, path: Path) -> dict:  # noqa: ANN001
    """Production exporter: accepts CanonicalSceneMesh and records its digest."""
    from .reality_contract import require_canonical_mesh

    receipt = require_canonical_mesh(mesh, "gltf_export")
    payload = {"volumes": [{
        "solid": mesh.as_dict(),
        "source_ids": sorted({
            str(source)
            for row in mesh.provenance
            for source in row.get("source_ids", [])
        }),
    }]}
    metadata = export_canonical_gltf(payload, path)
    metadata["input_mesh_digest"] = receipt.input_mesh_digest
    metadata["mesh_digest"] = receipt.input_mesh_digest
    stored = json.loads(path.read_text("utf-8"))
    stored["extras"].update(metadata)
    stored["meshes"][0]["extras"].update(metadata)
    path.write_text(json.dumps(stored, separators=(",", ":")), "utf-8")
    return metadata


__all__ = [
    "canonical_mesh_arrays", "export_canonical_gltf", "export_canonical_mesh_gltf",
    "mesh_digest",
]
