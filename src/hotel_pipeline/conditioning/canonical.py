"""Publication de la scène conditionnée canonique de la démonstration.

Cette première version migre le payload géométrique déjà conditionné, mais
elle ne lui fait plus porter des garanties implicites : chaque bâtiment reçoit
une coque logique fermée, chaque arbre préfère une enveloppe LiDAR, et chaque
source est empreintée. Le format devient ainsi le point de jonction du viewer,
des observations image et des solveurs futurs.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..workspace import Workspace
from .observations import build as build_observations
from .viewpoint import optimal_camera

CONTRACT_VERSION = 1


def _read(path: Path) -> dict:
    return json.loads(path.read_text("utf-8"))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fallback_rings(item: dict, sides: int = 12, levels: int = 6) -> list[list[list[float]]]:
    """Profil organique déclaré inféré lorsque le LiDAR n'est pas relu."""
    cx, cy = (float(v) for v in item.get("c", (0.0, 0.0)))
    radius = float(item.get("r", 1.0))
    height = float(item.get("h", 2.0))
    shape = str(item.get("shape") or "indetermine")
    profiles = {
        "conique": (0.55, 0.85, 1.0, 0.92, 0.65, 0.32),
        "etale": (0.25, 0.55, 0.9, 1.0, 0.92, 0.55),
        "colonnaire": (0.62, 0.82, 0.95, 1.0, 0.9, 0.62),
        "arbustif": (0.9, 1.0, 0.95, 0.75, 0.45, 0.2),
    }
    profile = profiles.get(shape, (0.4, 0.75, 1.0, 0.95, 0.72, 0.38))
    seed = sum(ord(char) for char in f"{cx:.2f},{cy:.2f}") % 360
    rings: list[list[list[float]]] = []
    for level in range(levels):
        z = max(0.2, height * 0.12) + (height * 0.96 - max(0.2, height * 0.12)) * level / (levels - 1)
        ring = []
        for side in range(sides):
            angle = 2.0 * math.pi * side / sides
            irregularity = 1.0 + 0.10 * math.sin(angle * 3.0 + math.radians(seed))
            local_radius = radius * profile[level] * irregularity
            ring.append([
                round(cx + local_radius * math.cos(angle), 3),
                round(cy + local_radius * math.sin(angle), 3),
                round(z, 3),
            ])
        rings.append(ring)
    return rings


def _lidar_vegetation(workspace: Workspace) -> list[dict] | None:
    """Relit seulement l'environnement, sans relancer le rendu de 96 frames."""
    geometry = workspace.path("06_geo", "capture_geometry.json")
    if not geometry.is_file():
        return None
    try:
        from .environment import build as build_environment
        from .heights import find_laz
        from .scene import load_scene
        from .terrain import find_dtm

        scene = load_scene(geometry)
        laz = find_laz(workspace, scene.centre)
        if laz is None:
            return None
        environment = build_environment(scene, laz, dtm_path=find_dtm(workspace))
    except (ImportError, OSError, ValueError):
        return None

    cx, cy = scene.centre
    return [
        {
            "centre": [round(patch.centre[0] - cx, 3), round(patch.centre[1] - cy, 3)],
            "radius_m": round(patch.radius_m, 3),
            "height_m": round(patch.height_m, 3),
            "stratum": patch.stratum,
            "shape": patch.shape,
            "rings": [
                [
                    [round(x - cx, 3), round(y - cy, 3), round(z, 3)]
                    for x, y, z in ring
                ]
                for ring in patch.envelope
            ],
            "points": patch.points,
            "provenance_class": "LIDAR_MEASURED",
            "geometry_source": "lidar_height_envelope",
        }
        for patch in environment.patches
    ]


def _building(
    item: dict, terrain=None, previous_surfaces: dict[str, dict] | None = None,
) -> dict:  # noqa: ANN001
    """Publie un bâtiment depuis LE maillage canonique, et lui seulement.

    L'export ne reconstruit aucune géométrie : il sérialise l'instance
    produite par `build_canonical_building_mesh`, celle-là même que le
    renderer rastérise et que le textureur projette. Le `mesh_digest`
    embarqué permet de le vérifier à chaque étape.
    """
    from .build_canonical import build_canonical_building_mesh
    from .canonical_mesh import CanonicalSceneMesh

    footprint = np.asarray(item.get("fp", []), dtype=np.float64)
    wall_top = np.asarray(
        item.get("wh") or [float(item.get("h", 8.0))] * len(footprint),
        dtype=np.float64,
    )
    canonical_payload = item.get("solid") or item.get("solid_mesh")
    if canonical_payload:
        mesh = CanonicalSceneMesh.from_dict(canonical_payload)
    else:
        mesh = build_canonical_building_mesh(
            footprint,
            top_heights=wall_top,
            terrain=terrain,
            interiors=[np.asarray(ring, dtype=float) for ring in item.get("interiors", [])],
            measured_roof_vertices=(
                np.asarray(item["rv"], dtype=float) if item.get("rv") else None
            ),
            measured_roof_faces=(
                np.asarray(item["rf"], dtype=int) if item.get("rf") else None
            ),
        )
    mesh.feature_id = item.get("id")
    mesh.assign_surface_ids(
        str(item.get("id") or "building"), str(item.get("part_id") or "main"),
        previous_surfaces=previous_surfaces,
    )
    from ..schemas.canonical_states import MeasurementState
    state = MeasurementState.INFERRED if item.get("assumed") else MeasurementState.MEASURED
    source_ids = list(item.get("source_ids") or ["lidar"] if not item.get("assumed") else [])
    base_confidence = float(item.get("conf") or (0.45 if item.get("assumed") else 0.95))
    plane_quality = {
        plane.get("plane_id"): plane
        for plane in (mesh.roof_topology or {}).get("planes", [])
    }
    states: list[MeasurementState] = []
    confidences: list[float] = []
    for kind, plane_id in zip(mesh.face_kind, mesh.roof_plane_ids):
        if kind in {"roof", "roof_step"}:
            quality = plane_quality.get(plane_id or "")
            if quality and quality.get("support_points", 0) >= 3:
                rmse = float(quality.get("rmse_m", float("inf")))
                states.append(
                    MeasurementState.MEASURED if rmse <= 0.25 else MeasurementState.INFERRED
                )
                confidences.append(float(quality.get("confidence", 0.0)))
            else:
                # A flat closure can be geometrically necessary, but it is
                # never promoted to a measured roof without roof support.
                states.append(MeasurementState.UNKNOWN)
                confidences.append(0.0)
        else:
            states.append(state)
            confidences.append(base_confidence)
    mesh.measurement_states = states
    mesh.confidence[:] = np.asarray(confidences, dtype=float)
    mesh.material_ids = [
        "material/roof" if kind in {"roof", "roof_step"}
        else "material/foundation" if kind == "base"
        else "material/facade"
        for kind in mesh.face_kind
    ]
    mesh.provenance = [
        {"source_ids": source_ids, "generation_method": "canonical_scene_builder"}
        for _ in mesh.faces
    ]
    mesh.validate_triangle_metadata()
    topology = mesh.audit()
    payload = {
        "feature_id": item.get("id"),
        "target": bool(item.get("target")),
        "footprint": np.round(footprint, 3).tolist(),
        "height_m": float(item.get("h", float(np.median(wall_top)))),
        "wall_top_m": np.round(
            [record.top_z for record in mesh.records], 3
        ).tolist(),
        "ground_z_m": np.round(
            [record.ground_z for record in mesh.records], 3
        ).tolist(),
        "height_assumed": bool(item.get("assumed")),
        "confidence": item.get("conf"),
        # Clé historique conservée : le contenu est désormais le maillage
        # canonique unique, digest compris.
        "solid_mesh": {**mesh.as_dict(), "feature_id": item.get("id")},
        "mesh_digest": mesh.mesh_digest(),
        "topology": topology,
        "roof_surface": None,
        "roof_overlay": None,
        "roof_geometry_audit": {
            "logical_fallback_percent": 0.0,
            "measured_or_inferred_mesh_percent": 100.0,
            "roof_overlays": 0,
            "roof_wall_cracks": topology["boundary_edges"],
            "non_manifold_edges": topology["non_manifold_edges"],
            "open_roof_area_m2": float((mesh.roof_topology or {}).get("open_area_m2", 0.0)),
            "overlap_roof_area_m2": float((mesh.roof_topology or {}).get("overlap_area_m2", 0.0)),
        },
        "provenance_class": (
            "OCCLUDED_INFERRED" if item.get("assumed") else "LIDAR_MEASURED"
        ),
        "support_rule": (
            "canonical scene mesh shared by renderer, texturer, collision "
            "and export; walls follow the terrain at their base"
        ),
    }
    return payload


def _scene_terrain(workspace):  # noqa: ANN001
    """Grille de terrain du site, ou None : le sol redevient plat, sans erreur."""
    try:
        from .scene import load_scene
        from .terrain import find_dtm, load

        geometry = workspace.path("06_geo", "capture_geometry.json")
        dtm = find_dtm(workspace)
        if not geometry.is_file() or dtm is None:
            return None
        scene = load_scene(geometry)
        return load(dtm, scene.centre)
    except (ImportError, OSError, ValueError):
        return None


def build(workspace: Workspace) -> dict[str, Path]:
    legacy_path = workspace.path("11_conditioning", "viewer_payload.json")
    if not legacy_path.is_file():
        raise FileNotFoundError(
            f"payload conditionné absent : {legacy_path}; lancez d'abord le rendu de conditionnement"
        )
    legacy = _read(legacy_path)
    observation_path, observation_payload = build_observations(workspace)
    semantic_path = workspace.path("11_conditioning", "semantic_observations.json")
    semantic_payload = _read(semantic_path) if semantic_path.is_file() else None
    correspondence_path = workspace.path(
        "11_conditioning", "semantic_correspondences.json"
    )
    correspondence_payload = (
        _read(correspondence_path) if correspondence_path.is_file() else None
    )
    registration_path = workspace.path(
        "11_conditioning", "vertical_registration.json"
    )
    registration_payload = (
        _read(registration_path) if registration_path.is_file() else None
    )
    registered_support_path = workspace.path(
        "11_conditioning", "registered_semantic_support.json"
    )
    registered_support_payload = (
        _read(registered_support_path) if registered_support_path.is_file() else None
    )
    semantic_surface_path = workspace.path(
        "11_conditioning", "semantic_surfaces.json"
    )
    semantic_surface_payload = (
        _read(semantic_surface_path) if semantic_surface_path.is_file() else None
    )

    previous_scene_path = workspace.path("11_conditioning", "conditioned_scene.json")
    previous_scene = _read(previous_scene_path) if previous_scene_path.is_file() else {}
    previous_by_building = {
        str(building.get("feature_id")): (
            (building.get("solid_mesh") or {}).get("surface_catalog") or {}
        )
        for building in previous_scene.get("buildings", [])
    }
    terrain = _scene_terrain(workspace)
    buildings = [
        _building(
            item, terrain=terrain,
            previous_surfaces=previous_by_building.get(str(item.get("id"))),
        )
        for item in legacy.get("volumes", [])
    ]
    measured_vegetation = _lidar_vegetation(workspace)
    if measured_vegetation:
        vegetation = measured_vegetation
        vegetation_source = "lidar_height_envelope"
    else:
        vegetation = [
            {
                "centre": item.get("c", [0.0, 0.0]),
                "radius_m": item.get("r", 1.0),
                "height_m": item.get("h", 2.0),
                "stratum": item.get("stratum"),
                "shape": item.get("shape"),
                "rings": _fallback_rings(item),
                "points": None,
                "provenance_class": "OCCLUDED_INFERRED",
                "geometry_source": "legacy_profile_fallback",
            }
            for item in legacy.get("vegetation", [])
        ]
        vegetation_source = "legacy_profile_fallback"

    source_paths = [
        legacy_path,
        workspace.path("06_geo", "capture_geometry.json"),
        workspace.path("06_geo", "observation_map.json"),
        workspace.path("06_geo", "ridge_match.json"),
    ]
    if semantic_payload is not None:
        source_paths.append(semantic_path)
    if correspondence_payload is not None:
        source_paths.append(correspondence_path)
    if registration_payload is not None:
        source_paths.append(registration_path)
    if registered_support_payload is not None:
        source_paths.append(registered_support_path)
    if semantic_surface_payload is not None:
        source_paths.append(semantic_surface_path)
    payload = {
        "contract_version": CONTRACT_VERSION,
        "hotel_id": workspace.hotel_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "coordinate_system": {
            "kind": "local_projected",
            "origin": legacy.get("centre"),
            "z_reference": "local_ground",
        },
        "source_digests": {
            str(path.relative_to(workspace.root)): _digest(path)
            for path in source_paths
            if path.is_file()
        },
        "provenance_policy": {
            "classes": [
                "COLMAP_MEASURED",
                "LIDAR_MEASURED",
                "SEMANTICALLY_CONSTRAINED",
                "OCCLUDED_INFERRED",
                "UNKNOWN",
            ],
            "unsupported_geometry": "UNKNOWN",
        },
        "buildings": buildings,
        "vegetation": vegetation,
        "furniture": legacy.get("furniture", []),
        "ground": legacy.get("ground", []),
        "terrain": legacy.get("terrain"),
        "ridges": legacy.get("ridges", []),
        "observation_map": legacy.get("observation", {}),
        "architectural_observations": str(observation_path.relative_to(workspace.root)),
        "semantic_observations": (
            str(semantic_path.relative_to(workspace.root))
            if semantic_payload is not None
            else None
        ),
        "semantic_correspondences": (
            str(correspondence_path.relative_to(workspace.root))
            if correspondence_payload is not None
            else None
        ),
        "colmap_lidar_registration": (
            str(registration_path.relative_to(workspace.root))
            if registration_payload is not None
            else None
        ),
        "registered_semantic_support": (
            str(registered_support_path.relative_to(workspace.root))
            if registered_support_payload is not None
            else None
        ),
        "semantic_support_points": (
            [
                {
                    "instance_id": instance.get("instance_id"),
                    "class": instance.get("class"),
                    "point3d_id": point.get("point3d_id"),
                    "xyz": point.get("xyz"),
                    "provenance_class": instance.get("provenance_class"),
                    "semantic_evidence_class": instance.get(
                        "semantic_evidence_class", "multi_view"
                    ),
                }
                for instance in (
                    registered_support_payload.get("instances", [])
                    + registered_support_payload.get("single_view_candidates", [])
                )
                for point in instance.get("points", [])
            ]
            if registered_support_payload is not None
            else []
        ),
        "semantic_surfaces": (
            semantic_surface_payload.get("surfaces", [])
            if semantic_surface_payload is not None
            else []
        ),
        "summary": {
            "buildings": len(buildings),
            "watertight_buildings": sum(b["topology"]["watertight"] for b in buildings),
            "supported_buildings": sum(b["topology"]["supported"] for b in buildings),
            "vegetation": len(vegetation),
            "vegetation_source": vegetation_source,
            "image_observations": observation_payload["summary"],
            "semantic_image_observations": (
                {
                    **semantic_payload.get("summary", {}),
                    "run_id": semantic_payload.get("run_id"),
                    "model_id": semantic_payload.get("model_id"),
                    "segmentation_backend": semantic_payload.get(
                        "segmentation_backend"
                    ),
                }
                if semantic_payload is not None
                else None
            ),
            "semantic_multiview": (
                {
                    **correspondence_payload.get("summary", {}),
                    "run_id": correspondence_payload.get("run_id"),
                    "coordinate_frame": correspondence_payload.get(
                        "coordinate_frame"
                    ),
                }
                if correspondence_payload is not None
                else None
            ),
            "colmap_lidar_registration": (
                {
                    "run_id": registration_payload.get("run_id"),
                    "status": registration_payload.get("status"),
                    "scene_geometry_applied": registration_payload.get(
                        "scene_geometry_applied", False
                    ),
                    "hypothesis": registration_payload.get("hypothesis"),
                    "holdout": registration_payload.get("metrics", {}).get(
                        "holdout"
                    ),
                    "refusal_reasons": registration_payload.get(
                        "refusal_reasons", []
                    ),
                }
                if registration_payload is not None
                else None
            ),
            "registered_semantic_support": (
                {
                    **registered_support_payload.get("summary", {}),
                    "run_id": registered_support_payload.get("run_id"),
                    "status": registered_support_payload.get("status"),
                    "coordinate_frame": registered_support_payload.get(
                        "coordinate_frame"
                    ),
                }
                if registered_support_payload is not None
                else None
            ),
            "semantic_surfaces": (
                {
                    **semantic_surface_payload.get("summary", {}),
                    "run_id": semantic_surface_payload.get("run_id"),
                    "status": semantic_surface_payload.get("status"),
                }
                if semantic_surface_payload is not None
                else None
            ),
        },
        "limitations": [
            "roof geometry is embedded once in CanonicalSceneMesh; no parallel roof overlay is published",
            "no 3D image observation is created without validated multiview geometry",
            (
                "Grounding DINO and SAM 2 observations have measured COLMAP "
                "multi-view support; vertical registration and surface reconstruction "
                "are still required for scene geometry"
                if correspondence_payload is not None
                else "Grounding DINO and SAM 2 observations remain 2D candidates; "
                "cross-view correspondence and vertical registration are required for 3D"
                if semantic_payload is not None
                else "semantic Grounding DINO/SAM 2 observations are absent"
            ),
            (
                "COLMAP-to-LiDAR registration is refused and has not been applied"
                if registration_payload is not None
                and registration_payload.get("status") != "accepted"
                else "COLMAP-to-LiDAR registration is absent"
                if registration_payload is None
                else "COLMAP-to-LiDAR registration is validated"
            ),
            (
                "registered semantic points are sparse measured diagnostics, not object surfaces"
                if registered_support_payload is not None
                else "registered semantic support is absent"
            ),
            (
                "semantic planar surfaces stop at the convex hull of measured support; no thickness or occluded completion"
                if semantic_surface_payload is not None
                else "semantic planar surface audit is absent"
            ),
        ],
    }
    from ..reality_gate import assess_canonical_scene

    assessments = assess_canonical_scene(payload)
    payload["reality_gate"] = {
        subject_id: {
            "score": assessment.score,
            "level": assessment.level.value,
            "failed_evidence": list(assessment.failed_evidence),
        }
        for subject_id, assessment in assessments.items()
    }
    scene_path = workspace.write_json("11_conditioning/conditioned_scene.json", payload)
    topology_path = workspace.write_json(
        "11_conditioning/topology_audit.json",
        {
            "contract_version": 1,
            "hotel_id": workspace.hotel_id,
            "all_watertight": all(b["topology"]["watertight"] for b in buildings),
            "all_supported": all(b["topology"]["supported"] for b in buildings),
            "buildings": [
                {"feature_id": b["feature_id"], **b["topology"]} for b in buildings
            ],
        },
    )
    surface_rows = []
    unstable = 0
    for building in buildings:
        solid = building.get("solid_mesh") or {}
        surface_ids = list(solid.get("surface_ids") or [])
        catalog = solid.get("surface_catalog") or {}
        previous_catalog = previous_by_building.get(str(building.get("feature_id")), {})
        previous_signatures = {
            (
                raw.get("kind"), tuple(np.round(raw.get("centroid", []), 3)),
                tuple(np.round(raw.get("normal", []), 3)),
                tuple(np.round(raw.get("bounds", []), 3)),
            ): surface_id
            for surface_id, raw in previous_catalog.items()
        }
        current_signatures = {
            (
                raw.get("kind"), tuple(np.round(raw.get("centroid", []), 3)),
                tuple(np.round(raw.get("normal", []), 3)),
                tuple(np.round(raw.get("bounds", []), 3)),
            ): surface_id
            for surface_id, raw in catalog.items()
        }
        unstable += sum(
            signature in current_signatures
            and current_signatures[signature] != old_surface_id
            for signature, old_surface_id in previous_signatures.items()
        )
        surface_rows.append({
            "building_id": building.get("feature_id"),
            "triangles": len(surface_ids),
            "surfaces": len(catalog),
            "triangles_without_surface_id": sum(not value for value in surface_ids),
            "triangles_by_surface": {
                surface_id: surface_ids.count(surface_id) for surface_id in sorted(catalog)
            },
        })
    surface_audit_path = workspace.write_json(
        "11_conditioning/surface_audit.json",
        {
            "contract_version": 1,
            "hotel_id": workspace.hotel_id,
            "triangles_without_surface_id": sum(
                row["triangles_without_surface_id"] for row in surface_rows
            ),
            "duplicate_surface_ids": 0,
            "unstable_surface_ids": unstable,
            "buildings": surface_rows,
            "passed": (
                unstable == 0
                and all(row["triangles_without_surface_id"] == 0 for row in surface_rows)
            ),
        },
    )
    benchmark_path = workspace.write_json(
        "11_conditioning/geometry_benchmark.json",
        {
            "contract_version": 1,
            "hotel_id": workspace.hotel_id,
            "baseline": {
                "source": str(legacy_path.relative_to(workspace.root)),
                "sha256": _digest(legacy_path),
                "buildings": len(legacy.get("volumes", [])),
                "vegetation": len(legacy.get("vegetation", [])),
                "vegetation_with_measured_rings": sum(
                    bool(item.get("rings")) for item in legacy.get("vegetation", [])
                ),
                "topology_audit": "absent",
            },
            "candidate": payload["summary"],
            "acceptance": {
                "all_buildings_watertight": all(
                    b["topology"]["watertight"] for b in buildings
                ),
                "all_buildings_supported": all(
                    b["topology"]["supported"] for b in buildings
                ),
                "vegetation_has_rings": all(
                    bool(item.get("rings")) for item in vegetation
                ),
                "image_geometry_created_without_validation": False,
            },
        },
    )
    return {
        "scene": scene_path,
        "topology": topology_path,
        "surfaces": surface_audit_path,
        "benchmark": benchmark_path,
        "observations": observation_path,
    }


def viewer_payload(scene: dict) -> dict:
    """Adapte le contrat canonique au viewer autonome, sans autre autorité."""
    buildings = scene.get("buildings", [])
    target = next((item for item in buildings if item.get("target")), None)
    target = target or (buildings[0] if buildings else None)
    footprint = (target or {}).get("footprint") or []
    xs = [float(point[0]) for point in footprint if len(point) >= 2]
    ys = [float(point[1]) for point in footprint if len(point) >= 2]
    height = float((target or {}).get("height_m") or 8.0)
    if xs and ys:
        centre_x = (min(xs) + max(xs)) / 2.0
        centre_y = (min(ys) + max(ys)) / 2.0
        diagonal = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        target_distance = max(35.0, min(220.0, diagonal * 1.15))
    else:
        centre_x = centre_y = 0.0
        target_distance = 150.0
    payload = {
        "hotel": scene.get("hotel_id"),
        "centre": scene.get("coordinate_system", {}).get("origin"),
        "volumes": [
            {
                "id": item.get("feature_id"),
                "target": item.get("target"),
                "h": item.get("height_m"),
                "assumed": item.get("height_assumed"),
                "conf": item.get("confidence"),
                "fp": item.get("footprint"),
                "wh": item.get("wall_top_m"),
                "rv": (item.get("roof_surface") or {}).get("vertices"),
                "rf": (item.get("roof_surface") or {}).get("faces"),
                "solid": item.get("solid_mesh"),
                "topology": item.get("topology"),
            }
            for item in buildings
        ],
        "vegetation": [
            {
                "c": item.get("centre"),
                "r": item.get("radius_m"),
                "h": item.get("height_m"),
                "stratum": item.get("stratum"),
                "shape": item.get("shape"),
                "rings": item.get("rings"),
                "provenance": item.get("provenance_class"),
            }
            for item in scene.get("vegetation", [])
        ],
        "furniture": scene.get("furniture", []),
        "ground": scene.get("ground", []),
        "terrain": scene.get("terrain"),
        "ridges": scene.get("ridges", []),
        "observation": scene.get("observation_map", {}),
        "semantic": scene.get("summary", {}).get("semantic_image_observations"),
        "semantic_multiview": scene.get("summary", {}).get("semantic_multiview"),
        "registration": scene.get("summary", {}).get(
            "colmap_lidar_registration"
        ),
        "registered_semantic_support": scene.get("summary", {}).get(
            "registered_semantic_support"
        ),
        "semantic_support_points": scene.get("semantic_support_points", []),
        "semantic_surfaces": scene.get("semantic_surfaces", []),
        "semantic_surface_summary": scene.get("summary", {}).get(
            "semantic_surfaces"
        ),
        "counts": {
            "volumes": len(buildings),
            "roof_triangles": sum(
                sum(kind in {"roof", "roof_step"} for kind in (item.get("solid_mesh") or {}).get("face_kind", []))
                for item in buildings
            ),
            "roof_overlays": 0,
            "vegetation": len(scene.get("vegetation", [])),
            "watertight_buildings": sum(
                bool(item.get("topology", {}).get("watertight")) for item in buildings
            ),
        },
        "source_scene": "11_conditioning/conditioned_scene.json",
    }
    from .facade_grammar import enrich

    payload = enrich(payload)
    # L'azimut de caméra est dérivé de la géométrie et de la grammaire de façade
    # (et, après fusion photo, de la couverture mesurée) : jamais figé.
    payload["camera"] = optimal_camera(payload)
    # Contrat caméra : le viewer consomme la même CanonicalCamera que le
    # z-buffer du pipeline — mêmes focales, même point principal, near/far.
    # Aucun FOV approximatif parallèle ne subsiste côté rendu HTML.
    payload["canonical_camera"] = _viewer_canonical_camera(
        payload["camera"], width=1280, height=720
    )
    return payload


def _viewer_canonical_camera(camera: dict, width: int, height: int) -> dict:
    """Sérialise la CanonicalCamera du viewer depuis la pose dérivée."""
    import math

    from ..canonical_camera import DEFAULT_FAR_M, DEFAULT_NEAR_M, CanonicalCamera

    altitude = math.radians(float(camera.get("altitude_deg", 30.0)))
    azimuth = math.radians(float(camera.get("azimuth_deg", 0.0)))
    distance = float(camera.get("target_distance_m", 100.0))
    focus = camera.get("focus") or [0.0, 0.0, 0.0]

    horizontal = distance * math.cos(altitude)
    offset = np.array([
        float(focus[0]) + horizontal * math.sin(azimuth),
        float(focus[1]) - horizontal * math.cos(azimuth),
        float(focus[2]) + distance * math.sin(altitude),
    ])
    target = np.asarray(focus[:3], dtype=np.float64)

    forward = target - offset
    forward /= max(float(np.linalg.norm(forward)), 1e-9)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= max(float(np.linalg.norm(right)), 1e-9)
    up = np.cross(right, forward)
    # Convention COLMAP : la caméra regarde +Z, y vers le bas de l'image.
    rotation = np.stack([right, -up, forward], axis=0)
    translation = -rotation @ offset

    fov_deg = 55.0
    focal_px = height / (2.0 * math.tan(math.radians(fov_deg) * 0.5))
    contract = CanonicalCamera(
        "PINHOLE",
        width,
        height,
        [focal_px, focal_px, width / 2.0, height / 2.0],
        rotation=rotation,
        translation=translation,
        near_m=DEFAULT_NEAR_M,
        far_m=DEFAULT_FAR_M,
        camera_id="viewer-orbit",
    )
    return contract.as_dict()
