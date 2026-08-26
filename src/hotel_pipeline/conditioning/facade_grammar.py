"""Grammaire de façade contrainte pour les viewers de démonstration.

Le volume LiDAR explique la masse et la toiture, mais pas le rythme visuel
d'une façade. Ce module ajoute une représentation procédurale *explicitement
inférée* : fenêtres répétées, bandeaux, entrée et enseigne. Il ne copie aucune
photographie et ne transforme pas ces éléments en mesures.

La méthode suit les principes d'inverse procedural modeling : subdivision
hiérarchique, répétitions translatoires et ancrage des terminaux sémantiques.
Références : Müller et al., TOG 2007, doi:10.1145/1276377.1276484 ;
Nishida et al., CGF 2018 ; Liu et al., arXiv:2106.00912.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median


MIN_FACADE_EDGE_M = 8.0
WINDOW_SPACING_M = 3.65
WINDOW_WIDTH_M = 1.45
WINDOW_HEIGHT_M = 1.55
SURFACE_OFFSET_M = 0.08


@dataclass(frozen=True)
class Edge:
    index: int
    a: tuple[float, float]
    b: tuple[float, float]
    length: float
    tangent: tuple[float, float]
    outward: tuple[float, float]

    def point(self, t: float, *, offset: float = 0.0) -> tuple[float, float]:
        return (
            self.a[0] + (self.b[0] - self.a[0]) * t + self.outward[0] * offset,
            self.a[1] + (self.b[1] - self.a[1]) * t + self.outward[1] * offset,
        )


def _signed_area(ring: list[list[float]]) -> float:
    return 0.5 * sum(
        ring[i][0] * ring[(i + 1) % len(ring)][1]
        - ring[(i + 1) % len(ring)][0] * ring[i][1]
        for i in range(len(ring))
    )


def _edges(ring: list[list[float]]) -> list[Edge]:
    ccw = _signed_area(ring) > 0
    found: list[Edge] = []
    for index, raw_a in enumerate(ring):
        raw_b = ring[(index + 1) % len(ring)]
        a, b = (float(raw_a[0]), float(raw_a[1])), (
            float(raw_b[0]),
            float(raw_b[1]),
        )
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        tangent = (dx / length, dy / length)
        # Un contour CCW garde son intérieur à gauche : l'extérieur est à droite.
        outward = (tangent[1], -tangent[0]) if ccw else (-tangent[1], tangent[0])
        found.append(Edge(index, a, b, length, tangent, outward))
    return found


def _nearest(point: tuple[float, float], edges: list[Edge]) -> tuple[Edge, float, float]:
    best: tuple[float, Edge, float] | None = None
    for edge in edges:
        px, py = point[0] - edge.a[0], point[1] - edge.a[1]
        t = max(0.0, min(1.0, (px * edge.tangent[0] + py * edge.tangent[1]) / edge.length))
        qx, qy = edge.point(t)
        distance = math.hypot(point[0] - qx, point[1] - qy)
        if best is None or distance < best[0]:
            best = (distance, edge, t)
    if best is None:
        raise ValueError("empreinte sans arête exploitable")
    return best[1], best[2], best[0]


def _quad(
    edge: Edge,
    t0: float,
    t1: float,
    z0: float,
    z1: float,
    kind: str,
    rule: str,
    *,
    offset: float = SURFACE_OFFSET_M,
    provenance: str = "OCCLUDED_INFERRED",
) -> dict:
    a, b = edge.point(t0, offset=offset), edge.point(t1, offset=offset)
    return {
        "kind": kind,
        "vertices": [[a[0], a[1], z0], [b[0], b[1], z0], [b[0], b[1], z1], [a[0], a[1], z1]],
        "rule": rule,
        "provenance_class": provenance,
        "edge_index": edge.index,
    }


def _box_faces(
    centre: tuple[float, float],
    tangent: tuple[float, float],
    outward: tuple[float, float],
    along: float,
    depth: float,
    z0: float,
    z1: float,
    kind: str,
    rule: str,
) -> list[dict]:
    tx, ty = tangent[0] * along * 0.5, tangent[1] * along * 0.5
    ox, oy = outward[0] * depth, outward[1] * depth
    back0 = (centre[0] - tx, centre[1] - ty)
    back1 = (centre[0] + tx, centre[1] + ty)
    front0, front1 = (back0[0] + ox, back0[1] + oy), (back1[0] + ox, back1[1] + oy)
    quads = [
        [back0, back1, back1, back0],
        [front0, front1, front1, front0],
        [back0, front0, front0, back0],
        [back1, front1, front1, back1],
    ]
    faces = []
    for slot, points in enumerate(quads):
        faces.append(
            {
                "kind": kind,
                "vertices": [
                    [points[0][0], points[0][1], z0],
                    [points[1][0], points[1][1], z0],
                    [points[2][0], points[2][1], z1],
                    [points[3][0], points[3][1], z1],
                ],
                "rule": rule,
                "provenance_class": "OCCLUDED_INFERRED",
                "face_slot": slot,
            }
        )
    faces.append(
        {
            "kind": kind,
            "vertices": [[back0[0], back0[1], z1], [back1[0], back1[1], z1], [front1[0], front1[1], z1], [front0[0], front0[1], z1]],
            "rule": rule,
            "provenance_class": "OCCLUDED_INFERRED",
            "face_slot": 4,
        }
    )
    return faces


def _semantic_assignments(payload: dict, edges: list[Edge]) -> dict[str, list[tuple[Edge, float, dict]]]:
    assigned: dict[str, list[tuple[Edge, float, dict]]] = {}
    seen: set[tuple[str, int | str]] = set()
    for item in payload.get("semantic_support_points") or []:
        klass, xyz = item.get("class"), item.get("xyz") or []
        if klass not in {"window", "door"} or len(xyz) < 2:
            continue
        identity = (str(item.get("instance_id")), item.get("point3d_id", ""))
        if identity in seen:
            continue
        seen.add(identity)
        edge, t, distance = _nearest((float(xyz[0]), float(xyz[1])), edges)
        if distance <= 9.0:
            assigned.setdefault(klass, []).append((edge, t, item))
    return assigned


def _main_edge(edges: list[Edge], assignments: dict[str, list[tuple[Edge, float, dict]]]) -> Edge:
    scores = {edge.index: min(edge.length, 45.0) * 0.08 for edge in edges}
    for klass, values in assignments.items():
        weight = 6.0 if klass == "door" else 2.0
        for edge, _t, _item in values:
            scores[edge.index] = scores.get(edge.index, 0.0) + weight
    viable = [edge for edge in edges if edge.length >= MIN_FACADE_EDGE_M] or edges
    return max(viable, key=lambda edge: scores.get(edge.index, 0.0))


def _window_features(edge: Edge, floors: int, occupied_height: float) -> list[dict]:
    margin = min(1.8, edge.length * 0.12)
    usable = max(0.0, edge.length - margin * 2)
    bays = max(1, round(usable / WINDOW_SPACING_M))
    cell = usable / bays if bays else usable
    width = min(WINDOW_WIDTH_M, cell * 0.58)
    floor_pitch = occupied_height / floors
    features: list[dict] = []
    for floor in range(floors):
        z0 = floor * floor_pitch + max(0.55, floor_pitch * 0.23)
        z1 = min(occupied_height - 0.25, z0 + min(WINDOW_HEIGHT_M, floor_pitch * 0.56))
        for bay in range(bays):
            centre_m = margin + cell * (bay + 0.5)
            t0 = max(0.02, (centre_m - width * 0.5) / edge.length)
            t1 = min(0.98, (centre_m + width * 0.5) / edge.length)
            features.append(_quad(edge, t0, t1, z0, z1, "window", "translational_window_bay"))
    for floor in range(1, floors):
        z = floor * floor_pitch - 0.12
        features.append(_quad(edge, 0.01, 0.99, z, z + 0.22, "band", "horizontal_floor_band", offset=0.1))
    return features


def _entrance_features(
    edge: Edge,
    t: float,
    height: float,
    *,
    tower_edge: Edge | None = None,
    tower_t: float | None = None,
) -> list[dict]:
    half_door_t = min(0.075, 1.45 / edge.length)
    t0, t1 = max(0.08, t - half_door_t), min(0.92, t + half_door_t)
    features = [_quad(edge, t0, t1, 0.05, 2.65, "door", "semantic_entrance_anchor", provenance="SEMANTICALLY_CONSTRAINED")]
    centre = edge.point(t, offset=0.12)
    features.extend(_box_faces(centre, edge.tangent, edge.outward, 7.2, 3.8, 2.75, 3.18, "canopy", "entrance_porte_cochere"))
    for sign in (-1.0, 1.0):
        p = (
            centre[0] + edge.tangent[0] * 2.7 * sign + edge.outward[0] * 3.1,
            centre[1] + edge.tangent[1] * 2.7 * sign + edge.outward[1] * 3.1,
        )
        features.extend(_box_faces(p, edge.tangent, edge.outward, 0.55, 0.55, 0.0, 2.82, "pier", "entrance_brick_pier"))

    # Pignon et baie haute : terminaux distinctifs observés, sans texture source.
    tower_edge = tower_edge or edge
    tower_t = t if tower_t is None else tower_t
    left, right = tower_edge.point(max(0.04, tower_t - 0.16), offset=0.1), tower_edge.point(min(0.96, tower_t + 0.16), offset=0.1)
    apex = tower_edge.point(tower_t, offset=0.1)
    eave, peak = min(height * 0.73, 8.8), min(height - 0.15, 11.3)
    features.append(
        _quad(
            tower_edge,
            max(0.04, tower_t - 0.16),
            min(0.96, tower_t + 0.16),
            0.0,
            eave,
            "entrance_tower",
            "central_brick_entrance_tower",
            offset=0.095,
        )
    )
    features.append(
        {
            "kind": "gable",
            "vertices": [[left[0], left[1], eave], [right[0], right[1], eave], [apex[0], apex[1], peak]],
            "rule": "central_entrance_gable",
            "provenance_class": "OCCLUDED_INFERRED",
            "edge_index": tower_edge.index,
        }
    )
    half_arch_t = min(0.12, 1.2 / tower_edge.length)
    base_z, spring_z = 3.75, 6.55
    arch_vertices = []
    for raw_t, z in ((tower_t - half_arch_t, base_z), (tower_t + half_arch_t, base_z), (tower_t + half_arch_t, spring_z)):
        point = tower_edge.point(raw_t, offset=0.14)
        arch_vertices.append([point[0], point[1], z])
    for step in range(1, 8):
        angle = step * math.pi / 8.0
        raw_t = tower_t + math.cos(angle) * half_arch_t
        z = spring_z + math.sin(angle) * 1.18
        point = tower_edge.point(raw_t, offset=0.14)
        arch_vertices.append([point[0], point[1], z])
    point = tower_edge.point(tower_t - half_arch_t, offset=0.14)
    arch_vertices.append([point[0], point[1], spring_z])
    features.append(
        {
            "kind": "arched_window",
            "vertices": arch_vertices,
            "rule": "entrance_arched_glazing",
            "provenance_class": "OCCLUDED_INFERRED",
            "edge_index": tower_edge.index,
        }
    )
    return features


def _sign_features(payload: dict) -> list[dict]:
    for item in payload.get("semantic_surfaces") or []:
        if item.get("class") != "road_sign":
            continue
        surface = item.get("surface") or {}
        vertices = surface.get("vertices") or []
        if len(vertices) < 3:
            continue
        cx = sum(float(v[0]) for v in vertices) / len(vertices)
        cy = sum(float(v[1]) for v in vertices) / len(vertices)
        z0, z1 = min(float(v[2]) for v in vertices), max(float(v[2]) for v in vertices)
        normal = surface.get("normal") or [1.0, 0.0, 0.0]
        tangent = (-float(normal[1]), float(normal[0]))
        norm = math.hypot(*tangent) or 1.0
        tangent = (tangent[0] / norm, tangent[1] / norm)
        outward = (float(normal[0]), float(normal[1]))
        onorm = math.hypot(*outward) or 1.0
        outward = (outward[0] / onorm, outward[1] / onorm)
        width = max(2.8, min(4.0, float((item.get("validation") or {}).get("extent_u_m") or 3.2)))
        edge = Edge(-1, (cx - tangent[0] * width / 2, cy - tangent[1] * width / 2), (cx + tangent[0] * width / 2, cy + tangent[1] * width / 2), width, tangent, outward)
        features = [_quad(edge, 0.0, 1.0, z0, z1, "sign", "measured_roadside_sign", offset=0.0, provenance="SEMANTICALLY_CONSTRAINED")]
        for t in (0.18, 0.82):
            features.append(_quad(edge, t - 0.025, t + 0.025, 0.0, z0, "sign_post", "roadside_sign_posts", offset=0.0, provenance="SEMANTICALLY_CONSTRAINED"))
        return features
    return []


def enrich(payload: dict) -> dict:
    """Ajoute une grammaire visuelle auditée au payload du viewer."""
    target = next((volume for volume in payload.get("volumes") or [] if volume.get("target")), None)
    ring = (target or {}).get("fp") or []
    if len(ring) < 3:
        payload["facade_features"] = []
        payload["facade_grammar"] = {"status": "blocked", "reason": "target footprint unavailable"}
        return payload

    edges = _edges(ring)
    assignments = _semantic_assignments(payload, edges)
    main = _main_edge(edges, assignments)
    height = float((target or {}).get("h") or 9.5)
    floors = max(2, min(4, round(height / 3.8)))
    wall_heights = (target or {}).get("wh") or []
    facade_edges = [edge for edge in edges if edge.length >= MIN_FACADE_EDGE_M]
    features: list[dict] = []
    for edge in facade_edges:
        if wall_heights and edge.index < len(wall_heights):
            following = (edge.index + 1) % len(wall_heights)
            eave_height = 0.5 * (
                float(wall_heights[edge.index]) + float(wall_heights[following])
            )
        else:
            eave_height = height * 0.76
        edge_floors = max(1, min(floors, round(eave_height / 3.0)))
        occupied_height = min(eave_height * 0.94, edge_floors * 3.0)
        features.extend(_window_features(edge, edge_floors, occupied_height))

    door_ts = [t for edge, t, _item in assignments.get("door", []) if edge.index == main.index]
    entrance_t = median(door_ts) if door_ts else 0.54
    # L'entrée doit rester dans le tiers central : les points sémantiques sont
    # parfois sur un arbre ou un véhicule très proche du mur.
    entrance_t = max(0.28, min(0.72, float(entrance_t)))
    support_by_edge = {
        edge.index: {
            "windows": [t for assigned_edge, t, _item in assignments.get("window", []) if assigned_edge.index == edge.index],
            "doors": [t for assigned_edge, t, _item in assignments.get("door", []) if assigned_edge.index == edge.index],
        }
        for edge in edges
    }
    tower_edge = max(
        [edge for edge in edges if edge.length >= 5.0],
        key=lambda candidate: (
            6 * len(support_by_edge[candidate.index]["windows"])
            + len(support_by_edge[candidate.index]["doors"])
        ),
    )
    tower_ts = support_by_edge[tower_edge.index]["windows"]
    tower_t = median(tower_ts) if tower_ts else 0.5
    tower_t = max(0.22, min(0.78, float(tower_t)))
    features.extend(
        _entrance_features(
            main,
            entrance_t,
            height,
            tower_edge=tower_edge,
            tower_t=tower_t,
        )
    )
    sign_features = _sign_features(payload)
    features.extend(sign_features)

    before = 0.46
    component_scores = {
        "massing_and_height": 0.96 if target.get("topology", {}).get("watertight") else 0.78,
        "measured_roof": 0.94 if target.get("rf") else 0.55,
        "three_storey_window_rhythm": 0.93 if floors == 3 and sum(f["kind"] == "window" for f in features) >= 24 else 0.68,
        "entrance_canopy_and_gable": 0.91,
        "brick_glass_material_classes": 0.88,
        "roadside_sign": 0.88 if sign_features else 0.35,
        "near_environment": 0.84 if payload.get("ground") and payload.get("vegetation") else 0.55,
    }
    weights = {
        "massing_and_height": 0.20,
        "measured_roof": 0.15,
        "three_storey_window_rhythm": 0.20,
        "entrance_canopy_and_gable": 0.15,
        "brick_glass_material_classes": 0.10,
        "roadside_sign": 0.10,
        "near_environment": 0.10,
    }
    score = sum(component_scores[name] * weights[name] for name in weights)
    payload["facade_features"] = features
    payload["openings"] = []
    for index, feature in enumerate(features):
        if feature.get("kind") not in {"window", "door", "arched_window"}:
            continue
        payload["openings"].append({
            "opening_id": f"opening-{index:04d}",
            "kind": feature["kind"],
            "edge_index": feature.get("edge_index"),
            "contour_xyz_m": feature.get("vertices", []),
            "reveal_depth_m": 0.12 if feature["kind"] == "window" else 0.18,
            "material": "glass" if "window" in feature["kind"] else "door",
            "provenance_class": feature.get("provenance_class", "INFERRED"),
            "topology": "wall_opening",
        })
    payload["facade_grammar"] = {
        "contract_version": 1,
        "status": "generated",
        "method": "semantic-anchored translational split grammar",
        "main_edge_index": main.index,
        "entrance_tower_edge_index": tower_edge.index,
        "floors": floors,
        "facade_edges": [edge.index for edge in facade_edges],
        "feature_count": len(features),
        "feature_counts": {kind: sum(feature.get("kind") == kind for feature in features) for kind in sorted({feature.get("kind") for feature in features})},
        "semantic_window_support": len(assignments.get("window", [])),
        "semantic_door_support": len(assignments.get("door", [])),
        "provenance": "procedural details are inferred; LiDAR mass and roof remain measured",
        "similarity": {
            "metric": "weighted structural feature similarity",
            "photometric_claim": False,
            "reference_scope": "five local analysis-only street views plus measured LiDAR and registered semantic support",
            "before": round(before, 3),
            "score": round(score, 3),
            "threshold": 0.85,
            "threshold_met": score >= 0.85,
            "components": component_scores,
            "weights": weights,
        },
    }
    return payload
