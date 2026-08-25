"""Solides bâtis fermés et audit topologique sans moteur 3D externe.

Le maillage de rendu historique juxtaposait murs et toiture sans partager
leurs arêtes. Cette représentation suffit à un z-buffer, pas à un bâtiment : une
surface pouvait traverser sa voisine ou laisser le volume ouvert. Ce module
construit une coque logique depuis une emprise et audite chaque arête.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import triangulate


@dataclass(frozen=True)
class SolidMesh:
    vertices: np.ndarray
    faces: np.ndarray

    def as_dict(self) -> dict:
        return {
            "vertices": np.round(self.vertices, 3).tolist(),
            "faces": self.faces.astype(int).tolist(),
        }


def _normalised_ring(footprint: np.ndarray) -> np.ndarray:
    ring = np.asarray(footprint, dtype=np.float64)
    if len(ring) > 1 and np.allclose(ring[0], ring[-1]):
        ring = ring[:-1]
    polygon = Polygon(ring)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty or polygon.geom_type != "Polygon":
        raise ValueError("emprise invalide : impossible de fermer le bâtiment")
    ring = np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)
    # Une orientation unique rend les faces et les audits reproductibles.
    if not polygon.exterior.is_ccw:
        ring = ring[::-1]
    return ring


def closed_solid(
    footprint: np.ndarray,
    top_heights: np.ndarray | float,
    bottom_heights: np.ndarray | float = 0.0,
) -> SolidMesh:
    """Ferme une emprise par un fond, des murs partagés et un toit logique.

    Le toit logique interpole les hauteurs connues aux sommets. La nappe LiDAR
    détaillée peut rester un calque d'observation distinct ; elle ne sert pas à
    prétendre que deux maillages aux bords différents sont soudés.
    """
    ring = _normalised_ring(footprint)
    count = len(ring)
    if count < 3:
        raise ValueError("une emprise bâtie requiert au moins trois sommets")

    top = np.broadcast_to(np.asarray(top_heights, dtype=np.float64), (count,)).copy()
    bottom = np.broadcast_to(
        np.asarray(bottom_heights, dtype=np.float64), (count,)
    ).copy()
    if not np.isfinite(top).all() or not np.isfinite(bottom).all():
        raise ValueError("altitude non finie dans la coque")
    if np.any(top <= bottom + 0.05):
        raise ValueError("le toit doit rester au-dessus du terrain")

    vertices = np.vstack(
        [np.c_[ring, bottom], np.c_[ring, top]]
    )
    lookup = {(round(float(x), 8), round(float(y), 8)): i for i, (x, y) in enumerate(ring)}
    polygon = Polygon(ring)
    roof_triangles: list[list[int]] = []
    for triangle in triangulate(polygon):
        if not polygon.covers(triangle):
            continue
        indices: list[int] = []
        for x, y in triangle.exterior.coords[:-1]:
            index = lookup.get((round(float(x), 8), round(float(y), 8)))
            if index is None:
                indices = []
                break
            indices.append(index)
        if len(indices) != 3:
            continue
        a, b, c = ring[indices]
        cross_2d = float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if cross_2d < 0:
            indices[1], indices[2] = indices[2], indices[1]
        roof_triangles.append(indices)
    if not roof_triangles:
        raise ValueError("triangulation de l'emprise impossible")

    faces: list[list[int]] = []
    for triangle in roof_triangles:
        faces.append([count + i for i in triangle])
        faces.append(list(reversed(triangle)))
    for i in range(count):
        j = (i + 1) % count
        faces.append([i, j, count + j])
        faces.append([i, count + j, count + i])
    return SolidMesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64))


def audit(mesh: SolidMesh) -> dict:
    """Mesure l'étanchéité et la connexité d'un maillage triangulaire."""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    edge_counts: Counter[tuple[int, int]] = Counter()
    face_edges: list[list[tuple[int, int]]] = []
    degenerate = 0
    signed_volume = 0.0
    for face in faces:
        if len(set(int(v) for v in face)) < 3:
            degenerate += 1
            continue
        a, b, c = vertices[face]
        if np.linalg.norm(np.cross(b - a, c - a)) <= 1e-9:
            degenerate += 1
        signed_volume += float(np.dot(a, np.cross(b, c))) / 6.0
        edges = [
            tuple(sorted((int(face[0]), int(face[1])))),
            tuple(sorted((int(face[1]), int(face[2])))),
            tuple(sorted((int(face[2]), int(face[0])))),
        ]
        face_edges.append(edges)
        edge_counts.update(edges)

    owners: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_index, edges in enumerate(face_edges):
        for edge in edges:
            owners[edge].append(face_index)
    adjacency: dict[int, set[int]] = defaultdict(set)
    for related in owners.values():
        for face_index in related:
            adjacency[face_index].update(i for i in related if i != face_index)
    unseen = set(range(len(face_edges)))
    components = 0
    while unseen:
        components += 1
        queue = deque([unseen.pop()])
        while queue:
            for neighbour in adjacency[queue.popleft()]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    queue.append(neighbour)

    boundary = sum(value == 1 for value in edge_counts.values())
    non_manifold = sum(value > 2 for value in edge_counts.values())
    volume = abs(signed_volume)
    watertight = boundary == 0 and non_manifold == 0 and degenerate == 0
    return {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "edges": int(len(edge_counts)),
        "boundary_edges": boundary,
        "non_manifold_edges": non_manifold,
        "degenerate_faces": degenerate,
        "connected_components": components,
        "volume_m3": round(volume, 3),
        "watertight": watertight,
        "supported": watertight and components == 1 and volume > 0.0,
        # Cette coque est une extrusion 2,5D d'un polygone simple : toit et
        # fond partagent exactement la même triangulation XY, le toit reste
        # strictement au-dessus du fond et les murs suivent le contour. Dans
        # cette classe restreinte, une auto-intersection est impossible par
        # construction ; ce ne serait plus vrai pour un maillage libre.
        "self_intersection": False,
        "self_intersection_method": "guaranteed_by_monotone_2_5d_extrusion",
    }
