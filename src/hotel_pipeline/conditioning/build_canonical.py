"""Assemblage du maillage canonique d'un bâtiment.

Un seul chemin mène au maillage : emprise normalisée en sommets identifiés,
pieds posés sur le terrain mesuré, toiture reconstruite depuis ses plans
quand ils le permettent, murs dérivés de la frontière du toit — jamais
l'inverse. La soudure finale referme le volume ; l'audit compte les arêtes
libres et doit en trouver zéro.

Les cours intérieures font partie du bâtiment : les anneaux intérieurs de
l'emprise produisent leurs propres murs — traversables par le raycast comme
les autres — et ne sont jamais refermés par un couvercle. La triangulation
contrainte garantit qu'aucun triangle du fond ou du toit ne recouvre la cour.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from .canonical_mesh import (
    CanonicalSceneMesh,
    FootprintVertex,
    WELD_TOLERANCE_M,
    footprint_records,
)
from .constrained_triangulation import triangulate_constrained
from .roof_planes import RoofDecomposition
from .roof_reconstruct import ReconstructedRoof, derive_wall_tops, reconstruct_roof

#: Hauteur minimale de mur : un toit reste strictement au-dessus du sol.
MIN_WALL_M = 0.5


def _terrain_height(terrain, x: float, y: float) -> float:  # noqa: ANN001
    """Altitude du terrain extérieur en un point, ou zéro sans relevé."""
    if terrain is None:
        return 0.0
    value = float(terrain.height_at(x, y))
    return value if np.isfinite(value) else 0.0


def _ring_records(ring: np.ndarray, terrain=None, top_z: float = 0.0):  # noqa: ANN001
    """Records d'un anneau (extérieur ou trou), orienté CCW."""
    records = footprint_records(ring, top_heights=top_z)
    if terrain is not None:
        for record in records:
            record.ground_z = _terrain_height(terrain, record.x, record.y)
    return records


def _cap_faces(
    polygon: Polygon,
    nodes: list[tuple[float, float]],
    flip: bool = False,
    index_offset: int = 0,
) -> list[list[int]]:
    """Triangles d'un couvercle par triangulation contrainte.

    ``polygon`` porte trous éventuels : aucune face n'est émise dans une cour.
    Les triangles sont indexés dans ``nodes``, décalés de ``index_offset``
    pour pointer sur la nappe du couvercle et non sur celle du fond. ``flip``
    retourne le sens : normale vers le bas pour le fond, vers le ciel pour
    le toit.
    """
    lookup = {
        (round(float(x), 6), round(float(y), 6)): index
        for index, (x, y) in enumerate(nodes)
    }
    faces: list[list[int]] = []
    for triangle in triangulate_constrained(polygon):
        node_ids = []
        for x, y in triangle:
            index = lookup.get((round(float(x), 6), round(float(y), 6)))
            if index is None:
                node_ids = []
                break
            node_ids.append(index)
        if len(node_ids) != 3 or len(set(node_ids)) != 3:
            continue
        indices = [i + index_offset for i in node_ids]
        a, b, c = (nodes[i] for i in node_ids)
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        # Fond (flip=True) : normale vers le bas ; couvercle (flip=False) :
        # normale vers le ciel.
        if (cross < 0) != bool(flip):
            indices.reverse()
        faces.append(list(indices))
    return faces


def _wall_strip(
    bottom_lookup: dict[tuple[float, float], int],
    top_lookup: dict[tuple[float, float], int],
    ring: list[FootprintVertex],
) -> list[list[int]]:
    """Quads verticaux le long d'un anneau, du pied au haut."""
    faces: list[list[int]] = []
    count = len(ring)
    for i in range(count):
        j = (i + 1) % count
        bi = bottom_lookup[(round(float(ring[i].x), 6), round(float(ring[i].y), 6))]
        bj = bottom_lookup[(round(float(ring[j].x), 6), round(float(ring[j].y), 6))]
        ti = top_lookup[(round(float(ring[i].x), 6), round(float(ring[i].y), 6))]
        tj = top_lookup[(round(float(ring[j].x), 6), round(float(ring[j].y), 6))]
        # Le pied des murs suit le terrain : ground_z par sommet, pas un
        # plan plat qui flotterait sur un site incliné.
        faces.extend([[bi, bj, tj], [bi, tj, ti]])
    return faces


def build_canonical_building_mesh(
    footprint: np.ndarray,
    top_heights: np.ndarray | float | None = None,
    terrain=None,  # noqa: ANN001 - TerrainGrid ou équivalent
    roof_decomposition: RoofDecomposition | None = None,
    weld_tolerance_m: float = WELD_TOLERANCE_M,
    interiors: list[np.ndarray] | None = None,
) -> CanonicalSceneMesh:
    """Construit LE maillage du bâtiment — celui que tout le monde consomme.

    Paramètres
    ----------
    footprint :
        Contour extérieur de l'emprise, dans n'importe quel sens CW/CCW.
    interiors :
        Anneaux intérieurs (cours, patios). Leur vide est conservé : murs
        intérieurs générés, aucun couvercle dessus, raycast inclus.
    top_heights :
        Hauteurs de toit par sommet, altitude de repli quand aucun pan de
        toiture reconstruit ne revendique le sommet.
    terrain :
        Grille de terrain (``height_at(x, y)``) pour poser les pieds de murs.
    roof_decomposition :
        Segmentation en plans ; présente, elle commande la reconstruction.
    """
    from ..logging import get_logger

    log = get_logger("canonical-building")

    records = footprint_records(footprint, top_heights=top_heights)
    for record in records:
        record.ground_z = _terrain_height(terrain, record.x, record.y)
    grounds = np.array([record.ground_z for record in records], dtype=np.float64)
    ring = np.array([[record.x, record.y] for record in records], dtype=np.float64)

    # Cours intérieures : chaque anneau devient un ruban de murs, jamais un
    # couvercle. L'orientation normalisée CCW garde les hauteurs liées à
    # leur sommet, où qu'ait été l'anneau fourni.
    hole_records: list[list[FootprintVertex]] = []
    for interior in interiors or []:
        try:
            hole_records.append(_ring_records(interior, terrain))
        except ValueError:
            log.info("anneau intérieur dégénéré ignoré")
    hole_rings = [
        np.array([[r.x, r.y] for r in hole], dtype=np.float64)
        for hole in hole_records
    ]
    polygon_with_holes = (
        Polygon(ring, hole_rings) if hole_rings else Polygon(ring)
    )
    if not polygon_with_holes.is_valid:
        polygon_with_holes = polygon_with_holes.buffer(0)

    boundary_nodes: list[tuple[float, float]] = (
        [(float(x), float(y)) for x, y in ring]
        + [
            (float(r.x), float(r.y))
            for hole in hole_records
            for r in hole
        ]
    )

    fallback_tops = np.array([record.top_z for record in records], dtype=np.float64)

    roof: ReconstructedRoof | None = None
    decomposition = roof_decomposition
    if (
        decomposition is not None
        and len(getattr(decomposition, "planes", [])) >= 2
        and len(fallback_tops) >= 3
    ):
        try:
            roof = reconstruct_roof(decomposition, polygon_with_holes)
        except (ValueError, TypeError) as exc:  # pragma: no cover - défensive
            log.info("reconstruction du toit impossible (%s) : extrusion", exc)
            roof = None

    vertices: list[np.ndarray] = []
    kinds: list[str] = []
    faces: list[list[int]] = []

    def add_vertex(point: np.ndarray) -> int:
        vertices.append(np.asarray(point, dtype=np.float64))
        return len(vertices) - 1

    if roof is not None and len(roof.plane_polygons) > 0:
        # Toit reconstruit : les murs dérivent de SA frontière. Chaque sommet
        # d'emprise interroge le pan propriétaire, pas le voisin le plus haut.
        all_records = records + [r for hole in hole_records for r in hole]
        tops = derive_wall_tops(all_records, roof, decomposition)
        tops = np.maximum(tops, [r.ground_z + MIN_WALL_M for r in all_records])
        for record, top in zip(all_records, tops):
            record.top_z = float(top)

        bottom_nodes: dict[tuple[float, float], int] = {}
        top_nodes: dict[tuple[float, float], int] = {}
        for node_index, (x, y) in enumerate(boundary_nodes):
            record = all_records[node_index]
            bottom_nodes[(round(x, 6), round(y, 6))] = add_vertex(
                np.array([x, y, record.ground_z])
            )
            top_nodes[(round(x, 6), round(y, 6))] = add_vertex(
                np.array([x, y, record.top_z])
            )

        # Fond posé sur le terrain, cours exclues ; normale vers le bas.
        for face in _cap_faces(
            polygon_with_holes, boundary_nodes, flip=True, index_offset=0
        ):
            faces.append(face)
            kinds.append("base")

        # Murs extérieurs ET murs de cour : même ruban, même consommation.
        exterior_ring = records
        for face in _wall_strip(bottom_nodes, top_nodes, exterior_ring):
            faces.append(face)
            kinds.append("wall")
        for hole in hole_records:
            for face in _wall_strip(bottom_nodes, top_nodes, hole):
                faces.append(face)
                kinds.append("wall")

        offset = len(vertices)
        for point in roof.vertices:
            add_vertex(point)
        for face_index, face in enumerate(roof.faces):
            faces.append([offset + int(v) for v in face])
            kind = "roof" if roof.face_plane[face_index] >= 0 else "roof_step"
            kinds.append(kind)
    else:
        # Extrusion canonique : fond contraint (cours vides), murs, couvercle
        # plat contraint. Aucune forme architecturale inventée.
        if fallback_tops.size == 0 or not np.isfinite(fallback_tops).all():
            raise ValueError("hauteur de toit manquante : impossible de fermer")
        tops = np.maximum(fallback_tops, grounds + MIN_WALL_M)
        for record, top in zip(records, tops):
            record.top_z = float(top)
        # Les parois de cour montent au niveau des avant-toits : la cour est
        # un vide du volume, pas une fosse séparée.
        eaves = float(np.max(tops))
        for hole in hole_records:
            for record in hole:
                record.top_z = max(eaves, record.ground_z + MIN_WALL_M)

        all_records = records + [r for hole in hole_records for r in hole]
        bottom_nodes: dict[tuple[float, float], int] = {}
        for node_index, (x, y) in enumerate(boundary_nodes):
            record = all_records[node_index]
            bottom_nodes[(round(x, 6), round(y, 6))] = add_vertex(
                np.array([x, y, record.ground_z])
            )

        for face in _cap_faces(polygon_with_holes, boundary_nodes, flip=True):
            faces.append(face)
            kinds.append("base")

        top_nodes: dict[tuple[float, float], int] = {}
        for node_index, (x, y) in enumerate(boundary_nodes):
            record = all_records[node_index]
            top_nodes[(round(x, 6), round(y, 6))] = add_vertex(
                np.array([x, y, record.top_z])
            )

        for face in _wall_strip(bottom_nodes, top_nodes, records):
            faces.append(face)
            kinds.append("wall")
        for hole in hole_records:
            for face in _wall_strip(bottom_nodes, top_nodes, hole):
                faces.append(face)
                kinds.append("wall")

        # Couvercle plat contraint : la cour reste ouverte sur le ciel.
        for face in _cap_faces(
            polygon_with_holes, boundary_nodes, index_offset=len(boundary_nodes)
        ):
            faces.append(face)
            kinds.append("roof")

    mesh = CanonicalSceneMesh(
        vertices=np.asarray(vertices, dtype=np.float64) if vertices else np.zeros((0, 3)),
        faces=np.asarray(faces, dtype=np.int64),
        face_kind=kinds,
        records=records,
    )
    welded = mesh.weld_vertices(weld_tolerance_m)
    welded.records = records
    welded.hole_records = hole_records
    return welded


def attach_canonical_meshes(scene, terrain=None) -> list[dict]:  # noqa: ANN001
    """Attach the authoritative mesh once, preserving roof planes and courtyards."""
    attached: list[dict] = []
    for prism in scene.prisms:
        top = prism.height_m
        if prism.roof_vertices is not None and len(prism.roof_vertices):
            top = np.asarray(prism.roof_vertices, dtype=float)[:, 2].max()
        prism.canonical_mesh = build_canonical_building_mesh(
            prism.footprint,
            top_heights=float(top),
            terrain=terrain,
            roof_decomposition=prism.roof_planes,
            interiors=list(getattr(prism, "interior_rings", [])),
        )
        prism.canonical_mesh.feature_id = prism.feature_id
        attached.append({
            "feature_id": prism.feature_id,
            "mesh_digest": prism.canonical_mesh.mesh_digest(),
            "faces": len(prism.canonical_mesh.faces),
        })
    return attached


__all__ = ["attach_canonical_meshes", "build_canonical_building_mesh"]
