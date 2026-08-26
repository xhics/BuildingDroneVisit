"""Assemblage du maillage canonique d'un bâtiment.

Un seul chemin mène au maillage : emprise normalisée en sommets identifiés,
pieds posés sur le terrain mesuré, toiture reconstruite depuis ses plans
quand ils le permettent, murs dérivés de la frontière du toit — jamais
l'inverse. La soudure finale referme le volume ; l'audit compte les arêtes
libres et doit en trouver zéro.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from .canonical_mesh import (
    CanonicalSceneMesh,
    WELD_TOLERANCE_M,
    footprint_records,
)
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


def _bottom_cap(
    ring: np.ndarray,
    grounds: np.ndarray,
    _lookup=None,  # noqa: ANN001 - paramètre historique, ignoré
) -> tuple[list[list[int]], np.ndarray]:
    """Fond du bâtiment : chaque sommet à son altitude de terrain.

    Sur terrain incliné, aucune extrémité de mur ne flotte ni ne pénètre le
    sol : le fond suit exactement la même nappe que les pieds des murs.
    Les triangles sont indexés dans l'ordre du fond retourné ; le maillage
    final les décale de sa base commune.
    """
    bottom_vertices = np.column_stack([ring, grounds])
    lookup = {
        (round(float(x), 6), round(float(y), 6)): index
        for index, (x, y) in enumerate(ring)
    }
    faces: list[list[int]] = []
    polygon = Polygon(ring)
    from shapely.ops import triangulate as shapely_triangulate

    for triangle in shapely_triangulate(polygon):
        if not polygon.covers(triangle):
            continue
        indices = []
        for x, y in triangle.exterior.coords[:-1]:
            index = lookup.get((round(float(x), 6), round(float(y), 6)))
            if index is None:
                # La triangulation de Delaunay d'un anneau ne crée pas de
                # sommet : si elle le faisait, on abandonne ce triangle.
                indices = []
                break
            indices.append(index)
        if len(indices) == 3:
            a, b, c = ring[indices]
            cross = float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
            if cross < 0:
                indices.reverse()
            faces.append(list(indices))
    return faces, bottom_vertices


def build_canonical_building_mesh(
    footprint: np.ndarray,
    top_heights: np.ndarray | float | None = None,
    terrain=None,  # noqa: ANN001 - TerrainGrid ou équivalent
    roof_decomposition: RoofDecomposition | None = None,
    weld_tolerance_m: float = WELD_TOLERANCE_M,
) -> CanonicalSceneMesh:
    """Construit LE maillage du bâtiment — celui que tout le monde consomme.

    Paramètres
    ----------
    footprint :
        Contour de l'emprise, dans n'importe quel sens CW/CCW.
    top_heights :
        Hauteurs de toit par sommet, utilisées comme altitude de repli quand
        aucun pan de toiture reconstruit ne revendique le sommet.
    terrain :
        Grille de terrain (``height_at(x, y)``) pour poser les pieds de murs.
    roof_decomposition :
        Segmentation en plans ; présente, elle commande la reconstruction.
    """
    records = footprint_records(footprint, top_heights=top_heights)

    # Le pied du mur suit le terrain extérieur ; le niveau structurel du
    # plancher resterait un attribut distinct, non confondu avec ce sol.
    for record in records:
        record.ground_z = _terrain_height(terrain, record.x, record.y)
    grounds = np.array([record.ground_z for record in records], dtype=np.float64)
    ring = np.array([[record.x, record.y] for record in records], dtype=np.float64)

    fallback_tops = np.array(
        [record.top_z for record in records], dtype=np.float64
    )

    roof: ReconstructedRoof | None = None
    decomposition = roof_decomposition
    if (
        decomposition is not None
        and len(getattr(decomposition, "planes", [])) >= 2
        and len(fallback_tops) >= 3
    ):
        try:
            roof = reconstruct_roof(decomposition, Polygon(ring))
        except (ValueError, TypeError) as exc:  # pragma: no cover - défensive
            from ..logging import get_logger

            get_logger("canonical-building").info(
                "reconstruction du toit impossible (%s) : extrusion", exc
            )
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
        tops = derive_wall_tops(records, roof, decomposition)
        tops = np.maximum(tops, grounds + MIN_WALL_M)
        for record, top in zip(records, tops):
            record.top_z = float(top)

        top_indices = [add_vertex(np.array([r.x, r.y, r.top_z])) for r in records]
        bottom_lookup: dict[tuple[float, float], int] = {}

        # Fond posé sur le terrain.
        cap_faces, cap_bottoms = _bottom_cap(ring, grounds, {})
        base_offset = len(vertices)
        for point in cap_bottoms:
            index = add_vertex(point)
            bottom_lookup[(round(float(point[0]), 6), round(float(point[1]), 6))] = index
        for face in cap_faces:
            faces.append([base_offset + i for i in face])
            kinds.append("base")

        # Murs : du pied au haut dérivé du toit, sommet pour sommet.
        count = len(records)
        for i in range(count):
            j = (i + 1) % count
            bi = bottom_lookup[(round(float(ring[i][0]), 6), round(float(ring[i][1]), 6))]
            bj = bottom_lookup[(round(float(ring[j][0]), 6), round(float(ring[j][1]), 6))]
            ti = top_indices[i]
            tj = top_indices[j]
            faces.extend([[bi, bj, tj], [bi, tj, ti]])
            kinds.extend(["wall", "wall"])

        # Surface de toit reconstruite, sommets partagés compris.
        offset = len(vertices)
        for point in roof.vertices:
            add_vertex(point)
        for face_index, face in enumerate(roof.faces):
            faces.append([offset + int(v) for v in face])
            kind = "roof" if roof.face_plane[face_index] >= 0 else "roof_step"
            kinds.append(kind)
    else:
        # Extrusion canonique : même code pour tous les consommateurs, plus
        # aucun extrudeur parallèle dans le pipeline.
        if fallback_tops.size == 0 or not np.isfinite(fallback_tops).all():
            raise ValueError("hauteur de toit manquante : impossible de fermer")
        tops = np.maximum(fallback_tops, grounds + MIN_WALL_M)
        for record, top in zip(records, tops):
            record.top_z = float(top)

        cap_faces, cap_bottoms = _bottom_cap(ring, grounds, {})
        bottom_lookup: dict[tuple[float, float], int] = {}
        for point in cap_bottoms:
            bottom_lookup[(round(float(point[0]), 6), round(float(point[1]), 6))] = (
                add_vertex(point)
            )
        for face in cap_faces:
            faces.append(list(face))
            kinds.append("base")

        count = len(records)
        top_indices = [add_vertex(np.array([r.x, r.y, r.top_z])) for r in records]
        for i in range(count):
            j = (i + 1) % count
            bi = bottom_lookup[(round(float(ring[i][0]), 6), round(float(ring[i][1]), 6))]
            bj = bottom_lookup[(round(float(ring[j][0]), 6), round(float(ring[j][1]), 6))]
            faces.extend([[bi, bj, top_indices[j]], [bi, top_indices[j], top_indices[i]]])
            kinds.extend(["wall", "wall"])

        # Couvercle plat : même triangulation XY que le fond.
        top_polygon = Polygon(ring)
        from shapely.ops import triangulate as shapely_triangulate

        lookup_top = {
            (round(float(record.x), 6), round(float(record.y), 6)): index
            for index, record in enumerate(records)
        }
        for triangle in shapely_triangulate(top_polygon):
            if not top_polygon.covers(triangle):
                continue
            ring_positions = []
            for x, y in triangle.exterior.coords[:-1]:
                position = lookup_top.get((round(float(x), 6), round(float(y), 6)))
                if position is None:
                    ring_positions = []
                    break
                ring_positions.append(position)
            if len(ring_positions) == 3:
                pa, pb, pc = (records[position] for position in ring_positions)
                cross = float(
                    (pb.x - pa.x) * (pc.y - pa.y) - (pb.y - pa.y) * (pc.x - pa.x)
                )
                if cross < 0:
                    ring_positions.reverse()
                faces.append([top_indices[position] for position in ring_positions])
                kinds.append("roof")

    mesh = CanonicalSceneMesh(
        vertices=np.asarray(vertices, dtype=np.float64) if vertices else np.zeros((0, 3)),
        faces=np.asarray(faces, dtype=np.int64),
        face_kind=kinds,
        records=records,
    )
    welded = mesh.weld_vertices(weld_tolerance_m)
    welded.records = records
    return welded


def attach_canonical_meshes(
    scene,  # noqa: ANN001 - ConditioningScene
    terrain=None,  # noqa: ANN001
) -> dict:
    """Donne à chaque prisme son maillage canonique, une seule fois.

    Après cette passe, le renderer, le textureur, la collision et l'export
    lisent tous `prism.canonical_mesh` : aucune extrusion locale ne subsiste.
    """
    built = 0
    for prism in scene.prisms:
        if getattr(prism, "canonical_mesh", None) is not None:
            continue
        try:
            mesh = build_canonical_building_mesh(
                prism.footprint,
                top_heights=(
                    np.asarray(prism.roof_vertices[:, 2])
                    if prism.roof_measured
                    else prism.height_m
                ),
                terrain=terrain,
                roof_decomposition=getattr(prism, "roof_planes", None),
            )
        except (ValueError, TypeError):
            continue
        mesh.feature_id = prism.feature_id
        mesh.provenance_class = "OCCLUDED_INFERRED" if prism.height_assumed else "LIDAR_MEASURED"
        prism.canonical_mesh = mesh
        built += 1
    return {
        "canonical_meshes_built": built,
        "prisms": len(scene.prisms),
        "digests": {p.feature_id: p.canonical_mesh.mesh_digest() for p in scene.prisms if p.canonical_mesh},
    }
