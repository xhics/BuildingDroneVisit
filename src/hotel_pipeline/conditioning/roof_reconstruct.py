"""Reconstruction du toit : les plans se coupent, le maillage en hérite.

Détecter des plans ne reconstruit pas un toit. Tant que chaque pan garde sa
nappe de sommets propre, deux versants voisins ne partagent **rien** : leur
faîtage est un doublon approximatif, avec chevauchements et fissures à la clé.

La reconstruction ici suit la chaîne LOD2 complète :

1. intersection 3D **exacte** de chaque paire de plans adjacents — l'arête
   n'est pas cherchée dans le nuage, elle est déduite des équations ;
2. graphe pan ↔ arête (faîtage, noue, rives), qui dit quelles découpes
   s'appliquent à quels pans ;
3. découpage de chaque pan par ces lignes et par l'emprise — les polygones
   obtenus pavent le toit sans recouvrement ni trou ;
4. triangulation dans un registre de sommets partagé : une extrémité de
   faîtage est **le même sommet** pour les deux pans, au floatant près.

Les décrochements verticaux — un toit haut contre un toit bas, sans pente
intermédiaire — ne sont plus lissés en pente : la discontinuité produit une
**face verticale explicite** entre les deux surfaces.

Les murs, enfin, ne tirent plus leur hauteur vers le toit haut voisin :
chaque arête d'emprise appartient à un pan (celui dont le polygone découpé
la porte), et c'est l'équation de **ce** plan qui donne l'altitude du sommet
de mur. L'aile basse reste basse.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from ..logging import get_logger
from .roof_planes import (
    MIN_PLANE_POINTS,
    RIDGE_ADJACENCY_M,
    RIDGE_MIN_ANGLE_DEG,
    RoofDecomposition,
    RoofPlane,
    _intersect_planes,
)

log = get_logger("conditioning-roof-reconstruct")

#: En dessous de cette aire, un fragment de pan ne vaut pas une face.
MIN_POLYGON_AREA_M2 = 2.0

#: Décrochement vertical minimal entre deux surfaces quasi parallèles pour
#: exiger une face verticale franche plutôt qu'une pseudo-pente.
STEP_HEIGHT_M = 0.8

#: Angle sous lequel deux pans sont tenus pour quasi parallèles : leur
#: rencontre n'est pas un faîtage mais un décrochement potentiel.
PARALLEL_ANGLE_DEG = 15.0

#: Marge de découpe, en mètres : elle évite que deux polygones adjacents ne
#: se disputent une lame de recouvrement d'un centimètre de large.
CLIP_SLACK_M = 1e-6


@dataclass
class ReconstructedRoof:
    """La surface de toit reconstruite, pans confondus et soudés."""

    vertices: np.ndarray
    faces: np.ndarray
    #: Index du plan propriétaire de chaque face ; -1 pour une face
    #: verticale de décrochement.
    face_plane: list[int]
    #: Polygone final de chaque pan, en XY, après toutes les découpes.
    plane_polygons: dict[int, Polygon] = field(default_factory=dict)
    topology: RoofTopology | None = None

    @property
    def step_faces(self) -> list[int]:
        return [i for i, plane in enumerate(self.face_plane) if plane < 0]

    def boundary_edges_xy(self) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """Arêtes du bord de toiture, vues de dessus."""
        from collections import Counter

        counts: Counter[tuple[int, int]] = Counter()
        for face in self.faces:
            a, b, c = (int(v) for v in face)
            counts.update(
                {
                    tuple(sorted((a, b))),
                    tuple(sorted((b, c))),
                    tuple(sorted((c, a))),
                }
            )
        return [
            (
                (float(self.vertices[a][0]), float(self.vertices[a][1])),
                (float(self.vertices[b][0]), float(self.vertices[b][1])),
            )
            for (a, b), count in counts.items()
            if count == 1
        ]


class _VertexRegistry:
    """Registre partagé : une position est un sommet, quel que soit le pan.

    La recherche par plus proche voisin évite le piège du quantifiage : deux
    découpes de la même arête ne tombent jamais au même floatant près, et un
    arrondi par grille peut les séparer d'un poil de cellule.
    """

    #: Deux positions à moins d'un dixième de millimètre sont le même sommet.
    TOLERANCE_M = 1e-4

    def __init__(self, tolerance_m: float = TOLERANCE_M) -> None:
        self.tolerance = tolerance_m
        self.vertices: list[np.ndarray] = []

    def index(self, point: np.ndarray) -> int:
        point = np.asarray(point, dtype=np.float64)
        if self.vertices:
            from scipy.spatial import cKDTree

            tree = cKDTree(np.asarray(self.vertices))
            distance, nearest = tree.query(point)
            if distance <= self.tolerance:
                return int(nearest)
        self.vertices.append(point)
        return len(self.vertices) - 1


def _half_plane_polygon(
    start: np.ndarray, direction: np.ndarray, keep_point: np.ndarray
) -> Polygon:
    """Grand rectangle couvrant le demi-plan délimité par la droite donnée."""
    # La coupe est une opération XY : la droite d'intersection 3D est
    # projetée à plat, le demi-plan garde le côté du pan propriétaire.
    flat = np.asarray(direction[:2], dtype=np.float64)
    norm = float(np.linalg.norm(flat))
    if norm < 1e-9:
        return Polygon()
    unit = flat / norm
    normal = np.array([-unit[1], unit[0]])
    side = float(np.dot(np.asarray(keep_point[:2]) - start[:2], normal))
    if side < 0:
        normal = -normal
    span = 1.0e5
    corners = [
        start[:2] + unit * span,
        start[:2] - unit * span,
        start[:2] - unit * span + normal * span,
        start[:2] + unit * span + normal * span,
    ]
    return Polygon(corners)


#: Distance en deçà de laquelle deux pans sont tenus pour adjacents quand
#: ils forment un faîtage, en mètres.
RIDGE_CONTACT_M = RIDGE_ADJACENCY_M

#: Les décrochements peuvent séparer davantage : un toit haut s'arrête, le
#: toit bas commence un peu plus loin. Rayon et effectif assouplis.
STEP_CONTACT_M = 4.0
STEP_MIN_CONTACT_POINTS = 5


@dataclass(frozen=True)
class RoofTopologyEdge:
    """Finite exact plane/plane intersection used by the canonical roof."""

    edge_id: str
    kind: str
    plane_ids: tuple[str, str]
    start: np.ndarray
    end: np.ndarray

    def as_dict(self) -> dict:
        return {
            "edge_id": self.edge_id,
            "kind": self.kind,
            "plane_ids": list(self.plane_ids),
            "start": np.round(self.start, 6).tolist(),
            "end": np.round(self.end, 6).tolist(),
            "length_m": round(float(np.linalg.norm(self.end - self.start)), 6),
        }


@dataclass
class RoofTopology:
    """Roof planes and their physical junctions, not a display overlay."""

    planes: list[RoofPlane]
    boundaries: dict[str, list[list[float]]] = field(default_factory=dict)
    ridges: list[RoofTopologyEdge] = field(default_factory=list)
    valleys: list[RoofTopologyEdge] = field(default_factory=list)
    hips: list[RoofTopologyEdge] = field(default_factory=list)
    steps: list[RoofTopologyEdge] = field(default_factory=list)
    parapets: list[RoofTopologyEdge] = field(default_factory=list)
    open_area_m2: float = 0.0
    overlap_area_m2: float = 0.0

    def as_dict(self) -> dict:
        return {
            "planes": [plane.as_dict() for plane in self.planes],
            "boundaries": self.boundaries,
            "ridges": [edge.as_dict() for edge in self.ridges],
            "valleys": [edge.as_dict() for edge in self.valleys],
            "hips": [edge.as_dict() for edge in self.hips],
            "steps": [edge.as_dict() for edge in self.steps],
            "parapets": [edge.as_dict() for edge in self.parapets],
            "open_area_m2": round(float(self.open_area_m2), 6),
            "overlap_area_m2": round(float(self.overlap_area_m2), 6),
        }


def _adjacency(
    decomposition: RoofDecomposition,
    contact_m: float,
    min_points: int = 8,
) -> list[tuple[int, int]]:
    """Paires de pans qui se touchent quelque part, par proximité de points."""
    from scipy.spatial import cKDTree

    planes = decomposition.planes
    pairs: list[tuple[int, int]] = []
    trees = [
        cKDTree(plane.points[:, :2])
        for plane in planes
        if len(plane.points) >= MIN_PLANE_POINTS
    ]
    indices = [
        i for i, plane in enumerate(planes) if len(plane.points) >= MIN_PLANE_POINTS
    ]
    for a_pos in range(len(indices)):
        for b_pos in range(a_pos + 1, len(indices)):
            i, j = indices[a_pos], indices[b_pos]
            close = trees[b_pos].query(planes[i].points[:, :2])[0]
            if bool((close <= contact_m).sum() >= min_points):
                pairs.append((i, j))
    return pairs


def _angle_between(a: RoofPlane, b: RoofPlane) -> float:
    cosine = abs(float(np.dot(a.normal, b.normal)))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _clip_polygon_by_ridge(
    polygon: Polygon, ridge_start: np.ndarray, ridge_dir: np.ndarray, keep: np.ndarray
) -> Polygon:
    half = _half_plane_polygon(ridge_start, ridge_dir, keep)
    clipped = polygon.intersection(half)
    return clipped if isinstance(clipped, Polygon) else Polygon()


def _triangulate_with_z(polygon: Polygon, plane: RoofPlane, registry: _VertexRegistry):
    """Triangule un polygone XY, z porté par l'équation du plan."""
    from shapely.ops import triangulate as shapely_triangulate

    faces: list[tuple[int, int, int]] = []
    for triangle in shapely_triangulate(polygon):
        if not polygon.covers(triangle):
            continue
        coords = np.asarray(triangle.exterior.coords[:-1], dtype=np.float64)
        points = np.column_stack([
            coords[:, 0],
            coords[:, 1],
            [plane.height_at(x, y) for x, y in coords],
        ])
        faces.append(tuple(registry.index(point) for point in points))
    return faces


def _finite_intersection_edge(
    pair: tuple[int, int], line: tuple[np.ndarray, np.ndarray],
    footprint: Polygon, decomposition: RoofDecomposition,
) -> RoofTopologyEdge | None:
    """Clip an exact infinite plane intersection to the physical roof extent."""
    i, j = pair
    point, direction = line
    span = max(100.0, float(np.hypot(*(np.ptp(np.asarray(footprint.exterior.coords), axis=0)[:2]))) * 4.0)
    candidate = LineString([point[:2] - direction[:2] * span, point[:2] + direction[:2] * span])
    clipped = candidate.intersection(footprint)
    pieces = list(clipped.geoms) if hasattr(clipped, "geoms") else [clipped]
    pieces = [piece for piece in pieces if isinstance(piece, LineString) and piece.length > 1e-6]
    if not pieces:
        return None
    segment = max(pieces, key=lambda piece: piece.length)
    xy = np.asarray([segment.coords[0], segment.coords[-1]], dtype=float)
    first, second = decomposition.planes[i], decomposition.planes[j]
    start = np.array([xy[0, 0], xy[0, 1], first.height_at(*xy[0])])
    end = np.array([xy[1, 0], xy[1, 1], first.height_at(*xy[1])])
    mid_z = float((start[2] + end[2]) * 0.5)
    reference_z = float((first.origin[2] + second.origin[2]) * 0.5)
    if abs(end[2] - start[2]) <= 0.05:
        kind = "ridge" if mid_z >= reference_z else "valley"
    else:
        kind = "hip"
    return RoofTopologyEdge(
        edge_id=f"{kind}_{i:02d}_{j:02d}", kind=kind,
        plane_ids=(first.plane_id, second.plane_id), start=start, end=end,
    )


def reconstruct_roof(
    decomposition: RoofDecomposition,
    footprint: Polygon,
    min_area_m2: float = MIN_POLYGON_AREA_M2,
) -> ReconstructedRoof | None:
    """Reconstruit le toit complet depuis ses plans, arêtes comprises.

    Retourne None quand il n'y a rien à reconstruire ensemble : moins de
    deux pans exploitables, ou aucun découpage productif.
    """
    seen_plane_ids: set[str] = set()
    for index, plane in enumerate(decomposition.planes):
        if plane.plane_id == "plane_unassigned" or plane.plane_id in seen_plane_ids:
            plane.plane_id = f"plane_{index:02d}"
        seen_plane_ids.add(plane.plane_id)
    usable = [
        (index, plane)
        for index, plane in enumerate(decomposition.planes)
        if len(plane.points) >= MIN_PLANE_POINTS
    ]
    if len(usable) < 1:
        return None

    footprint = footprint if footprint.is_valid else footprint.buffer(0)
    if footprint.is_empty or footprint.geom_type != "Polygon":
        return None

    registry = _VertexRegistry()
    polygons: dict[int, Polygon] = {}
    for index, plane in usable:
        hull = Polygon(plane.points[:, :2]).convex_hull.buffer(CLIP_SLACK_M)
        clipped = hull.intersection(footprint)
        polygons[index] = clipped if isinstance(clipped, Polygon) else Polygon()

    pairs = _adjacency(decomposition, RIDGE_CONTACT_M)
    step_candidates = [
        pair for pair in _adjacency(decomposition, STEP_CONTACT_M, STEP_MIN_CONTACT_POINTS)
        if pair not in pairs
    ]
    # Graphe pan ↔ arête : chaque paire adjacente produit soit un faîtage
    # (coupe par demi-plan), soit un décrochement (face verticale).
    ridges: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    steps: list[tuple[int, int]] = []
    for i, j in sorted(pairs):
        first, second = decomposition.planes[i], decomposition.planes[j]
        angle = _angle_between(first, second)
        line = _intersect_planes(first, second)
        if angle >= PARALLEL_ANGLE_DEG:
            if line is None or angle < RIDGE_MIN_ANGLE_DEG:
                continue
            ridges[(i, j)] = line
        else:
            # Quasi parallèles et proches : décrochement si les altitudes
            # divergent franchement.
            centre_i = first.origin
            centre_j = second.origin
            gap = abs(first.height_at(*centre_j[:2]) - centre_j[2])
            gap = max(gap, abs(second.height_at(*centre_i[:2]) - centre_i[2]))
            if gap >= STEP_HEIGHT_M:
                steps.append((i, j))
    for i, j in sorted(step_candidates):
        first, second = decomposition.planes[i], decomposition.planes[j]
        if _angle_between(first, second) >= PARALLEL_ANGLE_DEG:
            continue  # des pans en pente qui s'écartent ne font pas un mur
        centre_i = first.origin
        centre_j = second.origin
        gap = abs(first.height_at(*centre_j[:2]) - centre_j[2])
        gap = max(gap, abs(second.height_at(*centre_i[:2]) - centre_i[2]))
        if gap >= STEP_HEIGHT_M:
            steps.append((i, j))

    # Coupe de chaque pan par les faîtages qui le bordent.
    ridge_plane_indices = {index for pair in ridges for index in pair}
    if len(usable) == 1:
        ridge_plane_indices.add(usable[0][0])
    for index in ridge_plane_indices:
        polygons[index] = footprint

    for (i, j), (start, direction) in ridges.items():
        for owner, other in ((i, j), (j, i)):
            polygon = polygons.get(owner)
            if polygon is None or polygon.is_empty:
                continue
            keep = decomposition.planes[owner].origin
            polygons[owner] = _clip_polygon_by_ridge(polygon, start, direction, keep)

    # Décrochements : le toit bas perd le recouvrement sous le toit haut,
    # ou est étendu jusqu'au toit haut quand les deux nappes s'écartent.
    step_pairs: list[tuple[int, int]] = []
    for i, j in steps:
        zi = decomposition.planes[i].origin[2]
        zj = decomposition.planes[j].origin[2]
        low_index, high_index = (j, i) if zi > zj else (i, j)
        high = polygons.get(high_index)
        low = polygons.get(low_index)
        if high is None or low is None or high.is_empty or low.is_empty:
            continue
        overlap = low.intersection(high)
        if overlap.is_empty or not isinstance(overlap, Polygon):
            # Nappes disjointes : l'intervalle entre les deux bords revient
            # au pan bas, qui vient mourir contre le bord du pan haut. La
            # jonction devient alors une frontière commune, découpable.
            gap = low.distance(high)
            if gap > STEP_CONTACT_M:
                continue
            bridge = Polygon(
                list(high.exterior.coords) + list(low.exterior.coords)
            ).convex_hull
            infill = bridge.difference(low).difference(high)
            pieces = (
                list(infill.geoms) if infill.geom_type == "MultiPolygon" else [infill]
            )
            additions = [
                piece
                for piece in pieces
                if isinstance(piece, Polygon) and piece.area >= min_area_m2 * 0.25
            ]
            if not additions:
                continue
            low = low.union(unary_union(additions) if len(additions) > 1 else additions[0])
        if isinstance(overlap, Polygon) or not low.intersection(high).is_empty:
            cut = low.difference(high.buffer(CLIP_SLACK_M))
        else:
            cut = low
        pieces = (
            list(cut.geoms) if cut.geom_type == "MultiPolygon" else [cut]
        )
        merged: list[Polygon] = []
        for piece in pieces:
            if isinstance(piece, Polygon) and piece.area >= min_area_m2:
                merged.append(piece)
        if merged:
            merged_polygon = (
                unary_union(merged) if len(merged) > 1 else merged[0]
            )
            if isinstance(merged_polygon, Polygon):
                polygons[low_index] = merged_polygon
                step_pairs.append((low_index, high_index))

    faces: list[list[int]] = []
    face_plane: list[int] = []
    for index, polygon in polygons.items():
        if polygon is None or polygon.is_empty or not isinstance(polygon, Polygon):
            continue
        if polygon.area < min_area_m2:
            continue
        plane = decomposition.planes[index]
        for tri in _triangulate_with_z(polygon, plane, registry):
            faces.append(list(tri))
            face_plane.append(index)
        polygons[index] = polygon

    # Faces verticales franches le long des frontières de découpe : le toit
    # haut et le toit bas sont joints par un mur, jamais par une pente.
    topology_step_edges: list[RoofTopologyEdge] = []
    for low_index, high_index in step_pairs:
        low_polygon = polygons.get(low_index)
        if low_polygon is None or low_polygon.is_empty or not isinstance(low_polygon, Polygon):
            continue
        high = polygons.get(high_index)
        if high is None or not isinstance(high, Polygon):
            continue
        low_plane = decomposition.planes[low_index]
        high_plane = decomposition.planes[high_index]
        coords = np.asarray(low_polygon.exterior.coords[:-1], dtype=np.float64)
        count = len(coords)
        for k in range(count):
            a = coords[k]
            b = coords[(k + 1) % count]
            mid = (a + b) / 2.0
            if high.boundary.distance(Point(mid)) > 0.05:
                continue
            if not high.covers(Point(mid)) and high.exterior.distance(Point(mid)) > 0.05:
                continue
            za = low_plane.height_at(a[0], a[1])
            zb = low_plane.height_at(b[0], b[1])
            ta = high_plane.height_at(a[0], a[1])
            tb = high_plane.height_at(b[0], b[1])
            if max(ta - za, tb - zb) < STEP_HEIGHT_M * 0.5:
                continue
            ia = registry.index(np.array([a[0], a[1], za]))
            ib = registry.index(np.array([b[0], b[1], zb]))
            itb = registry.index(np.array([b[0], b[1], tb]))
            ita = registry.index(np.array([a[0], a[1], ta]))
            faces.extend([[ia, ib, itb], [ia, itb, ita]])
            face_plane.extend([-1, -1])
            topology_step_edges.append(RoofTopologyEdge(
                edge_id=f"step_{low_index:02d}_{high_index:02d}_{k:03d}",
                kind="step",
                plane_ids=(low_plane.plane_id, high_plane.plane_id),
                start=np.array([a[0], a[1], za]),
                end=np.array([b[0], b[1], zb]),
            ))

    if not faces:
        return None

    topology = RoofTopology(planes=[plane for _, plane in usable])
    topology.boundaries = {
        decomposition.planes[index].plane_id: np.round(
            np.asarray(poly.exterior.coords[:-1], dtype=float), 6
        ).tolist()
        for index, poly in polygons.items()
        if isinstance(poly, Polygon) and not poly.is_empty
    }
    for index, poly in polygons.items():
        if isinstance(poly, Polygon) and not poly.is_empty:
            decomposition.planes[index].polygon_boundary = topology.boundaries[
                decomposition.planes[index].plane_id
            ]
    for pair, line in sorted(ridges.items()):
        edge = _finite_intersection_edge(pair, line, footprint, decomposition)
        if edge is None:
            continue
        getattr(topology, f"{edge.kind}s").append(edge)
    topology.steps = topology_step_edges
    roof_polygons = [
        poly for poly in polygons.values()
        if isinstance(poly, Polygon) and not poly.is_empty
    ]
    if roof_polygons:
        covered = unary_union(roof_polygons)
        topology.open_area_m2 = float(max(0.0, footprint.area - covered.intersection(footprint).area))
        topology.overlap_area_m2 = float(max(
            0.0, sum(poly.area for poly in roof_polygons) - covered.area
        ))

    roof = ReconstructedRoof(
        vertices=np.asarray(registry.vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        face_plane=face_plane,
        plane_polygons={
            index: poly for index, poly in polygons.items() if isinstance(poly, Polygon)
        },
        topology=topology,
    )
    log.info(
        "%s : toit reconstruit — %d sommets partagés, %d faces dont %d décrochement(s)",
        decomposition.feature_id,
        len(roof.vertices),
        len(roof.faces),
        len(roof.step_faces),
    )
    return roof


def owning_plane_for_edge(
    roof: ReconstructedRoof, decomposition: RoofDecomposition, x: float, y: float
) -> RoofPlane | None:
    """Le pan propriétaire d'un point de l'emprise, par couverture réelle.

    Un mur ne monte jamais vers le toit haut voisin : son altitude vient du
    plan dont le polygone découpé couvre effectivement ce point.
    """
    point = Point(x, y)
    for index, polygon in roof.plane_polygons.items():
        if polygon.covers(point):
            return decomposition.planes[index]
    best = None
    best_distance = float("inf")
    for index, polygon in roof.plane_polygons.items():
        distance = polygon.distance(point)
        if distance < best_distance:
            best_distance = distance
            best = decomposition.planes[index]
    return best


def derive_wall_tops(
    records,  # noqa: ANN001 - séquence de FootprintVertex
    roof: ReconstructedRoof,
    decomposition: RoofDecomposition,
) -> np.ndarray:
    """Altitude du haut de mur à chaque sommet d'emprise, pan par pan.

    Chaque sommet interroge le pan qui lui appartient — non le maximum des
    toitures voisines. L'aile basse accolée au corps haut reste basse.
    """
    tops = np.empty(len(records), dtype=np.float64)
    for position, record in enumerate(records):
        plane = owning_plane_for_edge(roof, decomposition, record.x, record.y)
        tops[position] = (
            plane.height_at(record.x, record.y)
            if plane is not None
            else float(record.top_z)
        )
    return tops
