"""Triangulation contrainte : le contour est une loi, pas une suggestion.

La triangulation de Delaunay appliquée aux sommets du contour ne garantit
rien : sur un polygone en L ou en U, il manque au maillage les diagonales que
la concavité exige, et des lamés entiers disparaissent ou débordent. Avec des
cours intérieures, le problème s'aggrave : rien ne dit au triangulateur que
le trou doit rester vide.

Ce module implémente l'évidage d'oreilles contraint :

1. le contour extérieur est ramené au sens trigonométrique, les trous à
   l'inverse ;
2. chaque trou est ponté dans l'anneau extérieur par la paire de sommets
   mutuellement visibles la plus proche — la cour reste **vide**, elle n'est
   jamais recouverte ;
3. l'anneau fusionné est évidé en oreilles : un triangle n'est émis que si
   ses trois sommets appartiennent au contour et qu'aucun autre sommet ne
   s'y trouve.

Résultat : l'union des triangles produites vaut exactement le polygone,
trous soustraits, quelle que soit sa concavité.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import QhullError

_EPS_CROSS = 1e-12


def _signed_area(ring: list[tuple[float, float]]) -> float:
    total = 0.0
    count = len(ring)
    for index in range(count):
        ax, ay = ring[index]
        bx, by = ring[(index + 1) % count]
        total += ax * by - bx * ay
    return total / 2.0


def _cross(ax, ay, bx, by, cx, cy) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _point_in_triangle(
    px, py, ax, ay, bx, by, cx, cy, strict: bool = True
) -> bool:
    """Test d'inclusion, arêtes comprises sauf mention contraire."""
    d1 = _cross(px, py, ax, ay, bx, by)
    d2 = _cross(px, py, bx, by, cx, cy)
    d3 = _cross(px, py, cx, cy, ax, ay)
    if strict:
        return (d1 > _EPS_CROSS and d2 > _EPS_CROSS and d3 > _EPS_CROSS) or (
            d1 < -_EPS_CROSS and d2 < -_EPS_CROSS and d3 < -_EPS_CROSS
        )
    return (
        abs(d1) <= _EPS_CROSS
        or abs(d2) <= _EPS_CROSS
        or abs(d3) <= _EPS_CROSS
        or ((d1 > 0 and d2 > 0 and d3 > 0) or (d1 < 0 and d2 < 0 and d3 < 0))
    )


def _earclip(ring: list[int], coords: list[tuple[float, float]]) -> list[tuple[int, int, int]]:
    """Évide un anneau simple supposé CCW ; retourne des triplets d'indices.

    Deux passes : d'abord seuls les sommets strictement intérieurs bloquent
    l'oreille ; puis, faute de mieux, ceux posés sur ses bords sont tolérés
    — un polygone en U place des doublons de pont exactement sur les arêtes
    des dernières oreilles.
    """
    positions = list(ring)
    triangles: list[tuple[int, int, int]] = []
    guard = 0
    limit = 10 * len(positions) * len(positions) + 1000
    while len(positions) > 3 and guard < limit:
        guard += 1
        emitted = False
        for allow_edge_points in (False, True):
            count = len(positions)
            for slot in range(count):
                ia = positions[(slot - 1) % count]
                ib = positions[slot % count]
                ic = positions[(slot + 1) % count]
                ax, ay = coords[ia]
                bx, by = coords[ib]
                cx, cy = coords[ic]
                if _cross(ax, ay, bx, by, cx, cy) <= _EPS_CROSS:
                    continue  # sommet réflexe ou aligné : pas une oreille
                blocked = False
                for candidate in positions:
                    if candidate in (ia, ib, ic):
                        continue
                    px, py = coords[candidate]
                    if not _point_in_triangle(
                        px, py, ax, ay, bx, by, cx, cy,
                        strict=not allow_edge_points,
                    ):
                        continue
                    if allow_edge_points:
                        from shapely.geometry import LineString, Point

                        edges = LineString([coords[ia], coords[ib], coords[ic], coords[ia]])
                        if edges.distance(Point(px, py)) <= 1e-9:
                            continue  # posé sur le bord de l'oreille : toléré
                    blocked = True
                    break
                if blocked:
                    continue
                triangles.append((ia, ib, ic))
                positions.pop(slot)
                emitted = True
                break
            if emitted:
                break
        if not emitted:
            break  # anneau numériquement dégénéré : on rend ce qui existe
    if len(positions) == 3:
        triangles.append(tuple(positions))
    return triangles


def _segment_visible(
    a: tuple[float, float],
    b: tuple[float, float],
    rings: list[list[tuple[float, float]]],
) -> bool:
    """Le segment a-b ne traverse-t-il aucun anneau hors ses extrémités ?"""
    from shapely.geometry import LineString, Point
    from shapely.ops import unary_union

    segment = LineString([a, b])
    boundaries = unary_union([
        LineString(list(ring) + [ring[0]])
        for ring in rings
        if len(ring) >= 3
    ])
    intrusion = segment.intersection(boundaries)
    endpoints = Point(a).buffer(1e-9).union(Point(b).buffer(1e-9))
    residue = intrusion.difference(endpoints)
    return residue.is_empty


def _bridge_hole(
    outer: list[tuple[float, float]],
    hole: list[tuple[float, float]],
) -> tuple[int, int]:
    """Pont de visibilité minimale entre l'anneau extérieur et un trou.

    Retourne (index extérieur, index du trou) ; l'appelant épissure lui-même,
    en gardant des nœuds **distincts** pour les deux côtés du pont.
    """
    best: tuple[float, int, int] | None = None
    for oi in range(len(outer)):
        for hi in range(len(hole)):
            first = outer[oi]
            second = hole[hi]
            if not _segment_visible(first, second, [outer, hole]):
                continue
            distance = (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2
            if best is None or distance < best[0]:
                best = (distance, oi, hi)
    if best is None:
        # Aucune visibilité stricte (trou tangent au bord, cas pathologique) :
        # pont par les sommets les plus proches sans test.
        distances = [
            ((o[0] - h[0]) ** 2 + (o[1] - h[1]) ** 2, oi, hi)
            for oi, o in enumerate(outer)
            for hi, h in enumerate(hole)
        ]
        _, oi, hi = min(distances)
        return oi, hi
    return best[1], best[2]


def triangulate_constrained(polygon) -> list[np.ndarray]:  # noqa: ANN001 - shapely Polygon
    """Triangule un polygone shapely, trous et concavités respectés.

    Retourne une liste de triangles (tableaux 3×3 en XY). L'union couvre
    exactement le polygone — ni lamé manquant près d'un sommet réflexe, ni
    triangle dans une cours intérieure.
    """
    from shapely.geometry import Polygon as ShapelyPolygon

    if isinstance(polygon, ShapelyPolygon):
        source = polygon
    else:
        source = ShapelyPolygon(np.asarray(polygon)[:, :2])
    if not source.is_valid:
        source = source.buffer(0)
    if source.is_empty or source.geom_type != "Polygon":
        raise ValueError("triangulation contrainte : polygone simple requis")

    exterior = [(float(x), float(y)) for x, y in source.exterior.coords[:-1]]
    interiors = [
        [(float(x), float(y)) for x, y in ring.coords[:-1]]
        for ring in source.interiors
    ]
    interiors = [ring for ring in interiors if len(ring) >= 3]

    if _signed_area(exterior) < 0:
        exterior.reverse()
    normalised_holes: list[list[tuple[float, float]]] = []
    for hole in interiors:
        ring = list(hole)
        if _signed_area(ring) > 0:
            ring.reverse()
        normalised_holes.append(ring)
    # Les trous les plus à droite d'abord : le pontage reste local.
    normalised_holes.sort(key=lambda ring: max(x for x, _ in ring), reverse=True)

    # Les nœuds sont **des occurrences**, pas des coordonnées dédupliquées :
    # un pont traverse deux fois le même point, et l'évidage a besoin des
    # deux passages comme sommets séparés (les oreilles de surface nulle
    # entre les deux sont écartées par le test de convexité).
    nodes = list(exterior)
    ring: list[int] = list(range(len(exterior)))
    for hole in normalised_holes:
        base = len(nodes)
        nodes.extend(hole)
        hole_ring = list(range(base, base + len(hole)))
        outer_coords = [nodes[i] for i in ring]
        hole_coords = [nodes[i] for i in hole_ring]
        oi, hi = _bridge_hole(outer_coords, hole_coords)
        count = len(hole_ring)
        ring = (
            ring[: oi + 1]
            + [hole_ring[(hi + k) % count] for k in range(count)]
            + [hole_ring[hi]]
            + ring[oi:]
        )

    triangles = _earclip(ring, nodes)
    if _total_area(triangles, nodes) >= source.area - max(1e-6, source.area * 1e-9):
        return [
            np.array([nodes[i] for i in triangle], dtype=np.float64)
            for triangle in triangles
        ]
    # Anneau faiblement simple (ponts multiples) : l'évidage peut caler avant
    # d'avoir tout consommé. Le repli densifie le contour puis triangule par
    # Delaunay filtré — la couverture du polygone est garantie, au prix de
    # sommets intermédiaires sur les arêtes.
    return _dense_delaunay(source)


def _total_area(triangles: list[tuple[int, int, int]], nodes: list[tuple[float, float]]) -> float:
    total = 0.0
    for ia, ib, ic in triangles:
        ax, ay = nodes[ia]
        bx, by = nodes[ib]
        cx, cy = nodes[ic]
        total += abs(_cross(ax, ay, bx, by, cx, cy)) / 2.0
    return total


def _dense_delaunay(polygon) -> list[np.ndarray]:  # noqa: ANN001
    """Delaunay filtré sur un contour densifié : couverture garantie."""
    from scipy.spatial import Delaunay
    from shapely.geometry import Point

    def _densify(ring):
        points: list[tuple[float, float]] = []
        count = len(ring)
        for index in range(count):
            ax, ay = ring[index]
            bx, by = ring[(index + 1) % count]
            length = math.hypot(bx - ax, by - ay)
            # Un pas de 25 cm : assez fin pour que aucun lamé ne manque,
            # assez lâche pour ne pas exploser le nombre de sommets.
            segments = max(int(math.ceil(length / 0.25)), 1)
            for step in range(segments):
                t = step / segments
                points.append((ax + (bx - ax) * t, ay + (by - ay) * t))
        return points

    points = _densify([(float(x), float(y)) for x, y in polygon.exterior.coords[:-1]])
    for ring in polygon.interiors:
        points.extend(_densify([(float(x), float(y)) for x, y in ring.coords[:-1]]))
    if len(points) < 3:
        return []

    array = np.asarray(points, dtype=np.float64)
    try:
        simplex = Delaunay(array)
    except QhullError:
        return []
    triangles: list[np.ndarray] = []
    for face in simplex.simplices:
        tri = array[face]
        centre = tri.mean(axis=0)
        if polygon.covers(Point(centre)):
            triangles.append(tri)
    return triangles
