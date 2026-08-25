"""Sélection automatique du point de vue du viewer.

Le viewer ne doit jamais cadrer un bâtiment au hasard : l'azimut de caméra
est choisi par un algorithme à partir de la géométrie mesurée et de la
*couverture photographique observée* de chaque face. On maximise la surface
de façade visible qui porte réellement une texture (et non un proxy), tout en
favorisant légèrement la face d'entrée pour la vue « héro ».

Tout est dérivé des données du payload ; aucune valeur magique (ex. 210°)
n'est écrite en dur. En l'absence de couverture mesurée, on retombe sur la
face d'entrée / principale déduite de la grammaire de façade.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

MIN_FACADE_EDGE_M = 8.0
ENTRANCE_WEIGHT = 1.6
SWEEP_STEP_DEG = 0.5


@dataclass(frozen=True)
class _Edge:
    index: int
    length: float
    bearing_deg: float
    coverage: float


def _signed_area(ring: list[list[float]]) -> float:
    return 0.5 * sum(
        ring[i][0] * ring[(i + 1) % len(ring)][1]
        - ring[(i + 1) % len(ring)][0] * ring[i][1]
        for i in range(len(ring))
    )


def _edges(ring: list[list[float]]) -> list[_Edge]:
    ccw = _signed_area(ring) > 0
    found: list[_Edge] = []
    for index, raw_a in enumerate(ring):
        raw_b = ring[(index + 1) % len(ring)]
        a = (float(raw_a[0]), float(raw_a[1]))
        b = (float(raw_b[0]), float(raw_b[1]))
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        tangent = (dx / length, dy / length)
        outward = (tangent[1], -tangent[0]) if ccw else (-tangent[1], tangent[0])
        bearing = math.degrees(math.atan2(outward[0], outward[1])) % 360.0
        found.append(_Edge(index, length, bearing, 0.0))
    return found


def _coverage_by_edge(payload: dict) -> dict[int, float]:
    fusion = payload.get("reference_fusion") or {}
    coverage: dict[int, float] = {}
    for texture in fusion.get("textures") or []:
        edge_index = int(texture.get("edge_index", -1))
        observed = float(texture.get("observed_fraction") or 0.0)
        disagreement = float(texture.get("disagreement_fraction") or 0.0)
        if edge_index < 0 or observed <= 0.0:
            continue
        # Une face en désaccord inter-vues est moins fiable à montrer.
        coverage[edge_index] = observed * (1.0 - min(0.5, disagreement))
    return coverage


def _geometry(payload: dict) -> tuple[list[_Edge], float, float, float]:
    target = next(
        (volume for volume in payload.get("volumes") or [] if volume.get("target")),
        None,
    )
    target = target or (payload.get("volumes") or [None])[0]
    ring = (target or {}).get("fp") or []
    height = float((target or {}).get("h") or 9.5)
    xs = [float(point[0]) for point in ring if len(point) >= 2]
    ys = [float(point[1]) for point in ring if len(point) >= 2]
    if xs and ys:
        centre_x = (min(xs) + max(xs)) / 2.0
        centre_y = (min(ys) + max(ys)) / 2.0
        diagonal = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        target_distance = max(35.0, min(220.0, diagonal * 1.15))
    else:
        centre_x = centre_y = 0.0
        target_distance = 150.0
    edges = _edges(ring)
    coverage = _coverage_by_edge(payload)
    for edge in edges:
        object.__setattr__(edge, "coverage", coverage.get(edge.index, 0.0))
    return edges, height, target_distance, target_distance, (centre_x, centre_y)


def _grammar_priority_edges(payload: dict) -> list[int]:
    grammar = payload.get("facade_grammar") or {}
    priority: list[int] = []
    for key in ("entrance_tower_edge_index", "main_edge_index"):
        value = grammar.get(key)
        if isinstance(value, int) and value >= 0:
            priority.append(value)
    priority.extend(grammar.get("facade_edges") or [])
    # dédupliquer en conservant l'ordre
    seen: set[int] = set()
    return [edge for edge in priority if not (edge in seen or seen.add(edge))]


def _visibility(theta_deg: float, bearing_deg: float, power: float = 1.4) -> float:
    delta = math.radians(((bearing_deg - theta_deg + 180.0) % 360.0) - 180.0)
    cos = math.cos(delta)
    return max(0.0, cos) ** power


def _score(theta_deg: float, edges: list[_Edge], priority: list[int]) -> float:
    priority_set = set(priority[:2])
    total = 0.0
    for edge in edges:
        if edge.length < MIN_FACADE_EDGE_M and edge.index not in priority_set:
            continue
        visible = _visibility(theta_deg, edge.bearing_deg)
        if visible <= 0.0:
            continue
        weight = edge.coverage
        if edge.index in priority_set:
            weight += ENTRANCE_WEIGHT * max(0.15, edge.coverage)
        total += edge.length * weight * visible
    return total


def _best_azimuth(edges: list[_Edge], priority: list[int]) -> float:
    best_theta = 0.0
    best_score = -1.0
    theta = 0.0
    while theta < 360.0:
        value = _score(theta, edges, priority)
        if value > best_score:
            best_score = value
            best_theta = theta
        theta += SWEEP_STEP_DEG
    return best_theta


def optimal_camera(payload: dict) -> dict:
    """Caméra dérivée de la couverture mesurée et de la grammaire de façade.

    Renvoie un dictionnaire ``camera`` complet et traçable. En l'absence de
    couverture photographique, l'azimut retombe sur la face d'entrée / principale.
    """
    edges, height, target_distance, _span, (centre_x, centre_y) = _geometry(payload)
    priority = _grammar_priority_edges(payload)

    if edges and any(edge.coverage > 0.0 for edge in edges):
        azimuth = _best_azimuth(edges, priority)
        source = "measured_coverage_optimization"
    elif priority:
        # Aucune texture : cadrer la face d'entrée déduite de la grammaire.
        by_index = {edge.index: edge for edge in edges}
        leading = next((by_index.get(index) for index in priority if index in by_index), None)
        azimuth = leading.bearing_deg if leading else 0.0
        source = "facade_grammar_entrance"
    else:
        azimuth = 0.0
        source = "target_building_bounds"

    focus = [round(centre_x, 3), round(centre_y, 3), round(height * 0.30, 3)]
    altitude = max(14.0, min(32.0, math.degrees(math.atan2(height * 0.7, target_distance))))
    facade_altitude = max(2.0, min(8.0, math.degrees(math.atan2(height * 0.18, target_distance * 0.5))))
    return {
        "focus": focus,
        "target_distance_m": round(target_distance, 3),
        "context_distance_m": round(max(150.0, target_distance * 2.4), 3),
        "azimuth_deg": round(azimuth, 1),
        "altitude_deg": round(altitude, 1),
        "facade_azimuth_deg": round(azimuth, 1),
        "facade_altitude_deg": round(facade_altitude, 1),
        "source": source,
    }


__all__ = ["optimal_camera"]
