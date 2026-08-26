"""Le toit comme graphe : ce que les arêtes disent ensemble, non séparément.

Une arête de toiture prise isolément discrimine mal. Mesuré sur ce pilote, les
dix arêtes exploitables couvrent vingt-deux degrés d'étendue angulaire — neuf
d'entre elles sont pratiquement parallèles. Beaucoup de bâtiments possèdent une
longue ligne horizontale ; peu possèdent le **même agencement** d'arêtes.

Ce module transforme donc les arêtes mesurées en graphe : les nœuds sont leurs
extrémités et leurs intersections, les liens sont les arêtes elles-mêmes, et
les relations — partager un pan, se rejoindre, être parallèles — portent la
signature du toit.

**Pourquoi c'est plus fort qu'une ligne isolée.** Apparier une arête à un
segment détecté laisse le choix entre plusieurs candidats plausibles : sur ce
pilote, quatre-vingt-six associations restent ambiguës faute de départage. Un
sous-graphe, lui, impose des contraintes croisées : si deux arêtes se
rejoignent en 3D, leurs segments doivent se rejoindre à l'image. Une
correspondance qui satisfait la longueur mais viole l'incidence est écartée
sans avoir à choisir un seuil de plus.

**Ce que le graphe ne fait pas.** Il ne crée aucune arête : il décrit celles
que la segmentation a mesurées. Un toit dont la segmentation a manqué un pan
aura un graphe incomplet, et le module le dit plutôt que de combler.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..logging import get_logger

log = get_logger("geo-ridge-graph")

#: Distance, en mètres, en deçà de laquelle deux extrémités d'arêtes sont
#: tenues pour le même point du toit. Les arêtes viennent d'intersections de
#: plans ajustés : leurs extrémités ne coïncident jamais exactement.
JUNCTION_TOLERANCE_M = 1.5

#: Écart angulaire, en degrés, sous lequel deux arêtes sont dites parallèles.
PARALLEL_TOLERANCE_DEG = 8.0

#: Écart à quatre-vingt-dix degrés toléré pour qualifier une perpendicularité.
PERPENDICULAR_TOLERANCE_DEG = 12.0

#: Longueur minimale, en mètres, d'une arête retenue dans le graphe. En deçà,
#: son orientation est dominée par le bruit d'ajustement des plans.
MIN_EDGE_LENGTH_M = 3.0


@dataclass
class RidgeNode:
    """Un point du toit où des arêtes se rencontrent, ou s'arrêtent."""

    index: int
    position: np.ndarray
    edges: list[int] = field(default_factory=list)

    @property
    def degree(self) -> int:
        return len(self.edges)

    @property
    def kind(self) -> str:
        """Ce que ce nœud est, selon ce qui y aboutit."""
        if self.degree >= 3:
            return "carrefour"
        if self.degree == 2:
            return "jonction"
        return "extremite"

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "position": [round(float(v), 2) for v in self.position],
            "degree": self.degree,
            "kind": self.kind,
            "edges": self.edges,
        }


@dataclass
class RidgeEdge:
    """Une arête du graphe, et ce qui la relie aux autres."""

    index: int
    ridge_index: int
    node_a: int
    node_b: int
    length_m: float
    angle_deg: float
    kind: str
    #: Arêtes partageant un pan de toiture avec celle-ci.
    coplanar_with: list[int] = field(default_factory=list)
    #: Arêtes qui la rejoignent en un nœud commun.
    adjacent_to: list[int] = field(default_factory=list)
    parallel_to: list[int] = field(default_factory=list)
    perpendicular_to: list[int] = field(default_factory=list)

    @property
    def signature(self) -> tuple:
        """Ce qui caractérise cette arête dans le graphe, hors sa position.

        Deux arêtes de même signature sont interchangeables : c'est ce qui
        rend une association ambiguë, et il vaut mieux le savoir que de
        trancher au hasard.
        """
        return (
            len(self.adjacent_to),
            len(self.coplanar_with),
            len(self.parallel_to),
            len(self.perpendicular_to),
            round(self.length_m / 5.0),
        )

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "ridge_index": self.ridge_index,
            "nodes": [self.node_a, self.node_b],
            "length_m": round(self.length_m, 2),
            "angle_deg": round(self.angle_deg, 1),
            "kind": self.kind,
            "adjacent_to": self.adjacent_to,
            "coplanar_with": self.coplanar_with,
            "parallel_to": self.parallel_to,
            "perpendicular_to": self.perpendicular_to,
            "signature": list(self.signature),
        }


@dataclass
class RidgeGraph:
    """Le toit décrit par ses arêtes et leurs relations."""

    nodes: list[RidgeNode] = field(default_factory=list)
    edges: list[RidgeEdge] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    @property
    def junctions(self) -> list[RidgeNode]:
        """Nœuds où plusieurs arêtes se rencontrent : les points saillants."""
        return [node for node in self.nodes if node.degree >= 2]

    def distinctive(self) -> list[RidgeEdge]:
        """Arêtes dont la signature est unique dans ce graphe.

        Ce sont elles qui identifient une pose sans ambiguïté : une arête
        partageant sa signature avec une autre ne dit pas laquelle des deux on
        regarde.
        """
        counts: dict[tuple, int] = {}
        for edge in self.edges:
            counts[edge.signature] = counts.get(edge.signature, 0) + 1
        return [edge for edge in self.edges if counts[edge.signature] == 1]

    def components(self) -> list[list[int]]:
        """Groupes d'arêtes reliées entre elles."""
        seen: set[int] = set()
        groups: list[list[int]] = []
        for edge in self.edges:
            if edge.index in seen:
                continue
            stack, group = [edge.index], []
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                group.append(current)
                stack.extend(
                    other
                    for other in self.edges[current].adjacent_to
                    if other not in seen
                )
            groups.append(sorted(group))
        return groups

    def as_dict(self) -> dict:
        groups = self.components()
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "junction_count": len(self.junctions),
            "distinctive_count": len(self.distinctive()),
            "component_count": len(groups),
            "largest_component": max((len(g) for g in groups), default=0),
            "nodes": [n.as_dict() for n in self.nodes],
            "edges": [e.as_dict() for e in self.edges],
            "provenance": self.provenance,
            "caveats": [
                "le graphe décrit les arêtes mesurées, il n'en invente aucune : "
                "un pan manqué par la segmentation laisse un graphe incomplet",
                "deux arêtes de même signature sont interchangeables — le "
                "graphe le signale au lieu de trancher",
            ],
        }


def _angle_between(first: float, second: float) -> float:
    """Écart entre deux orientations non orientées, dans [0, 90]."""
    gap = abs(first - second) % 180.0
    return min(gap, 180.0 - gap)


def _plane_angle(ridge) -> float:  # noqa: ANN001
    """Orientation d'une arête au sol, en degrés dans [0, 180)."""
    delta = np.asarray(ridge.end) - np.asarray(ridge.start)
    return math.degrees(math.atan2(float(delta[1]), float(delta[0]))) % 180.0


def build(ridges, tolerance_m: float = JUNCTION_TOLERANCE_M) -> RidgeGraph:
    """Construit le graphe depuis les arêtes vectorisées d'une toiture."""
    graph = RidgeGraph()
    kept = [
        (index, ridge)
        for index, ridge in enumerate(ridges)
        if ridge.length_m >= MIN_EDGE_LENGTH_M
    ]
    if not kept:
        log.info("aucune arête assez longue pour un graphe")
        return graph

    # Les extrémités proches sont fondues en un nœud : deux arêtes issues
    # d'ajustements distincts ne se rejoignent jamais au millimètre.
    positions: list[np.ndarray] = []

    def node_for(point: np.ndarray) -> int:
        for slot, existing in enumerate(positions):
            if float(np.linalg.norm(existing - point)) <= tolerance_m:
                return slot
        positions.append(np.asarray(point, dtype=np.float64))
        graph.nodes.append(
            RidgeNode(index=len(positions) - 1, position=positions[-1])
        )
        return len(positions) - 1

    for slot, (ridge_index, ridge) in enumerate(kept):
        a = node_for(np.asarray(ridge.start, dtype=np.float64))
        b = node_for(np.asarray(ridge.end, dtype=np.float64))
        edge = RidgeEdge(
            index=slot,
            ridge_index=ridge_index,
            node_a=a,
            node_b=b,
            length_m=float(ridge.length_m),
            angle_deg=_plane_angle(ridge),
            kind=str(ridge.kind),
        )
        graph.edges.append(edge)
        graph.nodes[a].edges.append(slot)
        if b != a:
            graph.nodes[b].edges.append(slot)

    # Relations. Chacune dit quelque chose de différent : partager un pan n'est
    # pas se toucher, et deux arêtes parallèles peuvent être aux antipodes.
    planes = {
        slot: set(getattr(ridge, "plane_indices", (-1, -1)))
        for slot, (_ri, ridge) in enumerate(kept)
    }
    for edge in graph.edges:
        for other in graph.edges:
            if other.index == edge.index:
                continue
            shared_nodes = {edge.node_a, edge.node_b} & {other.node_a, other.node_b}
            if shared_nodes:
                edge.adjacent_to.append(other.index)
            shared_planes = planes[edge.index] & planes[other.index] - {-1}
            if shared_planes:
                edge.coplanar_with.append(other.index)
            gap = _angle_between(edge.angle_deg, other.angle_deg)
            if gap <= PARALLEL_TOLERANCE_DEG:
                edge.parallel_to.append(other.index)
            elif abs(gap - 90.0) <= PERPENDICULAR_TOLERANCE_DEG:
                edge.perpendicular_to.append(other.index)

    graph.provenance = {
        "junction_tolerance_m": tolerance_m,
        "parallel_tolerance_deg": PARALLEL_TOLERANCE_DEG,
        "perpendicular_tolerance_deg": PERPENDICULAR_TOLERANCE_DEG,
        "min_edge_length_m": MIN_EDGE_LENGTH_M,
        "ridges_supplied": len(ridges),
        "ridges_kept": len(kept),
    }
    log.info(
        "graphe de toiture : %d arête(s), %d nœud(s), %d carrefour(s), "
        "%d arête(s) distinctive(s)",
        len(graph.edges),
        len(graph.nodes),
        sum(1 for n in graph.nodes if n.degree >= 3),
        len(graph.distinctive()),
    )
    return graph


def consistent_pairs(graph: RidgeGraph, matches: dict[int, tuple]) -> dict[int, bool]:
    """Vérifie qu'un jeu d'associations respecte la topologie du toit.

    `matches` associe un index d'arête au segment 2D retenu. Deux arêtes qui se
    rejoignent en 3D doivent avoir des segments qui se rejoignent à l'image :
    c'est la contrainte croisée que l'appariement arête par arête ne pose pas.

    Une arête sans voisine appariée est tenue pour cohérente : rien ne la
    contredit, et l'absence de contrainte n'est pas une violation.
    """
    verdicts: dict[int, bool] = {}
    for index, segment in matches.items():
        if index >= len(graph.edges):
            verdicts[index] = False
            continue
        edge = graph.edges[index]
        neighbours = [n for n in edge.adjacent_to if n in matches]
        if not neighbours:
            verdicts[index] = True
            continue

        ok = False
        for other in neighbours:
            partner = matches[other]
            # Les segments doivent partager un voisinage à l'image, comme
            # leurs arêtes partagent un nœud en 3D.
            ends = [
                (segment[0], segment[1]),
                (segment[2], segment[3]),
                (partner[0], partner[1]),
                (partner[2], partner[3]),
            ]
            closest = min(
                math.hypot(ends[i][0] - ends[j][0], ends[i][1] - ends[j][1])
                for i in (0, 1)
                for j in (2, 3)
            )
            span = max(
                math.hypot(segment[2] - segment[0], segment[3] - segment[1]),
                math.hypot(partner[2] - partner[0], partner[3] - partner[1]),
            )
            if closest <= max(span * 0.4, 30.0):
                ok = True
                break
        verdicts[index] = ok
    return verdicts


def validate_roof_graph(graph: RidgeGraph) -> dict:
    """Reject impossible crossings between ridges that share no junction."""
    from shapely.geometry import LineString
    crossings = []
    for i, first in enumerate(graph.edges):
        a0, a1 = graph.nodes[first.node_a].position, graph.nodes[first.node_b].position
        for second in graph.edges[i+1:]:
            if {first.node_a, first.node_b} & {second.node_a, second.node_b}:
                continue
            b0, b1 = graph.nodes[second.node_a].position, graph.nodes[second.node_b].position
            if LineString([a0[:2],a1[:2]]).crosses(LineString([b0[:2],b1[:2]])):
                crossings.append([first.index, second.index])
    return {"passed":not crossings, "impossible_crossings":crossings, "component_count":len(graph.components()), "support_required":"geometric_intersection+normal_compatibility+lidar_or_photo"}


__all__ = [
    "JUNCTION_TOLERANCE_M",
    "MIN_EDGE_LENGTH_M",
    "PARALLEL_TOLERANCE_DEG",
    "PERPENDICULAR_TOLERANCE_DEG",
    "RidgeEdge",
    "RidgeGraph",
    "RidgeNode",
    "build",
    "consistent_pairs",
    "validate_roof_graph",
]
