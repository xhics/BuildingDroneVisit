"""Graphe de toiture : ce que les arêtes disent ensemble."""

from __future__ import annotations

import numpy as np
import pytest

from hotel_pipeline.geo.ridge_graph import (
    JUNCTION_TOLERANCE_M,
    MIN_EDGE_LENGTH_M,
    PARALLEL_TOLERANCE_DEG,
    build,
    consistent_pairs,
)


class _Ridge:
    def __init__(self, start, end, kind="faitage", planes=(0, 1)):
        self.start = np.asarray(start, dtype=float)
        self.end = np.asarray(end, dtype=float)
        self.kind = kind
        self.plane_indices = planes

    @property
    def length_m(self) -> float:
        return float(np.linalg.norm(self.end - self.start))


def _L(x0, y0, x1, y1, z=10.0, planes=(0, 1)):
    return _Ridge((x0, y0, z), (x1, y1, z), planes=planes)


class TestConstruction:
    def test_two_meeting_ridges_share_a_node(self):
        graph = build([_L(0, 0, 20, 0), _L(20, 0, 20, 20)])
        assert graph.node_count if hasattr(graph, "node_count") else len(graph.nodes) == 3
        assert graph.edges[0].adjacent_to == [1]

    def test_disjoint_ridges_do_not_touch(self):
        graph = build([_L(0, 0, 20, 0), _L(100, 100, 120, 100)])
        assert graph.edges[0].adjacent_to == []
        assert len(graph.components()) == 2

    def test_close_endpoints_are_merged(self):
        """Deux ajustements distincts ne se rejoignent jamais au millimètre."""
        gap = JUNCTION_TOLERANCE_M * 0.5
        graph = build([_L(0, 0, 20, 0), _L(20 + gap, 0, 20, 20)])
        assert len(graph.nodes) == 3

    def test_short_ridges_are_excluded(self):
        tiny = MIN_EDGE_LENGTH_M * 0.5
        graph = build([_L(0, 0, tiny, 0), _L(0, 0, 20, 0)])
        assert len(graph.edges) == 1

    def test_an_empty_input_yields_an_empty_graph(self):
        graph = build([])
        assert graph.edges == []
        assert graph.nodes == []


class TestRelations:
    def test_parallel_ridges_are_linked_as_such(self):
        graph = build([_L(0, 0, 20, 0), _L(0, 30, 20, 30)])
        assert graph.edges[1].index in graph.edges[0].parallel_to

    def test_perpendicular_ridges_are_linked_as_such(self):
        graph = build([_L(0, 0, 20, 0), _L(50, 0, 50, 20)])
        assert graph.edges[1].index in graph.edges[0].perpendicular_to

    def test_a_ridge_is_not_related_to_itself(self):
        graph = build([_L(0, 0, 20, 0)])
        assert graph.edges[0].parallel_to == []
        assert graph.edges[0].adjacent_to == []

    def test_shared_planes_make_ridges_coplanar(self):
        graph = build(
            [_L(0, 0, 20, 0, planes=(0, 1)), _L(0, 40, 20, 40, planes=(1, 2))]
        )
        assert graph.edges[1].index in graph.edges[0].coplanar_with

    def test_unrelated_planes_are_not_coplanar(self):
        graph = build(
            [_L(0, 0, 20, 0, planes=(0, 1)), _L(0, 40, 20, 40, planes=(5, 6))]
        )
        assert graph.edges[0].coplanar_with == []

    def test_parallel_tolerance_is_narrow(self):
        assert 0.0 < PARALLEL_TOLERANCE_DEG < 45.0


class TestDistinctiveness:
    def test_symmetric_ridges_share_a_signature(self):
        """Deux arêtes interchangeables ne doivent pas passer pour uniques."""
        graph = build([_L(0, 0, 20, 0), _L(0, 40, 20, 40)])
        assert graph.distinctive() == []

    def test_a_ridge_with_its_own_neighbourhood_is_distinctive(self):
        graph = build(
            [
                _L(0, 0, 20, 0),
                _L(20, 0, 20, 20),
                _L(20, 20, 40, 20),
                _L(100, 0, 130, 0),
            ]
        )
        assert graph.distinctive()

    def test_junctions_are_counted(self):
        graph = build([_L(0, 0, 20, 0), _L(20, 0, 20, 20), _L(20, 0, 40, 10)])
        assert graph.junctions


class TestConsistency:
    def _square(self):
        return build([_L(0, 0, 20, 0), _L(20, 0, 20, 20)])

    def test_neighbouring_segments_that_meet_are_consistent(self):
        graph = self._square()
        matches = {0: (100.0, 100.0, 300.0, 100.0), 1: (300.0, 100.0, 300.0, 260.0)}
        assert all(consistent_pairs(graph, matches).values())

    def test_neighbouring_segments_far_apart_are_refused(self):
        """Deux arêtes qui se rejoignent en 3D ne peuvent pas être aux antipodes."""
        graph = self._square()
        matches = {0: (100.0, 100.0, 300.0, 100.0), 1: (900.0, 700.0, 900.0, 860.0)}
        assert not all(consistent_pairs(graph, matches).values())

    def test_a_lone_match_is_not_contradicted(self):
        """L'absence de contrainte n'est pas une violation."""
        graph = self._square()
        assert consistent_pairs(graph, {0: (100.0, 100.0, 300.0, 100.0)})[0]

    def test_an_unknown_edge_is_refused(self):
        graph = self._square()
        assert consistent_pairs(graph, {99: (0.0, 0.0, 10.0, 0.0)})[99] is False


class TestReport:
    def test_report_serialises_with_caveats(self):
        payload = build([_L(0, 0, 20, 0), _L(20, 0, 20, 20)]).as_dict()
        assert payload["edge_count"] == 2
        assert payload["caveats"]
        assert any("n'en invente aucune" in c for c in payload["caveats"])

    def test_components_partition_the_edges(self):
        graph = build([_L(0, 0, 20, 0), _L(20, 0, 20, 20), _L(200, 200, 220, 200)])
        groups = graph.components()
        assert sum(len(g) for g in groups) == len(graph.edges)
