"""Carte des observations : ce qui est triangulable, et ce qui manque."""

from __future__ import annotations

import math

import pytest

from hotel_pipeline.geo.observation_map import (
    MAX_INCIDENCE_DEG,
    MAX_TRIANGULATION_DEG,
    MAX_USEFUL_DISTANCE_M,
    MIN_TRIANGULATION_DEG,
    TARGET_VIEWS,
    build,
    recommend,
)


class _Sample:
    """Point de mur regardant vers le nord (normale +y)."""

    def __init__(self, x: float, y: float = 0.0, normal=(0.0, 1.0)):
        self.x = x
        self.y = y
        self.normal = normal


WALL = [_Sample(x) for x in (0.0, 5.0, 10.0)]
ALL = [0, 1, 2]


def _view(name: str, x: float, y: float, indices=None):
    return (name, (x, y), ALL if indices is None else indices)


class TestTriangulation:
    def test_two_well_separated_views_triangulate(self):
        # Vingt degrés d'écart : au-dessus du minimum, sous le plafond.
        found = build(WALL, [_view("A", -8.0, 30.0), _view("B", 18.0, 30.0)])
        assert all(c.triangulable for c in found.cells)

    def test_a_single_view_never_triangulates(self):
        found = build(WALL, [_view("A", 5.0, 30.0)])
        assert not any(c.triangulable for c in found.cells)
        assert found.cells[0].status == "vue_unique"

    def test_two_views_from_the_same_spot_are_one_observation(self):
        """Dix vues du même trottoir n'apportent aucune parallaxe."""
        found = build(WALL, [_view("A", 5.0, 30.0), _view("B", 5.2, 30.0)])
        assert not any(c.triangulable for c in found.cells)
        assert found.cells[1].status == "parallaxe_insuffisante"

    def test_widely_separated_views_do_not_match(self):
        """Au-delà du plafond, la façade change trop d'aspect pour s'apparier.

        Les deux vues restent d'incidence acceptable : c'est bien l'écart
        entre elles qu'on teste, non leur rasance.
        """
        found = build(WALL, [_view("A", -22.0, 18.0), _view("B", 32.0, 18.0)])
        assert found.cells[1].view_count == 2
        assert found.cells[1].status == "vues_trop_ecartees"

    def test_unseen_cell_is_reported_as_such(self):
        found = build(WALL, [("A", (5.0, 30.0), [0])])
        assert found.cells[2].status == "aucune_vue"
        assert found.cells[2].view_count == 0


class TestRejections:
    def test_a_view_too_far_is_ignored(self):
        far = MAX_USEFUL_DISTANCE_M + 50.0
        found = build(WALL, [_view("A", -20.0, far), _view("B", 30.0, far)])
        assert all(c.view_count == 0 for c in found.cells)

    def test_a_grazing_view_is_ignored(self):
        """Vue en enfilade : quelques pixels décrivent plusieurs mètres."""
        # Presque dans le plan du mur, donc incidence proche de 90°.
        found = build(WALL, [("A", (200.0, 0.5), ALL)])
        assert found.cells[1].view_count == 0

    def test_out_of_range_indices_are_skipped(self):
        found = build(WALL, [("A", (5.0, 30.0), [0, 99, -3])])
        assert found.cells[0].view_count == 1

    def test_empty_observations_leave_everything_unseen(self):
        found = build(WALL, [])
        assert all(c.status == "aucune_vue" for c in found.cells)


class TestMargin:
    def test_two_views_are_triangulable_without_margin(self):
        found = build(WALL, [_view("A", -8.0, 30.0), _view("B", 18.0, 30.0)])
        assert found.cells[1].status == "triangulable_sans_marge"

    def test_the_target_count_grants_full_status(self):
        views = [_view("A", -8.0, 30.0), _view("B", 5.0, 30.0), _view("C", 18.0, 30.0)]
        assert len(views) >= TARGET_VIEWS
        found = build(WALL, views)
        assert found.cells[1].status == "observe"


class TestRecommendations:
    def test_gaps_produce_a_direction_to_shoot_from(self):
        found = recommend(build(WALL, []))
        assert found.missing
        first = found.missing[0]
        assert 0.0 <= first.bearing_deg < 360.0
        assert first.cells_gained > 0

    def test_a_complete_wall_needs_nothing(self):
        views = [_view("A", -8.0, 30.0), _view("B", 5.0, 30.0), _view("C", 18.0, 30.0)]
        assert recommend(build(WALL, views)).missing == []

    def test_recommended_position_faces_the_wall(self):
        """On se place dans l'axe du mur, là où l'incidence est nulle."""
        found = recommend(build(WALL, []))
        # Le mur regarde vers +y : la position doit être du côté +y.
        assert found.missing[0].y > 0

    def test_recommendations_are_ranked_by_gain(self):
        found = recommend(build(WALL, []))
        gains = [m.cells_gained for m in found.missing]
        assert gains == sorted(gains, reverse=True)


class TestReport:
    def test_report_serialises_with_caveats(self):
        payload = recommend(build(WALL, [_view("A", 5.0, 30.0)])).as_dict()
        assert payload["cell_count"] == 3
        assert payload["caveats"]
        assert "by_facade" in payload

    def test_facade_summary_counts_what_it_says(self):
        found = build(WALL, [("A", (5.0, 30.0), [0])], facade_id="MUR")
        summary = found.by_facade()["MUR"]
        assert summary["total"] == 3
        assert summary["unseen"] == 2

    def test_thresholds_are_coherent(self):
        assert 0.0 < MIN_TRIANGULATION_DEG < MAX_TRIANGULATION_DEG < 180.0
        assert 0.0 < MAX_INCIDENCE_DEG < 90.0
        assert MAX_USEFUL_DISTANCE_M > 0.0


class TestConnectivity:
    """Une prise qui ne recouvre rien forme un îlot que le solve ignore."""

    def _ring(self, count: int = 12, radius: float = 15.0):
        """Un bâtiment rond : chaque cellule regarde vers l'extérieur."""
        cells = []
        for index in range(count):
            angle = 2 * math.pi * index / count
            cells.append(
                _Sample(
                    radius * math.cos(angle),
                    radius * math.sin(angle),
                    (math.cos(angle), math.sin(angle)),
                )
            )
        return cells

    def test_a_recommendation_reports_what_it_shares(self):
        ring = self._ring()
        # Une vue existante couvre un côté.
        found = recommend(build(ring, [("A", (60.0, 0.0), [0, 1, 11])]))
        assert found.missing
        assert all(hasattr(m, "shared_with_existing") for m in found.missing)

    def test_an_isolated_sector_is_flagged_or_bridged(self):
        """Soit on le rattache, soit on le déclare isolé — jamais en silence."""
        ring = self._ring()
        found = recommend(build(ring, []))
        for entry in found.missing:
            # Chaque prise dit explicitement si elle se rattache.
            assert isinstance(entry.connected, bool)

    def test_links_are_reciprocal(self):
        """Si B se rattache à A, A doit se rattacher à B."""
        ring = self._ring()
        found = recommend(build(ring, []))
        for slot, entry in enumerate(found.missing):
            for other in entry.linked_to:
                if other < len(found.missing):
                    assert slot in found.missing[other].linked_to or (
                        "liaison" in found.missing[other].reason
                    )

    def test_a_bridge_names_itself(self):
        ring = self._ring()
        found = recommend(build(ring, [("A", (60.0, 0.0), [0])]))
        bridges = [m for m in found.missing if "liaison" in m.reason]
        for bridge in bridges:
            # Une liaison doit toucher quelque chose des deux côtés.
            assert bridge.linked_to or bridge.shared_with_existing > 0

    def test_serialised_recommendation_carries_connectivity(self):
        ring = self._ring()
        payload = recommend(build(ring, [("A", (60.0, 0.0), [0, 1])])).as_dict()
        for entry in payload["missing"]:
            assert "connected" in entry
            assert "shared_with_existing" in entry
            assert "linked_to" in entry

    def test_reach_stays_within_the_measured_limits(self):
        """La portée d'une prise applique les seuils de `build`, pas les siens."""
        from hotel_pipeline.geo.observation_map import (
            REACH_HALF_ANGLE_DEG,
            _cells_reached,
        )

        ring = self._ring()
        found = build(ring, [])
        # Depuis très loin, aucune cellule n'est atteinte.
        assert _cells_reached(found.cells, (5000.0, 0.0), 0.0, 30.0) == set()
        assert 0.0 < REACH_HALF_ANGLE_DEG < 180.0
