"""Association des arêtes de toiture aux segments détectés."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hotel_pipeline.geo.ridge_match import (
    AMBIGUITY_RATIO,
    ANGLE_TOLERANCE_DEG,
    BASE_RADIUS_PX,
    MIN_SEGMENT_PX,
    RidgeMatchReport,
    RidgeProjection,
    match_one,
    project_ridge,
    search_radius_px,
)


def _projection(
    start=(100.0, 200.0), end=(400.0, 205.0), distance=40.0, in_frame=True
) -> RidgeProjection:
    return RidgeProjection(
        ridge_index=0,
        asset_id="A",
        start_px=start,
        end_px=end,
        distance_m=distance,
        in_frame=in_frame,
    )


class _Camera:
    """Caméra pinhole regardant vers +x."""

    width, height = 1280, 720

    def __init__(self, position=(0.0, 0.0, 2.5)):
        self.position = np.asarray(position, dtype=float)
        self.f = 900.0

    def project(self, points):
        d = np.asarray(points, dtype=float) - self.position
        z = d[:, 0]
        if np.any(z <= 0.5):
            return None, z
        return (
            np.c_[
                self.width / 2 + self.f * d[:, 1] / z,
                self.height / 2 - self.f * d[:, 2] / z,
            ],
            z,
        )


class TestSearchRadius:
    def test_radius_grows_with_pose_uncertainty(self):
        tight = search_radius_px(_projection(), 0.5, 900.0)
        loose = search_radius_px(_projection(), 8.0, 900.0)
        assert loose > tight

    def test_radius_shrinks_with_distance(self):
        near = search_radius_px(_projection(distance=10.0), 3.0, 900.0)
        far = search_radius_px(_projection(distance=100.0), 3.0, 900.0)
        assert near > far

    def test_radius_never_falls_below_the_base(self):
        assert search_radius_px(_projection(distance=9999.0), 0.0, 900.0) >= (
            BASE_RADIUS_PX
        )

    def test_degenerate_distance_is_handled(self):
        assert search_radius_px(_projection(distance=0.0), 3.0, 900.0) > 0


class TestProjection:
    def test_a_ridge_in_front_projects(self):
        found = project_ridge(
            np.array([30.0, -5.0, 12.0]), np.array([30.0, 5.0, 12.0]),
            _Camera(), 0, "A",
        )
        assert found is not None
        assert found.in_frame

    def test_a_ridge_behind_the_camera_yields_nothing(self):
        assert project_ridge(
            np.array([-30.0, 0.0, 12.0]), np.array([-30.0, 5.0, 12.0]),
            _Camera(), 0, "A",
        ) is None

    def test_distance_is_measured_to_the_middle(self):
        found = project_ridge(
            np.array([40.0, 0.0, 2.5]), np.array([60.0, 0.0, 2.5]),
            _Camera(), 0, "A",
        )
        assert found.distance_m == pytest.approx(50.0, abs=0.5)


class TestMatching:
    def test_an_aligned_segment_is_retained(self):
        segments = [(100.0, 200.0, 400.0, 205.0)]
        found = match_one(_projection(), segments)
        assert found.matched
        assert found.offset_px < 2.0

    def test_a_parallel_but_offset_segment_is_refused(self):
        """Un segment de même orientation, mais ailleurs, ne décrit pas l'arête."""
        segments = [(100.0, 600.0, 400.0, 605.0)]
        assert not match_one(_projection(), segments).matched

    def test_a_crossing_segment_is_refused(self):
        segments = [(250.0, 50.0, 255.0, 400.0)]
        found = match_one(_projection(), segments)
        assert not found.matched
        assert "aucun segment" in found.reason

    def test_two_equivalent_candidates_are_ambiguous(self):
        """Deux segments de coût voisin ne tranchent rien.

        Aucun ne coïncide avec l'arête : ils l'encadrent symétriquement, ce
        qui est le cas gênant — une corniche et son ombre, deux niveaux de
        bardage.
        """
        segments = [
            (100.0, 194.0, 400.0, 199.0),
            (100.0, 206.0, 400.0, 211.0),
        ]
        found = match_one(_projection(), segments)
        assert found.ambiguous
        assert not found.matched

    def test_an_exact_match_is_not_ambiguous(self):
        """Un candidat parfait tranche, même si un second est proche."""
        segments = [
            (100.0, 200.0, 400.0, 205.0),
            (100.0, 208.0, 400.0, 213.0),
        ]
        found = match_one(_projection(), segments)
        assert found.matched
        assert not found.ambiguous

    def test_a_ridge_out_of_frame_is_not_attempted(self):
        found = match_one(_projection(in_frame=False), [(0.0, 0.0, 300.0, 5.0)])
        assert not found.matched
        assert "hors du cadre" in found.reason

    def test_a_ridge_too_short_on_screen_is_not_attempted(self):
        """Le cas dominant sur ce corpus : quelques pixels de long."""
        tiny = _projection(start=(100.0, 200.0), end=(108.0, 201.0))
        found = match_one(tiny, [(100.0, 200.0, 108.0, 201.0)])
        assert not found.matched
        assert "trop courte" in found.reason

    def test_a_short_segment_cannot_describe_a_long_ridge(self):
        segments = [(100.0, 200.0, 130.0, 200.5)]
        assert not match_one(_projection(), segments).matched

    def test_no_segments_at_all_is_reported(self):
        found = match_one(_projection(), [])
        assert not found.matched
        assert found.cost is None


class TestReport:
    def _match(self, ridge, asset, matched=True):
        from hotel_pipeline.geo.ridge_match import RidgeMatch

        return RidgeMatch(
            ridge_index=ridge,
            asset_id=asset,
            segment=(0.0, 0.0, 10.0, 0.0) if matched else None,
            cost=0.1 if matched else None,
        )

    def test_a_ridge_seen_twice_is_counted(self):
        report = RidgeMatchReport(
            matches=[self._match(0, "A"), self._match(0, "B"), self._match(1, "A")]
        )
        payload = report.as_dict()
        assert payload["ridges_found"] == 2
        assert payload["ridges_in_two_or_more"] == 1

    def test_report_carries_its_caveats(self):
        payload = RidgeMatchReport().as_dict()
        assert payload["caveats"]
        # La limite qui compte : une arête vue une fois ne contraint rien.
        assert any("une seule image" in c for c in payload["caveats"])

    def test_thresholds_are_coherent(self):
        assert 0.0 < ANGLE_TOLERANCE_DEG < 90.0
        assert BASE_RADIUS_PX > 0.0
        assert MIN_SEGMENT_PX > 0.0
        assert 0.0 < AMBIGUITY_RATIO <= 1.0


class TestDistanceFamilies:
    """Un segment vertical ne décrit pas un faîtage, où qu'il passe."""

    def test_compatible_segments_are_kept_apart(self):
        from hotel_pipeline.geo.ridge_match import distance_families

        families = distance_families(
            [(0.0, 0.0, 100.0, 2.0), (0.0, 0.0, 2.0, 100.0)], 0.0
        )
        assert len(families["compatible"]) == 1
        assert len(families["ecarte"]) == 1

    def test_oblique_segments_form_their_own_family(self):
        from hotel_pipeline.geo.ridge_match import (
            ANGLE_TOLERANCE_DEG,
            HORIZONTAL_BAND_DEG,
            distance_families,
        )

        middle = (ANGLE_TOLERANCE_DEG + HORIZONTAL_BAND_DEG) / 2
        import math

        length = 100.0
        segment = (
            0.0, 0.0,
            length * math.cos(math.radians(middle)),
            length * math.sin(math.radians(middle)),
        )
        families = distance_families([segment], 0.0)
        assert families["oblique"] and not families["compatible"]

    def test_every_segment_lands_in_exactly_one_family(self):
        from hotel_pipeline.geo.ridge_match import distance_families

        segments = [(0.0, 0.0, 100.0, float(k)) for k in range(0, 120, 20)]
        families = distance_families(segments, 0.0)
        assert sum(len(v) for v in families.values()) == len(segments)


class TestDisambiguation:
    """Le départage vient du voisinage, non de la proximité."""

    def _graph(self):
        import numpy as np

        from hotel_pipeline.geo.ridge_graph import build

        class _R:
            def __init__(self, a, b):
                self.start = np.array(a, dtype=float)
                self.end = np.array(b, dtype=float)
                self.kind = "faitage"
                self.plane_indices = (0, 1)

            @property
            def length_m(self):
                return float(np.linalg.norm(self.end - self.start))

        return build([_R((0, 0, 10), (20, 0, 10)), _R((20, 0, 10), (20, 20, 10))])

    def _match(self, ridge, segment, alternatives, ambiguous=True):
        from hotel_pipeline.geo.ridge_match import RidgeMatch

        return RidgeMatch(
            ridge_index=ridge,
            asset_id="A",
            segment=segment,
            cost=0.5,
            ambiguous=ambiguous,
            alternatives=alternatives,
        )

    def test_a_consistent_alternative_is_adopted(self):
        from hotel_pipeline.geo.ridge_graph import consistent_pairs
        from hotel_pipeline.geo.ridge_match import disambiguate

        graph = self._graph()
        anchor = self._match(1, (300.0, 100.0, 300.0, 260.0), [], ambiguous=False)
        # Le premier candidat est loin de l'ancre, le second la rejoint.
        candidate = self._match(
            0, (900.0, 700.0, 1100.0, 700.0), [(100.0, 100.0, 300.0, 100.0)]
        )
        resolved = disambiguate([anchor, candidate], graph, consistent_pairs)
        assert resolved == 1
        assert not candidate.ambiguous
        assert "topologie" in candidate.reason

    def test_without_alternatives_nothing_is_forced(self):
        from hotel_pipeline.geo.ridge_graph import consistent_pairs
        from hotel_pipeline.geo.ridge_match import disambiguate

        lonely = self._match(0, (0.0, 0.0, 10.0, 0.0), [])
        assert disambiguate([lonely], self._graph(), consistent_pairs) == 0
        assert lonely.ambiguous

    def test_a_settled_match_is_left_alone(self):
        from hotel_pipeline.geo.ridge_graph import consistent_pairs
        from hotel_pipeline.geo.ridge_match import disambiguate

        settled = self._match(0, (0.0, 0.0, 10.0, 0.0), [(1.0, 1.0, 11.0, 1.0)], False)
        assert disambiguate([settled], self._graph(), consistent_pairs) == 0
        assert settled.segment == (0.0, 0.0, 10.0, 0.0)
