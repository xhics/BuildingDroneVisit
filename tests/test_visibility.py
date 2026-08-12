"""Le cliché regarde-t-il le bâtiment ? (§11, §14)

Sur le corpus réel du WelcomINNS, 140 images sont classées « extérieur » mais
6 seulement cadrent l'hôtel : l'imagerie de roulage longe la voie sans la
viser. C'est ce que ces tests verrouillent.
"""

from __future__ import annotations

import pytest

from hotel_pipeline.visibility import angular_difference, assess, bearing_deg

# Carré d'environ 100 m de côté, centré sur (45.5740, -73.4433).
BUILDING = (
    "POLYGON ((-73.44380 45.57355, -73.44280 45.57355, "
    "-73.44280 45.57445, -73.44380 45.57445, -73.44380 45.57355))"
)


class TestBearing:
    def test_due_north(self):
        assert bearing_deg(45.0, -73.0, 45.01, -73.0) == pytest.approx(0, abs=1)

    def test_due_east(self):
        assert bearing_deg(45.0, -73.0, 45.0, -72.99) == pytest.approx(90, abs=1)

    def test_due_south(self):
        assert bearing_deg(45.0, -73.0, 44.99, -73.0) == pytest.approx(180, abs=1)


class TestAngularDifference:
    def test_wraps_around_north(self):
        assert angular_difference(350, 10) == pytest.approx(20)

    def test_is_symmetric(self):
        assert angular_difference(10, 350) == angular_difference(350, 10)

    def test_opposite_is_180(self):
        assert angular_difference(0, 180) == pytest.approx(180)


class TestAssess:
    def test_camera_pointing_at_building_sees_it(self):
        """Caméra au sud, cap au nord : le bâtiment est dans le champ."""
        result = assess(45.5725, -73.4433, heading_deg=0.0, building_wkt=BUILDING)
        assert result.visible

    def test_camera_driving_past_does_not_see_it(self):
        """Le cas dominant du corpus réel : on longe la voie, cap perpendiculaire."""
        result = assess(45.5725, -73.4433, heading_deg=90.0, building_wkt=BUILDING)
        assert not result.visible
        assert "hors champ" in result.reason

    def test_camera_facing_away_does_not_see_it(self):
        result = assess(45.5725, -73.4433, heading_deg=180.0, building_wkt=BUILDING)
        assert not result.visible

    def test_too_far_is_rejected_whatever_the_heading(self):
        result = assess(45.5500, -73.4433, heading_deg=0.0, building_wkt=BUILDING)
        assert not result.visible
        assert "trop loin" in result.reason

    def test_unknown_heading_is_not_assumed_visible(self):
        """Sans cap, la proximité ne prouve rien — on ne tranche pas en sa faveur."""
        result = assess(45.5725, -73.4433, heading_deg=None, building_wkt=BUILDING)
        assert not result.visible
        assert "cap inconnu" in result.reason

    def test_distance_is_measured_to_the_footprint_not_the_centroid(self):
        """De près, la façade remplit l'image même si le centroïde est plus loin."""
        result = assess(45.5734, -73.4433, heading_deg=0.0, building_wkt=BUILDING)
        assert result.distance_m < 30
