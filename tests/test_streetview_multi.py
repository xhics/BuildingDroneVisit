"""Street View multi-position (Lot 1B §7, §14).

L'étape 2 a chiffré le défaut du collecteur précédent : 8 fichiers pour un
seul point de vue. Ces tests verrouillent le comportement corrigé — des
positions indépendantes, un cap dirigé vers l'empreinte, et aucune fuite de
clé dans le manifeste.
"""

from __future__ import annotations

import pytest

from hotel_pipeline.collectors import streetview
from hotel_pipeline.collectors.streetview import Panorama, sample_road_network

BUILDING = (
    "POLYGON ((-73.44380 45.57355, -73.44280 45.57355, "
    "-73.44280 45.57445, -73.44380 45.57445, -73.44380 45.57355))"
)

#: Une voie est-ouest passant au sud du bâtiment.
ROAD = [
    {
        "type": "way",
        "id": 1,
        "tags": {"highway": "residential"},
        "geometry": [
            {"lat": 45.5730, "lon": -73.4450},
            {"lat": 45.5730, "lon": -73.4420},
        ],
    }
]


class TestSampling:
    def test_samples_are_spaced_in_metres(self):
        """L'espacement ne doit pas dépendre de la finesse de numérisation OSM."""
        samples = sample_road_network(ROAD, spacing_m=15.0)
        assert len(samples) > 5

        from hotel_pipeline.visibility import haversine_m

        gaps = [
            haversine_m(a[0], a[1], b[0], b[1]) for a, b in zip(samples, samples[1:])
        ]
        assert all(10 <= gap <= 20 for gap in gaps), gaps

    def test_denser_spacing_yields_more_samples(self):
        assert len(sample_road_network(ROAD, 10.0)) > len(sample_road_network(ROAD, 30.0))

    def test_empty_network_yields_no_samples(self):
        assert sample_road_network([]) == []

    def test_degenerate_segment_is_skipped(self):
        road = [{"geometry": [{"lat": 45.57, "lon": -73.44}, {"lat": 45.57, "lon": -73.44}]}]
        assert sample_road_network(road) == []


class TestPanoramaDeduplication:
    """Plusieurs points d'échantillonnage tombent sur un même panorama."""

    @pytest.fixture
    def collector(self, monkeypatch):
        monkeypatch.setenv("HOTEL_PIPELINE_OFFLINE", "0")
        monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "clef-de-test-secrete")
        return streetview

    def test_repeated_panorama_counted_once(self, collector, monkeypatch):
        monkeypatch.setattr(
            collector,
            "panorama_at",
            lambda lat, lon, key, radius_m=25: Panorama("PANO-A", 45.5730, -73.4433, "2025-06", "©"),
        )
        images = collector.collect(45.5741, -73.4433, BUILDING, road_elements=ROAD)
        assert len(images) == 1

    def test_distinct_panoramas_are_all_kept(self, collector, monkeypatch):
        counter = {"n": 0}

        def fake(lat, lon, key, radius_m=25):
            counter["n"] += 1
            return Panorama(f"PANO-{counter['n']}", lat, lon, "2025-06", "©")

        monkeypatch.setattr(collector, "panorama_at", fake)
        images = collector.collect(45.5741, -73.4433, BUILDING, road_elements=ROAD)
        assert len({i.source_id for i in images}) == len(images) > 1

    def test_missing_panorama_is_skipped(self, collector, monkeypatch):
        monkeypatch.setattr(collector, "panorama_at", lambda *a, **k: None)
        assert collector.collect(45.5741, -73.4433, BUILDING, road_elements=ROAD) == []


class TestHeadingAndFraming:
    @pytest.fixture
    def collector(self, monkeypatch):
        monkeypatch.setenv("HOTEL_PIPELINE_OFFLINE", "0")
        monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "clef-de-test-secrete")
        return streetview

    def test_heading_points_at_the_building(self, collector, monkeypatch):
        """Depuis le sud, la caméra doit regarder vers le nord."""
        monkeypatch.setattr(
            collector,
            "panorama_at",
            lambda *a, **k: Panorama("PANO-S", 45.5725, -73.4433, "2025-06", "©"),
        )
        images = collector.collect(45.5741, -73.4433, BUILDING, road_elements=ROAD)
        assert images[0].heading_deg == pytest.approx(0, abs=15)

    def test_far_panorama_is_rejected(self, collector, monkeypatch):
        monkeypatch.setattr(
            collector,
            "panorama_at",
            lambda *a, **k: Panorama("PANO-LOIN", 45.5900, -73.4433, "2025-06", "©"),
        )
        assert collector.collect(45.5741, -73.4433, BUILDING, road_elements=ROAD) == []

    def test_capture_date_is_preserved(self, collector, monkeypatch):
        """L'historique ne doit pas être confondu avec l'entrée actuelle."""
        monkeypatch.setattr(
            collector,
            "panorama_at",
            lambda *a, **k: Panorama("PANO-A", 45.5730, -73.4433, "2019-08", "©"),
        )
        image = collector.collect(45.5741, -73.4433, BUILDING, road_elements=ROAD)[0]
        assert image.captured_year == 2019
        assert image.extra["date"] == "2019-08"


class TestSecrets:
    def test_api_key_never_enters_the_manifest_url(self, monkeypatch):
        monkeypatch.setenv("HOTEL_PIPELINE_OFFLINE", "0")
        monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "clef-de-test-secrete")
        monkeypatch.setattr(
            streetview,
            "panorama_at",
            lambda *a, **k: Panorama("PANO-A", 45.5730, -73.4433, "2025-06", "©"),
        )
        image = streetview.collect(45.5741, -73.4433, BUILDING, road_elements=ROAD)[0]

        assert "clef-de-test-secrete" not in image.url
        assert "key=" not in image.url
        # La clé n'apparaît qu'au moment du téléchargement.
        assert "clef-de-test-secrete" in streetview.sign_url(image)
