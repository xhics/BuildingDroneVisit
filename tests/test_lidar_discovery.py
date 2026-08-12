"""Découverte des tuiles LiDAR (Lot 1B §9).

Trois exigences, toutes issues de contraintes réelles du service québécois :

- l'index complet pèse 159 Mo en GPKG, le WFS répond en kilo-octets ;
- une boîte englobante ne prouve pas une couverture ;
- un échec TLS est une panne, pas une absence de donnée.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import requests

from hotel_pipeline.geo import CoverageState, discover
from hotel_pipeline.geo import lidar

#: Empreinte du WelcomINNS, en WGS84.
FOOTPRINT = (
    "POLYGON ((-73.44380 45.57355, -73.44280 45.57355, "
    "-73.44280 45.57445, -73.44380 45.57445, -73.44380 45.57355))"
)

#: Tuile réellement retournée par le WFS pour ce site.
TILE_URL = (
    "https://diffusion.mern.gouv.qc.ca/diffusion/RGQ/Lidar/"
    "2023_Cmm_Lidar_Den15_DonneesClassifiees/Mtm8/Laz/23_3095048F08_DC.LAZ"
)
TILE_BYTES = 226_499_879


def feature(coords, **properties):
    base = {
        "tuile": "23_3095048F08_DC",
        "projet": "2023_CMM_LiDAR",
        "date_acquisition": "2023-11-21",
        "densite": 15,
        "classification": "1, 2, 6, 7",
        "format": "LAZ 1.4",
        "crs": "EPSG:2950",
        "referentiel_vertical": "CGVD1928",
        "licence": "CC BY 4.0",
        "taille": "216 Mo",
        "url": TILE_URL,
    }
    base.update(properties)
    return {
        "type": "Feature",
        "properties": base,
        "geometry": {"type": "Polygon", "coordinates": [coords]},
    }


COVERING = [(-73.4450, 45.5730), (-73.4420, 45.5730), (-73.4420, 45.5750),
            (-73.4450, 45.5750), (-73.4450, 45.5730)]
NEIGHBOURING = [(-73.4400, 45.5730), (-73.4380, 45.5730), (-73.4380, 45.5750),
                (-73.4400, 45.5750), (-73.4400, 45.5730)]


@pytest.fixture
def online(monkeypatch):
    monkeypatch.setenv("HOTEL_PIPELINE_OFFLINE", "0")


def stub_wfs(monkeypatch, features):
    monkeypatch.setattr(lidar, "_query_wfs", lambda bbox, url=lidar.WFS_URL: {"features": features})


def stub_size(monkeypatch, size=TILE_BYTES):
    monkeypatch.setattr(lidar, "exact_size", lambda url: size)


class TestCoverageIsIntersectionNotBoundingBox:
    def test_a_tile_covering_the_footprint_confirms_coverage(self, online, monkeypatch):
        stub_wfs(monkeypatch, [feature(COVERING)])
        stub_size(monkeypatch)
        result = discover(FOOTPRINT)
        assert result.state is CoverageState.COVERED
        assert len(result.tiles) == 1

    def test_a_neighbouring_tile_does_not_confirm_coverage(self, online, monkeypatch):
        """Le WFS filtre sur une boîte élargie : la voisine y entre sans toucher."""
        stub_wfs(monkeypatch, [feature(NEIGHBOURING)])
        result = discover(FOOTPRINT)
        assert result.state is CoverageState.NOT_COVERED
        assert result.considered == 1
        assert result.tiles == []

    def test_only_intersecting_tiles_are_kept(self, online, monkeypatch):
        stub_wfs(monkeypatch, [feature(NEIGHBOURING), feature(COVERING)])
        stub_size(monkeypatch)
        result = discover(FOOTPRINT)
        assert result.considered == 2
        assert len(result.tiles) == 1

    def test_no_feature_means_not_covered(self, online, monkeypatch):
        stub_wfs(monkeypatch, [])
        assert discover(FOOTPRINT).state is CoverageState.NOT_COVERED


class TestTlsFailureIsNotAnAbsence:
    def test_tls_error_yields_discovery_error(self, online, monkeypatch):
        """Lire un échec TLS comme « pas de LiDAR ici » serait une faute."""
        def explode(bbox, url=lidar.WFS_URL):
            raise requests.exceptions.SSLError("certificate verify failed")

        monkeypatch.setattr(lidar, "_query_wfs", explode)
        result = discover(FOOTPRINT)
        assert result.state is CoverageState.DISCOVERY_ERROR
        assert result.state is not CoverageState.NOT_COVERED
        assert "TLS" in result.error

    def test_network_error_also_yields_discovery_error(self, online, monkeypatch):
        def explode(bbox, url=lidar.WFS_URL):
            raise requests.ConnectionError("hôte injoignable")

        monkeypatch.setattr(lidar, "_query_wfs", explode)
        assert discover(FOOTPRINT).state is CoverageState.DISCOVERY_ERROR

    def test_tls_verification_is_never_disabled(self):
        """Aucun appel ne doit passer `verify=False`, même en dépannage."""
        source = Path(lidar.__file__).read_text("utf-8")
        assert "verify=False" not in source
        assert "verify = False" not in source

    def test_no_module_disables_tls_verification(self):
        root = Path(lidar.__file__).parent.parent
        offenders = [
            path.name
            for path in root.rglob("*.py")
            if "verify=False" in path.read_text("utf-8")
        ]
        assert offenders == []


class TestTileMetadata:
    def test_fields_are_parsed_from_the_service(self, online, monkeypatch):
        stub_wfs(monkeypatch, [feature(COVERING)])
        stub_size(monkeypatch)
        tile = discover(FOOTPRINT).tiles[0]

        assert tile.tile_id == "23_3095048F08_DC"
        assert tile.project == "2023_CMM_LiDAR"
        assert tile.acquired_on == date(2023, 11, 21)
        assert tile.point_density_per_m2 == 15.0
        assert tile.crs_horizontal == "EPSG:2950"
        assert tile.crs_vertical == "CGVD1928"
        assert tile.licence == "CC BY 4.0"

    def test_exact_size_supersedes_the_announced_volume(self, online, monkeypatch):
        """Le consentement se demande sur l'exact : un arrondi n'engage rien."""
        stub_wfs(monkeypatch, [feature(COVERING)])
        stub_size(monkeypatch)
        result = discover(FOOTPRINT)
        assert result.tiles[0].announced_size == "216 Mo"
        assert result.total_bytes == TILE_BYTES

    def test_missing_size_does_not_break_discovery(self, online, monkeypatch):
        stub_wfs(monkeypatch, [feature(COVERING)])
        monkeypatch.setattr(
            lidar, "exact_size", lambda url: (_ for _ in ()).throw(requests.ConnectionError())
        )
        result = discover(FOOTPRINT)
        assert result.state is CoverageState.COVERED
        assert result.total_bytes == 0

    def test_a_feature_without_url_is_ignored(self, online, monkeypatch):
        stub_wfs(monkeypatch, [feature(COVERING, url=None)])
        assert discover(FOOTPRINT).state is CoverageState.NOT_COVERED


class TestQueryShape:
    def test_bbox_is_widened_around_the_footprint(self):
        bbox = lidar.bbox_around(FOOTPRINT, margin_deg=0.001)
        miny, minx, maxy, maxx, srs = bbox.split(",")
        assert srs == "EPSG:4326"
        assert float(miny) < 45.57355
        assert float(maxx) > -73.44280

    def test_discovery_never_downloads_a_laz(self, online, monkeypatch):
        """Seules des métadonnées et une requête HEAD sont émises."""
        calls: list[str] = []
        monkeypatch.setattr(
            lidar, "_query_wfs", lambda bbox, url=lidar.WFS_URL: calls.append("wfs")
            or {"features": [feature(COVERING)]}
        )
        monkeypatch.setattr(lidar, "exact_size", lambda url: calls.append("head") or TILE_BYTES)

        discover(FOOTPRINT)
        assert calls == ["wfs", "head"]

    def test_sizes_can_be_skipped_entirely(self, online, monkeypatch):
        stub_wfs(monkeypatch, [feature(COVERING)])
        monkeypatch.setattr(
            lidar, "exact_size", lambda url: pytest.fail("aucune requête HEAD attendue")
        )
        result = discover(FOOTPRINT, measure_sizes=False)
        assert result.state is CoverageState.COVERED


class TestRealServiceResponse:
    """Fixture reprenant la réponse réelle du service québécois.

    Mes fixtures précédentes reproduisaient mes suppositions : elles passaient
    toutes alors que la découverte réelle rendait `not_covered` avec zéro
    entité. Deux défauts s'y cachaient — un ordre d'axes inversé et des noms
    d'attributs inconnus.
    """

    @pytest.fixture
    def payload(self) -> dict:
        import json

        path = Path(__file__).parent / "fixtures" / "lidar_wfs_boucherville.json"
        return json.loads(path.read_text("utf-8"))

    @pytest.fixture
    def discovered(self, online, monkeypatch, payload):
        monkeypatch.setattr(lidar, "_query_wfs", lambda bbox, url=lidar.WFS_URL: payload)
        monkeypatch.setattr(
            lidar, "exact_size", lambda url: TILE_BYTES if "3095048F08" in url else 1
        )
        return discover(FOOTPRINT)

    def test_four_tiles_are_considered(self, discovered):
        assert discovered.considered == 4

    def test_exactly_one_tile_intersects_the_footprint(self, discovered):
        """Trois voisines entrent dans la boîte élargie sans toucher le bâtiment."""
        assert len(discovered.tiles) == 1
        assert discovered.tiles[0].tile_id == "23_3095048F08_DC"

    def test_coverage_is_confirmed(self, discovered):
        assert discovered.state is CoverageState.COVERED

    def test_download_url_is_read_from_the_real_attribute(self, discovered):
        """`TELECHARGEMENT_TUILE` n'était pas reconnu : to_tile rendait None."""
        assert discovered.tiles[0].url.endswith("23_3095048F08_DC.LAZ")

    def test_density_is_extracted_from_a_unit_bearing_string(self, discovered):
        """« 15 pts/m2 » cassait la conversion directe en float."""
        assert discovered.tiles[0].point_density_per_m2 == 15.0

    def test_epsg_code_is_normalised(self, discovered):
        """Un nombre nu n'est pas un référentiel."""
        assert discovered.tiles[0].crs_horizontal == "EPSG:2950"

    def test_vertical_datum_is_preserved(self, discovered):
        assert discovered.tiles[0].crs_vertical == "CGVD 1928"

    def test_exact_size_is_measured_on_the_retained_tile_only(self, discovered):
        assert discovered.total_bytes == TILE_BYTES

    def test_announced_and_exact_sizes_both_survive(self, discovered):
        tile = discovered.tiles[0]
        assert tile.announced_size == "216 Mo"
        assert tile.exact_size_bytes == TILE_BYTES


class TestBboxAxisOrder:
    def test_longitude_comes_first(self):
        """Ce GeoServer attend lon,lat — l'ordre inverse rend zéro entité."""
        bbox = lidar.bbox_around(FOOTPRINT, margin_deg=0.001)
        first, second, third, fourth, srs = bbox.split(",")

        assert float(first) < -73.0   # longitude
        assert 45.0 < float(second) < 46.0  # latitude
        assert float(third) < -73.0
        assert 45.0 < float(fourth) < 46.0
        assert srs == "EPSG:4326"

    def test_bbox_widens_in_both_axes(self):
        tight = lidar.bbox_around(FOOTPRINT, margin_deg=0.0)
        wide = lidar.bbox_around(FOOTPRINT, margin_deg=0.01)
        assert float(wide.split(",")[0]) < float(tight.split(",")[0])
        assert float(wide.split(",")[1]) < float(tight.split(",")[1])
