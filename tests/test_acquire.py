"""Protocole d'acquisition d'une tuile (Lot 1B §9).

Un téléchargement partiel portant le nom du fichier final est pire qu'un
échec : tout ce qui suit le croira valide. Le nommage n'intervient donc
qu'après la taille exacte, la signature et l'empreinte.
"""

from __future__ import annotations

import hashlib

import pytest
import requests

from hotel_pipeline.geo.acquire import (
    LAS_SIGNATURE,
    AcquisitionError,
    download_tile,
    provenance_from,
)
from hotel_pipeline.geo.lidar import TileCandidate

CONTENT = LAS_SIGNATURE + b"charge utile de test" * 8
SIZE = len(CONTENT)
DIGEST = hashlib.sha256(CONTENT).hexdigest()
URL = "https://diffusion.mern.gouv.qc.ca/…/23_3095048F08_DC.LAZ"


class FakeResponse:
    def __init__(self, content=CONTENT, headers=None, status=200):
        self._content = content
        self.headers = headers if headers is not None else {
            "Content-Length": str(len(content)),
            "ETag": '"abc-123"',
            "Last-Modified": "Tue, 21 Nov 2023 10:00:00 GMT",
        }
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f"{self.status}")

    def iter_content(self, size):
        for start in range(0, len(self._content), size):
            yield self._content[start : start + size]


@pytest.fixture
def online(monkeypatch):
    monkeypatch.setenv("HOTEL_PIPELINE_OFFLINE", "0")


def stub_get(monkeypatch, response):
    monkeypatch.setattr(
        "hotel_pipeline.geo.acquire.requests.get",
        lambda *a, **k: response,
    )


class TestSuccessfulAcquisition:
    def test_file_is_written_and_named(self, online, monkeypatch, tmp_path):
        stub_get(monkeypatch, FakeResponse())
        target = tmp_path / "tile.LAZ"
        result = download_tile(URL, target, SIZE)

        assert result.succeeded
        assert target.is_file()
        assert target.read_bytes() == CONTENT

    def test_digest_and_size_are_recorded(self, online, monkeypatch, tmp_path):
        stub_get(monkeypatch, FakeResponse())
        result = download_tile(URL, tmp_path / "tile.LAZ", SIZE)
        assert result.sha256 == DIGEST
        assert result.size_bytes == SIZE

    def test_http_headers_are_kept(self, online, monkeypatch, tmp_path):
        stub_get(monkeypatch, FakeResponse())
        result = download_tile(URL, tmp_path / "tile.LAZ", SIZE)
        assert result.headers["ETag"] == '"abc-123"'
        assert "Last-Modified" in result.headers

    def test_no_partial_file_remains(self, online, monkeypatch, tmp_path):
        stub_get(monkeypatch, FakeResponse())
        download_tile(URL, tmp_path / "tile.LAZ", SIZE)
        assert list(tmp_path.glob("*.part")) == []


class TestRefusalsLeaveNothingBehind:
    def test_short_download_is_refused(self, online, monkeypatch, tmp_path):
        stub_get(monkeypatch, FakeResponse(content=CONTENT[:10], headers={}))
        target = tmp_path / "tile.LAZ"
        result = download_tile(URL, target, SIZE)

        assert not result.succeeded
        assert "incomplet" in result.error
        assert not target.exists()

    def test_wrong_announced_length_is_refused_before_writing(
        self, online, monkeypatch, tmp_path
    ):
        """Un volume différent de celui autorisé n'est pas téléchargé."""
        stub_get(monkeypatch, FakeResponse(headers={"Content-Length": "999999"}))
        result = download_tile(URL, tmp_path / "tile.LAZ", SIZE)
        assert "taille annoncée" in result.error

    def test_wrong_signature_is_refused(self, online, monkeypatch, tmp_path):
        payload = b"<html>404</html>" + b"x" * 100
        stub_get(monkeypatch, FakeResponse(content=payload, headers={}))
        target = tmp_path / "tile.LAZ"
        result = download_tile(URL, target, len(payload))

        assert "LAS/LAZ" in result.error
        assert not target.exists()

    def test_partial_file_is_removed_on_failure(self, online, monkeypatch, tmp_path):
        stub_get(monkeypatch, FakeResponse(content=CONTENT[:10], headers={}))
        download_tile(URL, tmp_path / "tile.LAZ", SIZE)
        assert list(tmp_path.glob("*.part")) == []

    def test_network_error_is_reported_not_raised(self, online, monkeypatch, tmp_path):
        def explode(*a, **k):
            raise requests.ConnectionError("coupure")

        monkeypatch.setattr("hotel_pipeline.geo.acquire.requests.get", explode)
        result = download_tile(URL, tmp_path / "tile.LAZ", SIZE)
        assert not result.succeeded
        assert "coupure" in result.error


class TestProvenanceOnlyFromSuccess:
    @pytest.fixture
    def tile(self) -> TileCandidate:
        from datetime import date

        return TileCandidate(
            tile_id="23_3095048F08_DC",
            url=URL,
            acquired_on=date(2023, 11, 21),
            point_density_per_m2=15.0,
            crs_horizontal="EPSG:2950",
            crs_vertical="CGVD 1928",
            licence="CC BY 4.0",
            classification="1,2,6,7",
            file_format="LAZ 1.4",
        )

    def test_successful_acquisition_yields_a_citable_source(
        self, online, monkeypatch, tmp_path, tile
    ):
        stub_get(monkeypatch, FakeResponse())
        result = download_tile(URL, tmp_path / "tile.LAZ", SIZE)
        provenance = provenance_from(result, tile)

        assert provenance.is_citable() == []
        assert provenance.file_digest == DIGEST
        assert provenance.crs_vertical == "CGVD 1928"
        assert provenance.vintage == "2023"

    def test_failed_acquisition_yields_no_source(self, online, monkeypatch, tmp_path, tile):
        """Une source citable sans fichier rendrait toute dérivation invérifiable."""
        stub_get(monkeypatch, FakeResponse(content=b"xx", headers={}))
        result = download_tile(URL, tmp_path / "tile.LAZ", SIZE)

        with pytest.raises(AcquisitionError, match="échouée"):
            provenance_from(result, tile)

    def test_provenance_declares_it_carries_elevation(
        self, online, monkeypatch, tmp_path, tile
    ):
        stub_get(monkeypatch, FakeResponse())
        result = download_tile(URL, tmp_path / "tile.LAZ", SIZE)
        assert provenance_from(result, tile).carries_elevation is True
