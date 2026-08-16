"""Télécharger sous plafond, vérifier, publier — ou rien (collecte V2).

Le consentement porte sur 133 030 octets exacts. Sans borne, un serveur servant
davantage remplirait le disque avant qu'on s'en aperçoive : `bytes_written`
était inscrit **après** l'écriture, donc trop tard pour refuser.

Ce que ces tests protègent : aucun cas d'échec ne doit laisser un asset publié
ni un fichier final partiel.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from hotel_pipeline.download import (
    Budget,
    DownloadRefused,
    check_dimensions,
    inspect,
    stream_to,
    verify,
)


def _jpeg(width=256, height=256) -> bytes:
    image = Image.new("RGB", (width, height), (200, 40, 60))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


class FakeStream:
    """Une réponse qui rend son corps par morceaux, comme le réseau."""

    def __init__(self, payload: bytes, chunk: int = 1024, declared=None):
        self.payload = payload
        self.chunk = chunk
        self.headers = {
            "Content-Length": str(
                declared if declared is not None else len(payload)
            )
        }
        self.status_code = 200
        self.consumed = 0

    def iter_content(self, _size):
        for start in range(0, len(self.payload), self.chunk):
            piece = self.payload[start:start + self.chunk]
            self.consumed += len(piece)
            yield piece

    def raise_for_status(self):
        return None


class FakeRequest:
    def __init__(self, candidate_id="c1", width=256, height=256):
        self.candidate_id = candidate_id
        self.digest = "d" * 16
        self.width_px = width
        self.height_px = height


# --- le plafond arrête le flux, il ne le constate pas -------------------------


def test_the_stream_stops_before_writing_the_overflowing_chunk(tmp_path) -> None:
    """Vérifier après l'écriture aurait déjà mis les octets sur le disque.

    « Refuser » ne serait alors plus qu'un constat.
    """
    payload = b"x" * 10_000
    target = tmp_path / "image.jpg"
    stream = FakeStream(payload, chunk=1000)

    with pytest.raises(DownloadRefused, match="dépassement"):
        stream_to(stream, target, ceiling=5_000, declared=10_000)

    # Le fichier ouvert existe, mais borné : c'est l'appelant qui le supprime.
    assert target.stat().st_size <= 5_000
    assert stream.consumed < len(payload), "le flux n'a pas été lu jusqu'au bout"


def test_a_body_shorter_than_declared_is_refused(tmp_path) -> None:
    """Le HEAD et le GET ne décriraient plus la même réponse."""
    target = tmp_path / "image.jpg"
    stream = FakeStream(b"y" * 500, declared=4096)

    with pytest.raises(DownloadRefused, match="ne décrivent pas la même"):
        stream_to(stream, target, ceiling=4096, declared=4096)


def test_a_body_matching_its_declaration_passes(tmp_path) -> None:
    """Sans quoi le plafond bloquerait aussi les cas conformes."""
    payload = _jpeg()
    target = tmp_path / "image.jpg"

    written = stream_to(
        FakeStream(payload), target, ceiling=len(payload), declared=len(payload)
    )

    assert written == len(payload)
    assert target.read_bytes() == payload


# --- deux plafonds simultanés --------------------------------------------------


def test_the_tighter_of_the_two_ceilings_applies() -> None:
    """Le plafond global seul laisserait un fichier consommer la part des
    autres."""
    budget = Budget(total_consented=133_030, per_request={"c1": 7_575})

    assert budget.ceiling_for("c1") == 7_575, "le plafond individuel est plus bas"

    budget.spent = 130_000
    assert budget.ceiling_for("c1") == 3_030, "le reste du lot est plus bas"


def test_an_unknown_individual_ceiling_falls_back_on_the_remainder() -> None:
    budget = Budget(total_consented=1_000, per_request={})

    assert budget.ceiling_for("inconnu") == 1_000
    budget.spent = 1_200
    assert budget.ceiling_for("inconnu") == 0, "jamais négatif"


# --- le format se décode, il ne se déduit pas ---------------------------------


def test_a_corrupt_file_is_refused(tmp_path) -> None:
    """Un serveur qui sert une page d'erreur en `.jpg` passerait sinon pour une
    image, et l'asset publierait un fichier illisible."""
    target = tmp_path / "image.jpg"
    target.write_bytes(b"<html>503 Service Unavailable</html>")

    with pytest.raises(DownloadRefused, match="non décodable"):
        inspect(target)


def test_a_truncated_image_is_refused(tmp_path) -> None:
    """Une image coupée en deux se décode parfois à moitié."""
    target = tmp_path / "image.jpg"
    target.write_bytes(_jpeg()[: len(_jpeg()) // 3])

    with pytest.raises(DownloadRefused):
        inspect(target)


def test_an_unaccepted_format_is_refused(tmp_path) -> None:
    """Le format est décodé, puis confronté à ce qui est accepté."""
    target = tmp_path / "image.gif"
    Image.new("RGB", (64, 64), (0, 0, 0)).save(target, format="GIF")

    with pytest.raises(DownloadRefused, match="hors des formats acceptés"):
        inspect(target)


def test_a_valid_image_yields_its_format_and_digest(tmp_path) -> None:
    target = tmp_path / "image.jpg"
    target.write_bytes(_jpeg(320, 240))

    fmt, width, height, digest = inspect(target)

    assert fmt == "JPEG"
    assert (width, height) == (320, 240)
    assert len(digest) == 64


# --- les dimensions suivent le contrat du fournisseur -------------------------


def test_street_view_must_match_exactly() -> None:
    """Il rend la taille demandée : autre chose est une autre image."""
    check_dimensions("street_view", FakeRequest(width=256, height=256), 256, 256)

    with pytest.raises(DownloadRefused, match="au lieu de"):
        check_dimensions("street_view", FakeRequest(width=256, height=256), 320, 240)


def test_mapillary_only_bounds_the_longest_side() -> None:
    """Sa miniature garde le rapport de l'original : exiger l'égalité stricte
    rejetterait des images conformes."""
    request = FakeRequest(width=256, height=256)

    check_dimensions("mapillary", request, 256, 192)
    check_dimensions("mapillary", request, 144, 256)

    with pytest.raises(DownloadRefused, match="plus grand côté"):
        check_dimensions("mapillary", request, 512, 384)


def test_an_undeclared_source_is_not_rejected_on_an_invented_rule() -> None:
    """On ne refuse pas sur une règle qu'on vient d'inventer."""
    check_dimensions("source-inconnue", FakeRequest(), 1234, 567)


# --- la vérification complète --------------------------------------------------


def test_verify_reports_what_it_measured(tmp_path) -> None:
    target = tmp_path / "image.jpg"
    payload = _jpeg(256, 256)
    target.write_bytes(payload)

    outcome = verify(target, "street_view", FakeRequest())

    assert outcome.image_format == "JPEG"
    assert (outcome.width, outcome.height) == (256, 256)
    assert outcome.bytes_staged == len(payload)
    assert outcome.request_digest == "d" * 16
    assert outcome.refused is None
