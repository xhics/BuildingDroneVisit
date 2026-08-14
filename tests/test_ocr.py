"""Lecture d'enseigne et provenance complète (collecte V2, étape 4).

Ce qui est éprouvé : l'OCR ne lit que des fichiers acquis, ne suppose aucune
langue, refuse un fichier modifié depuis sa mesure, et n'écrase jamais un
verdict humain. Chaque lecture porte de quoi être rejouée.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone

import pytest

from hotel_pipeline import ocr
from hotel_pipeline.ocr import OcrRefused, apply, read_asset, run
from hotel_pipeline.schemas import (
    Asset,
    Blinding,
    PropertyMatchStatus,
    ReviewDecision,
    ReviewEntry,
)
from hotel_pipeline.schemas.assets import DECISION_STATUS, VISIBILITY_OF

EXPECTED = ["Hôtel Test", "HotelTest"]
EXCLUDED = ["Hôtel Concurrent"]


def image_file(tmp_path, name: str = "vue.jpg") -> tuple:
    """Un vrai JPEG texturé, et son empreinte."""
    import hashlib

    from PIL import Image

    picture = Image.new("RGB", (32, 32), (10, 120, 200))
    for x in range(32):
        for y in range(32):
            if (x // 8 + y // 8) % 2:
                picture.putpixel((x, y), (240, 30, 60))
    buffer = io.BytesIO()
    picture.save(buffer, format="JPEG")
    payload = buffer.getvalue()

    path = tmp_path / name
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def asset(tmp_path, history=(), **overrides) -> Asset:
    path, digest = image_file(tmp_path, overrides.pop("name", "vue.jpg"))
    history = list(history)
    fields = dict(
        id="mapillary-1", source="mapillary", source_url_or_id="1",
        rights="open_data", ai_eligible=False, confidence=0.5, category="facade",
        checksum=digest, local_path=str(path),
    )
    if history:
        last = history[-1]
        fields.update(
            review_history=history,
            target_visibility_decision=last.decision,
            review_status=DECISION_STATUS[last.decision],
            target_building_visible=VISIBILITY_OF[last.decision],
            reviewer=last.decided_by, review_rationale=last.rationale,
            review_evidence=last.evidence,
        )
    fields.update(overrides)
    return Asset(**fields)


def entry(decision=ReviewDecision.REJECTED) -> ReviewEntry:
    return ReviewEntry(
        decision=decision, decided_by="Claude (Opus 5)", rationale="motif",
        evidence=["preuve"], reviewed_checksum="a" * 64, blinding=Blinding.UNBLINDED,
    )


class Reader:
    """Lecteur d'essai : rend un texte fixé, sans modèle ni réseau."""

    def __init__(self, text: str = "HOTEL TEST — 1195") -> None:
        self.text = text
        self.calls: list[str] = []

    def read(self, path) -> str:  # noqa: ANN001
        self.calls.append(str(path))
        return self.text


# --- l'OCR vient après l'acquisition ------------------------------------------


def test_an_asset_without_a_file_is_never_read(tmp_path) -> None:
    """À la découverte, aucune image n'existe : lire n'aurait pas de sens."""
    reader = Reader()
    subject = asset(tmp_path, local_path=None)

    with pytest.raises(OcrRefused, match="aucun fichier local"):
        read_asset(subject, reader, ["fr"], EXPECTED, EXCLUDED)

    assert reader.calls == []


def test_a_missing_file_is_reported_not_guessed(tmp_path) -> None:
    subject = asset(tmp_path, local_path=str(tmp_path / "absente.jpg"))

    with pytest.raises(OcrRefused, match="fichier absent"):
        read_asset(subject, Reader(), ["fr"], EXPECTED, EXCLUDED)


def test_a_file_changed_since_acquisition_is_refused(tmp_path) -> None:
    """La lecture porterait sur autre chose que ce qui a été mesuré."""
    subject = asset(tmp_path)
    (tmp_path / "vue.jpg").write_bytes(b"remplace")

    with pytest.raises(OcrRefused, match="a changé depuis son acquisition"):
        read_asset(subject, Reader(), ["fr"], EXPECTED, EXCLUDED)


# --- aucune langue n'est supposée ---------------------------------------------


def test_no_language_is_assumed(tmp_path) -> None:
    """« fr, en » était le repli du pilote, pas une propriété du monde."""
    reader = Reader()

    with pytest.raises(OcrRefused, match="aucune langue"):
        read_asset(asset(tmp_path), reader, [], EXPECTED, EXCLUDED)

    assert reader.calls == []


def test_the_batch_refuses_before_reading_anything(tmp_path) -> None:
    reader = Reader()

    with pytest.raises(OcrRefused, match="ne se supposent pas"):
        run([asset(tmp_path)], reader, [], EXPECTED, EXCLUDED)

    assert reader.calls == []


# --- chaque lecture se rejoue -------------------------------------------------


def test_a_reading_carries_everything_needed_to_replay_it(tmp_path) -> None:
    subject = asset(tmp_path)

    reading = read_asset(
        subject, Reader(), ["fr", "en"], EXPECTED, EXCLUDED,
        engine="easyocr", engine_version="1.7.1",
    )

    published = reading.as_dict()
    assert published["engine"] == "easyocr"
    assert published["engine_version"] == "1.7.1"
    assert published["languages"] == ["fr", "en"]
    assert published["file_digest"] == subject.checksum
    assert datetime.fromisoformat(published["read_at"]).tzinfo is not None


def test_the_verdict_is_derived_from_the_text_not_stated(tmp_path) -> None:
    """Le recalculer depuis le texte et le profil doit rendre le même verdict."""
    from hotel_pipeline.triage import evaluate

    reading = read_asset(
        asset(tmp_path), Reader("HOTEL TEST"), ["fr"], EXPECTED, EXCLUDED
    )

    assert reading.status is PropertyMatchStatus.MATCH
    assert evaluate(reading.text, EXPECTED, EXCLUDED).status is reading.status


def test_a_competitor_sign_disqualifies_the_view(tmp_path) -> None:
    reading = read_asset(
        asset(tmp_path), Reader("HÔTEL CONCURRENT"), ["fr"], EXPECTED, EXCLUDED
    )

    assert reading.status is PropertyMatchStatus.MISMATCH
    assert reading.matched_term == "Hôtel Concurrent"


def test_an_unreadable_sign_stays_uncertain(tmp_path) -> None:
    reading = read_asset(asset(tmp_path), Reader(""), ["fr"], EXPECTED, EXCLUDED)

    assert reading.status is PropertyMatchStatus.UNCERTAIN


# --- une lecture n'écrase jamais un verdict humain ----------------------------


def test_a_human_review_survives_the_reading(tmp_path) -> None:
    """L'OCR informe l'appartenance ; la revue la tranche."""
    reviewed = asset(tmp_path, [entry(ReviewDecision.REJECTED)])
    reading = read_asset(reviewed, Reader("HOTEL TEST"), ["fr"], EXPECTED, EXCLUDED)

    updated = apply(reviewed, reading)

    # Le texte lu est conservé — il documente — mais le verdict ne bouge pas.
    assert updated.sign_text == "HOTEL TEST"
    assert updated.property_match_status is reviewed.property_match_status
    assert updated.review_status is reviewed.review_status


def test_an_unreviewed_asset_receives_the_reading_verdict(tmp_path) -> None:
    subject = asset(tmp_path)
    reading = read_asset(subject, Reader("HOTEL TEST"), ["fr"], EXPECTED, EXCLUDED)

    updated = apply(subject, reading)

    assert updated.property_match_status is PropertyMatchStatus.MATCH


# --- le rapport dit ce qu'une enseigne n'établit pas --------------------------


def test_the_report_separates_read_from_skipped(tmp_path) -> None:
    readable = asset(tmp_path, name="a.jpg")
    unreadable = asset(tmp_path, name="b.jpg", id="mapillary-2", local_path=None)

    _, readings, report = run(
        [readable, unreadable], Reader(), ["fr"], EXPECTED, EXCLUDED
    )

    assert report.read == 1
    assert len(readings) == 1
    assert "mapillary-2" in report.skipped


def test_the_report_states_what_a_sign_does_not_establish(tmp_path) -> None:
    _, _, report = run([asset(tmp_path)], Reader(), ["fr"], EXPECTED, EXCLUDED)

    assert "jamais une visibilité" in report.as_dict()["note"]


def test_the_report_names_the_engine_that_read(tmp_path) -> None:
    _, _, report = run(
        [asset(tmp_path)], Reader(), ["fr"], EXPECTED, EXCLUDED,
        engine="easyocr", engine_version="1.7.1",
    )

    published = report.as_dict()
    assert published["engine"] == "easyocr"
    assert published["languages"] == ["fr"]


def test_no_property_name_lives_in_the_module(tmp_path) -> None:
    """Les noms réels vivent au profil, jamais dans la logique."""
    import inspect

    source = inspect.getsource(ocr).lower()

    for name in ("welcominns", "mortagne", "tetra", "isomed"):
        assert name not in source
