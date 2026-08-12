"""Lecture d'enseigne par OCR (plan directeur §4, §14).

C'est la brique la plus rentable du tri : lire « WelcomINNS » sur une photo
confirme automatiquement `property_match_status`, et lire « Mortagne »
disqualifie l'image. Le risque nº1 du §3 — confondre l'hôtel avec son voisin —
devient ainsi mesurable au lieu d'être supposé.

Google Cloud Vision est utilisé pour cela seul ; le tri par catégorie revient
à OpenCLIP, gratuit et local.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from ..logging import get_logger
from ..schemas import PropertyMatchStatus

log = get_logger("sign-ocr")


def normalise(text: str) -> str:
    """Minuscules sans accents ni ponctuation, pour comparer des enseignes."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", stripped).strip()


@dataclass
class SignReading:
    text: str
    status: PropertyMatchStatus
    matched_term: str | None = None


def evaluate(
    text: str, expected_terms: list[str], excluded_terms: list[str]
) -> SignReading:
    """Confronte un texte lu aux termes attendus et exclus.

    Séparé de l'appel réseau : c'est la logique de décision, et elle se teste
    sans clé ni service.
    """
    haystack = normalise(text)

    for term in excluded_terms:
        needle = normalise(term)
        if needle and needle in haystack:
            return SignReading(text, PropertyMatchStatus.MISMATCH, term)

    for term in expected_terms:
        needle = normalise(term)
        if needle and needle in haystack:
            return SignReading(text, PropertyMatchStatus.MATCH, term)

    return SignReading(text, PropertyMatchStatus.UNCERTAIN)


class LocalReader:
    """OCR local par EasyOCR, sans clé ni service.

    EasyOCR vise le texte en scène — enseignes, angles, éclairage variable —
    là où Tesseract vise le document scanné. Le modèle est chargé une seule
    fois, l'initialisation étant coûteuse.
    """

    def __init__(self, languages: tuple[str, ...] = ("fr", "en")) -> None:
        import easyocr

        log.info("chargement d'EasyOCR (%s)", ", ".join(languages))
        self._reader = easyocr.Reader(list(languages), gpu=False, verbose=False)

    def read(self, image_path: Path) -> str:
        results = self._reader.readtext(str(image_path), detail=0)
        return " ".join(results)


def read_text_vision(image_path: Path) -> str:
    """OCR par Google Cloud Vision — repli si l'OCR local est indisponible."""
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_path.read_bytes())
    response = client.text_detection(image=image)

    if response.error.message:
        raise RuntimeError(f"Vision API : {response.error.message}")

    annotations = response.text_annotations
    return annotations[0].description if annotations else ""


def get_reader():
    """Retourne un lecteur OCR, local de préférence.

    L'OCR local suffit à cet usage et évite une dépendance facturée ; Vision
    n'est qu'un repli.
    """
    try:
        return LocalReader()
    except ImportError:
        log.warning("EasyOCR absent — repli sur Google Cloud Vision")

        class _VisionReader:
            def read(self, image_path: Path) -> str:
                return read_text_vision(image_path)

        return _VisionReader()
