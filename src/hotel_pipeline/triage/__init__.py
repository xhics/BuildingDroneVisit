"""Tri assisté des médias (plan directeur §11)."""

from .dedup import group_duplicates, phash
from .quality import QualityIssues, basic_scores, normalised_quality
from .sign_ocr import SignReading, evaluate, normalise

__all__ = [
    "QualityIssues",
    "SignReading",
    "basic_scores",
    "evaluate",
    "group_duplicates",
    "normalise",
    "normalised_quality",
    "phash",
]
