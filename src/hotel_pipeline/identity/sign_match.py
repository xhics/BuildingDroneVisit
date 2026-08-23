"""Appariement tolérant d'un nom d'enseigne dans un texte d'OCR.

`triage.sign_ocr.evaluate` exige le terme sur des limites de mots, ce qui est
la bonne règle pour un texte propre. Une enseigne photographiée de loin ne
donne pas un texte propre : mesuré sur ce pilote, l'OCR rend « TETRA 1205
TECH » — le numéro civique s'insère entre les deux mots — et « ecokn » pour
WELCOMINNS sur une vue à contre-jour.

Deux tolérances, chacune motivée par un de ces cas :

1. **les intrus sont ignorés** : les termes du nom recherché doivent
   apparaître dans l'ordre, pas nécessairement côte à côte ;
2. **une approximation est admise** sur un mot long, pour absorber les lettres
   qu'un OCR confond, sans jamais rapprocher deux noms réellement distincts.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from ..triage.sign_ocr import normalise

#: En deçà, deux mots ne sont pas des lectures du même mot. Réglé haut : une
#: enseigne mal lue reste plus proche d'elle-même que d'un nom concurrent.
FUZZY_RATIO = 0.78

#: Un mot court se déforme trop vite pour être apparié de façon approchée.
MIN_FUZZY_LENGTH = 6


@dataclass
class SignMatch:
    matched: bool
    term: str | None
    method: str
    score: float


def _tokens(text: str) -> list[str]:
    return [t for t in normalise(text).split() if t]


def _subsequence(haystack: list[str], needle: list[str]) -> bool:
    """Les mots recherchés apparaissent-ils dans l'ordre, intrus admis ?"""
    position = 0
    for word in needle:
        while position < len(haystack) and haystack[position] != word:
            position += 1
        if position == len(haystack):
            return False
        position += 1
    return True


def _fuzzy_hit(haystack: list[str], word: str) -> tuple[bool, float]:
    best = 0.0
    for candidate in haystack:
        if abs(len(candidate) - len(word)) > max(3, len(word) // 3):
            continue
        ratio = SequenceMatcher(None, candidate, word).ratio()
        best = max(best, ratio)
    return best >= FUZZY_RATIO, best


def find_term(text: str, term: str) -> SignMatch:
    """Cherche un nom d'enseigne dans un texte d'OCR, avec tolérance."""
    haystack = _tokens(text)
    needle = _tokens(term)
    if not haystack or not needle:
        return SignMatch(False, None, "empty", 0.0)

    joined = " ".join(haystack)
    if f" {' '.join(needle)} " in f" {joined} ":
        return SignMatch(True, term, "exact", 1.0)

    if len(needle) > 1 and _subsequence(haystack, needle):
        return SignMatch(True, term, "subsequence", 0.9)

    # Un nom d'un seul mot, assez long pour rester reconnaissable dégradé.
    if len(needle) == 1 and len(needle[0]) >= MIN_FUZZY_LENGTH:
        hit, score = _fuzzy_hit(haystack, needle[0])
        if hit:
            return SignMatch(True, term, "fuzzy", score)

    # Nom composé dont chaque mot long est retrouvé approximativement.
    if len(needle) > 1:
        scores = []
        for word in needle:
            if len(word) < MIN_FUZZY_LENGTH:
                continue
            hit, score = _fuzzy_hit(haystack, word)
            if not hit:
                scores = []
                break
            scores.append(score)
        if scores:
            return SignMatch(True, term, "fuzzy_all", sum(scores) / len(scores))

    return SignMatch(False, None, "absent", 0.0)


def evaluate(
    text: str, expected_terms: list[str], excluded_terms: list[str]
) -> tuple[str, str | None, str]:
    """Statut d'appartenance d'après l'enseigne lue.

    L'attendu l'emporte sur l'exclu, comme dans `triage.sign_ocr` : une image
    portant le nom de l'établissement lui appartient, quels que soient les
    autres commerces visibles dans le cadre.
    """
    for term in expected_terms:
        hit = find_term(text, term)
        if hit.matched:
            return "match", term, hit.method
    for term in excluded_terms:
        hit = find_term(text, term)
        if hit.matched:
            return "mismatch", term, hit.method
    return "uncertain", None, "absent"

#: Écart maximal, en unités, entre deux numéros civiques encore tenus pour
#: voisins. Au-delà, le numéro lu appartient à une autre portion de rue et ne
#: dit rien de l'établissement.
CIVIC_NEIGHBOUR_SPAN = 40


def civic_number(address: str) -> str | None:
    """Numéro civique d'une adresse, s'il en porte un en tête."""
    tokens = normalise(address).split()
    for token in tokens:
        if token.isdigit() and 2 <= len(token) <= 5:
            return token
    return None


def contradicts_civic(text: str, expected_civic: str | None) -> str | None:
    """Numéro civique voisin lu dans l'image, s'il en dément l'appartenance.

    Un immeuble porte souvent son numéro en grand sur sa façade. Mesuré sur ce
    pilote, l'OCR lisait « 1205 » sur le voisin sans qu'aucun nom d'enseigne
    n'apparaisse : la ressemblance d'embedding l'emportait alors, et le
    bâtiment d'à côté remontait en tête des références.

    Seuls les numéros **proches** comptent : un numéro éloigné appartient à une
    autre rue ou à un panneau publicitaire, et ne prouve rien.
    """
    if not expected_civic or not expected_civic.isdigit():
        return None

    target = int(expected_civic)
    for token in normalise(text).split():
        if not token.isdigit() or not 2 <= len(token) <= 5:
            continue
        if token == expected_civic:
            # Le bon numéro : l'image appartient au site, pas l'inverse.
            return None
        value = int(token)
        if abs(value - target) <= CIVIC_NEIGHBOUR_SPAN:
            return token
    return None
