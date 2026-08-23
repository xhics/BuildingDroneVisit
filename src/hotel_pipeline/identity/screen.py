"""Dépistage automatique d'un corpus : qui montre l'établissement, qui non.

Le module enchaîne trois modèles et aucune heuristique de métadonnée :

1. l'**OCR d'enseigne** propose des ancres — lire « WELCOMINNS » sur un panneau
   est une preuve d'identité qu'aucun champ du manifeste ne porte ;
2. l'**embedding** mesure la ressemblance de chaque image aux ancres ;
3. la **notation d'attributs** en langage naturel évalue si l'image est
   exploitable comme référence — cadrage, occultation, saison.

Le troisième point mérite d'être explicite : « le bâtiment est-il assez grand
dans le cadre ? » est décidé par le modèle sur l'image, pas par une règle sur
la distance au centroïde. C'est ce qui distingue une vue de façade d'une vue
lointaine où le bâtiment occupe trois pour cent des pixels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..logging import get_logger
from .anchors import Anchor, AnchorSet
from .embedding import EmbeddingIndex, cosine
from .candidates import USABLE_RIGHTS
from .verdict import (
    IdentityStatus,
    IdentityVerdict,
    calibrate_threshold,
    judge,
    uncertain_band,
)

log = get_logger("identity-screen")

#: Attributs jugés par le modèle, décrits en langage naturel. Chaque entrée
#: oppose une affirmation à des alternatives concrètes : CLIP n'encode pas la
#: négation, une alternative doit décrire une scène, jamais son absence.
ATTRIBUTE_PROMPTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "close_framing": (
        "a large building filling most of the photograph, seen from nearby",
        (
            "a wide landscape where buildings are small and far away on the horizon",
            "an empty road, roundabout or parking lot in the foreground",
            "a close-up of a road sign, pole or traffic furniture",
        ),
    ),
    "facade_visible": (
        "the front facade of a building with its windows and main entrance visible",
        (
            "the back or side of a building, mostly blank wall",
            "a building hidden behind trees, fences or parked vehicles",
            "an interior room seen from inside",
        ),
    ),
    "is_photograph": (
        "a photograph of a real building taken outdoors",
        (
            "a flat graphic logo, wordmark or text banner on a plain background",
            "a printed map, floor plan or diagram",
            "a screenshot of a web page",
        ),
    ),
    "unobstructed": (
        "a clear unobstructed view of a building",
        (
            "a building largely hidden behind bare tree branches",
            "a view blocked by a truck, bus or large vehicle",
        ),
    ),
    "summer": (
        "a sunny summer day with green leaves and blue sky",
        (
            "a winter scene with snow on the ground and bare trees",
            "an overcast grey day with flat diffuse light",
        ),
    ),
}

#: Nature du bâti, décrite en langage naturel. Ce prompt tranche là où la
#: ressemblance aux ancres hésite : mesuré sur ce pilote, il note 0,72 sur les
#: images retenues contre 0,001 sur les indécises. Il reste court à dessein —
#: une description longue et détaillée se dilue et ne discrimine plus rien.
BUILT_FORM_PROMPT: tuple[str, tuple[str, ...]] = (
    "a long two-storey commercial building",
    (
        "a detached family house",
        "trees and lawn only",
        "a road seen from a car",
    ),
)

#: En deçà, une image indécise ne montre pas un bâti de la bonne nature et le
#: doute est tranché en rejet ; au-delà, elle est retenue.
BUILT_FORM_ACCEPT = 0.35

#: Un attribut n'est retenu que s'il l'emporte franchement sur ses alternatives.
ATTRIBUTE_ACCEPT = 0.55

#: En deçà, l'image relève du visuel graphique — logo, plan, capture d'écran —
#: et ne peut pas servir d'ancre, quel que soit le texte qu'on y lit.
PHOTOGRAPH_ACCEPT = 0.60

#: Côté minimal d'une image servant de référence de production. Une vignette
#: peut parfaitement montrer le bon bâtiment — mesuré sur ce pilote, une
#: acquisition de 200 px obtenait le meilleur score du corpus — sans porter
#: assez de détail pour guider un générateur.
MIN_REFERENCE_SIDE = 640


@dataclass
class ScreenedAsset:
    """Une image, son verdict d'identité et ce qu'elle vaut comme référence."""

    asset_id: str
    path: Path
    verdict: IdentityVerdict
    attributes: dict[str, float] = field(default_factory=dict)
    #: Ce qu'une enseigne lue dans l'image dit de son appartenance.
    sign_status: str | None = None
    sign_text: str | None = None
    #: Plus petit côté de l'image, en pixels.
    min_side: int | None = None
    #: Ce que le manifeste dit des droits de cette image. `unknown` par défaut :
    #: ne pas savoir n'est pas une autorisation.
    rights: str = "unknown"
    #: Ressemblance à un bâti de la nature attendue, quand elle a servi à
    #: lever une indécision.
    built_form: float | None = None

    @property
    def reference_score(self) -> float:
        """Aptitude à servir de référence, une fois l'identité établie.

        `facade_visible` agit en **facteur**, non en terme additif. Mesuré sur
        le corpus : un immeuble de bureaux voisin, tout en verre, obtenait une
        ressemblance d'embedding de 0,83 — supérieure à celle des vraies vues
        de l'hôtel — parce que la scène de rue enneigée domine le vecteur. Son
        score de façade, lui, restait à 0,11. Additionné, il passait devant ;
        multiplié, il tombe. Une image où aucune façade n'est lisible ne peut
        pas servir de référence, quel que soit son score d'identité.
        """
        if self.verdict.status is not IdentityStatus.MATCH:
            return 0.0
        # Les droits sont **rapportés**, jamais arbitrés ici : le classement
        # suit la seule qualité visuelle, et le statut voyage avec l'image pour
        # que la décision d'usage se prenne ailleurs, en connaissance de cause.
        framing = self.attributes.get("close_framing", 0.0)
        facade = self.attributes.get("facade_visible", 0.0)
        clear = self.attributes.get("unobstructed", 0.0)
        base = self.verdict.score * 0.55 + framing * 0.3 + clear * 0.15
        score = base * (0.25 + 0.75 * facade)
        # Une image trop petite reste informative sur l'identité, mais ne peut
        # pas servir de référence : la pénalité est progressive, pour ne pas
        # écarter d'un cheveu une image à peine sous la barre.
        if self.min_side is not None and self.min_side < MIN_REFERENCE_SIDE:
            score *= max(0.15, self.min_side / MIN_REFERENCE_SIDE)
        return float(score)

    def as_dict(self) -> dict:
        return {
            **self.verdict.as_dict(),
            "path": str(self.path),
            "reference_score": round(self.reference_score, 4),
            "attributes": {k: round(v, 4) for k, v in self.attributes.items()},
            "sign_status": self.sign_status,
            "sign_text": self.sign_text,
            "min_side": self.min_side,
            "rights": self.rights,
            "rights_cleared": self.rights in USABLE_RIGHTS,
            "built_form": None if self.built_form is None else round(self.built_form, 3),
        }


@dataclass
class ScreeningResult:
    """Le tri complet d'un corpus, et ce qu'il autorise."""

    hotel_id: str
    assets: list[ScreenedAsset]
    threshold: float
    threshold_reason: str
    anchor_summary: dict

    def by_status(self, status: IdentityStatus) -> list[ScreenedAsset]:
        return [a for a in self.assets if a.verdict.status is status]

    def rights_summary(self) -> dict[str, int]:
        """Répartition des droits parmi les images retenues."""
        counts: dict[str, int] = {}
        for asset in self.by_status(IdentityStatus.MATCH):
            counts[asset.rights] = counts.get(asset.rights, 0) + 1
        return counts

    def best_references(self, limit: int = 8) -> list[ScreenedAsset]:
        """Les meilleures références, identité établie d'abord."""
        matched = self.by_status(IdentityStatus.MATCH)
        return sorted(matched, key=lambda a: a.reference_score, reverse=True)[:limit]

    def as_dict(self) -> dict:
        counts = {
            str(status): len(self.by_status(status)) for status in IdentityStatus
        }
        return {
            "hotel_id": self.hotel_id,
            "screened_at": datetime.now(timezone.utc).isoformat(),
            "threshold": round(self.threshold, 4),
            "threshold_reason": self.threshold_reason,
            "anchors": self.anchor_summary,
            "counts": counts,
            "rights_summary": self.rights_summary(),
            "assets": [a.as_dict() for a in self.assets],
            "caveats": [
                "le verdict porte sur la ressemblance à des ancres : une ancre "
                "fausse inverse proprement tout le tri",
                "`uncertain` n'est pas un échec du modèle mais une demande de "
                "revue humaine : ces images ne doivent pas partir en production",
                "les droits viennent du manifeste et sont rapportés tels "
                "quels : le classement ne les arbitre pas, il les transporte "
                "pour que la décision d'usage se prenne en connaissance",
            ],
        }


def discover_anchors_by_sign(
    candidates: list[tuple[str, Path]],
    expected_names: list[str],
    excluded_names: list[str] | None = None,
    limit: int = 5,
    budget: int = 60,
    photo_scores: dict[str, float] | None = None,
) -> list[Anchor]:
    """Propose des ancres en lisant les enseignes, sans intervention humaine.

    C'est le seul moyen d'établir l'identité **sans rien présupposer** : le nom
    de l'établissement, écrit sur son propre panneau, est une preuve intrinsèque
    à l'image. Toute autre amorce — la proximité au centroïde, l'azimut — ne
    ferait que présumer ce qu'elle prétend établir.

    Trois précautions, chacune tirée d'un défaut mesuré :

    1. **l'ordre de lecture n'est pas quelconque.** L'OCR coûte plusieurs
       secondes par image ; lire un corpus entier prendrait des heures. Les
       candidates sont donc triées par vraisemblance d'enseigne avant lecture,
       et seules les `budget` premières sont ouvertes ;
    2. **l'appariement est tolérant.** Un panneau photographié de loin rend
       « TETRA 1205 TECH » ou « Tsomed » — un test sur limites de mots échoue
       sur les deux ;
    3. **un nom concurrent disqualifie l'image.** Lire l'enseigne du voisin
       dans le même cadre ne fait pas de la photographie une ancre.
    """
    from . import sign_match

    excluded_names = excluded_names or []
    found: list[Anchor] = []
    for asset_id, path in candidates[:budget]:
        if len(found) >= limit:
            break
        try:
            text = _sign_reader().read(path)
        except Exception as exc:  # pragma: no cover - dépend du moteur d'OCR
            log.info("OCR indisponible sur %s : %s", path.name, exc)
            continue

        status, term, method = sign_match.evaluate(
            text, expected_names, excluded_names
        )
        if status != "match":
            continue

        # Le nom lu ne suffit pas : un logo de la marque le porte aussi, et
        # mesuré sur ce pilote un wordmark du site officiel arrivait en tête
        # des propositions. Une ancre calibre toute la suite du tri par sa
        # ressemblance visuelle — elle doit donc montrer le bâtiment, pas son
        # identité graphique.
        if photo_scores is not None:
            photographic = photo_scores.get(asset_id, 1.0)
            if photographic < PHOTOGRAPH_ACCEPT:
                log.info(
                    "écarté, visuel non photographique (%.2f) : %s",
                    photographic,
                    path.name,
                )
                continue

        found.append(
            Anchor(
                asset_id=asset_id,
                path=path,
                origin="sign_ocr",
                evidence=(
                    f"enseigne lue ({method}) : {text.strip()[:70]!r} "
                    f"— terme {term!r}"
                ),
            )
        )
        log.info("ancre par enseigne : %s (%s)", path.name, method)
    return found


def rank_sign_candidates(
    index: EmbeddingIndex, candidates: list[tuple[str, Path]]
) -> tuple[list[tuple[str, Path]], dict[str, float]]:
    """Trie les images par vraisemblance de porter une enseigne lisible.

    Le tri est fait par le modèle, sur l'image, et non par une règle sur les
    métadonnées : aucun champ du manifeste ne dit si un panneau est visible et
    lisible. Il évite d'ouvrir l'OCR sur des vues d'intérieur ou de chaussée.
    """
    prompts = [
        "a large sign or illuminated lettering on a pole, with readable text",
        "a building facade with a company name written on it",
        "an empty road, parking lot or plain landscape with no readable text",
        "an indoor room seen from inside",
    ]
    text_vectors = index.embedder.encode_text(prompts)

    photo_positive, photo_alternatives = ATTRIBUTE_PROMPTS["is_photograph"]
    photo_vectors = index.embedder.encode_text(
        [photo_positive, *photo_alternatives]
    )

    scored: list[tuple[float, str, Path]] = []
    photographic: dict[str, float] = {}
    for asset_id, path in candidates:
        try:
            vector = index.vector_of(path)
        except (OSError, ValueError):
            continue
        logits = np.array([cosine(vector, v) for v in text_vectors]) * 100.0
        exp = np.exp(logits - logits.max())
        probabilities = exp / exp.sum()
        # Les deux premiers intitulés décrivent du texte lisible ; les deux
        # derniers décrivent son absence.
        scored.append((float(probabilities[0] + probabilities[1]), asset_id, path))

        photo_logits = np.array([cosine(vector, v) for v in photo_vectors]) * 100.0
        photo_exp = np.exp(photo_logits - photo_logits.max())
        photographic[asset_id] = float((photo_exp / photo_exp.sum())[0])

    scored.sort(reverse=True, key=lambda item: item[0])
    return [(asset_id, path) for _, asset_id, path in scored], photographic


def score_attributes(
    index: EmbeddingIndex, image_vector: np.ndarray, text_cache: dict
) -> dict[str, float]:
    """Note les attributs d'une image contre des descriptions concurrentes."""
    scores: dict[str, float] = {}
    for name, (positive, alternatives) in ATTRIBUTE_PROMPTS.items():
        vectors = text_cache.get(name)
        if vectors is None:
            vectors = index.embedder.encode_text([positive, *alternatives])
            text_cache[name] = vectors
        logits = np.array([cosine(image_vector, v) for v in vectors]) * 100.0
        exp = np.exp(logits - logits.max())
        scores[name] = float((exp / exp.sum())[0])
    return scores


def _min_side(path: Path) -> int | None:
    """Plus petit côté d'une image, sans la décoder entièrement."""
    try:
        from PIL import Image

        with Image.open(path) as raw:
            return int(min(raw.size))
    except Exception:  # pragma: no cover - dépend du fichier
        return None



@lru_cache(maxsize=1)
def _sign_reader():
    """Charge le moteur d'OCR une seule fois par processus.

    Sans ce cache, EasyOCR se réinitialisait à chaque image : cinq secondes
    par appel, pour un moteur strictement identique d'un appel au suivant.
    """
    from ..triage import sign_ocr

    return sign_ocr.get_reader()



def read_sign(
    path: Path,
    expected: list[str],
    excluded: list[str],
    expected_civic: str | None = None,
):
    """Lit l'enseigne d'une image et confronte le texte aux noms attendus.

    C'est la seule preuve d'identité **discriminante** du dispositif. La
    ressemblance d'embedding mesure la ressemblance d'une scène, ce qui ne
    suffit pas entre voisins : mesuré sur ce pilote, l'immeuble de bureaux du
    1205 obtient 0,80 contre les ancres de l'hôtel du 1195 — même rue, même
    neige, même lumière grise. Seul le texte lu sur le panneau distingue
    « WELCOMINNS » de « TETRA TECH ».
    """
    from . import sign_match

    try:
        text = _sign_reader().read(path)
    except Exception as exc:  # pragma: no cover - dépend du moteur d'OCR
        log.info("OCR indisponible sur %s : %s", path.name, exc)
        return None, None
    status, _term, _method = sign_match.evaluate(text, expected, excluded)
    if status == "uncertain":
        # Aucun nom lisible, mais un numéro civique voisin suffit : un immeuble
        # porte le sien en grand, et c'est parfois le seul indice disponible.
        neighbour = sign_match.contradicts_civic(text, expected_civic)
        if neighbour:
            log.info("numéro civique voisin lu (%s) : %s", neighbour, path.name)
            status = "mismatch"
    return status, text.strip()[:120]


def _resolve_uncertain(
    assets: list[ScreenedAsset],
    index: EmbeddingIndex,
    vectors: dict[str, np.ndarray],
) -> int:
    """Écarte les indécises que la nature du bâti dément.

    Les images indécises ne sont pas ambiguës en elles-mêmes : elles tombent
    dans la bande étroite qui entoure le seuil — six centièmes de large sur ce
    pilote, pour quatre-vingt-une images. Les départager demande un second
    signal, indépendant de la distance aux ancres.

    Celui-ci décrit ce que le site **est** : un long bâtiment de deux étages.
    Une rue résidentielle ou un stationnement vide n'y ressemblent pas, quelle
    que soit leur proximité d'embedding à une ancre hivernale.

    Il ne sert qu'à **écarter**. Promouvoir sur ce seul critère ferait entrer
    l'immeuble de bureaux voisin, qui correspond parfaitement à la
    description : la forme dit ce qu'une image n'est pas, jamais qu'elle est
    le bon bâtiment.
    """
    undecided = [a for a in assets if a.verdict.status is IdentityStatus.UNCERTAIN]
    if not undecided:
        return 0

    positive, alternatives = BUILT_FORM_PROMPT
    text_vectors = index.embedder.encode_text([positive, *alternatives])

    resolved = 0
    for asset in undecided:
        vector = vectors.get(asset.asset_id)
        if vector is None:
            continue
        logits = np.array([cosine(vector, v) for v in text_vectors]) * 100.0
        exp = np.exp(logits - logits.max())
        score = float((exp / exp.sum())[0])

        asset.built_form = score

        # La forme du bâti **écarte**, elle ne promeut pas. Un immeuble de
        # bureaux voisin ressemble parfaitement à « un long bâtiment de deux
        # étages » — mesuré à 1,00 sur ce pilote — et repassait `match` sans
        # qu'aucune enseigne ne l'ait démenti. Le doute qui subsiste après
        # cette épreuve reste un doute.
        if score >= BUILT_FORM_ACCEPT:
            continue

        asset.verdict = IdentityVerdict(
            asset.asset_id,
            IdentityStatus.MISMATCH,
            asset.verdict.score,
            asset.verdict.threshold,
            asset.verdict.nearest_anchor,
            f"indécision levée par la nature du bâti : {score:.0%} de "
            f"ressemblance à un long bâtiment de deux étages, insuffisant",
        )
        resolved += 1

    if resolved:
        log.info("%d indécision(s) tranchée(s) sur la forme du bâti", resolved)
    return resolved


def _apply_sign_evidence(
    assets: list[ScreenedAsset],
    expected: list[str],
    excluded: list[str],
    budget: int,
    expected_civic: str | None = None,
) -> None:
    """Confronte les meilleures candidates à leur enseigne, et tranche.

    La preuve textuelle l'emporte sur la ressemblance d'embedding : elle est
    spécifique là où celle-ci ne l'est pas. Un immeuble de bureaux voisin peut
    ressembler à l'hôtel — même rue, même saison, même lumière — mais son
    panneau ne porte pas le même nom.
    """
    ranked = sorted(
        (a for a in assets if a.verdict.status is IdentityStatus.MATCH),
        key=lambda a: a.reference_score,
        reverse=True,
    )[:budget]

    for asset in ranked:
        status, text = read_sign(asset.path, expected, excluded, expected_civic)
        asset.sign_status, asset.sign_text = status, text
        if status == "mismatch":
            asset.verdict = IdentityVerdict(
                asset.asset_id,
                IdentityStatus.MISMATCH,
                asset.verdict.score,
                asset.verdict.threshold,
                asset.verdict.nearest_anchor,
                f"ressemblance {asset.verdict.score:.3f} démentie par "
                f"l'enseigne lue : {text!r}",
            )
            asset.attributes = {}
            log.info("démenti par enseigne : %s", asset.path.name)
        elif status == "match":
            log.info("identité confirmée par enseigne : %s", asset.path.name)


def screen_assets(
    hotel_id: str,
    candidates: list[tuple[str, Path]],
    anchor_set: AnchorSet,
    index: EmbeddingIndex,
    with_attributes: bool = True,
    expected_names: list[str] | None = None,
    excluded_names: list[str] | None = None,
    sign_budget: int = 12,
    rights_by_id: dict[str, str] | None = None,
    expected_civic: str | None = None,
) -> ScreeningResult:
    """Juge chaque image du corpus contre les ancres, puis calibre le seuil."""
    if not candidates:
        raise ValueError("aucune image à dépister")

    anchor_set.embed(index)
    coherence = anchor_set.coherence()

    scored: list[tuple[str, Path, float, str | None, np.ndarray, int | None]] = []
    seen: set[str] = set()
    for asset_id, path in candidates:
        # Le même fichier atteint par deux chemins ne doit être jugé qu'une
        # fois : sans cela, une acquisition recopiée occupe deux rangs du
        # classement final avec le même score.
        try:
            fingerprint = EmbeddingIndex.digest(path)
        except OSError as exc:
            log.info("image illisible, écartée : %s (%s)", path, exc)
            continue
        if fingerprint in seen:
            continue
        seen.add(fingerprint)

        try:
            vector = index.vector_of(path)
        except (OSError, ValueError) as exc:
            log.info("image illisible, écartée : %s (%s)", path, exc)
            continue
        score, nearest = anchor_set.similarity(vector)
        scored.append((asset_id, path, score, nearest, vector, _min_side(path)))

    all_scores = [s for _, _, s, _, _, _ in scored]
    threshold, reason = calibrate_threshold(all_scores)
    band = uncertain_band(all_scores)
    log.info("seuil calibré à %.3f — %s", threshold, reason)

    text_cache: dict = {}
    vectors: dict[str, np.ndarray] = {}
    assets: list[ScreenedAsset] = []
    for asset_id, path, score, nearest, vector, min_side in scored:
        verdict = judge(
            asset_id, score, threshold, nearest, len(anchor_set), coherence, band
        )
        attributes: dict[str, float] = {}
        # Les attributs ne servent qu'à classer les images retenues : les
        # calculer sur un corpus entier coûterait sans rien décider.
        if with_attributes and verdict.status is IdentityStatus.MATCH:
            attributes = score_attributes(index, vector, text_cache)

        assets.append(
            ScreenedAsset(
                asset_id,
                path,
                verdict,
                attributes,
                min_side=min_side,
                rights=(rights_by_id or {}).get(asset_id, "unknown"),
            )
        )
        vectors[asset_id] = vector

    # Les indécisions sont levées avant l'OCR : celui-ci ne lit que les
    # meilleures candidates, et une image tranchée entre-temps peut y entrer.
    _resolve_uncertain(assets, index, vectors)

    # L'OCR coûte plusieurs secondes par image : le passer sur tout le corpus
    # prendrait des dizaines de minutes pour ne rien changer aux rejets déjà
    # tranchés. Il n'est appelé que là où il décide — en tête de classement,
    # là où une image s'apprête à partir en production.
    if expected_names or excluded_names or expected_civic:
        _apply_sign_evidence(
            assets,
            expected_names or [],
            excluded_names or [],
            sign_budget,
            expected_civic,
        )

    index.save()
    return ScreeningResult(
        hotel_id=hotel_id,
        assets=assets,
        threshold=threshold,
        threshold_reason=reason,
        anchor_summary=anchor_set.as_dict(),
    )
