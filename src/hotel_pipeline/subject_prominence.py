"""Prominence du sujet, mesurée **sur les pixels** (Lot 2).

`appearance_quality` déduisait la place du bâtiment dans le cadre d'un calcul
géométrique : distance, champ de vision, cap. C'est une **prédiction**, et elle
s'est révélée fausse sur le corpus réel — une vue à 63 m notée 0,99
« reference_grade » montrait l'hôtel comme un petit bloc lointain derrière un
parc-o-bus, une clôture et un lampadaire. La géométrie ne sait pas ce qui se
met devant, ni ce que l'objectif cadre vraiment.

Ce module répond à la même question en regardant l'image. Il n'invente rien :
il compare l'image à une description du sujet recherché et à des descriptions
de ce que le corpus contient réellement quand le sujet manque — route vide,
horizon lointain, poteaux, pavillon.

Deux règles, héritées de la cascade de classification et vérifiées ici :

1. **Une alternative décrit une scène concrète, jamais une négation.** CLIP
   n'encode pas la négation : « une photo sans bâtiment » contient le mot
   bâtiment et l'emporte sur les photos de bâtiments.
2. **Le score propose, il ne décide pas.** Le classifieur du Lot 1B avait 26 %
   de rappel et retirait 271 assets que personne n'avait regardés. Ici, le
   score **classe** une file de sélection ; il ne retire rien, et la porte
   d'apparence le déclare inféré, non mesuré au sens géométrique.

Mesuré sur le pilote, contre un jugement visuel :

```text
vue rapprochée de la façade      0,921
vue lointaine derrière parc-o-bus 0,001
neige et lampadaires              0,000
intérieur de concession           0,001
immeuble de bureaux voisin        0,202
```
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .logging import get_logger

log = get_logger("prominence")

#: Ce qu'on cherche : le sujet occupant réellement le cadre.
SUBJECT_PROMPTS: tuple[str, ...] = (
    "a large hotel or motel building filling most of the photograph, "
    "its facade clearly visible",
)

#: Ce que le corpus contient quand le sujet manque. Scènes concrètes, tirées
#: de ce qu'on a réellement vu en revue : chaussée, horizon, poteaux, pavillon.
ABSENCE_PROMPTS: tuple[str, ...] = (
    "an empty road, parking lot or snowbank with no building nearby",
    "a distant skyline where buildings are tiny specks on the horizon",
    "trees, street lights and utility poles occupying the frame",
    "a residential house or small commercial storefront",
)

#: Au-dessus, le sujet est franchement au premier plan.
#:
#: **Calibré**, non choisi : mesuré contre 47 décisions humaines du pilote.
#:
#: ```text
#: seuil  rappel  précision  F1
#:  0,15    50 %      61 %   0,55
#:  0,20    45 %      71 %   0,56   <- meilleur F1
#:  0,45    32 %      78 %   0,45
#:  0,60    32 %     100 %   0,48   <- valeur retenue
#: ```
#:
#: 0,60 ne maximise pas le F1 : il maximise la **précision**. C'est le bon
#: compromis ici parce qu'un faux positif entre dans les références
#: d'apparence d'une vidéo commerciale — une concession voisine prise pour
#: l'hôtel —, tandis qu'un faux négatif ne fait que laisser une vue de côté,
#: où `ABSENT_THRESHOLD` la récupère comme partielle.
#:
#: Le rappel de 32 % est donc assumé, non ignoré : il est la raison d'être de
#: la bande « incidente », et `facade_segments` y puise la couverture que ce
#: seuil écarte.
PROMINENT_THRESHOLD = 0.60

#: En dessous, le sujet est absent ou anecdotique.
#:
#: 0,15 retient la moitié des vues confirmées (rappel 50 %) : c'est la bande
#: des vues **partielles**, celles dont l'union couvre ce qu'aucune ne montre
#: seule. Les écarter ferait perdre 33° d'arc sur la façade principale du
#: pilote.
ABSENT_THRESHOLD = 0.15


@dataclass
class ProminenceReading:
    """Ce que les pixels disent de la place du sujet."""

    asset_id: str
    score: float
    verdict: str
    #: Le modèle a-t-il pu lire l'image ? Faux = aucune mesure, pas un zéro.
    measured: bool = True
    reason: str | None = None

    def as_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "score": round(self.score, 4),
            "verdict": self.verdict,
            "measured": self.measured,
            "reason": self.reason,
        }


def _verdict(
    score: float,
    accept: float = PROMINENT_THRESHOLD,
    partial: float = ABSENT_THRESHOLD,
) -> str:
    if score >= accept:
        return "subject_prominent"
    if score >= partial:
        return "subject_incidental"
    return "subject_absent"


class ProminenceReader:
    """Lit la prominence du sujet sur un lot d'images.

    Le modèle est chargé une fois : l'initialisation coûte ~10 s, l'inférence
    ~0,2 s par image. Instancier par image rendrait un lot de 200 vues
    inutilisable.
    """

    def __init__(self, classifier=None, policy=None):  # noqa: ANN001
        from .schemas.policy import DEFAULT_POLICY

        policy = policy or DEFAULT_POLICY
        if classifier is None:
            from .triage.classify import Classifier

            classifier = Classifier(policy=policy)
        self._classifier = classifier
        self._text_features = None
        # Les descriptions viennent de la **politique** : celles d'un motel de
        # banlieue ne décrivent pas un hôtel de centre-ville, et les figer au
        # module rendait le portage impossible sans éditer le code.
        geometry = getattr(policy, "geometry", None)
        self.subject_prompts: tuple[str, ...] = tuple(
            [getattr(geometry, "subject_prompt", None)] if geometry
            and getattr(geometry, "subject_prompt", None) else SUBJECT_PROMPTS
        )
        self.absence_prompts: tuple[str, ...] = tuple(
            getattr(geometry, "absence_prompts", None) or ABSENCE_PROMPTS
        )
        self.accept_threshold: float = float(
            getattr(geometry, "prominence_accept", PROMINENT_THRESHOLD)
        )
        self.partial_threshold: float = float(
            getattr(geometry, "prominence_partial", ABSENT_THRESHOLD)
        )

    def _encode_prompts(self):  # noqa: ANN202
        if self._text_features is not None:
            return self._text_features

        import torch

        classifier = self._classifier
        tokens = classifier.tokenizer(
            list(self.subject_prompts) + list(self.absence_prompts)
        ).to(classifier.device)
        with torch.no_grad():
            features = classifier.model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)
        self._text_features = features
        return features

    def read(self, asset_id: str, image_path: Path) -> ProminenceReading:
        """Prominence du sujet sur une image, ou l'aveu qu'on n'a pas pu lire."""
        import torch
        from PIL import Image

        if not Path(image_path).is_file():
            return ProminenceReading(
                asset_id=asset_id, score=0.0, verdict="unmeasured",
                measured=False, reason="fichier absent",
            )

        try:
            with Image.open(image_path) as handle:
                tensor = self._classifier.preprocess(
                    handle.convert("RGB")
                ).unsqueeze(0).to(self._classifier.device)
        except Exception as exc:  # image illisible, tronquée, format inconnu
            return ProminenceReading(
                asset_id=asset_id, score=0.0, verdict="unmeasured",
                measured=False, reason=f"image illisible : {exc}"[:120],
            )

        text_features = self._encode_prompts()
        with torch.no_grad():
            image_features = self._classifier.model.encode_image(tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            probabilities = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]

        # Somme des descriptions du sujet : plusieurs formulations décrivent la
        # même scène, et n'en garder qu'une ferait dépendre le score du hasard
        # de la rédaction.
        score = float(probabilities[: len(self.subject_prompts)].sum())
        return ProminenceReading(
            asset_id=asset_id,
            score=score,
            verdict=_verdict(score, self.accept_threshold, self.partial_threshold),
        )

    def read_many(self, items: list[tuple[str, Path]]) -> list[ProminenceReading]:
        readings = [self.read(asset_id, path) for asset_id, path in items]
        measured = sum(1 for r in readings if r.measured)
        log.info(
            "prominence : %d image(s) lue(s), %d non mesurée(s)",
            measured, len(readings) - measured,
        )
        return readings


def reference_strength(
    reading: "ProminenceReading",
    target_building_visible: bool | None,
) -> tuple[float | None, str]:
    """Force continue de la référence, plutôt qu'un oui/non.

    `is_reference_grade` rendait un booléen : une vue notée 0,59 et une notée
    0,61 devenaient « non » et « oui », et rien en aval ne savait qu'elles
    étaient presque identiques. Le seuil reste utile pour trancher, mais il ne
    doit pas **effacer** ce qu'il a mesuré.

    Retourne `(force, motif)`. `None` signifie non mesuré — distinct de 0,0,
    qui affirme que le sujet est absent.
    """
    if not reading.measured:
        return None, reading.reason or "prominence non mesurée"
    if target_building_visible is not True:
        return 0.0, (
            "bâtiment au premier plan, mais identité non établie : "
            "le voisinage compte concessions et bureaux"
        )
    # L'identité est établie : la force est celle mesurée sur les pixels.
    return reading.score, "sujet mesuré et identité établie"


def is_reference_grade(
    reading: "ProminenceReading",
    target_building_visible: bool | None,
) -> tuple[bool, str]:
    """Le sujet est-il au premier plan **et** est-ce bien le nôtre ?

    Deux questions distinctes qu'aucun score seul ne tranche. Mesuré sur le
    pilote : les neuf vues les mieux notées comprenaient trois concessions
    automobiles voisines, toutes à 160-202 m. Le modèle voit « un grand
    bâtiment qui remplit le cadre » — il ne sait pas *lequel*.

    L'identité vient de `target_building_visible`, établie par le cadrage
    mesuré ou par une revue humaine. La prominence vient des pixels. Une
    référence d'apparence exige les deux : montrer beaucoup, et montrer nous.
    """
    if not reading.measured:
        return False, reading.reason or "prominence non mesurée"
    if reading.verdict != "subject_prominent":
        return False, f"sujet non proéminent ({reading.verdict})"
    if target_building_visible is not True:
        return False, (
            "bâtiment au premier plan, mais identité non établie : "
            "le voisinage compte concessions et bureaux"
        )
    return True, "sujet proéminent et identité établie"


__all__ = [
    "is_reference_grade",
    "reference_strength",
    "ABSENCE_PROMPTS",
    "ABSENT_THRESHOLD",
    "PROMINENT_THRESHOLD",
    "SUBJECT_PROMPTS",
    "ProminenceReader",
    "ProminenceReading",
]
