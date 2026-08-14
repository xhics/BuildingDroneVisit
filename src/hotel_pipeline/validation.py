"""Validation conditionnelle d'une cohorte étiquetée (Lot 1B V2).

Ce module compare des étiquettes humaines à des prédictions **gelées avant
l'étiquetage**. Il ne calibre rien : il décrit ce qu'un jeu conditionnel
permet de dire, et refuse d'énoncer ce qu'il ne permet pas.

Quatre populations, jamais mélangées :

```text
blind_first_pass        les décisions prises sans voir le système
unblinded_existing      celles prises avant l'aveuglement, en voyant le diagnostic
unblinded_adjudication  les corrections de seconde passe, avec séquence et mesures
operational_final       la dernière décision applicable, quelle qu'en soit l'origine
```

Seule la première juge le système. Les autres servent au pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from .logging import get_logger
from .schemas import Asset, Blinding, ReviewDecision

log = get_logger("validation")

#: Ce que cette cohorte ne mesure pas, inscrit dans le rapport lui-même.
LIMITS = [
    "sélection issue du modèle : le rappel sur les 189 vues Mapillary n'est pas "
    "mesurable, les faux négatifs étant exclus par construction",
    "deux séquences seulement : aucune inférence statistique robuste, les vues "
    "d'une même séquence n'étant pas indépendantes",
    "un site, un réviseur : aucune généralisation à d'autres établissements",
    "aucun seuil de modèle ne peut être réglé ici — il faudrait un échantillon "
    "stratifié distinct, séparé réglage/validation au niveau des séquences",
]


def first_blind(asset: Asset):  # noqa: ANN201
    """Première décision aveugle d'un asset, s'il en a une."""
    return next(
        (e for e in asset.review_history if e.blinding is Blinding.BLIND), None
    )


def first_blind_geometry(asset: Asset):  # noqa: ANN201
    return next(
        (e for e in asset.geometry_history if e.blinding is Blinding.BLIND), None
    )


@dataclass
class ValidationReport:
    """Résultats, séparés par population et par séquence."""

    hotel_id: str = ""
    title: str = "validation conditionnelle des candidats Mapillary"
    built_at: str = ""
    cohort_digest: str = ""
    protocol_id: str = ""
    predictions_digest: str = ""
    sequence_register_digest: str = ""
    sequence_correlation: str = "unknown"

    blind_first_pass: dict = field(default_factory=dict)
    unblinded_existing: dict = field(default_factory=dict)
    unblinded_adjudication: dict = field(default_factory=dict)
    operational_final: dict = field(default_factory=dict)

    by_sequence: dict = field(default_factory=dict)
    confusions: list[dict] = field(default_factory=list)
    geometry: dict = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "title": self.title,
            "hotel_id": self.hotel_id,
            "built_at": self.built_at,
            "bindings": {
                "cohort_digest": self.cohort_digest,
                "protocol_id": self.protocol_id,
                "predictions_digest": self.predictions_digest,
                "sequence_register_digest": self.sequence_register_digest,
                "sequence_correlation": self.sequence_correlation,
            },
            "blind_first_pass": self.blind_first_pass,
            "unblinded_existing": self.unblinded_existing,
            "unblinded_adjudication": self.unblinded_adjudication,
            "operational_final": self.operational_final,
            "by_sequence": self.by_sequence,
            "confusions": self.confusions,
            "geometry_suitability": self.geometry,
            "limits": LIMITS,
            "caveats": self.caveats,
        }


def _counts(decisions: list[str]) -> dict[str, int]:
    return {
        value: decisions.count(value)
        for value in ("confirmed", "rejected", "unresolved")
        if decisions.count(value)
    }


def blind_pass(assets: list[Asset], predictions: dict) -> dict:
    """Confronte les étiquettes aveugles à l'instantané **gelé**.

    Les prédictions viennent du fichier figé avant l'étiquetage : les
    recalculer aujourd'hui les comparerait à un système que ces mêmes
    étiquettes ont déjà modifié.
    """
    frozen = {row["asset_id"]: row for row in predictions.get("predictions", [])}
    decisions: list[str] = []
    resolved_truth: list[bool] = []
    resolved_predicted: list[bool] = []
    rows = []

    for asset in sorted(assets, key=lambda a: a.id):
        entry = first_blind(asset)
        if entry is None:
            continue
        decisions.append(entry.decision.value)
        predicted = frozen.get(asset.id, {})

        # La prédiction évaluée est celle que le système portait alors :
        # `target_building_visible` gelé, non recalculé.
        expected = predicted.get("target_building_visible")
        row = {
            "asset_id": asset.id,
            "human": entry.decision.value,
            "predicted_target_visible": expected,
            "predicted_role": predicted.get("role"),
            "building_score": (predicted.get("subject_scores") or {}).get("building"),
        }
        if entry.decision is not ReviewDecision.UNRESOLVED:
            truth = entry.decision is ReviewDecision.CONFIRMED
            resolved_truth.append(truth)
            resolved_predicted.append(expected is True)
            row["agreement"] = (expected is True) == truth
        rows.append(row)

    positives = sum(1 for t, p in zip(resolved_truth, resolved_predicted) if t and p)
    false_positives = sum(
        1 for t, p in zip(resolved_truth, resolved_predicted) if p and not t
    )
    false_negatives = sum(
        1 for t, p in zip(resolved_truth, resolved_predicted) if t and not p
    )

    return {
        "labels": len(decisions),
        "three_class": _counts(decisions),
        "undecided_rate": (
            round(decisions.count("unresolved") / len(decisions), 4) if decisions else None
        ),
        # Les métriques binaires ne portent que sur les cas tranchés : inclure
        # les indécises reviendrait à leur prêter une vérité qu'elles nient.
        "resolved_only": {
            "count": len(resolved_truth),
            "true_positive": positives,
            "false_positive": false_positives,
            "false_negative": false_negatives,
            "precision_among_detected": (
                round(positives / (positives + false_positives), 4)
                if positives + false_positives
                else None
            ),
            "note": (
                "précision parmi les candidats détectés ; ce n'est pas une "
                "précision générale Mapillary, et le rappel n'est pas mesurable"
            ),
        },
        "rows": rows,
    }


def by_sequence(assets: list[Asset], register: dict) -> dict:
    """Résultats par séquence Mapillary.

    Les vues d'une même séquence sont prises à quelques secondes d'intervalle
    depuis un véhicule en mouvement : les traiter comme indépendantes
    gonflerait la confiance qu'on peut mettre dans un total.
    """
    sequences = {
        entry["asset_id"]: entry.get("sequence_id") or "sans-séquence"
        for entry in register.get("entries", [])
    }
    grouped: dict[str, list[str]] = {}
    for asset in assets:
        entry = first_blind(asset)
        if entry is None:
            continue
        grouped.setdefault(sequences.get(asset.id, "sans-séquence"), []).append(
            entry.decision.value
        )
    return {
        sequence: {"labels": len(values), **_counts(values)}
        for sequence, values in sorted(grouped.items())
    }


def confusions(assets: list[Asset]) -> list[dict]:
    """Rejets, classés par nature de la confusion.

    Un concurrent du même type ne se confond pas comme un immeuble de bureaux :
    l'un partage la fonction, l'enseigne et souvent l'architecture, l'autre ne
    partage que le voisinage. Les compter ensemble masquerait le cas difficile.
    """
    kinds = {
        "competitor_same_kind": ["mortagne", "hôtel", "hotel"],
        "neighbouring_office": ["1205", "1201", "tetra", "isomed", "à louer", "bureaux"],
        "no_building": ["stationnement", "rond-point", "carrefour", "arbres", "autobus"],
    }
    found = []
    for asset in sorted(assets, key=lambda a: a.id):
        entry = first_blind(asset)
        if entry is None or entry.decision is not ReviewDecision.REJECTED:
            continue
        text = f"{entry.rationale} {' '.join(entry.evidence)}".lower()
        kind = "other"
        for name, markers in kinds.items():
            if any(marker in text for marker in markers):
                kind = name
                break
        found.append(
            {"asset_id": asset.id, "kind": kind, "rationale": entry.rationale}
        )
    return found


def geometry_summary(assets: list[Asset], blind_only: bool = True) -> dict:
    """Aptitudes attribuées, et sur quelles mesures.

    Par défaut, seules les appréciations aveugles : celles rendues après
    l'arbitrage ont vu les mesures, et ne peuvent pas juger le système.
    """
    rows = []
    for asset in sorted(assets, key=lambda a: a.id):
        entry = (
            first_blind_geometry(asset)
            if blind_only
            else (asset.geometry_history[-1] if asset.geometry_history else None)
        )
        if entry is None:
            continue
        rows.append(
            {
                "asset_id": asset.id,
                "suitability": entry.suitability.value,
                "measurements": entry.measurements,
                "rationale": entry.rationale,
            }
        )
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["suitability"]] = counts.get(row["suitability"], 0) + 1
    return {
        "scope": "aveugle seulement" if blind_only else "dernière appréciation",
        "counts": counts,
        "rows": rows,
        "note": (
            "appréciations humaines ; les mesures sont des estimations visuelles "
            "de cadrage, non des calculs — aucun seuil n'est réglé sur six images"
        ),
    }


def population(assets: list[Asset], blinding: Blinding, after_blind: bool) -> dict:
    """Décisions d'une population donnée, hors première passe aveugle."""
    decisions = []
    for asset in assets:
        blind = first_blind(asset)
        for entry in asset.review_history:
            if entry is blind:
                continue
            if entry.blinding is not blinding:
                continue
            posterior = blind is not None and entry.decided_at > blind.decided_at
            if posterior == after_blind:
                decisions.append(entry.decision.value)
    return {"labels": len(decisions), **_counts(decisions)}


def operational(assets: list[Asset]) -> dict:
    """Dernière décision applicable, quelle que soit son origine."""
    decisions = [
        a.target_visibility_decision.value for a in assets if a.review_history
    ]
    return {"labels": len(decisions), **_counts(decisions)}
