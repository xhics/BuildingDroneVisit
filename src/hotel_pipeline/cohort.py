"""Vérité terrain : cohorte, séquences et instantané de prédictions.

Trois précautions, sans lesquelles la revue mesurerait le système avec ses
propres réponses.

**L'aveuglement.** La planche d'analyse montre rôle, motif, scores et ligne de
vue. Excellente pour comprendre ; désastreuse pour étiqueter, puisque le
réviseur y lirait la réponse du système avant de produire les étiquettes qui
serviront à le juger.

**La portée.** La cohorte est définie par `contains_building` ou le sujet
`building`, tous deux issus du modèle. Elle mesure donc les confusions **parmi
les images détectées**, jamais le rappel : les faux négatifs en sont exclus par
construction, et aucun chiffre tiré d'elle ne peut prétendre le contraire.

**La corrélation.** Les assets historiques ne portent ni `sequence_id` ni
provenance d'acquisition. Leur regroupement par séquence se demande à la
source ; à défaut, il reste `unknown` — une proximité géographique n'est pas
une séquence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .logging import get_logger
from .schemas import Asset, Subject

log = get_logger("cohort")

#: Définition exacte de la cohorte, inscrite au rapport : sa portée en dépend.
COHORT_DEFINITION = (
    "assets de source `mapillary` dont `contains_building` est vrai ou dont les "
    "sujets contiennent `building` — sélection issue du modèle, donc mesurant "
    "la précision parmi les images détectées, jamais le rappel"
)


def members(assets: list[Asset], source: str = "mapillary") -> list[Asset]:
    return [
        a
        for a in assets
        if a.source == source and (a.contains_building or Subject.BUILDING in a.subjects)
    ]


def cohort_digest(assets: list[Asset]) -> str:
    """Empreinte de la cohorte, indépendante de l'ordre du manifeste."""
    payload = "|".join(sorted(f"{a.id}:{a.checksum}" for a in assets))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def blind_order(assets: list[Asset], digest: str) -> list[Asset]:
    """Ordre mélangé, déterministe, dérivé de l'empreinte de cohorte.

    Présenter les vues dans l'ordre du manifeste placerait côte à côte les
    images voisines d'une même séquence : le réviseur jugerait la seconde en
    ayant la première en tête.
    """
    def key(asset: Asset) -> str:
        return hashlib.sha256(f"{digest}:{asset.id}".encode("utf-8")).hexdigest()

    return sorted(assets, key=key)


# --- registre de séquences ----------------------------------------------------


@dataclass
class SequenceRegister:
    hotel_id: str
    retrieved_at: str = ""
    endpoint: str = "https://graph.mapillary.com/{image_id}"
    correlation: str = "unknown"
    entries: list[dict] = field(default_factory=list)
    response_digest: str | None = None
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "hotel_id": self.hotel_id,
            "retrieved_at": self.retrieved_at,
            "endpoint": self.endpoint,
            # `known` seulement si la source a répondu : un regroupement par
            # distance n'est pas une séquence, et ne doit pas en prendre le nom.
            "sequence_correlation": self.correlation,
            "response_digest": self.response_digest,
            "error": self.error,
            "entries": self.entries,
            "sequences": self.by_sequence(),
        }

    def by_sequence(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for entry in self.entries:
            sequence = entry.get("sequence_id") or "sans-séquence"
            grouped.setdefault(sequence, []).append(entry["asset_id"])
        return {k: sorted(v) for k, v in sorted(grouped.items())}


def build_register(assets: list[Asset], hotel_id: str, fetch) -> SequenceRegister:  # noqa: ANN001
    """Interroge la source pour les identifiants fournisseur de la cohorte.

    Les identifiants viennent de `asset.id` : les anciennes URL de CDN ne sont
    pas une identité durable, et ont d'ailleurs expiré.
    """
    register = SequenceRegister(
        hotel_id=hotel_id, retrieved_at=datetime.now(timezone.utc).isoformat()
    )
    provider_ids = {a.id.split("-", 1)[1]: a for a in assets}

    try:
        found = fetch(sorted(provider_ids))
    except Exception as exc:  # noqa: BLE001 — panne réseau, quelle qu'en soit la cause
        register.correlation = "unknown"
        register.error = f"{type(exc).__name__} : {exc}"
        log.warning("séquences non récupérées : %s", exc)
        return register

    register.response_digest = hashlib.sha256(
        json.dumps(found, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]

    for provider_id, asset in sorted(provider_ids.items()):
        payload = found.get(provider_id)
        register.entries.append(
            {
                "asset_id": asset.id,
                "provider_image_id": provider_id,
                "sequence_id": (payload or {}).get("sequence"),
                "captured_at": (payload or {}).get("captured_at"),
                "found": payload is not None,
            }
        )

    retrieved = [e for e in register.entries if e["sequence_id"]]
    register.correlation = "known" if len(retrieved) == len(register.entries) else "partial"
    if not retrieved:
        register.correlation = "unknown"
    log.info(
        "séquences : %d/%d image(s), %d séquence(s)",
        len(retrieved), len(register.entries), len(register.by_sequence()),
    )
    return register


# --- instantané des prédictions ------------------------------------------------


def predictions(assets: list[Asset], policy, visibility: dict | None = None) -> dict:  # noqa: ANN001
    """Ce que le système prédit **avant** tout étiquetage.

    Publié séparément et daté : comparer ensuite les décisions humaines à des
    prédictions recalculées après coup comparerait les étiquettes à un système
    qu'elles ont déjà modifié.
    """
    from .roles import role_for

    measures = visibility or {}
    rows = []
    for asset in sorted(assets, key=lambda a: a.id):
        role, reason = role_for(asset, policy)
        measure = measures.get(asset.id, {})
        rows.append(
            {
                "asset_id": asset.id,
                "checksum": asset.checksum,
                "subject_scores": asset.subject_scores,
                "subjects": [s.value for s in asset.subjects],
                "contains_building": asset.contains_building,
                "target_building_visible": asset.target_building_visible,
                "target_evidence": asset.target_evidence,
                "review_status": asset.review_status.value,
                "geometry_suitability": asset.geometry_suitability.value,
                "role": role.value,
                "role_reason": reason,
                "view_sector": asset.view_sector.value,
                "line_of_sight_status": asset.line_of_sight_status,
                "proven_clear_fraction": measure.get("proven_clear_fraction"),
                "risk_unknown_height_fraction": measure.get("risk_unknown_height_fraction"),
                "distance_m": measure.get("distance_m"),
                "target_in_frame_fraction": asset.target_in_frame_fraction,
                "already_reviewed": bool(asset.review_history),
            }
        )

    return {
        "cohort_definition": COHORT_DEFINITION,
        "cohort_digest": cohort_digest(assets),
        "members": len(rows),
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "measures": [
                "précision parmi les candidats détectés",
                "confusions WelcomINNS / Tetra Tech / Toyota",
                "taux d'indécision",
                "aptitude géométrique parmi les images confirmées",
            ],
            "cannot_measure": [
                "le rappel sur les 189 vues Mapillary — les faux négatifs sont "
                "exclus par construction",
                "un seuil OpenCLIP, qui exigerait un échantillon stratifié "
                "distinct, séparé réglage/validation au niveau des séquences",
            ],
        },
        "predictions": rows,
    }
