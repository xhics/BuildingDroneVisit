"""Retirer un plan de la circulation sans effacer ce qu'il disait (collecte V2).

Trois brouillons ont été produits avant le contrat `ResolvedAcquisitionRequest`.
Ils annonçaient `256` là où la requête portait `thumb_2048` : leurs neuf
acquisitions auraient été refusées à l'exécution, et leur provenance aurait
décrit un fichier qui n'était pas celui du disque.

Les supprimer effacerait ce qui a réellement été planifié à ce moment-là. Les
réécrire serait pire : un plan qui change après coup ne se relit plus comme ce
qu'il fut. On publie donc un **événement** à côté, et c'est lui qui les retire
de la circulation.

```text
plan          intact, jamais réécrit, jamais supprimé
invalidation  événement immuable qui le nomme, avec son SHA et un motif
sélection     `_latest_plan` ignore ce qu'une invalidation committed nomme
```

Le SHA est ce qui rend l'événement vérifiable : il nomme **ce** fichier-là, non
un identifiant qu'un fichier ultérieur pourrait reprendre. Un motif structuré
plutôt qu'une phrase : c'est ce qui permettra de retrouver toutes les
invalidations d'une même cause.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

from .logging import get_logger

log = get_logger("plan-invalidation")


class InvalidationReason(StrEnum):
    """Pourquoi un plan cesse d'être exécutable.

    Structuré, non libre : une phrase ne se recherche pas, et deux
    invalidations de même cause doivent se reconnaître.
    """

    #: Produit avant que la résolution planifiée n'atteigne la requête
    #: fournisseur. Le plan annonçait une résolution, la requête en portait une
    #: autre, la provenance décrivait la première.
    PRE_RESOLVED_ACQUISITION_REQUEST_CONTRACT = (
        "pre_resolved_acquisition_request_contract"
    )

    #: Les besoins ont changé depuis : le plan répond à une question qu'on ne
    #: pose plus.
    STALE_DEMANDS = "stale_demands"

    #: Décision humaine, motivée dans le champ libre qui l'accompagne.
    OPERATOR_DECISION = "operator_decision"


class InvalidationRefused(RuntimeError):
    """Rien n'a été publié."""


@dataclass
class InvalidatedPlan:
    """Un plan nommé par une invalidation, avec de quoi le reconnaître."""

    plan_id: str
    sha256: str

    def as_dict(self) -> dict:
        return {"plan_id": self.plan_id, "sha256": self.sha256}


@dataclass
class Invalidation:
    """L'événement. Immuable une fois publié."""

    invalidation_id: str
    reason: InvalidationReason
    plans: list[InvalidatedPlan] = field(default_factory=list)

    #: Ce qu'un lecteur doit comprendre, en plus du motif structuré.
    rationale: str = ""
    declared_at: str = ""

    def as_dict(self, **extra) -> dict:
        payload = {
            "invalidation_id": self.invalidation_id,
            "reason": self.reason.value,
            "rationale": self.rationale,
            "declared_at": self.declared_at,
            "plans": [plan.as_dict() for plan in self.plans],
            "note": (
                "les plans nommés restent intacts sur le disque : cet "
                "événement les retire de la circulation, il n'efface pas ce "
                "qu'ils disaient"
            ),
        }
        payload.update(extra)
        return payload


def build(
    plan_paths: list[Path],
    reason: InvalidationReason,
    rationale: str = "",
) -> Invalidation:
    """Construit l'événement à partir des fichiers **existants**.

    Chaque plan est lu pour en prendre l'empreinte : nommer un identifiant sans
    son SHA laisserait l'invalidation porter sur un fichier qu'on n'a pas vu.
    Aucun caractère générique, aucune plage : ce qui n'est pas nommé n'est pas
    invalidé.
    """
    from .transaction import new_transaction_id, sha_of_file

    if not plan_paths:
        raise InvalidationRefused(
            "aucun plan nommé : une invalidation sans objet n'invalide rien"
        )
    if not rationale.strip():
        raise InvalidationRefused(
            "invalidation sans motif lisible : le code structuré dit la "
            "catégorie, non ce qu'un relecteur doit comprendre"
        )

    plans: list[InvalidatedPlan] = []
    for path in plan_paths:
        digest = sha_of_file(path)
        if digest is None:
            raise InvalidationRefused(
                f"{path.name} : plan introuvable — on n'invalide pas ce qu'on "
                "n'a pas lu"
            )
        payload = json.loads(path.read_text("utf-8"))
        plans.append(
            InvalidatedPlan(plan_id=payload["plan_id"], sha256=digest)
        )

    return Invalidation(
        invalidation_id=new_transaction_id(),
        reason=reason,
        plans=plans,
        rationale=rationale.strip(),
        declared_at=datetime.now(timezone.utc).isoformat(),
    )


def invalidated_plan_ids(directory: Path) -> set[str]:
    """Identifiants retirés par une invalidation **committed**.

    Un manifeste préparé n'invalide rien : il décrit une intention, et rien ne
    dit encore qu'elle a abouti. Le lire comme un fait ferait disparaître un
    plan qu'aucun événement n'a retiré.
    """
    retired: set[str] = set()
    for path in sorted(directory.glob("plan_invalidation_*_committed.json")):
        payload = json.loads(path.read_text("utf-8"))
        for plan in payload.get("plans", []):
            plan_id = plan.get("plan_id")
            if plan_id:
                retired.add(plan_id)
    return retired
