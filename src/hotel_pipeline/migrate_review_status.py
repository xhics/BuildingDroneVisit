"""Migration du statut de revue vers l'état terminal `human_unresolved`.

Les manifestes antérieurs écrivaient `needs_review` pour une revue non
conclusive. Cette valeur mêlait deux situations opposées — personne n'a jugé,
et personne ne peut trancher — si bien qu'une image examinée revenait dans la
file à chaque exécution.

La migration travaille sur le **JSON brut**, avant validation : le manifeste à
convertir est justement celui que le modèle refuse désormais. Elle ne touche ni
aux historiques, ni aux décisions, ni aux verdicts de visibilité — elle renomme
un état, et seulement lorsque l'historique le prouve.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .logging import get_logger
from .schemas import AssetManifest, ReviewDecision, ReviewStatus

log = get_logger("migrate-review-status")

_LEGACY = ReviewStatus.NEEDS_REVIEW.value
_TERMINAL = ReviewStatus.HUMAN_UNRESOLVED.value


@dataclass
class StatusMigrationReport:
    """Ce que la migration a changé, et ce qu'elle a délibérément laissé."""

    total: int = 0
    converted: int = 0
    already_terminal: int = 0
    never_reviewed: int = 0
    converted_ids: list[str] = field(default_factory=list)
    untouched_decisions: bool = True

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "converted": self.converted,
            "already_terminal": self.already_terminal,
            "never_reviewed": self.never_reviewed,
            "converted_ids": self.converted_ids,
            "untouched_decisions": self.untouched_decisions,
            "note": (
                "seul le statut change, et seulement là où l'historique porte "
                "une dernière décision 'unresolved' : décisions, historiques, "
                "empreintes et verdicts de visibilité sont inchangés"
            ),
        }


def needs_migration(payload: dict) -> bool:
    """Le manifeste brut contient-il au moins un statut à convertir ?"""
    return any(_is_legacy(asset) for asset in payload.get("assets", []))


def _is_legacy(asset: dict) -> bool:
    """Une revue close sans conclusion, encore déclarée « en attente ».

    L'historique est seul juge : convertir sur la foi du champ plat
    `target_visibility_decision` reviendrait à faire confiance à la valeur par
    défaut, qui vaut `unresolved` pour tout asset jamais examiné.
    """
    history = asset.get("review_history") or []
    if not history:
        return False
    return (
        asset.get("review_status") == _LEGACY
        and history[-1].get("decision") == ReviewDecision.UNRESOLVED.value
    )


def migrate_payload(payload: dict) -> tuple[dict, StatusMigrationReport]:
    """Convertit un manifeste brut. Aucune autre clé n'est touchée."""
    report = StatusMigrationReport()
    assets = payload.get("assets", [])
    report.total = len(assets)

    for asset in assets:
        history = asset.get("review_history") or []
        if not history:
            report.never_reviewed += 1
            continue
        if asset.get("review_status") == _TERMINAL:
            report.already_terminal += 1
            continue
        if _is_legacy(asset):
            asset["review_status"] = _TERMINAL
            report.converted += 1
            report.converted_ids.append(asset.get("id", "?"))

    log.info(
        "statut de revue : %d converti(s) sur %d asset(s)", report.converted, report.total
    )
    return payload, report


def migrate_file(path: Path) -> tuple[AssetManifest, StatusMigrationReport]:
    """Migre le fichier, puis **valide** le résultat avant de le rendre.

    Valider après conversion est le contrôle qui compte : si la migration
    laissait une incohérence, le manifeste ne se construirait pas, et rien ne
    serait écrit.
    """
    payload = json.loads(path.read_text("utf-8"))
    before = _without_status(payload)

    migrated, report = migrate_payload(payload)
    manifest = AssetManifest.model_validate(migrated)

    report.untouched_decisions = before == _without_status(migrated)
    if not report.untouched_decisions:
        raise ValueError(
            "migration refusée : un champ autre que le statut de revue a changé"
        )
    return manifest, report


def _without_status(payload: dict) -> str:
    """Le manifeste privé de son seul champ migrable, sous forme comparable.

    Comparer les textes après avoir retiré `review_status` prouve la non-perte
    sans avoir à énumérer ce qui doit être préservé : tout le reste est dans la
    comparaison, y compris les champs qu'on n'aurait pas pensé à citer.
    """
    stripped = [
        {key: value for key, value in asset.items() if key != "review_status"}
        for asset in payload.get("assets", [])
    ]
    return json.dumps(stripped, sort_keys=True, ensure_ascii=False)
