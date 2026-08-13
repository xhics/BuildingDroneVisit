"""Registre versionnable des décisions humaines (Lot 1B §6).

`work/` est ignoré par Git : les décisions y vivaient donc hors du dépôt, et
un « commit des décisions » n'aurait contenu aucune décision. Elles sont
pourtant ce que le pipeline ne peut pas régénérer — tout le reste se recalcule.

Le registre les extrait dans `decisions/<hotel>/asset_reviews.json` : les
historiques de visibilité et d'aptitude, l'identifiant et l'empreinte de
chaque image, jamais les images elles-mêmes. Rejouer une revue consiste alors
à réappliquer ce fichier à un manifeste, en vérifiant que chaque décision
porte bien sur l'image qu'elle dit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .logging import get_logger
from .schemas import Asset, GeometryEntry, ReviewEntry
from .schemas.assets import DECISION_STATUS, VISIBILITY_OF

log = get_logger("decisions")

#: Racine du registre, hors `work/` pour être versionnable.
DECISIONS_DIR = Path("decisions")

REGISTER_NAME = "asset_reviews.json"


class RegisterRefused(RuntimeError):
    """Le registre n'a pas été appliqué, et rien n'a été modifié."""


@dataclass
class AssetDecisions:
    asset_id: str
    checksum: str
    source: str = ""
    source_url_or_id: str = ""
    review_history: list[dict] = field(default_factory=list)
    geometry_history: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "checksum": self.checksum,
            "source": self.source,
            "source_url_or_id": self.source_url_or_id,
            "review_history": self.review_history,
            "geometry_history": self.geometry_history,
        }


@dataclass
class Register:
    hotel_id: str
    exported_at: str = ""
    decisions: list[AssetDecisions] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "hotel_id": self.hotel_id,
            "exported_at": self.exported_at,
            "note": (
                "Décisions humaines seules. Les images ne sont pas versionnées ; "
                "leur empreinte l'est, et conditionne l'application du registre."
            ),
            "decisions": [d.as_dict() for d in self.decisions],
        }


def export(assets: list[Asset], hotel_id: str) -> Register:
    """Extrait tout ce qui a été décidé, et rien d'autre."""
    import json

    register = Register(
        hotel_id=hotel_id, exported_at=datetime.now(timezone.utc).isoformat()
    )
    for asset in sorted(assets, key=lambda a: a.id):
        if not (asset.review_history or asset.geometry_history):
            continue
        register.decisions.append(
            AssetDecisions(
                asset_id=asset.id,
                checksum=asset.checksum,
                source=asset.source,
                source_url_or_id=asset.source_url_or_id,
                review_history=[json.loads(e.model_dump_json()) for e in asset.review_history],
                geometry_history=[
                    json.loads(e.model_dump_json()) for e in asset.geometry_history
                ],
            )
        )
    log.info("registre : %d asset(s) portant une décision", len(register.decisions))
    return register


def apply(assets: list[Asset], payload: dict, strict: bool = True) -> dict:  # noqa: A001
    """Réapplique un registre à un manifeste, empreintes vérifiées.

    Une décision porte sur une image précise. Si l'empreinte du manifeste
    diverge de celle inscrite, l'image n'est plus celle qui a été jugée : la
    décision ne la suit pas, et le registre est refusé plutôt qu'appliqué de
    travers.

    Rien n'est modifié tant que tout n'a pas été vérifié.
    """
    by_id = {a.id: index for index, a in enumerate(assets)}
    planned: list[tuple[int, dict]] = []
    problems: list[str] = []
    unknown: list[str] = []

    for record in payload.get("decisions", []):
        asset_id = record["asset_id"]
        index = by_id.get(asset_id)
        if index is None:
            unknown.append(asset_id)
            continue

        asset = assets[index]
        if asset.checksum != record["checksum"]:
            problems.append(
                f"{asset_id} : empreinte {asset.checksum[:12]}… au manifeste, "
                f"{record['checksum'][:12]}… au registre — l'image a changé"
            )
            continue

        update: dict = {}
        reviews = [ReviewEntry.model_validate(e) for e in record.get("review_history", [])]
        if reviews:
            for entry in reviews:
                if entry.reviewed_checksum != asset.checksum:
                    problems.append(
                        f"{asset_id} : une décision de visibilité porte sur "
                        f"{entry.reviewed_checksum[:12]}…, pas sur l'image actuelle"
                    )
            last = reviews[-1]
            update.update(
                review_history=reviews,
                target_visibility_decision=last.decision,
                review_status=DECISION_STATUS[last.decision],
                target_building_visible=VISIBILITY_OF[last.decision],
                reviewer=last.decided_by,
                reviewed_at=last.decided_at,
                review_rationale=last.rationale,
                review_evidence=last.evidence,
                target_evidence=f"revue humaine : {last.rationale}",
            )

        geometry = [GeometryEntry.model_validate(e) for e in record.get("geometry_history", [])]
        if geometry:
            for entry in geometry:
                if entry.reviewed_checksum != asset.checksum:
                    problems.append(
                        f"{asset_id} : une appréciation d'aptitude porte sur "
                        f"{entry.reviewed_checksum[:12]}…, pas sur l'image actuelle"
                    )
            update.update(
                geometry_history=geometry,
                geometry_suitability=geometry[-1].suitability,
            )

        if update:
            planned.append((index, update))

    if unknown and strict:
        problems.append(
            f"{len(unknown)} décision(s) sans asset correspondant : {sorted(unknown)[:5]}"
        )
    if problems:
        raise RegisterRefused("; ".join(problems))

    for index, update in planned:
        # Revalidation systématique : `model_copy` ne vérifie rien, et un
        # registre corrompu passerait les invariants du manifeste.
        assets[index] = Asset.model_validate(
            assets[index].model_copy(update=update).model_dump()
        )

    log.info("registre appliqué à %d asset(s)", len(planned))
    return {"applied": len(planned), "unknown": sorted(unknown)}


def path_for(hotel_id: str, root: Path | None = None) -> Path:
    return (root or DECISIONS_DIR) / hotel_id / REGISTER_NAME
