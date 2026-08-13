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

from pydantic import ValidationError

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


def apply(  # noqa: A001
    assets: list[Asset], payload: dict, strict: bool = True, hotel_id: str | None = None
) -> dict:
    """Réapplique un registre à un manifeste, empreintes vérifiées.

    Une décision porte sur une image précise. Si l'empreinte du manifeste
    diverge de celle inscrite, l'image n'est plus celle qui a été jugée : la
    décision ne la suit pas, et le registre est refusé plutôt qu'appliqué de
    travers.

    Rien n'est modifié tant que tout n'a pas été vérifié.
    """
    by_id = {a.id: index for index, a in enumerate(assets)}
    problems: list[str] = []
    unknown: list[str] = []

    # Chaque candidat est **construit et validé** ici, jamais posé dans
    # `assets`. La version précédente revalidait dans la boucle d'écriture :
    # un registre dont la seconde entrée violait la filiation laissait la
    # première appliquée, et le manifeste sortait à demi modifié d'un appel
    # annoncé comme atomique.
    candidates: list[tuple[int, Asset]] = []

    if not isinstance(payload, dict):
        raise RegisterRefused("registre illisible : un objet JSON est attendu")

    # Un registre appartient à un établissement. L'appliquer à un autre
    # produirait des décisions plausibles sur les mauvaises images.
    declared = payload.get("hotel_id")
    if hotel_id is not None and declared != hotel_id:
        raise RegisterRefused(
            f"registre de {declared!r} appliqué à {hotel_id!r} — "
            "les décisions ne portent pas sur ce corpus"
        )

    records = payload.get("decisions")
    if not isinstance(records, list):
        raise RegisterRefused("registre sans liste 'decisions'")

    seen: set[str] = set()
    for position, record in enumerate(records):
        if not isinstance(record, dict) or "asset_id" not in record or "checksum" not in record:
            problems.append(f"entrée n° {position + 1} : 'asset_id' ou 'checksum' manquant")
            continue

        asset_id = record["asset_id"]
        if asset_id in seen:
            # Deux entrées pour le même asset : la seconde écraserait
            # silencieusement la première, et l'ordre du fichier déciderait de
            # l'historique retenu.
            problems.append(f"{asset_id} : entrée dupliquée dans le registre")
            continue
        seen.add(asset_id)

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

        try:
            update = _update_for(asset, record, problems)
        except ValidationError as exc:
            # Une erreur de schéma dans le registre est un refus, pas une
            # exception qui traverse l'appelant : elle décrit une donnée, non
            # un défaut de programme.
            problems.append(f"{asset_id} : entrée de registre invalide — {exc}")
            continue

        if not update:
            continue

        try:
            candidates.append(
                (index, Asset.model_validate(asset.model_copy(update=update).model_dump()))
            )
        except ValidationError as exc:
            problems.append(
                f"{asset_id} : décisions incohérentes avec le manifeste — {exc}"
            )

    if unknown and strict:
        problems.append(
            f"{len(unknown)} décision(s) sans asset correspondant : {sorted(unknown)[:5]}"
        )
    if problems:
        raise RegisterRefused("; ".join(problems))

    # Seul point d'écriture, atteint uniquement si tout a été vérifié.
    for index, candidate in candidates:
        assets[index] = candidate

    log.info("registre appliqué à %d asset(s)", len(candidates))
    return {"applied": len(candidates), "unknown": sorted(unknown)}


def _update_for(asset: Asset, record: dict, problems: list[str]) -> dict:
    """Champs qu'une entrée de registre poserait sur cet asset.

    Les empreintes sont vérifiées entrée par entrée : une décision porte sur
    une image précise, et un historique dont une seule entrée vise autre chose
    n'est pas rejouable.
    """
    update: dict = {}

    reviews = [ReviewEntry.model_validate(e) for e in record.get("review_history", [])]
    for entry in reviews:
        if entry.reviewed_checksum != asset.checksum:
            problems.append(
                f"{asset.id} : une décision de visibilité porte sur "
                f"{entry.reviewed_checksum[:12]}…, pas sur l'image actuelle"
            )
    if reviews:
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
    for entry in geometry:
        if entry.reviewed_checksum != asset.checksum:
            problems.append(
                f"{asset.id} : une appréciation d'aptitude porte sur "
                f"{entry.reviewed_checksum[:12]}…, pas sur l'image actuelle"
            )
    if geometry:
        update.update(
            geometry_history=geometry,
            geometry_suitability=geometry[-1].suitability,
        )

    return update


def verify_files(assets: list[Asset], workspace_root: Path | None = None) -> list[str]:
    """Confronte l'empreinte déclarée au **contenu réel** des fichiers jugés.

    L'accord registre/manifeste ne prouve que la cohérence de deux
    déclarations. Si l'image sur disque a changé depuis, les deux restent
    d'accord et la décision porte pourtant sur autre chose.
    """
    import hashlib

    problems: list[str] = []
    for asset in assets:
        if not (asset.review_history or asset.geometry_history) or not asset.local_path:
            continue
        path = Path(asset.local_path)
        if workspace_root and not path.is_absolute():
            path = workspace_root / path
        if not path.is_file():
            problems.append(f"{asset.id} : image jugée absente ({path})")
            continue

        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != asset.checksum:
            problems.append(
                f"{asset.id} : fichier {actual[:12]}… au lieu de "
                f"{asset.checksum[:12]}… — l'image a changé depuis sa revue"
            )
    return problems


def path_for(hotel_id: str, root: Path | None = None) -> Path:
    return (root or DECISIONS_DIR) / hotel_id / REGISTER_NAME
