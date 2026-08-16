"""Muter un fichier suivi, ou n'en rien faire (collecte V2).

Un reçu écrit **après** la mutation peut manquer : l'interruption laisse un
fichier migré sans trace, qui se relit comme s'il avait toujours eu cette
forme. Un reçu écrit **avant** peut mentir : il affirme une migration qui n'a
pas eu lieu. Les deux ordres sont faux, parce que le problème n'est pas
l'ordre — c'est qu'il manque un état intermédiaire.

D'où trois temps, dont le second seul est irréversible :

```text
prepared    ce qu'on s'apprête à faire, avec les deux SHA. Rien n'a bougé.
mutation    remplacement atomique — le fichier passe de before à after
committed   ce qui a été fait, référençant le manifeste préparé
```

À la reprise, l'empreinte du fichier tranche sans ambiguïté : au SHA initial la
transaction n'a pas été appliquée ; au SHA final elle l'a été et seul le reçu
manque ; à toute autre valeur quelqu'un est passé entre-temps, et deviner
serait pire que refuser.

`write_text` n'est pas atomique : une écriture interrompue laisse un fichier
tronqué, ni l'un ni l'autre état. On écrit donc à côté, puis on renomme —
`os.replace` est atomique sur un même système de fichiers.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .logging import get_logger

log = get_logger("transaction")


class TransactionConflict(RuntimeError):
    """Le fichier n'est ni dans l'état attendu avant, ni dans celui d'après."""


def sha_of(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def sha_of_file(path: Path) -> str | None:
    """Empreinte du fichier, ou `None` s'il n'existe pas."""
    if not path.is_file():
        return None
    return sha_of(path.read_text("utf-8"))


def write_atomic(path: Path, content: str) -> None:
    """Écrit tout ou rien.

    `write_text` tronque puis écrit : interrompue, elle laisse un fichier
    partiel qui n'est aucun des deux états. On passe par un fichier temporaire
    du **même** répertoire — `os.replace` n'est atomique qu'au sein d'un
    système de fichiers — puis on renomme.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    )
    try:
        with handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


@dataclass
class Transaction:
    """Une mutation de fichier, dans ses trois temps."""

    transaction_id: str
    target: Path
    sha_before: str | None
    sha_after: str
    kind: str

    #: Ce que la mutation apporte — chemins ajoutés, motif d'invalidation. Le
    #: manifeste préparé le porte : sans lui, un reçu de récupération ne
    #: saurait pas dire **ce qui** avait été prévu.
    intent: dict = field(default_factory=dict)

    prepared_at: str = ""

    def as_dict(self, **extra) -> dict:
        payload = {
            "transaction_id": self.transaction_id,
            "kind": self.kind,
            "target": str(self.target),
            "sha256_before": self.sha_before,
            "sha256_after": self.sha_after,
            "prepared_at": self.prepared_at,
            "intent": self.intent,
        }
        payload.update(extra)
        return payload


def new_transaction_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def prepare(
    target: Path, after_text: str, kind: str, intent: dict | None = None,
) -> Transaction:
    """Décrit la mutation à venir. **Rien n'a bougé** à ce stade."""
    return Transaction(
        transaction_id=new_transaction_id(),
        target=target,
        sha_before=sha_of_file(target),
        sha_after=sha_of(after_text),
        kind=kind,
        intent=dict(intent or {}),
        prepared_at=datetime.now(timezone.utc).isoformat(),
    )


def commit(
    transaction: Transaction,
    after_text: str,
    publish_prepared,  # noqa: ANN001 — callable(dict) -> None
    publish_committed,  # noqa: ANN001 — callable(dict) -> None
) -> dict:
    """Applique la mutation entre deux publications.

    Le manifeste préparé part avant, le reçu après. Entre les deux, une
    revérification : si le fichier a changé depuis la préparation, quelqu'un
    est passé, et écraser son travail serait pire que refuser.
    """
    publish_prepared(transaction.as_dict(state="prepared"))

    current = sha_of_file(transaction.target)
    if current != transaction.sha_before:
        raise TransactionConflict(
            f"{transaction.target.name} a changé depuis la préparation "
            f"({transaction.sha_before} attendu, {current} trouvé) : "
            "la mutation n'est pas appliquée"
        )

    write_atomic(transaction.target, after_text)

    receipt = transaction.as_dict(
        state="committed",
        committed_at=datetime.now(timezone.utc).isoformat(),
        recovered=False,
    )
    publish_committed(receipt)
    log.info(
        "transaction %s appliquée : %s → %s",
        transaction.transaction_id, transaction.sha_before, transaction.sha_after,
    )
    return receipt


def recover(prepared: dict, target: Path) -> dict:
    """Tranche le sort d'une transaction préparée dont le reçu manque.

    L'empreinte du fichier fait foi : elle dit ce qui s'est passé, là où une
    date ou un ordre d'écriture ne diraient que ce qu'on espérait.
    """
    current = sha_of_file(target)

    if current == prepared.get("sha256_before"):
        return dict(
            prepared,
            state="abandoned",
            recovered=True,
            resolved_at=datetime.now(timezone.utc).isoformat(),
            resolution=(
                "le fichier porte encore son empreinte initiale : la mutation "
                "n'a pas été appliquée, et rien n'est à défaire"
            ),
        )

    if current == prepared.get("sha256_after"):
        return dict(
            prepared,
            state="committed",
            recovered=True,
            resolved_at=datetime.now(timezone.utc).isoformat(),
            resolution=(
                "le fichier porte l'empreinte prévue : la mutation a bien eu "
                "lieu, seul son reçu manquait"
            ),
        )

    raise TransactionConflict(
        f"{target.name} porte {current}, ni l'empreinte initiale "
        f"({prepared.get('sha256_before')}) ni la finale "
        f"({prepared.get('sha256_after')}) : quelqu'un est passé entre-temps. "
        "Deviner serait pire que refuser."
    )


def pending(directory: Path, prefix: str) -> list[dict]:
    """Manifestes préparés sans reçu correspondant.

    Un manifeste préparé n'est **jamais** modifié : c'est le reçu qui atteste,
    et son absence qui signale une transaction à reprendre.
    """
    prepared_files = sorted(directory.glob(f"{prefix}_*_prepared.json"))
    committed = {
        path.name.replace("_committed.json", "")
        for path in directory.glob(f"{prefix}_*_committed.json")
    }
    return [
        json.loads(path.read_text("utf-8"))
        for path in prepared_files
        if path.name.replace("_prepared.json", "") not in committed
    ]
