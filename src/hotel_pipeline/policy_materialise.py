"""Écrire noir sur blanc ce que le code comblait en silence (collecte V2).

Dix chemins de la politique du pilote venaient des valeurs par défaut du code :
deux seuils `geometry`, six paramètres `collection`, et les sections `coverage`
et `adaptive_search` entières. Le fichier paraissait complet ; il l'était pour
la validation, pas pour la lecture.

Ce que cela coûtait : un relecteur ne pouvait pas savoir sur quels seuils un
manifeste avait été produit, et une mise à jour du code aurait déplacé
silencieusement des décisions déjà prises.

**Ce n'est pas un recalibrage.** Les valeurs effectives sont inchangées : ce
sont exactement celles que le code appliquait. La version ne bouge donc pas —
elle qualifie la politique, non sa représentation — et le `policy_digest` reste
identique, ce que le reçu prouve plutôt que d'en donner l'assurance.

```text
avant   fichier partiel + valeurs du code  → digest D
après   fichier complet                    → digest D
```

Un digest qui bougerait signalerait une valeur modifiée en chemin : le reçu le
rendrait visible, et la migration serait à refuser.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .logging import get_logger

log = get_logger("policy-materialise")


class MaterialisationRefused(RuntimeError):
    """La migration changerait une valeur : rien n'est écrit."""


@dataclass
class MaterialisationReceipt:
    """Ce qui a été rendu explicite, et la preuve que rien n'a changé.

    Append-only : chaque migration laisse son reçu, aucun ne remplace un autre.
    Une représentation qui change sans trace se relit comme une politique qui
    n'aurait jamais eu d'autre forme.
    """

    policy_path: str = ""
    sha_before: str = ""
    sha_after: str = ""

    #: Les deux doivent être **identiques** : c'est tout l'objet du reçu.
    digest_before: str = ""
    digest_after: str = ""

    version_before: str = ""
    version_after: str = ""

    #: Transaction qui a porté la mutation. Le manifeste préparé du même
    #: identifiant dit ce qui était prévu, avant qu'on ne sache si ce fut fait.
    transaction_id: str = ""

    #: Chemins rendus explicites, avec la valeur inscrite.
    materialised: dict = field(default_factory=dict)

    #: Champs déjà présents dont la valeur aurait changé. Toujours vide quand
    #: la migration aboutit — le remplir, c'est la refuser.
    altered: dict = field(default_factory=dict)
    removed: list = field(default_factory=list)

    @property
    def values_unchanged(self) -> bool:
        return (
            self.digest_before == self.digest_after
            and not self.altered
            and not self.removed
        )

    def as_dict(self) -> dict:
        return {
            "policy_path": self.policy_path,
            "sha256_before": self.sha_before,
            "sha256_after": self.sha_after,
            "policy_digest_before": self.digest_before,
            "policy_digest_after": self.digest_after,
            "version_before": self.version_before,
            "version_after": self.version_after,
            "transaction_id": self.transaction_id,
            "materialised_paths": sorted(self.materialised),
            "materialised_values": self.materialised,
            "altered_fields": self.altered,
            "removed_fields": self.removed,
            "values_unchanged": self.values_unchanged,
            "note": (
                "migration de **représentation** : les valeurs effectives sont "
                "celles que le code appliquait déjà. La version ne bouge pas — "
                "elle qualifie la politique, non sa forme — et l'égalité des "
                "deux empreintes le prouve"
            ),
        }


def _sha(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _flatten(value, prefix: str = "") -> dict:
    """Aplatit un document en chemins pointés, pour comparer sans ambiguïté."""
    flat: dict = {}
    if isinstance(value, dict):
        for key, item in value.items():
            flat.update(_flatten(item, f"{prefix}{key}."))
    else:
        flat[prefix.rstrip(".")] = value
    return flat


def materialise(
    policy_path: Path,
    publish_receipt=None,  # noqa: ANN001 — rétrocompatibilité des tests
    publish_prepared=None,  # noqa: ANN001
) -> MaterialisationReceipt:
    """Rend explicite tout ce que le code comblait, sans rien changer d'autre.

    Le fichier n'est réécrit qu'après vérification : si une valeur avait
    changé, le reçu le dirait et rien ne serait publié. Écrire d'abord et
    vérifier ensuite laisserait un fichier faux sur le disque le temps de s'en
    apercevoir.
    """
    from .context import implicit_paths
    from .provenance import policy_digest
    from .schemas import PipelinePolicy

    before_text = policy_path.read_text("utf-8")
    raw_before = json.loads(before_text)
    policy = PipelinePolicy.model_validate(raw_before)

    receipt = MaterialisationReceipt(
        policy_path=str(policy_path),
        sha_before=_sha(before_text),
        digest_before=policy_digest(policy),
        version_before=str(raw_before.get("version", "")),
    )

    paths = implicit_paths(policy, raw_before)
    if not paths:
        receipt.sha_after = receipt.sha_before
        receipt.digest_after = receipt.digest_before
        receipt.version_after = receipt.version_before
        log.info("politique déjà complète : rien à matérialiser")
        return receipt

    # Le document **complet** tel que le modèle le voit : c'est exactement ce
    # que le code appliquait, sérialisé.
    raw_after = json.loads(policy.model_dump_json())

    flat_before = _flatten(raw_before)
    flat_after = _flatten(raw_after)

    for path, value in sorted(flat_before.items()):
        if path not in flat_after:
            receipt.removed.append(path)
        elif flat_after[path] != value:
            receipt.altered[path] = {"before": value, "after": flat_after[path]}

    for path in sorted(set(flat_after) - set(flat_before)):
        receipt.materialised[path] = flat_after[path]

    after_text = json.dumps(raw_after, indent=2, ensure_ascii=False) + "\n"
    receipt.sha_after = _sha(after_text)
    receipt.digest_after = policy_digest(PipelinePolicy.model_validate(raw_after))
    receipt.version_after = str(raw_after.get("version", ""))

    if not receipt.values_unchanged:
        raise MaterialisationRefused(
            "la migration changerait la politique effective — "
            f"empreinte {receipt.digest_before} → {receipt.digest_after}, "
            f"{len(receipt.altered)} champ(s) modifié(s), "
            f"{len(receipt.removed)} disparu(s). Rien n'a été écrit."
        )
    if receipt.version_before != receipt.version_after:
        raise MaterialisationRefused(
            "la version a bougé alors que les valeurs sont identiques : une "
            "migration de représentation ne recalibre rien"
        )

    # Trois temps, dont le second seul est irréversible. Un reçu avant la
    # mutation peut mentir ; un reçu après peut manquer. Le manifeste préparé
    # lève l'ambiguïté : à la reprise, l'empreinte du fichier tranche.
    from .transaction import commit, prepare

    transaction = prepare(
        policy_path, after_text, kind="policy_materialisation",
        intent={
            "materialised_paths": sorted(receipt.materialised),
            "policy_digest": receipt.digest_after,
            "version": receipt.version_after,
        },
    )
    receipt.transaction_id = transaction.transaction_id

    commit(
        transaction, after_text,
        publish_prepared=publish_prepared or (lambda _payload: None),
        publish_committed=(
            (lambda _payload: publish_receipt(receipt))
            if publish_receipt is not None
            else (lambda _payload: None)
        ),
    )
    log.info(
        "politique matérialisée : %d chemin(s) explicites, empreinte %s "
        "inchangée",
        len(receipt.materialised), receipt.digest_after,
    )
    return receipt
