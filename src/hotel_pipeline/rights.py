"""Droits d'usage : acquisition factuelle, décision séparée (collecte V2).

`assets acquire --rights owned` permettait d'écrire un statut juridique sans la
moindre preuve. L'acquisition constate un fait — ce fichier vient de là — et ne
tranche rien : une source tierce téléchargée est `public_uncleared`, quelles
que soient les intentions de l'opérateur.

Deux gestes distincts s'y substituent, et les confondre est précisément ce
qu'on empêche :

```text
rights clear        une autorisation existe et se prouve
                    → licensed / open_data / owned, avec preuves

rights assume-risk  aucune autorisation, et on avance quand même
                    → reste public_uncleared, rights_encumbered=true
                    → l'acceptation du risque est tracée, l'état juridique intact
```

Le second n'améliore pas les droits : il déclare qu'on les enfreint peut-être,
en connaissance de cause. Falsifier l'état juridique pour se donner le droit de
continuer aurait rendu le manifeste inutilisable comme preuve de diligence.
"""

from __future__ import annotations

from .logging import get_logger
from .schemas.enums import Rights
from .schemas.rights import CLEARABLE, RightsAction, RightsDecision

log = get_logger("rights")

def effect(decision: RightsDecision) -> dict:
    """Ce qu'une décision change sur l'asset, et rien de plus.

    `assume_risk` ne touche pas `rights` : c'est tout l'objet de la séparation.
    L'état juridique reste ce qu'il est — non établi — et le risque assumé se
    lit à côté.
    """
    if decision.action is RightsAction.CLEAR:
        return {
            "rights": decision.granted_rights,
            "rights_encumbered": False,
            "rights_note": f"{decision.scope} — {decision.rationale}",
        }
    if decision.action is RightsAction.ASSUME_RISK:
        return {
            "rights_encumbered": True,
            "rights_note": f"risque assumé ({decision.scope}) — {decision.rationale}",
        }
    return {
        "rights": Rights.PUBLIC_UNCLEARED,
        "rights_encumbered": False,
        "rights_note": f"révoqué — {decision.rationale}",
    }


def apply(asset, decision: RightsDecision):  # noqa: ANN001, ANN201
    """Inscrit une décision à l'historique, et en applique l'effet.

    Refuse une décision portant sur un autre fichier : les droits examinés
    étaient ceux d'une image qui n'est plus celle-ci.
    """
    if decision.reviewed_checksum != asset.checksum:
        raise ValueError(
            f"asset {asset.id!r} : décision prise sur l'empreinte "
            f"{decision.reviewed_checksum[:12]}…, le fichier porte "
            f"{asset.checksum[:12]}… — l'examen ne portait pas sur ce fichier"
        )

    history = [*asset.rights_history, decision]
    log.info(
        "%s : droits — %s par %s (%s)",
        asset.id, decision.action.value, decision.decided_by, decision.scope,
    )
    return asset.model_copy(update={"rights_history": history, **effect(decision)})


def acquisition_rights(source_licence_claim: str | None = None) -> dict:
    """Ce qu'une acquisition écrit, et c'est tout.

    Un fichier téléchargé d'une source tierce est `public_uncleared` : indexé
    publiquement, droits non établis. La licence revendiquée par le fournisseur
    est conservée comme **revendication**, jamais comme autorisation — afficher
    « CC BY » ne prouve pas qu'on détenait les droits de l'accorder.
    """
    return {
        "rights": Rights.PUBLIC_UNCLEARED,
        "rights_encumbered": False,
        "rights_note": (
            f"licence revendiquée par la source : {source_licence_claim}"
            if source_licence_claim
            else None
        ),
    }
