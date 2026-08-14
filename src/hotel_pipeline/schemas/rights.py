"""Vocabulaire des décisions de droits (collecte V2).

Séparé de `hotel_pipeline.rights`, qui porte la logique : `Asset` doit pouvoir
déclarer son historique sans dépendre du module qui l'applique.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import Rights


#: Droits qu'une décision d'autorisation peut établir. `public_uncleared` et
#: `unknown` n'y figurent pas : ils décrivent une absence, et on ne décide pas
#: d'une absence.
CLEARABLE: frozenset[Rights] = frozenset(
    {Rights.OWNED, Rights.LICENSED, Rights.OPEN_DATA}
)


class RightsAction(StrEnum):
    """Ce qu'une décision fait."""

    #: Une autorisation existe, et la décision l'enregistre.
    CLEAR = "clear"

    #: Aucune autorisation ; le risque est accepté et tracé.
    ASSUME_RISK = "assume_risk"

    #: Retour en arrière : la preuve invoquée ne tient pas.
    REVOKE = "revoke"


class RightsDecision(BaseModel):
    """Une décision de droits, immuable et opposable.

    Immuable par convention, comme les revues : on n'édite jamais une décision,
    on en ajoute une qui la corrige. L'empreinte du fichier y figure — une
    autorisation porte sur ce qui a été examiné, et si le fichier change, elle
    ne le suit pas.
    """

    model_config = ConfigDict(extra="forbid")

    action: RightsAction
    decided_by: str = Field(min_length=1)
    decided_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    rationale: str = Field(min_length=1)

    #: Empreinte du fichier au moment de la décision.
    reviewed_checksum: str = Field(min_length=64, max_length=64)

    #: Portée : ce que l'autorisation couvre. « usage interne » et « diffusion
    #: publique » ne sont pas la même permission, et les confondre ferait
    #: publier ce qui n'était autorisé qu'en interne.
    scope: str = Field(min_length=1)

    #: Droits établis. Obligatoire pour `clear`, interdit ailleurs : accepter
    #: un risque n'améliore aucun droit.
    granted_rights: Rights | None = None

    #: Preuves. Obligatoires pour `clear` — une autorisation sans preuve est
    #: une affirmation — et exigées aussi pour `assume_risk`, où elles disent
    #: ce qui a été examiné avant d'accepter.
    evidence: list[str] = Field(default_factory=list)

    #: Licence revendiquée **par la source**, conservée telle quelle. Ce n'est
    #: pas une autorisation : un fournisseur qui affiche « CC BY » ne prouve
    #: pas qu'il détenait les droits de l'accorder.
    source_licence_claim: str | None = None

    supersedes_index: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _action_matches_its_effect(self) -> "RightsDecision":
        if self.action is RightsAction.CLEAR:
            if self.granted_rights is None:
                raise ValueError(
                    "décision « clear » sans droits établis : elle n'accorde rien"
                )
            if self.granted_rights not in CLEARABLE:
                raise ValueError(
                    f"« clear » ne peut pas établir {self.granted_rights.value!r} : "
                    f"ce statut décrit une absence de droits. Attendu l'un de "
                    f"{sorted(r.value for r in CLEARABLE)}"
                )
            if not self.evidence:
                raise ValueError(
                    "autorisation sans preuve : une autorisation qu'on ne peut "
                    "pas produire est une affirmation"
                )
        elif self.granted_rights is not None:
            raise ValueError(
                f"« {self.action.value} » n'établit aucun droit : accepter un "
                "risque ou révoquer n'améliore pas l'état juridique"
            )

        if self.action is RightsAction.ASSUME_RISK and not self.evidence:
            raise ValueError(
                "risque accepté sans preuve de ce qui a été examiné : "
                "l'acceptation doit dire sur quoi elle porte"
            )
        return self
