"""Décider comment reconstruire ce site — et le motiver (collecte V2).

Deux questions distinctes, que rien ne doit fondre en une :

```text
CaptureDemand + DemandAssessment   ce qui est photographiquement couvert
SiteManifest                       quels objets existent, sont ciblables,
                                   non résolus ou inapplicables
```

Les mêler ferait d'un objet non résolu une lacune de couverture, ou d'une
façade non photographiée un objet inexistant. Ce sont deux manques qui
appellent deux réponses : l'un demande une existence et une géométrie, l'autre
une prise de vue.

**Deux axes, non un seul.** La route dit par quels matériaux le site se
reconstruit ; le statut dit si l'on peut engager :

```text
path             PATH_A_OPEN_3D | PATH_B_PHOTO_FIRST | PATH_C_GEO_FIRST
                 PATH_D_HYBRID  | REJECT
decision_status  ready | capture_required | blocked_prerequisites
                 validation_required
```

**Ce qui déclenche une campagne, et ce qui n'en déclenche pas** :

```text
non ciblable, non critique   forbidden_claim + action de résolution
                             jamais capture_required : aucune caméra
                             ne comble l'absence d'un objet
ciblable, sans photo
ni proxy qualifié            capture_required
ciblable, couvert par un
proxy qualifié               compatible avec ready, sous restrictions
```

**Jamais depuis le nombre brut d'images.** Trois cent treize vues autour d'un
bâtiment peuvent n'en montrer aucune façade utilement. Ce qui compte est
`meets()` — vues **et** continuité mesurée — et l'**union** des points de vue.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .logging import get_logger
from .schemas.acquisition import CaptureDemand, DemandAssessment, DemandStatus
from .schemas.enums import RouterPath

log = get_logger("router")

#: Version du **contrat de décision** : ce que le Router promet d'examiner et
#: dans quel ordre. Elle entre dans l'identité de la décision — deux versions
#: n'ont pas jugé selon les mêmes règles, et leurs verdicts ne se comparent pas
#: même à entrées identiques.
#:
#: 2 : les empreintes du manifeste d'évaluation et de son rapport sont
#: distinctes, et la version entre dans `input_digest`.
ROUTER_CONTRACT_VERSION = 2


class DecisionStatus(StrEnum):
    """Peut-on engager, et sinon qu'est-ce qui manque ?"""

    #: Prêt à engager la reconstruction **par cette route**. Jamais
    #: `ENVIRONMENT_3D_READY`, qui conclut la Phase 1.
    READY = "ready"

    #: Un besoin **ciblable** manque de photo et de proxy : cela se prend à la
    #: caméra.
    CAPTURE_REQUIRED = "capture_required"

    #: Un objet critique n'est pas établi : on ne saurait pas où viser.
    BLOCKED_PREREQUISITES = "blocked_prerequisites"

    #: Couverture suffisante, validation manquante — Gate G5 (SfM).
    VALIDATION_REQUIRED = "validation_required"


class ObjectStanding(StrEnum):
    """Ce que le site dit d'un objet — non ce qu'une photo en montre."""

    #: Établi et géoréférencé : on sait le viser.
    TARGETABLE = "targetable"

    #: Connu, mais sans géométrie : on sait qu'il existe, pas où.
    KNOWN_NOT_TARGETABLE = "known_not_targetable"

    #: Ni existence, ni état temporel établis. Rien ne peut en être affirmé.
    #:
    #: `PARKING_HOTEL` est ici, et non « réfuté » : c'est l'**association** au
    #: stationnement candidat qui a été démentie, pas l'existence d'un
    #: stationnement. Inventer un état « objet réfuté » changerait le sens de
    #: ce qui a été constaté.
    UNRESOLVED = "unresolved"


class MissingInput(RuntimeError):
    """Une entrée absente, implicite ou périmée : la décision est refusée."""


class DecisionConflict(RuntimeError):
    """Deux décisions portent la même identité mais divergent.

    À entrées identiques, le verdict doit être identique. Une divergence
    signifie qu'une entrée non déclarée a pesé — republier écraserait la trace
    de ce défaut.
    """


#: Les empreintes sans lesquelles la décision ne se rattache à rien.
REQUIRED_INPUTS: tuple[str, ...] = (
    "demands_digest",
    #: Le manifeste d'évaluation **et** son rapport : le premier porte les
    #: verdicts, le second les identifiants de points de vue. Une seule
    #: empreinte laissait trois rapports différents produire la même identité.
    "assessment_manifest_digest",
    "assessment_report_digest",
    "site_manifest_digest",
    "capture_geometry_digest",
    "asset_manifest_digest",
    "visibility_application_digest",
    "spatial_reference_digest",
    "preview_assessment_digest",
    "policy_digest",
)


class InputManifest(BaseModel):
    """Ce sur quoi la décision est prise, nommé et empreint.

    Fermé : un champ libre laisserait passer une entrée oubliée, et la décision
    paraîtrait fondée sur un corpus qu'elle n'a pas lu.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    demands_digest: str = ""
    assessment_manifest_digest: str = ""
    assessment_report_digest: str = ""
    site_manifest_digest: str = ""
    capture_geometry_digest: str = ""
    asset_manifest_digest: str = ""
    visibility_application_digest: str = ""
    spatial_reference_digest: str = ""
    preview_assessment_digest: str = ""
    policy_digest: str = ""

    #: Facettes de politique réellement consommées : le digest global change à
    #: chaque retouche, y compris sur une facette étrangère à cette décision.
    policy_facets: tuple[str, ...] = ()

    #: Version du contrat, dans l'identité : deux versions n'ont pas jugé selon
    #: les mêmes règles.
    contract_version: int = Field(default=ROUTER_CONTRACT_VERSION, ge=1)

    @model_validator(mode="after")
    def _the_two_assessment_digests_differ(self) -> "InputManifest":
        """Le manifeste et son rapport sont deux fichiers.

        Les confondre — ou recopier l'un dans l'autre — reproduirait le défaut
        qu'ils existent pour éviter.
        """
        both = self.assessment_manifest_digest, self.assessment_report_digest
        if all(d.strip() for d in both) and both[0] == both[1]:
            raise ValueError(
                "le manifeste d'évaluation et son rapport portent la même "
                "empreinte : ce sont deux fichiers, et les confondre laisserait "
                "des rapports différents produire la même décision"
            )
        return self

    def check(self) -> None:
        """Refuse la décision plutôt que de la rendre sur un corpus inconnu."""
        absents = [
            name for name in REQUIRED_INPUTS
            if not (getattr(self, name) or "").strip()
        ]
        if absents:
            raise MissingInput(
                "entrées manquantes, la décision est refusée : "
                + ", ".join(sorted(absents))
            )

    def as_dict(self) -> dict:
        payload = {name: getattr(self, name) for name in REQUIRED_INPUTS}
        payload["policy_facets"] = sorted(self.policy_facets)
        payload["contract_version"] = self.contract_version
        return payload

    @property
    def digest(self) -> str:
        """Empreinte **déterministe** des entrées.

        Distincte de `decided_at` : un simple rejeu doit rendre la même
        décision, sinon deux documents divergeraient sans qu'aucune entrée
        n'ait changé.
        """
        material = json.dumps(self.as_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


class DemandStanding(BaseModel):
    """Où en est **un** besoin — jugé par le besoin lui-même.

    `viewpoints_required` n'a **pas** de défaut : la valeur vient de la
    politique (`coverage.building_viewpoints_required`), matérialisée dans
    chaque `CaptureDemand`. En inventer une ici ferait décider le Router sur un
    seuil que personne n'a arbitré.
    """

    model_config = ConfigDict(extra="forbid")

    demand_id: str
    status: DemandStatus
    viewpoints_required: int = Field(ge=1)
    viewpoints_found: int = Field(default=0, ge=0)

    #: Résultat de `DemandAssessment.meets(CaptureDemand)` : vues, continuité
    #: **mesurée**, niveau de continuité.
    meets_demand: bool = False

    targetable: bool = True
    viewpoint_ids: tuple[str, ...] = ()
    note: str = ""

    @property
    def satisfied(self) -> bool:
        """Satisfait **et** clos : `meets()` ne regarde pas le statut, qui peut
        déclarer le besoin inatteignable pour une raison qu'aucune métrique ne
        porte."""
        return self.meets_demand and self.status is DemandStatus.MET

    def as_dict(self) -> dict:
        return {
            "demand_id": self.demand_id,
            "status": self.status.value,
            "viewpoints_found": self.viewpoints_found,
            "viewpoints_required": self.viewpoints_required,
            "meets_demand": self.meets_demand,
            "targetable": self.targetable,
            "viewpoint_ids": sorted(self.viewpoint_ids),
            "note": self.note,
        }


def standing_for(
    demand: CaptureDemand,
    assessment: DemandAssessment,
    targetable: bool,
    viewpoint_ids: tuple[str, ...] = (),
    note: str = "",
) -> DemandStanding:
    """Construit l'état d'un besoin **depuis ses propres contrats**."""
    return DemandStanding(
        demand_id=demand.demand_id,
        status=assessment.status,
        viewpoints_required=demand.viewpoints_required,
        viewpoints_found=assessment.viewpoints_found,
        meets_demand=assessment.meets(demand),
        targetable=targetable,
        viewpoint_ids=viewpoint_ids,
        note=note,
    )


class ProxyZone(BaseModel):
    """Un artefact géométrique, et **ce qu'il couvre nommément**.

    Un proxy qualifié ne comble pas ce qu'il ne touche pas : un modèle de
    terrain ne donne ni la façade arrière, ni l'entrée. Sans portée déclarée,
    n'importe quel proxy suffirait à rendre une route hybride, et le document
    annoncerait une couverture qui n'existe pas.

    « Qualifié » ne se déduit pas de la présence d'un fichier : il faut un
    verdict de qualification et les artefacts qui le portent, empreintes
    comprises. `capture_geometry.json` existe sur tout site ayant tourné une
    fois — s'en contenter qualifierait des proxies jamais éprouvés.
    """

    model_config = ConfigDict(extra="forbid")

    zone: str
    artifact: str

    #: Verdict de qualification, non une présence de fichier.
    qualified: bool = False

    #: Rapport qui porte le verdict, et son empreinte.
    qualification_report: str = ""
    qualification_digest: str = ""

    #: Artefacts effectivement retenus, avec leurs empreintes : sans elles, on
    #: ne saurait pas laquelle des dérivations a été employée.
    source_artifacts: tuple[str, ...] = ()
    source_digests: tuple[str, ...] = ()

    covered_objects: tuple[str, ...] = ()
    covered_demands: tuple[str, ...] = ()

    #: Toujours faux : un proxy donne une forme, jamais une apparence.
    appearance_provided: bool = False

    camera_restrictions: tuple[str, ...] = ()
    note: str = ""

    @model_validator(mode="after")
    def _a_qualified_proxy_carries_its_evidence(self) -> "ProxyZone":
        """Qualifié sans preuve serait une affirmation, non un constat."""
        if self.qualified:
            if not self.qualification_digest.strip():
                raise ValueError(
                    f"proxy {self.zone!r} déclaré qualifié sans empreinte de "
                    "rapport : la qualification serait invérifiable"
                )
            if not self.source_artifacts:
                raise ValueError(
                    f"proxy {self.zone!r} déclaré qualifié sans artefact "
                    "source : on ne saurait pas ce qui a été mesuré"
                )
        if self.appearance_provided:
            raise ValueError(
                "un proxy géométrique ne fournit jamais l'apparence : un rendu "
                "texturé sur une forme non observée passerait pour une photo"
            )
        return self

    def covers_object(self, kind: str) -> bool:
        return self.qualified and kind in self.covered_objects

    def covers_demand(self, demand_id: str) -> bool:
        return self.qualified and demand_id in self.covered_demands

    def as_dict(self) -> dict:
        return {
            "zone": self.zone,
            "artifact": self.artifact,
            "qualified": self.qualified,
            "qualification_report": self.qualification_report,
            "qualification_digest": self.qualification_digest,
            "source_artifacts": list(self.source_artifacts),
            "source_digests": list(self.source_digests),
            "covered_objects": sorted(self.covered_objects),
            "covered_demands": sorted(self.covered_demands),
            "appearance_provided": self.appearance_provided,
            "camera_restrictions": list(self.camera_restrictions),
            "note": self.note,
        }


class RouterDecision(BaseModel):
    """La décision, et de quoi la contester."""

    model_config = ConfigDict(extra="forbid")

    hotel_id: str
    path: RouterPath
    decision_status: DecisionStatus
    inputs: InputManifest
    contract_version: int = Field(default=ROUTER_CONTRACT_VERSION, ge=1)

    #: --- côté photographique ------------------------------------------------
    demands_satisfied: list[str] = Field(default_factory=list)
    demands_partial: list[str] = Field(default_factory=list)
    demands_open: list[str] = Field(default_factory=list)
    demands_not_targetable: list[str] = Field(default_factory=list)
    demands_unreachable: list[str] = Field(default_factory=list)

    #: **Union** des identifiants, non une somme.
    independent_viewpoints: int = Field(default=0, ge=0)
    viewpoint_ids: list[str] = Field(default_factory=list)

    #: --- côté site -----------------------------------------------------------
    objects_by_standing: dict = Field(default_factory=dict)
    critical_objects_unestablished: list[str] = Field(default_factory=list)

    #: --- ce qui comble, et par quoi ------------------------------------------
    proxy_zones: list[ProxyZone] = Field(default_factory=list)
    appearance_gaps: list[str] = Field(default_factory=list)
    camera_restrictions: list[str] = Field(default_factory=list)

    #: --- ce qui reste à faire -------------------------------------------------
    rationale: str = ""
    blocking: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    limits: list[str] = Field(default_factory=list)

    decided_at: str = ""

    @model_validator(mode="after")
    def _a_status_matches_what_it_claims(self) -> "RouterDecision":
        """Invariants structurels : le statut doit refléter les listes.

        Sans eux, une décision pourrait annoncer `ready` en portant un objet
        critique non établi — et le document ferait autorité.
        """
        if self.decision_status is DecisionStatus.BLOCKED_PREREQUISITES:
            if not self.critical_objects_unestablished and not self.blocking:
                raise ValueError(
                    "décision bloquée sans prérequis manquant nommé : ce qui "
                    "bloque doit être dit, sinon rien ne peut être débloqué"
                )
        elif self.critical_objects_unestablished:
            raise ValueError(
                "un objet critique n'est pas établi, mais la décision ne "
                f"bloque pas : {sorted(self.critical_objects_unestablished)}"
            )

        if self.independent_viewpoints != len(set(self.viewpoint_ids)):
            raise ValueError(
                f"{self.independent_viewpoints} point(s) de vue annoncé(s) pour "
                f"{len(set(self.viewpoint_ids))} identifiant(s) distinct(s) : "
                "un panorama servant plusieurs besoins reste un point de vue"
            )

        if self.path is RouterPath.PATH_B_PHOTO_FIRST:
            if self.decision_status is DecisionStatus.READY:
                raise ValueError(
                    "Path B ne peut pas être « prêt » avant la validation SfM "
                    "(Gate G5) : ce serait livrer une préparation comme un "
                    "résultat acquis"
                )
        return self

    @property
    def input_digest(self) -> str:
        return self.inputs.digest

    def as_dict(self) -> dict:
        return {
            "hotel_id": self.hotel_id,
            "path": self.path.value,
            "decision_status": self.decision_status.value,
            "contract_version": self.contract_version,
            "input_digest": self.input_digest,
            "inputs": self.inputs.as_dict(),
            "photographic": {
                "satisfied": sorted(self.demands_satisfied),
                "partial": sorted(self.demands_partial),
                "open": sorted(self.demands_open),
                "not_targetable": sorted(self.demands_not_targetable),
                "unreachable": sorted(self.demands_unreachable),
                "independent_viewpoints": self.independent_viewpoints,
                "viewpoint_ids": sorted(self.viewpoint_ids),
            },
            "site": {
                "by_standing": self.objects_by_standing,
                "critical_unestablished": sorted(
                    self.critical_objects_unestablished
                ),
            },
            "geometric_proxies": [zone.as_dict() for zone in self.proxy_zones],
            "appearance_gaps": sorted(self.appearance_gaps),
            "camera_restrictions": sorted(set(self.camera_restrictions)),
            "rationale": self.rationale,
            "blocking": self.blocking,
            "forbidden_claims": self.forbidden_claims,
            "next_actions": self.next_actions,
            "decided_at": self.decided_at,
            "limits": self.limits or [
                "la couverture se juge sur les besoins, jamais sur le nombre "
                "d'images : trois cents vues peuvent n'en montrer aucune "
                "façade utilement",
                "un proxy géométrique donne une forme, non une apparence",
                "un objet non résolu n'est pas une lacune photographique : il "
                "demande une existence et une géométrie, non une prise de vue",
                "« ready » signifie prêt à engager cette route, jamais "
                "ENVIRONMENT_3D_READY, qui conclut la Phase 1",
                "Path A et Path C ne sont pas implémentés : ce Router est "
                "opérationnel pour Path B et Path D, non générique",
            ],
        }


#: Le contenu **sémantique** d'une décision : tout sauf l'instant où elle a été
#: rendue. Deux décisions de même identité doivent s'accorder là-dessus.
def semantic_payload(payload: dict) -> dict:
    """Ce qui doit être identique à entrées identiques.

    `decided_at` change à chaque exécution sans qu'aucune entrée bouge : le
    comparer ferait échouer tout rejeu légitime.
    """
    return {key: value for key, value in payload.items() if key != "decided_at"}


def compare_with_existing(fresh: dict, existing: dict) -> None:
    """Refuse une divergence à identité égale.

    Une différence signifie qu'une entrée non déclarée a pesé sur le verdict.
    Republier écraserait la trace de ce défaut ; `--force` ne doit pas le
    permettre, car forcer ne rend pas la décision cohérente.
    """
    if semantic_payload(fresh) != semantic_payload(existing):
        écarts = sorted(
            key for key in set(semantic_payload(fresh)) | set(semantic_payload(existing))
            if semantic_payload(fresh).get(key) != semantic_payload(existing).get(key)
        )
        raise DecisionConflict(
            "une décision de même identité existe et diverge sur "
            f"{', '.join(écarts)} : une entrée non déclarée a pesé sur le "
            "verdict, et forcer n'y changerait rien"
        )


#: Objets sans lesquels on ne saurait pas où pointer une caméra.
CRITICAL_OBJECTS: frozenset[str] = frozenset({"BUILDING_MAIN"})


def standing_of(state: str, has_geometry: bool) -> ObjectStanding:
    """Traduit l'état d'un objet de site en portée décisionnelle."""
    if state == "unresolved":
        return ObjectStanding.UNRESOLVED
    return (
        ObjectStanding.TARGETABLE if has_geometry
        else ObjectStanding.KNOWN_NOT_TARGETABLE
    )


def _claims_forbidden_by(objects: dict) -> list[str]:
    """Ce dont rien ne peut être affirmé."""
    return [
        f"{kind} : ni existence, ni état temporel, ni géométrie établis — "
        "aucune affirmation à son sujet"
        for kind, standing in sorted(objects.items())
        if standing is ObjectStanding.UNRESOLVED
    ]


def decide(
    hotel_id: str,
    demands: list[DemandStanding],
    objects: dict,
    proxies: list[ProxyZone],
    inputs: InputManifest,
) -> RouterDecision:
    """Arrête une route et un statut, en citant ce qui les fonde."""
    inputs.check()

    satisfied = [d.demand_id for d in demands if d.satisfied and d.targetable]
    unreachable = [
        d.demand_id for d in demands if d.status is DemandStatus.UNREACHABLE
    ]
    partial = [
        d.demand_id for d in demands
        if d.targetable and not d.satisfied and d.viewpoints_found > 0
        and d.demand_id not in unreachable
    ]
    open_demands = [
        d.demand_id for d in demands
        if d.targetable and d.viewpoints_found == 0
        and d.demand_id not in unreachable
    ]
    not_targetable = [
        d.demand_id for d in demands
        if not d.targetable and d.demand_id not in unreachable
    ]

    # Union, non somme : un panorama servant trois besoins reste un point de vue.
    vues: set[str] = set()
    for demand in demands:
        vues.update(demand.viewpoint_ids)

    critiques = [
        kind for kind, standing in objects.items()
        if kind in CRITICAL_OBJECTS and standing is not ObjectStanding.TARGETABLE
    ]

    # Un proxy ne comble que ce qu'il déclare couvrir.
    comblés = {
        demand_id for demand_id in partial + open_demands
        if any(zone.covers_demand(demand_id) for zone in proxies)
    }
    qualifiés = [zone for zone in proxies if zone.qualified]

    #: **Seuls les besoins ciblables** pèsent sur le statut. Un besoin sans
    #: cible n'appelle pas une caméra : aucune prise de vue ne comble l'absence
    #: d'un objet dont on ignore s'il existe. Le compter ici confondrait les
    #: deux sources que le Router sépare.
    reste = [
        demand_id for demand_id in partial + open_demands
        if demand_id not in comblés
    ]

    decision_fields = dict(
        hotel_id=hotel_id,
        inputs=inputs,
        demands_satisfied=satisfied,
        demands_partial=partial,
        demands_open=open_demands,
        demands_not_targetable=not_targetable,
        demands_unreachable=unreachable,
        independent_viewpoints=len(vues),
        viewpoint_ids=sorted(vues),
        objects_by_standing={
            standing.value: sorted(
                kind for kind, s in objects.items() if s is standing
            )
            for standing in ObjectStanding
        },
        critical_objects_unestablished=critiques,
        proxy_zones=proxies,
        appearance_gaps=sorted(comblés),
        camera_restrictions=[
            restriction for zone in qualifiés
            for restriction in zone.camera_restrictions
        ],
        forbidden_claims=_claims_forbidden_by(objects),
        decided_at=datetime.now(timezone.utc).isoformat(),
    )

    #: Ce qu'un objet non ciblable appelle : une résolution, jamais une caméra.
    résolutions = [
        f"établir existence, état temporel et géométrie de {demand_id}"
        for demand_id in sorted(not_targetable)
    ]

    # 1. Un prérequis manquant l'emporte sur tout le reste.
    if critiques:
        return RouterDecision(
            path=RouterPath.REJECT,
            decision_status=DecisionStatus.BLOCKED_PREREQUISITES,
            blocking=[
                f"{kind} : objet critique non établi — on ne saurait pas où viser"
                for kind in sorted(critiques)
            ],
            rationale=(
                "un objet critique n'est pas établi : photographier sans savoir "
                "quoi viser ne produirait rien d'exploitable"
            ),
            next_actions=["établir la géométrie des objets critiques"] + résolutions,
            **decision_fields,
        )

    # 2. Tout ce qui est ciblable est couvert par la photographie : Path B.
    if not partial and not open_demands:
        return RouterDecision(
            path=RouterPath.PATH_B_PHOTO_FIRST,
            # Gate G5 : aucune reconstruction SfM n'a été éprouvée avant le
            # Lot 2. Annoncer « prêt » anticiperait un résultat qu'on n'a pas.
            decision_status=DecisionStatus.VALIDATION_REQUIRED,
            rationale=(
                f"les {len(satisfied)} besoin(s) ciblables sont satisfaits au "
                f"sens de meets() par {len(vues)} point(s) de vue "
                "indépendant(s) ; la validation SfM (Gate G5) reste à faire"
            ),
            next_actions=["valider la reconstruction SfM (Gate G5)"] + résolutions,
            **decision_fields,
        )

    # 3. Des proxies qualifiés comblent nommément ce que la photo ne couvre pas.
    if comblés:
        return RouterDecision(
            path=RouterPath.PATH_D_HYBRID,
            decision_status=(
                DecisionStatus.READY if not reste
                else DecisionStatus.CAPTURE_REQUIRED
            ),
            rationale=(
                f"{len(satisfied)} besoin(s) satisfait(s), {len(partial)} "
                f"partiel(s), {len(open_demands)} ouvert(s) ; "
                f"{len(comblés)} besoin(s) reposent sur un proxy géométrique "
                "qualifié qui les couvre nommément, "
                f"{len(not_targetable)} sans cible établie. La reconstruction "
                "hybride est possible, en sachant ce qui vient d'une photo et "
                "ce qui vient d'une forme."
            ),
            next_actions=[
                f"chercher des vues pour {demand_id}" for demand_id in sorted(reste)
            ] + résolutions,
            **decision_fields,
        )

    # 4. Ni couverture, ni proxy qui couvre ce qui manque.
    return RouterDecision(
        path=RouterPath.PATH_D_HYBRID if qualifiés else RouterPath.REJECT,
        decision_status=DecisionStatus.CAPTURE_REQUIRED,
        rationale=(
            f"{len(reste)} besoin(s) ciblables sans couverture photographique "
            "ni proxy qui les couvre nommément : ce qui manque se prend à la "
            "caméra"
        ),
        next_actions=[
            f"capturer {demand_id}" for demand_id in sorted(reste)
        ] + résolutions,
        **decision_fields,
    )
