"""Décider comment reconstruire ce site — et le motiver (collecte V2).

Deux questions distinctes, que rien ne doit fondre en une :

```text
CaptureDemand + DemandAssessment   ce qui est photographiquement couvert
SiteManifest                       quels objets existent, sont ciblables,
                                   non résolus ou inapplicables
```

Les mêler ferait d'un objet non résolu une lacune de couverture, ou d'une
façade non photographiée un objet inexistant. Ce sont deux manques qui
appellent deux réponses : l'un demande une localisation ou une preuve, l'autre
une prise de vue.

**Deux axes, non un seul.** La route dit par quels matériaux le site se
reconstruit ; le statut dit si l'on peut engager. `capture_required` et
`blocked_prerequisites` sont des états d'une route, non des routes :

```text
path             PATH_A_OPEN_3D | PATH_B_PHOTO_FIRST | PATH_C_GEO_FIRST
                 PATH_D_HYBRID  | REJECT
decision_status  ready | capture_required | blocked_prerequisites
                 validation_required
```

**Jamais depuis le nombre brut d'images.** Trois cent treize vues autour d'un
bâtiment peuvent n'en montrer aucune façade utilement ; sur ce site, six
acquisitions ont été réfutées une à une. Ce qui compte est `meets()` — vues
**et** continuité mesurée — et l'**union** des points de vue : un panorama qui
sert trois besoins reste un point de vue.

La décision cite ce sur quoi elle se fonde, et ses entrées sont **fermées** :
une empreinte absente refuse la décision plutôt que de la rendre sur un corpus
inconnu.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum

from .logging import get_logger
from .schemas.acquisition import CaptureDemand, DemandAssessment, DemandStatus
from .schemas.enums import RouterPath

log = get_logger("router")

#: Version du **contrat de décision** : ce que le Router promet d'examiner et
#: dans quel ordre. La changer périme les décisions antérieures, qui ont été
#: prises sur d'autres règles.
ROUTER_CONTRACT_VERSION = 1


class DecisionStatus(StrEnum):
    """Peut-on engager, et sinon qu'est-ce qui manque ?

    Orthogonal à la route : `PATH_D_HYBRID` peut être `ready` ou
    `capture_required` selon ce qui est couvert, sans changer de matériaux.
    """

    #: Prêt à engager la reconstruction **par cette route**. Jamais
    #: `ENVIRONMENT_3D_READY`, qui est un résultat de fin de Phase 1.
    READY = "ready"

    #: Ce qui manque se prend à la caméra. Aucun obstacle ne l'empêche : c'est
    #: une campagne, non un blocage.
    CAPTURE_REQUIRED = "capture_required"

    #: Quelque chose doit être établi avant toute capture : un objet critique
    #: non localisé, un référentiel absent. Envoyer quelqu'un sur place sans
    #: cela ferait photographier au hasard.
    BLOCKED_PREREQUISITES = "blocked_prerequisites"

    #: La couverture paraît suffisante, mais une validation manque encore —
    #: Gate G5 (SfM) pour Path B. Avant le Lot 2, aucune reconstruction
    #: photogrammétrique n'a été éprouvée : l'annoncer prête serait annoncer
    #: un résultat qu'on n'a pas.
    VALIDATION_REQUIRED = "validation_required"


class ObjectStanding(StrEnum):
    """Ce que le site dit d'un objet — non ce qu'une photo en montre."""

    #: Établi et géoréférencé : on sait le viser.
    TARGETABLE = "targetable"

    #: Connu, mais sans géométrie : on sait qu'il existe, pas où.
    KNOWN_NOT_TARGETABLE = "known_not_targetable"

    #: Instancié au gabarit, sans que rien n'établisse son existence. Ni son
    #: existence, ni son état temporel ne sont acquis : rien ne peut en être
    #: affirmé.
    UNRESOLVED = "unresolved"

    #: Démenti par une preuve : ce n'est pas une absence de données, c'est un
    #: constat. Sur ce site, le stationnement associé par proximité montrait le
    #: bâtiment voisin.
    REFUTED = "refuted"


class MissingInput(RuntimeError):
    """Une entrée absente, implicite ou périmée : la décision est refusée.

    Rendre une route sur un corpus inconnu produirait un document qui paraît
    fondé sans l'être — pire qu'une absence de décision.
    """


#: Les empreintes sans lesquelles la décision ne se rattache à rien. Une
#: décision se relit des mois plus tard : ce qu'elle ne nomme pas ne pourra
#: plus être retrouvé.
REQUIRED_INPUTS: tuple[str, ...] = (
    "demands_digest",
    "assessment_report_digest",
    "site_manifest_digest",
    "capture_geometry_digest",
    "asset_manifest_digest",
    "visibility_application_digest",
    "spatial_reference_digest",
    "preview_assessment_digest",
    "policy_digest",
)


@dataclass(frozen=True)
class InputManifest:
    """Ce sur quoi la décision est prise, nommé et empreint.

    Fermé : un champ libre laisserait passer une entrée oubliée, et la
    décision paraîtrait fondée sur un corpus qu'elle n'a pas lu.
    """

    demands_digest: str = ""
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


@dataclass
class DemandStanding:
    """Où en est **un** besoin — jugé par le besoin lui-même.

    `viewpoints_required` n'a **pas** de défaut : la valeur vient de la
    politique (`coverage.building_viewpoints_required`), matérialisée dans
    chaque `CaptureDemand`. En inventer une ici ferait décider le Router sur un
    seuil que personne n'a arbitré.
    """

    demand_id: str
    status: DemandStatus
    viewpoints_required: int
    viewpoints_found: int = 0

    #: Résultat de `DemandAssessment.meets(CaptureDemand)` : vues, continuité
    #: **mesurée**, niveau de continuité. Un compte de vues seul laisserait
    #: deux clichés sans recouvrement passer pour une couverture SfM.
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
    """Construit l'état d'un besoin **depuis ses propres contrats**.

    Le seuil de points de vue est lu sur le besoin, jamais choisi ici.
    """
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


@dataclass(frozen=True)
class ProxyZone:
    """Un artefact géométrique, et **ce qu'il couvre nommément**.

    Un proxy qualifié ne comble pas ce qu'il ne touche pas : un modèle de
    terrain ne donne ni la façade arrière, ni l'entrée. Sans déclaration de
    portée, n'importe quel proxy qualifié suffirait à rendre une route hybride
    — et le document annoncerait une couverture qui n'existe pas.

    Un proxy donne une **forme**, jamais une apparence : `appearance_provided`
    reste faux, et les plans rapprochés restent interdits sur ces zones, faute
    de quoi un rendu texturé passerait pour une observation.
    """

    zone: str
    artifact: str
    qualified: bool

    #: Objets de site que ce proxy couvre réellement.
    covered_objects: tuple[str, ...] = ()

    #: Besoins dont il comble la part **géométrique**, jamais l'apparence.
    covered_demands: tuple[str, ...] = ()

    #: Toujours faux : un proxy ne montre pas à quoi les choses ressemblent.
    appearance_provided: bool = False

    #: Ce que la caméra ne doit pas faire sur ces zones.
    camera_restrictions: tuple[str, ...] = ()

    note: str = ""

    def covers_object(self, kind: str) -> bool:
        return self.qualified and kind in self.covered_objects

    def covers_demand(self, demand_id: str) -> bool:
        return self.qualified and demand_id in self.covered_demands

    def as_dict(self) -> dict:
        return {
            "zone": self.zone,
            "artifact": self.artifact,
            "qualified": self.qualified,
            "covered_objects": sorted(self.covered_objects),
            "covered_demands": sorted(self.covered_demands),
            "appearance_provided": self.appearance_provided,
            "camera_restrictions": list(self.camera_restrictions),
            "note": self.note,
        }


@dataclass
class RouterDecision:
    """La décision, et de quoi la contester."""

    hotel_id: str
    path: RouterPath
    decision_status: DecisionStatus
    inputs: InputManifest
    contract_version: int = ROUTER_CONTRACT_VERSION

    #: --- ce qui fonde la décision, côté photographique ---------------------
    demands_satisfied: list[str] = field(default_factory=list)
    demands_partial: list[str] = field(default_factory=list)
    demands_open: list[str] = field(default_factory=list)
    demands_not_targetable: list[str] = field(default_factory=list)
    demands_unreachable: list[str] = field(default_factory=list)

    #: **Union** des identifiants, non une somme : un panorama servant trois
    #: besoins reste un point de vue.
    independent_viewpoints: int = 0
    viewpoint_ids: list[str] = field(default_factory=list)

    #: --- ce qui fonde la décision, côté site --------------------------------
    objects_by_standing: dict = field(default_factory=dict)
    critical_objects_unestablished: list[str] = field(default_factory=list)

    #: --- ce qui comble, et par quoi ----------------------------------------
    proxy_zones: list[ProxyZone] = field(default_factory=list)

    #: Ce que les proxies **ne** fournissent pas : dit explicitement, sans quoi
    #: la route hybride laisserait croire à une couverture complète.
    appearance_gaps: list[str] = field(default_factory=list)
    camera_restrictions: list[str] = field(default_factory=list)

    #: --- ce qui reste à faire ----------------------------------------------
    rationale: str = ""
    blocking: list[str] = field(default_factory=list)
    forbidden_claims: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    limits: list[str] = field(default_factory=list)

    decided_at: str = ""

    @property
    def input_digest(self) -> str:
        return self.inputs.digest

    def as_dict(self) -> dict:
        return {
            "hotel_id": self.hotel_id,
            "path": self.path.value,
            "decision_status": self.decision_status.value,
            "contract_version": self.contract_version,
            #: Déterministe : un rejeu sans changement d'entrée rend la même
            #: valeur, quand `decided_at` diffère à chaque exécution.
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
            ],
        }


#: Objets sans lesquels on ne saurait pas où pointer une caméra. Leur absence
#: bloque ; celle des autres se documente.
CRITICAL_OBJECTS: frozenset[str] = frozenset({"BUILDING_MAIN"})


def standing_of(state: str, has_geometry: bool, refuted: bool = False) -> ObjectStanding:
    """Traduit l'état d'un objet de site en portée décisionnelle."""
    if refuted:
        return ObjectStanding.REFUTED
    if state == "unresolved":
        return ObjectStanding.UNRESOLVED
    return (
        ObjectStanding.TARGETABLE if has_geometry
        else ObjectStanding.KNOWN_NOT_TARGETABLE
    )


def _claims_forbidden_by(objects: dict) -> list[str]:
    """Ce dont rien ne peut être affirmé.

    Un objet non résolu n'a ni existence, ni état temporel établis : dire
    « l'entrée actuelle se trouve là » supposerait deux faits qu'aucun artefact
    ne porte.
    """
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
    """Arrête une route et un statut, en citant ce qui les fonde.

    L'ordre des tests n'est pas indifférent : un prérequis manquant l'emporte
    sur toute couverture, car photographier sans savoir quoi viser ne produit
    rien d'exploitable.
    """
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
    independants = len(vues) if vues else 0

    critiques = [
        kind for kind, standing in objects.items()
        if kind in CRITICAL_OBJECTS and standing is not ObjectStanding.TARGETABLE
    ]

    # Un proxy ne comble que ce qu'il déclare couvrir.
    comblés = {
        demand_id for demand_id in partial + open_demands + not_targetable
        if any(zone.covers_demand(demand_id) for zone in proxies)
    }
    qualifiés = [zone for zone in proxies if zone.qualified]

    decision = RouterDecision(
        hotel_id=hotel_id,
        path=RouterPath.REJECT,
        decision_status=DecisionStatus.BLOCKED_PREREQUISITES,
        inputs=inputs,
        demands_satisfied=satisfied,
        demands_partial=partial,
        demands_open=open_demands,
        demands_not_targetable=not_targetable,
        demands_unreachable=unreachable,
        independent_viewpoints=independants,
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

    # 1. Un prérequis manquant l'emporte sur tout le reste.
    if critiques:
        decision.decision_status = DecisionStatus.BLOCKED_PREREQUISITES
        decision.blocking = [
            f"{kind} : objet critique non établi — on ne saurait pas où viser"
            for kind in sorted(critiques)
        ]
        decision.rationale = (
            "un objet critique n'est pas établi : photographier sans savoir "
            "quoi viser ne produirait rien d'exploitable"
        )
        decision.next_actions = ["établir la géométrie des objets critiques"]
        return decision

    reste = [
        demand_id for demand_id in partial + open_demands + not_targetable
        if demand_id not in comblés
    ]

    # 2. Tout couvert par la photographie : Path B — mais non validé.
    if not partial and not open_demands and not not_targetable:
        decision.path = RouterPath.PATH_B_PHOTO_FIRST
        # Gate G5 : aucune reconstruction SfM n'a été éprouvée avant le Lot 2.
        # Annoncer « prêt » anticiperait un résultat qu'on n'a pas.
        decision.decision_status = DecisionStatus.VALIDATION_REQUIRED
        decision.rationale = (
            f"les {len(satisfied)} besoin(s) sont satisfaits au sens de "
            f"meets() par {independants} point(s) de vue indépendant(s) ; "
            "la validation SfM (Gate G5) reste à faire"
        )
        decision.next_actions = ["valider la reconstruction SfM (Gate G5)"]
        return decision

    # 3. Des proxies qualifiés comblent nommément ce que la photo ne couvre pas.
    if comblés:
        decision.path = RouterPath.PATH_D_HYBRID
        decision.decision_status = (
            DecisionStatus.READY if not reste
            else DecisionStatus.CAPTURE_REQUIRED
        )
        decision.rationale = (
            f"{len(satisfied)} besoin(s) satisfait(s), {len(partial)} "
            f"partiel(s), {len(open_demands)} ouvert(s) ; "
            f"{len(comblés)} besoin(s) reposent sur un proxy géométrique "
            "qualifié qui les couvre nommément. La reconstruction hybride est "
            "possible, en sachant ce qui vient d'une photo et ce qui vient "
            "d'une forme."
        )
        decision.next_actions = [
            f"chercher des vues pour {demand_id}"
            for demand_id in sorted(partial + open_demands)
        ]
        decision.next_actions.extend(
            f"établir existence, état temporel et géométrie de {demand_id}"
            for demand_id in sorted(not_targetable)
        )
        return decision

    # 4. Ni couverture, ni proxy qui couvre ce qui manque.
    decision.path = RouterPath.PATH_D_HYBRID if qualifiés else RouterPath.REJECT
    decision.decision_status = DecisionStatus.CAPTURE_REQUIRED
    decision.rationale = (
        f"{len(reste)} besoin(s) sans couverture photographique ni proxy qui "
        "les couvre nommément : ce qui manque se prend à la caméra"
    )
    decision.next_actions = [
        f"capturer {demand_id}" for demand_id in sorted(partial + open_demands)
    ]
    decision.next_actions.extend(
        f"établir existence, état temporel et géométrie de {demand_id}"
        for demand_id in sorted(not_targetable)
    )
    return decision
