"""Décider comment reconstruire ce site — et le motiver (collecte V2).

Deux questions distinctes, que rien ne doit fondre en une :

```text
CaptureDemand + évaluations   ce qui est photographiquement couvert
SiteManifest                  quels objets existent, sont ciblables,
                              non résolus ou inapplicables
```

Les mêler ferait d'un objet non résolu une lacune de couverture, ou d'une
façade non photographiée un objet inexistant. Ce sont deux manques qui
appellent deux réponses : l'un demande une localisation ou une preuve, l'autre
une prise de vue.

**Jamais depuis le nombre brut d'images.** Trois cent treize vues autour d'un
bâtiment peuvent n'en montrer aucune façade utilement ; sur ce site, six
acquisitions ont été réfutées une à une. Ce qui compte est le nombre de besoins
satisfaits et de points de vue **indépendants** — deux cadrages d'un même
panorama n'en font qu'un.

La décision cite ce sur quoi elle se fonde : besoins par état, points de vue,
zones couvertes par proxy géométrique, artefacts actifs. Une décision qu'on ne
peut pas contester n'est pas une décision, c'est un verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum

from .logging import get_logger

log = get_logger("router")

#: Version du **contrat de décision** : ce que le Router promet d'examiner et
#: dans quel ordre. La changer périme les décisions antérieures, qui ont été
#: prises sur d'autres règles.
ROUTER_CONTRACT_VERSION = 1


class Route(StrEnum):
    """Comment ce site peut être reconstruit, en l'état."""

    #: Les photos existantes suffisent : chaque besoin est couvert par assez de
    #: points de vue indépendants, et la géométrie n'apporte que du contexte.
    PHOTO_FIRST_READY = "photo_first_ready"

    #: Les photos couvrent une partie ; le reste s'appuie sur des proxies
    #: géométriques qualifiés — toiture, terrain. Reconstruire est possible, en
    #: sachant ce qui vient d'où.
    HYBRID_READY = "hybrid_ready"

    #: Ce qui manque se prend à la caméra. Aucun obstacle ne l'empêche : c'est
    #: une campagne, non un blocage.
    CAPTURE_REQUIRED = "capture_required"

    #: Quelque chose doit être établi avant toute capture : un objet critique
    #: non localisé, un référentiel absent, une hauteur non qualifiée. Envoyer
    #: quelqu'un sur place sans cela ferait photographier au hasard.
    BLOCKED_PREREQUISITES = "blocked_prerequisites"


class ObjectStanding(StrEnum):
    """Ce que le site dit d'un objet — non ce qu'une photo en montre."""

    #: Établi et géoréférencé : on sait le viser.
    TARGETABLE = "targetable"

    #: Connu, mais sans géométrie : on sait qu'il existe, pas où.
    KNOWN_NOT_TARGETABLE = "known_not_targetable"

    #: Instancié au gabarit, sans que rien n'établisse son existence.
    UNRESOLVED = "unresolved"

    #: Démenti par une preuve : ce n'est pas une absence de données, c'est un
    #: constat. Sur ce site, le stationnement associé par proximité montrait le
    #: bâtiment voisin.
    REFUTED = "refuted"


@dataclass
class DemandStanding:
    """Où en est **un** besoin photographique."""

    demand_id: str
    status: str
    viewpoints_found: int = 0
    viewpoints_required: int = 1
    targetable: bool = True
    note: str = ""

    @property
    def satisfied(self) -> bool:
        return self.viewpoints_found >= self.viewpoints_required

    def as_dict(self) -> dict:
        return {
            "demand_id": self.demand_id,
            "status": self.status,
            "viewpoints_found": self.viewpoints_found,
            "viewpoints_required": self.viewpoints_required,
            "targetable": self.targetable,
            "note": self.note,
        }


@dataclass
class ProxyZone:
    """Une zone qu'un artefact géométrique couvre, faute de photo.

    Un proxy n'est pas une photo : il donne une forme, non une apparence. Le
    dire évite qu'un rendu texturé passe pour une observation.
    """

    zone: str
    artifact: str
    qualified: bool
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "zone": self.zone, "artifact": self.artifact,
            "qualified": self.qualified, "note": self.note,
        }


@dataclass
class RouterDecision:
    """La décision, et de quoi la contester."""

    hotel_id: str
    route: Route
    contract_version: int = ROUTER_CONTRACT_VERSION

    #: --- ce qui fonde la décision, côté photographique ---------------------
    demands_satisfied: list[str] = field(default_factory=list)
    demands_partial: list[str] = field(default_factory=list)
    demands_open: list[str] = field(default_factory=list)
    demands_not_targetable: list[str] = field(default_factory=list)

    #: Points de vue **indépendants** servant au moins un besoin. Deux cadrages
    #: d'un même panorama n'en font qu'un.
    independent_viewpoints: int = 0

    #: --- ce qui fonde la décision, côté site --------------------------------
    #: Objets par état. Un objet démenti n'est pas un objet manquant.
    objects_by_standing: dict = field(default_factory=dict)

    #: Objets **critiques** non établis : ceux dont l'absence empêche de
    #: décider où pointer une caméra.
    critical_objects_unestablished: list[str] = field(default_factory=list)

    #: --- ce qui comble, et par quoi ----------------------------------------
    proxy_zones: list[ProxyZone] = field(default_factory=list)
    active_artifacts: list[str] = field(default_factory=list)

    #: --- ce qui reste à faire ----------------------------------------------
    rationale: str = ""
    blocking: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    limits: list[str] = field(default_factory=list)

    decided_at: str = ""
    inputs: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "hotel_id": self.hotel_id,
            "route": self.route.value,
            "contract_version": self.contract_version,
            "photographic": {
                "satisfied": sorted(self.demands_satisfied),
                "partial": sorted(self.demands_partial),
                "open": sorted(self.demands_open),
                "not_targetable": sorted(self.demands_not_targetable),
                "independent_viewpoints": self.independent_viewpoints,
            },
            "site": {
                "by_standing": self.objects_by_standing,
                "critical_unestablished": sorted(
                    self.critical_objects_unestablished
                ),
            },
            "geometric_proxies": [zone.as_dict() for zone in self.proxy_zones],
            "active_artifacts": sorted(self.active_artifacts),
            "rationale": self.rationale,
            "blocking": self.blocking,
            "next_actions": self.next_actions,
            "inputs": self.inputs,
            "decided_at": self.decided_at,
            "limits": self.limits or [
                "la couverture se juge sur les besoins, jamais sur le nombre "
                "d'images : trois cents vues peuvent n'en montrer aucune "
                "façade utilement",
                "un proxy géométrique donne une forme, non une apparence",
                "un objet non résolu n'est pas une lacune photographique : il "
                "demande une localisation, non une prise de vue",
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


def decide(
    hotel_id: str,
    demands: list[DemandStanding],
    objects: dict,
    proxies: list[ProxyZone],
    artifacts: list[str],
    inputs: dict | None = None,
) -> RouterDecision:
    """Arrête une route, en citant ce qui la fonde.

    L'ordre des tests n'est pas indifférent : un prérequis manquant l'emporte
    sur toute couverture, car photographier sans savoir quoi viser ne produit
    rien d'exploitable.
    """
    satisfied = [d.demand_id for d in demands if d.satisfied and d.targetable]
    partial = [
        d.demand_id for d in demands
        if d.targetable and not d.satisfied and d.viewpoints_found > 0
    ]
    open_demands = [
        d.demand_id for d in demands
        if d.targetable and d.viewpoints_found == 0
    ]
    not_targetable = [d.demand_id for d in demands if not d.targetable]

    critiques = [
        kind for kind, standing in objects.items()
        if kind in CRITICAL_OBJECTS
        and standing not in (ObjectStanding.TARGETABLE,)
    ]

    viewpoints = sum(d.viewpoints_found for d in demands)
    qualifiés = [zone for zone in proxies if zone.qualified]

    decision = RouterDecision(
        hotel_id=hotel_id,
        route=Route.BLOCKED_PREREQUISITES,
        demands_satisfied=satisfied,
        demands_partial=partial,
        demands_open=open_demands,
        demands_not_targetable=not_targetable,
        independent_viewpoints=viewpoints,
        objects_by_standing={
            standing.value: sorted(
                kind for kind, s in objects.items() if s is standing
            )
            for standing in ObjectStanding
        },
        critical_objects_unestablished=critiques,
        proxy_zones=proxies,
        active_artifacts=artifacts,
        inputs=dict(inputs or {}),
        decided_at=datetime.now(timezone.utc).isoformat(),
    )

    # 1. Un prérequis manquant l'emporte sur tout le reste.
    if critiques:
        decision.blocking = [
            f"{kind} : objet critique non établi — on ne saurait pas où viser"
            for kind in critiques
        ]
        decision.rationale = (
            "un objet critique n'est pas établi : photographier sans savoir "
            "quoi viser ne produirait rien d'exploitable"
        )
        decision.next_actions = ["établir la géométrie des objets critiques"]
        return decision

    # 2. Tout couvert, sans recours à un proxy : les photos suffisent.
    if not partial and not open_demands and not not_targetable:
        decision.route = Route.PHOTO_FIRST_READY
        decision.rationale = (
            f"les {len(satisfied)} besoin(s) sont couverts par "
            f"{viewpoints} point(s) de vue indépendant(s) ; la géométrie "
            "n'apporte que du contexte"
        )
        return decision

    # 3. Des proxies qualifiés comblent ce que la photo ne couvre pas.
    if qualifiés:
        decision.route = Route.HYBRID_READY
        decision.rationale = (
            f"{len(satisfied)} besoin(s) couvert(s), {len(partial)} "
            f"partiel(s), {len(open_demands)} ouvert(s) ; "
            f"{len(qualifiés)} zone(s) reposent sur un proxy géométrique "
            "qualifié. La reconstruction est possible, en sachant ce qui "
            "vient d'une photo et ce qui vient d'une forme."
        )
        decision.next_actions = [
            f"chercher des vues pour {demand_id}"
            for demand_id in sorted(partial + open_demands)
        ]
        if not_targetable:
            decision.next_actions.extend(
                f"localiser {demand_id} : connu, mais sans géométrie à viser"
                for demand_id in sorted(not_targetable)
            )
        return decision

    # 4. Ni couverture, ni proxy : il faut aller photographier.
    decision.route = Route.CAPTURE_REQUIRED
    decision.rationale = (
        f"{len(open_demands)} besoin(s) sans aucune vue et aucun proxy "
        "qualifié pour y suppléer : ce qui manque se prend à la caméra"
    )
    decision.next_actions = [
        f"capturer {demand_id}" for demand_id in sorted(open_demands + partial)
    ]
    return decision
