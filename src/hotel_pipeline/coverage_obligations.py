"""Obligations de couverture : du gabarit aux besoins (collecte V2).

Entre les objets du gabarit et les `CaptureDemand`, un chaînon manquait. Sans
lui, un manifeste de besoins pouvait omettre la façade arrière et paraître
complet : le Router aurait compté des besoins tous satisfaits, sans savoir
qu'un objet n'en avait jamais eu.

La chaîne d'autorité, dans l'ordre :

```text
objets du gabarit
→ obligations de couverture applicables
→ CaptureDemand
→ DemandAssessment sur un corpus précis
→ Router
```

Une obligation obligatoire produit au moins une demande, **ou** porte
`not_applicable` / `waived` avec motif et preuve. Aucune ne disparaît en
silence — c'est tout ce que ce module garantit, et c'est ce qui empêche un
manifeste d'oublier volontairement l'arrière.

Ce module ne dit pas si un besoin est satisfait : c'est l'affaire du corpus et
de `DemandAssessment`. Il dit qu'aucun objet n'a été oublié.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .logging import get_logger
from .schemas.acquisition import CaptureIntent, TargetKind

log = get_logger("coverage-obligations")


class ObligationStatus(StrEnum):
    """Ce qu'il advient d'une obligation, pour un site donné."""

    #: Au moins une demande la couvre.
    DEMANDED = "demanded"

    #: L'objet n'existe pas ici — un établissement sans stationnement.
    NOT_APPLICABLE = "not_applicable"

    #: L'objet existe, et on renonce à le couvrir, en connaissance de cause.
    WAIVED = "waived"

    #: Aucune demande, aucune dispense : l'oubli qu'on cherche à empêcher.
    UNMET = "unmet"


@dataclass(frozen=True)
class CoverageObligation:
    """Ce qu'un objet du gabarit exige comme couverture photographique.

    L'obligation vit ici, pas dans le besoin : un besoin dit ce qu'on cherche,
    une obligation dit ce qu'on **doit** chercher. Confondre les deux laissait
    le manifeste définir ses propres exigences.
    """

    object_id: str
    intent: CaptureIntent
    target_kind: TargetKind

    #: Cible attendue de la demande. Pour un secteur, le nom du secteur ; pour
    #: un objet, l'identifiant de l'objet lui-même.
    expected_target_ref: str

    mandatory: bool = True
    rationale: str = ""


#: Obligations du gabarit. Génériques : aucun établissement n'y est nommé, et
#: leur applicabilité à un site se décide par `not_applicable`, jamais en
#: retirant une ligne.
OBLIGATIONS: tuple[CoverageObligation, ...] = (
    CoverageObligation(
        "FACADE_PRIMARY", CaptureIntent.BUILDING_CAPTURE, TargetKind.VIEW_SECTOR,
        "front", rationale="la façade d'adresse fonde la reconnaissance du site",
    ),
    CoverageObligation(
        "FACADE_LEFT", CaptureIntent.BUILDING_CAPTURE, TargetKind.VIEW_SECTOR,
        "left", rationale="un volume ne se reconstruit pas depuis une seule face",
    ),
    CoverageObligation(
        "FACADE_RIGHT", CaptureIntent.BUILDING_CAPTURE, TargetKind.VIEW_SECTOR,
        "right", rationale="un volume ne se reconstruit pas depuis une seule face",
    ),
    CoverageObligation(
        "FACADE_REAR", CaptureIntent.BUILDING_CAPTURE, TargetKind.VIEW_SECTOR,
        "rear",
        rationale=(
            "la face la plus souvent absente, et celle qu'un manifeste "
            "incomplet omet sans le dire"
        ),
    ),
    CoverageObligation(
        "ENTRANCE_MAIN_CURRENT", CaptureIntent.BUILDING_CAPTURE,
        TargetKind.SITE_OBJECT, "ENTRANCE_MAIN_CURRENT",
        rationale="l'entrée porte l'apparence la plus datée du bâtiment",
    ),
    CoverageObligation(
        "PROPERTY_SIGN", CaptureIntent.BUILDING_CAPTURE, TargetKind.SITE_OBJECT,
        "PROPERTY_SIGN",
        rationale="l'enseigne établit l'appartenance là où la géométrie ne peut pas",
    ),
    CoverageObligation(
        "ACCESS_ROAD_MAIN", CaptureIntent.CONTEXT_CAPTURE,
        TargetKind.CONTEXT_CORRIDOR, "ACCESS_ROAD_MAIN",
        rationale="l'accès documente comment on arrive, non le bâtiment",
    ),
    CoverageObligation(
        "PARKING_HOTEL", CaptureIntent.CONTEXT_CAPTURE, TargetKind.SITE_OBJECT,
        "PARKING_HOTEL", mandatory=False,
        rationale="tous les établissements n'en ont pas",
    ),
    CoverageObligation(
        "DRIVEWAY_MAIN", CaptureIntent.CONTEXT_CAPTURE, TargetKind.TRANSITION,
        "DRIVEWAY_MAIN", mandatory=False,
        rationale="la transition voie-entrée, quand elle existe",
    ),
)


#: Objets du gabarit **sans** obligation photographique, et pourquoi. Les
#: nommer rend leur absence vérifiable : un objet ajouté au gabarit et oublié
#: ici se signalerait, au lieu de n'exiger rien en silence.
NO_PHOTOGRAPHIC_OBLIGATION: dict[str, str] = {
    "PROPERTY_PARCEL": (
        "limite cadastrale : elle s'établit sur un registre officiel, aucune "
        "photographie ne la prouve"
    ),
    "TERRAIN_MAIN": (
        "surface de sol : elle se dérive d'un nuage LiDAR, une vue de rue n'en "
        "montre que ce qui n'est pas masqué"
    ),
    "ROOFLINE_MAIN": (
        "toiture : invisible depuis la voirie, elle vient de l'aérien ou du "
        "LiDAR — l'exiger en photo produirait une obligation intenable"
    ),
    "BUILDING_MAIN": (
        "le volume lui-même n'est pas une vue : il se couvre par ses façades, "
        "qui portent chacune leur obligation"
    ),
}


class ObligationWaiver(BaseModel):
    """Renoncer à couvrir un objet — décision, jamais omission.

    Immuable, comme les autres décisions : motif et preuve obligatoires,
    puisque renoncer sans dire pourquoi interdit d'y revenir.
    """

    model_config = ConfigDict(extra="forbid")

    object_id: str = Field(min_length=1)
    status: ObligationStatus
    decided_by: str = Field(min_length=1)
    decided_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    rationale: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _only_dispenses_are_declared(self) -> "ObligationWaiver":
        if self.status not in (ObligationStatus.NOT_APPLICABLE, ObligationStatus.WAIVED):
            raise ValueError(
                f"{self.object_id!r} : une dispense déclare `not_applicable` ou "
                f"`waived`, jamais {self.status.value!r} — les autres états se "
                "constatent, ils ne se décident pas"
            )
        if not self.evidence:
            raise ValueError(
                f"{self.object_id!r} : dispense sans preuve. Déclarer qu'un "
                "objet n'existe pas, ou qu'on renonce à le couvrir, demande de "
                "dire sur quoi on se fonde"
            )
        return self


@dataclass
class CoverageReport:
    """État de chaque obligation, et ce qui manque."""

    by_status: dict[str, list[str]] = field(default_factory=dict)
    demands_by_object: dict[str, list[str]] = field(default_factory=dict)
    orphan_demands: list[str] = field(default_factory=list)

    @property
    def unmet(self) -> list[str]:
        return sorted(self.by_status.get(ObligationStatus.UNMET.value, []))

    @property
    def complete(self) -> bool:
        """Aucune obligation obligatoire n'est laissée sans réponse."""
        return not self.unmet

    def as_dict(self) -> dict:
        return {
            "by_status": {k: sorted(v) for k, v in sorted(self.by_status.items())},
            "demands_by_object": self.demands_by_object,
            "orphan_demands": self.orphan_demands,
            "complete": self.complete,
            "note": (
                "une obligation couverte n'est pas un besoin satisfait : ce "
                "rapport dit qu'aucun objet n'a été oublié, pas qu'on possède "
                "les vues"
            ),
        }


def assess(demands: list, waivers: list[ObligationWaiver] | None = None) -> CoverageReport:  # noqa: ANN001
    """Confronte les besoins déclarés aux obligations du gabarit.

    Une obligation non couverte et non dispensée est `unmet` : c'est le seul
    état que ce module refuse de laisser passer en silence.
    """
    dispensed = {waiver.object_id: waiver for waiver in (waivers or [])}
    report = CoverageReport()

    for obligation in OBLIGATIONS:
        matching = [
            demand.demand_id
            for demand in demands
            if demand.intent is obligation.intent
            and demand.target_kind is obligation.target_kind
            and demand.target_ref == obligation.expected_target_ref
        ]
        if matching:
            status = ObligationStatus.DEMANDED
            report.demands_by_object[obligation.object_id] = sorted(matching)
        elif obligation.object_id in dispensed:
            status = dispensed[obligation.object_id].status
        elif obligation.mandatory:
            status = ObligationStatus.UNMET
        else:
            # Facultative et non demandée : rien n'est dû, et le dire
            # explicitement évite qu'on la croie oubliée.
            status = ObligationStatus.NOT_APPLICABLE

        report.by_status.setdefault(status.value, []).append(obligation.object_id)

    covered = {
        demand_id
        for identifiers in report.demands_by_object.values()
        for demand_id in identifiers
    }
    # Un besoin qui ne sert aucune obligation n'est pas fautif — on peut
    # vouloir davantage que le minimum — mais il se signale, pour qu'une cible
    # mal orthographiée ne passe pas pour une exigence supplémentaire.
    report.orphan_demands = sorted(
        demand.demand_id for demand in demands if demand.demand_id not in covered
    )

    log.info(
        "obligations : %d couverte(s), %d non couverte(s), %d besoin(s) hors gabarit",
        len(report.demands_by_object), len(report.unmet), len(report.orphan_demands),
    )
    return report


def missing_demands(report: CoverageReport) -> list[CoverageObligation]:
    """Obligations qu'il reste à traduire en besoins."""
    unmet = set(report.unmet)
    return [
        obligation for obligation in OBLIGATIONS if obligation.object_id in unmet
    ]
