"""Sélection des acquisitions (collecte V2, étape 2).

Entre la découverte et le téléchargement, une décision : lesquels de ces
candidats servent lesquels besoins, et à quel volume. Rien n'est acquis ici
non plus — le plan est un engagement écrit, soumis au consentement, et
l'acquisition n'exécutera que ce qu'il porte.

Deux séparations gouvernent le module.

**Le besoin et son évaluation.** Un candidat n'est ni bon ni mauvais : il est
bon *pour un besoin*. Une vue lointaine cadre mal la façade et documente
parfaitement la voie d'accès — trois verdicts pour une image, et les réduire à
un seul obligerait à en taire deux.

**Le volume connu et le volume inconnu.** Une taille absente n'est pas zéro.
Additionner comme nulles les tailles qu'on ignore annoncerait un total « exact »
faux, et ferait consentir à un volume qui n'a pas été montré.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from .logging import get_logger
from .schemas.acquisition import (
    AcquisitionPlan,
    CandidateEvaluation,
    CaptureCandidate,
    CaptureDemand,
    CaptureIntent,
    Eligibility,
    PlannedAcquisition,
    PlanStatus,
    VolumeStatus,
)

log = get_logger("plan")


class PlanRefused(RuntimeError):
    """Rien n'a été planifié, et rien n'a été écrit."""


@dataclass
class PlanReport:
    """Ce qui a été retenu, écarté, et ce que le volume laisse indéterminé."""

    plan_id: str = ""
    candidates: int = 0
    evaluations: int = 0
    selected: int = 0
    rejected_by_reason: dict[str, int] = field(default_factory=dict)
    preview_required: int = 0
    demands_served: dict[str, int] = field(default_factory=dict)
    demands_unserved: list[str] = field(default_factory=list)

    known_bytes: int = 0
    unknown_size_items: int = 0
    volume_status: str = VolumeStatus.UNKNOWN.value

    def as_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "candidates": self.candidates,
            "evaluations": self.evaluations,
            "selected": self.selected,
            "rejected_by_reason": self.rejected_by_reason,
            "preview_required": self.preview_required,
            "demands": {
                "served": self.demands_served,
                "unserved": self.demands_unserved,
            },
            "volume": {
                "known_bytes": self.known_bytes,
                "unknown_size_items": self.unknown_size_items,
                "status": self.volume_status,
                "note": (
                    "un volume inconnu n'est pas un volume nul : les "
                    "acquisitions sans taille annoncée sont comptées à part, "
                    "et aucun total « exact » n'est publié tant qu'il en reste"
                ),
            },
            "bytes_downloaded": 0,
        }


def evaluate(
    candidate: CaptureCandidate, demand: CaptureDemand, geometry=None,  # noqa: ANN001
) -> CandidateEvaluation:
    """Ce que ce candidat vaut **pour ce besoin**.

    À ce stade, aucune image n'existe : la géométrie est calculée sur des
    métadonnées, et ses valeurs sont des espérances. Un candidat dont on ne
    sait rien n'est donc pas rejeté — il demande une vérification par
    miniature, ce qui est une réponse, pas un doute.
    """
    from .schemas.acquisition import CandidateGeometry

    measured = geometry or CandidateGeometry()

    if candidate.camera_lat is None or candidate.camera_lon is None:
        return CandidateEvaluation(
            candidate_id=candidate.candidate_id, demand_id=demand.demand_id,
            intent=demand.intent, eligibility=Eligibility.REJECTED,
            geometry=measured,
            rejection_reason=(
                "position de caméra inconnue : la géométrie ne peut rien dire "
                "de ce que la vue montre"
            ),
        )

    width = measured.unclipped_width_fraction
    if width is not None and width < demand.min_projected_width_fraction:
        return CandidateEvaluation(
            candidate_id=candidate.candidate_id, demand_id=demand.demand_id,
            intent=demand.intent, eligibility=Eligibility.REJECTED,
            geometry=measured,
            rejection_reason=(
                f"taille projetée espérée {width:.3f} sous le minimum "
                f"{demand.min_projected_width_fraction:.3f} du besoin"
            ),
        )

    visible = measured.visible_fraction
    if visible is not None and visible < demand.min_visible_fraction:
        return CandidateEvaluation(
            candidate_id=candidate.candidate_id, demand_id=demand.demand_id,
            intent=demand.intent, eligibility=Eligibility.REJECTED,
            geometry=measured,
            rejection_reason=(
                f"part visible espérée {visible:.3f} sous le minimum "
                f"{demand.min_visible_fraction:.3f} du besoin"
            ),
        )

    # Rien ne contredit le besoin, mais rien ne l'établit non plus : sans
    # mesure de cadrage, engager la pleine résolution reviendrait à parier.
    if width is None and demand.min_projected_width_fraction > 0:
        return CandidateEvaluation(
            candidate_id=candidate.candidate_id, demand_id=demand.demand_id,
            intent=demand.intent, eligibility=Eligibility.PREVIEW_REQUIRED,
            geometry=measured,
        )

    return CandidateEvaluation(
        candidate_id=candidate.candidate_id, demand_id=demand.demand_id,
        intent=demand.intent, eligibility=Eligibility.ELIGIBLE, geometry=measured,
    )


def select(
    evaluations: list[CandidateEvaluation], demands: list[CaptureDemand],
    sizes: dict[str, int] | None = None,
) -> list[PlannedAcquisition]:
    """Retient ce qui sert un besoin, en respectant le nombre attendu.

    Le nombre demandé porte sur des **points de vue**, non sur des fichiers :
    on ne retient donc pas deux fois la même position. Faute de position
    fiable à ce stade, le candidat lui-même fait office de point de vue, et le
    dédoublonnage géométrique reste au niveau où il se mesure.
    """
    announced = sizes or {}
    by_demand: dict[str, list[CandidateEvaluation]] = {}
    for evaluation in evaluations:
        if evaluation.eligibility is Eligibility.REJECTED:
            continue
        by_demand.setdefault(evaluation.demand_id, []).append(evaluation)

    wanted: dict[str, list[str]] = {}
    intents: dict[str, set[CaptureIntent]] = {}
    reasons: dict[str, list[str]] = {}

    for demand in demands:
        retained = sorted(
            by_demand.get(demand.demand_id, []),
            # L'éligible passe avant celui qui demande une vérification : à
            # besoin égal, mieux vaut ce qui est établi que ce qui reste à voir.
            key=lambda e: (e.eligibility is Eligibility.PREVIEW_REQUIRED, e.candidate_id),
        )[: demand.viewpoints_required]

        for evaluation in retained:
            wanted.setdefault(evaluation.candidate_id, []).append(demand.demand_id)
            intents.setdefault(evaluation.candidate_id, set()).add(demand.intent)
            reasons.setdefault(evaluation.candidate_id, []).append(
                f"{demand.demand_id} ({evaluation.eligibility.value})"
            )

    planned = []
    for candidate_id in sorted(wanted):
        served = sorted(wanted[candidate_id])
        chosen = sorted(intents[candidate_id], key=lambda i: i.value)
        planned.append(
            PlannedAcquisition(
                candidate_id=candidate_id,
                intents=chosen,
                primary_intent=chosen[0],
                serves_demands=served,
                # `None` et non zéro : une taille absente est inconnue.
                expected_bytes=announced.get(candidate_id),
                selection_rationale=(
                    "retenu pour " + ", ".join(sorted(reasons[candidate_id]))
                ),
            )
        )
    return planned


def build(
    hotel_id: str,
    candidates: list[CaptureCandidate],
    demands: list[CaptureDemand],
    digests: dict[str, str | None],
    geometries: dict[tuple[str, str], object] | None = None,
    sizes: dict[str, int] | None = None,
    plan_id: str | None = None,
) -> tuple[AcquisitionPlan, list[CandidateEvaluation], PlanReport]:
    """Évalue chaque candidat pour chaque besoin, puis arrête un plan.

    Le plan naît **brouillon**. Il ne devient exécutable qu'après consentement
    explicite sur le volume : c'est le seul moment où la question « combien
    d'octets » a une réponse à montrer.
    """
    if not demands:
        raise PlanRefused(
            "aucun besoin déclaré : un plan sans objectif ne sélectionne rien, "
            "il justifie après coup"
        )
    if not candidates:
        raise PlanRefused(
            "aucun candidat : lancez « assets discover » avant de planifier"
        )

    known_geometry = geometries or {}
    evaluations: list[CandidateEvaluation] = []
    for candidate in candidates:
        for demand in demands:
            evaluations.append(
                evaluate(
                    candidate, demand,
                    known_geometry.get((candidate.candidate_id, demand.demand_id)),
                )
            )

    planned = select(evaluations, demands, sizes)
    plan = AcquisitionPlan(
        plan_id=plan_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
        hotel_id=hotel_id,
        status=PlanStatus.DRAFT,
        acquisitions=planned,
        **{name: digests.get(name) for name in _PLAN_DIGEST_FIELDS},
    )

    report = _report(plan, candidates, evaluations, demands)
    log.info(
        "plan %s : %d candidat(s), %d évaluation(s), %d retenue(s), volume %s",
        plan.plan_id, len(candidates), len(evaluations), len(planned),
        plan.volume_status.value,
    )
    return plan, evaluations, report


#: Empreintes qu'un plan porte. Tenues ici pour qu'un ajout au schéma ne passe
#: pas inaperçu : le plan les exige toutes pour devenir exécutable.
_PLAN_DIGEST_FIELDS = (
    "candidate_manifest_digest", "demand_digest", "policy_digest",
    "site_manifest_digest", "spatial_manifest_digest", "corpus_digest",
    "road_geometry_digest", "obstacle_geometry_digest",
)


def _report(
    plan: AcquisitionPlan, candidates: list[CaptureCandidate],
    evaluations: list[CandidateEvaluation], demands: list[CaptureDemand],
) -> PlanReport:
    report = PlanReport(
        plan_id=plan.plan_id,
        candidates=len(candidates),
        evaluations=len(evaluations),
        selected=len(plan.acquisitions),
    )

    for evaluation in evaluations:
        if evaluation.eligibility is Eligibility.REJECTED:
            key = (evaluation.rejection_reason or "").split(" :")[0].split(" sous")[0]
            report.rejected_by_reason[key] = report.rejected_by_reason.get(key, 0) + 1
        elif evaluation.eligibility is Eligibility.PREVIEW_REQUIRED:
            report.preview_required += 1

    served: dict[str, int] = {}
    for acquisition in plan.acquisitions:
        for demand_id in acquisition.serves_demands:
            served[demand_id] = served.get(demand_id, 0) + 1
    report.demands_served = dict(sorted(served.items()))
    report.demands_unserved = sorted(
        demand.demand_id for demand in demands if demand.demand_id not in served
    )

    report.known_bytes = plan.known_bytes
    report.unknown_size_items = len(plan.unknown_size_items)
    report.volume_status = plan.volume_status.value
    return report


def consent(plan: AcquisitionPlan, digests: dict[str, str | None]) -> AcquisitionPlan:
    """Rend le plan exécutable, une fois le volume montré et accepté.

    Le passage n'est pas un changement d'étiquette : le schéma exige alors
    **toutes** les empreintes. Un plan qu'on ne peut pas rattacher à un état ne
    s'acquiert pas — il aurait choisi ses images pour un autre.
    """
    updated = plan.model_copy(
        update={
            "status": PlanStatus.EXECUTABLE,
            **{
                name: digests.get(name) or getattr(plan, name)
                for name in _PLAN_DIGEST_FIELDS
            },
        }
    )
    missing = updated.missing_digests()
    if missing:
        raise PlanRefused(
            f"plan {plan.plan_id!r} : empreinte(s) manquante(s) {missing} — "
            "un plan qu'on ne peut pas rattacher à un état ne s'acquiert pas"
        )
    # Revalider : `model_copy` ne rejoue pas les validateurs, et un plan
    # exécutable vide passerait.
    return AcquisitionPlan.model_validate(updated.model_dump())
