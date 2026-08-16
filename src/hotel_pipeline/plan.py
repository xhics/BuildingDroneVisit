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

    #: Registre des appels de **cette** commande — les mesures de volume, quand
    #: elles ont lieu. Un plan héritant des appels de la découverte annoncerait
    #: un coût qui n'est pas le sien.
    transport: dict = field(default_factory=dict)
    selected: int = 0
    rejected_by_reason: dict[str, int] = field(default_factory=dict)
    preview_required: int = 0
    #: Besoins pour lesquels une acquisition est **prévue**. Jamais « servis » :
    #: un besoin n'est servi qu'après qualification du fichier acquis, et
    #: `preview_required` signifie précisément qu'on ne sait pas encore.
    demands_planned: dict[str, int] = field(default_factory=dict)
    demands_planned_pending_preview: dict[str, int] = field(default_factory=dict)
    demands_unplanned: list[str] = field(default_factory=list)

    known_bytes: int = 0
    unknown_size_items: int = 0
    volume_status: str = VolumeStatus.UNKNOWN.value

    def as_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "candidates": self.candidates,
            "transport": self.transport,
            "evaluations": self.evaluations,
            "selected": self.selected,
            "rejected_by_reason": self.rejected_by_reason,
            "preview_required": self.preview_required,
            "demands": {
                "planned": self.demands_planned,
                "planned_pending_preview": self.demands_planned_pending_preview,
                "unplanned": self.demands_unplanned,
                "note": (
                    "« prévu » n'est pas « servi » : un besoin n'est servi "
                    "qu'après qualification du fichier acquis, et une "
                    "acquisition en vérification ne l'établit pas"
                ),
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

    # Le secteur d'abord : une vue prise du mauvais côté ne montre pas la
    # cible demandée, quelle que soit sa distance ou sa netteté.
    if measured.wrong_sector:
        return CandidateEvaluation(
            candidate_id=candidate.candidate_id, demand_id=demand.demand_id,
            intent=demand.intent, eligibility=Eligibility.REJECTED,
            geometry=measured,
            rejection_reason=(
                "observée depuis un côté que le besoin n'accepte pas : le "
                "secteur se juge sur la position de l'observateur"
            ),
        )

    # Une zone interdite aux plans rapprochés n'est pas un avertissement : la
    # vue prise depuis là ne sert pas ce besoin, et le schéma la validait sans
    # que rien ne l'écarte.
    if measured.forbidden_zones_entered:
        return CandidateEvaluation(
            candidate_id=candidate.candidate_id, demand_id=demand.demand_id,
            intent=demand.intent, eligibility=Eligibility.REJECTED,
            geometry=measured,
            rejection_reason=(
                "caméra située dans une zone interdite au besoin : "
                + ", ".join(measured.forbidden_zones_entered)
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

    # La cible entre-t-elle dans le cadre ? La largeur apparente ne le dit
    # pas : une cible immense à moitié hors champ paraît excellente sur elle.
    if (
        measured.in_frame_fraction is not None
        and demand.min_visible_fraction > 0
        and measured.in_frame_fraction < demand.min_visible_fraction
    ):
        return CandidateEvaluation(
            candidate_id=candidate.candidate_id, demand_id=demand.demand_id,
            intent=demand.intent, eligibility=Eligibility.REJECTED,
            geometry=measured,
            rejection_reason=(
                f"part dans le cadre {measured.in_frame_fraction:.3f} sous le "
                f"minimum {demand.min_visible_fraction:.3f} du besoin"
            ),
        )

    # Rien ne contredit le besoin, mais rien ne l'établit non plus : **toute**
    # métrique qu'il exige et qu'on ignore impose une vérification. Ne
    # regarder que la largeur laissait passer une vue dont la part visible
    # était inconnue alors que le besoin en exigeait une.
    unknown = _unknown_required_metrics(measured, demand)
    if unknown:
        return CandidateEvaluation(
            candidate_id=candidate.candidate_id, demand_id=demand.demand_id,
            intent=demand.intent, eligibility=Eligibility.PREVIEW_REQUIRED,
            geometry=measured,
        )

    return CandidateEvaluation(
        candidate_id=candidate.candidate_id, demand_id=demand.demand_id,
        intent=demand.intent, eligibility=Eligibility.ELIGIBLE, geometry=measured,
    )


def _unknown_required_metrics(measured, demand) -> list[str]:  # noqa: ANN001
    """Métriques exigées par le besoin et absentes de la mesure.

    Un besoin qui n'exige rien ne rend rien inconnu : c'est l'exigence qui
    crée l'obligation de mesurer, pas la mesure qui crée l'exigence.
    """
    missing = []
    if demand.min_projected_width_fraction > 0 and measured.unclipped_width_fraction is None:
        missing.append("taille projetée")
    if demand.min_visible_fraction > 0 and measured.in_frame_fraction is None:
        missing.append("part dans le cadre")
    if demand.min_visible_fraction > 0 and measured.visible_fraction is None:
        missing.append("part non masquée")
    return missing


def group_viewpoints(candidates: list, separation_m: float) -> dict[str, str]:
    """Attribue un point de vue à chaque candidat, par distance **réelle**.

    Deux cadrages d'un même panorama sont deux acquisitions et **un seul**
    point de vue : les compter deux fois ferait croire un besoin servi par
    deux observations indépendantes alors qu'il n'y en a qu'une, et un SfM
    n'en tirerait aucune parallaxe.

    Le regroupement se faisait par grille de latitude/longitude, ce qui
    séparait deux caméras distantes de six mètres tombant de part et d'autre
    d'une frontière de cellule — et en réunissait deux distantes de quatorze au
    sein d'une même cellule. On compare donc des distances, pas des cases.
    """
    assigned: dict[str, str] = {}
    anchors: list[tuple[str, float, float]] = []

    # Ordre stable : le regroupement ne doit pas dépendre de l'ordre d'arrivée.
    for candidate in sorted(candidates, key=lambda c: c.candidate_id):
        if candidate.panorama_id:
            assigned[candidate.candidate_id] = f"pano:{candidate.panorama_id}"
            continue
        if candidate.camera_lat is None or candidate.camera_lon is None:
            assigned[candidate.candidate_id] = f"sans-position:{candidate.candidate_id}"
            continue

        near = next(
            (
                name for name, lat, lon in anchors
                if _distance_m(candidate.camera_lat, candidate.camera_lon, lat, lon)
                <= separation_m
            ),
            None,
        )
        if near is None:
            near = f"pos:{candidate.candidate_id}"
            anchors.append((near, candidate.camera_lat, candidate.camera_lon))
        assigned[candidate.candidate_id] = near

    return assigned


def _distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance approchée en mètres, suffisante à cette échelle.

    Un calcul géodésique complet n'apporterait rien sur quelques dizaines de
    mètres, et exigerait le contexte spatial que ce module n'a pas.
    """
    import math

    mean_lat = math.radians((lat1 + lat2) / 2.0)
    dx = (lon2 - lon1) * 111_320.0 * math.cos(mean_lat)
    dy = (lat2 - lat1) * 110_540.0
    return math.hypot(dx, dy)


def select(
    evaluations: list[CandidateEvaluation], demands: list[CaptureDemand],
    sizes: dict[str, int] | None = None,
    candidates: dict | None = None,
    separation_m: float = 10.0,
    levels: dict[tuple[str, str], str] | None = None,
    preview_resolution: str = "256",
    full_resolution: str = "2048",
) -> list[PlannedAcquisition]:
    """Retient ce qui sert un besoin, en respectant le nombre attendu.

    Le nombre demandé porte sur des **points de vue**, non sur des fichiers :
    deux cadrages d'un même panorama ne comptent que pour un. Sans le
    dictionnaire des candidats, la position est inconnue et chaque candidat
    compte pour lui-même — l'appelant qui veut le décompte juste le fournit.

    `levels` porte le niveau que la **recherche** a prononcé. Le plan ne peut
    pas le contredire : un candidat qu'aucun niveau n'autorise n'entre pas au
    plan, et une preview y entre en miniature. Sans cette contrainte, les trois
    listes publiées restaient informatives et le plan téléchargeait en pleine
    résolution ce que la recherche bornait à l'aperçu.
    """
    announced = sizes or {}
    known = candidates or {}
    viewpoints = group_viewpoints(list(known.values()), separation_m)
    # `None` et `{}` ne disent pas la même chose, et les confondre réactivait
    # tout : un registre vide signifie « la recherche a eu lieu et n'a rien
    # recommandé », non « aucune contrainte ».
    legacy = levels is None
    graded = dict(levels or {})

    by_demand: dict[str, list[CandidateEvaluation]] = {}
    unrecommended: set[tuple[str, str]] = set()
    for evaluation in evaluations:
        if evaluation.eligibility is Eligibility.REJECTED:
            continue
        # L'autorisation vaut pour **ce besoin**, non pour le candidat en
        # général : une vue pleinement acquérable pour le stationnement n'est
        # pas autorisée à servir une façade qui ne l'a jamais recommandée.
        if not legacy and (evaluation.candidate_id, evaluation.demand_id) not in graded:
            unrecommended.add((evaluation.candidate_id, evaluation.demand_id))
            continue
        by_demand.setdefault(evaluation.demand_id, []).append(evaluation)

    wanted: dict[str, list[str]] = {}
    intents: dict[str, set[CaptureIntent]] = {}
    reasons: dict[str, list[str]] = {}

    for demand in demands:
        ordered = sorted(
            by_demand.get(demand.demand_id, []),
            # L'éligible passe avant celui qui demande une vérification : à
            # besoin égal, mieux vaut ce qui est établi que ce qui reste à voir.
            key=lambda e: (e.eligibility is Eligibility.PREVIEW_REQUIRED, e.candidate_id),
        )

        retained, seen_viewpoints = [], set()
        for evaluation in ordered:
            viewpoint = viewpoints.get(
                evaluation.candidate_id, f"candidat:{evaluation.candidate_id}"
            )
            if viewpoint in seen_viewpoints:
                continue
            seen_viewpoints.add(viewpoint)
            retained.append(evaluation)
            if len(retained) >= demand.viewpoints_required:
                break

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

        # Une preview se vérifie en miniature. La planifier en pleine
        # résolution dépenserait le volume avant de savoir s'il le valait.
        # Le niveau **le plus prudent** parmi les besoins servis : autoriser
        # la pleine résolution parce qu'un seul besoin s'en contentait perdrait
        # la réserve des autres.
        served_levels = [
            graded[(candidate_id, demand_id)]
            for demand_id in served
            if (candidate_id, demand_id) in graded
        ]
        is_preview = any(
            value in ("recommended_for_preview", "recommended_for_enrichment")
            for value in served_levels
        )
        level = (
            "recommended_for_preview" if is_preview
            else (served_levels[0] if served_levels else None)
        )
        # « Le mieux que la source sache faire » quand aucun besoin servi
        # n'exige un nombre de pixels. Demander 2048 à une source qui plafonne
        # à 640 la rendait inéligible ; l'intention laisse chaque fournisseur
        # répondre au maximum de sa capacité.
        exige_un_nombre = any(
            getattr(demand, "min_projected_width_fraction", 0.0) > 0
            or getattr(demand, "min_visible_fraction", 0.0) > 0
            for demand in demands
            if demand.demand_id in served
        )
        resolution = (
            preview_resolution if is_preview
            else (full_resolution if exige_un_nombre else "full_available")
        )
        planned.append(
            PlannedAcquisition(
                candidate_id=candidate_id,
                intents=chosen,
                primary_intent=chosen[0],
                serves_demands=served,
                # `None` et non zéro : une taille absente est inconnue.
                resolution=resolution,
                demand_levels={
                    demand_id: graded[(candidate_id, demand_id)]
                    for demand_id in served
                    if (candidate_id, demand_id) in graded
                },
                expected_bytes=announced.get(candidate_id),
                selection_rationale=(
                    "retenu pour " + ", ".join(sorted(reasons[candidate_id]))
                    + (
                        f" — {level}, planifié en {resolution}"
                        if level else ""
                    )
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
    separation_m: float = 10.0,
    policy=None,  # noqa: ANN001 — pour inscrire les facettes consommées
    levels: dict[tuple[str, str], str] | None = None,
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

    collection = getattr(policy, "collection", None)
    planned = select(
        evaluations, demands, sizes,
        candidates={c.candidate_id: c for c in candidates},
        separation_m=separation_m,
        levels=levels,
        preview_resolution=getattr(collection, "preview_resolution", "256"),
        full_resolution=getattr(collection, "full_resolution", "2048"),
    )
    plan = AcquisitionPlan(
        plan_id=plan_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
        hotel_id=hotel_id,
        status=PlanStatus.DRAFT,
        acquisitions=planned,
        # Les facettes réellement lues : c'est par elles que le plan périme.
        policy_dependency_digests=(
            _facet_digests(policy) if policy is not None else {}
        ),
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


def _facet_digests(policy) -> dict[str, str]:  # noqa: ANN001
    from .policy_facets import dependency_digests

    return dependency_digests(policy, "AcquisitionPlan")


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

    pending = {
        (e.candidate_id, e.demand_id)
        for e in evaluations
        if e.eligibility is Eligibility.PREVIEW_REQUIRED
    }
    planned: dict[str, int] = {}
    to_verify: dict[str, int] = {}
    for acquisition in plan.acquisitions:
        for demand_id in acquisition.serves_demands:
            planned[demand_id] = planned.get(demand_id, 0) + 1
            if (acquisition.candidate_id, demand_id) in pending:
                to_verify[demand_id] = to_verify.get(demand_id, 0) + 1

    report.demands_planned = dict(sorted(planned.items()))
    report.demands_planned_pending_preview = dict(sorted(to_verify.items()))
    report.demands_unplanned = sorted(
        demand.demand_id for demand in demands if demand.demand_id not in planned
    )

    report.known_bytes = plan.known_bytes
    report.unknown_size_items = len(plan.unknown_size_items)
    report.volume_status = plan.volume_status.value
    return report


def consent(  # noqa: ANN001
    plan: AcquisitionPlan,
    digests: dict[str, str | None],
    measured_from: str | None = None,
    download_contract_version: int | None = None,
) -> AcquisitionPlan:
    """Rend le plan exécutable, une fois le volume montré et accepté.

    Le passage n'est pas un changement d'étiquette : le schéma exige alors
    **toutes** les empreintes. Un plan qu'on ne peut pas rattacher à un état ne
    s'acquiert pas — il aurait choisi ses images pour un autre.

    L'accord s'attache aussi aux **requêtes** et au plafond. Sans cet ancrage,
    réécrire une résolution après coup téléchargerait autre chose sous le même
    consentement : le statut disait « accepté » sans dire de quoi.
    """
    # Le volume d'abord : c'est ce qu'on montre avant de dire à quoi il
    # s'applique. Un total partiel se refuse quel que soit l'état des requêtes.
    if plan.unknown_size_items:
        raise PlanRefused(
            f"{len(plan.unknown_size_items)} taille(s) inconnue(s) : consentir "
            "à un total dont une part n'est pas mesurée serait consentir à ce "
            "qui n'a pas été montré"
        )

    ungrounded = [
        acquisition.candidate_id
        for acquisition in plan.acquisitions
        if not acquisition.request_digest
    ]
    if ungrounded:
        raise PlanRefused(
            f"acquisition(s) sans empreinte de requête : {sorted(ungrounded)} — "
            "consentir sans savoir ce qui sera demandé n'engage rien"
        )

    updated = plan.model_copy(
        update={
            "status": PlanStatus.EXECUTABLE,
            "consented_max_bytes": plan.known_bytes,
            "consented_request_digests": sorted(
                a.request_digest for a in plan.acquisitions
            ),
            "consented_from_plan_id": measured_from or plan.plan_id,
            "consented_download_contract_version": download_contract_version,
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
