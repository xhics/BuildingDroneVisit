"""Recherche adaptative : chercher là où il manque (collecte V2).

La découverte interrogeait un centre et un rayon, puis laissait le plan trier.
Elle ne savait pas quels secteurs étaient déficitaires — et le lui apprendre
dans le collecteur aurait recréé les objectifs de couverture hors des
obligations, donc deux autorités qui divergent.

Elle consomme donc l'évaluation des besoins, et n'interroge que ce qui reste
ouvert. Deux passes, dans cet ordre :

```text
1. exploration    trouver les secteurs sans aucune vue
2. expansion      prolonger les vues prometteuses, par séquence
```

Trois principes gouvernent les mesures, et aucun n'est un score unique.

**La continuité et la parallaxe s'opposent.** Une vue voisine d'une ancre est
mauvaise pour la géométrie et excellente pour le chaînage. Les fondre en un
chiffre ferait rejeter comme doublon ce qui ferme un trou de continuité.

**L'inconnu n'est pas zéro.** Une séquence non interrogée donne
`continuity_gain = None`, avec la cause ; une absence d'ancre donne
`parallax_gain = None`, avec la sienne. Zéro dirait « mesuré, et nul ».

**Le bâtiment n'est jamais une ancre.** C'est une cible géométrique, pas une
position caméra. Sans ancre compatible, la priorité vient du manque de
couverture — et la diversité entre nouveaux candidats se mesure entre eux.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

from .logging import get_logger

log = get_logger("adaptive-search")


class SequenceStatus(StrEnum):
    """Ce qu'on sait de l'appartenance d'un candidat à une séquence."""

    #: Première passe : l'enrichissement n'a pas eu lieu.
    NOT_QUERIED = "not_queried"

    #: Le fournisseur a rendu une séquence.
    KNOWN = "known"

    #: Interrogé, sans séquence en réponse — l'image n'en a pas.
    NOT_RETURNED = "not_returned"

    #: L'enrichissement a échoué. La vue reste utilisable en géométrie ; seul
    #: son gain de continuité est indisponible.
    QUERY_ERROR = "query_error"


class ContinuityLevel(StrEnum):
    """Jusqu'où va ce qu'on sait du recouvrement."""

    #: Deux vues de la même séquence : le recouvrement est **plausible**.
    #: Appartenir à une séquence ne le prouve pas — un véhicule tourne.
    POTENTIAL = "potential"

    #: Mesuré sur les images acquises.
    ACHIEVED = "achieved"


@dataclass
class CandidateMeasure:
    """Ce qu'un candidat vaut **pour un besoin**, mesure par mesure.

    Aucun agrégat : les qualités s'opposent, et les fondre ferait disparaître
    celle qu'on n'a pas cherchée.
    """

    candidate_id: str
    demand_id: str

    coverage_gap_priority: int = 0
    sector_novelty: bool = False

    distance_to_target_m: float | None = None
    distance_to_nearest_anchor_m: float | None = None
    bearing_delta_from_anchor_deg: float | None = None

    #: `None` tant qu'aucune séquence n'est connue — jamais zéro.
    continuity_gain: float | None = None
    continuity_level: ContinuityLevel | None = None
    continuity_reason: str = ""

    #: Mesures **géométriques**, indépendantes de toute préférence. Elles
    #: restent identiques si la fonction de choix change, et c'est ce qui rend
    #: un rapport relisible après un changement de politique.
    baseline_to_anchor_m: float | None = None

    #: Préférence **issue de la politique**, à ne pas confondre avec une
    #: mesure. `None` quand aucune ancre géométrique ne sert ce besoin.
    parallax_utility: float | None = None
    parallax_reason: str = ""

    sequence_id: str | None = None
    sequence_status: SequenceStatus = SequenceStatus.NOT_QUERIED
    corridor_ref: str | None = None

    heading_is_measured: bool = False
    rejection_reason: str | None = None

    def as_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "demand_id": self.demand_id,
            "coverage_gap_priority": self.coverage_gap_priority,
            "sector_novelty": self.sector_novelty,
            "distance_to_target_m": self.distance_to_target_m,
            "distance_to_nearest_anchor_m": self.distance_to_nearest_anchor_m,
            "bearing_delta_from_anchor_deg": self.bearing_delta_from_anchor_deg,
            "continuity": {
                "gain": self.continuity_gain,
                "level": self.continuity_level.value if self.continuity_level else None,
                "reason": self.continuity_reason,
                "sequence_id": self.sequence_id,
                "sequence_status": self.sequence_status.value,
            },
            "parallax": {
                # Mesures brutes d'abord : elles survivent au changement de
                # préférence, et un rapport doit rester lisible après.
                "bearing_delta_deg": self.bearing_delta_from_anchor_deg,
                "baseline_m": self.baseline_to_anchor_m,
                # Puis la préférence, étiquetée comme telle.
                "utility_policy_score": self.parallax_utility,
                "reason": self.parallax_reason,
            },
            "corridor_ref": self.corridor_ref,
            "heading_is_measured": self.heading_is_measured,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class GeometryAnchor:
    """Un point de vue existant qui sert **ce** besoin.

    Distinct d'une graine de séquence : celle-ci sert à explorer, sans porter
    aucun crédit de couverture. Les confondre ferait croire que l'arrière
    possède une ancre parce qu'une image du coin avant-droit est identifiée.
    """

    viewpoint_id: str
    demand_id: str
    lat: float
    lon: float
    suitability: str


@dataclass
class SearchReport:
    """Ce qui a été cherché, pour quels besoins, et ce qui en est ressorti.

    Structurellement lié au manifeste qu'il accompagne : un horodatage commun
    ne prouve rien, et un rapport orphelin ne se rattache à aucun état.
    """

    run_id: str = ""
    hotel_id: str = ""

    #: Filiation : ce à quoi cette recherche se rapporte.
    demand_digest: str | None = None
    demand_assessment_digest: str | None = None
    asset_manifest_digest: str | None = None
    capture_geometry_digest: str | None = None
    coarse_response_digest: str | None = None
    policy_dependency_digests: dict[str, str] = field(default_factory=dict)

    #: Tous les candidats considérés, recommandés ou non.
    candidates_considered: list[str] = field(default_factory=list)
    recommended: dict[str, list[str]] = field(default_factory=dict)

    demands_searched: list[str] = field(default_factory=list)
    demands_skipped: dict[str, str] = field(default_factory=dict)
    anchors_by_demand: dict[str, list[str]] = field(default_factory=dict)

    #: Distinguer quatre issues qu'un simple zéro confondrait.
    source_not_queried: dict[str, str] = field(default_factory=dict)
    zero_results: list[str] = field(default_factory=list)
    query_errors: dict[str, str] = field(default_factory=dict)
    all_rejected: dict[str, int] = field(default_factory=dict)

    enrichment_calls: int = 0
    measures: list[CandidateMeasure] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "hotel_id": self.hotel_id,
            "lineage": {
                "demand_digest": self.demand_digest,
                "demand_assessment_digest": self.demand_assessment_digest,
                "asset_manifest_digest": self.asset_manifest_digest,
                "capture_geometry_digest": self.capture_geometry_digest,
                "coarse_response_digest": self.coarse_response_digest,
            },
            "policy_dependency_digests": self.policy_dependency_digests,
            "candidates_considered": sorted(self.candidates_considered),
            "recommended": {k: sorted(v) for k, v in sorted(self.recommended.items())},
            "demands_searched": sorted(self.demands_searched),
            "demands_skipped": self.demands_skipped,
            "anchors_by_demand": self.anchors_by_demand,
            "outcomes": {
                "source_not_queried": self.source_not_queried,
                "zero_results": sorted(self.zero_results),
                "query_errors": self.query_errors,
                "all_candidates_rejected": self.all_rejected,
            },
            "enrichment_calls": self.enrichment_calls,
            "measures": [measure.as_dict() for measure in self.measures],
            "bytes_downloaded": 0,
            "note": (
                "un gain de continuité « potential » dit que deux vues "
                "appartiennent à la même séquence, non qu'elles se recouvrent : "
                "seule la mesure sur images acquises l'établit"
            ),
        }


def open_demands(assessment, demands: list) -> list:  # noqa: ANN001
    """Besoins qu'il reste à servir. Un besoin satisfait n'est pas recherché."""
    from .schemas.acquisition import DemandStatus

    served = {
        item.demand_id
        for item in getattr(assessment, "assessments", [])
        if item.status is DemandStatus.MET
    }
    return [demand for demand in demands if demand.demand_id not in served]


def anchors_for(demand, assets: list, viewpoints: dict, sectors: dict) -> list[GeometryAnchor]:  # noqa: ANN001
    """Ancres géométriques **compatibles avec ce besoin**.

    Une vue du coin avant-droit n'est une ancre que pour son propre secteur :
    la traiter comme ancre générale ferait croire à l'arrière qu'il en possède
    une, et la recherche chercherait de la parallaxe autour d'un point qui ne
    voit pas la cible.
    """
    from .demands_assess import counts_towards
    from .schemas.acquisition import TargetKind

    if demand.target_kind is not TargetKind.VIEW_SECTOR:
        return []

    found: dict[str, GeometryAnchor] = {}
    for asset in assets:
        usable, _ = counts_towards(asset)
        if not usable:
            continue
        observed = sectors.get(asset.id) or asset.view_sector.value
        if observed != demand.target_ref:
            continue
        if asset.camera_lat is None or asset.camera_lon is None:
            continue
        viewpoint = viewpoints.get(asset.id, f"asset:{asset.id}")
        found.setdefault(
            viewpoint,
            GeometryAnchor(
                viewpoint_id=viewpoint, demand_id=demand.demand_id,
                lat=asset.camera_lat, lon=asset.camera_lon,
                suitability=asset.geometry_suitability.value,
            ),
        )
    return list(found.values())


def coverage_gap_priority(demand, assessment) -> int:  # noqa: ANN001
    """Combien ce besoin manque, en priorité de recherche.

    L'ordre est celui de la spécification : un secteur sans aucune vue passe
    avant un secteur sous son minimum, qui passe avant tout gain marginal.
    """
    from .schemas.acquisition import DemandStatus

    item = next(
        (a for a in getattr(assessment, "assessments", [])
         if a.demand_id == demand.demand_id),
        None,
    )
    if item is None:
        return 3
    if item.status is DemandStatus.MET:
        return 0
    if item.viewpoints_found == 0:
        return 3  # aucun point de vue : la priorité la plus haute
    if item.viewpoints_found < demand.viewpoints_required:
        return 2  # sous son minimum
    return 1  # servi, mais perfectible


def measure_candidate(
    candidate,  # noqa: ANN001
    demand,  # noqa: ANN001
    anchors: list[GeometryAnchor],
    priority: int,
    target_lat: float | None = None,
    target_lon: float | None = None,
    sequence_of: dict[str, str] | None = None,
    anchor_sequences: set[str] | None = None,
    sequence_status: SequenceStatus = SequenceStatus.NOT_QUERIED,
    policy=None,  # noqa: ANN001 — AdaptiveSearchPolicy
) -> CandidateMeasure:
    """Mesure un candidat contre un besoin, qualité par qualité.

    Ni agrégat, ni valeur par défaut : ce qu'on ignore reste `None`, avec sa
    cause. Un zéro affirmerait une mesure qu'on n'a pas faite.
    """
    from .visibility import bearing_deg, haversine_m

    measure = CandidateMeasure(
        candidate_id=candidate.candidate_id,
        demand_id=demand.demand_id,
        coverage_gap_priority=priority,
        sector_novelty=not anchors,
        heading_is_measured=candidate.heading_is_measured,
    )

    if candidate.camera_lat is None or candidate.camera_lon is None:
        measure.rejection_reason = "position de caméra inconnue"
        return measure

    if target_lat is not None and target_lon is not None:
        measure.distance_to_target_m = round(
            haversine_m(candidate.camera_lat, candidate.camera_lon,
                        target_lat, target_lon), 1
        )

    # --- parallaxe : contre les ancres **de ce besoin**, ou rien -------------
    if not anchors:
        measure.parallax_utility = None
        measure.parallax_reason = (
            "aucune ancre géométrique compatible avec ce besoin — le bâtiment "
            "est une cible, jamais une position caméra"
        )
    else:
        nearest = min(
            anchors,
            key=lambda a: haversine_m(
                candidate.camera_lat, candidate.camera_lon, a.lat, a.lon
            ),
        )
        separation = haversine_m(
            candidate.camera_lat, candidate.camera_lon, nearest.lat, nearest.lon
        )
        measure.distance_to_nearest_anchor_m = round(separation, 1)
        measure.baseline_to_anchor_m = round(separation, 1)

        if target_lat is not None and target_lon is not None:
            from_candidate = bearing_deg(
                candidate.camera_lat, candidate.camera_lon, target_lat, target_lon
            )
            from_anchor = bearing_deg(nearest.lat, nearest.lon, target_lat, target_lon)
            delta = abs((from_candidate - from_anchor + 180.0) % 360.0 - 180.0)
            measure.bearing_delta_from_anchor_deg = round(delta, 1)

            if policy is None:
                measure.parallax_reason = (
                    f"écart de {delta:.1f}° depuis {nearest.viewpoint_id} — "
                    "aucune politique de recherche : la mesure est publiée, "
                    "la préférence n'est pas calculée"
                )
            else:
                measure.parallax_utility = parallax_utility(delta, separation, policy)
                measure.parallax_reason = (
                    f"écart de {delta:.1f}° et base de {separation:.1f} m depuis "
                    f"{nearest.viewpoint_id} — {_utility_note(delta, policy)}"
                )
        else:
            measure.parallax_reason = "position de la cible inconnue"

    # --- continuité : seulement si une séquence est connue -------------------
    sequences = sequence_of or {}
    known_sequences = anchor_sequences or set()
    measure.sequence_status = sequence_status
    measure.sequence_id = sequences.get(candidate.candidate_id)

    if sequence_status is SequenceStatus.NOT_QUERIED:
        measure.continuity_reason = (
            "séquence non interrogée : l'enrichissement vient après la "
            "présélection"
        )
    elif sequence_status is SequenceStatus.QUERY_ERROR:
        measure.continuity_reason = (
            "enrichissement en échec : la vue reste utilisable en géométrie, "
            "son gain de continuité est indisponible"
        )
    elif measure.sequence_id is None:
        measure.continuity_reason = "le fournisseur n'a rendu aucune séquence"
    elif measure.sequence_id in known_sequences:
        # Même séquence qu'une vue déjà retenue : recouvrement **plausible**.
        measure.continuity_gain = 1.0
        measure.continuity_level = ContinuityLevel.POTENTIAL
        measure.continuity_reason = (
            f"même séquence {measure.sequence_id} qu'une vue retenue — "
            "recouvrement plausible, non mesuré"
        )
    else:
        measure.continuity_gain = 0.0
        measure.continuity_level = ContinuityLevel.POTENTIAL
        measure.continuity_reason = "séquence connue, distincte de celles retenues"

    return measure


def parallax_utility(bearing_delta_deg: float, baseline_m: float, policy) -> float:  # noqa: ANN001
    """Utilité d'un écart angulaire, **selon la politique**. Non monotone.

    Un écart plus grand donne plus de profondeur jusqu'à un point, puis les
    deux vues cessent de partager assez de surface pour qu'un appariement
    fonctionne. Préférer systématiquement le plus grand angle — ce que faisait
    `delta / 90` — sélectionnait de la diversité en croyant sélectionner de la
    parallaxe : à 100°, deux vues peuvent ne montrer presque rien en commun.

    Aucun seuil n'est écrit ici : les trois viennent de la politique, et
    changer la préférence ne doit toucher aucune mesure brute.
    """
    if baseline_m < policy.baseline_min_m:
        # Un écart angulaire mesuré sur deux positions confondues ne
        # correspond à aucune parallaxe exploitable.
        return 0.0

    low = policy.parallax_preferred_min_deg
    high = policy.parallax_preferred_max_deg

    if bearing_delta_deg < low:
        # Montée linéaire : peu de profondeur, mais du recouvrement.
        return round(max(bearing_delta_deg / low, 0.0) * 0.8, 4) if low else 0.0
    if bearing_delta_deg <= high:
        return 1.0

    penalised = 1.0 - (bearing_delta_deg - high) * policy.parallax_excess_penalty
    return round(max(penalised, 0.0), 4)


def _utility_note(bearing_delta_deg: float, policy) -> str:  # noqa: ANN001
    if bearing_delta_deg < policy.parallax_preferred_min_deg:
        return "sous la plage préférée : peu de profondeur"
    if bearing_delta_deg <= policy.parallax_preferred_max_deg:
        return "dans la plage préférée"
    return "au-delà de la plage préférée : recouvrement probablement dégradé"


def pairwise_parallax_utility(
    bearing_delta_deg: float, baseline_m: float, policy  # noqa: ANN001
) -> float:
    """Même préférence, entre deux **nouveaux** candidats.

    Un candidat à 25° peut être préféré à un candidat à 100° : le second
    apporte de la diversité, non de la parallaxe exploitable.
    """
    return parallax_utility(bearing_delta_deg, baseline_m, policy)


def pairwise_parallax_potential(first, second, target_lat: float, target_lon: float) -> dict:  # noqa: ANN001
    """Diversité entre deux **nouveaux** candidats.

    L'absence d'ancre existante ne veut pas dire qu'on peut retenir deux vues
    quasi identiques. Ce gain-là se mesure entre candidats, et reste
    `potential` : aucune image n'a été inspectée.
    """
    from .visibility import bearing_deg, haversine_m

    baseline = haversine_m(
        first.camera_lat, first.camera_lon, second.camera_lat, second.camera_lon
    )
    first_bearing = bearing_deg(
        first.camera_lat, first.camera_lon, target_lat, target_lon
    )
    second_bearing = bearing_deg(
        second.camera_lat, second.camera_lon, target_lat, target_lon
    )
    delta = abs((first_bearing - second_bearing + 180.0) % 360.0 - 180.0)

    return {
        "baseline_m": round(baseline, 1),
        "bearing_delta_deg": round(delta, 1),
        "level": ContinuityLevel.POTENTIAL.value,
        "note": (
            "potentiel : mesuré sur des positions annoncées, aucune image "
            "n'ayant été inspectée"
        ),
    }


def select_for_demand(
    measures: list[CandidateMeasure],
    candidates: dict,
    demand,  # noqa: ANN001
    target_lat: float | None,
    target_lon: float | None,
    wanted: int,
    policy=None,  # noqa: ANN001 — AdaptiveSearchPolicy
) -> list[str]:
    """Retient les vues d'un besoin, la seconde par gain marginal.

    Déterministe et indépendant de l'ordre de l'API : les égalités se départagent
    par identifiant, sans quoi deux exécutions rendraient deux plans.
    """
    eligible = [m for m in measures if m.rejection_reason is None]

    if policy is not None:
        # Le classement ne borne rien : sans ce filtre, le premier du tri est
        # retenu même s'il est à des kilomètres de la cible.
        limit = policy.max_distance_to_target_m
        for measure in eligible:
            if (
                measure.distance_to_target_m is not None
                and measure.distance_to_target_m > limit
            ):
                measure.rejection_reason = (
                    f"à {measure.distance_to_target_m:.0f} m de la cible, "
                    f"au-delà de la portée retenue ({limit:.0f} m)"
                )
        eligible = [m for m in eligible if m.rejection_reason is None]

    if not eligible:
        return []

    ordered = sorted(
        eligible,
        key=lambda m: (
            -m.coverage_gap_priority,
            -(m.parallax_utility if m.parallax_utility is not None else 0.0),
            -(m.continuity_gain if m.continuity_gain is not None else 0.0),
            m.distance_to_target_m if m.distance_to_target_m is not None else math.inf,
            m.candidate_id,
        ),
    )

    retained = [ordered[0].candidate_id]
    if wanted <= 1 or target_lat is None or target_lon is None:
        return retained[:wanted]

    # Les suivantes par **gain marginal** contre les retenues : sans cela, deux
    # vues quasi identiques passeraient toutes deux.
    while len(retained) < wanted:
        best, best_gain = None, -1.0
        for measure in ordered:
            if measure.candidate_id in retained:
                continue
            candidate = candidates.get(measure.candidate_id)
            if candidate is None or candidate.camera_lat is None:
                continue
            # La préférence, non l'angle brut : maximiser l'écart retiendrait
            # deux vues sans surface commune, en croyant maximiser la parallaxe.
            gains = []
            for kept in retained:
                other = candidates.get(kept)
                if other is None:
                    continue
                pair = pairwise_parallax_potential(
                    other, candidate, target_lat, target_lon
                )
                gains.append(
                    pairwise_parallax_utility(
                        pair["bearing_delta_deg"], pair["baseline_m"], policy
                    )
                    if policy is not None
                    else pair["bearing_delta_deg"]
                )
            marginal = min(gains) if gains else 0.0
            if marginal > best_gain or (
                marginal == best_gain and best and measure.candidate_id < best
            ):
                best, best_gain = measure.candidate_id, marginal
        if best is None:
            break
        retained.append(best)

    return retained


def shortlist_for_enrichment(
    measures_by_demand: dict[str, list[CandidateMeasure]], per_demand: int
) -> list[str]:
    """Identifiants à enrichir d'une séquence, bornés et **dédoublonnés**.

    L'enrichissement coûte un appel par image. Un candidat servant deux besoins
    n'en vaut qu'un : l'appeler deux fois dépenserait le budget sur une
    information qu'on possède déjà.

    L'ordre est déterministe : la priorité de couverture d'abord, puis
    l'identifiant. Deux exécutions rendent la même liste.
    """
    wanted: list[str] = []
    seen: set[str] = set()

    for demand_id in sorted(measures_by_demand):
        eligible = [
            measure for measure in measures_by_demand[demand_id]
            if measure.rejection_reason is None
        ]
        ordered = sorted(
            eligible,
            key=lambda m: (
                -m.coverage_gap_priority,
                -(m.parallax_utility if m.parallax_utility is not None else 0.0),
                m.distance_to_target_m if m.distance_to_target_m is not None else math.inf,
                m.candidate_id,
            ),
        )
        for measure in ordered[:per_demand]:
            if measure.candidate_id in seen:
                continue
            seen.add(measure.candidate_id)
            wanted.append(measure.candidate_id)

    return wanted


def expand_sequences(
    seeds: dict[str, str],
    members_of,  # noqa: ANN001 — callable(sequence_id) -> list de candidats
    target_lat: float,
    target_lon: float,
    policy,  # noqa: ANN001 — CollectionPolicy
    known_ids: set[str],
) -> tuple[list, dict[str, str], int]:
    """Prolonge les séquences prometteuses, **dans la zone utile seulement**.

    Suivre une séquence entière la mènerait ailleurs : un véhicule roule, et
    ses vues suivantes montrent une autre rue. Deux bornes s'appliquent — un
    nombre de membres et une distance à la cible — et chaque rejet porte son
    motif.
    """
    from .visibility import haversine_m

    added: list = []
    rejected: dict[str, str] = {}
    calls = 0

    for sequence_id in sorted(set(seeds.values())):
        try:
            members = members_of(sequence_id)
            calls += 1
        except (OSError, RuntimeError, ValueError) as exc:
            rejected[sequence_id] = f"expansion impossible : {exc}"
            continue

        kept = 0
        for member in members:
            if member.candidate_id in known_ids:
                continue
            if kept >= policy.sequence_expansion_max_members:
                rejected[member.candidate_id] = (
                    f"au-delà de {policy.sequence_expansion_max_members} membres "
                    "explorés pour cette séquence"
                )
                continue
            if member.camera_lat is None or member.camera_lon is None:
                rejected[member.candidate_id] = "position inconnue"
                continue
            distance = haversine_m(
                member.camera_lat, member.camera_lon, target_lat, target_lon
            )
            if distance > policy.sequence_expansion_max_distance_m:
                rejected[member.candidate_id] = (
                    f"à {distance:.0f} m de la cible, au-delà de "
                    f"{policy.sequence_expansion_max_distance_m:.0f} m — hors zone utile"
                )
                continue
            added.append(member)
            known_ids.add(member.candidate_id)
            kept += 1

    return added, rejected, calls
