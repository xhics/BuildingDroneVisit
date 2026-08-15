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

    #: `None` quand aucune ancre géométrique ne sert ce besoin.
    parallax_gain: float | None = None
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
                "gain_against_existing": self.parallax_gain,
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
    """Ce qui a été cherché, pour quels besoins, et ce qui en est ressorti."""

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
        measure.parallax_gain = None
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

        if target_lat is not None and target_lon is not None:
            from_candidate = bearing_deg(
                candidate.camera_lat, candidate.camera_lon, target_lat, target_lon
            )
            from_anchor = bearing_deg(nearest.lat, nearest.lon, target_lat, target_lon)
            delta = abs((from_candidate - from_anchor + 180.0) % 360.0 - 180.0)
            measure.bearing_delta_from_anchor_deg = round(delta, 1)
            # Le gain de parallaxe croît avec l'écart angulaire vers la cible :
            # c'est lui qui donne de la profondeur, non la distance seule.
            measure.parallax_gain = round(min(delta / 90.0, 1.0), 4)
            measure.parallax_reason = (
                f"écart de {delta:.1f}° vers la cible depuis {nearest.viewpoint_id}"
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
) -> list[str]:
    """Retient les vues d'un besoin, la seconde par gain marginal.

    Déterministe et indépendant de l'ordre de l'API : les égalités se départagent
    par identifiant, sans quoi deux exécutions rendraient deux plans.
    """
    eligible = [m for m in measures if m.rejection_reason is None]
    if not eligible:
        return []

    ordered = sorted(
        eligible,
        key=lambda m: (
            -m.coverage_gap_priority,
            -(m.parallax_gain if m.parallax_gain is not None else 0.0),
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
            gains = [
                pairwise_parallax_potential(
                    candidates[kept], candidate, target_lat, target_lon
                )["bearing_delta_deg"]
                for kept in retained
                if candidates.get(kept) is not None
            ]
            marginal = min(gains) if gains else 0.0
            if marginal > best_gain or (
                marginal == best_gain and best and measure.candidate_id < best
            ):
                best, best_gain = measure.candidate_id, marginal
        if best is None:
            break
        retained.append(best)

    return retained
