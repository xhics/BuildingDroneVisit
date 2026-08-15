"""Découverte de candidats — **métadonnées seulement** (collecte V2, étape 1).

Aucun octet d'image ne traverse ce module. Il interroge les index des sources,
retient ce qu'elles disent d'une prise de vue possible, et s'arrête là. Le
choix vient ensuite (`plan`), le téléchargement encore après (`acquire`), et
l'OCR après l'acquisition — jamais avant, puisqu'à ce stade aucune image
n'existe.

Trois règles gouvernent le résultat :

- **le besoin juge la collecte**, non l'inverse : les candidats sont évalués
  contre des `CaptureDemand` énoncées d'avance, et un candidat sans besoin à
  servir n'est pas un candidat ;
- **aucune URL n'est conservée** : elle expire, ou porte une clé. Le manifeste
  garde de quoi la reconstruire, et le schéma refuse le reste ;
- **ce qui est annoncé n'est pas ce qui est mesuré** : les dimensions viennent
  du fournisseur, et le resteront jusqu'à ce qu'un fichier existe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from .logging import get_logger
from .provenance import digest_of
from .schemas.acquisition import CandidateManifest, CaptureCandidate

log = get_logger("discover")


class DiscoveryRefused(RuntimeError):
    """Rien n'a été interrogé, et rien n'a été écrit."""


@dataclass
class DiscoveryReport:
    """Ce qui a été demandé, à qui, et ce qui en est revenu."""

    run_id: str = ""
    sources_queried: list[str] = field(default_factory=list)
    sources_skipped: dict[str, str] = field(default_factory=dict)
    candidates_by_source: dict[str, int] = field(default_factory=dict)
    duplicates_dropped: int = 0

    #: Ce que la découverte **ne** dit pas. Inscrit au rapport pour qu'aucun
    #: lecteur ne prenne un candidat pour une image utilisable.
    limits: list[str] = field(default_factory=list)

    #: Appels émis, par source et par étage de recherche — jamais des candidats
    #: rendus. Confondre les deux ferait passer une source prolixe pour une
    #: source souvent interrogée.
    requests_by_source: dict = field(default_factory=dict)

    #: Rapport de recherche adaptative, quand elle a eu lieu.
    search: object = None

    def counts_by_source(self, unique: list, recommended: set) -> dict:
        """Effectifs par source : « zéro » cesse d'être ambigu."""
        from .schemas.acquisition import SourceCandidateCounts

        by_source: dict = {}
        for source, returned in self.candidates_by_source.items():
            kept = [c for c in unique if c.source == source]
            advised = [c for c in kept if c.candidate_id in recommended]
            by_source[source] = SourceCandidateCounts(
                returned=returned,
                unique=len(kept),
                recommended=len(advised),
                rejected=len(kept) - len(advised),
            )
        return by_source

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "sources_queried": self.sources_queried,
            "sources_skipped": self.sources_skipped,
            "candidates_by_source": self.candidates_by_source,
            "duplicates_dropped": self.duplicates_dropped,
            "requests_by_source": {
                source: counts.as_dict()
                for source, counts in self.requests_by_source.items()
            },
            "adaptive_search": self.search.as_dict() if self.search else None,
            "bytes_downloaded": 0,
            "limits": self.limits or LIMITS,
        }


#: Ce qu'une découverte ne peut pas établir. Générique : aucun nombre de
#: corpus, aucun nom d'établissement.
LIMITS = [
    "aucune image n'a été téléchargée : les dimensions sont annoncées par la "
    "source, non mesurées",
    "aucun cadrage n'est calculé ici — il dépend d'une géométrie de capture et "
    "d'un référentiel résolu",
    "la présence d'un candidat ne dit rien de ce qu'il montre : seule une "
    "revue ou une mesure le dira",
]


def candidates_from(source: str, images: list) -> list[CaptureCandidate]:
    """Convertit ce qu'une source d'index rend en candidats.

    C'est ici que les adresses disparaissent. Un collecteur rend une URL de
    vignette — signée, ou expirante ; la conserver mettrait une clé d'API dans
    un manifeste versionné, ou garantirait un lien mort. On garde de quoi la
    reconstruire, et `CaptureCandidate` refuse le reste.

    Les dimensions annoncées ne sont pas recopiées comme mesurées : elles
    n'existeront qu'après acquisition d'un fichier.
    """
    from .schemas.acquisition import capture_identity

    candidates: list[CaptureCandidate] = []
    for image in images:
        provider_id = str(image.source_id)
        candidates.append(
            CaptureCandidate(
                candidate_id=capture_identity(source, provider_id),
                source=source,
                provider_id=provider_id,
                camera_lat=image.lat,
                camera_lon=image.lon,
                original_heading_deg=image.heading_deg,
                heading_is_measured=image.heading_is_measured,
                captured_at=_captured_at(image),
                # Le nécessaire à la reconstruction de l'adresse, et rien qui
                # ressemble à un secret ou à une URL.
                request_spec={
                    "provider_id": provider_id,
                    "resolution": "thumb_2048",
                },
                available_resolutions=["thumb_2048"],
            )
        )
    return candidates


def _captured_at(image):  # noqa: ANN001, ANN201
    """Année de capture rendue par le collecteur, ramenée à une date.

    L'année seule est ce que la source publie ; la présenter comme un instant
    précis lui prêterait une exactitude qu'elle n'a pas. On la place donc au
    1er janvier, et le champ garde son sens : « pas plus précis que l'année ».
    """
    year = getattr(image, "captured_year", None)
    if not year:
        return None
    return datetime(int(year), 1, 1, tzinfo=timezone.utc)


def deduplicate(candidates: list[CaptureCandidate]) -> tuple[list[CaptureCandidate], int]:
    """Écarte les doublons d'identité, en conservant le premier vu.

    Deux interrogations d'un même index rendent les mêmes vues ; les empiler
    gonflerait le volume annoncé au plan, donc le consentement demandé.
    """
    seen: dict[str, CaptureCandidate] = {}
    dropped = 0
    for candidate in candidates:
        if candidate.candidate_id in seen:
            dropped += 1
            continue
        seen[candidate.candidate_id] = candidate
    return list(seen.values()), dropped


def discover(
    hotel_id: str,
    demands,  # noqa: ANN001 — CaptureDemandManifest
    queries: dict,
    run_id: str | None = None,
    demand_digest: str | None = None,
    policy_digest: str | None = None,
    search=None,  # noqa: ANN001 — AdaptiveContext
) -> tuple[CandidateManifest, DiscoveryReport]:
    """Construit le manifeste de candidats à partir de réponses déjà obtenues.

    `queries` associe un nom de source à une liste de candidats déjà
    construits, ou à une chaîne expliquant pourquoi elle n'a pas été
    interrogée. Séparer l'appel réseau de la construction rend la découverte
    rejouable, et testable sans clé.

    Une source en panne n'est pas une source vide : le motif est conservé, et
    le plan qui suivra saura qu'il ne juge pas un corpus complet.
    """
    if not demands.demands:
        raise DiscoveryRefused(
            "aucun besoin déclaré : la découverte serait un ramassage sans "
            "objectif, et le corpus définirait ce qu'on cherchait"
        )

    report = DiscoveryReport(
        run_id=run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    collected: list[CaptureCandidate] = []

    for source in sorted(queries):
        outcome = queries[source]
        if isinstance(outcome, str):
            report.sources_skipped[source] = outcome
            log.info("source %s non interrogée : %s", source, outcome)
            continue

        report.sources_queried.append(source)
        report.candidates_by_source[source] = len(outcome)
        collected.extend(outcome)

    unique, dropped = deduplicate(collected)
    report.duplicates_dropped = dropped

    evaluations, recommended, search_report = _adaptive_pass(
        hotel_id, unique, search, report,
    )

    # Les deux objets sont construits — et validés — avant toute écriture. Une
    # incohérence entre eux ne doit laisser ni manifeste neuf ni rapport
    # orphelin : c'est le couple qui est publié, ou rien.
    manifest = CandidateManifest(
        hotel_id=hotel_id,
        # `queries` porte le nombre d'entités rendues **par source
        # interrogée** : sans lui, un zéro ne distingue pas une source vide
        # d'une source jamais appelée.
        queries=dict(report.candidates_by_source),
        requests_by_source=report.requests_by_source,
        candidates_by_source=report.counts_by_source(unique, recommended),
        # Tous les candidats restent au manifeste, y compris les non
        # recommandés. Les retirer effacerait la trace de ce qui a été vu puis
        # écarté : « rien à l'arrière » ne se distinguerait plus de « rien
        # cherché à l'arrière ». La recherche présélectionne pour enrichir ;
        # seul le plan décide ce qui sera acquis.
        candidates=sorted(unique, key=lambda c: c.candidate_id),
        evaluations=evaluations,
        recommended_for_plan=sorted(recommended),
        adaptive_search_run_id=search_report.run_id if search_report else None,
        adaptive_search_report_digest=(
            digest_of(search_report.as_dict()) if search_report else None
        ),
        demand_digest=demand_digest,
        policy_digest=policy_digest,
    )
    report.search = search_report
    log.info(
        "découverte %s : %d candidat(s) de %d source(s), %d doublon(s) écarté(s), "
        "0 octet téléchargé",
        report.run_id, len(unique), len(report.sources_queried), dropped,
    )
    return manifest, report


def _adaptive_pass(hotel_id: str, candidates: list, search, report):  # noqa: ANN001
    """Mesure chaque candidat contre les besoins ouverts, et **recommande**.

    Recommander n'est pas décider. `assets plan` reste seul à trancher ce qui
    sera acquis ; ce qui est produit ici sert à choisir quoi enrichir, et à
    expliquer pourquoi un candidat n'a pas été retenu. Un candidat écarté
    conserve donc son évaluation.
    """
    from .adaptive_search import SearchReport, measure_candidate, select_for_demand
    from .schemas.acquisition import CandidateEvaluation, Eligibility

    if search is None or not getattr(search, "outstanding", None):
        return [], set(), None

    target_lat, target_lon = search.target or (None, None)
    published = SearchReport(
        run_id=report.run_id,
        hotel_id=hotel_id,
        demands_searched=[d.demand_id for d in search.outstanding],
        demands_skipped=dict(search.skipped),
        anchors_by_demand={
            demand_id: [anchor.viewpoint_id for anchor in anchors]
            for demand_id, anchors in search.anchors.items()
        },
        candidates_considered=[c.candidate_id for c in candidates],
        **(search.lineage or {}),
    )

    evaluations: list[CandidateEvaluation] = []
    recommended: set[str] = set()
    by_id = {c.candidate_id: c for c in candidates}

    for demand in search.outstanding:
        anchors = search.anchors.get(demand.demand_id, [])
        # Un besoin sans ancre compatible est le plus urgent : c'est un secteur
        # que rien ne couvre encore.
        priority = 3 if not anchors else 2

        measures = [
            measure_candidate(
                candidate, demand, anchors, priority,
                target_lat=target_lat, target_lon=target_lon,
                policy=search.policy,
            )
            for candidate in candidates
        ]
        published.measures.extend(measures)

        retained = select_for_demand(
            measures, by_id, demand, target_lat, target_lon,
            wanted=demand.viewpoints_required, policy=search.policy,
        )
        published.recommended[demand.demand_id] = list(retained)
        recommended.update(retained)

        if measures and not retained:
            # Tout écarter est un résultat, pas un silence : sans cette trace,
            # « aucun candidat proposé » et « aucun candidat mesuré » se
            # confondraient au rapport.
            published.all_rejected[demand.demand_id] = len(measures)

        evaluations.extend(
            CandidateEvaluation(
                candidate_id=measure.candidate_id,
                demand_id=demand.demand_id,
                intent=demand.intent,
                eligibility=(
                    Eligibility.REJECTED if measure.rejection_reason
                    else Eligibility.PREVIEW_REQUIRED
                ),
                rejection_reason=measure.rejection_reason,
            )
            for measure in measures
        )

    log.info(
        "recherche adaptative : %d besoin(s) ouvert(s), %d candidat(s) "
        "recommandé(s) sur %d mesuré(s) — le plan reste seul à décider",
        len(search.outstanding), len(recommended), len(candidates),
    )
    return evaluations, recommended, published
