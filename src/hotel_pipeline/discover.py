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
from .schemas.acquisition import (
    CandidateManifest,
    CaptureCandidate,
    DemandRecommendation,
)

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

    #: Cadrages regroupés faute de différer : `écarté` → `retenu`.
    framings_merged: dict = field(default_factory=dict)

    #: Cadrages, panoramas et points de vue comptés **séparément**. Un seul
    #: chiffre les confondrait, et 1442 candidats pour 721 panoramas se lirait
    #: comme un doublon alors que ce sont deux acquisitions légitimes.
    viewpoint_counts: dict = field(default_factory=dict)

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
            "viewpoint_counts": self.viewpoint_counts,
            "framings_merged": self.framings_merged,
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
                # Le trajet dont la vue fait partie, quand la source le publie.
                # Sans lui, la continuité restait inconnue et tout besoin
                # l'exigeant demeurait borné à l'aperçu.
                sequence_id=getattr(image, "sequence_id", None),
                # Ce que la caméra déclare. Le cadrage exige un cap, un champ
                # de vision et une largeur : il en manquait deux sur trois.
                camera_type=getattr(image, "camera_type", None),
                requested_fov_deg=_declared_fov(image),
                advertised_width=getattr(image, "width_px", None),
                advertised_height=getattr(image, "height_px", None),
                # Le nécessaire à la reconstruction de l'adresse, et rien qui
                # ressemble à un secret ou à une URL.
                request_spec={
                    "provider_id": provider_id,
                    "resolution": "thumb_2048",
                },
                # `thumb_256` existe à l'API et n'était pas déclaré : un plan
                # demandant un aperçu était refusé faute de l'avoir dit.
                available_resolutions=sorted(
                    {"thumb_256", "thumb_2048", "256", "2048"}
                ),
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


def merge_near_identical_framings(
    candidates: list[CaptureCandidate], bearing_tolerance_deg: float,
) -> tuple[list[CaptureCandidate], dict[str, str]]:
    """Réunit les cadrages d'un même panorama dont les caps se confondent.

    Deux cadrages séparés de 1,5° montrent la même chose et coûtent deux
    requêtes. Les fusionner n'est pas une déduplication d'identité — ce sont
    bien deux acquisitions possibles — mais un choix : à contenu identique, une
    seule suffit.

    Les écartés sont **rendus**, avec le cadrage qui les remplace. Les perdre
    ferait disparaître la trace de ce qui a été proposé puis regroupé.
    """
    if bearing_tolerance_deg <= 0:
        return candidates, {}

    kept: list[CaptureCandidate] = []
    merged: dict[str, str] = {}
    by_panorama: dict[tuple, list[CaptureCandidate]] = {}

    for candidate in sorted(candidates, key=lambda c: c.candidate_id):
        if not candidate.panorama_id:
            kept.append(candidate)
            continue
        heading = candidate.requested_heading_deg
        if heading is None:
            kept.append(candidate)
            continue

        # Deux cadrages ne se confondent que si **tout** le reste coïncide.
        # Comparer les seuls caps réunirait un gros plan et un grand angle pris
        # dans la même direction : ils ne montrent pas la même chose.
        signature = (
            candidate.panorama_id,
            candidate.requested_fov_deg,
            candidate.requested_pitch_deg,
            candidate.advertised_width,
            candidate.advertised_height,
        )
        group = by_panorama.setdefault(signature, [])
        twin = next(
            (
                other for other in group
                if abs(
                    (other.requested_heading_deg - heading + 180.0) % 360.0 - 180.0
                ) <= bearing_tolerance_deg
            ),
            None,
        )
        if twin is not None:
            merged[candidate.candidate_id] = twin.candidate_id
            continue
        group.append(candidate)
        kept.append(candidate)

    return kept, merged


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

    tolerance = getattr(search, "framing_merge_bearing_deg", None) if search else None
    if tolerance:
        unique, merged = merge_near_identical_framings(unique, tolerance)
        # Trace d'audit : les regroupés sont nommés, avec leur remplaçant. Un
        # simple compteur ne dirait pas **lequel** a été proposé puis écarté.
        report.framings_merged = merged

    evaluations, recommended, search_report, by_level, recommendations = (
        _adaptive_pass(hotel_id, unique, search, report)
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
        recommendations=recommendations,
        recommended_for_enrichment=sorted(by_level["enrichment"]),
        recommended_for_preview=sorted(by_level["preview"]),
        eligible_for_full_acquisition=sorted(by_level["full"]),
        adaptive_search_run_id=search_report.run_id if search_report else None,
        adaptive_search_report_digest=(
            digest_of(search_report.as_dict()) if search_report else None
        ),
        demand_digest=demand_digest,
        policy_digest=policy_digest,
    )
    report.search = search_report
    panoramas = {c.panorama_id for c in unique if c.panorama_id}
    report.viewpoint_counts = {
        # Trois nombres distincts, parce que ce sont trois choses distinctes.
        "framing_candidates": len(unique),
        "distinct_panoramas": len(panoramas),
        "viewpoints": len(set(_viewpoints_of(unique, search).values())),
        "note": (
            "deux cadrages d'un même panorama sont deux acquisitions et un "
            "seul point de vue : les quotas se comptent en points de vue"
        ),
    }
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
    from .adaptive_search import (
        SearchReport,
        SequenceStatus,
        distance_distribution,
        measure_candidate,
        select_for_demand,
    )
    from .schemas.acquisition import CandidateEvaluation, Eligibility

    empty_levels = {"enrichment": set(), "preview": set(), "full": set()}
    if search is None or not getattr(search, "outstanding", None):
        return [], set(), None, empty_levels, []

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

    _declare_stages(published, search)

    evaluations: list[CandidateEvaluation] = []
    recommended: set[str] = set()
    by_id = {c.candidate_id: c for c in candidates}

    # Les quotas se comptent en points de vue. Ceux du corpus existant sont
    # déjà groupés ; les candidats le sont ici, par la même règle.
    viewpoints = dict(getattr(search, "viewpoints", None) or {})
    viewpoints.update(_viewpoints_of(candidates, search))

    # Séquences rendues par les sources. Une source qui n'en publie pas laisse
    # le statut à `NOT_RETURNED` : « le fournisseur n'en a pas rendu » n'est pas
    # « nous n'avons pas demandé ».
    sequences = {
        candidate.candidate_id: candidate.sequence_id
        for candidate in candidates
        if candidate.sequence_id
    }
    sequence_status = (
        SequenceStatus.KNOWN if sequences else SequenceStatus.NOT_RETURNED
    )

    for demand in search.outstanding:
        anchors = search.anchors.get(demand.demand_id, [])
        # Séquences déjà servies par une ancre de **ce** besoin : y appartenir
        # rend le recouvrement plausible, jamais certain.
        anchor_sequences = {
            sequences[anchor.viewpoint_id]
            for anchor in anchors
            if anchor.viewpoint_id in sequences
        }
        # Un besoin sans ancre compatible est le plus urgent : c'est un secteur
        # que rien ne couvre encore.
        priority = 3 if not anchors else 2

        measures = [
            measure_candidate(
                candidate, demand, anchors, priority,
                target_lat=target_lat, target_lon=target_lon,
                policy=search.policy,
                sector=getattr(search, "sector", None),
                sequence_of=sequences,
                anchor_sequences=anchor_sequences,
                sequence_status=sequence_status,
            )
            for candidate in candidates
        ]
        published.measures.extend(measures)

        retained = select_for_demand(
            measures, by_id, demand, target_lat, target_lon,
            wanted=demand.viewpoints_required, policy=search.policy,
            viewpoints=viewpoints,
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

    # Ce que le seuil de distance écarte, besoin par besoin : il reste non
    # calibré, et son effet doit être lisible avant qu'on le modifie.
    if search.policy is not None:
        published.distance_distribution = distance_distribution(
            published.measures, search.policy.automatic_candidate_max_distance_m
        )

    log.info(
        "recherche adaptative : %d besoin(s) ouvert(s), %d candidat(s) "
        "recommandé(s) sur %d mesuré(s) — le plan reste seul à décider",
        len(search.outstanding), len(recommended), len(candidates),
    )
    # Une mesure peut servir plusieurs besoins à des niveaux différents. Le
    # **plus prudent** l'emporte : autoriser une acquisition complète parce
    # qu'un autre besoin s'en contentait perdrait la réserve du premier.
    by_level = {"enrichment": set(), "preview": set(), "full": set()}
    graded = {}
    for measure in published.measures:
        level = measure.recommendation_level
        if level is None or measure.candidate_id not in recommended:
            continue
        rank = {"recommended_for_enrichment": 0,
                "recommended_for_preview": 1,
                "eligible_for_full_acquisition": 2}[level.value]
        graded[measure.candidate_id] = min(
            graded.get(measure.candidate_id, rank), rank
        )
    for candidate_id, rank in graded.items():
        by_level[("enrichment", "preview", "full")[rank]].add(candidate_id)

    # L'autorité : couple par couple. Un niveau porté par le seul candidat
    # laissait une autorisation obtenue pour un besoin en couvrir un autre.
    recommendations = [
        DemandRecommendation(
            candidate_id=measure.candidate_id,
            demand_id=measure.demand_id,
            level=measure.recommendation_level.value,
            # Le schéma refuse un motif vide : une autorisation muette ne se
            # conteste pas. Si `_grade` en oubliait un, mieux vaut le dire que
            # publier une chaîne vide qui aurait l'air d'une explication.
            reason=(
                measure.recommendation_reason
                or "motif non renseigné par la recherche"
            ),
            unmeasured_requirements=list(measure.unmeasured_requirements),
        )
        for measure in published.measures
        if measure.recommendation_level is not None
        and measure.candidate_id in published.recommended.get(measure.demand_id, [])
    ]

    # Le résumé doit **dériver** des couples, sinon les deux divergent et le
    # schéma refuse le manifeste — ce qui est le comportement voulu.
    by_level = {"enrichment": set(), "preview": set(), "full": set()}
    short = {
        "recommended_for_enrichment": "enrichment",
        "recommended_for_preview": "preview",
        "eligible_for_full_acquisition": "full",
    }
    ranked = {"enrichment": 0, "preview": 1, "full": 2}
    best: dict[str, str] = {}
    for entry in recommendations:
        key = short[entry.level]
        if entry.candidate_id not in best or ranked[key] < ranked[best[entry.candidate_id]]:
            best[entry.candidate_id] = key
    for candidate_id, key in best.items():
        by_level[key].add(candidate_id)

    return evaluations, recommended, published, by_level, recommendations


def _viewpoints_of(candidates: list, search) -> dict:  # noqa: ANN001
    """Point de vue de chaque candidat, par la règle du plan — non une seconde.

    Deux cadrages d'un même panorama sont deux acquisitions et une seule
    observation : `pano:<panorama_id>` les réunit, et un SfM ne tirerait aucune
    parallaxe de leur paire.
    """
    from .plan import group_viewpoints

    separation = getattr(search, "viewpoint_separation_m", None)
    if separation is None:
        # Sans seuil, on ne groupe **que** par panorama : deux positions
        # distinctes restent deux points de vue plutôt que d'être réunies par
        # une distance inventée ici.
        return {
            c.candidate_id: (
                f"pano:{c.panorama_id}" if c.panorama_id else c.candidate_id
            )
            for c in candidates
        }
    return group_viewpoints(candidates, separation_m=separation)


def _declare_stages(published, search) -> None:  # noqa: ANN001
    """Dit ce qui n'a pas eu lieu — un zéro ne le dirait pas.

    Une étape non exécutée et une étape sans résultat produisent le même
    compteur à zéro. Sans cette déclaration, la seconde passe pouvait rester
    non câblée en présentant les mêmes chiffres qu'une recherche complète.
    """
    enrich = getattr(search, "enrich_sequences", None)
    if enrich is None:
        published.stages_skipped["metadata_enrichment"] = (
            "aucun client d'enrichissement de séquence fourni : la continuité "
            "reste inconnue, elle n'est pas nulle"
        )
        published.stages_skipped["sequence_expansion"] = (
            "sans enrichissement préalable, aucune séquence à prolonger"
        )

    if not getattr(search, "requests_by_source", None):
        published.stages_skipped["request_provenance"] = (
            "les collecteurs ne rendent pas encore leur décompte d'appels : "
            "le coût réel des requêtes n'est pas mesuré"
        )


#: Plafond du champ de vision qu'un candidat peut déclarer. Une image
#: sphérique voit à 360°, ce que le schéma refuse — et à juste titre : « cadrer »
#: n'a pas de sens avant qu'un cap et une ouverture soient choisis. La vue
#: reste au manifeste, son cadrage attend l'extraction.
MAX_DECLARABLE_FOV_DEG = 120.0


def _declared_fov(image) -> float | None:  # noqa: ANN001
    """Champ de vision utilisable pour juger un cadrage, ou `None`.

    Un panorama complet n'a pas de cadrage tant qu'on n'en a pas extrait une
    vue : rendre 360° ferait croire à une mesure là où il n'y a qu'une
    promesse.
    """
    fov = getattr(image, "fov_deg", None)
    if fov is None or fov <= 0 or fov > MAX_DECLARABLE_FOV_DEG:
        return None
    return fov
