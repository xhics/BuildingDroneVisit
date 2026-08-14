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

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "sources_queried": self.sources_queried,
            "sources_skipped": self.sources_skipped,
            "candidates_by_source": self.candidates_by_source,
            "duplicates_dropped": self.duplicates_dropped,
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

    manifest = CandidateManifest(
        hotel_id=hotel_id,
        # `queries` porte le nombre d'entités rendues **par source
        # interrogée** : sans lui, un zéro ne distingue pas une source vide
        # d'une source jamais appelée.
        queries=dict(report.candidates_by_source),
        demand_digest=demand_digest,
        policy_digest=policy_digest,
        candidates=sorted(unique, key=lambda c: c.candidate_id),
    )
    log.info(
        "découverte %s : %d candidat(s) de %d source(s), %d doublon(s) écarté(s), "
        "0 octet téléchargé",
        report.run_id, len(unique), len(report.sources_queried), dropped,
    )
    return manifest, report
