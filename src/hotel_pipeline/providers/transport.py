"""Le seul endroit d'où part un appel réseau (collecte V2).

Le compteur précédent vivait dans `cached_call` : il comptait les **cache
misses**, non les requêtes. Deux pages Mapillary produisaient deux `GET` et un
seul décompte, les échecs n'étaient pas comptés, et plusieurs appels
contournaient le cache entièrement.

Ce module inverse la charge : rien ne part sans passer par lui, et chaque
tentative est inscrite **avant** l'appel — sinon un échec réseau
disparaîtrait avec l'exception.

```text
tentative   inscrite avant l'appel, toujours
issue       succès, erreur HTTP, erreur réseau, servi par le cache
étage       recherche grossière, enrichissement, mesure, acquisition
page        une par requête d'une même pagination
```

Ce que le registre ne porte **jamais** : URL, jeton, en-tête d'autorisation.
Un rapport versionné ne doit pas rendre un secret lisible, et une URL signée en
est un.
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import StrEnum

from ..logging import get_logger

log = get_logger("transport")


class NetworkMode(StrEnum):
    """Ce que l'exécution en cours s'autorise.

    Trois états, non deux : « le cache suffit » et « rien n'est permis » sont
    des situations différentes, et les confondre ferait passer un corpus figé
    pour une absence de données.
    """

    #: Cache d'abord, réseau sur miss.
    ONLINE = "online"

    #: Cache autorisé ; un miss est **refusé** sans appeler le producteur.
    #: C'est le mode des rejeux sur réponses authentiques figées.
    CACHE_ONLY = "cache_only"

    #: Ni cache ni réseau. Garde-fou des tests unitaires.
    FORBIDDEN = "forbidden"


class Stage(StrEnum):
    """À quel étage du travail cet appel appartient."""

    COARSE_SEARCH = "coarse_search"
    METADATA_ENRICHMENT = "metadata_enrichment"
    SEQUENCE_EXPANSION = "sequence_expansion"

    #: Résolution d'adresse : Mapillary ne publie pas d'URL durable, il faut la
    #: redemander. C'est un appel distinct de celui qui suit.
    URL_RESOLUTION = "url_resolution"

    VOLUME_PROBE = "volume_probe"
    DOWNLOAD = "download"


class Outcome(StrEnum):
    SUCCESS = "success"
    HTTP_ERROR = "http_error"
    NETWORK_ERROR = "network_error"
    CACHE_HIT = "cache_hit"

    #: Refusé par le mode réseau, sans qu'aucun paquet ne parte.
    REFUSED = "refused"


class NetworkRefused(RuntimeError):
    """Le mode en cours interdit cet appel."""


@dataclass
class Attempt:
    """Une tentative, réussie ou non. Inscrite avant l'appel."""

    source: str
    stage: Stage
    method: str
    outcome: Outcome = Outcome.SUCCESS

    #: Rang dans une pagination — la première page est 1. Sans lui, deux pages
    #: se lisaient comme une seule requête.
    page: int = 1

    status_code: int | None = None
    bytes_received: int | None = None
    error: str | None = None

    #: Ce à quoi l'appel se rapporte, quand une requête d'acquisition l'a
    #: motivé. Jamais l'URL : elle porte le jeton.
    request_digest: str | None = None

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "stage": self.stage.value,
            "method": self.method,
            "outcome": self.outcome.value,
            "page": self.page,
            "status_code": self.status_code,
            "bytes_received": self.bytes_received,
            "error": self.error,
            "request_digest": self.request_digest,
        }


@dataclass
class Ledger:
    """Ce qui a été tenté pendant **une** exécution.

    Porté par un `ContextVar` : deux commandes concurrentes ne doivent pas
    additionner leurs appels, et un total cumulé entre deux exécutions ne dirait
    rien de l'une ni de l'autre.
    """

    attempts: list[Attempt] = field(default_factory=list)

    #: Ce que l'exécution s'attendait à émettre au plus. La pagination interdit
    #: souvent un nombre exact d'avance : annoncer un plafond vaut mieux que se
    #: taire, et le comparer à l'effectif dit si l'estimation valait.
    planned_max_requests: dict[str, int] = field(default_factory=dict)

    def record(self, attempt: Attempt) -> Attempt:
        self.attempts.append(attempt)
        return attempt

    def by_source(self) -> dict:
        """Décompte par source, issue par issue."""
        counts: dict[str, dict] = {}
        for attempt in self.attempts:
            row = counts.setdefault(
                attempt.source,
                {
                    "attempted": 0, "succeeded": 0, "http_errors": 0,
                    "network_errors": 0, "cache_hits": 0, "refused": 0,
                    "bytes_received": 0, "by_stage": {},
                },
            )
            # Une lecture de cache n'est pas une tentative réseau : les compter
            # ensemble surestimerait le coût, et les taire le masquerait.
            if attempt.outcome is Outcome.CACHE_HIT:
                row["cache_hits"] += 1
            elif attempt.outcome is Outcome.REFUSED:
                row["refused"] += 1
            else:
                row["attempted"] += 1
                if attempt.outcome is Outcome.SUCCESS:
                    row["succeeded"] += 1
                elif attempt.outcome is Outcome.HTTP_ERROR:
                    row["http_errors"] += 1
                else:
                    row["network_errors"] += 1

            if attempt.bytes_received:
                row["bytes_received"] += attempt.bytes_received
            stage = row["by_stage"].setdefault(attempt.stage.value, 0)
            row["by_stage"][attempt.stage.value] = stage + 1
        return counts

    def as_dict(self) -> dict:
        by_source = self.by_source()
        return {
            "planned_max_requests": self.planned_max_requests,
            "actual_requests": {
                source: row["attempted"] for source, row in by_source.items()
            },
            "by_source": by_source,
            "note": (
                "tentatives réseau réellement émises, pagination et échecs "
                "compris. Les lectures de cache sont comptées à part : les "
                "confondre surestimerait le coût"
            ),
        }


_LEDGER: ContextVar[Ledger | None] = ContextVar("transport_ledger", default=None)
_MODE: ContextVar[NetworkMode | None] = ContextVar("transport_mode", default=None)


def current_mode() -> NetworkMode:
    """Mode en vigueur, l'environnement faisant foi en dernier ressort."""
    explicit = _MODE.get()
    if explicit is not None:
        return explicit
    if os.environ.get("HOTEL_PIPELINE_OFFLINE") == "1":
        return NetworkMode.FORBIDDEN
    if os.environ.get("HOTEL_PIPELINE_CACHE_ONLY") == "1":
        return NetworkMode.CACHE_ONLY
    return NetworkMode.ONLINE


def set_mode(mode: NetworkMode | None):  # noqa: ANN201
    return _MODE.set(mode)


def ledger() -> Ledger:
    """Registre de l'exécution en cours, créé à la demande."""
    current = _LEDGER.get()
    if current is None:
        current = Ledger()
        _LEDGER.set(current)
    return current


def reset_ledger() -> Ledger:
    """Ouvre un registre neuf. À appeler au début de chaque commande."""
    fresh = Ledger()
    _LEDGER.set(fresh)
    return fresh


def record_cache_hit(source: str, stage: Stage) -> None:
    """Une réponse servie sans réseau. Comptée, mais à part."""
    ledger().record(
        Attempt(source=source, stage=stage, method="CACHE", outcome=Outcome.CACHE_HIT)
    )


def guard(source: str, stage: Stage, what: str) -> None:
    """Refuse l'appel si le mode ne l'autorise pas, et l'inscrit.

    Le refus est **inscrit** : sans cela, une exécution en mode fermé
    présenterait les mêmes compteurs qu'une exécution sans besoin réseau.
    """
    mode = current_mode()
    if mode is NetworkMode.ONLINE:
        return

    ledger().record(
        Attempt(
            source=source, stage=stage, method="-", outcome=Outcome.REFUSED,
            error=f"mode {mode.value}",
        )
    )
    if mode is NetworkMode.CACHE_ONLY:
        raise NetworkRefused(
            f"mode cache_only — {what} : la réponse n'est pas au cache, et "
            "aucun appel n'est émis. Rejouez sur un cache complet, ou passez "
            "en mode online."
        )
    raise NetworkRefused(f"mode hors ligne actif — appel {what} refusé")


def request(
    source: str,
    stage: Stage,
    method: str,
    caller,  # noqa: ANN001 — callable() -> réponse
    page: int = 1,
    request_digest: str | None = None,
    what: str | None = None,
):  # noqa: ANN201
    """Émet un appel, en l'inscrivant **avant** de le tenter.

    L'inscription précède l'appel : une erreur réseau emporterait l'exception
    et, avec elle, la trace de la tentative. Un registre qui ne compte que les
    succès ne mesure pas un coût, il mesure une chance.
    """
    guard(source, stage, what or f"{method} {source}")

    attempt = ledger().record(
        Attempt(
            source=source, stage=stage, method=method, page=page,
            request_digest=request_digest,
        )
    )
    try:
        response = caller()
    except Exception as exc:  # noqa: BLE001 — l'issue est inscrite, puis relancée
        attempt.outcome = Outcome.NETWORK_ERROR
        attempt.error = f"{type(exc).__name__}: {str(exc)[:120]}"
        raise

    attempt.status_code = getattr(response, "status_code", None)
    if attempt.status_code is not None and attempt.status_code >= 400:
        attempt.outcome = Outcome.HTTP_ERROR
        attempt.error = f"HTTP {attempt.status_code}"

    declared = None
    headers = getattr(response, "headers", None)
    if headers is not None:
        declared = headers.get("Content-Length")
    if declared is not None:
        try:
            attempt.bytes_received = int(declared)
        except (TypeError, ValueError):
            attempt.bytes_received = None

    return response
