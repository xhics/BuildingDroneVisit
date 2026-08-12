"""Temporalité par portée (Lot 1B, audit du câblage).

Un statut global ne suffit pas : une photographie peut montrer une entrée
rénovée et une façade inchangée. L'évaluation se fait donc **par portée** —
`entrance`, `facade`, `roof`, `signage` — et chacune répond séparément.

Cinq règles, dans cet ordre :

1. dériver uniquement depuis une vraie date de capture et un jalon attesté ;
2. rendre `unknown` pour une approbation seule, une date de publication ou une
   date absente ;
3. autoriser l'image pour la géométrie selon la politique ;
4. n'interdire que son usage d'apparence, dans la portée concernée ;
5. conserver une décision humaine explicite, prioritaire et jamais recalculée.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .logging import get_logger
from .schemas import Asset, PropertyProfile, TemporalStatus
from .schemas.policy import DEFAULT_POLICY, PipelinePolicy

log = get_logger("temporal")

#: Sujets par lesquels une portée devient visible dans une image.
#:
#: Les deux vocabulaires ne coïncident pas : la portée `signage` désigne des
#: travaux d'enseigne, le sujet `sign` désigne une enseigne à l'écran. Sans
#: cette table, `signage` ne se déclenchait jamais.
SCOPE_SUBJECTS: dict[str, tuple[str, ...]] = {
    "entrance": ("entrance",),
    "signage": ("sign",),
    "facade": ("building",),
    "roof": ("roof",),
    "grounds": ("grounds",),
    "parking": ("parking",),
}


def subjects_for_scope(scope: str) -> tuple[str, ...]:
    """Sujets révélant une portée. Une portée inconnue se cherche sous son nom."""
    return SCOPE_SUBJECTS.get(scope, (scope,))


@dataclass
class TemporalReport:
    total: int = 0
    by_scope: dict[str, dict[str, int]] = field(default_factory=dict)
    human_decisions: int = 0
    sensitive_unknown: int = 0

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "by_scope": self.by_scope,
            "human_decisions": self.human_decisions,
            "sensitive_scopes_undetermined": self.sensitive_unknown,
        }


def derive_scope(
    asset: Asset, profile: PropertyProfile | None, scope: str
) -> tuple[TemporalStatus, str]:
    """Statut d'une portée pour un asset, et la méthode qui l'établit.

    Une année de capture ne situe pas une image à l'intérieur de l'année des
    travaux : la comparaison se fait donc strictement, et l'année du chantier
    elle-même reste indécise.
    """
    decision = next((d for d in asset.temporal_decisions if d.scope == scope), None)
    if decision is not None:
        return decision.status, f"revue humaine ({decision.decided_by})"

    if profile is None:
        return TemporalStatus.UNKNOWN, "aucun profil"

    event = profile.latest_event(scope)
    if event is None:
        return TemporalStatus.UNKNOWN, "aucun travaux déclaré pour cette portée"

    if asset.capture_year is None:
        return TemporalStatus.UNKNOWN, "date de capture inconnue"

    if event.started_on and asset.capture_year < event.started_on.year:
        return TemporalStatus.BEFORE_EVENT, "capture antérieure au début attesté"

    if event.establishes_current_appearance and asset.capture_year > event.completed_on.year:
        return TemporalStatus.CURRENT_CONFIRMED, "capture postérieure à l'achèvement confirmé"

    # Approbation seule, chantier en cours, ou achèvement non confirmé.
    return TemporalStatus.UNKNOWN, "jalon insuffisant pour trancher"


def assess(
    assets: list[Asset],
    profile: PropertyProfile | None,
    policy: PipelinePolicy = DEFAULT_POLICY,
) -> TemporalReport:
    """Évalue chaque portée déclarée, en place."""
    report = TemporalReport(total=len(assets))
    scopes = sorted({event.scope for event in (profile.renovation_events if profile else [])})
    if not scopes:
        scopes = list(policy.temporal.sensitive_scopes)

    for index, asset in enumerate(assets):
        by_scope: dict[str, TemporalStatus] = {}
        methods: list[str] = []

        for scope in scopes:
            status, method = derive_scope(asset, profile, scope)
            by_scope[scope] = status
            methods.append(f"{scope}:{method}")
            counts = report.by_scope.setdefault(scope, {})
            counts[status.value] = counts.get(status.value, 0) + 1

        if asset.temporal_decisions:
            report.human_decisions += 1

        assets[index] = asset.model_copy(
            update={
                "temporal_by_scope": by_scope,
                "temporal_status": _aggregate(by_scope),
                "temporal_method": " | ".join(methods) or None,
            }
        )

    report.sensitive_unknown = len(
        [a for a in assets if undetermined_sensitive_scopes(a, policy)]
    )
    log.info(
        "temporalité : %d asset(s), %d décision(s) humaine(s), "
        "%d avec une portée sensible indéterminée",
        report.total,
        report.human_decisions,
        report.sensitive_unknown,
    )
    return report


def _aggregate(by_scope: dict[str, TemporalStatus]) -> TemporalStatus:
    """Statut global, retenu au plus restrictif.

    Il ne sert qu'aux résumés : les décisions se prennent par portée.
    """
    statuses = set(by_scope.values())
    if TemporalStatus.BEFORE_EVENT in statuses:
        return TemporalStatus.BEFORE_EVENT
    if statuses == {TemporalStatus.CURRENT_CONFIRMED}:
        return TemporalStatus.CURRENT_CONFIRMED
    return TemporalStatus.UNKNOWN


def undetermined_sensitive_scopes(
    asset: Asset, policy: PipelinePolicy = DEFAULT_POLICY
) -> list[str]:
    """Portées sensibles que l'asset **montre** sans qu'on sache les dater.

    Une image qui ne montre pas l'entrée n'a pas à être datée sur l'entrée :
    bloquer tout le corpus parce qu'une portée sensible existe reviendrait à
    interdire la géométrie pour une question d'apparence.
    """
    from .schemas import Subject

    shown = {s.value for s in asset.subjects}
    return [
        scope
        for scope in policy.temporal.sensitive_scopes
        if shown.intersection(subjects_for_scope(scope))
        and asset.temporal_by_scope.get(scope, TemporalStatus.UNKNOWN)
        is TemporalStatus.UNKNOWN
    ]


def appearance_allowed(
    asset: Asset, scope: str, policy: PipelinePolicy = DEFAULT_POLICY
) -> bool:
    """L'asset peut-il servir de référence d'apparence pour cette portée ?"""
    status = asset.temporal_by_scope.get(scope, TemporalStatus.UNKNOWN)
    if status is TemporalStatus.CURRENT_CONFIRMED:
        return True
    if status is TemporalStatus.UNKNOWN:
        return policy.temporal.allow_unknown_for_appearance
    return False
