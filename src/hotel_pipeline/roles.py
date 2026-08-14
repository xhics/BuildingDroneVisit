"""Affectation des rôles de reconstruction (Lot 1B §4).

Aucune source n'est écartée : elles sont affectées. Une photographie
promotionnelle sans position caméra ne peut pas porter de géométrie — c'est
structurel, pas une question de qualité — mais elle reste la meilleure preuve
d'apparence et de date dont on dispose.

Les règles sont déterministes et ordonnées : la première qui s'applique fixe le
rôle, et la raison est enregistrée à côté.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .logging import get_logger
from .schemas.policy import DEFAULT_POLICY, PipelinePolicy
from .schemas import (
    Asset,
    ClusterRole,
    GeometrySuitability,
    PropertyMatchStatus,
    ReconstructionRole,
    ReviewStatus,
    Subject,
    TemporalStatus,
)

log = get_logger("roles")


@dataclass
class RoleReport:
    counts: dict[str, int] = field(default_factory=dict)
    reasons: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"roles": self.counts, "reasons": self.reasons}


def role_for(
    asset: Asset, policy: PipelinePolicy = DEFAULT_POLICY
) -> tuple[ReconstructionRole, str]:
    """Rôle d'un asset et sa justification.

    Porter de la géométrie exige **toutes** les conditions suivantes :

    ```text
    caméra située
    + bâtiment cible réellement visible
    + aucune occultation non arbitrée
    + revue acceptée
    + point de vue actif
    + aptitude géométrique établie
    + temporalité admissible
    ```

    La version précédente se contentait de « position connue + un bâtiment
    visible », ce qui a promu 20 vues Street View montrant un concessionnaire
    automobile, une épicerie et un bureau d'ingénierie.
    """
    if Subject.INTERIOR in asset.subjects:
        return ReconstructionRole.REFERENCE_ONLY, "intérieur, hors périmètre extérieur"

    if asset.property_match_status is PropertyMatchStatus.MISMATCH:
        return ReconstructionRole.REJECT, "enseigne d'un autre établissement"

    positioned = asset.camera_lat is not None and asset.camera_lon is not None
    contains = bool(asset.contains_building or Subject.BUILDING in asset.subjects)

    if positioned:
        # Un blocage intégral peut être partagé entre plusieurs obstacles :
        # `occluded_by`, singulier, reste alors vide, et ne consulter que lui
        # laisserait passer une vue prouvée bouchée.
        if asset.line_of_sight_status == "blocked":
            return (
                ReconstructionRole.CONTEXT_LOCK,
                "ligne de vue intégralement bloquée, mesurée",
            )
        if asset.occluded_by:
            return ReconstructionRole.CONTEXT_LOCK, "ligne de visée masquée par un voisin"
        if asset.target_building_visible is not True:
            # Le rôle est le même — verrou de contexte — mais le motif n'est
            # pas le même : « pas encore jugé » et « jugé, indécidable » ne
            # demandent pas la même suite.
            if asset.review_status is ReviewStatus.HUMAN_UNRESOLVED:
                return (
                    ReconstructionRole.CONTEXT_LOCK,
                    "revue close sans conclusion : preuves insuffisantes",
                )
            return (
                ReconstructionRole.CONTEXT_LOCK,
                "bâtiment cible non établi" if contains else "environnement seulement",
            )
        if asset.review_status is ReviewStatus.REJECTED:
            return ReconstructionRole.REJECT, "rejeté en revue"
        if asset.review_status is ReviewStatus.NEEDS_REVIEW:
            return ReconstructionRole.CONTEXT_LOCK, "en attente de revue humaine"
        # Exiger un statut actif, et non seulement « pas inactif » : un asset
        # créé avant la déduplication porte `None` et franchissait le Router.
        if asset.cluster_role not in (ClusterRole.CANONICAL, ClusterRole.OVERLAP):
            return ReconstructionRole.CONTEXT_LOCK, "point de vue non arbitré ou déjà couvert"
        # L'identité ne dit rien de la structure. Une vue peut montrer sans
        # conteste le bon bâtiment et n'apporter aucune façade exploitable :
        # promouvoir sur la seule reconnaissance confondait « c'est bien lui »
        # et « on peut le reconstruire avec ça ».
        if not asset.carries_geometry:
            return (
                ReconstructionRole.CONTEXT_LOCK,
                "aptitude géométrique non évaluée"
                if asset.geometry_suitability is GeometrySuitability.UNASSESSED
                else "structure insuffisante pour la géométrie",
            )
        # La géométrie d'un volume change peu : une vue non datée reste
        # exploitable pour la structure. L'apparence, non — mais c'est un
        # usage distinct, exprimé par la politique et non par le rôle seul.
        if asset.temporal_status is TemporalStatus.BEFORE_EVENT:
            return ReconstructionRole.TEXTURE_REFERENCE, "antérieur aux travaux déclarés"
        if (
            asset.temporal_status is TemporalStatus.UNKNOWN
            and not policy.temporal.allow_unknown_for_geometry
        ):
            return ReconstructionRole.CONTEXT_LOCK, "datation inconnue, exigée par la politique"
        return ReconstructionRole.PHOTO_GEOMETRY, "cible visible, située et arbitrée"

    if Subject.SIGN in asset.subjects or (asset.sign_text or "").strip():
        return ReconstructionRole.IDENTITY_EVIDENCE, "enseigne lisible, sans position"

    if Subject.ENTRANCE in asset.subjects or contains:
        return ReconstructionRole.TEXTURE_REFERENCE, "apparence exploitable, sans position"

    return ReconstructionRole.REFERENCE_ONLY, "ni géométrie ni apparence exploitable"


def assign(assets: list[Asset], policy: PipelinePolicy = DEFAULT_POLICY) -> RoleReport:
    """Affecte un rôle à chaque asset, en place."""
    report = RoleReport()

    for index, asset in enumerate(assets):
        role, reason = role_for(asset, policy)
        assets[index] = asset.model_copy(update={"reconstruction_role": role})
        report.counts[role.value] = report.counts.get(role.value, 0) + 1
        report.reasons[reason] = report.reasons.get(reason, 0) + 1

    log.info("rôles affectés : %s", report.counts)
    return report
