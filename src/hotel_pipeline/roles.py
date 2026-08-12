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
from .schemas import (
    Asset,
    ClusterRole,
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


def role_for(asset: Asset) -> tuple[ReconstructionRole, str]:
    """Rôle d'un asset et sa justification.

    Porter de la géométrie exige **toutes** les conditions suivantes :

    ```text
    caméra située
    + bâtiment cible réellement visible
    + aucune occultation non arbitrée
    + revue acceptée
    + point de vue actif
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
        if asset.occluded_by:
            return ReconstructionRole.CONTEXT_LOCK, "ligne de visée masquée par un voisin"
        if asset.target_building_visible is not True:
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
        if asset.temporal_status is TemporalStatus.BEFORE_EVENT:
            return ReconstructionRole.TEXTURE_REFERENCE, "antérieur à la rénovation"
        return ReconstructionRole.PHOTO_GEOMETRY, "cible visible, située et arbitrée"

    if Subject.SIGN in asset.subjects or (asset.sign_text or "").strip():
        return ReconstructionRole.IDENTITY_EVIDENCE, "enseigne lisible, sans position"

    if Subject.ENTRANCE in asset.subjects or contains:
        return ReconstructionRole.TEXTURE_REFERENCE, "apparence exploitable, sans position"

    return ReconstructionRole.REFERENCE_ONLY, "ni géométrie ni apparence exploitable"


def assign(assets: list[Asset]) -> RoleReport:
    """Affecte un rôle à chaque asset, en place."""
    report = RoleReport()

    for index, asset in enumerate(assets):
        role, reason = role_for(asset)
        assets[index] = asset.model_copy(update={"reconstruction_role": role})
        report.counts[role.value] = report.counts.get(role.value, 0) + 1
        report.reasons[reason] = report.reasons.get(reason, 0) + 1

    log.info("rôles affectés : %s", report.counts)
    return report
