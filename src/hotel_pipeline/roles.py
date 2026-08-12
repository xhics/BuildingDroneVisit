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
from .schemas import Asset, ReconstructionRole, Subject

log = get_logger("roles")


@dataclass
class RoleReport:
    counts: dict[str, int] = field(default_factory=dict)
    reasons: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"roles": self.counts, "reasons": self.reasons}


def role_for(asset: Asset) -> tuple[ReconstructionRole, str]:
    """Rôle d'un asset et sa justification.

    L'ordre compte : on écarte d'abord ce qui nuit, on retient ensuite ce qui
    porte la géométrie, et on affecte enfin ce qui ne porte qu'une apparence.
    """
    if Subject.INTERIOR in asset.subjects:
        return ReconstructionRole.REFERENCE_ONLY, "intérieur, hors périmètre extérieur"

    if asset.property_match_status.value == "mismatch":
        return ReconstructionRole.REJECT, "enseigne d'un autre établissement"

    positioned = asset.camera_lat is not None and asset.camera_lon is not None
    shows_building = Subject.BUILDING in asset.subjects

    # Seule une image située peut être triangulée. Sans position ni cap, une
    # photographie ne se rattache à aucun point de vue.
    if positioned and shows_building:
        return ReconstructionRole.PHOTO_GEOMETRY, "position connue et bâtiment visible"

    if positioned and not shows_building:
        return ReconstructionRole.CONTEXT_LOCK, "position connue, environnement seulement"

    if Subject.SIGN in asset.subjects or (asset.sign_text or "").strip():
        return ReconstructionRole.IDENTITY_EVIDENCE, "enseigne lisible, sans position"

    if Subject.ENTRANCE in asset.subjects or shows_building:
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
