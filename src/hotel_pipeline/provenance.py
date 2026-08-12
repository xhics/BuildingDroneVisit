"""Provenance des rapports (Lot 1B, généricité).

Un rapport sans provenance n'est pas reproductible : on ne sait ni avec quels
seuils il a été produit, ni sur quel profil d'établissement, ni si l'un ou
l'autre a changé depuis.

Chaque rapport porte donc la version de la politique, l'identifiant de sa
calibration, l'identifiant du profil, et une empreinte des deux. L'empreinte
détecte ce qu'une version ne dit pas : une modification locale non versionnée.
"""

from __future__ import annotations

import hashlib

from .schemas import PipelinePolicy, PropertyProfile


def _digest(payload: str) -> str:
    """Empreinte courte, suffisante pour détecter une divergence."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def policy_digest(policy: PipelinePolicy) -> str:
    return _digest(policy.model_dump_json())


def profile_digest(profile: PropertyProfile) -> str:
    return _digest(profile.model_dump_json())


def provenance(
    policy: PipelinePolicy, profile: PropertyProfile | None = None
) -> dict[str, str]:
    """Bloc de provenance à insérer dans tout rapport."""
    block = {
        "policy_version": policy.version,
        "calibration_id": policy.model.calibration_id,
        "calibrated_on_sites": str(policy.model.calibrated_on_sites),
        "policy_digest": policy_digest(policy),
    }
    if profile is not None:
        block["property_profile_id"] = profile.property_id
        block["property_profile_digest"] = profile_digest(profile)
    return block


def stamp(report: dict, policy: PipelinePolicy, profile: PropertyProfile | None = None) -> dict:
    """Appose la provenance sur un rapport déjà sérialisé."""
    return {**report, "provenance": provenance(policy, profile)}
