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


def digest_of(payload: object) -> str:
    """Empreinte d'un document déjà lu, stable à l'ordre des clés près.

    Deux écritures d'un même contenu doivent rendre la même empreinte : sans
    tri, un manifeste relu puis réécrit paraîtrait avoir changé.
    """
    import json

    return _digest(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    )


def provenance(
    policy: PipelinePolicy, profile: PropertyProfile | None = None
) -> dict[str, str]:
    """Bloc de provenance à insérer dans tout rapport."""
    block = {
        "policy_version": policy.version,
        # Deux calibrations distinctes : celle du classifieur photographique et
        # celle des seuils géospatiaux. Les confondre laisserait croire que les
        # secondes reposent sur les 36 images du jeu de validation.
        "model_calibration_id": policy.model.calibration_id,
        "model_calibrated_on_sites": str(policy.model.calibrated_on_sites),
        "terrain_calibration_id": policy.terrain.calibration_id,
        "terrain_calibrated_on_sites": str(policy.terrain.calibrated_on_sites),
        "qualification_status": policy.qualification.status,
        "qualification_intended_use": policy.qualification.intended_use,
        "qualification_calibration_id": policy.qualification.calibration_id,
        "policy_digest": policy_digest(policy),
    }
    if profile is not None:
        block["property_profile_id"] = profile.property_id
        block["property_profile_digest"] = profile_digest(profile)
    return block


def stamp(report: dict, policy: PipelinePolicy, profile: PropertyProfile | None = None) -> dict:
    """Appose la provenance sur un rapport déjà sérialisé."""
    return {**report, "provenance": provenance(policy, profile)}
