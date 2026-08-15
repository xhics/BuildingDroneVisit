"""Facettes de politique : ce dont une production dépend réellement.

`policy_digest` est l'empreinte **complète**. Elle sert la provenance — dire
avec quels réglages un rapport a été produit — et elle bouge dès qu'un seuil
change, où qu'il soit. La prendre pour une dépendance périmerait le nuage LiDAR
parce qu'une ouverture sectorielle a bougé.

D'où deux niveaux, à ne pas confondre :

```text
policy_digest              empreinte complète, pour la provenance
policy_dependency_digests  empreintes des seules facettes consommées
```

Les facettes se déclarent **par champ**, non par section. `geometry` porte à la
fois les seuils d'adjacence — qui décident quel bâtiment est la cible — et
l'ouverture sectorielle — qui décide quel candidat sert quel besoin. Les
regrouper ferait périmer une résolution de bâtiment parce qu'un cadrage a
changé.

Les dépendances transitives ne sont pas répétées : modifier la collecte périme
le manifeste de candidats, dont l'empreinte périme à son tour le plan qui le
cite. Chaque production ne déclare donc que ce qu'elle **lit** elle-même.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum

from .logging import get_logger

log = get_logger("policy-facets")


class Facet(StrEnum):
    """Un groupe de réglages consommé par les mêmes productions."""

    COLLECTION_DISCOVERY = "collection_discovery"
    CANDIDATE_GEOMETRY = "candidate_geometry"
    COVERAGE_TARGETS = "coverage_targets"
    VISIBILITY = "visibility"
    BUILDING_RESOLUTION = "building_resolution"
    DEDUPLICATION = "deduplication"
    CLASSIFICATION_MODEL = "classification_model"
    TEMPORAL = "temporal"
    TERRAIN_DERIVATION = "terrain_derivation"
    GEOSPATIAL_QUALIFICATION = "geospatial_qualification"


#: Champs de chaque facette, en chemins pointés. Une valeur qui n'y figure pas
#: ne périme rien : c'est un choix, vérifié par un test qui refuse qu'un champ
#: de politique reste orphelin sans être explicitement écarté.
FACET_FIELDS: dict[Facet, tuple[str, ...]] = {
    Facet.COLLECTION_DISCOVERY: (
        "collection.radius_m",
        "collection.road_radius_m",
        "collection.sample_spacing_m",
        "collection.snap_radius_m",
        "collection.max_panorama_distance_m",
        "collection.image_fov_deg",
        "collection.wide_fov_deg",
    ),
    Facet.CANDIDATE_GEOMETRY: (
        "geometry.half_fov_deg",
        "geometry.max_distance_m",
        "geometry.sector_observer_half_width_deg",
        "geometry.viewpoint_separation_m",
    ),
    # Ce qu'une obligation exige. Séparée de `candidate_geometry` : l'une dit
    # combien de vues il faut, l'autre comment on juge une vue. Modifier la
    # première périme les besoins ; modifier la seconde périme les évaluations.
    Facet.COVERAGE_TARGETS: (
        "coverage.building_viewpoints_required",
        "coverage.context_viewpoints_required",
        "coverage.building_continuity_required",
        "coverage.context_continuity_required",
        "coverage.building_min_projected_width",
        "coverage.context_min_projected_width",
        "coverage.building_min_visible_fraction",
        "coverage.context_min_visible_fraction",
    ),
    Facet.VISIBILITY: (
        "visibility.max_angular_step_deg",
        "visibility.min_angular_cells",
        "visibility.corridor_sample_step_m",
        "visibility.intersection_tolerance_m",
        "visibility.output_precision",
        "visibility.sampling_method",
        "visibility.projection_model",
    ),
    # L'adjacence décide **quel bâtiment** est la cible : elle appartient à la
    # résolution, non au cadrage, bien qu'elle vive dans la même section.
    Facet.BUILDING_RESOLUTION: (
        "geometry.adjacency_strong_m",
        "geometry.adjacency_max_m",
    ),
    Facet.DEDUPLICATION: (
        "dedup.phash_hamming_threshold",
        "dedup.position_tolerance_m",
        "dedup.bearing_tolerance_deg",
        "dedup.max_overlap_per_cluster",
    ),
    Facet.CLASSIFICATION_MODEL: (
        "model.model_name",
        "model.pretrained",
        "model.subject_accept",
        "model.subject_reject",
        "model.review_confidence_floor",
    ),
    Facet.TEMPORAL: (
        "temporal.allow_unknown_for_geometry",
        "temporal.allow_unknown_for_appearance",
        "temporal.require_current_for_sensitive_zones",
        "temporal.sensitive_scopes",
    ),
    Facet.TERRAIN_DERIVATION: (
        "terrain.cell_m",
        "terrain.ring_m",
        "terrain.search_radius_m",
        "terrain.min_truth_coverage",
        "terrain.min_ring_coverage",
        "terrain.min_reconstructed",
        "terrain.max_building_points_per_m2",
        "terrain.min_trials",
    ),
    Facet.GEOSPATIAL_QUALIFICATION: (
        "qualification.status",
        "qualification.intended_use",
        "qualification.terrain",
        "qualification.roofline",
    ),
}

#: Champs délibérément hors facette. Les nommer rend leur exclusion vérifiable :
#: un champ ajouté à la politique et oublié se signalerait, au lieu de ne rien
#: périmer en silence.
UNSCOPED_FIELDS: frozenset[str] = frozenset(
    {
        # La version identifie la politique, elle n'en règle rien.
        "version",
        # Les calibrations décrivent d'où viennent les seuils ; changer un
        # identifiant de campagne ne change aucune valeur, donc ne périme rien.
        "model.calibration_id", "model.calibrated_on_sites",
        "terrain.calibration_id", "terrain.calibrated_on_sites",
        "qualification.calibration_id", "qualification.calibrated_on_sites",
    }
)


#: Ce que chaque production **lit** de la politique. Les dépendances
#: transitives n'y figurent pas : elles passent par les empreintes d'entrée.
CONSUMERS: dict[str, tuple[Facet, ...]] = {
    "CaptureDemandManifest": (Facet.COVERAGE_TARGETS,),
    "DemandAssessmentManifest": (Facet.COVERAGE_TARGETS, Facet.CANDIDATE_GEOMETRY),
    "CandidateManifest": (Facet.COLLECTION_DISCOVERY,),
    "CandidateEvaluation": (Facet.CANDIDATE_GEOMETRY, Facet.VISIBILITY),
    "AcquisitionPlan": (Facet.CANDIDATE_GEOMETRY, Facet.VISIBILITY),
    "CaptureGeometryManifest": (Facet.BUILDING_RESOLUTION,),
    "VisibilityRun": (Facet.VISIBILITY,),
    "DuplicateReport": (Facet.DEDUPLICATION,),
    "ClassificationReport": (Facet.CLASSIFICATION_MODEL,),
    "TemporalReport": (Facet.TEMPORAL,),
    "DerivedRaster": (Facet.TERRAIN_DERIVATION,),
    "QualificationReport": (Facet.GEOSPATIAL_QUALIFICATION,),
    # Un fichier téléchargé est identifié par son empreinte : aucun seuil de
    # politique n'intervient dans ce qu'il est.
    "AcquiredLaz": (),
    "AcquiredImage": (),
}


def _value_at(policy, path: str):  # noqa: ANN001, ANN201
    current = policy
    for part in path.split("."):
        current = getattr(current, part)
    return current


def facet_digest(policy, facet: Facet) -> str:  # noqa: ANN001
    """Empreinte des seuls champs d'une facette.

    Sérialisés par chemin trié : ajouter un champ à la facette change
    l'empreinte, ce qui est voulu — la production dépend désormais de lui.
    """
    payload = {
        path: _serialisable(_value_at(policy, path))
        for path in sorted(FACET_FIELDS[facet])
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _serialisable(value):  # noqa: ANN001, ANN201
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def dependency_digests(policy, production: str) -> dict[str, str]:  # noqa: ANN001
    """Empreintes des facettes que cette production lit réellement.

    Une production inconnue est une erreur, non un dictionnaire vide : le
    silence ferait d'un oubli une absence de dépendance, et rien ne
    périmerait plus.
    """
    if production not in CONSUMERS:
        raise KeyError(
            f"production {production!r} sans dépendances déclarées — "
            f"ajoutez-la à CONSUMERS. Connues : {sorted(CONSUMERS)}"
        )
    return {
        facet.value: facet_digest(policy, facet)
        for facet in CONSUMERS[production]
    }


def stale_facets(
    recorded: dict[str, str], policy, production: str  # noqa: ANN001
) -> list[str]:
    """Facettes ayant bougé depuis qu'une production les a citées.

    Une facette **absente** de ce qui a été enregistré n'est pas une facette
    inchangée : la production a été écrite avant que la dépendance existe, et
    on ne peut pas affirmer qu'elle vaut encore. Elle est donc signalée.
    """
    current = dependency_digests(policy, production)
    problems = []
    for name, digest in current.items():
        seen = recorded.get(name)
        if seen is None:
            problems.append(
                f"{name} : dépendance absente de la production — écrite avant "
                "que cette facette soit déclarée, sa validité n'est pas établie"
            )
        elif seen != digest:
            problems.append(f"{name} : {seen[:12]}… ≠ {digest[:12]}… courant")
    return problems


def describe(policy) -> dict:  # noqa: ANN001
    """Les deux niveaux, côte à côte, pour un rapport."""
    from .provenance import policy_digest

    return {
        "policy_digest": policy_digest(policy),
        "facets": {
            facet.value: facet_digest(policy, facet) for facet in Facet
        },
        "note": (
            "l'empreinte complète sert la provenance ; les empreintes de "
            "facette servent la péremption. Un rapport cite les deux, et une "
            "facette qu'il ne lit pas ne le périme jamais."
        ),
    }
