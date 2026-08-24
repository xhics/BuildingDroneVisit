"""Livrables canoniques de couverture et de contexte du Lot 1B.

Le Router choisit une route. Ce module publie ce que cette route permet
réellement de montrer, les zones qui restent des proxies et les faits du site
qui demeurent indéterminés. Il ne mute jamais :class:`SiteManifest`.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .geo.catalog import CoverageBasis, CoverageState, GeoSource, route
from .provenance import digest_of
from .schemas.enums import ObjectState, ReconstructionRole, Rights
from .schemas.geometry import GeometryResolutionStatus, GeometryRole


LOT1B_COVERAGE_CONTRACT_VERSION = 2


class AcquisitionState(StrEnum):
    AVAILABLE_NOT_ACQUIRED = "available_not_acquired"
    ACQUIRED = "acquired"
    MANUAL_ACQUISITION_REQUIRED = "manual_acquisition_required"
    NOT_APPLICABLE = "not_applicable"


class ConfidenceLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class Lot1BStatus(StrEnum):
    READY_FOR_LOT_2 = "ready_for_lot_2"
    INCOMPLETE_CAPTURE_REQUIRED = "incomplete_capture_required"
    BLOCKED_PREREQUISITES = "blocked_prerequisites"


class ConstraintSeverity(StrEnum):
    HARD = "hard"
    REQUIRED = "required"
    ADVISORY = "advisory"


class SourceCoverage(BaseModel):
    """Couverture et acquisition sont deux axes indépendants."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    dataset: str
    coverage_state: CoverageState
    acquisition_state: AcquisitionState
    coverage_basis: CoverageBasis
    vintage: str | None = None
    resolution_m: float | None = None
    evidence: list[str] = Field(min_length=1)
    establishes: list[str] = Field(default_factory=list)
    cannot_establish: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _manual_is_not_covered_by_fiction(self) -> "SourceCoverage":
        if self.acquisition_state is AcquisitionState.MANUAL_ACQUISITION_REQUIRED:
            if self.coverage_state is not CoverageState.MANUAL_ACQUISITION_REQUIRED:
                raise ValueError("une acquisition manuelle exige l'état de couverture homonyme")
        return self


class ObjectRecheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    state_before: ObjectState
    state_after_review: ObjectState
    geometry_available: bool
    evidence_checked: list[str] = Field(min_length=1)
    finding: str = Field(min_length=1)
    next_action: str = Field(min_length=1)

    @model_validator(mode="after")
    def _review_does_not_promote_without_new_evidence(self) -> "ObjectRecheck":
        if (
            self.state_before is ObjectState.UNRESOLVED
            and self.state_after_review is not ObjectState.UNRESOLVED
        ):
            raise ValueError("une relecture sans nouvelle preuve ne promeut pas un objet")
        return self


class ContextManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: int = LOT1B_COVERAGE_CONTRACT_VERSION
    hotel_id: str
    generated_at: str
    execution_mode: str = "local_only"
    network_requests: int = Field(default=0, ge=0)
    provenance: dict[str, str]
    input_digests: dict[str, str]
    territories: list[str]
    working_crs: str
    context_anchor_counts: dict[str, int]
    source_coverage: list[SourceCoverage]
    object_rechecks: list[ObjectRecheck]
    preservation_rules: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _closed_inputs_and_unique_sources(self) -> "ContextManifest":
        required = {
            "site_manifest", "asset_manifest", "spatial_manifest",
            "capture_geometry", "router_decision", "source_registry",
        }
        if set(self.input_digests) != required:
            raise ValueError(
                f"empreintes de contexte incomplètes : attendu {sorted(required)}"
            )
        if any(not value.strip() for value in self.input_digests.values()):
            raise ValueError("une empreinte de contexte est vide")
        ids = [source.source_id for source in self.source_coverage]
        if len(ids) != len(set(ids)):
            raise ValueError("source de contexte dupliquée")
        kinds = [row.kind for row in self.object_rechecks]
        if len(kinds) != len(set(kinds)):
            raise ValueError("objet réexaminé plusieurs fois")
        return self


class CameraConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    constraint_id: str
    zone_ref: str
    rule: str
    severity: ConstraintSeverity
    rationale: str
    evidence_refs: list[str] = Field(min_length=1)
    min_distance_m: float | None = None
    allowed_angles_deg: str | None = None
    detail_level: str | None = None
    proof_required: str | None = None


class CameraConstraintsManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: int = LOT1B_COVERAGE_CONTRACT_VERSION
    hotel_id: str
    generated_at: str
    router_decision_digest: str
    provenance: dict[str, str]
    constraints: list[CameraConstraint] = Field(min_length=1)

    @model_validator(mode="after")
    def _constraint_ids_are_unique(self) -> "CameraConstraintsManifest":
        ids = [row.constraint_id for row in self.constraints]
        if len(ids) != len(set(ids)):
            raise ValueError("identifiant de contrainte caméra dupliqué")
        return self


class CoverageReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: int = LOT1B_COVERAGE_CONTRACT_VERSION
    hotel_id: str
    generated_at: str
    execution_mode: str = "local_only"
    network_requests: int = Field(default=0, ge=0)
    provenance: dict[str, str]
    input_digest: str
    router: dict
    demands: dict
    assets: dict
    visibility: dict
    geometry: dict
    rights: dict
    sources: dict
    unresolved_objects: list[str]
    lot_1b_status: Lot1BStatus
    blocking_reasons: list[str]
    limitations: list[str]
    outputs: dict[str, str]


class DedupRobustnessEvidence(BaseModel):
    """Preuve séparée : un rapport courant n'est pas une validation robuste."""

    model_config = ConfigDict(extra="forbid")

    contract_version: int = 2
    #: Empreinte complète au moment de l'exécution, pour la provenance. Elle
    #: n'est pas la dépendance : des champs aval (visibilité, rôles) peuvent
    #: légitimement changer sans altérer une seule comparaison robuste.
    asset_manifest_sha256: str = Field(min_length=64, max_length=64)
    robust_input_digest: str = Field(min_length=1)
    algorithm: str = Field(min_length=1)
    policy_digest: str = Field(min_length=1)
    dedup_policy_digest: str = Field(min_length=1)
    production_used: bool
    candidate_pairs: int = Field(ge=0)
    matched_pairs: int = Field(ge=0)
    crop_regression_passed: bool
    watermark_regression_passed: bool
    distinct_regression_passed: bool

    @model_validator(mode="after")
    def _must_prove_the_production_algorithm(self) -> "DedupRobustnessEvidence":
        if not (
            self.production_used
            and self.crop_regression_passed
            and self.watermark_regression_passed
            and self.distinct_regression_passed
        ):
            raise ValueError(
                "la preuve robuste exige l'algorithme de production et les deux régressions"
            )
        return self


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _router_decision(workspace) -> tuple[Path, dict]:  # noqa: ANN001
    folder = workspace.path("10_validation")
    invalidated: set[str] = set()
    registry = folder / "router_invalidations.json"
    if registry.is_file():
        invalidated = {
            row["decision_file"]
            for row in json.loads(registry.read_text("utf-8")).get("invalidations", [])
        }
    candidates = sorted(
        (p for p in folder.glob("router_decision_*.json") if p.name not in invalidated),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError("aucune décision Router courante")
    path = candidates[-1]
    return path, json.loads(path.read_text("utf-8"))


def _source_result(source: GeoSource, lidar: dict | None) -> SourceCoverage:
    if source.source_id == "lidar-quebec" and lidar:
        state = CoverageState(lidar.get("coverage", "unknown"))
        acquired = bool((lidar.get("tiles") or []))
        return SourceCoverage(
            source_id=source.source_id,
            dataset=source.dataset,
            coverage_state=state,
            acquisition_state=(
                AcquisitionState.ACQUIRED if acquired
                else AcquisitionState.AVAILABLE_NOT_ACQUIRED
            ),
            coverage_basis=source.coverage_basis,
            vintage=(lidar.get("tiles") or [{}])[0].get("acquired_on"),
            resolution_m=source.resolution_m,
            evidence=["06_geo/lidar_discovery.json : intersection réelle de la tuile"],
            establishes=list(source.establishes),
            cannot_establish=list(source.cannot_establish),
            limitations=["altimétrie et volume seulement ; aucune apparence"],
        )
    if not source.acquisition_automated:
        return SourceCoverage(
            source_id=source.source_id,
            dataset=source.dataset,
            coverage_state=CoverageState.MANUAL_ACQUISITION_REQUIRED,
            acquisition_state=AcquisitionState.MANUAL_ACQUISITION_REQUIRED,
            coverage_basis=source.coverage_basis,
            vintage=source.vintage,
            resolution_m=source.resolution_m,
            evidence=[source.index_url or source.url],
            establishes=list(source.establishes),
            cannot_establish=list(source.cannot_establish),
            limitations=[
                "consultation ou extraction manuelle requise ; aucune géométrie cadastrale acquise",
                "la représentation cadastrale ne vaut pas arpentage ni titre de propriété",
            ],
        )
    if source.coverage_basis is CoverageBasis.PUBLISHER_DECLARED_TERRITORY:
        return SourceCoverage(
            source_id=source.source_id,
            dataset=source.dataset,
            coverage_state=CoverageState.COVERED,
            acquisition_state=AcquisitionState.AVAILABLE_NOT_ACQUIRED,
            coverage_basis=source.coverage_basis,
            vintage=source.vintage,
            resolution_m=source.resolution_m,
            evidence=[source.index_url or source.url],
            establishes=list(source.establishes),
            cannot_establish=list(source.cannot_establish),
            limitations=[
                "couverture déclarée pour le territoire, pas intersection d'une tuile",
                "résolution de 5 m : verrou de contexte, pas découpage de toiture ou parcelle",
            ],
        )
    return SourceCoverage(
        source_id=source.source_id,
        dataset=source.dataset,
        coverage_state=CoverageState.UNKNOWN,
        acquisition_state=AcquisitionState.AVAILABLE_NOT_ACQUIRED,
        coverage_basis=source.coverage_basis,
        vintage=source.vintage,
        resolution_m=source.resolution_m,
        evidence=[source.index_url or source.url],
        establishes=list(source.establishes),
        cannot_establish=list(source.cannot_establish),
        limitations=["index non interrogé : inconnu n'est pas non couvert"],
    )


_RECHECK_ACTIONS = {
    "PARKING_HOTEL": "obtenir une preuve visuelle reliée à une géométrie candidate distincte du 1205",
    "ENTRANCE_MAIN_CURRENT": "obtenir une vue datée et localisée après achèvement confirmé des travaux",
    "PROPERTY_PARCEL": "commander ou exporter l'extrait cadastral officiel et le vérifier",
    "DRIVEWAY_MAIN": "établir son existence puis sa géométrie depuis une preuve de terrain ou officielle",
    "PARK_AND_RIDE": "localiser le terminus et vérifier sa relation distinct_from avec la propriété",
}


def _rechecks(site) -> list[ObjectRecheck]:  # noqa: ANN001
    rows: list[ObjectRecheck] = []
    by_kind = {obj.kind: obj for obj in site.objects}
    for kind, action in _RECHECK_ACTIONS.items():
        obj = by_kind[kind]
        evidence = list(obj.evidence) or [obj.unresolved_reason or "aucune preuve positive"]
        rows.append(ObjectRecheck(
            kind=kind,
            state_before=obj.state,
            state_after_review=obj.state,
            geometry_available=bool(obj.geometry_wkt),
            evidence_checked=evidence,
            finding=obj.unresolved_reason or "preuve insuffisante pour changer l'état",
            next_action=action,
        ))
    return rows


def _positionless_kinds(router: dict, geometry) -> list[str]:  # noqa: ANN001
    """Objets sans position utilisable, sans condamner les proxies mesurés."""
    known_without_site_wkt = set(
        router["site"]["by_standing"].get("known_not_targetable", [])
    )
    unresolved = set(router["site"]["by_standing"].get("unresolved", []))
    proxy_objects = {
        kind
        for zone in router.get("geometric_proxies", [])
        if zone.get("qualified")
        for kind in zone.get("covered_objects", [])
    }
    resolved_features = {
        item.feature_id
        for item in geometry.geometries
        if item.resolution_status is GeometryResolutionStatus.RESOLVED
    }
    return sorted((unresolved | known_without_site_wkt) - proxy_objects - resolved_features)


def _no_claim_kinds(router: dict, geometry, site) -> list[str]:  # noqa: ANN001
    """Objets dont **rien** ne peut être affirmé : l'existence même manque.

    Distinct de `_blind_field_kinds` : un objet établi par la photographie mais
    dépourvu de contour existe, et l'interdire d'affirmation reviendrait à taire
    une preuve. Seul un objet non résolu tombe ici.
    """
    established = {
        obj.kind for obj in site.objects
        if obj.state in (ObjectState.CONFIRMED, ObjectState.INFERRED)
    }
    return sorted(set(_positionless_kinds(router, geometry)) - established)


def _blind_field_kinds(router: dict, geometry, site) -> list[str]:  # noqa: ANN001
    """Objets réels que **rien n'a photographiés** : les champs visuels morts.

    Leur existence est établie, leur apparence ne l'est pas. Une caméra qui les
    cadre montrerait une forme sans texture observée — ce qui se lit comme une
    reconstruction alors que ce n'en est pas une. Ils se contournent, ils ne se
    nient pas.
    """
    established = {
        obj.kind for obj in site.objects
        if obj.state in (ObjectState.CONFIRMED, ObjectState.INFERRED)
    }
    return sorted(set(_positionless_kinds(router, geometry)) & established)


_FACADE_BY_SECTOR = {
    "front": "FACADE_PRIMARY",
    "front_left_corner": "FACADE_PRIMARY",
    "left": "FACADE_LEFT",
    "rear_left_corner": "FACADE_REAR",
    "rear": "FACADE_REAR",
    "rear_right_corner": "FACADE_REAR",
    "right": "FACADE_RIGHT",
    "front_right_corner": "FACADE_PRIMARY",
}

_FACADE_VIEWPOINT_THRESHOLDS = {
    "FACADE_PRIMARY": 8,
    "FACADE_LEFT": 5,
    "FACADE_RIGHT": 5,
    "FACADE_REAR": 3,
}


def _per_facade_viewpoint_counts(assets) -> dict[str, int]:  # noqa: ANN001
    """Compte les viewpoint_cluster uniques par façade.

    L'unité est le cluster, non le fichier : un panorama comptant pour
    plusieurs besoins reste un seul point de vue indépendant.
    """
    counts: dict[str, set[str]] = {kind: set() for kind in _FACADE_BY_SECTOR.values()}
    for asset in assets.assets:
        if asset.view_sector is None or asset.view_sector.value == "unknown":
            continue
        facade = _FACADE_BY_SECTOR.get(asset.view_sector.value)
        if facade is None:
            continue
        cluster = asset.viewpoint_cluster or asset.id
        counts[facade].add(cluster)
    return {kind: len(clusters) for kind, clusters in counts.items()}


def _zone_state_for_facade(
    kind: str,
    appearance_coverage: str,
    viewpoint_count: int,
    has_synthetic: bool,
) -> str:
    """Déduit l'état de zone trusted / proxy / unobserved.

    - ``trusted`` : couverture photographique suffisante et viewpoints >= seuil.
    - ``proxy`` : apparence partielle ou complétée synthétiquement.
    - ``unobserved`` : aucune apparence observée ni synthétique.
    """
    threshold = _FACADE_VIEWPOINT_THRESHOLDS.get(kind, 3)
    if appearance_coverage == "full" and viewpoint_count >= threshold:
        return "trusted"
    if appearance_coverage == "partial" or has_synthetic:
        return "proxy"
    if appearance_coverage == "none":
        return "unobserved"
    return "proxy"


def _completion_findings(
    *,
    uncovered_capture_demands: list[str],
    unresolved_objects: list[str],
    source_coverage: list[SourceCoverage],
    asset_count: int,
    duplicate_report_files: int | None,
    dedup_robustness_current: bool,
    source_registry_complete: bool,
    uncleared_rights: int,
) -> tuple[list[str], list[str]]:
    """Sépare ce qui empêche DONE de ce qui borne seulement l'usage.

    Les droits assumés par l'opérateur restent visibles, mais ne redeviennent
    pas subrepticement un Gate. À l'inverse, l'orthophoto et la déduplication
    font partie de la définition normative du Lot 1B et doivent apparaître
    dans les blocages même si le Router peut déjà choisir Path D.
    """
    blockers = [
        f"{demand_id} reste sans preuve photographique ni proxy qualifié"
        for demand_id in uncovered_capture_demands
    ]
    unresolved = set(unresolved_objects)
    if "PROPERTY_PARCEL" in unresolved:
        blockers.append("la parcelle cadastrale n'est pas acquise")
    critical = sorted(unresolved & {"ENTRANCE_MAIN_CURRENT", "PARKING_HOTEL"})
    if critical:
        blockers.append(
            "objet(s) critique(s) non établi(s) : " + ", ".join(critical)
        )

    cmm = next((row for row in source_coverage if row.source_id == "cmm-ortho"), None)
    if cmm is None or cmm.acquisition_state is not AcquisitionState.ACQUIRED:
        blockers.append("l'orthophoto couvrant la propriété n'est pas acquise et qualifiée")
    if duplicate_report_files != asset_count:
        blockers.append(
            "la déduplication ne couvre pas le manifeste courant : "
            f"{duplicate_report_files!r}/{asset_count} assets"
        )
    elif not dedup_robustness_current:
        blockers.append(
            "la déduplication courante n'a pas de preuve robuste aux recadrages et filigranes"
        )
    if not source_registry_complete:
        blockers.append(
            "le registre canonique ne clôt pas les familles photographiques prioritaires"
        )

    limitations = [
        f"{uncleared_rights} assets restent public_uncleared ou unknown ; "
        "ils ne deviennent ni textures ni preuves de production"
    ]
    return blockers, limitations


def _as_wgs84(shape, working_crs: str):  # noqa: ANN001, ANN201
    """Normalise une forme pour GeoJSON, sans étiqueter des mètres en degrés."""
    minx, miny, maxx, maxy = shape.bounds
    if -180 <= minx <= maxx <= 180 and -90 <= miny <= maxy <= 90:
        return shape
    from pyproj import Transformer
    from shapely.ops import transform

    transformer = Transformer.from_crs(working_crs, "EPSG:4326", always_xy=True)
    return transform(transformer.transform, shape)


def _robust_dedup_is_current(workspace, policy=None) -> bool:  # noqa: ANN001
    """Vérifie les dépendances robustes, jamais le manifeste entier.

    La SHA complète reste dans la preuve comme photographie d'exécution. La
    péremption porte sur les entrées réellement consommées et, quand la
    politique est disponible, sur la facette de déduplication uniquement.
    """
    path = workspace.path("01_sources", "dedup_robustness_report.json")
    if not path.is_file():
        return False
    try:
        evidence = DedupRobustnessEvidence.model_validate_json(path.read_text("utf-8"))
    except (OSError, ValueError):
        return False
    try:
        assets = workspace.read_assets()
    except (OSError, ValueError):
        return False
    if assets is None:
        return False
    from .dedup_levels import robust_input_digest

    if evidence.robust_input_digest != robust_input_digest(assets.assets):
        return False
    if policy is not None:
        from .policy_facets import Facet, facet_digest

        if evidence.dedup_policy_digest != facet_digest(policy, Facet.DEDUPLICATION):
            return False
    return True


def _synthesize_blind_facades_from_satellite(
    by_kind, footprint_obj, measured, orthophoto_data=None, orthophoto_source_id=None,
) -> list:  # noqa: ANN001
    """Crée des complétions synthétiques pour les façades aveugles via satellite.
    
    Retourne une liste de SyntheticCompletion si des façades aveugles peuvent
    être complétées par une orthophoto disponible ; retourne [] sinon.
    
    La logique cherche une orthophoto (GéoMont, Google, etc.) et teste
    chaque façade qui a union_fraction==0 pour voir si elle est visible.
    
    L'orthophoto CMM à 5 m est réservée au contexte global ; elle ne sert
    pas à la complétion d'apparence des façades.
    """
    synthetics = []
    
    blind_facades = [
        (kind, obj)
        for kind, obj in by_kind.items()
        if kind.startswith("FACADE_") and kind in measured
        and measured[kind].get("union_fraction", 0.0) <= 0.0
    ]
    if not blind_facades:
        return synthetics
    
    if orthophoto_data is None or orthophoto_source_id is None:
        return synthetics
    
    from .geo.satellite_completion import synthesize_completion_from_orthophoto
    
    for facade_kind, facade_obj in blind_facades:
        if not facade_obj.geometry_wkt:
            continue
        
        synthetic = synthesize_completion_from_orthophoto(
            facade_kind=facade_kind,
            facade_geometry_wkt=facade_obj.geometry_wkt,
            footprint_geometry_wkt=footprint_obj.geometry_wkt,
            orthophoto_source_id=orthophoto_source_id,
            orthophoto_data=orthophoto_data,
        )
        if synthetic:
            synthetics.append(synthetic)
    
    return synthetics


def measure_facade_coverage(site, assets, geometry, policy, orthophoto_data=None, orthophoto_source_id=None) -> dict:  # noqa: ANN001
    """Couverture d'apparence réellement reçue par chaque mur.

    Remplace un littéral — « partial » pour FACADE_PRIMARY, « none » pour les
    autres — qui n'était vrai que sur le premier site mesuré. Un second
    bâtiment en aurait hérité le verdict sans qu'aucune image ne le fonde.

    Seuls les assets **porteurs de géométrie** comptent : une vue écartée en
    revue ou périmée par la déduplication ne texture rien. Les obstacles
    viennent des bâtiments voisins déjà résolus, pas d'une hypothèse.

    Les sujets sont dédupliqués par ``viewpoint_cluster`` : un panorama
    comptant pour plusieurs besoins reste un seul point de vue indépendant.
    """
    from pyproj import Transformer
    from shapely import wkt as shapely_wkt
    from shapely.ops import transform as shapely_transform

    from .geo.facade_coverage import coverage_from_subjects, sample_facade

    by_kind = {obj.kind: obj for obj in site.objects}
    footprint_obj = by_kind.get("BUILDING_MAIN")
    if footprint_obj is None or not footprint_obj.geometry_wkt:
        return {}

    working_crs = geometry.working_crs
    to_working = Transformer.from_crs("EPSG:4326", working_crs, always_xy=True)

    def project(shape):  # noqa: ANN001
        return shapely_transform(
            lambda x, y, z=None: to_working.transform(x, y), shape
        )

    footprint = project(shapely_wkt.loads(footprint_obj.geometry_wkt))

    obstacles = [
        shapely_wkt.loads(item.projected_wkt)
        for item in geometry.geometries
        if item.role is GeometryRole.OBSTACLE_BUILDING and item.projected_wkt
    ]

    fov_deg = float(policy.collection.image_fov_deg)
    
    seen_clusters: set[str] = set()
    subjects = []
    for asset in assets.assets:
        if asset.reconstruction_role is not ReconstructionRole.PHOTO_GEOMETRY:
            continue
        if asset.camera_lat is None or asset.camera_lon is None:
            continue
        cluster = asset.viewpoint_cluster or asset.id
        if cluster in seen_clusters:
            continue
        seen_clusters.add(cluster)
        subjects.append((
            cluster,
            to_working.transform(asset.camera_lon, asset.camera_lat),
            asset.heading_deg,
            fov_deg,
        ))

    measured: dict[str, dict] = {}
    for kind, obj in sorted(by_kind.items()):
        if not kind.startswith("FACADE_") or not obj.geometry_wkt:
            continue
        samples = sample_facade(shapely_wkt.loads(obj.geometry_wkt), footprint)
        coverage = coverage_from_subjects(
            kind, samples, subjects, footprint, obstacles
        )
        measured[kind] = coverage.as_dict()
    
    synthetics = _synthesize_blind_facades_from_satellite(
        by_kind, footprint_obj, measured,
        orthophoto_data=orthophoto_data,
        orthophoto_source_id=orthophoto_source_id,
    )
    if synthetics:
        from .geo.satellite_completion import merge_with_measured_coverage
        measured = merge_with_measured_coverage(measured, synthetics)
    
    return measured

def _geometry_feature(
    obj, *, confidence: ConfidenceLevel, use: str, working_crs: str,
    appearance_coverage: str = "none", geometric_support_coverage: str = "none",
    measurement: dict | None = None, zone_state: str | None = None,
    viewpoint_count: int | None = None,
) -> dict:  # noqa: ANN001
    from shapely import wkt
    from shapely.geometry import mapping

    shape = wkt.loads(obj.geometry_wkt) if obj.geometry_wkt else None
    geometry = mapping(_as_wgs84(shape, working_crs)) if shape else None

    properties = {
        "kind": obj.kind,
        "state": obj.state.value,
        "confidence": confidence.value,
        "use": use,
        "appearance_coverage": appearance_coverage,
        "geometric_support_coverage": geometric_support_coverage,
    }
    if zone_state is not None:
        properties["zone_state"] = zone_state
    if viewpoint_count is not None:
        properties["independent_viewpoints"] = viewpoint_count
    if measurement:
        properties["appearance_measurement"] = measurement
    return {
        "type": "Feature",
        "id": obj.object_id,
        "geometry": geometry,
        "properties": properties,
    }


def build(workspace) -> dict[str, Path]:  # noqa: ANN001
    """Construit et publie les livrables sans muter les vérités amont."""
    site = workspace.read_site()
    assets = workspace.read_assets()
    spatial = workspace.read_spatial()
    if site is None or assets is None or spatial is None:
        raise FileNotFoundError("site, assets et manifeste spatial sont requis")

    from .context import PipelineContext
    from .geo.geometry_loader import load_capture_geometry

    context, _warning = PipelineContext.for_workspace(workspace)
    geometry, _legacy = load_capture_geometry(
        workspace.path("06_geo/capture_geometry.json"), context.spatial_reference
    )
    router_path, router = _router_decision(workspace)
    source_registry_path = workspace.path("00_manifest", "source_registry.json")
    if not source_registry_path.is_file():
        raise FileNotFoundError(
            "registre des sources absent : exécuter `sources registry`"
        )
    from .source_registry import SourceRegistry

    source_registry = SourceRegistry.model_validate_json(
        source_registry_path.read_text("utf-8")
    )
    if source_registry.hotel_id != workspace.hotel_id:
        raise ValueError("registre des sources d'un autre établissement")
    lidar_path = workspace.path("06_geo/lidar_discovery.json")
    lidar = json.loads(lidar_path.read_text("utf-8")) if lidar_path.is_file() else None

    building = spatial.candidate(spatial.confirmed_building_id)
    if building is None:
        raise ValueError("BUILDING_MAIN confirmé absent du manifeste spatial")
    routing = route(building.centroid_lat, building.centroid_lon)
    sources = [_source_result(source, lidar) for source in routing.territorial_candidates]
    rechecks = _rechecks(site)

    input_paths = {
        "site_manifest": workspace.site_path,
        "asset_manifest": workspace.assets_path,
        "spatial_manifest": workspace.spatial_path,
        "capture_geometry": workspace.path("06_geo/capture_geometry.json"),
        "router_decision": router_path,
        "source_registry": source_registry_path,
    }
    input_digests = {name: _sha256(path) for name, path in input_paths.items()}
    now = datetime.now(timezone.utc).isoformat()

    role_counts = Counter(a.reconstruction_role.value for a in assets.assets)
    rights_counts = Counter(a.rights.value for a in assets.assets)
    usable_geometry = [
        a for a in assets.assets
        if a.reconstruction_role is ReconstructionRole.PHOTO_GEOMETRY
        and a.rights in {Rights.OWNED, Rights.LICENSED, Rights.OPEN_DATA}
    ]
    unresolved = sorted(o.kind for o in site.unresolved())
    source_payload = {row.source_id: row.model_dump(mode="json") for row in sources}
    source_payload["photographic_registry"] = {
        "file": "00_manifest/source_registry.json",
        "sha256": _sha256(source_registry_path),
        "required_families": source_registry.required_families,
        "closed_families": source_registry.closed_families,
        "closure_complete": source_registry.closure_complete,
    }
    demands = router["photographic"]
    qualified_proxies = [z for z in router["geometric_proxies"] if z["qualified"]]
    roof_proxy = next((z for z in qualified_proxies if z["zone"] == "ROOFLINE_MAIN"), {})
    uncovered_capture_demands = sorted(
        (set(demands["open"]) | set(demands["partial"]))
        - set(router.get("appearance_gaps") or [])
    )
    no_claim_kinds = _no_claim_kinds(router, geometry, site)
    per_facade_viewpoints = _per_facade_viewpoint_counts(assets)
    
    orthophoto_for_appearance = None
    orthophoto_source_id = None
    for candidate in routing.territorial_candidates:
        if candidate.source_id == "cmm-ortho":
            continue
        if candidate.resolution_m is not None and candidate.resolution_m < 5.0:
            orthophoto_for_appearance = {
                "resolution_cm": candidate.resolution_m * 100.0,
                "coverage_fraction": 1.0,
                "notes": candidate.notes or "",
            }
            orthophoto_source_id = candidate.source_id
            break
    
    facade_coverage = measure_facade_coverage(
        site, assets, geometry, context.policy,
        orthophoto_data=orthophoto_for_appearance,
        orthophoto_source_id=orthophoto_source_id,
    )
    blind_field_kinds = sorted(set(_blind_field_kinds(router, geometry, site)) | {
        kind for kind, row in facade_coverage.items()
        if row.get("appearance_coverage") == "none"
    })

    context_manifest = ContextManifest(
        hotel_id=workspace.hotel_id,
        generated_at=now,
        provenance=context.provenance,
        input_digests=input_digests,
        territories=sorted(routing.territories),
        working_crs=geometry.working_crs,
        context_anchor_counts={
            "target_buildings": len(geometry.resolved(GeometryRole.TARGET_BUILDING)),
            "road_corridors": len(geometry.corridors),
            "obstacle_buildings": len(geometry.resolved(GeometryRole.OBSTACLE_BUILDING)),
            "context_lock_assets": role_counts[ReconstructionRole.CONTEXT_LOCK.value],
        },
        source_coverage=sources,
        object_rechecks=rechecks,
        preservation_rules=[
            "conserver les bâtiments voisins, routes et accès tels que portés par la géométrie de contexte",
            "ne jamais générer une parcelle, une entrée, une allée ou un parc-o-bus non établis",
            "les assets de contexte aux droits non clarifiés guident la vérification mais ne deviennent pas textures de production",
            "l'orthophoto CMM à 5 m verrouille le contexte global seulement",
        ],
    )

    constraints = [
        CameraConstraint(
            constraint_id="proxy-close-up",
            zone_ref="BUILDING_MAIN",
            rule="avoid_close_up",
            severity=ConstraintSeverity.HARD,
            rationale="les quatre façades sont des volumes proxy sans apparence complète",
            evidence_refs=[router_path.name],
            min_distance_m=25.0,
            allowed_angles_deg="360",
            detail_level="facade",
            proof_required="acquisition et qualification d'une vue rapprochée sur chaque façade",
        ),
        CameraConstraint(
            constraint_id="roof-gaps",
            zone_ref="ROOFLINE_MAIN",
            rule="avoid_missing_roof_cells",
            severity=ConstraintSeverity.HARD,
            rationale=roof_proxy.get("note") or "la toiture qualifiée conserve des lacunes",
            evidence_refs=[roof_proxy.get("qualification_report") or "06_geo/qualification_report"],
            min_distance_m=None,
            allowed_angles_deg="45-135",
            detail_level="roof",
            proof_required="reconstruction SfM ou acquisition drone orthogonale",
        ),
    ]
    if no_claim_kinds:
        constraints.append(CameraConstraint(
            constraint_id="unresolved-claims",
            zone_ref=",".join(no_claim_kinds),
            rule="do_not_show_as_fact",
            severity=ConstraintSeverity.HARD,
            rationale="existence, association, état courant ou géométrie non établis",
            evidence_refs=["00_manifest/site_manifest.json"],
        ))
    if blind_field_kinds:
        # Champs visuels morts : l'objet existe et sa position est connue, mais
        # aucune photographie n'en donne l'apparence. Le cadrer produirait une
        # forme sans texture observée, qui se lirait comme une reconstruction.
        constraints.append(CameraConstraint(
            constraint_id="blind-visual-fields",
            zone_ref=",".join(blind_field_kinds),
            rule="avoid_framing_no_observed_appearance",
            severity=ConstraintSeverity.HARD,
            rationale=(
                "existence établie, apparence jamais observée : cadrer ces "
                "objets montrerait une forme sans texture mesurée"
            ),
            evidence_refs=[
                "00_manifest/site_manifest.json",
                "01_sources/preview_assessments.json",
            ],
        ))
    constraints.extend(
        CameraConstraint(
            constraint_id=f"capture-{demand_id.split(':', 1)[-1].lower()}",
            zone_ref=demand_id.split(":", 1)[-1],
            rule="preview_then_human_review",
            severity=ConstraintSeverity.REQUIRED,
            rationale="besoin ciblable sans photo ni proxy qualifié",
            evidence_refs=[router_path.name, "06_geo/capture_geometry.json"],
        )
        for demand_id in uncovered_capture_demands
    )
    camera_manifest = CameraConstraintsManifest(
        hotel_id=workspace.hotel_id,
        generated_at=now,
        router_decision_digest=_sha256(router_path),
        provenance=context.provenance,
        constraints=constraints,
    )

    features = []
    by_kind = {o.kind: o for o in site.objects}
    
    # Couverture du bâtiment = union des mesures de façades, pas un littéral.
    # Sans façades mesurées, on recule à "partial" (cas d'absence de données).
    # Couverture du bâtiment : moyenne des façades **pondérée par leur
    # longueur**, non le maximum. Le maximum laissait une seule façade
    # complète marquer le bâtiment entier comme couvert — un motel photographié
    # de face passait ainsi pour intégralement observé, ses trois autres murs
    # n'ayant jamais été vus.
    #
    # Le poids vient du nombre de points échantillonnés, proportionnel à la
    # longueur du mur : un pignon de six mètres ne pèse pas autant qu'une
    # façade de trente.
    kinds = ("FACADE_PRIMARY", "FACADE_LEFT", "FACADE_RIGHT", "FACADE_REAR")
    measured = [
        (
            facade_coverage.get(kind, {}).get("appearance_union_fraction", 0.0),
            facade_coverage.get(kind, {}).get("sampled", 0) or 0,
        )
        for kind in kinds
        if kind in facade_coverage
    ]
    total_weight = sum(weight for _fraction, weight in measured)
    if total_weight > 0:
        building_union = sum(
            fraction * weight for fraction, weight in measured
        ) / total_weight
    elif measured:
        # Sans poids disponible, la moyenne simple : elle reste plus honnête
        # que le maximum, qui ne décrit qu'un seul mur.
        building_union = sum(f for f, _w in measured) / len(measured)
    else:
        building_union = 0.0

    # Le mur le moins couvert est rapporté à part : une moyenne acceptable peut
    # masquer une façade entièrement aveugle, et c'est elle qui décide si un
    # plan large est réalisable.
    worst_facade = min((f for f, _w in measured), default=0.0)
    building_coverage = (
        "full" if building_union >= 0.9
        else "none" if building_union <= 0.0
        else "partial"
    )
    
    features.append(_geometry_feature(
        by_kind["BUILDING_MAIN"], confidence=ConfidenceLevel.HIGH,
        use="confirmed_footprint_proxy_volume", working_crs=geometry.working_crs,
        appearance_coverage=building_coverage,
        geometric_support_coverage=building_coverage,
        zone_state=(
            "trusted" if building_coverage == "full"
            else "unobserved" if building_coverage == "none"
            else "proxy"
        ),
    ))
    for kind in ("FACADE_PRIMARY", "FACADE_LEFT", "FACADE_RIGHT", "FACADE_REAR"):
        mesure = facade_coverage.get(kind, {})
        couverture = mesure.get("appearance_coverage", "none")
        support = mesure.get("geometric_support_coverage", "none")
        has_synthetic = "synthesis" in mesure
        vp_count = per_facade_viewpoints.get(kind, 0)
        zone_state = _zone_state_for_facade(kind, couverture, vp_count, has_synthetic)
        features.append(_geometry_feature(
            by_kind[kind], confidence=ConfidenceLevel.MEDIUM,
            use=(
                "qualified_geometry_proxy_no_appearance" if couverture == "none"
                else "qualified_geometry_with_observed_appearance"
            ),
            working_crs=geometry.working_crs,
            appearance_coverage=couverture,
            geometric_support_coverage=support,
            measurement=mesure,
            zone_state=zone_state,
            viewpoint_count=vp_count,
        ))
    access = next(g for g in geometry.geometries if g.feature_id == "ACCESS_ROAD_MAIN")
    from shapely import wkt
    from shapely.geometry import mapping
    features.append({
        "type": "Feature", "id": access.feature_id,
        "geometry": mapping(wkt.loads(access.wgs84_wkt)),
        "properties": {"kind": "ACCESS_ROAD_MAIN", "state": "inferred", "confidence": "medium", "use": "capture_required", "access_status": "restricted"},
    })
    for row in rechecks:
        features.append({
            "type": "Feature", "id": f"unresolved:{row.kind}", "geometry": None,
            "properties": {"kind": row.kind, "state": "unresolved", "confidence": "none", "use": "forbidden_claim", "reason": row.finding},
        })
    for kind in sorted(set(no_claim_kinds) - {row.kind for row in rechecks}):
        obj = by_kind[kind]
        features.append({
            "type": "Feature", "id": f"unlocated:{kind}", "geometry": None,
            "properties": {
                "kind": kind, "state": obj.state.value,
                "confidence": "low", "use": "known_but_location_unknown",
                "reason": "objet connu sans géométrie propre",
            },
        })
    zone_confidence = {
        "type": "FeatureCollection",
        "name": "lot_1b_zone_confidence",
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": features,
    }

    applied_runs = {a.visibility_run_id for a in assets.assets if a.visibility_run_id}
    if len(applied_runs) != 1:
        raise ValueError(
            f"visibilité courante ambiguë : {len(applied_runs)} run(s) porté(s)"
        )
    visibility_run_id = next(iter(applied_runs))
    application_files = sorted(
        workspace.path("06_geo").glob(
            f"visibility_application_{visibility_run_id}_*.json"
        ),
        key=lambda p: p.stat().st_mtime,
    )
    if not application_files:
        raise FileNotFoundError(
            f"reçu d'application de visibilité absent pour {visibility_run_id}"
        )
    visibility_file = application_files[-1]
    visibility = json.loads(visibility_file.read_text("utf-8"))
    visibility_counts = Counter(
        (
            getattr(a.line_of_sight_status, "value", a.line_of_sight_status)
            if a.line_of_sight_status else "not_assessed"
        )
        for a in assets.assets
    )
    duplicate_path = workspace.path("01_sources", "duplicate_report.json")
    duplicate_report_files = None
    if duplicate_path.is_file():
        duplicate_report_files = json.loads(
            duplicate_path.read_text("utf-8")
        ).get("files")
    uncleared_rights = (
        rights_counts[Rights.PUBLIC_UNCLEARED.value]
        + rights_counts[Rights.UNKNOWN.value]
    )
    blockers, limitations = _completion_findings(
        uncovered_capture_demands=uncovered_capture_demands,
        unresolved_objects=unresolved,
        source_coverage=sources,
        asset_count=len(assets.assets),
        duplicate_report_files=duplicate_report_files,
        dedup_robustness_current=_robust_dedup_is_current(workspace, context.policy),
        source_registry_complete=source_registry.closure_complete,
        uncleared_rights=uncleared_rights,
    )
    report = CoverageReport(
        hotel_id=workspace.hotel_id,
        generated_at=now,
        provenance=context.provenance,
        input_digest=digest_of({**input_digests, "contract_version": LOT1B_COVERAGE_CONTRACT_VERSION}),
        router={
            "file": router_path.name,
            "sha256": _sha256(router_path),
            "path": router["path"],
            "decision_status": router["decision_status"],
            "input_digest": router["input_digest"],
        },
        demands=demands,
        assets={"total": len(assets.assets), "roles": dict(sorted(role_counts.items()))},
        visibility={
            "run_id": visibility["run_id"],
            "assets_updated": visibility["assets_updated"],
            "by_status": dict(sorted(visibility_counts.items())),
            "note": "clear/partial/at_risk décrivent la ligne de vue, pas l'identité ni le cadrage",
        },
        geometry={
            "working_crs": geometry.working_crs,
            "front_azimuth_deg": spatial.front_azimuth_deg,
            "qualified_proxies": [z["zone"] for z in qualified_proxies],
            "road_corridors": len(geometry.corridors),
            "obstacles": len(geometry.resolved(GeometryRole.OBSTACLE_BUILDING)),
        },
        rights={
            "by_status": dict(sorted(rights_counts.items())),
            "uncleared_or_unknown": uncleared_rights,
            "photo_geometry_with_production_rights": len(usable_geometry),
            "rule": "un asset non clarifié ne devient pas une texture ou une preuve de production",
        },
        sources=source_payload,
        unresolved_objects=unresolved,
        lot_1b_status=Lot1BStatus.INCOMPLETE_CAPTURE_REQUIRED,
        blocking_reasons=blockers,
        limitations=limitations,
        outputs={
            "coverage_report": "coverage/coverage_report.json",
            "context_manifest": "coverage/context_manifest.json",
            "camera_constraints": "coverage/camera_constraints.json",
            "zone_confidence": "coverage/zone_confidence.geojson",
            "video_prompts": "scene_package/video_prompts.json",
            "lot_1b_report": "work/<hotel>/LOT_1B_REPORT.md",
        },
    )

    output_dir = workspace.path("coverage")
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace.write_json("coverage/context_manifest.json", context_manifest.model_dump(mode="json"))
    workspace.write_json("coverage/camera_constraints.json", camera_manifest.model_dump(mode="json"))
    workspace.write_json("coverage/zone_confidence.geojson", zone_confidence)
    workspace.write_json("coverage/coverage_report.json", report.model_dump(mode="json"))

    capture_lines = "\n".join(f"- `{demand}`" for demand in uncovered_capture_demands)
    blind_lines = "\n".join(f"- `{kind}`" for kind in blind_field_kinds)
    brief = f"""# Brief de capture complémentaire — {workspace.hotel_id}

## Besoins à couvrir

{capture_lines or '- aucun'}

## Champs visuels morts à éviter

{blind_lines or '- aucun'}

## Consignes de capture

- partir exclusivement des cibles et corridors résolus cités par ces besoins ;
- acquérir d'abord une miniature, puis imposer une revue humaine ;
- ne pas élargir aux autres besoins ;
- ne pas créditer l'entrée, l'enseigne ou le stationnement depuis cette vue ;
- ne lancer aucune prise de vue rapprochée sur une façade proxy ;
- hauteur de caméra : 1,2 m à 1,8 m (niveau piéton) ;
- distance recommandée : 10 m à 30 m de la façade ciblée ;
- recouvrement visuel demandé : 60 % à 80 % entre images consécutives ;
- sens de déplacement : continu, reliant façade, coins, côtés, arrière et stationnement ;
- lumière homogène, absence de pluie si possible ;
- aucune exigence de matériel spécialisé.

## Zones déjà couvertes à ne pas refaire

- toute zone `trusted` de `zone_confidence.geojson` ;
- toute vue déjà classée `confirmed` dans `asset_reviews.json` ;
- tout point de vue dont le `viewpoint_cluster` est déjà canonique.

Le Router reste `{router['path']} / {router['decision_status']}`. Une capture ne
change ce statut qu'après aperçu, décision humaine et nouvelle évaluation des besoins.
"""
    workspace.write_text("coverage/capture_brief.md", brief)

    source_lines = "\n".join(
        f"- {row.source_id}: {row.coverage_state.value} / {row.acquisition_state.value}"
        for row in sources
    )
    unresolved_lines = ", ".join(f"`{kind}`" for kind in unresolved)
    blind_lines = ", ".join(f"`{kind}`" for kind in blind_field_kinds)
    lot_report = f"""# Rapport Lot 1B — {workspace.hotel_id}

## Verdict

Le Lot 1B n'est pas clos. La route canonique est **{router['path']}** avec le
statut **{router['decision_status']}**. Le volume, la toiture et le terrain sont
exploitables comme proxies qualifiés ; leur apparence ne l'est pas.

## Couverture

- {len(assets.assets)} assets, dont {role_counts[ReconstructionRole.PHOTO_GEOMETRY.value]} porteurs de géométrie ;
- {demands['independent_viewpoints']} point de vue indépendant ;
- besoins satisfaits : {len(demands['satisfied'])}, partiels : {len(demands['partial'])}, ouverts : {len(demands['open'])} ;
- capture complémentaire : {', '.join(uncovered_capture_demands) or 'aucune'} ;
- droits : {rights_counts[Rights.PUBLIC_UNCLEARED.value] + rights_counts[Rights.UNKNOWN.value]} assets non clarifiés.

## Zones de confiance

Chaque façade est classée `trusted`, `proxy` ou `unobserved` dans
`zone_confidence.geojson` selon sa couverture d'apparence mesurée et le nombre
de points de vue indépendants qui l'observent.

## Contexte, orthophoto et cadastre

{source_lines}

## Objets réexaminés

Les objets suivants restent `unresolved` : {unresolved_lines}. Cette
relecture n'a apporté aucune preuve nouvelle et ne les promeut donc pas.

## Champs visuels morts

Les objets suivants sont établis mais jamais observés : {blind_lines}.
Ils ne sont pas interdits d'affirmation : leur existence est connue.
Ils doivent être évités par une caméra future, qui montrerait une forme
sans texture mesurée.

## Condition de fermeture

Obtenir et juger une vue de l'accès ; acquérir et qualifier l'orthophoto ;
acquérir l'extrait cadastral ; établir l'entrée actuelle et l'association du
stationnement ; remettre la déduplication au niveau du manifeste courant avec
les contrôles recadrage/filigrane ; et publier le registre canonique qui clôt
chaque famille photographique prioritaire. Régénérer ensuite ce rapport et
republier le Router si ses entrées changent. Aucun travail du Lot 2 n'est engagé.
Un export hybride/proxy peut être produit séparément ; il ne ferme aucun de ces
gates et ne vaut pas reconstruction SfM. Les droits non clarifiés bornent
l'usage des images, sans être présentés comme la cause du manque de couverture.
"""
    lot_report_path = workspace.write_text("LOT_1B_REPORT.md", lot_report)

    return {
        "context_manifest": output_dir / "context_manifest.json",
        "camera_constraints": output_dir / "camera_constraints.json",
        "coverage_report": output_dir / "coverage_report.json",
        "zone_confidence": output_dir / "zone_confidence.geojson",
        "capture_brief": output_dir / "capture_brief.md",
        "lot_1b_report": lot_report_path,
    }
