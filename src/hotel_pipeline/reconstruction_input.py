"""Préparation de l'entrée de reconstruction Lot 2.

Ce module crée un `ReconstructionInputManifest` immuable qui snapshot
le corpus Lot 1B sélectionné pour la reconstruction. Tous les backends
(COLMAP, GLUEMAP, MP-SfM, MapAnything, VGGT) reçoivent exactement
les mêmes données.

Il produit également un `ReconstructionSelectionManifest` détaillé par asset
(selected, rejected, auxiliary, texture_only) avec motifs, et sépare
les cohortes temporelles (current_confirmed, historical, unknown).
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .schemas.reconstruction import (
    AssetTargetSupport,
    Criticality,
    ReconstructionInputManifest,
    ReconstructionRole,
    ReconstructionSelection,
    ReconstructionSelectionManifest,
    ReconstructionTarget,
    ReconstructionTargetKind,
    SupportRole,
    SupportType,
)
from .schemas.enums import Rights
from .workspace import Workspace


# Secteur d'une asset -> cible de reconstruction qu'elle soutient.
# Remplace les littéraux `PRIMARY` / `SECONDARY` gravés partout dans le code :
# la criticité est une décision par cible, pas une catégorie de surface.
_SECTOR_TO_TARGET = {
    "front": "FACADE_PRIMARY",
    "front_left_corner": "FACADE_PRIMARY",
    "front_right_corner": "FACADE_PRIMARY",
    "left": "FACADE_LEFT",
    "right": "FACADE_RIGHT",
    "rear": "FACADE_REAR",
    "rear_left_corner": "FACADE_REAR",
    "rear_right_corner": "FACADE_REAR",
}

# Objet du site -> criticité promotionnelle.
# Un `BUILDING_MAIN` confirmé est ce que la vidéo doit montrer ; un objet
# secondaire est un SHOULD_SHOW ; un objet non résolu est du contexte.
_KIND_CRITICALITY = {
    "BUILDING_MAIN": Criticality.MUST_SHOW,
    "BUILDING": Criticality.MUST_SHOW,
    "ENTRANCE": Criticality.MUST_SHOW,
    "POOL": Criticality.SHOULD_SHOW,
    "GARDEN": Criticality.SHOULD_SHOW,
    "ROOF": Criticality.SHOULD_SHOW,
    "PARK_RIDE": Criticality.CONTEXT_ONLY,
    "PARKING": Criticality.CONTEXT_ONLY,
    "ROAD": Criticality.CONTEXT_ONLY,
    "PEDESTRIAN": Criticality.CONTEXT_ONLY,
    "TREE": Criticality.CONTEXT_ONLY,
    "WATER": Criticality.CONTEXT_ONLY,
    "FACADE_PRIMARY": Criticality.MUST_SHOW,
    "FACADE_LEFT": Criticality.SHOULD_SHOW,
    "FACADE_RIGHT": Criticality.SHOULD_SHOW,
    "FACADE_REAR": Criticality.SHOULD_SHOW,
}

# Cibles promotionnelles par défaut, indépendantes du site manifest.
# Un hôtel a toujours une façade principale, une entrée, un toit, un sol.
_DEFAULT_TARGETS = [
    ReconstructionTarget(
        target_id="FACADE_PRIMARY",
        kind=ReconstructionTargetKind.SURFACE,
        criticality=Criticality.MUST_SHOW,
        required_fidelity=0.9,
        allowed_support=[
            SupportType.MEASURED_PHOTO,
            SupportType.MEASURED_LIDAR,
            SupportType.MULTIVIEW_RECONSTRUCTED,
            SupportType.FEEDFORWARD_INFERRED,
        ],
        maximum_generative_completion=0.0,
        maximum_inferred_geometry=0.5,
        minimum_camera_distance_m=8.0,
    ),
    ReconstructionTarget(
        target_id="FACADE_LEFT",
        kind=ReconstructionTargetKind.SURFACE,
        criticality=Criticality.SHOULD_SHOW,
        required_fidelity=0.7,
        allowed_support=[
            SupportType.MEASURED_PHOTO,
            SupportType.MEASURED_LIDAR,
            SupportType.MULTIVIEW_RECONSTRUCTED,
            SupportType.FEEDFORWARD_INFERRED,
        ],
        maximum_generative_completion=0.0,
        maximum_inferred_geometry=0.5,
    ),
    ReconstructionTarget(
        target_id="FACADE_RIGHT",
        kind=ReconstructionTargetKind.SURFACE,
        criticality=Criticality.SHOULD_SHOW,
        required_fidelity=0.7,
        allowed_support=[
            SupportType.MEASURED_PHOTO,
            SupportType.MEASURED_LIDAR,
            SupportType.MULTIVIEW_RECONSTRUCTED,
            SupportType.FEEDFORWARD_INFERRED,
        ],
        maximum_generative_completion=0.0,
        maximum_inferred_geometry=0.5,
    ),
    ReconstructionTarget(
        target_id="FACADE_REAR",
        kind=ReconstructionTargetKind.SURFACE,
        criticality=Criticality.SHOULD_SHOW,
        required_fidelity=0.7,
        allowed_support=[
            SupportType.MEASURED_PHOTO,
            SupportType.MEASURED_LIDAR,
            SupportType.MULTIVIEW_RECONSTRUCTED,
            SupportType.FEEDFORWARD_INFERRED,
        ],
        maximum_generative_completion=0.0,
        maximum_inferred_geometry=0.5,
    ),
    ReconstructionTarget(
        target_id="ROOF",
        kind=ReconstructionTargetKind.SURFACE,
        criticality=Criticality.SHOULD_SHOW,
        required_fidelity=0.6,
        allowed_support=[
            SupportType.MEASURED_PHOTO,
            SupportType.MEASURED_LIDAR,
            SupportType.MULTIVIEW_RECONSTRUCTED,
            SupportType.GEOSPATIAL_PROXY,
        ],
        maximum_generative_completion=0.0,
        maximum_inferred_geometry=0.3,
    ),
    ReconstructionTarget(
        target_id="ENTRANCE",
        kind=ReconstructionTargetKind.OBJECT,
        criticality=Criticality.MUST_SHOW,
        required_fidelity=0.85,
        allowed_support=[
            SupportType.MEASURED_PHOTO,
            SupportType.MULTIVIEW_RECONSTRUCTED,
            SupportType.FEEDFORWARD_INFERRED,
        ],
        maximum_generative_completion=0.0,
        maximum_inferred_geometry=0.4,
    ),
    ReconstructionTarget(
        target_id="POOL",
        kind=ReconstructionTargetKind.OBJECT,
        criticality=Criticality.SHOULD_SHOW,
        required_fidelity=0.6,
        allowed_support=[
            SupportType.MEASURED_PHOTO,
            SupportType.MULTIVIEW_RECONSTRUCTED,
            SupportType.FEEDFORWARD_INFERRED,
        ],
        maximum_generative_completion=0.0,
        maximum_inferred_geometry=0.4,
    ),
    ReconstructionTarget(
        target_id="GARDEN",
        kind=ReconstructionTargetKind.AREA,
        criticality=Criticality.SHOULD_SHOW,
        required_fidelity=0.5,
        allowed_support=[
            SupportType.MEASURED_PHOTO,
            SupportType.MULTIVIEW_RECONSTRUCTED,
            SupportType.FEEDFORWARD_INFERRED,
        ],
        maximum_generative_completion=0.0,
        maximum_inferred_geometry=0.4,
    ),
]


def _derive_targets(workspace: Workspace) -> list[ReconstructionTarget]:
    """Dérive les cibles de reconstruction depuis le site manifest.

    Les cibles promotionnelles par défaut sont toujours présentes ; un
    objet confirmé du site les enrichit ou les upgrade (un `BUILDING_MAIN`
    confirmé passe `FACADE_PRIMARY` en MUST_SHOW avec preuve).
    """
    targets = {t.target_id: t for t in _DEFAULT_TARGETS}
    try:
        from .schemas import SiteManifest
        site = SiteManifest.model_validate_json(workspace.site_path.read_text("utf-8"))
    except Exception:
        return list(targets.values())

    for obj in site.objects:
        criticality = _KIND_CRITICALITY.get(obj.kind)
        if criticality is None:
            continue

        # L'importance promotionnelle est **déclarée** ; l'état de la
        # géométrie est **constaté**. Les fondre — en rabattant sur
        # CONTEXT_ONLY tout objet non confirmé — faisait déclarer
        # NOT_APPLICABLE les quatre façades, dont la principale : la porte
        # de fidélité cessait d'évaluer la surface même que le produit doit
        # montrer, et le silence passait pour un succès.
        #
        # Une cible MUST_SHOW dont la géométrie n'est qu'inférée reste
        # MUST_SHOW. Le manque se dit — `geometry_confirmed=False` — il ne
        # dispense de rien.
        state = obj.state.value if obj.state is not None else None
        confirmed = state == "confirmed"

        target_id = obj.kind
        if target_id in targets:
            targets[target_id] = targets[target_id].model_copy(
                update={
                    "criticality": criticality,
                    "geometry_state": state,
                    "geometry_confirmed": confirmed,
                }
            )
        else:
            kind = (
                ReconstructionTargetKind.OBJECT
                if obj.kind in {"POOL", "ENTRANCE"}
                else ReconstructionTargetKind.AREA
            )
            targets[target_id] = ReconstructionTarget(
                target_id=target_id,
                kind=kind,
                criticality=criticality,
                geometry_state=state,
                geometry_confirmed=confirmed,
                required_fidelity=0.5,
                allowed_support=[
                    SupportType.MEASURED_PHOTO,
                    SupportType.MULTIVIEW_RECONSTRUCTED,
                    SupportType.FEEDFORWARD_INFERRED,
                ],
                maximum_generative_completion=0.0,
                maximum_inferred_geometry=0.4,
            )
    return list(targets.values())


def _derive_asset_target_support(
    selected_ids: list[str],
    workspace: Workspace,
) -> list[AssetTargetSupport]:
    """Associe chaque asset sélectionné à la cible qu'il soutient.

    Remplace `tier_assignment` : le soutien est une paire (asset, cible)
    avec un rôle et un taux de couverture, pas une étiquette globale.
    """
    from .schemas import AssetManifest
    assets = AssetManifest.model_validate_json(workspace.assets_path.read_text("utf-8"))
    by_id = {a.id: a for a in assets.assets}

    supports: list[AssetTargetSupport] = []
    for asset_id in selected_ids:
        asset = by_id.get(asset_id)
        if asset is None:
            continue
        sector = asset.view_sector.value if asset.view_sector else "unknown"
        target_id = _SECTOR_TO_TARGET.get(sector, "ROOF")
        supports.append(AssetTargetSupport(
            asset_id=asset_id,
            target_id=target_id,
            support_role=SupportRole.PRIMARY,
            coverage_fraction=1.0,
            quality_score=asset.quality_score or 0.0,
            reconstruction_role=asset.reconstruction_role or ReconstructionRole.REFERENCE_ONLY,
        ))
    return supports


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_input(
    hotel_id: str,
    *,
    allowed_backends: list[str] | None = None,
    mask_set_digest: str | None = None,
) -> tuple[ReconstructionInputManifest, ReconstructionSelectionManifest]:
    """Crée les manifestes d'entrée de reconstruction pour le Lot 2.

    Retourne:
        (ReconstructionInputManifest, ReconstructionSelectionManifest)
    """
    workspace = Workspace(hotel_id)

    # La décision Router est publiée horodatée et peut être invalidée : il
    # n'existe aucun `router_decision.json` statique. On résout la décision
    # courante par le même chemin que le Lot 1B, sans quoi le Lot 2 exigerait
    # un fichier que le pipeline ne produit jamais.
    from .lot1b_coverage import _router_decision

    static_router = workspace.path("10_validation/router_decision.json")
    if static_router.is_file():
        # Forme statique (fixtures, imports externes).
        router_decision_path = static_router
    else:
        # Forme publiée par le pipeline : horodatée et potentiellement
        # invalidée. On résout la décision courante par le même chemin que le
        # Lot 1B, sinon le Lot 2 exigerait un fichier jamais produit.
        try:
            router_decision_path, _ = _router_decision(workspace)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"manifeste requis absent : router_decision ({exc})"
            ) from exc

    required_files = {
        "asset_manifest": workspace.assets_path,
        "spatial_manifest": workspace.spatial_path,
        "site_manifest": workspace.site_path,
        "router_decision": router_decision_path,
        "coverage": workspace.path("coverage/coverage_report.json"),
    }
    for name, path in required_files.items():
        if not path.is_file():
            raise FileNotFoundError(f"manifeste requis absent : {name} ({path})")

    digests = {
        "asset_manifest": _sha256(workspace.assets_path),
        "spatial_manifest": _sha256(workspace.spatial_path),
        "site_manifest": _sha256(workspace.site_path),
        "router_decision": _sha256(required_files["router_decision"]),
        "coverage": _sha256(required_files["coverage"]),
    }

    from .schemas import AssetManifest
    assets = AssetManifest.model_validate_json(workspace.assets_path.read_text("utf-8"))

    selected_ids: list[str] = []
    excluded_ids: list[str] = []
    selection_reasons: dict[str, str] = {}
    selections: list[ReconstructionSelection] = []
    temporal_cohorts: dict[str, list[str]] = defaultdict(list)

    for asset in assets.assets:
        cohort = _temporal_cohort(asset)
        if cohort:
            temporal_cohorts[cohort].append(asset.id)

        if asset.reconstruction_role is ReconstructionRole.PHOTO_GEOMETRY:
            if asset.rights in {Rights.OWNED, Rights.LICENSED, Rights.OPEN_DATA}:
                selected_ids.append(asset.id)
                selections.append(ReconstructionSelection(
                    asset_id=asset.id,
                    decision="selected",
                    reason=_selected_reason(asset),
                    reconstruction_role=asset.reconstruction_role,
                ))
            else:
                excluded_ids.append(asset.id)
                selection_reasons[asset.id] = f"droits non clarifiés ({asset.rights.value})"
                selections.append(ReconstructionSelection(
                    asset_id=asset.id,
                    decision="rejected",
                    reason=selection_reasons[asset.id],
                    reconstruction_role=asset.reconstruction_role,
                ))
        elif asset.reconstruction_role is ReconstructionRole.TEXTURE_REFERENCE:
            excluded_ids.append(asset.id)
            selection_reasons[asset.id] = "rôle texture_only"
            selections.append(ReconstructionSelection(
                asset_id=asset.id,
                decision="texture_only",
                reason=selection_reasons[asset.id],
                reconstruction_role=asset.reconstruction_role,
            ))
        else:
            excluded_ids.append(asset.id)
            selection_reasons[asset.id] = (
                f"rôle {asset.reconstruction_role.value} hors reconstruction"
            )
            selections.append(ReconstructionSelection(
                asset_id=asset.id,
                decision="rejected",
                reason=selection_reasons[asset.id],
                reconstruction_role=asset.reconstruction_role,
            ))

    if not selected_ids:
        raise ValueError(
            "aucun asset sélectionnable pour la reconstruction : "
            "vérifier les rôles et les droits"
        )

    reconstruction_input_id = (
        f"recon-{hotel_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )

    targets = _derive_targets(workspace)
    asset_target_support = _derive_asset_target_support(selected_ids, workspace)

    input_manifest = ReconstructionInputManifest(
        reconstruction_input_id=reconstruction_input_id,
        hotel_id=hotel_id,
        asset_manifest_digest=digests["asset_manifest"],
        spatial_manifest_digest=digests["spatial_manifest"],
        site_manifest_digest=digests["site_manifest"],
        coverage_digest=digests["coverage"],
        router_decision_digest=digests["router_decision"],
        targets=targets,
        asset_target_support=asset_target_support,
        selected_asset_ids=selected_ids,
        excluded_asset_ids=excluded_ids,
        selection_reasons=selection_reasons,
        mask_set_digest=mask_set_digest,
        allowed_backends=allowed_backends or ["colmap_incremental"],
        temporal_cohorts=dict(temporal_cohorts),
    )

    selection_manifest = ReconstructionSelectionManifest(
        reconstruction_input_id=reconstruction_input_id,
        selections=selections,
    )

    return input_manifest, selection_manifest


def _temporal_cohort(asset) -> str | None:
    """Détermine la cohorte temporelle d'un asset (point 17)."""
    try:
        from .temporal import TemporalStatus
        if asset.temporal_status is TemporalStatus.CURRENT_CONFIRMED:
            return "current_confirmed"
        if asset.temporal_status is TemporalStatus.HISTORICAL:
            return "historical"
        if asset.temporal_status is TemporalStatus.UNKNOWN:
            return "unknown"
    except Exception:
        pass
    return None


def _selected_reason(asset) -> str:
    """Raison de sélection d'un asset."""
    reasons = []
    if asset.reconstruction_role:
        reasons.append(f"rôle={asset.reconstruction_role.value}")
    if asset.view_sector and asset.view_sector.value != "unknown":
        reasons.append(f"secteur={asset.view_sector.value}")
    if asset.viewpoint_cluster:
        reasons.append(f"cluster={asset.viewpoint_cluster}")
    if asset.duplicate_group:
        reasons.append(f"duplicate_group={asset.duplicate_group}")
    return "; ".join(reasons) if reasons else "photo_geometry éligible"


def publish_input(manifest: ReconstructionInputManifest, workspace: Workspace) -> Path:
    """Publie le ReconstructionInputManifest sous 07_reconstruction/."""
    output_dir = workspace.path("07_reconstruction")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"reconstruction_input_{manifest.reconstruction_input_id}.json"
    workspace.write_json(output_path, manifest.model_dump(mode="json"))
    return output_path


def publish_selection(manifest: ReconstructionSelectionManifest, workspace: Workspace) -> Path:
    """Publie le ReconstructionSelectionManifest sous 07_reconstruction/."""
    output_dir = workspace.path("07_reconstruction")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"reconstruction_selection_{manifest.reconstruction_input_id}.json"
    workspace.write_json(output_path, manifest.model_dump(mode="json"))
    return output_path
