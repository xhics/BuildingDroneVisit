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
    ReconstructionInputManifest,
    ReconstructionSelection,
    ReconstructionSelectionManifest,
)
from .schemas.enums import ReconstructionRole, Rights
from .workspace import Workspace


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

    required_files = {
        "asset_manifest": workspace.assets_path,
        "spatial_manifest": workspace.spatial_path,
        "site_manifest": workspace.site_path,
        "router_decision": workspace.path("10_validation/router_decision.json"),
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

    input_manifest = ReconstructionInputManifest(
        reconstruction_input_id=reconstruction_input_id,
        asset_manifest_digest=digests["asset_manifest"],
        spatial_manifest_digest=digests["spatial_manifest"],
        site_manifest_digest=digests["site_manifest"],
        coverage_digest=digests["coverage"],
        router_decision_digest=digests["router_decision"],
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
