"""Préparation de l'entrée de reconstruction Lot 2.

Ce module crée un `ReconstructionInputManifest` immuable qui snapshot
le corpus Lot 1B sélectionné pour la reconstruction. Tous les backends
(COLMAP, GLUEMAP, MP-SfM, MapAnything, VGGT) reçoivent exactement
les mêmes données.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .schemas.reconstruction import (
    ReconstructionInputManifest,
    ReconstructionSelection,
    ReconstructionSelectionManifest,
)
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
) -> ReconstructionInputManifest:
    """Crée un ReconstructionInputManifest immuable pour le Lot 2.

    Arguments:
      hotel_id: identifiant de l'hôtel
      allowed_backends: backends autorisés (défaut: colmap_incremental)
      mask_set_digest: empreinte du jeu de masques SfM, si applicable

    Retourne:
      ReconstructionInputManifest prêt à être publié.
    """
    workspace = Workspace(hotel_id)

    # Vérifier que les manifestes sources existent
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

    # Calculer les empreintes des manifestes sources
    digests = {
        "asset_manifest": _sha256(workspace.assets_path),
        "spatial_manifest": _sha256(workspace.spatial_path),
        "site_manifest": _sha256(workspace.site_path),
        "router_decision": _sha256(required_files["router_decision"]),
        "coverage": _sha256(required_files["coverage"]),
    }

    # Charger les assets et sélectionner ceux éligibles pour la reconstruction
    from .schemas import AssetManifest
    from .schemas.enums import ReconstructionRole, Rights

    assets = AssetManifest.model_validate_json(workspace.assets_path.read_text("utf-8"))

    selected_ids: list[str] = []
    excluded_ids: list[str] = []
    selection_reasons: dict[str, str] = {}

    for asset in assets.assets:
        if asset.reconstruction_role is ReconstructionRole.PHOTO_GEOMETRY:
            if asset.rights in {Rights.OWNED, Rights.LICENSED, Rights.OPEN_DATA}:
                selected_ids.append(asset.id)
            else:
                excluded_ids.append(asset.id)
                selection_reasons[asset.id] = f"droits non clarifiés ({asset.rights.value})"
        elif asset.reconstruction_role in (
            ReconstructionRole.TEXTURE_REFERENCE,
            ReconstructionRole.CONTEXT_LOCK,
            ReconstructionRole.IDENTITY_EVIDENCE,
        ):
            excluded_ids.append(asset.id)
            selection_reasons[asset.id] = (
                f"rôle {asset.reconstruction_role.value} hors reconstruction"
            )

    if not selected_ids:
        raise ValueError(
            "aucun asset sélectionnable pour la reconstruction : "
            "vérifier les rôles et les droits"
        )

    reconstruction_input_id = (
        f"recon-{hotel_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )

    return ReconstructionInputManifest(
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
    )


def publish_input(manifest: ReconstructionInputManifest, workspace: Workspace) -> Path:
    """Publie le ReconstructionInputManifest sous 07_reconstruction/."""
    output_dir = workspace.path("07_reconstruction")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"reconstruction_input_{manifest.reconstruction_input_id}.json"
    workspace.write_json(output_path, manifest.model_dump(mode="json"))
    return output_path
