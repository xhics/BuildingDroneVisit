"""Arborescence de travail `work/<hotel>/` (plan directeur §18).

Chaque commande lit des entrées versionnées, produit des sorties versionnées,
et détecte proprement un résultat existant. `--force` est le seul moyen de
réécrire.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from .schemas import ProjectManifest

if TYPE_CHECKING:
    from .schemas import AssetManifest
    from .schemas.spatial import SpatialManifest

#: Sous-répertoires de l'espace de travail, dans l'ordre du pipeline (§18).
SUBDIRS: tuple[str, ...] = (
    "00_manifest",
    "01_sources",
    "02_images",
    "02_images/reference_only",
    "02_images/production_eligible",
    "03_preflight",
    "04_masks",
    "05_colmap",
    "06_geo",
    "07_reconstruction",
    "08_composite",
    "09_confidence",
    "10_validation",
)

MANIFEST_NAME = "project_manifest.json"
SPATIAL_MANIFEST_NAME = "spatial_manifest.json"
ASSET_MANIFEST_NAME = "asset_manifest.json"
POLICY_NAME = "pipeline_policy.json"
SITE_MANIFEST_NAME = "site_manifest.json"
REPORT_NAME = "report.json"


def work_root() -> Path:
    """Racine des espaces de travail.

    Surchargeable par ``HOTEL_PIPELINE_WORK`` : sur la VM GPU, le travail vit
    sur le Container Disk, pas dans le dépôt.
    """
    return Path(os.environ.get("HOTEL_PIPELINE_WORK", "work")).resolve()


class Workspace:
    """Espace de travail d'un hôtel."""

    def __init__(self, hotel_id: str, root: Path | None = None) -> None:
        self.hotel_id = hotel_id
        self.root = (root or work_root()) / hotel_id

    # -- arborescence ----------------------------------------------------

    def create(self) -> None:
        for subdir in SUBDIRS:
            (self.root / subdir).mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        return self.root.is_dir()

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    @property
    def manifest_path(self) -> Path:
        return self.root / "00_manifest" / MANIFEST_NAME

    @property
    def policy_path(self) -> Path:
        """Politique effective du projet.

        Elle vit dans l'espace de travail, non dans le répertoire courant :
        chercher `pipeline_policy.json` relativement au cwd faisait dépendre
        le résultat de l'endroit d'où la commande était lancée.
        """
        return self.root / "00_manifest" / POLICY_NAME

    @property
    def report_path(self) -> Path:
        return self.root / REPORT_NAME

    # -- manifeste -------------------------------------------------------

    def read_manifest(self) -> ProjectManifest:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"aucun manifeste pour {self.hotel_id!r}. "
                f"Lancez d'abord : hotel-pipeline init {self.hotel_id} --address \"...\""
            )
        return ProjectManifest.model_validate_json(self.manifest_path.read_text("utf-8"))

    def write_manifest(self, manifest: ProjectManifest) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.manifest_path, manifest.model_dump_json(indent=2))

    def write_json(self, relative: str, payload: object) -> Path:
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(target, json.dumps(payload, indent=2, ensure_ascii=False))
        return target

    def write_report(
        self, relative: str, report, context, production: str | None = None  # noqa: ANN001
    ) -> Path:
        """Écrit un rapport en y apposant sa provenance.

        Point de passage **obligatoire** : `write_json` reste disponible pour
        les données brutes, mais tout rapport doit dire avec quelle politique
        et quel profil il a été produit. Sans cela, un chiffre n'est pas
        reproductible — les rapports du WelcomINNS n'en portaient aucune.

        `production` ajoute les empreintes des facettes **lues** par ce type de
        production. L'empreinte complète reste inscrite quoi qu'il arrive : les
        deux niveaux ne se remplacent pas.
        """
        from .provenance import stamp, with_dependencies

        payload = report.as_dict() if hasattr(report, "as_dict") else dict(report)
        stamped = (
            with_dependencies(payload, context.policy, production, context.profile)
            if production
            else stamp(payload, context.policy, context.profile)
        )
        return self.write_json(relative, stamped)

    def read_json(self, relative: str) -> object | None:
        target = self.root / relative
        if not target.is_file():
            return None
        return json.loads(target.read_text("utf-8"))

    # -- manifeste spatial -----------------------------------------------

    @property
    def spatial_path(self) -> Path:
        return self.root / "00_manifest" / SPATIAL_MANIFEST_NAME

    def read_spatial(self) -> "SpatialManifest | None":
        if not self.spatial_path.is_file():
            return None
        from .schemas.spatial import SpatialManifest

        return SpatialManifest.model_validate_json(self.spatial_path.read_text("utf-8"))

    def write_spatial(self, spatial: "SpatialManifest") -> None:
        self.spatial_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.spatial_path, spatial.model_dump_json(indent=2))

    # -- manifeste d'assets ----------------------------------------------

    @property
    def assets_path(self) -> Path:
        return self.root / "00_manifest" / ASSET_MANIFEST_NAME

    def read_assets(self) -> "AssetManifest | None":
        if not self.assets_path.is_file():
            return None
        from .schemas import AssetManifest

        return AssetManifest.model_validate_json(self.assets_path.read_text("utf-8"))

    @property
    def site_path(self) -> Path:
        return self.root / "00_manifest" / SITE_MANIFEST_NAME

    def read_site(self):  # noqa: ANN201
        if not self.site_path.is_file():
            return None
        from .schemas.site import SiteManifest

        return SiteManifest.model_validate_json(self.site_path.read_text("utf-8"))

    def write_site(self, site) -> None:  # noqa: ANN001
        self.site_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.site_path, site.model_dump_json(indent=2))

    def write_assets(self, assets: "AssetManifest") -> None:
        self.assets_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.assets_path, assets.model_dump_json(indent=2))


def _atomic_write(path: Path, text: str) -> None:
    """Écriture atomique : un run interrompu ne laisse pas de fichier tronqué.

    Une VM préemptée en cours d'écriture produirait autrement un artefact
    partiel que la reprise considérerait comme valide.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text + "\n", encoding="utf-8")
    tmp.replace(path)
