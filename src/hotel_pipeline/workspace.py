"""Arborescence de travail `work/<hotel>/` (plan directeur §18).

Chaque commande lit des entrées versionnées, produit des sorties versionnées,
et détecte proprement un résultat existant. `--force` est le seul moyen de
réécrire.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .schemas import ProjectManifest

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


def _atomic_write(path: Path, text: str) -> None:
    """Écriture atomique : un run interrompu ne laisse pas de fichier tronqué.

    Une VM préemptée en cours d'écriture produirait autrement un artefact
    partiel que la reprise considérerait comme valide.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text + "\n", encoding="utf-8")
    tmp.replace(path)
