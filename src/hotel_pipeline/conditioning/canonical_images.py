"""Identité canonique des images : une table, aucune correspondance par nom.

Problème 34 — l'identité entre l'image originale, l'asset, le masque et
l'image COLMAP était appariée par nom ou par stem. Ce module remplace ce
matching fragile par un ``CanonicalImageId`` unique et une table de
correspondance explicite :

    canonical_image_id <-> asset_id <-> normalized_path <-> colmap_image_id <-> mask_id <-> checksum

Résolution par contenu uniquement : le sha256 du fichier référencé. Une
image COLMAP résout vers **exactement un asset ou aucun** — jamais deux.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

from ..logging import get_logger
from ..workspace import Workspace

log = get_logger("canonical-images")

INDEX_RELATIVE = "11_conditioning/canonical_images.json"
TABLE_CONTRACT_VERSION = 1


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _workspace_relative(workspace: Workspace, path: Path) -> str | None:
    try:
        return path.resolve().relative_to(workspace.root.resolve()).as_posix()
    except ValueError:
        return None


@dataclass(frozen=True)
class CanonicalImageRecord:
    canonical_image_id: str
    asset_id: str
    normalized_path: str
    checksum: str
    colmap_image_id: int | None = None
    colmap_name: str | None = None
    mask_id: str | None = None

    def with_colmap(self, image_id: int, name: str) -> "CanonicalImageRecord":
        return replace(self, colmap_image_id=int(image_id), colmap_name=str(name))


@dataclass
class CanonicalImageTable:
    records: list[CanonicalImageRecord]
    unresolved_colmap: list[dict[str, Any]] = field(default_factory=list)
    ambiguous_assets: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._by_canonical: dict[str, CanonicalImageRecord] = {}
        self._by_asset: dict[str, CanonicalImageRecord] = {}
        self._by_checksum: dict[str, CanonicalImageRecord] = {}
        self._by_colmap: dict[int, CanonicalImageRecord] = {}
        for record in self.records:
            if record.canonical_image_id in self._by_canonical:
                raise ValueError(f"canonical_image_id dupliqué : {record.canonical_image_id}")
            if record.asset_id in self._by_asset:
                raise ValueError(f"asset_id dupliqué dans la table : {record.asset_id}")
            self._by_canonical[record.canonical_image_id] = record
            self._by_asset[record.asset_id] = record
            self._by_checksum.setdefault(record.checksum, record)
            if record.colmap_image_id is not None:
                if record.colmap_image_id in self._by_colmap:
                    raise ValueError(
                        f"colmap_image_id {record.colmap_image_id} résout vers plusieurs assets"
                    )
                self._by_colmap[record.colmap_image_id] = record

    @classmethod
    def build(cls, workspace, assets, reconstruction=None, model_dir=None):
        records_by_checksum = {}
        asset_order = []
        ambiguous = []
        seen_paths = set()
        for asset in assets:
            asset_id = str(asset.get("id", ""))
            raw = asset.get("local_path")
            local = _resolve_local_path(workspace, raw)
            if local is None or str(local) in seen_paths:
                continue
            seen_paths.add(str(local))
            try:
                checksum = str(asset.get("checksum") or file_sha256(local))
            except OSError:
                continue
            normalized = _workspace_relative(workspace, local) or local.as_posix()
            record = CanonicalImageRecord(
                canonical_image_id=f"ci-{checksum[:16]}",
                asset_id=asset_id,
                normalized_path=normalized,
                checksum=checksum,
                mask_id=asset_id,
            )
            existing = records_by_checksum.get(checksum)
            if existing is not None:
                ambiguous.append(asset_id)
                continue
            records_by_checksum[checksum] = record
            asset_order.append(record)
        unresolved = []
        if reconstruction is not None:
            candidate_dirs = []
            if model_dir is not None:
                candidate_dirs.append(model_dir / "images")
            candidate_dirs.append(workspace.root)
            for image_id, model_image in sorted(_iter_colmap_images(reconstruction)):
                name = str(model_image.name)
                resolved = None
                for directory in candidate_dirs:
                    file_path = directory / name
                    if not file_path.is_file():
                        continue
                    try:
                        digest = file_sha256(file_path)
                    except OSError:
                        continue
                    hit = records_by_checksum.get(digest)
                    if hit is not None:
                        resolved = hit.with_colmap(image_id, name)
                        break
                if resolved is None:
                    unresolved.append({"colmap_image_id": int(image_id), "name": name})
                else:
                    asset_order[asset_order.index(records_by_checksum[resolved.checksum])] = resolved
        return cls(records=asset_order, unresolved_colmap=unresolved, ambiguous_assets=ambiguous)

    def resolve_colmap(self, image_id):
        return self._by_colmap.get(int(image_id))

    def resolve_asset(self, asset_id):
        return self._by_asset.get(str(asset_id))

    def resolve_mask(self, mask_id):
        for record in self.records:
            if record.mask_id == mask_id:
                return record
        return None

    def validate(self):
        errors = []
        seen_ids = set()
        seen_colmap = {}
        for record in self.records:
            if record.canonical_image_id in seen_ids:
                errors.append(f"canonical dupliqué : {record.canonical_image_id}")
            seen_ids.add(record.canonical_image_id)
            if record.colmap_image_id is not None:
                owner = seen_colmap.get(record.colmap_image_id)
                if owner is not None and owner != record.asset_id:
                    errors.append(f"colmap_image_id {record.colmap_image_id} partagé entre {owner} et {record.asset_id}")
                seen_colmap[record.colmap_image_id] = record.asset_id
        return errors

    def to_payload(self):
        return {
            "contract_version": TABLE_CONTRACT_VERSION,
            "records": [
                {
                    "canonical_image_id": r.canonical_image_id,
                    "asset_id": r.asset_id,
                    "normalized_path": r.normalized_path,
                    "checksum": r.checksum,
                    "colmap_image_id": r.colmap_image_id,
                    "colmap_name": r.colmap_name,
                    "mask_id": r.mask_id,
                }
                for r in self.records
            ],
            "unresolved_colmap": list(self.unresolved_colmap),
            "ambiguous_assets": list(self.ambiguous_assets),
        }

    def save(self, workspace):
        return workspace.write_json(INDEX_RELATIVE, self.to_payload())

    @classmethod
    def from_workspace(cls, workspace):
        path = workspace.path(*INDEX_RELATIVE.split("/"))
        payload = json.loads(path.read_text("utf-8"))
        if int(payload.get("contract_version", 0)) < TABLE_CONTRACT_VERSION:
            raise ValueError("table d'identité canonique : contrat inconnu")
        records = [CanonicalImageRecord(**item) for item in payload.get("records", [])]
        return cls(records=records, unresolved_colmap=payload.get("unresolved_colmap", []), ambiguous_assets=payload.get("ambiguous_assets", []))


def _iter_colmap_images(reconstruction):
    images = getattr(reconstruction, "images", {})
    for key, value in images.items():
        yield key, value


def _resolve_local_path(workspace, raw):
    if not raw:
        return None
    candidate = Path(str(raw))
    if candidate.is_file():
        return candidate
    normalized = str(raw).replace("\\", "/")
    marker = f"/work/{workspace.hotel_id}/"
    if marker in normalized:
        relocated = workspace.root / normalized.split(marker, 1)[1]
        if relocated.is_file():
            return relocated
    direct = workspace.root / normalized
    if direct.is_file():
        return direct
    return None


__all__ = ["CanonicalImageRecord", "CanonicalImageTable", "file_sha256"]
