"""Candidats au dépistage, pris dans le manifeste plutôt que sur le disque.

Le dépistage parcourait le système de fichiers, quand tout le reste du pipeline
raisonne sur des **assets** qualifiés. Les deux ne se rejoignaient jamais : sur
ce pilote, cent quarante-trois des cent soixante-trois images dépistées
n'existaient pas au manifeste, et les références proposées ressortaient donc
avec un statut de droits inconnu.

C'est une garantie que le dépôt tient partout ailleurs — les droits sont un
citoyen de première classe — et que la couche d'identité contournait.

Le module fait le pont : il rend les candidats **avec** leur asset, pour que le
verdict d'identité hérite des droits, de l'azimut et de la cohorte temporelle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..logging import get_logger

log = get_logger("identity-candidates")

#: Droits sous lesquels une image peut servir de référence de production.
#: `unknown` n'y figure pas : ne pas savoir n'est pas une autorisation.
USABLE_RIGHTS = frozenset({"open_data", "owner_licensed", "public_domain"})


@dataclass
class Candidate:
    """Une image à dépister, et ce que le manifeste en dit déjà."""

    asset_id: str
    path: Path
    rights: str
    bearing_deg: float | None
    temporal_status: str | None
    #: Vrai quand l'image vient du manifeste ; faux pour un recadrage dérivé.
    in_manifest: bool = True
    #: Asset dont l'image dérive, pour un recadrage.
    source_asset_id: str | None = None

    @property
    def rights_cleared(self) -> bool:
        return self.rights in USABLE_RIGHTS

    def as_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "rights": self.rights,
            "rights_cleared": self.rights_cleared,
            "bearing_deg": None if self.bearing_deg is None else round(self.bearing_deg, 1),
            "temporal_status": self.temporal_status,
            "in_manifest": self.in_manifest,
            "source_asset_id": self.source_asset_id,
        }


@dataclass
class CandidateSet:
    """Les candidats d'un site, et ce qui a été écarté."""

    candidates: list[Candidate] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.candidates)

    def pairs(self) -> list[tuple[str, Path]]:
        """Forme attendue par le dépistage."""
        return [(c.asset_id, c.path) for c in self.candidates]

    def by_id(self) -> dict[str, Candidate]:
        return {c.asset_id: c for c in self.candidates}

    def as_dict(self) -> dict:
        cleared = sum(1 for c in self.candidates if c.rights_cleared)
        return {
            "count": len(self.candidates),
            "rights_cleared": cleared,
            "rights_unclear": len(self.candidates) - cleared,
            "skipped": self.skipped,
        }


def _bearing_of(asset: dict) -> float | None:
    raw = asset.get("bearing_from_building_deg")
    if raw in (None, "None", ""):
        return None
    try:
        return float(raw) % 360.0
    except (TypeError, ValueError):
        return None


def collect(
    manifest_path: Path,
    extra_folders: list[Path] | None = None,
    require_rights: bool = False,
) -> CandidateSet:
    """Rassemble les candidats depuis le manifeste, recadrages compris.

    Les recadrages vivent sur le disque sans exister au manifeste : ils sont
    admis, mais rattachés à l'asset dont ils dérivent pour en hériter les
    droits. Un recadrage dont la source reste introuvable garde un statut
    `unknown`, jamais un statut emprunté.
    """
    import json

    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    assets = payload.get("assets", [])
    by_id = {a["id"]: a for a in assets}

    result = CandidateSet()
    result.skipped = {"fichier_absent": 0, "droits": 0, "source_inconnue": 0}
    seen: set[Path] = set()

    for asset in assets:
        local = asset.get("local_path")
        if not local:
            result.skipped["fichier_absent"] += 1
            continue
        path = Path(local)
        if not path.is_file():
            result.skipped["fichier_absent"] += 1
            continue

        rights = str(asset.get("rights", "unknown"))
        if require_rights and rights not in USABLE_RIGHTS:
            result.skipped["droits"] += 1
            continue

        seen.add(path.resolve())
        result.candidates.append(
            Candidate(
                asset_id=str(asset["id"]),
                path=path,
                rights=rights,
                bearing_deg=_bearing_of(asset),
                temporal_status=str(asset.get("temporal_status") or "unknown"),
            )
        )

    for folder in extra_folders or []:
        folder = Path(folder)
        if not folder.is_dir():
            continue
        for image in sorted(folder.rglob("*.jpg")) + sorted(folder.rglob("*.png")):
            if image.resolve() in seen:
                continue
            source = _resolve_source(image.stem, by_id)
            rights = str(source.get("rights", "unknown")) if source else "unknown"
            if source is None:
                result.skipped["source_inconnue"] += 1
            if require_rights and rights not in USABLE_RIGHTS:
                result.skipped["droits"] += 1
                continue
            result.candidates.append(
                Candidate(
                    asset_id=image.stem,
                    path=image,
                    rights=rights,
                    bearing_deg=_bearing_of(source) if source else None,
                    temporal_status=(
                        str(source.get("temporal_status") or "unknown")
                        if source
                        else "unknown"
                    ),
                    in_manifest=False,
                    source_asset_id=str(source["id"]) if source else None,
                )
            )

    log.info(
        "candidats : %d (%d avec droits établis), écartés %s",
        len(result.candidates),
        sum(1 for c in result.candidates if c.rights_cleared),
        result.skipped,
    )
    return result


def _resolve_source(name: str, by_id: dict) -> dict | None:
    """Retrouve l'asset dont un recadrage dérive, par son jeton d'identifiant."""
    direct = by_id.get(name)
    if direct is not None:
        return direct
    for token in name.split("_")[1:]:
        if len(token) < 10:
            continue
        for key, asset in by_id.items():
            if token in key:
                return asset
    return None
