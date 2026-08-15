"""Acquisition et fusion non destructive (Lot 1B V2).

`assets gather --force` reconstruisait le manifeste depuis la collecte. Or le
manifeste porte désormais ce que le pipeline ne sait pas régénérer : neuf
décisions humaines, leurs motifs, leurs preuves et leurs empreintes. Une
recollecte pouvait donc effacer une revue par un simple `--force`.

La fusion est ici **atomique et additive** : un asset existant n'est jamais
modifié, un nouvel asset est ajouté, une collision d'identifiant portant un
contenu différent est refusée, et une erreur partielle laisse le manifeste
identique octet pour octet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from .logging import get_logger
from .schemas import Asset

log = get_logger("acquisition")


class AcquisitionRefused(RuntimeError):
    """Le lot n'a pas été fusionné, et rien n'a été modifié."""


@dataclass
class MergeReport:
    added: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    run_id: str = ""
    plan_id: str = ""
    plan_digest: str = ""

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "plan_id": self.plan_id,
            "plan_digest": self.plan_digest,
            "added": sorted(self.added),
            "unchanged": sorted(self.unchanged),
            "counts": {"added": len(self.added), "unchanged": len(self.unchanged)},
        }


#: Forme admise d'un identifiant d'exécution : horodatage compact, rien
#: d'autre. Tout le reste — séparateur, `..`, chemin absolu — permettrait de
#: sortir du répertoire d'acquisition.
RUN_ID_PATTERN = re.compile(r"^(\d{8}T\d{6})(\d{6})?Z$")


def run_directory(workspace, run_id: str) -> Path:  # noqa: ANN001
    """Répertoire d'exécution d'une acquisition, garanti sous 02_images.

    Les nouveaux fichiers ne se mélangent pas aux 329 historiques : on doit
    pouvoir dire ce qu'une exécution a produit, et la défaire sans toucher au
    reste. Encore faut-il que l'identifiant ne puisse pas désigner autre
    chose : un `run_id` malformé sortirait de l'arborescence.
    """
    match = RUN_ID_PATTERN.match(run_id or "")
    if not match:
        raise AcquisitionRefused(
            f"identifiant d'exécution invalide : {run_id!r} — attendu un "
            "horodatage de la forme 20260813T140000000000Z"
        )
    try:
        # La forme ne suffit pas : `20261340T250000Z` la respecte sans désigner
        # aucun instant, et un répertoire ainsi nommé ne se replacerait pas
        # dans l'ordre des exécutions.
        datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")
    except ValueError as exc:
        raise AcquisitionRefused(
            f"identifiant d'exécution invalide : {run_id!r} — {exc}"
        ) from exc

    root = workspace.path("02_images", "acquisitions").resolve()
    target = (root / run_id).resolve()
    # Ceinture et bretelles : la forme est déjà contrainte, mais la
    # containment se vérifie sur le chemin résolu, pas sur la chaîne.
    if not target.is_relative_to(root):
        raise AcquisitionRefused(f"répertoire hors de l'arborescence : {target}")
    return target


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def merge(existing: list[Asset], acquired: list[Asset]) -> MergeReport:
    """Fusionne un lot acquis dans un manifeste, sans jamais l'écraser.

    Rien n'est écrit tant que le lot entier n'a pas été vérifié : une fusion à
    moitié appliquée serait pire qu'un échec, puisqu'elle paraîtrait complète.

    Un identifiant déjà présent n'est **pas** une erreur en soi — une
    acquisition rejouée retrouve les mêmes images. Il ne l'est que si le
    contenu diffère : l'asset existant porte alors peut-être des décisions
    prises sur une autre image.
    """
    report = MergeReport()
    by_id = {a.id: a for a in existing}
    problems: list[str] = []
    additions: list[Asset] = []

    seen: set[str] = set()
    for asset in acquired:
        if asset.id in seen:
            problems.append(f"{asset.id} : deux fois dans le lot acquis")
            continue
        seen.add(asset.id)

        current = by_id.get(asset.id)
        if current is None:
            additions.append(asset)
            continue

        if current.checksum != asset.checksum:
            problems.append(
                f"{asset.id} : identifiant déjà pris par un fichier différent "
                f"({current.checksum[:12]}… ≠ {asset.checksum[:12]}…) — "
                "les décisions existantes porteraient sur une autre image"
            )
            continue

        # Même identifiant, même contenu : on ne touche à rien. Réécrire
        # l'asset perdrait sa revue, ses rôles et sa grappe pour rien.
        report.unchanged.append(asset.id)

    if problems:
        raise AcquisitionRefused("; ".join(problems))

    existing.extend(additions)
    report.added = [a.id for a in additions]
    log.info(
        "fusion : %d ajouté(s), %d inchangé(s)", len(report.added), len(report.unchanged)
    )
    return report


def verify_acquired(assets: list[Asset], workspace_root: Path | None = None) -> list[str]:
    """Contrôle que chaque fichier acquis existe et correspond à son empreinte."""
    import hashlib

    problems: list[str] = []
    for asset in assets:
        if not asset.local_path:
            problems.append(f"{asset.id} : acquis sans fichier local")
            continue
        path = Path(asset.local_path)
        if workspace_root and not path.is_absolute():
            path = workspace_root / path
        if not path.is_file():
            problems.append(f"{asset.id} : fichier absent ({path})")
            continue

        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != asset.checksum:
            problems.append(
                f"{asset.id} : empreinte du fichier différente de celle déclarée"
            )
    return problems


def plan_is_current(plan, digests: dict[str, str | None], policy=None) -> list[str]:  # noqa: ANN001
    """Confronte un plan à l'état courant, sur une **liste fermée** de champs.

    Ne vérifier que les clés transmises laissait l'appelant décider de ce qui
    serait contrôlé : un plan sans lien avec le site passait pour courant. Ici,
    une empreinte absente — du plan comme de l'état courant — est un refus, et
    non un silence.

    `policy_digest` est comparé pour la **provenance** seulement quand aucune
    facette n'est déclarée. Dès que le plan porte ses dépendances, ce sont
    elles qui jugent : un seuil de terrain, ou une calibration renommée,
    n'a aucune raison de périmer une sélection photographique.
    """
    from .schemas.acquisition import REQUIRED_PLAN_DIGESTS, PlanStatus

    stale: list[str] = []
    if getattr(plan, "status", None) is PlanStatus.DRAFT:
        stale.append("plan à l'état de brouillon : un brouillon ne s'acquiert pas")

    recorded_facets = getattr(plan, "policy_dependency_digests", None) or {}
    if recorded_facets and policy is not None:
        from .policy_facets import stale_facets

        stale.extend(stale_facets(recorded_facets, policy, "AcquisitionPlan"))

    for name in REQUIRED_PLAN_DIGESTS:
        # La politique est jugée par ses facettes dès qu'elles existent :
        # comparer aussi l'empreinte complète rendrait la finesse inutile.
        if name == "policy_digest" and recorded_facets and policy is not None:
            continue
        recorded = getattr(plan, name, None)
        current = digests.get(name)
        if not recorded:
            stale.append(f"{name} : absent du plan")
            continue
        if not current:
            stale.append(f"{name} : état courant inconnu — comparaison impossible")
            continue
        if recorded != current:
            stale.append(f"{name} : plan {recorded[:12]}… ≠ courant {current[:12]}…")
    return stale


@dataclass
class Measured:
    """Ce que le fichier acquis est réellement."""

    checksum: str
    width: int
    height: int
    size_bytes: int
    image_format: str


def measure(
    path: Path,
    expected_size: tuple[int, int] | None = None,
    allow_uniform: bool = False,
) -> Measured:
    """Empreinte, format, dimensions et poids du fichier réellement acquis.

    Recopier les dimensions annoncées par le fournisseur masquerait un rendu
    tronqué, redimensionné ou remplacé par une image d'erreur. Un fichier
    illisible n'est pas non plus une image « aux dimensions inconnues » : c'est
    une acquisition ratée, et elle est refusée plutôt que promue en asset.
    """
    import hashlib

    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)

    if size == 0:
        raise AcquisitionRefused(f"{path.name} : fichier vide")

    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            image_format = (image.format or "").upper()
            uniform = _is_uniform(image)
    except ImportError as exc:  # pragma: no cover — Pillow est une dépendance
        raise AcquisitionRefused(
            f"{path.name} : impossible de contrôler l'image, Pillow absent"
        ) from exc
    except (OSError, ValueError) as exc:
        raise AcquisitionRefused(
            f"{path.name} : fichier illisible comme image — une acquisition "
            f"ratée ne devient pas un asset ({exc})"
        ) from exc

    if expected_size and (width, height) != expected_size:
        raise AcquisitionRefused(
            f"{path.name} : {width}×{height} au lieu des {expected_size[0]}×"
            f"{expected_size[1]} demandés — rendu tronqué ou redimensionné"
        )

    if uniform and not allow_uniform:
        # Street View rend une vignette grise « no imagery » avec un code 200 :
        # le téléchargement réussit, l'image ne montre rien.
        raise AcquisitionRefused(
            f"{path.name} : image d'une seule couleur — réponse de remplacement "
            "plutôt que photographie"
        )

    return Measured(digest.hexdigest(), width, height, size, image_format)


def _is_uniform(image) -> bool:  # noqa: ANN001
    """L'image est-elle d'une seule couleur ?"""
    extrema = image.convert("RGB").getextrema()
    return all(low == high for low, high in extrema)


def as_asset(candidate, provenance, local_path: Path, rights) -> Asset:  # noqa: ANN001
    """Construit l'asset d'un candidat acquis, provenance et mesures comprises.

    L'identité durable vient du fournisseur, jamais de l'URL : celles des CDN
    expirent, et une URL signée porterait la clé d'API jusque dans le
    manifeste. Les propriétés du fichier, elles, sont mesurées ici — pas
    reprises du candidat.
    """
    from .schemas import AssetCategory, ExteriorInterior

    measured = measure(local_path)

    try:
        return Asset(
            id=candidate.candidate_id,
            source=candidate.source,
            source_url_or_id=candidate.provider_id,
            rights=rights,
            ai_eligible=False,
            confidence=0.5,
            category=AssetCategory.OTHER,
            # `exterior` ne se présume pas : sans preuve fournisseur — le
            # `source=outdoor` de Street View, par exemple — l'inconnu reste
            # inconnu. Une vue d'intérieur déclarée extérieure fausserait la
            # couverture et le rôle.
            exterior_or_interior=(
                ExteriorInterior.EXTERIOR
                if candidate.outdoor_evidence
                else ExteriorInterior.UNKNOWN
            ),
            checksum=measured.checksum,
            local_path=str(local_path),
            width=measured.width,
            height=measured.height,
            file_size_bytes=measured.size_bytes,
            camera_lat=candidate.camera_lat,
            camera_lon=candidate.camera_lon,
            heading_deg=(
                candidate.requested_heading_deg
                if candidate.requested_heading_deg is not None
                else candidate.computed_heading_deg or candidate.original_heading_deg
            ),
            heading_is_measured=candidate.heading_is_measured,
            capture_year=candidate.captured_at.year if candidate.captured_at else None,
            acquisition=provenance,
        )
    except ValidationError as exc:
        raise AcquisitionRefused(
            f"{candidate.candidate_id} : asset invalide — {exc}"
        ) from exc
