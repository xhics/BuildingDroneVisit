"""Déduplication à quatre niveaux (Lot 1B §5).

La déduplication classique supprime ce qui se ressemble. Ici elle doit faire
l'inverse sur un point précis : **conserver le recouvrement utile**, car c'est
lui qui rendra un SfM possible. Elle hiérarchise donc au lieu d'effacer.

Niveau 1  fichier identique          — checksum
Niveau 2  même photographie republiée — pHash
Niveau 3  même point de vue           — position, azimut, distance
Niveau 4  recouvrement utile          — canonique + jusqu'à deux vues gardées

Les Gates comptent les points de vue du niveau 3, jamais les fichiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .logging import get_logger
from .schemas import Asset, ClusterRole, Rights
from .visibility import angular_difference, bearing_deg, haversine_m

log = get_logger("dedup-levels")

#: Deux caméras distantes de moins de cela occupent la même position utile.
POSITION_TOLERANCE_M = 10.0

#: Et regardent le bâtiment sous le même angle si leurs azimuts concordent.
BEARING_TOLERANCE_DEG = 25.0

#: Vues supplémentaires conservées par point de vue, au-delà de la canonique.
#: Deux suffisent à porter un déplacement exploitable sans gonfler le compte.
MAX_OVERLAP_PER_CLUSTER = 2

#: Qualité de provenance, du meilleur au moins bon. Départage les canoniques
#: à résolution égale : une source aux droits établis prime.
PROVENANCE_RANK: dict[Rights, int] = {
    Rights.OWNED: 0,
    Rights.LICENSED: 1,
    Rights.OPEN_DATA: 2,
    Rights.PUBLIC_UNCLEARED: 3,
    Rights.UNKNOWN: 4,
}


@dataclass
class DedupReport:
    files: int = 0
    exact_groups: int = 0
    perceptual_groups: int = 0
    viewpoints: int = 0
    canonical: int = 0
    overlap: int = 0
    inactive: int = 0
    by_source_family: dict[str, dict[str, int]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "files": self.files,
            "unique_by_checksum": self.exact_groups,
            "unique_photographs": self.perceptual_groups,
            "independent_viewpoints": self.viewpoints,
            "roles": {
                "canonical": self.canonical,
                "overlap": self.overlap,
                "inactive": self.inactive,
            },
            "by_source_family": self.by_source_family,
        }


# --- niveau 1 : fichier identique ---------------------------------------


def exact_groups(assets: list[Asset]) -> dict[str, str]:
    """Regroupe par checksum. Un checksum non calculé ne regroupe rien."""
    groups: dict[str, str] = {}
    seen: dict[str, str] = {}
    placeholder = "0" * 64

    for asset in sorted(assets, key=lambda a: a.id):
        if not asset.checksum or asset.checksum == placeholder:
            continue
        groups[asset.id] = seen.setdefault(asset.checksum, asset.id)
    return groups


# --- niveau 2 : même photographie republiée ------------------------------


def perceptual_groups(assets: list[Asset], threshold: int = 6) -> dict[str, str]:
    """Regroupe les republications par pHash.

    Une republication recompressée, redimensionnée ou filigranée conserve un
    pHash proche. Le seuil est volontairement conservateur : mieux vaut deux
    groupes pour une même photo qu'une fusion erronée de deux clichés.
    """
    from .triage.dedup import group_duplicates

    hashes = {a.id: a.phash for a in assets if a.phash}
    return group_duplicates(hashes, threshold=threshold)


# --- niveau 3 : même point de vue ----------------------------------------


def viewpoint_groups(
    assets: list[Asset],
    building_lat: float,
    building_lon: float,
    position_tolerance_m: float = POSITION_TOLERANCE_M,
    bearing_tolerance_deg: float = BEARING_TOLERANCE_DEG,
) -> tuple[dict[str, str], dict[str, float]]:
    """Regroupe les prises de vue géolocalisées par position et azimut.

    Deux fichiers pris pratiquement au même endroit, regardant le bâtiment
    depuis la même direction, comptent comme **un seul** point de vue — même
    s'ils sont visuellement différents.

    Les images sans position ne peuvent pas être situées : chacune forme son
    propre point de vue, sauf republication déjà détectée au niveau 2.
    """
    clusters: dict[str, str] = {}
    bearings: dict[str, float] = {}
    representatives: list[tuple[str, float, float, float]] = []

    geolocated = [a for a in assets if a.camera_lat is not None and a.camera_lon is not None]

    for asset in sorted(geolocated, key=lambda a: a.id):
        bearing = bearing_deg(building_lat, building_lon, asset.camera_lat, asset.camera_lon)
        bearings[asset.id] = bearing

        match = None
        for rep_id, rep_lat, rep_lon, rep_bearing in representatives:
            distance = haversine_m(asset.camera_lat, asset.camera_lon, rep_lat, rep_lon)
            if (
                distance <= position_tolerance_m
                and angular_difference(bearing, rep_bearing) <= bearing_tolerance_deg
            ):
                match = rep_id
                break

        if match is None:
            representatives.append((asset.id, asset.camera_lat, asset.camera_lon, bearing))
            clusters[asset.id] = asset.id
        else:
            clusters[asset.id] = match

    # Sans position : le point de vue se rabat sur la photographie unique.
    for asset in assets:
        if asset.id in clusters:
            continue
        clusters[asset.id] = asset.perceptual_duplicate_group or asset.id

    log.info(
        "points de vue : %d indépendant(s) pour %d fichier(s)",
        len(set(clusters.values())),
        len(assets),
    )
    return clusters, bearings


# --- niveau 4 : recouvrement utile ---------------------------------------


def _quality_key(asset: Asset) -> tuple:
    """Ordre de préférence pour le fichier canonique d'un groupe.

    Résolution d'abord, puis poids — une recompression agressive perd du
    détail à dimensions égales —, puis qualité de provenance.
    """
    pixels = (asset.width or 0) * (asset.height or 0)
    return (
        -pixels,
        -(asset.file_size_bytes or 0),
        PROVENANCE_RANK.get(asset.rights, 99),
        asset.id,
    )


def assign_roles(assets: list[Asset], max_overlap: int = MAX_OVERLAP_PER_CLUSTER) -> None:
    """Attribue canonique, recouvrement et inactif au sein de chaque point de vue.

    Rien n'est supprimé : les fichiers surnuméraires restent au registre en
    `inactive`, consultables et réactivables.
    """
    by_cluster: dict[str, list[Asset]] = {}
    for asset in assets:
        by_cluster.setdefault(asset.viewpoint_cluster or asset.id, []).append(asset)

    roles: dict[str, ClusterRole] = {}
    for members in by_cluster.values():
        ordered = sorted(members, key=_quality_key)
        roles[ordered[0].id] = ClusterRole.CANONICAL
        for asset in ordered[1 : 1 + max_overlap]:
            roles[asset.id] = ClusterRole.OVERLAP
        for asset in ordered[1 + max_overlap :]:
            roles[asset.id] = ClusterRole.INACTIVE

    for index, asset in enumerate(assets):
        assets[index] = asset.model_copy(update={"cluster_role": roles[asset.id]})


# --- orchestration -------------------------------------------------------


def measure_files(assets: list[Asset]) -> int:
    """Renseigne dimensions et poids depuis les fichiers présents."""
    from PIL import Image

    measured = 0
    for index, asset in enumerate(assets):
        if not asset.local_path or asset.width is not None:
            continue
        path = Path(asset.local_path)
        if not path.is_file():
            continue
        try:
            with Image.open(path) as image:
                width, height = image.size
        except OSError as exc:
            log.warning("dimensions illisibles pour %s : %s", asset.id, exc)
            continue

        assets[index] = asset.model_copy(
            update={
                "width": width,
                "height": height,
                "file_size_bytes": path.stat().st_size,
            }
        )
        measured += 1
    return measured


def run(assets: list[Asset], building_lat: float, building_lon: float) -> DedupReport:
    """Applique les quatre niveaux et produit le rapport du §5."""
    measure_files(assets)

    exact = exact_groups(assets)
    perceptual = perceptual_groups(assets)

    for index, asset in enumerate(assets):
        assets[index] = asset.model_copy(
            update={
                "exact_duplicate_group": exact.get(asset.id),
                "perceptual_duplicate_group": perceptual.get(asset.id),
            }
        )

    clusters, bearings = viewpoint_groups(assets, building_lat, building_lon)
    for index, asset in enumerate(assets):
        assets[index] = asset.model_copy(
            update={
                "viewpoint_cluster": clusters.get(asset.id),
                "bearing_from_building_deg": bearings.get(asset.id),
            }
        )

    assign_roles(assets)

    report = DedupReport(
        files=len(assets),
        exact_groups=len(set(exact.values())) if exact else 0,
        perceptual_groups=len({a.perceptual_duplicate_group or a.id for a in assets}),
        viewpoints=len({a.viewpoint_cluster for a in assets if a.viewpoint_cluster}),
        canonical=len([a for a in assets if a.cluster_role is ClusterRole.CANONICAL]),
        overlap=len([a for a in assets if a.cluster_role is ClusterRole.OVERLAP]),
        inactive=len([a for a in assets if a.cluster_role is ClusterRole.INACTIVE]),
    )

    for asset in assets:
        family = asset.source_family or asset.source
        entry = report.by_source_family.setdefault(
            family, {"files": 0, "photographs": 0, "viewpoints": 0}
        )
        entry["files"] += 1

    for family in report.by_source_family:
        members = [a for a in assets if (a.source_family or a.source) == family]
        report.by_source_family[family]["photographs"] = len(
            {a.perceptual_duplicate_group or a.id for a in members}
        )
        report.by_source_family[family]["viewpoints"] = len(
            {a.viewpoint_cluster for a in members if a.viewpoint_cluster}
        )

    return report
