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
from .schemas import Asset, ClusterRole, GeometrySuitability, Rights
from .schemas.policy import DEFAULT_POLICY, PipelinePolicy
from .visibility import angular_difference, bearing_deg, haversine_m

log = get_logger("dedup-levels")

#: Deux caméras distantes de moins de cela occupent la même position utile.
POSITION_TOLERANCE_M = DEFAULT_POLICY.dedup.position_tolerance_m

#: Et regardent le bâtiment sous le même angle si leurs azimuts concordent.
BEARING_TOLERANCE_DEG = DEFAULT_POLICY.dedup.bearing_tolerance_deg

#: Vues supplémentaires conservées par point de vue, au-delà de la canonique.
#: Deux suffisent à porter un déplacement exploitable sans gonfler le compte.
MAX_OVERLAP_PER_CLUSTER = DEFAULT_POLICY.dedup.max_overlap_per_cluster

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
    robust_candidate_pairs: int = 0
    robust_matched_pairs: int = 0

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


def _plausible_republication_pair(first: Asset, second: Asset) -> bool:
    """Écarte les séquences de rue qui partagent légitimement leur contenu."""
    first_family = first.source_family or first.source
    second_family = second.source_family or second.source
    if first_family != second_family:
        return True
    return (
        first.camera_lat is None and first.camera_lon is None
        and second.camera_lat is None and second.camera_lon is None
    )


def robust_pairs(
    assets: list[Asset], *, region_cutoff: int, plausible_only: bool = True,
) -> list[tuple[str, str]]:
    """Détecte les recadrages sans confondre deux frames d'une même route."""
    import imagehash

    available = [asset for asset in assets if asset.crop_resistant_hash]
    parsed = {
        asset.id: imagehash.hex_to_multihash(asset.crop_resistant_hash)
        for asset in available
    }
    pairs: list[tuple[str, str]] = []
    for index, first in enumerate(sorted(available, key=lambda row: row.id)):
        for second in sorted(available, key=lambda row: row.id)[:index]:
            if plausible_only and not _plausible_republication_pair(first, second):
                continue
            if parsed[first.id].matches(
                parsed[second.id], region_cutoff=region_cutoff
            ):
                pairs.append((first.id, second.id))
    return pairs


def plausible_pair_count(assets: list[Asset], *, plausible_only: bool) -> int:
    available = [asset for asset in assets if asset.crop_resistant_hash]
    return sum(
        1
        for index, first in enumerate(available)
        for second in available[:index]
        if not plausible_only or _plausible_republication_pair(first, second)
    )


def robust_input_digest(assets: list[Asset]) -> str:
    """Empreinte fermée des seules données lues par le détecteur robuste.

    Une application de visibilité, une revue humaine ou un changement de rôle
    ne modifie pas ce que le détecteur de republications a comparé. Lier sa
    preuve au JSON complet du manifeste la périmait donc à chaque projection
    aval, bien que les fichiers et les signatures soient inchangés.

    Le contenu, la famille de source et la présence d'une position sont les
    seules entrées du filtrage robuste. Les coordonnées exactes sont conservées
    plutôt qu'un booléen afin qu'une correction de provenance spatiale déplace
    bien l'empreinte.
    """
    from .provenance import digest_of

    return digest_of([
        {
            "asset_id": asset.id,
            "checksum": asset.checksum,
            "crop_resistant_hash": asset.crop_resistant_hash,
            "source": asset.source,
            "source_family": asset.source_family,
            "camera_lat": asset.camera_lat,
            "camera_lon": asset.camera_lon,
        }
        for asset in sorted(assets, key=lambda row: row.id)
    ])


def robust_regression(region_cutoff: int) -> dict[str, bool]:
    """Régression exécutée par la commande, pas seulement par pytest."""
    import imagehash
    from PIL import Image, ImageDraw

    base = Image.new("RGB", (512, 384), "#d9d0c4")
    draw = ImageDraw.Draw(base)
    draw.rectangle((35, 85, 480, 340), fill="#8b3d2e")
    draw.rectangle((70, 120, 170, 280), fill="#4c708f")
    draw.polygon([(30, 90), (255, 20), (490, 90)], fill="#252525")
    for x in range(190, 450, 55):
        draw.rectangle((x, 140, x + 30, 210), fill="#e2e8ed")
    draw.text((170, 250), "HOTEL TEST", fill="white")
    cropped = base.crop((45, 25, 485, 365)).resize(base.size)
    watermarked = base.copy()
    overlay = ImageDraw.Draw(watermarked, "RGBA")
    overlay.rectangle((0, 315, 512, 384), fill=(255, 255, 255, 150))
    overlay.text((320, 335), "EXAMPLE.COM", fill=(0, 0, 0, 210))
    distinct = Image.new("RGB", base.size, "#224466")
    ImageDraw.Draw(distinct).ellipse((80, 40, 420, 360), fill="#ddcc22")

    hashes = {
        name: imagehash.crop_resistant_hash(image)
        for name, image in {
            "base": base, "crop": cropped, "watermark": watermarked,
            "distinct": distinct,
        }.items()
    }
    return {
        "crop": hashes["base"].matches(
            hashes["crop"], region_cutoff=region_cutoff
        ),
        "watermark": hashes["base"].matches(
            hashes["watermark"], region_cutoff=region_cutoff
        ),
        "distinct_rejected": not hashes["base"].matches(
            hashes["distinct"], region_cutoff=region_cutoff
        ),
    }


def _merge_photo_groups(
    assets: list[Asset], phash_groups: dict[str, str], pairs: list[tuple[str, str]],
) -> dict[str, str]:
    """Union transitive des groupes pHash et des preuves robustes."""
    ids = sorted(asset.id for asset in assets)
    parent = {asset_id: asset_id for asset_id in ids}

    def find(asset_id: str) -> str:
        while parent[asset_id] != asset_id:
            parent[asset_id] = parent[parent[asset_id]]
            asset_id = parent[asset_id]
        return asset_id

    def union(first: str, second: str) -> None:
        a, b = find(first), find(second)
        if a != b:
            parent[max(a, b)] = min(a, b)

    by_phash: dict[str, list[str]] = {}
    for asset_id, group_id in phash_groups.items():
        by_phash.setdefault(group_id, []).append(asset_id)
    for members in by_phash.values():
        for asset_id in members[1:]:
            union(members[0], asset_id)
    for first, second in pairs:
        union(first, second)
    return {asset_id: find(asset_id) for asset_id in ids}


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

    # L'ordre décide du résultat : un regroupement glouton produit un nombre de
    # points de vue différent selon l'ordre de parcours — 105 ou 118 sur le même
    # corpus, selon qu'on triait par identifiant ou par distance. L'ordre est
    # donc fixé ici, une fois pour toutes, et sur un critère utile : la vue la
    # plus proche du bâtiment nomme son groupe, départagée par la qualité et la
    # fraîcheur.
    def _order(asset: Asset) -> tuple:
        return (
            asset.target_distance_m if asset.target_distance_m is not None else float("inf"),
            -(asset.quality_score or 0.0),
            -(asset.capture_year or 1900),
            asset.id,
        )

    for asset in sorted(geolocated, key=_order):
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


#: Préférence sur l'aptitude géométrique, du plus utile au moins.
_SUITABILITY_RANK: dict[GeometrySuitability, int] = {
    GeometrySuitability.PRIMARY: 0,
    GeometrySuitability.AUXILIARY: 1,
    GeometrySuitability.UNASSESSED: 2,
    GeometrySuitability.INSUFFICIENT: 3,
}


def _quality_key(asset: Asset) -> tuple:
    """Ordre de préférence pour le fichier canonique d'un groupe.

    **La cible d'abord.** Choisir sur la seule résolution laissait un point de
    vue représenté par la vue la plus grande, fût-elle tournée ailleurs :
    l'unique fichier promu ne montrait alors pas le bâtiment, tandis que celui
    qui le montrait était rangé en `overlap` ou en `inactive`. Un point de vue
    doit être représenté par ce qu'il apporte, non par le poids de son JPEG.

    Ensuite seulement : aptitude géométrique, résolution, poids — une
    recompression agressive perd du détail à dimensions égales — et provenance.

    Fraîcheur temporelle et qualité mesurée départagent les ex aequo : une
    capture récente ou mieux notée prime pour la représentation canonique.
    """
    pixels = (asset.width or 0) * (asset.height or 0)
    return (
        # `False` trie avant `True` : la cible visible passe en tête.
        asset.target_building_visible is not True,
        _SUITABILITY_RANK.get(asset.geometry_suitability, 9),
        -pixels,
        -(asset.file_size_bytes or 0),
        PROVENANCE_RANK.get(asset.rights, 99),
        -(asset.capture_year or 1900),
        -(asset.quality_score or 0.0),
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


def measure_hashes(assets: list[Asset], *, robust: bool) -> tuple[int, int]:
    """Calcule les signatures absentes depuis le contenu local vérifié."""
    from .triage.dedup import crop_resistant_hash, phash

    phash_count = robust_count = 0
    for index, asset in enumerate(assets):
        if not asset.local_path:
            continue
        path = Path(asset.local_path)
        if not path.is_file():
            continue
        updates = {}
        try:
            if not asset.phash:
                updates["phash"] = phash(path)
                phash_count += 1
            if robust and not asset.crop_resistant_hash:
                updates["crop_resistant_hash"] = crop_resistant_hash(path)
                robust_count += 1
        except OSError as exc:
            log.warning("hash illisible pour %s : %s", asset.id, exc)
            continue
        if updates:
            assets[index] = asset.model_copy(update=updates)
    return phash_count, robust_count


def run(
    assets: list[Asset],
    building_lat: float,
    building_lon: float,
    policy: PipelinePolicy = DEFAULT_POLICY,
) -> DedupReport:
    """Applique les quatre niveaux et produit le rapport du §5."""
    measure_files(assets)
    measure_hashes(assets, robust=policy.dedup.robust_crop_hash_enabled)

    exact = exact_groups(assets)
    phash_only = perceptual_groups(
        assets, threshold=policy.dedup.phash_hamming_threshold
    )
    crop_pairs = (
        robust_pairs(
            assets,
            region_cutoff=policy.dedup.robust_region_cutoff,
            plausible_only=policy.dedup.robust_plausible_pairs_only,
        )
        if policy.dedup.robust_crop_hash_enabled else []
    )
    perceptual = _merge_photo_groups(assets, phash_only, crop_pairs)

    for index, asset in enumerate(assets):
        assets[index] = asset.model_copy(
            update={
                "exact_duplicate_group": exact.get(asset.id),
                "perceptual_duplicate_group": perceptual.get(asset.id),
            }
        )

    clusters, bearings = viewpoint_groups(
        assets,
        building_lat,
        building_lon,
        position_tolerance_m=policy.dedup.position_tolerance_m,
        bearing_tolerance_deg=policy.dedup.bearing_tolerance_deg,
    )
    for index, asset in enumerate(assets):
        assets[index] = asset.model_copy(
            update={
                "viewpoint_cluster": clusters.get(asset.id),
                "bearing_from_building_deg": bearings.get(asset.id),
            }
        )

    assign_roles(assets, max_overlap=policy.dedup.max_overlap_per_cluster)

    report = DedupReport(
        files=len(assets),
        exact_groups=len(set(exact.values())) if exact else 0,
        perceptual_groups=len({a.perceptual_duplicate_group or a.id for a in assets}),
        viewpoints=len({a.viewpoint_cluster for a in assets if a.viewpoint_cluster}),
        canonical=len([a for a in assets if a.cluster_role is ClusterRole.CANONICAL]),
        overlap=len([a for a in assets if a.cluster_role is ClusterRole.OVERLAP]),
        inactive=len([a for a in assets if a.cluster_role is ClusterRole.INACTIVE]),
        robust_candidate_pairs=plausible_pair_count(
            assets,
            plausible_only=policy.dedup.robust_plausible_pairs_only,
        ),
        robust_matched_pairs=len(crop_pairs),
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
