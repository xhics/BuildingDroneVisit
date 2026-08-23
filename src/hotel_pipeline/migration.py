"""Migration du manifeste vers la structure du Lot 1B (§13, étape 1).

Règle unique et non négociable : **on ne dérive que ce qui est déterministe.**

Le type de capture découle de la source, la famille aussi, le statut temporel
découle d'une version d'entrée déjà tranchée par un humain. En revanche le
secteur de vue, les sujets et le rôle de reconstruction ne se déduisent pas
d'une catégorie devinée par un classifieur : ils restent `unknown` et
`needs_review`.

Une catégorie ambiguë ne doit jamais devenir une certitude en traversant une
migration — c'est le critère d'acceptation de l'étape 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .logging import get_logger
from .schemas import (
    Asset,
    AssetManifest,
    CaptureType,
    EntranceVersion,
    ReviewStatus,
    Subject,
    TemporalStatus,
)

log = get_logger("migration")

#: Famille de source réelle, par collecteur. Plusieurs plateformes peuvent
#: republier une même famille : le tableau est le point unique où l'établir.
SOURCE_FAMILY: dict[str, str] = {
    "mapillary": "mapillary",
    "street_view": "google_streetview",
    "places": "google_places",
    "website": "hotel_website",
    "tripadvisor": "tripadvisor_traveler",
    "kartaview": "kartaview",
    "flickr": "flickr",
    "commons": "wikimedia_commons",
    "hotel": "hotel_direct",
}

#: Nature de la prise de vue, déductible de la source seule.
CAPTURE_TYPE: dict[str, CaptureType] = {
    "mapillary": CaptureType.STREET_IMAGERY,
    "street_view": CaptureType.STREET_IMAGERY,
    "places": CaptureType.TRAVELER,
    "tripadvisor": CaptureType.TRAVELER,
    "kartaview": CaptureType.STREET_IMAGERY,
    "flickr": CaptureType.TRAVELER,
    "commons": CaptureType.TRAVELER,
    "website": CaptureType.PROMOTIONAL,
    "hotel": CaptureType.HOTEL_CAPTURE,
}

#: Sources dont le cap est **choisi par nous** plutôt qu'observé. Street View
#: rend un panorama sphérique : la direction extraite exprime une intention de
#: cadrage. Sans cette correction, les assets créés avant l'ajout du champ
#: conservent la valeur par défaut et se voient créditer d'une preuve qu'ils
#: n'apportent pas.
CHOSEN_HEADING_SOURCES: frozenset[str] = frozenset({"street_view"})

TEMPORAL_FROM_ENTRANCE: dict[EntranceVersion, TemporalStatus] = {
    EntranceVersion.BEFORE_RENOVATION: TemporalStatus.BEFORE_EVENT,
    EntranceVersion.AFTER_RENOVATION: TemporalStatus.AFTER_EVENT,
    EntranceVersion.UNKNOWN: TemporalStatus.UNKNOWN,
}

#: Valeurs héritées, antérieures à la généralisation. Une date d'établissement
#: précis figurait alors dans le vocabulaire du schéma.
LEGACY_TEMPORAL: dict[str, str] = {
    "pre_2024": "before_renovation",
    "post_2024": "after_renovation",
}


@dataclass
class MigrationReport:
    total: int = 0
    already_migrated: int = 0
    source_family_set: int = 0
    capture_type_set: int = 0
    temporal_set: int = 0
    exact_groups: int = 0
    perceptual_groups: int = 0
    subjects_set: int = 0
    heading_corrected: int = 0
    left_unknown: dict[str, int] = field(default_factory=dict)
    unmapped_sources: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "already_migrated": self.already_migrated,
            "derived": {
                "source_family": self.source_family_set,
                "capture_type": self.capture_type_set,
                "temporal_status": self.temporal_set,
                "subjects": self.subjects_set,
                "heading_provenance_corrected": self.heading_corrected,
                "exact_duplicate_groups": self.exact_groups,
                "perceptual_duplicate_groups": self.perceptual_groups,
            },
            "left_unknown": self.left_unknown,
            "unmapped_sources": sorted(set(self.unmapped_sources)),
        }


def _subjects_from_evidence(asset: Asset) -> list[Subject]:
    """Sujets déductibles **sans** recourir à la catégorie devinée.

    Seules deux preuves sont acceptées ici : la géométrie, qui établit que le
    bâtiment est dans le champ, et l'OCR, qui établit qu'une enseigne a été
    lue. Tout le reste attend l'étape 3.
    """
    subjects: list[Subject] = []
    if asset.sees_building:
        subjects.append(Subject.BUILDING)
    if asset.sign_text and asset.sign_text.strip():
        subjects.append(Subject.SIGN)
    return subjects


def migrate(manifest: AssetManifest) -> tuple[AssetManifest, MigrationReport]:
    """Complète les champs déterministes, laisse le reste explicitement inconnu."""
    report = MigrationReport(total=len(manifest.assets))
    checksum_groups: dict[str, str] = {}

    for index, asset in enumerate(manifest.assets):
        # La provenance du cap se corrige même sur un asset déjà migré : le
        # champ est postérieur à leur création.
        if asset.source in CHOSEN_HEADING_SOURCES and asset.heading_is_measured:
            manifest.assets[index] = asset = asset.model_copy(
                update={"heading_is_measured": False}
            )
            report.heading_corrected += 1

        if asset.source_family is not None:
            report.already_migrated += 1
            continue

        updates: dict = {}

        family = SOURCE_FAMILY.get(asset.source)
        if family is None:
            report.unmapped_sources.append(asset.source)
            family = asset.source
        updates["source_family"] = family
        report.source_family_set += 1

        capture = CAPTURE_TYPE.get(asset.source)
        if capture is not None:
            updates["capture_type"] = capture
            report.capture_type_set += 1

        temporal = TEMPORAL_FROM_ENTRANCE[asset.entrance_version]
        if temporal is not TemporalStatus.UNKNOWN:
            updates["temporal_status"] = temporal
            report.temporal_set += 1

        # Le checksum est une identité exacte : aucun jugement n'est porté.
        if asset.checksum and asset.checksum != "0" * 64:
            updates["exact_duplicate_group"] = checksum_groups.setdefault(
                asset.checksum, asset.id
            )

        # Le regroupement perceptuel existant est repris tel quel.
        if asset.duplicate_group:
            updates["perceptual_duplicate_group"] = asset.duplicate_group

        subjects = _subjects_from_evidence(asset)
        if subjects:
            updates["subjects"] = subjects
            report.subjects_set += 1

        # La qualification héritée vient d'un classifieur à classe forcée :
        # elle est conservée mais marquée comme telle, et reste à revoir.
        if asset.exterior_or_interior.value != "unknown":
            updates["classification_method"] = "legacy_openclip_single_label"
            updates["review_status"] = ReviewStatus.NEEDS_REVIEW

        manifest.assets[index] = asset.model_copy(update=updates)

    migrated = manifest.assets
    report.exact_groups = len({a.exact_duplicate_group for a in migrated if a.exact_duplicate_group})
    report.perceptual_groups = len(
        {a.perceptual_duplicate_group for a in migrated if a.perceptual_duplicate_group}
    )
    report.left_unknown = {
        "view_sector": len([a for a in migrated if a.view_sector.value == "unknown"]),
        "temporal_status": len([a for a in migrated if a.temporal_status is TemporalStatus.UNKNOWN]),
        "subjects_empty": len([a for a in migrated if not a.subjects]),
        "needs_review": len([a for a in migrated if a.review_status is ReviewStatus.NEEDS_REVIEW]),
    }

    log.info(
        "migration : %d asset(s), %d déjà migré(s), %d laissé(s) en revue",
        report.total,
        report.already_migrated,
        report.left_unknown["needs_review"],
    )
    return manifest, report
