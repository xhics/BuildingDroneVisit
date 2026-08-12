"""Déduplication à quatre niveaux (Lot 1B §5, §14).

Le point délicat n'est pas de fusionner ce qui se ressemble : c'est de **ne
pas** fusionner deux positions réellement différentes, et de conserver le
recouvrement utile qui rendra un SfM possible.
"""

from __future__ import annotations

import pytest

from hotel_pipeline.dedup_levels import (
    assign_roles,
    exact_groups,
    run,
    viewpoint_groups,
)
from hotel_pipeline.schemas import Asset, AssetCategory, ClusterRole, Rights

BUILDING_LAT, BUILDING_LON = 45.5741, -73.4433


def make(asset_id: str, **overrides) -> Asset:
    fields = dict(
        id=asset_id,
        source="mapillary",
        source_url_or_id=f"https://x.invalid/{asset_id}.jpg",
        rights=Rights.OPEN_DATA,
        ai_eligible=False,
        confidence=0.8,
        category=AssetCategory.OTHER,
        checksum="a" * 64,
    )
    fields.update(overrides)
    return Asset(**fields)


class TestLevel1Exact:
    def test_identical_checksums_group_together(self):
        groups = exact_groups([make("a"), make("b"), make("c", checksum="b" * 64)])
        assert groups["a"] == groups["b"] != groups["c"]

    def test_placeholder_checksum_never_groups(self):
        """Un checksum non calculé ne doit pas fusionner des images distinctes."""
        groups = exact_groups([make("a", checksum="0" * 64), make("b", checksum="0" * 64)])
        assert groups == {}


class TestLevel3Viewpoints:
    def test_same_spot_same_direction_is_one_viewpoint(self):
        """Deux clichés pris au même endroit ne comptent qu'une fois."""
        south = dict(camera_lat=45.5730, camera_lon=-73.4433)
        clusters, _ = viewpoint_groups(
            [make("a", **south), make("b", **south)], BUILDING_LAT, BUILDING_LON
        )
        assert clusters["a"] == clusters["b"]

    def test_opposite_sides_are_not_merged(self):
        """Le risque le plus grave : fondre l'avant et l'arrière en un seul point."""
        clusters, _ = viewpoint_groups(
            [
                make("sud", camera_lat=45.5730, camera_lon=-73.4433),
                make("nord", camera_lat=45.5752, camera_lon=-73.4433),
            ],
            BUILDING_LAT,
            BUILDING_LON,
        )
        assert clusters["sud"] != clusters["nord"]

    def test_positions_beyond_tolerance_stay_distinct(self):
        clusters, _ = viewpoint_groups(
            [
                make("a", camera_lat=45.5730, camera_lon=-73.4433),
                make("b", camera_lat=45.5730, camera_lon=-73.4460),
            ],
            BUILDING_LAT,
            BUILDING_LON,
        )
        assert clusters["a"] != clusters["b"]

    def test_bearing_is_recorded(self):
        """L'azimut dit de quel côté se tient l'observateur, pas où il regarde."""
        _, bearings = viewpoint_groups(
            [make("sud", camera_lat=45.5730, camera_lon=-73.4433)], BUILDING_LAT, BUILDING_LON
        )
        assert bearings["sud"] == pytest.approx(180, abs=2)

    def test_unpositioned_assets_fall_back_to_photograph(self):
        """Sans position, deux republications d'une même photo restent un point."""
        clusters, _ = viewpoint_groups(
            [
                make("expedia", perceptual_duplicate_group="p1"),
                make("kayak", perceptual_duplicate_group="p1"),
                make("autre", perceptual_duplicate_group="p2"),
            ],
            BUILDING_LAT,
            BUILDING_LON,
        )
        assert clusters["expedia"] == clusters["kayak"] != clusters["autre"]


class TestLevel4UsefulOverlap:
    def test_best_resolution_becomes_canonical(self):
        assets = [
            make("petit", viewpoint_cluster="vp", width=640, height=480),
            make("grand", viewpoint_cluster="vp", width=2048, height=1536),
        ]
        assign_roles(assets)
        by_id = {a.id: a for a in assets}
        assert by_id["grand"].cluster_role is ClusterRole.CANONICAL
        assert by_id["petit"].cluster_role is ClusterRole.OVERLAP

    def test_recompressed_copy_loses_to_the_source(self):
        """À dimensions égales, la recompression a perdu du détail."""
        assets = [
            make("source", viewpoint_cluster="vp", width=1600, height=1200, file_size_bytes=900_000),
            make("copie", viewpoint_cluster="vp", width=1600, height=1200, file_size_bytes=90_000),
        ]
        assign_roles(assets)
        assert {a.id: a.cluster_role for a in assets}["source"] is ClusterRole.CANONICAL

    def test_better_provenance_breaks_ties(self):
        assets = [
            make("trouble", viewpoint_cluster="vp", width=800, height=600,
                 file_size_bytes=100, rights=Rights.PUBLIC_UNCLEARED),
            make("clair", viewpoint_cluster="vp", width=800, height=600,
                 file_size_bytes=100, rights=Rights.OPEN_DATA),
        ]
        assign_roles(assets)
        assert {a.id: a.cluster_role for a in assets}["clair"] is ClusterRole.CANONICAL

    def test_useful_overlap_is_kept(self):
        """Les vues successives portent le déplacement : elles ne sont pas jetées."""
        assets = [make(f"v{i}", viewpoint_cluster="vp", width=1000 - i, height=1000)
                  for i in range(3)]
        assign_roles(assets)
        roles = [a.cluster_role for a in assets]
        assert roles.count(ClusterRole.CANONICAL) == 1
        assert roles.count(ClusterRole.OVERLAP) == 2
        assert ClusterRole.INACTIVE not in roles

    def test_surplus_is_deactivated_not_deleted(self):
        assets = [make(f"v{i}", viewpoint_cluster="vp", width=1000 - i, height=1000)
                  for i in range(6)]
        assign_roles(assets)
        assert len(assets) == 6
        assert len([a for a in assets if a.cluster_role is ClusterRole.INACTIVE]) == 3


class TestReport:
    def test_report_separates_files_photographs_and_viewpoints(self):
        """Le rapport doit distinguer les trois unités (§5, validation)."""
        same = dict(camera_lat=45.5730, camera_lon=-73.4433)
        assets = [
            make("a", phash="ffffffffffffffff", **same),
            make("b", phash="ffffffffffffffff", **same),
            make("c", phash="0000000000000000", camera_lat=45.5752, camera_lon=-73.4433),
        ]
        report = run(assets, BUILDING_LAT, BUILDING_LON)

        assert report.files == 3
        assert report.perceptual_groups == 2
        assert report.viewpoints == 2

    def test_counts_are_broken_down_by_source_family(self):
        assets = [
            make("m1", source_family="mapillary", camera_lat=45.5730, camera_lon=-73.4433),
            make("e1", source_family="expedia_media"),
        ]
        report = run(assets, BUILDING_LAT, BUILDING_LON)
        assert set(report.by_source_family) == {"mapillary", "expedia_media"}
        assert report.by_source_family["mapillary"]["files"] == 1
