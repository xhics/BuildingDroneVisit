"""Rôles de reconstruction et occultation (Lot 1B §4, §7, §11).

Deux corrections apportées aux étapes déjà livrées :

- aucune source n'est écartée, elles sont **affectées** — une photographie sans
  position ne porte pas de géométrie, mais reste la meilleure preuve d'identité
  ou d'apparence ;
- le champ de vision seul ne suffit pas : un pavillon interposé annule la vue.
"""

from __future__ import annotations

import pytest

from hotel_pipeline.roles import assign, role_for
from hotel_pipeline.schemas import (
    Asset,
    AssetCategory,
    ClusterRole,
    PropertyMatchStatus,
    ReconstructionRole,
    ReviewStatus,
    Rights,
    Subject,
)
from hotel_pipeline.visibility import is_occluded, obstacles_from

BUILDING = (
    "POLYGON ((-73.44380 45.57355, -73.44280 45.57355, "
    "-73.44280 45.57445, -73.44380 45.57445, -73.44380 45.57355))"
)


def make(asset_id="a", **overrides) -> Asset:
    fields = dict(
        id=asset_id,
        source="mapillary",
        source_url_or_id="https://x.invalid/1.jpg",
        rights=Rights.OPEN_DATA,
        ai_eligible=False,
        confidence=0.5,
        category=AssetCategory.OTHER,
        checksum="a" * 64,
    )
    fields.update(overrides)
    return Asset(**fields)


class TestRoleAssignment:
    def test_generic_building_does_not_carry_geometry(self):
        """« Un bâtiment » n'est pas « le bâtiment ».

        Ce test encodait auparavant le défaut : position connue plus bâtiment
        détecté suffisaient. Cela a promu 20 vues Street View montrant
        Boucherville Toyota, Rachelle Béry et Tetra Tech.
        """
        asset = make(camera_lat=45.573, camera_lon=-73.443, subjects=[Subject.BUILDING])
        role, reason = role_for(asset)
        assert role is ReconstructionRole.CONTEXT_LOCK
        assert "non établi" in reason

    def test_confirmed_target_carries_geometry(self):
        asset = make(
            camera_lat=45.573,
            camera_lon=-73.443,
            subjects=[Subject.BUILDING],
            target_building_visible=True,
            review_status=ReviewStatus.AUTOMATIC_ACCEPTED,
            cluster_role=ClusterRole.CANONICAL,
        )
        role, _ = role_for(asset)
        assert role is ReconstructionRole.PHOTO_GEOMETRY

    def test_unarbitrated_cluster_never_carries_geometry(self):
        """Un asset antérieur à la déduplication porte `None` et passait."""
        asset = make(
            camera_lat=45.573, camera_lon=-73.443, subjects=[Subject.BUILDING],
            target_building_visible=True, review_status=ReviewStatus.AUTOMATIC_ACCEPTED,
            cluster_role=None,
        )
        assert role_for(asset)[0] is ReconstructionRole.CONTEXT_LOCK

    def test_occlusion_downgrades_to_context(self):
        asset = make(
            camera_lat=45.573, camera_lon=-73.443,
            subjects=[Subject.BUILDING], target_building_visible=True,
            review_status=ReviewStatus.AUTOMATIC_ACCEPTED, occluded_by="way/999",
        )
        assert role_for(asset)[0] is ReconstructionRole.CONTEXT_LOCK

    def test_pending_review_never_carries_geometry(self):
        asset = make(
            camera_lat=45.573, camera_lon=-73.443,
            subjects=[Subject.BUILDING], target_building_visible=True,
            review_status=ReviewStatus.NEEDS_REVIEW,
        )
        assert role_for(asset)[0] is ReconstructionRole.CONTEXT_LOCK

    def test_inactive_viewpoint_never_carries_geometry(self):
        asset = make(
            camera_lat=45.573, camera_lon=-73.443,
            subjects=[Subject.BUILDING], target_building_visible=True,
            review_status=ReviewStatus.AUTOMATIC_ACCEPTED,
            cluster_role=ClusterRole.INACTIVE,
        )
        assert role_for(asset)[0] is ReconstructionRole.CONTEXT_LOCK

    def test_pre_renovation_view_is_texture_only(self):
        from hotel_pipeline.schemas import TemporalStatus

        asset = make(
            camera_lat=45.573, camera_lon=-73.443,
            subjects=[Subject.BUILDING], target_building_visible=True,
            review_status=ReviewStatus.AUTOMATIC_ACCEPTED,
            cluster_role=ClusterRole.CANONICAL,
            temporal_status=TemporalStatus.BEFORE_EVENT,
        )
        assert role_for(asset)[0] is ReconstructionRole.TEXTURE_REFERENCE

    def test_unpositioned_view_cannot_carry_geometry(self):
        """Sans position, aucune triangulation — même sur une superbe façade."""
        asset = make(subjects=[Subject.BUILDING])
        role, reason = role_for(asset)
        assert role is ReconstructionRole.TEXTURE_REFERENCE
        assert "sans position" in reason

    def test_readable_sign_without_position_proves_identity(self):
        """Le site officiel a fourni le seul « CLUB ÉLITE WELCOMINNS » lu."""
        asset = make(sign_text="CLUB ÉLITE WELCOMINNS")
        role, _ = role_for(asset)
        assert role is ReconstructionRole.IDENTITY_EVIDENCE

    def test_positioned_view_without_building_locks_the_context(self):  # noqa: D401
        asset = make(camera_lat=45.573, camera_lon=-73.443, subjects=[Subject.ROAD])
        role, _ = role_for(asset)
        assert role is ReconstructionRole.CONTEXT_LOCK

    def test_interior_is_out_of_scope(self):
        asset = make(subjects=[Subject.INTERIOR, Subject.BUILDING])
        role, _ = role_for(asset)
        assert role is ReconstructionRole.REFERENCE_ONLY

    def test_competitor_sign_is_rejected(self):
        asset = make(
            camera_lat=45.573,
            camera_lon=-73.443,
            subjects=[Subject.BUILDING],
            property_match_status=PropertyMatchStatus.MISMATCH,
        )
        role, _ = role_for(asset)
        assert role is ReconstructionRole.REJECT

    def test_no_source_is_eliminated(self):
        """Chaque asset reçoit un rôle : aucun n'est retiré du registre."""
        assets = [
            make("geo", camera_lat=45.573, camera_lon=-73.443, subjects=[Subject.BUILDING],
                 target_building_visible=True, review_status=ReviewStatus.AUTOMATIC_ACCEPTED,
                 cluster_role=ClusterRole.CANONICAL),
            make("promo", subjects=[Subject.BUILDING]),
            make("interieur", subjects=[Subject.INTERIOR]),
            make("rien"),
        ]
        report = assign(assets)
        assert all(a.reconstruction_role is not None for a in assets)
        assert sum(report.counts.values()) == 4


class TestOcclusion:
    @pytest.fixture
    def blocking_house(self):
        return obstacles_from(
            [
                {
                    "type": "way",
                    "id": 999,
                    "tags": {"building": "house"},
                    "geometry": [
                        {"lat": 45.57300, "lon": -73.44360},
                        {"lat": 45.57300, "lon": -73.44300},
                        {"lat": 45.57330, "lon": -73.44300},
                        {"lat": 45.57330, "lon": -73.44360},
                        {"lat": 45.57300, "lon": -73.44360},
                    ],
                }
            ],
            exclude_id="way/54581348",
        )

    def test_interposed_building_blocks_the_view(self):
        """Le défaut de l'étape 4 : viser l'empreinte à travers un pavillon."""
        obstacles = obstacles_from(
            [
                {
                    "type": "way",
                    "id": 999,
                    "tags": {"building": "house"},
                    "geometry": [
                        {"lat": 45.57300, "lon": -73.44360},
                        {"lat": 45.57300, "lon": -73.44300},
                        {"lat": 45.57330, "lon": -73.44300},
                        {"lat": 45.57330, "lon": -73.44360},
                        {"lat": 45.57300, "lon": -73.44360},
                    ],
                }
            ],
            exclude_id="way/1",
        )
        blocker = is_occluded(45.5725, -73.4433, 45.57355, -73.4433, obstacles)
        assert blocker == "way/999"

    def test_clear_line_of_sight_is_not_blocked(self, blocking_house):
        """Une visée qui contourne l'obstacle reste dégagée."""
        assert is_occluded(45.5725, -73.4420, 45.57355, -73.4420, blocking_house) is None

    def test_target_building_never_occludes_itself(self):
        obstacles = obstacles_from(
            [
                {
                    "type": "way",
                    "id": 1,
                    "tags": {"building": "hotel"},
                    "geometry": [
                        {"lat": 45.57355, "lon": -73.44380},
                        {"lat": 45.57355, "lon": -73.44280},
                        {"lat": 45.57445, "lon": -73.44280},
                        {"lat": 45.57445, "lon": -73.44380},
                        {"lat": 45.57355, "lon": -73.44380},
                    ],
                }
            ],
            exclude_id="way/1",
        )
        assert obstacles == []

    def test_non_buildings_are_not_obstacles(self):
        """Un stationnement ne masque rien."""
        assert obstacles_from(
            [{"type": "way", "id": 2, "tags": {"amenity": "parking"}, "geometry": []}],
            exclude_id="way/1",
        ) == []


class TestDeterministicClustering:
    def test_viewpoint_grouping_does_not_depend_on_input_order(self):
        """Le regroupement glouton rendait 105 ou 118 points de vue selon l'ordre."""
        from hotel_pipeline.dedup_levels import viewpoint_groups

        assets = [
            make("a", camera_lat=45.5730, camera_lon=-73.4433, target_distance_m=40.0),
            make("b", camera_lat=45.5730, camera_lon=-73.4434, target_distance_m=41.0),
            make("c", camera_lat=45.5752, camera_lon=-73.4433, target_distance_m=90.0),
        ]
        forward, _ = viewpoint_groups(assets, 45.5741, -73.4433)
        backward, _ = viewpoint_groups(list(reversed(assets)), 45.5741, -73.4433)
        assert forward == backward
