"""Câblage effectif de PipelinePolicy et PropertyProfile (Lot 1B, généricité).

Des modèles bien conçus mais déclaratifs ne servent à rien : tant qu'un module
lit sa constante locale, le pipeline peut produire un résultat différent de ce
que la politique annonce. Ces tests vérifient que **changer la politique change
le comportement**.
"""

from __future__ import annotations

import pytest

from hotel_pipeline import dedup_levels, roles
from hotel_pipeline.provenance import policy_digest, profile_digest, provenance, stamp
from hotel_pipeline.resolve import build_candidates
from hotel_pipeline.schemas import (
    Asset,
    AssetCategory,
    ClusterRole,
    DEFAULT_POLICY,
    PipelinePolicy,
    PropertyProfile,
    ReconstructionRole,
    ReviewStatus,
    Rights,
    Subject,
    TemporalStatus,
)
from hotel_pipeline.schemas.spatial import GeocodeResult
from hotel_pipeline.visibility import annotate

BUILDING = (
    "POLYGON ((-73.44380 45.57355, -73.44280 45.57355, "
    "-73.44280 45.57445, -73.44380 45.57445, -73.44380 45.57355))"
)


def make(asset_id="a", **overrides) -> Asset:
    fields = dict(
        id=asset_id,
        source="mapillary",
        source_url_or_id="u",
        rights=Rights.OPEN_DATA,
        ai_eligible=False,
        confidence=0.5,
        category=AssetCategory.OTHER,
        checksum="a" * 64,
    )
    fields.update(overrides)
    return Asset(**fields)


class TestGeometryPolicyIsHonoured:
    def test_narrow_fov_rejects_what_wide_fov_accepts(self):
        assets = [make(camera_lat=45.5725, camera_lon=-73.4433, heading_deg=30.0)]
        wide = annotate(list(assets), BUILDING, policy=DEFAULT_POLICY)

        strict = DEFAULT_POLICY.model_copy(deep=True)
        strict.geometry.half_fov_deg = 10.0
        narrow = annotate(list(assets), BUILDING, policy=strict)

        assert wide == 1
        assert narrow == 0

    def test_short_max_distance_rejects_distant_cameras(self):
        assets = [make(camera_lat=45.5715, camera_lon=-73.4433, heading_deg=0.0)]
        close = DEFAULT_POLICY.model_copy(deep=True)
        close.geometry.max_distance_m = 50.0
        assert annotate(list(assets), BUILDING, policy=close) == 0


class TestDedupPolicyIsHonoured:
    def test_tighter_position_tolerance_splits_a_cluster(self):
        assets = [
            make("a", camera_lat=45.5730, camera_lon=-73.44330, target_distance_m=40.0),
            make("b", camera_lat=45.5730, camera_lon=-73.44340, target_distance_m=41.0),
        ]
        loose = dedup_levels.run(list(assets), 45.5741, -73.4433, policy=DEFAULT_POLICY)

        strict = DEFAULT_POLICY.model_copy(deep=True)
        strict.dedup.position_tolerance_m = 1.0
        tight = dedup_levels.run(list(assets), 45.5741, -73.4433, policy=strict)

        assert loose.viewpoints == 1
        assert tight.viewpoints == 2

    def test_overlap_budget_is_taken_from_the_policy(self):
        assets = [make(f"v{i}", viewpoint_cluster="vp", width=1000 - i, height=1000)
                  for i in range(5)]
        policy = DEFAULT_POLICY.model_copy(deep=True)
        policy.dedup.max_overlap_per_cluster = 0
        dedup_levels.assign_roles(assets, max_overlap=policy.dedup.max_overlap_per_cluster)
        assert len([a for a in assets if a.cluster_role is ClusterRole.OVERLAP]) == 0


class TestTemporalPolicyIsHonoured:
    def _asset(self):
        return make(
            camera_lat=45.573, camera_lon=-73.443, subjects=[Subject.BUILDING],
            target_building_visible=True, review_status=ReviewStatus.AUTOMATIC_ACCEPTED,
            cluster_role=ClusterRole.CANONICAL, temporal_status=TemporalStatus.UNKNOWN,
        )

    def test_unknown_date_allowed_for_geometry_by_default(self):
        assert roles.role_for(self._asset())[0] is ReconstructionRole.PHOTO_GEOMETRY

    def test_strict_policy_rejects_undated_views(self):
        strict = DEFAULT_POLICY.model_copy(deep=True)
        strict.temporal.allow_unknown_for_geometry = False
        role, reason = roles.role_for(self._asset(), strict)
        assert role is ReconstructionRole.CONTEXT_LOCK
        assert "politique" in reason


class TestProfileDrivesFootprintScoring:
    @pytest.fixture
    def elements(self):
        return [
            {
                "type": "way", "id": 1, "tags": {"building": "yes"},
                "geometry": [
                    {"lat": 45.57355, "lon": -73.44380},
                    {"lat": 45.57355, "lon": -73.44300},
                    {"lat": 45.57390, "lon": -73.44300},
                    {"lat": 45.57390, "lon": -73.44380},
                    {"lat": 45.57355, "lon": -73.44380},
                ],
            }
        ]

    def test_matching_profile_awards_the_footprint_bonus(self, elements):
        profile = PropertyProfile(
            property_id="p", address="a", official_name="X",
            footprint_min_m2=1000, footprint_max_m2=5000,
        )
        candidate = build_candidates(elements, GeocodeResult(lat=45.5737, lon=-73.4434,
                                                            provider="t"), profile)[0]
        assert any("emprise plausible" in r for r in candidate.score_reasons)

    def test_mismatched_footprint_never_eliminates(self, elements):
        """Hors plage, le candidat perd des points mais reste examinable."""
        tiny = PropertyProfile(
            property_id="p", address="a", official_name="X",
            footprint_min_m2=10, footprint_max_m2=20,
        )
        candidates = build_candidates(
            elements, GeocodeResult(lat=45.5737, lon=-73.4434, provider="t"), tiny
        )
        assert len(candidates) == 1
        assert any("à vérifier" in r for r in candidates[0].score_reasons)


class TestProvenance:
    @pytest.fixture
    def profile(self) -> PropertyProfile:
        return PropertyProfile(property_id="welcominns", address="a", official_name="X")

    def test_report_carries_policy_and_profile_identity(self, profile):
        block = provenance(DEFAULT_POLICY, profile)
        assert block["policy_version"] == DEFAULT_POLICY.version
        assert block["calibration_id"] == DEFAULT_POLICY.model.calibration_id
        assert block["property_profile_id"] == "welcominns"

    def test_digest_detects_an_unversioned_edit(self):
        """Une version ne dit rien d'une modification locale non publiée."""
        tweaked = DEFAULT_POLICY.model_copy(deep=True)
        tweaked.model.subject_accept = 0.51
        assert policy_digest(tweaked) != policy_digest(DEFAULT_POLICY)
        assert tweaked.version == DEFAULT_POLICY.version

    def test_profile_digest_changes_with_the_profile(self, profile):
        other = profile.model_copy(update={"official_name": "Y"})
        assert profile_digest(other) != profile_digest(profile)

    def test_stamp_preserves_the_report_body(self, profile):
        stamped = stamp({"files": 3}, DEFAULT_POLICY, profile)
        assert stamped["files"] == 3
        assert stamped["provenance"]["property_profile_id"] == "welcominns"

    def test_policy_reload_keeps_the_same_digest(self):
        reloaded = PipelinePolicy.model_validate_json(DEFAULT_POLICY.model_dump_json())
        assert policy_digest(reloaded) == policy_digest(DEFAULT_POLICY)
