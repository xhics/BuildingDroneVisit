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
from hotel_pipeline.visibility import assess as assess_view

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
    """La politique décide toujours du cadrage, mais plus de la visibilité.

    `annotate()` a été supprimé : il jugeait sur un seul rayon vers le point le
    plus proche, et ses 29 occultations se sont toutes révélées non prouvées.
    Ce qui reste ici est le calcul de cadrage, seul usage légitime du champ de
    vision.
    """

    def test_narrow_fov_rejects_what_wide_fov_accepts(self):
        wide = assess_view(45.5725, -73.4433, 30.0, BUILDING,
                           half_fov_deg=DEFAULT_POLICY.geometry.half_fov_deg,
                           max_distance_m=DEFAULT_POLICY.geometry.max_distance_m)
        narrow = assess_view(45.5725, -73.4433, 30.0, BUILDING,
                             half_fov_deg=10.0,
                             max_distance_m=DEFAULT_POLICY.geometry.max_distance_m)

        assert wide.visible is True
        assert narrow.visible is False

    def test_short_max_distance_rejects_distant_cameras(self):
        result = assess_view(45.5715, -73.4433, 0.0, BUILDING,
                             half_fov_deg=DEFAULT_POLICY.geometry.half_fov_deg,
                             max_distance_m=50.0)
        assert result.visible is False


class TestTheOldAnnotatorIsGone:
    """Tant qu'il restait accessible, une collecte pouvait le rappeler."""

    def test_no_module_calls_the_single_ray_annotator(self):
        import pathlib

        from hotel_pipeline import visibility

        assert not hasattr(visibility, "annotate")

        # Un appel se reconnaît à son import ou à son accès qualifié : les
        # commentaires qui expliquent la suppression, eux, doivent rester.
        callers = []
        for path in pathlib.Path("src/hotel_pipeline").rglob("*.py"):
            for number, line in enumerate(path.read_text("utf-8").splitlines(), 1):
                code = line.split("#", 1)[0]
                if "import annotate" in code or "visibility.annotate" in code:
                    callers.append(f"{path.name}:{number}")
        assert callers == []

    def test_gathering_no_longer_produces_visibility(self):
        import pathlib

        source = pathlib.Path("src/hotel_pipeline/steps.py").read_text("utf-8")
        code = "\n".join(line.split("#", 1)[0] for line in source.splitlines())

        assert "annotate" not in code
        # La collecte dit désormais où la visibilité se calcule.
        assert "visibility assess" in source


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
            **usable(),
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
            country_code="CA", timezone="America/Toronto", ocr_languages=["fr", "en"],
            property_id="p", address="a", official_name="X",
            footprint_min_m2=1000, footprint_max_m2=5000,
        )
        candidate = build_candidates(elements, GeocodeResult(lat=45.5737, lon=-73.4434,
                                                            provider="t"), profile)[0]
        assert any("emprise plausible" in r for r in candidate.score_reasons)

    def test_mismatched_footprint_never_eliminates(self, elements):
        """Hors plage, le candidat perd des points mais reste examinable."""
        tiny = PropertyProfile(
            country_code="CA", timezone="America/Toronto", ocr_languages=["fr", "en"],
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
        return PropertyProfile(country_code="CA", timezone="America/Toronto", ocr_languages=["fr", "en"], property_id="welcominns", address="a", official_name="X")

    def test_report_carries_policy_and_profile_identity(self, profile):
        block = provenance(DEFAULT_POLICY, profile)
        assert block["policy_version"] == DEFAULT_POLICY.version
        assert block["model_calibration_id"] == DEFAULT_POLICY.model.calibration_id
        assert block["property_profile_id"] == "welcominns"
        # La calibration terrain est distincte de celle du modèle photo : les
        # confondre laisserait croire que les seuils géospatiaux reposent sur
        # les images du jeu de validation. Les deux valant « non-calibré » par
        # défaut, la distinction se vérifie sur une politique qui les sépare —
        # comparer les défauts entre eux ne prouvait plus rien.
        separated = DEFAULT_POLICY.model_copy(deep=True)
        separated.model.calibration_id = "campagne-images"
        separated.model.calibrated_on_sites = 2
        distinct = provenance(separated, profile)

        assert distinct["model_calibration_id"] == "campagne-images"
        assert distinct["terrain_calibration_id"] != distinct["model_calibration_id"]
        assert distinct["terrain_calibrated_on_sites"] == "0"

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


def usable(suitability="primary", by="hm", rationale="façade franche, lignes raccordables"):
    """Champs d'une aptitude géométrique établie.

    Une vue n'est plus porteuse du seul fait qu'on y reconnaît l'hôtel :
    l'aptitude est une décision distincte, et elle exige son historique.
    """
    from hotel_pipeline.review import assessment_fields
    from hotel_pipeline.schemas import GeometrySuitability

    return assessment_fields(
        GeometrySuitability(suitability), by, rationale,
        ["cadrage et netteté vérifiés sur la façade"], "a" * 64,
    )
