"""Artefacts dérivés (Lot 1B §9).

Un WKT ne peut pas représenter honnêtement une surface 2,5D : il dit où, jamais
à quelle altitude, ni à quelle résolution, ni quelle part est mesurée plutôt
qu'interpolée.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from hotel_pipeline.schemas import (
    DerivedArtifact,
    GeoSourceProvenance,
    ObjectState,
    SiteManifest,
    SiteObject,
)

SOURCE_ID = "lidar-quebec-23_3095048F08_DC"


def source() -> GeoSourceProvenance:
    return GeoSourceProvenance(
        source_id=SOURCE_ID,
        dataset="Données LiDAR du Québec",
        vintage="2023",
        tile_id="23_3095048F08_DC",
        crs_horizontal="EPSG:2950",
        crs_vertical="CGVD 1928",
        point_density_per_m2=15.0,
        carries_elevation=True,
        licence="CC BY 4.0",
        retrieved_at=datetime(2026, 8, 12, tzinfo=timezone.utc),
        file_digest="fc6407b2",
    )


def artifact(**overrides) -> DerivedArtifact:
    fields = dict(
        artifact_id="dtm-0.5m",
        role="dtm",
        path="06_geo/derived/dtm_0.5m.tif",
        format="GeoTIFF",
        sha256="a" * 64,
        crs_horizontal="EPSG:2950",
        crs_vertical="CGVD 1928",
        resolution_m=0.5,
        nodata=-9999.0,
        algorithm_id="tin-linear-v1",
        parameters={"ring_m": "20", "aggregation": "median"},
        measured_fraction=0.09,
        interpolated_fraction=0.88,
        derived_from_sources=[SOURCE_ID],
    )
    fields.update(overrides)
    return DerivedArtifact(**fields)


class TestArtifactInvariants:
    def test_valid_artifact_is_accepted(self):
        assert artifact().resolution_m == 0.5

    def test_fractions_cannot_exceed_the_whole(self):
        with pytest.raises(ValueError, match="dépasse 1"):
            artifact(measured_fraction=0.7, interpolated_fraction=0.5)

    def test_nodata_fraction_is_derived_not_declared(self):
        """Le reste se déduit : le déclarer permettrait de le contredire."""
        assert artifact(measured_fraction=0.2, interpolated_fraction=0.7).nodata_fraction == (
            pytest.approx(0.1)
        )

    def test_elevation_artifact_requires_a_vertical_datum(self):
        with pytest.raises(ValueError, match="référentiel vertical"):
            artifact(crs_vertical=None)

    def test_a_non_elevation_artifact_may_omit_it(self):
        assert artifact(role="footprint_mask", crs_vertical=None).crs_vertical is None

    def test_an_artifact_without_a_source_is_refused(self):
        """Une dérivation sans origine n'est pas vérifiable."""
        with pytest.raises(ValueError, match="sans source"):
            artifact(derived_from_sources=[])

    def test_resolution_must_be_positive(self):
        with pytest.raises(ValueError):
            artifact(resolution_m=0)

    def test_algorithm_and_parameters_are_retained(self):
        produced = artifact()
        assert produced.algorithm_id == "tin-linear-v1"
        assert produced.parameters["aggregation"] == "median"


class TestManifestIntegrity:
    def test_artifact_sources_must_be_declared(self):
        with pytest.raises(ValueError, match="sources non"):
            SiteManifest(
                hotel_id="h",
                geo_sources=[],
                artifacts=[artifact()],
            )

    def test_duplicate_artifact_ids_are_refused(self):
        with pytest.raises(ValueError, match="artefact dupliqués"):
            SiteManifest(
                hotel_id="h",
                geo_sources=[source()],
                artifacts=[artifact(), artifact(role="ndsm")],
            )

    def test_object_cannot_reference_an_absent_artifact(self):
        with pytest.raises(ValueError, match="artefacts absents"):
            SiteManifest(
                hotel_id="h",
                geo_sources=[source()],
                objects=[
                    SiteObject(
                        object_id="h:TERRAIN_MAIN",
                        kind="TERRAIN_MAIN",
                        artifact_ids=["dtm-inexistant"],
                    )
                ],
            )

    def test_a_complete_derivation_is_accepted(self):
        manifest = SiteManifest(
            hotel_id="h",
            geo_sources=[source()],
            artifacts=[artifact()],
            objects=[
                SiteObject(
                    object_id="h:TERRAIN_MAIN",
                    kind="TERRAIN_MAIN",
                    state=ObjectState.INFERRED,
                    derived_from_sources=[SOURCE_ID],
                    derivation_method="TIN local sur classe 2, anneau de 20 m",
                    artifact_ids=["dtm-0.5m"],
                )
            ],
        )
        assert manifest.summary()["artifacts"] == 1
        assert manifest.artifact("dtm-0.5m").role == "dtm"
        assert [o.kind for o in manifest.derived()] == ["TERRAIN_MAIN"]
