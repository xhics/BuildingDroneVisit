"""Régressions de la chaîne de dérivation (Lot 1B §9).

Cinq défauts que la suite ne couvrait pas encore, chacun capable de produire un
résultat plausible et faux.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from shapely import contains_xy
from shapely.geometry import Polygon

from hotel_pipeline.geo.derive import (
    DeriveResult,
    supersede_missing,
    tile_covers,
    verify_publication,
    verify_written,
)
from hotel_pipeline.geo.raster import NODATA, GridSpec, write_geotiff, write_mask
from hotel_pipeline.geo.terrain import pseudo_footprint_validation
from hotel_pipeline.schemas import (
    DerivedArtifact,
    GeoSourceProvenance,
    SiteManifest,
)

#: Bâtiment oblique : sa boîte englobante vaut près du double de son aire.
OBLIQUE = Polygon([(0, 0), (40, 20), (36, 28), (-4, 8)])

GRID = GridSpec(
    origin_x=100.0, origin_y=200.0, cell_m=0.5, width=6, height=4, crs="EPSG:2950"
)


class TestRingOnAnObliqueFootprint:
    """Un anneau polygonal n'est pas une boîte élargie.

    Sur un bâtiment oblique, la boîte inclut des zones bien plus éloignées d'un
    côté que de l'autre : les appuis y seraient déséquilibrés.
    """

    def test_the_ring_excludes_the_footprint_itself(self):
        ring = OBLIQUE.buffer(20).difference(OBLIQUE)
        assert not ring.intersects(OBLIQUE.buffer(-0.5))

    def test_the_ring_is_narrower_than_the_widened_box(self):
        ring = OBLIQUE.buffer(20).difference(OBLIQUE)
        minx, miny, maxx, maxy = OBLIQUE.bounds
        box_area = (maxx - minx + 40) * (maxy - miny + 40) - OBLIQUE.area
        assert ring.area < box_area

    def test_ring_membership_is_uniform_around_the_shape(self):
        """Chaque point de l'anneau est à moins de vingt mètres du bâtiment."""
        ring = OBLIQUE.buffer(20).difference(OBLIQUE)
        rng = np.random.default_rng(11)
        minx, miny, maxx, maxy = ring.bounds
        x = rng.uniform(minx, maxx, 4000)
        y = rng.uniform(miny, maxy, 4000)
        inside_ring = contains_xy(ring, x, y)

        from shapely.geometry import Point

        distances = [OBLIQUE.distance(Point(px, py)) for px, py in
                     zip(x[inside_ring][:200], y[inside_ring][:200])]
        assert max(distances) <= 20.0 + 1e-6

    def test_a_widened_box_would_reach_much_further(self):
        """Ce que la boîte aurait inclus, et que l'anneau écarte."""
        from shapely.geometry import Point

        minx, miny, maxx, maxy = OBLIQUE.bounds
        corner = Point(minx - 20, miny - 20)
        assert OBLIQUE.distance(corner) > 20.0


class TestPseudoFootprintCoverage:
    def test_coverage_fractions_are_reported_per_trial(self):
        rng = np.random.default_rng(5)
        x = rng.uniform(0, 120, 30000)
        y = rng.uniform(0, 120, 30000)
        z = 30 + 0.01 * x

        small = Polygon([(0, 0), (12, 0), (12, 8), (0, 8)])
        trials, _ = pseudo_footprint_validation(
            x, y, z, small, ring_m=8.0, trials=2, cell_m=1.0
        )
        assert trials
        for trial in trials:
            assert 0 < trial["truth_coverage_fraction"] <= 1
            assert 0 < trial["ring_coverage_fraction"] <= 1
            assert trial["reconstructed_fraction"] >= 0.9

    def test_a_point_count_alone_would_have_accepted_a_sparse_trial(self):
        """Trente points suffisaient autrefois ; la couverture les refuse."""
        rng = np.random.default_rng(9)
        x = rng.uniform(0, 120, 2000)
        y = rng.uniform(0, 120, 2000)
        z = np.full(2000, 30.0)

        small = Polygon([(0, 0), (12, 0), (12, 8), (0, 8)])
        trials, rejected = pseudo_footprint_validation(
            x, y, z, small, ring_m=8.0, cell_m=0.5
        )
        assert trials == []
        assert any("couvrant" in reason for reason in rejected)


class TestSearchAreaWithinTile:
    BOUNDS = {"min_x": 0.0, "max_x": 1000.0, "min_y": 0.0, "max_y": 1000.0}

    def test_a_centred_footprint_is_fully_covered(self):
        footprint = Polygon([(400, 400), (450, 400), (450, 450), (400, 450)])
        assert tile_covers(self.BOUNDS, footprint, 150.0)

    def test_a_footprint_near_the_edge_is_not(self):
        """Sans ce contrôle, l'absence d'essai passerait pour une difficulté."""
        footprint = Polygon([(20, 20), (70, 20), (70, 70), (20, 70)])
        assert not tile_covers(self.BOUNDS, footprint, 150.0)

    def test_unknown_bounds_are_treated_as_not_covering(self):
        with pytest.raises(KeyError):
            tile_covers({}, OBLIQUE, 150.0)


class TestCorruptedRasterIsDetected:
    @pytest.fixture
    def expected(self):
        values = np.arange(GRID.width * GRID.height, dtype=float).reshape(
            GRID.width, GRID.height
        )
        values[0, 0] = np.nan
        return values

    def _result(self, path, name="dtm"):
        result = DeriveResult(grid=GRID.as_dict())
        result.layers = {name: str(path)}
        return result

    def test_correct_values_pass(self, tmp_path, expected):
        path = tmp_path / "dtm.tif"
        write_geotiff(path, expected, GRID)
        assert verify_written(self._result(path), GRID, {"dtm": expected}) == []

    def test_altered_values_are_caught(self, tmp_path, expected):
        """Un raster non vide mais faux passait le contrôle précédent."""
        path = tmp_path / "dtm.tif"
        corrupted = expected.copy()
        corrupted[2, 2] += 5.0
        write_geotiff(path, corrupted, GRID)

        problems = verify_written(self._result(path), GRID, {"dtm": expected})
        assert any("valeurs divergentes" in problem for problem in problems)

    def test_a_shifted_defined_mask_is_caught(self, tmp_path, expected):
        path = tmp_path / "dtm.tif"
        holed = expected.copy()
        holed[3, 1] = np.nan
        write_geotiff(path, holed, GRID)

        problems = verify_written(self._result(path), GRID, {"dtm": expected})
        assert any("définies d'un côté" in problem for problem in problems)

    def test_a_flipped_mask_is_caught(self, tmp_path):
        mask = np.zeros((GRID.width, GRID.height), dtype=bool)
        mask[0, 0] = True
        path = tmp_path / "ndsm_valid.tif"
        write_mask(path, np.flipud(mask), GRID)

        problems = verify_written(
            self._result(path, "ndsm_valid"), GRID, {"ndsm_valid": mask}
        )
        assert any("masque divergentes" in problem for problem in problems)


class TestSuccessivePublications:
    def _source(self) -> GeoSourceProvenance:
        from datetime import datetime, timezone

        return GeoSourceProvenance(
            source_id="lidar", dataset="LiDAR", vintage="2023", tile_id="t",
            crs_horizontal="EPSG:2950", crs_vertical="CGVD 1928",
            carries_elevation=True, licence="CC BY 4.0", file_digest="abc",
            retrieved_at=datetime(2026, 8, 12, tzinfo=timezone.utc),
        )

    def _artifact(self, artifact_id: str, path, parents=(), **overrides):
        fields = dict(
            artifact_id=artifact_id, role="dtm", path=str(path), format="GeoTIFF",
            sha256="a" * 64, crs_horizontal="EPSG:2950", crs_vertical="CGVD 1928",
            resolution_m=0.5, algorithm_id="v1", measured_fraction=0.0,
            interpolated_fraction=1.0, coverage_domain="footprint",
            derived_from_sources=["lidar"], derived_from_artifacts=list(parents),
        )
        fields.update(overrides)
        return DerivedArtifact(**fields)

    def test_two_runs_keep_distinct_identifiers_and_lineage(self, tmp_path):
        first = tmp_path / "run1.tif"
        second = tmp_path / "run2.tif"
        for path in (first, second):
            write_geotiff(path, np.zeros((GRID.width, GRID.height)), GRID)

        manifest = SiteManifest(
            hotel_id="h",
            geo_sources=[self._source()],
            artifacts=[
                self._artifact("dtm@run1", first),
                self._artifact("ndsm@run1", first, parents=["dtm@run1"], role="ndsm"),
                self._artifact("dtm@run2", second),
                self._artifact("ndsm@run2", second, parents=["dtm@run2"], role="ndsm"),
            ],
        )
        assert len(manifest.artifacts) == 4
        assert verify_publication(manifest) == []

    def test_a_missing_file_is_refused_while_active(self, tmp_path):
        manifest = SiteManifest(
            hotel_id="h",
            geo_sources=[self._source()],
            artifacts=[self._artifact("dtm@run1", tmp_path / "disparu.tif")],
        )
        problems = verify_publication(manifest)
        assert any("fichier absent" in problem for problem in problems)

    def test_invalidating_it_clears_the_publication(self, tmp_path):
        manifest = SiteManifest(
            hotel_id="h",
            geo_sources=[self._source()],
            artifacts=[self._artifact("dtm@run1", tmp_path / "disparu.tif")],
        )
        marked = supersede_missing(manifest, "réorganisation des publications")
        assert marked == 1
        assert verify_publication(manifest) == []
        assert manifest.artifacts[0].invalidation_reason

    def test_an_invalidated_artifact_stays_in_the_manifest(self, tmp_path):
        """Conserver n'est pas exposer : il reste consultable, jamais courant."""
        manifest = SiteManifest(
            hotel_id="h",
            geo_sources=[self._source()],
            artifacts=[self._artifact("dtm@run1", tmp_path / "disparu.tif")],
        )
        supersede_missing(manifest, "motif")
        assert len(manifest.artifacts) == 1
        assert manifest.active_artifacts() == []

    def test_an_invalidation_without_reason_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="sans motif"):
            self._artifact("dtm@run1", tmp_path / "x.tif", status="invalidated")

    def test_a_supersession_without_successor_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="sans successeur"):
            self._artifact("dtm@run1", tmp_path / "x.tif", status="superseded")
