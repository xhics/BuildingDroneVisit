"""Grille commune, orientation GeoTIFF et nDSM strict (Lot 1B §9).

Confondre les deux conventions produit un GeoTIFF parfaitement valide et
géographiquement retourné : aucune erreur, aucun avertissement, et un toit au
sud du bâtiment. D'où un test sur un tableau **asymétrique**, dont les quatre
coins diffèrent — un tableau symétrique passerait à l'envers.
"""

from __future__ import annotations

import numpy as np
import pytest

from hotel_pipeline.geo.raster import (
    NODATA,
    GridSpec,
    from_raster,
    normalised_height,
    sha256_file,
    to_raster,
    write_geotiff,
    write_mask,
)

GRID = GridSpec(
    origin_x=309226.0, origin_y=5048247.0, cell_m=0.5,
    width=4, height=3, crs="EPSG:2950",
)


def asymmetric() -> np.ndarray:
    """Valeurs distinctes par coin, en convention (colonne, ligne) Y-nord."""
    values = np.zeros((GRID.width, GRID.height))
    values[0, 0] = 11.0    # sud-ouest
    values[-1, 0] = 12.0   # sud-est
    values[0, -1] = 13.0   # nord-ouest
    values[-1, -1] = 14.0  # nord-est
    return values


class TestGridSpec:
    def test_north_is_derived_from_the_south_west_origin(self):
        assert GRID.north == 5048247.0 + 3 * 0.5

    def test_transform_starts_at_the_north_west_corner(self):
        transform = GRID.transform()
        assert transform.c == GRID.origin_x
        assert transform.f == GRID.north
        assert transform.e == -GRID.cell_m  # Y décroissant vers le sud

    def test_grids_are_comparable(self):
        assert GRID.matches(GridSpec(**GRID.as_dict()))
        assert not GRID.matches(GridSpec(**{**GRID.as_dict(), "cell_m": 1.0}))


class TestOrientation:
    def test_conversion_is_reversible(self):
        values = asymmetric()
        assert np.array_equal(from_raster(to_raster(values)), values)

    def test_first_raster_row_is_the_north(self):
        raster = to_raster(asymmetric())
        assert raster[0, 0] == 13.0   # nord-ouest en haut à gauche
        assert raster[0, -1] == 14.0  # nord-est
        assert raster[-1, 0] == 11.0  # sud-ouest en bas
        assert raster[-1, -1] == 12.0

    def test_shape_becomes_rows_by_columns(self):
        assert to_raster(asymmetric()).shape == (GRID.height, GRID.width)


class TestWrittenGeotiffIsGeographicallyCorrect:
    @pytest.fixture
    def written(self, tmp_path):
        path = tmp_path / "asymmetric.tif"
        write_geotiff(path, asymmetric(), GRID)
        return path

    def test_corners_land_at_the_right_coordinates(self, written):
        """Le contrôle décisif : relire et interroger par coordonnées."""
        import rasterio

        with rasterio.open(written) as source:
            def value_at(x, y):
                return next(source.sample([(x, y)]))[0]

            half = GRID.cell_m / 2
            assert value_at(GRID.origin_x + half, GRID.origin_y + half) == 11.0
            assert value_at(GRID.east - half, GRID.origin_y + half) == 12.0
            assert value_at(GRID.origin_x + half, GRID.north - half) == 13.0
            assert value_at(GRID.east - half, GRID.north - half) == 14.0

    def test_metadata_matches_the_grid(self, written):
        import rasterio

        with rasterio.open(written) as source:
            assert source.width == GRID.width
            assert source.height == GRID.height
            assert source.crs.to_string() == GRID.crs
            assert source.nodata == NODATA

    def test_nan_becomes_the_declared_nodata(self, tmp_path):
        import rasterio

        values = asymmetric()
        values[1, 1] = np.nan
        path = tmp_path / "holes.tif"
        write_geotiff(path, values, GRID)

        with rasterio.open(path) as source:
            band = source.read(1)
        assert (band == NODATA).sum() >= 1

    def test_no_partial_file_remains(self, written):
        assert list(written.parent.glob("*.part")) == []

    def test_mask_is_written_as_bytes(self, tmp_path):
        import rasterio

        mask = np.zeros((GRID.width, GRID.height), dtype=bool)
        mask[0, -1] = True  # nord-ouest
        path = tmp_path / "mask.tif"
        write_mask(path, mask, GRID)

        with rasterio.open(path) as source:
            assert source.dtypes[0] == "uint8"
            assert source.read(1)[0, 0] == 1  # première ligne = nord

    def test_digest_is_stable(self, written):
        assert sha256_file(written) == sha256_file(written)


class TestStrictNormalisedHeight:
    def test_height_is_the_difference_where_both_exist(self):
        dsm = np.array([[40.0]])
        dtm = np.array([[30.0]])
        valid = np.array([[True]])
        assert normalised_height(dsm, dtm, valid)[0, 0] == pytest.approx(10.0)

    def test_missing_terrain_yields_no_height(self):
        """Soustraire contre une altitude absente ferait entrer l'extrapolation."""
        dsm = np.array([[40.0]])
        dtm = np.array([[np.nan]])
        assert np.isnan(normalised_height(dsm, dtm, np.array([[True]]))[0, 0])

    def test_missing_roof_yields_no_height(self):
        dsm = np.array([[np.nan]])
        dtm = np.array([[30.0]])
        assert np.isnan(normalised_height(dsm, dtm, np.array([[True]]))[0, 0])

    def test_outside_the_valid_mask_yields_no_height(self):
        dsm = np.array([[40.0]])
        dtm = np.array([[30.0]])
        assert np.isnan(normalised_height(dsm, dtm, np.array([[False]]))[0, 0])

    def test_no_zero_is_ever_substituted(self):
        """Un zéro se lirait comme une hauteur nulle, donc comme une mesure."""
        dsm = np.array([[40.0, np.nan]])
        dtm = np.array([[np.nan, 30.0]])
        height = normalised_height(dsm, dtm, np.ones((1, 2), dtype=bool))
        assert np.isnan(height).all()
        assert not (height == 0).any()

    def test_negative_heights_are_preserved_not_clipped(self):
        """Une hauteur négative est un signal, pas une valeur à corriger."""
        dsm = np.array([[28.0]])
        dtm = np.array([[30.0]])
        assert normalised_height(dsm, dtm, np.array([[True]]))[0, 0] == pytest.approx(-2.0)


class TestShapeGuards:
    def test_write_refuses_a_mismatched_layer(self, tmp_path):
        """Une couche mal dimensionnée s'écrirait sans erreur visible."""
        with pytest.raises(ValueError, match="forme"):
            write_geotiff(tmp_path / "x.tif", np.zeros((2, 2)), GRID)

    def test_mask_write_refuses_too(self, tmp_path):
        with pytest.raises(ValueError, match="forme"):
            write_mask(tmp_path / "m.tif", np.zeros((9, 9), dtype=bool), GRID)

    def test_correct_shape_is_accepted(self, tmp_path):
        assert write_geotiff(
            tmp_path / "ok.tif", np.zeros((GRID.width, GRID.height)), GRID
        ).is_file()

    def test_height_refuses_broadcasting(self):
        """NumPy diffuserait silencieusement une ligne sur toute la grille."""
        with pytest.raises(ValueError, match="formes incompatibles"):
            normalised_height(
                np.zeros((4, 3)), np.zeros((1, 3)), np.ones((4, 3), dtype=bool)
            )

    def test_height_refuses_a_mismatched_mask(self):
        with pytest.raises(ValueError, match="formes incompatibles"):
            normalised_height(
                np.zeros((4, 3)), np.zeros((4, 3)), np.ones((2, 2), dtype=bool)
            )
