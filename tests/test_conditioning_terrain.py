"""Relief du terrain porté au sol rendu (module `conditioning.terrain`).

À ne pas confondre avec `test_terrain.py`, qui couvre l'interpolation du
terrain sous un bâtiment au Lot 1B : ce fichier-ci porte sur le relief tel
qu'il est posé sous les surfaces de la scène conditionnée.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from hotel_pipeline.conditioning.terrain import (  # noqa: E402
    MIN_RELIEF_M,
    TerrainGrid,
    load,
)


def _dtm(tmp_path: Path, values: np.ndarray, res: float = 0.5) -> Path:
    path = tmp_path / "dtm.tif"
    transform = rasterio.transform.from_origin(0.0, values.shape[0] * res, res, res)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:2950",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(values.astype("float32"), 1)
    return path


def test_un_terrain_en_pente_est_releve(tmp_path: Path) -> None:
    size = 200
    slope = np.tile(np.linspace(100.0, 103.0, size), (size, 1))
    path = _dtm(tmp_path, slope)

    grid = load(path, (50.0, 50.0), radius_m=45.0)

    assert grid is not None
    assert grid.relief_m > 1.0
    # Le relief est rapporté au sol médian : le centre reste proche de zéro.
    assert abs(grid.height_at(50.0, 50.0)) < 1.0


def test_un_terrain_plat_ne_porte_aucun_relief(tmp_path: Path) -> None:
    """Aucune ondulation n'est ajoutée pour faire vivant."""
    path = _dtm(tmp_path, np.full((200, 200), 100.0))

    assert load(path, (50.0, 50.0), radius_m=45.0) is None


def test_un_modele_absent_laisse_le_sol_plat(tmp_path: Path) -> None:
    assert load(tmp_path / "rien.tif", (0.0, 0.0)) is None


def test_le_relief_suit_la_pente_dans_le_bon_sens(tmp_path: Path) -> None:
    """Une inversion d'axe placerait la butte dans le creux."""
    size = 200
    # Altitude croissante vers le nord (y croissant).
    grid_values = np.tile(
        np.linspace(103.0, 100.0, size).reshape(-1, 1), (1, size)
    )
    path = _dtm(tmp_path, grid_values)

    grid = load(path, (50.0, 50.0), radius_m=45.0)

    assert grid is not None
    nord = grid.height_at(50.0, 85.0)
    sud = grid.height_at(50.0, 15.0)
    assert nord > sud


def test_un_point_hors_grille_reste_au_niveau_de_reference() -> None:
    grid = TerrainGrid(
        x0=0.0, y0=0.0, step_m=4.0,
        heights=np.zeros((5, 5)), reference_z=100.0,
    )
    assert grid.height_at(1e6, 1e6) == 0.0


def test_le_rapport_porte_ses_reserves() -> None:
    grid = TerrainGrid(
        x0=0.0, y0=0.0, step_m=4.0,
        heights=np.array([[0.0, 1.0], [0.5, 1.5]]), reference_z=100.0,
    )
    payload = grid.as_dict()

    assert payload["relief_m"] == pytest.approx(1.5)
    joined = " ".join(payload["caveats"])
    assert "sol nu" in joined
    assert "faire vivant" in joined


def test_le_sol_rendu_epouse_le_terrain() -> None:
    """Posé à plat, le sol faisait flotter les volumes sur un plan idéal."""
    from hotel_pipeline.conditioning.ground_polygons import GroundPatch
    from hotel_pipeline.conditioning.render import _patch_faces

    patch = GroundPatch(
        kind="vegetal",
        ring=[(0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0), (0.0, 0.0)],
    )
    grid = TerrainGrid(
        x0=0.0, y0=0.0, step_m=4.0,
        heights=np.array([[0.0, 0.0, 0.0], [0.0, 1.2, 1.2], [0.0, 1.2, 2.4]]),
        reference_z=100.0,
    )

    plat = np.concatenate(_patch_faces(patch))
    releve = np.concatenate(_patch_faces(patch, grid))

    assert np.allclose(plat[:, 2], 0.0)
    assert releve[:, 2].max() > 0.5
