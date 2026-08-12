"""Grille commune et écriture GeoTIFF (Lot 1B §9).

Deux conventions coexistent et se confondent facilement.

Les calculs sur nuage de points raisonnent naturellement en `(colonne, ligne)`
avec Y croissant vers le nord — c'est l'ordre des coordonnées projetées. Les
rasters, eux, s'écrivent en `(ligne, colonne)` avec la **première ligne au
nord**, donc Y décroissant.

Les confondre produit un GeoTIFF parfaitement valide et géographiquement
retourné : aucune erreur, aucun avertissement, et un toit qui se retrouve au
sud du bâtiment. La conversion est donc faite en un seul endroit, et vérifiée
par un test sur un tableau asymétrique dont les quatre coins diffèrent.

Toutes les couches d'un même site partagent une `GridSpec` unique : mêmes
origine, dimensions, résolution et référentiel. Sans cela, une soustraction
cellule à cellule fabrique des hauteurs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..logging import get_logger

log = get_logger("raster")

#: Valeur sans donnée des rasters d'altitude.
NODATA = -9999.0


@dataclass(frozen=True)
class GridSpec:
    """Grille partagée par toutes les couches d'un site.

    L'origine désigne le coin **sud-ouest**, comme les coordonnées projetées.
    La transformation raster, elle, part du coin nord-ouest : la conversion est
    faite ici plutôt que chez chaque appelant.
    """

    origin_x: float
    origin_y: float
    cell_m: float
    width: int   # colonnes, axe X
    height: int  # lignes, axe Y
    crs: str

    @property
    def north(self) -> float:
        return self.origin_y + self.height * self.cell_m

    @property
    def east(self) -> float:
        return self.origin_x + self.width * self.cell_m

    def transform(self):  # noqa: ANN201
        from rasterio.transform import from_origin

        return from_origin(self.origin_x, self.north, self.cell_m, self.cell_m)

    def cell_centres_xy(self):
        """Centres de cellules en convention `(colonne, ligne)`, Y vers le nord."""
        xs = self.origin_x + (np.arange(self.width) + 0.5) * self.cell_m
        ys = self.origin_y + (np.arange(self.height) + 0.5) * self.cell_m
        return np.meshgrid(xs, ys, indexing="ij")

    def matches(self, other: "GridSpec") -> bool:
        return (
            self.origin_x == other.origin_x
            and self.origin_y == other.origin_y
            and self.cell_m == other.cell_m
            and self.width == other.width
            and self.height == other.height
            and self.crs == other.crs
        )

    def as_dict(self) -> dict:
        return {
            "origin_x": self.origin_x,
            "origin_y": self.origin_y,
            "cell_m": self.cell_m,
            "width": self.width,
            "height": self.height,
            "crs": self.crs,
        }


def to_raster(array_xy: np.ndarray) -> np.ndarray:
    """`(colonne, ligne)` Y-nord → `(ligne, colonne)` première ligne au nord.

    Transposition **puis** inversion verticale. Omettre l'une des deux produit
    un raster valide et faux.
    """
    return np.flipud(array_xy.T)


def from_raster(array_rowcol: np.ndarray) -> np.ndarray:
    """Conversion inverse, pour relire un raster dans la convention de calcul."""
    return np.flipud(array_rowcol).T


def _check_shape(array_xy: np.ndarray, grid: GridSpec, label: str) -> None:
    """Refuse une couche qui n'est pas à la forme de la grille.

    Sans ce contrôle, une couche mal dimensionnée s'écrit quand même — après
    transposition elle prend l'allure d'un raster valide, et le décalage ne se
    voit qu'en superposant les couches.
    """
    expected = (grid.width, grid.height)
    actual = tuple(np.shape(array_xy))
    if actual != expected:
        raise ValueError(
            f"{label} : forme {actual} au lieu de {expected} — toutes les "
            "couches d'un site partagent la même grille"
        )


def write_geotiff(
    path: Path,
    array_xy: np.ndarray,
    grid: GridSpec,
    nodata: float = NODATA,
    dtype: str = "float32",
) -> Path:
    """Écrit une couche, en convertissant NaN vers la valeur sans donnée.

    La conversion n'intervient qu'à la sérialisation : les calculs raisonnent
    en NaN, seul état qui se propage correctement dans une soustraction.
    """
    import rasterio

    _check_shape(array_xy, grid, path.name)
    data = to_raster(np.asarray(array_xy, dtype=np.float64))
    if np.issubdtype(np.dtype(dtype), np.floating):
        data = np.where(np.isnan(data), nodata, data).astype(dtype)
    else:
        data = np.nan_to_num(data, nan=0).astype(dtype)

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")

    with rasterio.open(
        partial, "w", driver="GTiff", height=grid.height, width=grid.width,
        count=1, dtype=dtype, crs=grid.crs, transform=grid.transform(),
        nodata=nodata, compress="deflate", tiled=True,
    ) as destination:
        destination.write(data, 1)

    partial.replace(path)
    log.info("écrit %s (%d × %d, %s)", path.name, grid.width, grid.height, grid.crs)
    return path


def write_mask(path: Path, mask_xy: np.ndarray, grid: GridSpec) -> Path:
    """Écrit un masque booléen en entier 8 bits, sans valeur sans donnée."""
    import rasterio

    _check_shape(mask_xy, grid, path.name)
    data = to_raster(np.asarray(mask_xy)).astype("uint8")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")

    with rasterio.open(
        partial, "w", driver="GTiff", height=grid.height, width=grid.width,
        count=1, dtype="uint8", crs=grid.crs, transform=grid.transform(),
        compress="deflate", tiled=True,
    ) as destination:
        destination.write(data, 1)

    partial.replace(path)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalised_height(dsm_xy: np.ndarray, dtm_xy: np.ndarray, valid_xy: np.ndarray) -> np.ndarray:
    """nDSM strict : une hauteur n'existe que là où les deux surfaces existent.

    Aucun zéro, aucune valeur de remplissage, aucun `nan_to_num` : soustraire
    contre une altitude absente ou substituée ferait entrer l'extrapolation par
    la porte de derrière — non en interpolant, mais en soustrayant.
    """
    dsm = np.asarray(dsm_xy, dtype=np.float64)
    dtm = np.asarray(dtm_xy, dtype=np.float64)
    mask = np.asarray(valid_xy, dtype=bool)

    # Le broadcasting de NumPy accepterait des formes différentes et
    # produirait une surface plausible mais fausse.
    if not (dsm.shape == dtm.shape == mask.shape):
        raise ValueError(
            f"formes incompatibles : DSM {dsm.shape}, DTM {dtm.shape}, "
            f"masque {mask.shape}"
        )

    valid = mask & np.isfinite(dsm) & np.isfinite(dtm)

    height = np.full(dsm.shape, np.nan, dtype=np.float64)
    np.subtract(dsm, dtm, out=height, where=valid)
    height[~valid] = np.nan
    return height
