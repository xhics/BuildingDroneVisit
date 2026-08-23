"""Relief du terrain, lu dans le modèle numérique de terrain.

Le sol était posé à plat, à l'altitude zéro. Un stationnement en pente douce,
un talus de bordure, une allée qui descend vers l'entrée : rien de tout cela
n'apparaissait, et les volumes bâtis flottaient sur un plan idéal.

L'amplitude reste modeste — un peu plus d'un mètre sur ce pilote — mais elle
est mesurée, et c'est justement ce qui distingue un terrain d'un plan de
référence. Sur une vue rasante, un mètre de dénivelé se voit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..logging import get_logger

log = get_logger("conditioning-terrain")

#: En deçà, le terrain est plat à l'échelle du rendu : le porter en géométrie
#: ajouterait des sommets sans rien montrer.
MIN_RELIEF_M = 0.3

#: Pas d'échantillonnage du terrain, en mètres. Le relief d'un site est doux :
#: une maille large suffit et garde le maillage léger.
TERRAIN_STEP_M = 4.0


@dataclass
class TerrainGrid:
    """Altitudes du terrain, en CRS projeté, rapportées au sol médian."""

    x0: float
    y0: float
    step_m: float
    #: Hauteurs relatives au sol de référence, (lignes, colonnes).
    heights: np.ndarray
    reference_z: float
    provenance: dict = field(default_factory=dict)

    @property
    def relief_m(self) -> float:
        finite = self.heights[np.isfinite(self.heights)]
        return float(finite.max() - finite.min()) if finite.size else 0.0

    def height_at(self, x: float, y: float) -> float:
        """Altitude relative en un point, par plus proche cellule."""
        col = int(round((x - self.x0) / self.step_m))
        row = int(round((y - self.y0) / self.step_m))
        if not (0 <= row < self.heights.shape[0] and 0 <= col < self.heights.shape[1]):
            return 0.0
        value = self.heights[row, col]
        return 0.0 if not np.isfinite(value) else float(value)

    def as_dict(self) -> dict:
        return {
            "step_m": self.step_m,
            "shape": list(self.heights.shape),
            "reference_z": round(self.reference_z, 2),
            "relief_m": round(self.relief_m, 2),
            "provenance": self.provenance,
            "caveats": [
                "le relief vient du modèle de terrain dérivé du LiDAR : il "
                "décrit le sol nu, non les aménagements posés dessus",
                "un terrain plat à l'échelle du relevé reste rendu plat : "
                "aucune ondulation n'est ajoutée pour faire vivant",
            ],
        }


def load(
    dtm_path: Path,
    centre: tuple[float, float],
    radius_m: float = 160.0,
    step_m: float = TERRAIN_STEP_M,
) -> TerrainGrid | None:
    """Lit le relief autour d'un site, rapporté à son altitude médiane."""
    try:
        import rasterio
        from rasterio.windows import from_bounds
    except ImportError as exc:  # pragma: no cover - dépend de l'installation
        log.info("rasterio indisponible (%s) : terrain plat", exc)
        return None

    dtm_path = Path(dtm_path)
    if not dtm_path.is_file():
        log.info("modèle de terrain absent : %s", dtm_path)
        return None

    cx, cy = centre
    with rasterio.open(dtm_path) as src:
        nodata = src.nodata
        try:
            window = from_bounds(
                cx - radius_m, cy - radius_m, cx + radius_m, cy + radius_m,
                src.transform,
            )
            band = src.read(1, window=window, boundless=True, fill_value=nodata)
            transform = src.window_transform(window)
        except (ValueError, rasterio.errors.RasterioError) as exc:
            log.info("lecture du terrain impossible : %s", exc)
            return None

    valid = np.isfinite(band)
    if nodata is not None:
        valid &= band != nodata
    if valid.sum() < 16:
        log.info("modèle de terrain trop lacunaire dans l'emprise")
        return None

    reference = float(np.median(band[valid]))
    relative = np.where(valid, band - reference, np.nan)

    # Rééchantillonnage à la maille voulue : le relief est doux, une cellule
    # de quatre mètres le décrit sans multiplier les sommets.
    scale = max(int(round(step_m / abs(transform.a))), 1)
    if scale > 1:
        rows = (relative.shape[0] // scale) * scale
        cols = (relative.shape[1] // scale) * scale
        blocks = relative[:rows, :cols].reshape(
            rows // scale, scale, cols // scale, scale
        )
        # Un bloc entièrement hors emprise n'a aucune valeur : la médiane y est
        # indéfinie, et c'est le résultat voulu — pas une anomalie à signaler.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            relative = np.nanmedian(blocks, axis=(1, 3))
        origin_x = transform.c + scale * abs(transform.a) * 0.5
        origin_y = transform.f - scale * abs(transform.e) * 0.5
    else:
        origin_x, origin_y = transform.c, transform.f

    # Les lignes d'un raster descendent : on retourne pour que la ligne zéro
    # soit au sud, comme le repère projeté.
    relative = relative[::-1, :]
    origin_y -= abs(transform.e) * scale * (relative.shape[0] - 1)

    grid = TerrainGrid(
        x0=float(origin_x),
        y0=float(origin_y),
        step_m=float(scale * abs(transform.a)),
        heights=relative,
        reference_z=reference,
        provenance={"dtm": str(dtm_path), "radius_m": radius_m},
    )

    if grid.relief_m < MIN_RELIEF_M:
        log.info("terrain plat (%.2f m) : relief non porté", grid.relief_m)
        return None

    log.info(
        "terrain : %s cellules de %.0f m, relief %.2f m",
        "×".join(str(v) for v in grid.heights.shape),
        grid.step_m,
        grid.relief_m,
    )
    return grid


def find_dtm(workspace) -> Path | None:  # noqa: ANN001
    """Cherche le modèle de terrain qualifié le plus récent."""
    derived = workspace.path("06_geo", "derived")
    if not derived.is_dir():
        return None
    found = sorted(
        (p for p in derived.glob("*/dtm.tif") if "SUPERSEDED" not in str(p)),
        key=lambda p: p.parent.name,
    )
    return found[-1] if found else None
