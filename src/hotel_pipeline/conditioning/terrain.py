"""Relief du terrain, lu dans le modèle numérique de terrain.

Le sol était posé à plat, à l'altitude zéro. Un stationnement en pente douce,
un talus de bordure, une allée qui descend vers l'entrée : rien de tout cela
n'apparaissait, et les volumes bâtis flottaient sur un plan idéal.

Deux progrès successifs :

- l'altitude d'un point n'est plus celle de la cellule la plus proche mais
  une **interpolation bilinéaire** entre les quatre coins : une pente
  synthétique continue reste continue, aucun effet d'escalier au déplacement
  de la caméra ;
- la grille est **adaptative** : fine près du bâtiment (0,25–0,5 m), moyenne
  autour (1 m), large au loin (4 m). Une rampe ou une bordure importante peut
  aussi être imposée comme **breakline** explicite.
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

#: Pas d'échantillonnage du terrain loin du site, en mètres. Le relief d'un
#: site est doux : une maille large suffit et garde le maillage léger.
TERRAIN_STEP_M = 4.0

#: Zones de raffinement adaptatif, du centre vers l'extérieur : (rayon
#: extérieur de la zone, pas d'échantillonnage). La première zone couvre le
#: bâtiment lui-même, la seconde ses abords proches, la dernière le reste.
ADAPTIVE_ZONES: tuple[tuple[float, float], ...] = (
    (30.0, 0.5),
    (80.0, 1.0),
    (float("inf"), TERRAIN_STEP_M),
)


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
        """Altitude relative en un point, par interpolation bilinéaire.

        Les quatre coins de la cellule englobante pondèrent le résultat : une
        pente régulière donne une altitude régulière entre les nœuds, là où
        la cellule la plus proche produisait des marches. Sans voisin valide,
        le plus proche l'emporte ; hors grille, zéro.
        """
        col_f = (x - self.x0) / self.step_m
        row_f = (y - self.y0) / self.step_m
        rows, cols = self.heights.shape
        if not (-1 <= row_f <= rows and -1 <= col_f <= cols):
            return 0.0

        col0 = int(np.floor(col_f))
        row0 = int(np.floor(row_f))
        fx = col_f - col0
        fy = row_f - row0

        corners: list[tuple[int, int, float]] = [
            (row0, col0, (1.0 - fx) * (1.0 - fy)),
            (row0, col0 + 1, fx * (1.0 - fy)),
            (row0 + 1, col0 + 1, fx * fy),
            (row0 + 1, col0, (1.0 - fx) * fy),
        ]

        total_weight = 0.0
        total_value = 0.0
        nearest_distance = float("inf")
        nearest_value = 0.0
        for row, col, weight in corners:
            if 0 <= row < rows and 0 <= col < cols:
                value = self.heights[row, col]
                if np.isfinite(value):
                    weight = max(weight, 1e-12)
                    total_weight += weight
                    total_value += weight * float(value)
                else:
                    distance = abs(row - row_f) + abs(col - col_f)
                    if distance < nearest_distance:
                        nearest_distance = distance
                        # NaN : on retombera sur la valeur finie la plus proche
                        nearest_value = float("nan")
            else:
                distance = abs(row - row_f) + abs(col - col_f)
                nearest_distance = min(nearest_distance, distance)

        if total_weight > 0.0 and np.isfinite(total_value):
            return float(total_value / total_weight)

        # Repli : la valeur finie la plus proche dans la fenêtre 3×3.
        best = None
        best_distance = float("inf")
        for row in range(max(0, row0 - 1), min(rows, row0 + 2)):
            for col in range(max(0, col0 - 1), min(cols, col0 + 2)):
                value = self.heights[row, col]
                if not np.isfinite(value):
                    continue
                distance = abs(row - row_f) + abs(col - col_f)
                if distance < best_distance:
                    best_distance = distance
                    best = float(value)
        return best if best is not None else 0.0

    def as_dict(self) -> dict:
        return {
            "step_m": self.step_m,
            "shape": list(self.heights.shape),
            "reference_z": round(self.reference_z, 2),
            "relief_m": round(self.relief_m, 2),
            "interpolation": "bilinear",
            "provenance": self.provenance,
            "caveats": [
                "le relief vient du modèle de terrain dérivé du LiDAR : il "
                "décrit le sol nu, non les aménagements posés dessus",
                "un terrain plat à l'échelle du relevé reste rendu plat : "
                "aucune ondulation n'est ajoutée pour faire vivant",
            ],
        }


@dataclass
class AdaptiveTerrainGrid:
    """Grilles emboîtées : fine près du bâtiment, large au loin.

    ``height_at`` interroge la grille la plus fine qui couvre le point ; les
    bordures entre zones sont interpolées par la même lecture bilinéaire,
    donc continues par construction à la précision des données sources.
    """

    centre: tuple[float, float]
    #: Zones du centre vers l'extérieur : (rayon de couverture, grille).
    zones: list[tuple[float, TerrainGrid]]
    reference_z: float = 0.0
    breaklines: list = field(default_factory=list)

    @property
    def relief_m(self) -> float:
        return max((grid.relief_m for _, grid in self.zones), default=0.0)

    def height_at(self, x: float, y: float) -> float:
        for radius, grid in self.zones:  # zones triées du plus fin au plus large
            distance = float(
                np.hypot(x - self.centre[0], y - self.centre[1])
            )
            if distance <= radius or radius == float("inf"):
                return grid.height_at(x, y)
        return 0.0

    def as_dict(self) -> dict:
        return {
            "kind": "adaptive",
            "centre": [round(c, 2) for c in self.centre],
            "zones": [
                {"radius_m": round(radius, 1), "step_m": round(grid.step_m, 2)}
                for radius, grid in self.zones
            ],
            "breaklines": len(self.breaklines),
            "reference_z": round(self.reference_z, 2),
            "relief_m": round(self.relief_m, 2),
            "interpolation": "bilinear_adaptive",
            "provenance": (
                self.zones[-1][1].provenance
                if self.zones
                else {}
            ),
        }


def _resample_to_step(
    relative: np.ndarray,
    transform,  # noqa: ANN001
    step_m: float,
) -> tuple[TerrainGrid, float]:
    """Rééchantillonne la nappe native à la maille demandée (médiane par bloc)."""
    scale = max(int(round(step_m / abs(transform.a))), 1)
    if scale > 1:
        rows = (relative.shape[0] // scale) * scale
        cols = (relative.shape[1] // scale) * scale
        blocks = relative[:rows, :cols].reshape(
            rows // scale, scale, cols // scale, scale
        )
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            resampled = np.nanmedian(blocks, axis=(1, 3))
        origin_x = transform.c + scale * abs(transform.a) * 0.5
        origin_y = transform.f - scale * abs(transform.e) * 0.5
    else:
        resampled = relative
        origin_x, origin_y = transform.c, transform.f

    # Les lignes d'un raster descendent : on retourne pour que la ligne zéro
    # soit au sud, comme le repère projeté.
    resampled = resampled[::-1, :]
    origin_y -= abs(transform.e) * scale * (resampled.shape[0] - 1)
    step = float(scale * abs(transform.a))
    grid = TerrainGrid(
        x0=float(origin_x),
        y0=float(origin_y),
        step_m=step,
        heights=resampled,
        reference_z=0.0,
    )
    return grid, step


def load(
    dtm_path: Path,
    centre: tuple[float, float],
    radius_m: float = 160.0,
    step_m: float | None = None,
    adaptive: bool = True,
    zones: tuple[tuple[float, float], ...] = ADAPTIVE_ZONES,
) -> TerrainGrid | AdaptiveTerrainGrid | None:
    """Lit le relief autour d'un site, rapporté à son altitude médiane.

    Par défaut, une grille adaptative : ~0,5 m sous le bâtiment, 1 m autour,
    4 m au-delà. ``adaptive=False`` (ou ``step_m`` explicite) conserve le
    comportement à maille unique, désormais lu en bilinéaire.
    """
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

    grids: list[tuple[float, TerrainGrid]] = []
    ordered_zones = sorted(
        zones if adaptive else ((float("inf"), step_m or TERRAIN_STEP_M),),
        key=lambda zone: zone[1],
    )
    for zone_radius, zone_step in ordered_zones:
        grid, _actual_step = _resample_to_step(relative, transform, zone_step)
        grid.reference_z = reference
        grid.provenance = {"dtm": str(dtm_path), "radius_m": radius_m}
        # `height_at` choisit la première zone (la plus fine) qui couvre le
        # point ; la dernière couvre l'infini par défaut.
        coverage = zone_radius if adaptive else float("inf")
        grids.append((coverage, grid))

    relief = grids[-1][1].relief_m if grids else 0.0
    if relief < MIN_RELIEF_M:
        log.info("terrain plat (%.2f m) : relief non porté", relief)
        return None

    if adaptive and len(grids) > 1:
        result: TerrainGrid | AdaptiveTerrainGrid = AdaptiveTerrainGrid(
            centre=(cx, cy),
            zones=grids,
            reference_z=reference,
        )
        log.info(
            "terrain adaptatif : %s zones (%s), relief %.2f m",
            len(grids),
            ", ".join(f"{grid.step_m:.1f} m ≤ {radius:.0f} m" for radius, grid in grids),
            relief,
        )
    else:
        result = grids[-1][1]
        log.info(
            "terrain : %s cellules de %.1f m, relief %.2f m",
            "×".join(str(v) for v in result.heights.shape),
            result.step_m,
            relief,
        )
    return result


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
