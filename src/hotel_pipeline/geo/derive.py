"""Orchestration des dérivations géospatiales (Lot 1B §9).

Produit les rasters et leur rapport. **Ne qualifie aucun objet** : un GeoTIFF
produit n'est pas encore une géométrie qualifiée, et la décision de faire
passer `TERRAIN_MAIN` ou `ROOFLINE_MAIN` en `inferred` appartient à une étape
distincte, au vu des métriques.

Le déroulé suit un ordre qui protège le manifeste : tout est écrit dans un
répertoire de transit, relu et contrôlé — dimensions, référentiel,
transformation, valeur sans donnée, valeurs elles-mêmes — puis seulement
empreint, assemblé et publié. Un manifeste ne doit jamais référencer un fichier
qu'on n'a pas rouvert.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..logging import get_logger
from ..schemas.site import DerivedArtifact
from . import terrain
from .preflight import BUILDING, GROUND, UNCLASSIFIED
from .raster import (
    NODATA,
    GridSpec,
    normalised_height,
    sha256_file,
    write_geotiff,
    write_mask,
)

log = get_logger("derive")

ALGORITHM = "tin-linear-idw-diagnostic-v1"

#: Couches produites, dans l'ordre de dépendance.
LAYER_ROLES: dict[str, str] = {
    "dtm": "dtm",
    "both_defined": "mask",
    "tin_only": "mask",
    "idw_only": "mask",
    "neither_defined": "mask",
    "distance_to_support": "distance",
    "dsm_roof_class6": "dsm_roof",
    "class1_candidates": "unclassified_roof_candidates",
    "ndsm_valid": "mask",
    "ndsm": "ndsm",
}


@dataclass
class DeriveResult:
    grid: dict = field(default_factory=dict)
    layers: dict[str, str] = field(default_factory=dict)
    artifacts: list[DerivedArtifact] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    qa_flags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "grid": self.grid,
            "layers": self.layers,
            "artifacts": [a.model_dump(mode="json") for a in self.artifacts],
            "metrics": self.metrics,
            "qa_flags": self.qa_flags,
        }


# --- lecture du nuage ------------------------------------------------------


def _load_points(laz_path: Path, footprint, ring_m: float):  # noqa: ANN001
    """Points utiles, préfiltrés par boîte puis découpés au polygone.

    Le préfiltrage par boîte évite de tester vingt millions de points contre un
    polygone ; le découpage réel se fait ensuite, jamais sur la boîte seule.
    """
    import laspy
    from shapely import contains_xy

    with laspy.open(str(laz_path)) as reader:
        points = reader.read()

    x = np.asarray(points.x)
    y = np.asarray(points.y)
    z = np.asarray(points.z)
    codes = np.asarray(points.classification)

    minx, miny, maxx, maxy = footprint.bounds
    window = (
        (x >= minx - ring_m) & (x <= maxx + ring_m)
        & (y >= miny - ring_m) & (y <= maxy + ring_m)
    )
    x, y, z, codes = x[window], y[window], z[window], codes[window]

    inside = contains_xy(footprint, x, y)
    return x, y, z, codes, inside


# --- production ------------------------------------------------------------


def derive(
    laz_path: Path,
    footprint_projected,  # noqa: ANN001 — polygone Shapely en CRS de la tuile
    staging: Path,
    crs: str,
    crs_vertical: str,
    source_id: str,
    cell_m: float = terrain.CELL_M,
    ring_m: float = terrain.RING_M,
) -> DeriveResult:
    """Produit toutes les couches dans le répertoire de transit."""
    from shapely import contains_xy

    result = DeriveResult()
    minx, miny, maxx, maxy = footprint_projected.bounds
    origin = terrain.aligned_origin(minx, miny, cell_m)
    grid = GridSpec(
        origin_x=origin[0],
        origin_y=origin[1],
        cell_m=cell_m,
        width=int(np.ceil((maxx - origin[0]) / cell_m)),
        height=int(np.ceil((maxy - origin[1]) / cell_m)),
        crs=crs,
    )
    result.grid = grid.as_dict()

    gx, gy = grid.cell_centres_xy()
    footprint_mask = contains_xy(footprint_projected, gx, gy)
    domain_cells = int(footprint_mask.sum())

    x, y, z, codes, inside = _load_points(laz_path, footprint_projected, ring_m)

    # --- appuis de terrain : classe 2 du pourtour uniquement --------------
    ring = (codes == GROUND) & ~inside
    px, py, pz = terrain.aggregate_median(
        x[ring], y[ring], z[ring], origin, (grid.width, grid.height), cell_m
    )
    log.info("appuis de terrain : %d cellule(s) de classe 2 au pourtour", pz.size)

    flat_x, flat_y = gx.ravel(), gy.ravel()
    tin = terrain.interpolate_tin(px, py, pz, flat_x, flat_y).reshape(gx.shape)
    idw = terrain.interpolate_idw(px, py, pz, flat_x, flat_y).reshape(gx.shape)
    distances = terrain.support_distance(px, py, flat_x, flat_y).reshape(gx.shape)

    masks = terrain.definition_masks(tin, idw, footprint_mask)
    disagreement = terrain.compare_models(tin, idw, masks, domain_cells)
    rejected = terrain.rejected_extrapolation(masks, distances, domain_cells)

    # Le DTM de production est le TIN, restreint à l'empreinte. L'IDW ne
    # remplit jamais ce que le TIN refuse.
    dtm = np.where(masks.both_defined | masks.tin_only, tin, np.nan)

    # --- toiture : classe 6 dans l'empreinte -------------------------------
    roof = (codes == BUILDING) & inside
    dsm_roof = _median_grid(x[roof], y[roof], z[roof], grid)

    # --- candidats classe 1, séparés du toit -------------------------------
    roof_median = float(np.nanmedian(z[roof])) if roof.any() else np.nan
    high = (codes == UNCLASSIFIED) & inside & (z > roof_median)
    class1 = _median_grid(x[high], y[high], z[high], grid)

    # --- nDSM strict --------------------------------------------------------
    ndsm_valid = footprint_mask & np.isfinite(dtm) & np.isfinite(dsm_roof)
    ndsm = normalised_height(dsm_roof, dtm, ndsm_valid)

    layers = {
        "dtm": dtm,
        "both_defined": masks.both_defined,
        "tin_only": masks.tin_only,
        "idw_only": masks.idw_only,
        "neither_defined": masks.neither_defined,
        "distance_to_support": np.where(footprint_mask, distances, np.nan),
        "dsm_roof_class6": dsm_roof,
        "class1_candidates": class1,
        "ndsm_valid": ndsm_valid,
        "ndsm": ndsm,
    }

    staging.mkdir(parents=True, exist_ok=True)
    for name, array in layers.items():
        path = staging / f"{name}.tif"
        if array.dtype == bool:
            write_mask(path, array, grid)
        else:
            write_geotiff(path, array, grid)
        result.layers[name] = str(path)

    result.metrics = _metrics(
        grid, footprint_mask, dtm, dsm_roof, class1, ndsm, ndsm_valid,
        masks, disagreement, rejected, distances, px, py, pz, footprint_projected,
    )
    result.qa_flags = _qa_flags(result.metrics, ndsm)
    result.artifacts = _artifacts(
        result, grid, crs, crs_vertical, source_id, domain_cells, masks, ndsm_valid
    )
    return result


def _median_grid(x, y, z, grid: GridSpec) -> np.ndarray:  # noqa: ANN001
    """Médiane par cellule, sur la grille commune. NaN là où rien n'est observé."""
    values = np.full((grid.width, grid.height), np.nan)
    if x.size == 0:
        return values

    cx, cy, medians = terrain.aggregate_median(
        x, y, z, (grid.origin_x, grid.origin_y), (grid.width, grid.height), grid.cell_m
    )
    ix = ((cx - grid.origin_x) / grid.cell_m).astype(int)
    iy = ((cy - grid.origin_y) / grid.cell_m).astype(int)
    keep = (ix >= 0) & (ix < grid.width) & (iy >= 0) & (iy < grid.height)
    values[ix[keep], iy[keep]] = medians[keep]
    return values


# --- métriques et QA -------------------------------------------------------


def _metrics(
    grid, footprint_mask, dtm, dsm_roof, class1, ndsm, ndsm_valid,
    masks, disagreement, rejected, distances, px, py, pz, footprint,  # noqa: ANN001
) -> dict:
    domain = int(footprint_mask.sum())
    interior_distances = distances[footprint_mask]

    block = terrain.block_cross_validation(px, py, pz)
    width = footprint.bounds[2] - footprint.bounds[0]
    height = footprint.bounds[3] - footprint.bounds[1]
    pseudo = terrain.pseudo_building_validation(px, py, pz, width, height)

    heights = ndsm[np.isfinite(ndsm)]
    return {
        "footprint_cells": domain,
        "coverage": {
            "dtm_defined": _fraction(np.isfinite(dtm) & footprint_mask, domain),
            "roof_observed": _fraction(np.isfinite(dsm_roof) & footprint_mask, domain),
            "class1_candidates": _fraction(np.isfinite(class1) & footprint_mask, domain),
            "ndsm_valid": _fraction(ndsm_valid, domain),
        },
        "definition_masks": masks.counts(),
        "tin_vs_idw": disagreement.as_dict(),
        "extrapolation_rejected": rejected.as_dict(),
        "support_distance_in_footprint": {
            "p50_m": round(float(np.percentile(interior_distances, 50)), 2),
            "p95_m": round(float(np.percentile(interior_distances, 95)), 2),
            "max_m": round(float(interior_distances.max()), 2),
        },
        "block_validation": block.as_dict(),
        "pseudo_building_validation": pseudo,
        "height_statistics": {
            "count": int(heights.size),
            "min_m": round(float(heights.min()), 2) if heights.size else None,
            "median_m": round(float(np.median(heights)), 2) if heights.size else None,
            "p95_m": round(float(np.percentile(heights, 95)), 2) if heights.size else None,
            "max_m": round(float(heights.max()), 2) if heights.size else None,
            "negative_cells": int((heights < 0).sum()),
        },
    }


def _fraction(mask, domain: int) -> float:  # noqa: ANN001
    return round(int(np.asarray(mask).sum()) / max(domain, 1), 4)


def _qa_flags(metrics: dict, ndsm) -> list[str]:  # noqa: ANN001
    """Signale sans corriger. Une hauteur aberrante reste dans le raster."""
    flags: list[str] = []
    heights = metrics["height_statistics"]

    if heights["negative_cells"]:
        flags.append(
            f"{heights['negative_cells']} cellule(s) de hauteur négative — "
            "conservées telles quelles, jamais écrêtées"
        )
    if heights["max_m"] is not None and heights["max_m"] > 100:
        flags.append(f"hauteur maximale de {heights['max_m']} m — invraisemblable")
    if metrics["extrapolation_rejected"]["cells"]:
        flags.append(
            f"{metrics['extrapolation_rejected']['cells']} cellule(s) hors "
            "enveloppe convexe, laissées sans donnée"
        )
    if metrics["definition_masks"]["tin_only"]:
        flags.append(
            f"{metrics['definition_masks']['tin_only']} cellule(s) où le TIN "
            "répond sans l'IDW — anomalie, le TIN est le plus restrictif"
        )
    return flags


# --- artefacts -------------------------------------------------------------


def _artifacts(
    result: DeriveResult, grid: GridSpec, crs: str, crs_vertical: str,
    source_id: str, domain_cells: int, masks, ndsm_valid,  # noqa: ANN001
) -> list[DerivedArtifact]:
    """Assemble les artefacts, avec leur filiation réelle."""
    parents = {
        "ndsm_valid": ["dtm", "dsm_roof_class6"],
        "ndsm": ["dtm", "dsm_roof_class6", "ndsm_valid"],
        "both_defined": ["dtm"],
        "tin_only": ["dtm"],
        "idw_only": ["dtm"],
        "neither_defined": ["dtm"],
    }
    coverage = result.metrics["coverage"]
    fractions = {
        "dtm": (0.0, coverage["dtm_defined"]),
        "dsm_roof_class6": (coverage["roof_observed"], 0.0),
        "class1_candidates": (coverage["class1_candidates"], 0.0),
        # Chaque hauteur dépend du terrain interpolé : rien n'y est mesuré.
        "ndsm": (0.0, coverage["ndsm_valid"]),
    }

    artifacts: list[DerivedArtifact] = []
    for name, role in LAYER_ROLES.items():
        path = Path(result.layers[name])
        measured, interpolated = fractions.get(name, (1.0, 0.0))
        artifacts.append(
            DerivedArtifact(
                artifact_id=name,
                role=role,
                path=str(path),
                format="GeoTIFF",
                sha256=sha256_file(path),
                crs_horizontal=crs,
                crs_vertical=crs_vertical if role in _ELEVATION else None,
                resolution_m=grid.cell_m,
                nodata=NODATA if role != "mask" else None,
                algorithm_id=ALGORITHM,
                parameters={
                    "cell_m": str(grid.cell_m),
                    "ring_m": str(terrain.RING_M),
                    "aggregation": "median",
                    "interpolation": "tin_linear",
                },
                measured_fraction=measured,
                interpolated_fraction=interpolated,
                coverage_domain="footprint",
                derived_from_sources=[source_id],
                derived_from_artifacts=parents.get(name, []),
                produced_at=datetime.now(timezone.utc),
            )
        )
    return artifacts


#: Rôles dont les valeurs sont des altitudes ou des différences d'altitude.
_ELEVATION = {"dtm", "dsm_roof", "ndsm", "unclassified_roof_candidates"}


def verify_written(result: DeriveResult, grid: GridSpec) -> list[str]:
    """Rouvre chaque couche et confronte son en-tête à la grille.

    Un manifeste ne doit jamais référencer un fichier qu'on n'a pas rouvert :
    l'écriture peut réussir et produire un raster décalé ou mal projeté.
    """
    import rasterio

    problems: list[str] = []
    for name, path in result.layers.items():
        with rasterio.open(path) as source:
            if (source.width, source.height) != (grid.width, grid.height):
                problems.append(
                    f"{name} : {source.width}×{source.height} au lieu de "
                    f"{grid.width}×{grid.height}"
                )
            if source.crs is None or source.crs.to_string() != grid.crs:
                problems.append(f"{name} : référentiel {source.crs} au lieu de {grid.crs}")
            if not np.allclose(
                [source.transform.c, source.transform.f, source.transform.a],
                [grid.origin_x, grid.north, grid.cell_m],
            ):
                problems.append(f"{name} : transformation incohérente")
    return problems
