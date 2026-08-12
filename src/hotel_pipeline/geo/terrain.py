"""Terrain sous l'empreinte — interpolation et mesure de sa fiabilité.

Sous un bâtiment, il n'y a pas de sol : 0,8 point par mètre carré ici, contre
208 410 points au pourtour. Le terrain intérieur ne peut donc qu'être
**interpolé depuis son pourtour**, et l'enjeu n'est pas de produire une
surface — c'est de dire à quel point elle est fiable.

Trois mesures, parce qu'aucune ne suffit seule.

1. **Validation par blocs** sur le sol observé. Blocs spatiaux larges, sinon
   des points voisins presque identiques se retrouvent des deux côtés et
   l'erreur mesurée est fictive.
2. **Distance au support réel**, par cellule. Une cellule au centre d'un
   bâtiment de 40 m est à 15 m du plus proche point de sol : l'erreur de
   validation par blocs, mesurée à quelques mètres d'un appui, ne la décrit
   pas.
3. **Validation par faux bâtiment**. Masquer des polygones comparables dans
   des zones où le sol est connu, interpoler depuis leur seul pourtour, puis
   comparer au sol masqué. C'est la seule mesure qui reproduit la situation
   réelle.

Une cellule hors de l'enveloppe convexe des appuis n'est pas interpolée mais
extrapolée : elle reste `nodata`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..logging import get_logger

log = get_logger("terrain")

#: Pas de la grille commune à tous les produits.
CELL_M = 0.5

#: Anneau de sol autour de l'empreinte, en mètres.
RING_M = 20.0

#: Côté des blocs de validation croisée. Assez large pour que deux points
#: voisins ne se retrouvent pas de part et d'autre.
BLOCK_M = 10.0


@dataclass
class ValidationScores:
    """Erreurs d'une reconstruction confrontée à un sol connu."""

    n: int = 0
    bias: float | None = None
    mae: float | None = None
    rmse: float | None = None
    p95: float | None = None

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "bias_m": self.bias,
            "mae_m": self.mae,
            "rmse_m": self.rmse,
            "p95_m": self.p95,
        }


@dataclass
class SupportDistance:
    """Éloignement des cellules interpolées à leur appui le plus proche."""

    p50: float | None = None
    p95: float | None = None
    max: float | None = None

    def as_dict(self) -> dict:
        return {"p50_m": self.p50, "p95_m": self.p95, "max_m": self.max}


@dataclass
class TerrainResult:
    grid_origin: tuple[float, float] = (0.0, 0.0)
    cell_m: float = CELL_M
    shape: tuple[int, int] = (0, 0)

    measured_cells: int = 0
    interpolated_cells: int = 0
    nodata_cells: int = 0
    outside_hull_cells: int = 0

    block_validation: ValidationScores = field(default_factory=ValidationScores)
    support_distance: SupportDistance = field(default_factory=SupportDistance)
    pseudo_building: list[dict] = field(default_factory=list)
    model_disagreement: SupportDistance = field(default_factory=SupportDistance)

    warnings: list[str] = field(default_factory=list)

    @property
    def domain_cells(self) -> int:
        return self.measured_cells + self.interpolated_cells + self.nodata_cells

    def as_dict(self) -> dict:
        total = max(self.domain_cells, 1)
        return {
            "grid": {
                "origin_x": self.grid_origin[0],
                "origin_y": self.grid_origin[1],
                "cell_m": self.cell_m,
                "shape": list(self.shape),
            },
            "cells": {
                "measured": self.measured_cells,
                "interpolated": self.interpolated_cells,
                "nodata": self.nodata_cells,
                "outside_convex_hull": self.outside_hull_cells,
                "measured_fraction": round(self.measured_cells / total, 4),
                "interpolated_fraction": round(self.interpolated_cells / total, 4),
            },
            "block_validation": self.block_validation.as_dict(),
            "support_distance": self.support_distance.as_dict(),
            "pseudo_building_validation": self.pseudo_building,
            "tin_vs_idw_disagreement": self.model_disagreement.as_dict(),
            "warnings": self.warnings,
        }


# --- grille --------------------------------------------------------------


def aligned_origin(minx: float, miny: float, cell_m: float = CELL_M) -> tuple[float, float]:
    """Origine alignée sur un multiple du pas, fixée une fois pour tous les produits.

    Sans origine commune, terrain et toiture ne se soustraient pas cellule à
    cellule : un décalage d'un demi-pixel suffit à fabriquer des hauteurs.
    """
    return (np.floor(minx / cell_m) * cell_m, np.floor(miny / cell_m) * cell_m)


def cell_centres(origin: tuple[float, float], shape: tuple[int, int], cell_m: float):
    columns, rows = shape
    xs = origin[0] + (np.arange(columns) + 0.5) * cell_m
    ys = origin[1] + (np.arange(rows) + 0.5) * cell_m
    return np.meshgrid(xs, ys, indexing="ij")


def aggregate_median(x, y, z, origin, shape, cell_m: float):  # noqa: ANN001
    """Médiane par cellule, pour neutraliser le biais de densité.

    Une moyenne suivrait les zones sur-échantillonnées ; la médiane non.
    """
    columns, rows = shape
    ix = np.clip(((x - origin[0]) / cell_m).astype(int), 0, columns - 1)
    iy = np.clip(((y - origin[1]) / cell_m).astype(int), 0, rows - 1)
    keys = ix.astype(np.int64) * rows + iy.astype(np.int64)

    order = np.argsort(keys, kind="stable")
    keys_sorted, z_sorted = keys[order], z[order]
    unique_keys, starts = np.unique(keys_sorted, return_index=True)
    ends = np.append(starts[1:], keys_sorted.size)

    values = np.array([np.median(z_sorted[s:e]) for s, e in zip(starts, ends)])
    cx = origin[0] + (unique_keys // rows + 0.5) * cell_m
    cy = origin[1] + (unique_keys % rows + 0.5) * cell_m
    return cx, cy, values


# --- interpolation --------------------------------------------------------


def interpolate_tin(px, py, pz, qx, qy):  # noqa: ANN001
    """TIN linéaire. Rend NaN hors enveloppe convexe — jamais d'extrapolation."""
    from scipy.interpolate import LinearNDInterpolator

    interpolator = LinearNDInterpolator(np.column_stack([px, py]), pz)
    return interpolator(qx, qy)


def interpolate_idw(px, py, pz, qx, qy, k: int = 8, power: float = 2.0):  # noqa: ANN001
    """Pondération inverse à la distance — seconde méthode, pour le désaccord."""
    from scipy.spatial import cKDTree

    tree = cKDTree(np.column_stack([px, py]))
    k = min(k, len(pz))
    distances, indices = tree.query(np.column_stack([qx, qy]), k=k)
    if k == 1:
        distances, indices = distances[:, None], indices[:, None]

    distances = np.maximum(distances, 1e-6)
    weights = 1.0 / distances**power
    return (weights * pz[indices]).sum(axis=1) / weights.sum(axis=1)


def support_distance(px, py, qx, qy) -> np.ndarray:  # noqa: ANN001
    """Distance de chaque cellule au point de sol observé le plus proche."""
    from scipy.spatial import cKDTree

    tree = cKDTree(np.column_stack([px, py]))
    distances, _ = tree.query(np.column_stack([qx, qy]), k=1)
    return distances


# --- validations ----------------------------------------------------------


def _scores(truth, predicted) -> ValidationScores:  # noqa: ANN001
    valid = ~np.isnan(predicted)
    if not valid.any():
        return ValidationScores()
    error = predicted[valid] - truth[valid]
    return ValidationScores(
        n=int(valid.sum()),
        bias=round(float(np.mean(error)), 4),
        mae=round(float(np.mean(np.abs(error))), 4),
        rmse=round(float(np.sqrt(np.mean(error**2))), 4),
        p95=round(float(np.percentile(np.abs(error), 95)), 4),
    )


def block_cross_validation(px, py, pz, block_m: float = BLOCK_M) -> ValidationScores:  # noqa: ANN001
    """Validation croisée par blocs spatiaux.

    Retirer des points isolés donnerait une erreur fictive : leurs voisins
    immédiats restent dans l'échantillon d'apprentissage. Les blocs éliminent
    ce voisinage.
    """
    bx = np.floor(px / block_m).astype(int)
    by = np.floor(py / block_m).astype(int)
    blocks = np.unique(np.column_stack([bx, by]), axis=0)

    truths, predictions = [], []
    for block_x, block_y in blocks:
        held = (bx == block_x) & (by == block_y)
        if held.sum() < 3 or (~held).sum() < 4:
            continue
        predicted = interpolate_tin(px[~held], py[~held], pz[~held], px[held], py[held])
        truths.append(pz[held])
        predictions.append(predicted)

    if not truths:
        return ValidationScores()
    return _scores(np.concatenate(truths), np.concatenate(predictions))


def pseudo_building_validation(
    px, py, pz, width_m: float, height_m: float, ring_m: float = RING_M, trials: int = 5  # noqa: ANN001
) -> list[dict]:
    """Masquer un faux bâtiment là où le sol est connu, puis le reconstruire.

    C'est la seule mesure qui reproduit la situation réelle : un vide continu,
    de dimensions comparables, interpolé depuis son seul pourtour. La
    validation par blocs reste optimiste parce que ses trous sont petits.
    """
    results: list[dict] = []
    if px.size < 4:
        return results

    minx, maxx = float(px.min()), float(px.max())
    miny, maxy = float(py.min()), float(py.max())

    # Un faux bâtiment plus large que la zone de sol connue ne laisserait
    # aucun pourtour : renoncer vaut mieux que produire un essai dégénéré.
    needed_x = width_m + 2 * ring_m
    needed_y = height_m + 2 * ring_m
    if (maxx - minx) < needed_x or (maxy - miny) < needed_y:
        log.info(
            "zone de sol trop petite pour un faux bâtiment de %.0f × %.0f m",
            width_m,
            height_m,
        )
        return results

    rng = np.random.default_rng(seed=1195)
    attempts = 0
    while len(results) < trials and attempts < trials * 6:
        attempts += 1
        cx = rng.uniform(minx + needed_x / 2, maxx - needed_x / 2)
        cy = rng.uniform(miny + needed_y / 2, maxy - needed_y / 2)

        masked = (
            (np.abs(px - cx) <= width_m / 2) & (np.abs(py - cy) <= height_m / 2)
        )
        ring = (
            (np.abs(px - cx) <= width_m / 2 + ring_m)
            & (np.abs(py - cy) <= height_m / 2 + ring_m)
            & ~masked
        )
        if masked.sum() < 30 or ring.sum() < 50:
            continue

        predicted = interpolate_tin(
            px[ring], py[ring], pz[ring], px[masked], py[masked]
        )
        # Même règle que pour la production : l'erreur ne se mesure que là où
        # le TIN reconstruit. Sinon une méthode extrapolant partout serait
        # récompensée d'avoir répondu.
        scores = _scores(pz[masked], predicted)
        unsupported = int(np.isnan(predicted).sum())
        distances = support_distance(px[ring], py[ring], px[masked], py[masked])

        results.append(
            {
                "centre": [round(cx, 1), round(cy, 1)],
                "masked_points": int(masked.sum()),
                "reconstructed_points": int(masked.sum()) - unsupported,
                "unsupported_points": unsupported,
                **scores.as_dict(),
                "support_distance_p95_m": round(float(np.percentile(distances, 95)), 2),
                "support_distance_max_m": round(float(distances.max()), 2),
            }
        )

    log.info("validation par faux bâtiment : %d essai(s) exploitable(s)", len(results))
    return results


# --- domaine de comparaison et masques ------------------------------------


@dataclass
class Disagreement:
    """Écart entre le TIN de production et l'IDW de diagnostic.

    Mesuré **uniquement** là où les deux répondent. Le comparer là où le TIN
    renonce n'aurait aucun sens : l'écart y serait infini par construction, et
    récompenserait la méthode qui extrapole le plus.
    """

    compared_cells: int = 0
    compared_fraction: float = 0.0
    median_signed_bias: float | None = None
    mae: float | None = None
    rmse: float | None = None
    p95_abs: float | None = None
    max_abs: float | None = None

    def as_dict(self) -> dict:
        return {
            "compared_cells": self.compared_cells,
            "compared_fraction": self.compared_fraction,
            "median_signed_bias_m": self.median_signed_bias,
            "mae_m": self.mae,
            "rmse_m": self.rmse,
            "p95_abs_m": self.p95_abs,
            "max_abs_m": self.max_abs,
        }


@dataclass
class ExtrapolationRejected:
    """Cellules que seul l'IDW aurait remplies — et qui restent sans donnée.

    Ce n'est pas un désaccord entre méthodes : c'est une extrapolation refusée.
    Les compter à part empêche de présenter un refus comme une divergence.
    """

    cells: int = 0
    fraction: float = 0.0
    distance_p50: float | None = None
    distance_p95: float | None = None
    distance_max: float | None = None

    def as_dict(self) -> dict:
        return {
            "cells": self.cells,
            "fraction_of_footprint": self.fraction,
            "distance_to_ground_p50_m": self.distance_p50,
            "distance_to_ground_p95_m": self.distance_p95,
            "distance_to_ground_max_m": self.distance_max,
        }


@dataclass
class DefinitionMasks:
    """Quatre masques mutuellement exclusifs, sur la grille du domaine.

    `tin_only` devrait rester vide — le TIN est plus restrictif que l'IDW —
    mais le masque existe pour que l'anomalie soit visible si elle survient,
    plutôt que silencieusement absorbée.
    """

    both_defined: np.ndarray
    tin_only: np.ndarray
    idw_only: np.ndarray
    neither_defined: np.ndarray

    def counts(self) -> dict[str, int]:
        return {
            "both_defined": int(self.both_defined.sum()),
            "tin_only": int(self.tin_only.sum()),
            "idw_only": int(self.idw_only.sum()),
            "neither_defined": int(self.neither_defined.sum()),
        }

    def is_partition_of(self, domain_cells: int) -> bool:
        return sum(self.counts().values()) == domain_cells


def definition_masks(tin_values, idw_values, in_domain) -> DefinitionMasks:  # noqa: ANN001
    """Répartit les cellules du domaine selon les méthodes qui y répondent."""
    tin_ok = np.isfinite(tin_values) & in_domain
    idw_ok = np.isfinite(idw_values) & in_domain
    return DefinitionMasks(
        both_defined=tin_ok & idw_ok,
        tin_only=tin_ok & ~idw_ok,
        idw_only=idw_ok & ~tin_ok,
        neither_defined=in_domain & ~tin_ok & ~idw_ok,
    )


def compare_models(tin_values, idw_values, masks: DefinitionMasks, domain_cells: int) -> Disagreement:  # noqa: ANN001
    """Désaccord TIN/IDW sur le seul domaine commun."""
    both = masks.both_defined
    result = Disagreement(
        compared_cells=int(both.sum()),
        compared_fraction=round(int(both.sum()) / max(domain_cells, 1), 4),
    )
    if result.compared_cells == 0:
        return result

    error = np.asarray(idw_values)[both] - np.asarray(tin_values)[both]
    absolute = np.abs(error)
    result.median_signed_bias = round(float(np.median(error)), 4)
    result.mae = round(float(np.mean(absolute)), 4)
    result.rmse = round(float(np.sqrt(np.mean(error**2))), 4)
    result.p95_abs = round(float(np.percentile(absolute, 95)), 4)
    result.max_abs = round(float(absolute.max()), 4)
    return result


def rejected_extrapolation(
    masks: DefinitionMasks, distances_to_ground, domain_cells: int  # noqa: ANN001
) -> ExtrapolationRejected:
    """Décrit les cellules refusées, avec leur éloignement au sol réel."""
    idw_only = masks.idw_only
    result = ExtrapolationRejected(
        cells=int(idw_only.sum()),
        fraction=round(int(idw_only.sum()) / max(domain_cells, 1), 4),
    )
    if result.cells == 0:
        return result

    distances = np.asarray(distances_to_ground)[idw_only]
    result.distance_p50 = round(float(np.percentile(distances, 50)), 2)
    result.distance_p95 = round(float(np.percentile(distances, 95)), 2)
    result.distance_max = round(float(distances.max()), 2)
    return result
