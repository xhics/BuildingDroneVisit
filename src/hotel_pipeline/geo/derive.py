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

    #: Couches telles que produites en mémoire, pour confronter le relu à
    #: l'attendu. Exclues du rapport sérialisé.
    expected_layers: dict = field(default_factory=dict, repr=False)

    def as_dict(self) -> dict:
        return {
            "grid": self.grid,
            "layers": self.layers,
            "artifacts": [a.model_dump(mode="json") for a in self.artifacts],
            "metrics": self.metrics,
            "qa_flags": self.qa_flags,
        }


# --- lecture du nuage ------------------------------------------------------


def load_ground_within(laz_path: Path, footprint, radius_m: float):  # noqa: ANN001
    """Classe 2 dans un large voisinage, pour la validation par pseudo-empreinte.

    Cette validation a besoin d'une étendue de sol sans rapport avec l'anneau
    de production : elle doit pouvoir y creuser plusieurs vides de la taille du
    bâtiment, sans chevauchement.
    """
    import laspy

    with laspy.open(str(laz_path)) as reader:
        points = reader.read()

    x = np.asarray(points.x)
    y = np.asarray(points.y)
    minx, miny, maxx, maxy = footprint.bounds
    window = (
        (x >= minx - radius_m) & (x <= maxx + radius_m)
        & (y >= miny - radius_m) & (y <= maxy + radius_m)
        & (np.asarray(points.classification) == GROUND)
    )
    return x[window], y[window], np.asarray(points.z)[window], window.sum()


def tile_covers(laz_bounds: dict, footprint, radius_m: float) -> bool | str:
    """La zone de recherche tient-elle entièrement dans la tuile ?

    Rend `"unknown"` faute de bornes : lever une exception ferait échouer la
    dérivation, et rendre `False` affirmerait une troncature non constatée.
    L'ignorance est un troisième état, pas un cas particulier du refus.
    """
    required = ("min_x", "max_x", "min_y", "max_y")
    if not laz_bounds or any(key not in laz_bounds for key in required):
        return "unknown"

    minx, miny, maxx, maxy = footprint.bounds
    return (
        minx - radius_m >= laz_bounds["min_x"]
        and maxx + radius_m <= laz_bounds["max_x"]
        and miny - radius_m >= laz_bounds["min_y"]
        and maxy + radius_m <= laz_bounds["max_y"]
    )


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
    policy=None,  # noqa: ANN001 — PipelinePolicy
    laz_bounds: dict | None = None,
) -> DeriveResult:
    """Produit toutes les couches dans le répertoire de transit.

    Tous les paramètres géométriques viennent de la politique : sans cela, une
    politique posée dans l'espace de travail ne changerait pas la dérivation.
    """
    from shapely import contains_xy

    from ..schemas.policy import DEFAULT_POLICY

    limits = (policy or DEFAULT_POLICY).terrain
    cell_m = limits.cell_m
    ring_m = limits.ring_m
    search_radius_m = limits.search_radius_m

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

    laz_bounds = laz_bounds or {}
    x, y, z, codes, inside = _load_points(laz_path, footprint_projected, ring_m)

    # --- appuis de terrain : classe 2 dans l'anneau réel -------------------
    # `buffer(ring).difference(footprint)` plutôt que la boîte : sur un
    # bâtiment oblique, la boîte inclut des zones bien plus éloignées d'un
    # côté que de l'autre.
    ring_polygon = footprint_projected.buffer(ring_m).difference(footprint_projected)
    ring = (codes == GROUND) & contains_xy(ring_polygon, x, y)
    px, py, pz = terrain.aggregate_median(x[ring], y[ring], z[ring], origin, cell_m)
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

    # Validation par pseudo-empreinte, sur une étendue de sol bien plus large
    # que l'anneau de production.
    wide_x, wide_y, wide_z, _ = load_ground_within(laz_path, footprint_projected, search_radius_m)
    wide_px, wide_py, wide_pz = terrain.aggregate_median(wide_x, wide_y, wide_z, origin, cell_m)
    # Les exclusions ne servent à rien si l'orchestration ne les transmet pas :
    # le vrai bâtiment et les concentrations de classe 6 privent une zone de
    # sol pour la même raison.
    building_x = x[(codes == BUILDING)]
    building_y = y[(codes == BUILDING)]
    pseudo_result = terrain.pseudo_footprint_validation(
        wide_px, wide_py, wide_pz, footprint_projected, ring_m=ring_m,
        trials=limits.min_trials, search_radius_m=search_radius_m, cell_m=cell_m,
        policy=policy, real_footprint=footprint_projected,
        building_xy=(building_x, building_y),
    )
    search_within_tile = tile_covers(laz_bounds, footprint_projected, search_radius_m)

    result.expected_layers = layers
    result.metrics = _metrics(
        grid, footprint_mask, dtm, dsm_roof, class1, ndsm, ndsm_valid,
        masks, disagreement, rejected, distances, px, py, pz, footprint_projected,
        pseudo_result, search_within_tile,
    )
    result.qa_flags = _qa_flags(result.metrics, ndsm, limits)
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
        x, y, z, (grid.origin_x, grid.origin_y), grid.cell_m
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
    pseudo_result, search_within_tile,  # noqa: ANN001
) -> dict:
    domain = int(footprint_mask.sum())
    interior_distances = distances[footprint_mask]

    block = terrain.block_cross_validation(px, py, pz)
    pseudo, pseudo_rejected = pseudo_result

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
        "pseudo_footprint_validation": {
            "trials": pseudo,
            "rejected_candidates": pseudo_rejected,
            "search_area_within_tile": search_within_tile,
        },
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


def _qa_flags(metrics: dict, ndsm, limits) -> list[str]:  # noqa: ANN001
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
    pseudo = metrics["pseudo_footprint_validation"]
    if len(pseudo["trials"]) < limits.min_trials:
        flags.append(
            f"{len(pseudo['trials'])} essai(s) de pseudo-empreinte pour un "
            f"minimum de {limits.min_trials} — l'erreur sous le bâtiment reste "
            "estimée par une validation optimiste"
        )
    # La troncature se signale indépendamment du nombre d'essais : une zone
    # tronquée qui en produit trois reste une zone tronquée.
    if pseudo["search_area_within_tile"] is not True:
        flags.append(
            f"zone de recherche non entièrement couverte par la tuile "
            f"({pseudo['search_area_within_tile']}) — les essais ne portent "
            "que sur la portion disponible"
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


def verify_written(
    result: DeriveResult, grid: GridSpec, expected: dict[str, np.ndarray] | None = None
) -> list[str]:
    """Rouvre chaque couche et la confronte à la grille **et aux valeurs**.

    Un manifeste ne doit jamais référencer un fichier qu'on n'a pas rouvert :
    l'écriture peut réussir et produire un raster décalé, mal projeté, ou
    rempli de mauvaises valeurs. Contrôler qu'il n'est pas vide ne suffit pas —
    un raster corrompu est rarement vide.
    """
    import rasterio

    expected_transform = grid.transform()
    problems: list[str] = []

    for name, path in result.layers.items():
        role = LAYER_ROLES[name]
        with rasterio.open(path) as source:
            if (source.width, source.height) != (grid.width, grid.height):
                problems.append(
                    f"{name} : {source.width}×{source.height} au lieu de "
                    f"{grid.width}×{grid.height}"
                )
            if source.crs is None or source.crs.to_string() != grid.crs:
                problems.append(f"{name} : référentiel {source.crs} au lieu de {grid.crs}")

            # Les six coefficients, pas seulement l'origine et le pas : une
            # rotation ou un cisaillement passerait autrement inaperçu.
            # Nom distinct du paramètre `expected` : une variable locale
            # homonyme avait désactivé toute la comparaison de valeurs, en
            # silence — `name not in [six flottants]` étant toujours vrai.
            actual_transform = [source.transform[i] for i in range(6)]
            wanted_transform = [expected_transform[i] for i in range(6)]
            if not np.allclose(actual_transform, wanted_transform):
                problems.append(
                    f"{name} : transformation {actual_transform} au lieu de "
                    f"{wanted_transform}"
                )

            expected_dtype = "uint8" if role == "mask" else "float32"
            if source.dtypes[0] != expected_dtype:
                problems.append(
                    f"{name} : type {source.dtypes[0]} au lieu de {expected_dtype}"
                )

            expected_nodata = None if role == "mask" else NODATA
            if source.nodata != expected_nodata:
                problems.append(
                    f"{name} : nodata {source.nodata} au lieu de {expected_nodata}"
                )

            # Les valeurs elles-mêmes, comparées à ce qui devait être écrit.
            band = source.read(1)
            if role == "mask":
                if not np.isin(np.unique(band), (0, 1)).all():
                    problems.append(f"{name} : masque contenant autre chose que 0 et 1")
            elif np.ma.masked_equal(band, NODATA).count() == 0:
                problems.append(f"{name} : aucune valeur exploitable")

            if expected is None or name not in expected:
                continue

            problems.extend(_compare_values(name, role, band, expected[name], grid))

    return problems


def _compare_values(name: str, role: str, band, target, grid: GridSpec) -> list[str]:  # noqa: ANN001
    """Confronte cellule à cellule le raster relu à la couche attendue.

    La comparaison se fait dans la précision **réellement écrite** — float32 —
    sans quoi l'arrondi de sérialisation se lirait comme une corruption.
    """
    from .raster import to_raster

    problems: list[str] = []
    written = to_raster(np.asarray(target))

    if role == "mask":
        if not np.array_equal(band.astype(bool), written.astype(bool)):
            differing = int((band.astype(bool) != written.astype(bool)).sum())
            problems.append(f"{name} : {differing} cellule(s) de masque divergentes")
        return problems

    reference = np.where(np.isnan(written), NODATA, written).astype("float32")

    defined_read = band != NODATA
    defined_expected = reference != NODATA
    if not np.array_equal(defined_read, defined_expected):
        differing = int((defined_read != defined_expected).sum())
        problems.append(
            f"{name} : {differing} cellule(s) définies d'un côté seulement"
        )

    if not np.array_equal(band, reference, equal_nan=True):
        differing = int((band != reference).sum())
        problems.append(f"{name} : {differing} cellule(s) de valeurs divergentes")

    return problems


def verify_publication(site) -> list[str]:  # noqa: ANN001 — SiteManifest
    """Refuse un manifeste dont un artefact actif pointe vers un fichier absent.

    Conserver les productions antérieures est correct ; les laisser passer pour
    courantes ne l'est pas. Un artefact actif dont le chemin n'existe plus est
    une référence morte, et une qualification fondée dessus serait invérifiable.
    """
    problems: list[str] = []
    for artifact in site.active_artifacts():
        if not Path(artifact.path).is_file():
            problems.append(
                f"artefact actif {artifact.artifact_id!r} : fichier absent "
                f"({artifact.path}) — le marquer superseded ou invalidated"
            )
    return problems


def supersede_previous(site, new_artifacts) -> int:  # noqa: ANN001
    """Fait passer en `superseded` les actifs remplacés par une nouvelle série.

    Sans cela, deux exécutions restent actives simultanément et une
    qualification ne saurait pas laquelle choisir. La correspondance se fait
    par rôle **et** par chemin de couche, non par identifiant : celui-ci porte
    l'horodatage de son exécution.
    """
    successors = {}
    for artifact in new_artifacts:
        key = (artifact.role, Path(artifact.path).name)
        successors[key] = artifact.artifact_id

    new_ids = {a.artifact_id for a in new_artifacts}
    marked = 0
    for index, artifact in enumerate(site.artifacts):
        if artifact.artifact_id in new_ids or not artifact.is_active:
            continue
        successor = successors.get((artifact.role, Path(artifact.path).name))
        if successor:
            site.artifacts[index] = artifact.model_copy(
                update={"status": "superseded", "superseded_by": successor}
            )
            marked += 1
    return marked


def verify_digests(site) -> list[str]:  # noqa: ANN001
    """Confronte l'empreinte déclarée au contenu réel des artefacts actifs.

    L'existence d'un fichier ne prouve pas qu'il n'a pas changé depuis sa
    publication : un raster réécrit à la main passerait le contrôle de présence.
    """
    from .raster import sha256_file

    problems: list[str] = []
    for artifact in site.active_artifacts():
        path = Path(artifact.path)
        if not path.is_file():
            continue
        actual = sha256_file(path)
        if actual != artifact.sha256:
            problems.append(
                f"artefact {artifact.artifact_id!r} : empreinte {actual[:16]}… "
                f"au lieu de {artifact.sha256[:16]}… — le fichier a changé"
            )
    return problems


def supersede_missing(site, reason: str) -> int:  # noqa: ANN001
    """Marque comme invalidés les artefacts actifs sans fichier.

    Le motif est obligatoire : un rejet sans raison n'apprend rien à qui relira
    le manifeste dans six mois.
    """
    marked = 0
    for index, artifact in enumerate(site.artifacts):
        if artifact.is_active and not Path(artifact.path).is_file():
            site.artifacts[index] = artifact.model_copy(
                update={"status": "invalidated", "invalidation_reason": reason}
            )
            marked += 1
    return marked
