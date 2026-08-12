"""Préflight LAZ — ce que la tuile porte réellement (Lot 1B §9).

Une classification annoncée n'est pas une classification portée, et une densité
moyenne sur une tuile de plusieurs kilomètres carrés ne dit rien de l'empreinte
d'un bâtiment. Ce module mesure avant de décider d'une méthode.

L'indicateur décisif n'est pas un nombre de points mais une **proportion de
cellules occupées** : un toit représenté par quelques amas discontinus produit
un nombre de points flatteur et une enveloppe fausse. C'est précisément la
géométrie inventée que le plan directeur interdit.

Deux référentiels coexistent : la tuile est projetée — EPSG:2950 pour ce
pilote — et l'empreinte vient d'OSM en WGS84. Le découpage se fait donc après
reprojection ; le faire dans le mauvais référentiel rend un résultat vide ou
décalé, sans lever d'erreur.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..logging import get_logger

log = get_logger("laz-preflight")

#: Classes ASPRS utiles ici.
GROUND = 2
BUILDING = 6
UNCLASSIFIED = 1
NOISE = 7

CLASS_NAMES = {
    UNCLASSIFIED: "non classé",
    GROUND: "sol",
    BUILDING: "bâtiment",
    NOISE: "bruit",
}

#: Taille de cellule pour mesurer la couverture spatiale, en mètres. Un toit
#: d'hôtel se juge à cette échelle : plus fin surestime les trous, plus grossier
#: masque les discontinuités.
COVERAGE_CELL_M = 1.0

#: Marge autour de l'empreinte, en mètres, pour comparer intérieur et pourtour.
MARGIN_M = 20.0


@dataclass
class ClassStats:
    code: int
    name: str
    count: int = 0
    z_min: float | None = None
    z_median: float | None = None
    z_max: float | None = None

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "name": self.name,
            "count": self.count,
            "z_min": self.z_min,
            "z_median": self.z_median,
            "z_max": self.z_max,
        }


@dataclass
class PreflightReport:
    file: str = ""
    las_version: str | None = None
    point_format: int | None = None
    point_count: int = 0
    declared_crs: str | None = None
    bounds: dict[str, float] = field(default_factory=dict)

    footprint_classes: dict[int, ClassStats] = field(default_factory=dict)
    margin_classes: dict[int, ClassStats] = field(default_factory=dict)

    footprint_area_m2: float = 0.0
    ground_density_per_m2: float | None = None
    building_density_per_m2: float | None = None

    #: Proportion de cellules de l'empreinte contenant au moins un point de
    #: classe 6. C'est ce chiffre, et non le nombre de points, qui dit si une
    #: toiture est mesurable.
    roof_cell_coverage: float | None = None
    ground_cell_coverage: float | None = None

    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "file": self.file,
            "las_version": self.las_version,
            "point_format": self.point_format,
            "point_count": self.point_count,
            "declared_crs": self.declared_crs,
            "bounds": self.bounds,
            "footprint_area_m2": round(self.footprint_area_m2, 1),
            "classes_in_footprint": {
                str(c): s.as_dict() for c, s in sorted(self.footprint_classes.items())
            },
            "classes_in_margin": {
                str(c): s.as_dict() for c, s in sorted(self.margin_classes.items())
            },
            "effective_density": {
                "ground_per_m2": self.ground_density_per_m2,
                "building_per_m2": self.building_density_per_m2,
            },
            "cell_coverage": {
                "roof": self.roof_cell_coverage,
                "ground": self.ground_cell_coverage,
            },
            "warnings": self.warnings,
        }


def _stats(codes, z, code: int) -> ClassStats:  # noqa: ANN001
    import numpy as np

    stats = ClassStats(code=code, name=CLASS_NAMES.get(code, f"classe {code}"))
    selected = z[codes == code]
    stats.count = int(selected.size)
    if stats.count:
        stats.z_min = float(np.min(selected))
        stats.z_median = float(np.median(selected))
        stats.z_max = float(np.max(selected))
    return stats


def _cell_coverage(x, y, footprint, cell_m: float) -> float:  # noqa: ANN001
    """Proportion de cellules **du polygone** contenant au moins un point.

    Le dénominateur est le nombre de cellules dont le centre tombe dans
    l'empreinte, non celles de sa boîte englobante. Un bâtiment allongé et
    oblique n'occupe qu'une fraction de sa boîte : mesurer contre elle plafonne
    la couverture à cette fraction, et fait passer une toiture dense pour
    fragmentaire — 34 % annoncés là où la densité mesurée était de 25 points
    par mètre carré.
    """
    import numpy as np
    from shapely.vectorized import contains

    minx, miny, maxx, maxy = footprint.bounds
    columns = max(1, int((maxx - minx) / cell_m))
    rows = max(1, int((maxy - miny) / cell_m))

    # Centres de cellules, puis filtre par le polygone.
    cx = minx + (np.arange(columns) + 0.5) * cell_m
    cy = miny + (np.arange(rows) + 0.5) * cell_m
    grid_x, grid_y = np.meshgrid(cx, cy, indexing="ij")
    inside_grid = contains(footprint, grid_x.ravel(), grid_y.ravel())
    denominator = int(inside_grid.sum())
    if denominator == 0:
        return 0.0

    if x.size == 0:
        return 0.0

    ix = np.clip(((x - minx) / cell_m).astype(int), 0, columns - 1)
    iy = np.clip(((y - miny) / cell_m).astype(int), 0, rows - 1)
    occupied_keys = np.unique(ix.astype(np.int64) * rows + iy.astype(np.int64))

    # Ne compter que les cellules occupées qui appartiennent au polygone.
    inside_keys = np.flatnonzero(inside_grid)
    occupied_inside = np.intersect1d(occupied_keys, inside_keys, assume_unique=True).size
    return round(occupied_inside / denominator, 4)


def run(laz_path: Path, footprint_wkt: str, declared_crs: str) -> PreflightReport:
    """Mesure ce que la tuile porte sur l'empreinte, et autour.

    `footprint_wkt` est en WGS84 ; il est reprojeté vers le référentiel de la
    tuile avant tout découpage.
    """
    import laspy
    import numpy as np
    from pyproj import Transformer
    from shapely import wkt as shapely_wkt
    from shapely.ops import transform as shapely_transform
    from shapely.vectorized import contains

    report = PreflightReport(file=laz_path.name)

    with laspy.open(str(laz_path)) as reader:
        header = reader.header
        report.las_version = f"{header.version.major}.{header.version.minor}"
        report.point_format = int(header.point_format.id)
        report.point_count = int(header.point_count)
        report.bounds = {
            "min_x": float(header.mins[0]), "min_y": float(header.mins[1]),
            "min_z": float(header.mins[2]), "max_x": float(header.maxs[0]),
            "max_y": float(header.maxs[1]), "max_z": float(header.maxs[2]),
        }
        report.declared_crs = _declared_crs(header) or declared_crs
        points = reader.read()

    # L'empreinte OSM est en WGS84 ; la tuile est projetée.
    transformer = Transformer.from_crs("EPSG:4326", declared_crs, always_xy=True)
    footprint = shapely_transform(
        lambda xs, ys, zs=None: transformer.transform(xs, ys),
        shapely_wkt.loads(footprint_wkt),
    )
    report.footprint_area_m2 = footprint.area

    minx, miny, maxx, maxy = footprint.bounds
    if not _overlaps(report.bounds, footprint.bounds):
        report.warnings.append(
            "l'empreinte reprojetée ne recoupe pas les bornes de la tuile — "
            "vérifier le référentiel déclaré"
        )
        return report

    x = np.asarray(points.x)
    y = np.asarray(points.y)
    z = np.asarray(points.z)
    codes = np.asarray(points.classification)

    # Fenêtre grossière d'abord : tester 20 millions de points contre un
    # polygone serait inutilement coûteux.
    margin = MARGIN_M
    window = (
        (x >= minx - margin) & (x <= maxx + margin)
        & (y >= miny - margin) & (y <= maxy + margin)
    )
    xw, yw, zw, cw = x[window], y[window], z[window], codes[window]

    inside = contains(footprint, xw, yw)
    for code in (UNCLASSIFIED, GROUND, BUILDING, NOISE):
        report.footprint_classes[code] = _stats(cw[inside], zw[inside], code)
        report.margin_classes[code] = _stats(cw[~inside], zw[~inside], code)

    area = report.footprint_area_m2 or 1.0
    report.ground_density_per_m2 = round(
        report.footprint_classes[GROUND].count / area, 2
    )
    report.building_density_per_m2 = round(
        report.footprint_classes[BUILDING].count / area, 2
    )

    roof = (cw == BUILDING) & inside
    ground = (cw == GROUND) & inside
    report.roof_cell_coverage = _cell_coverage(
        xw[roof], yw[roof], footprint, COVERAGE_CELL_M
    )
    report.ground_cell_coverage = _cell_coverage(
        xw[ground], yw[ground], footprint, COVERAGE_CELL_M
    )

    _add_warnings(report)
    log.info(
        "préflight %s : %d point(s) dans l'empreinte, couverture toiture %.1f %%",
        laz_path.name,
        sum(s.count for s in report.footprint_classes.values()),
        (report.roof_cell_coverage or 0) * 100,
    )
    return report


def _declared_crs(header) -> str | None:  # noqa: ANN001
    """Référentiel déclaré par le fichier, à confronter à celui de l'index."""
    try:
        crs = header.parse_crs()
    except Exception:  # noqa: BLE001 — un en-tête sans CRS ne doit pas interrompre
        return None
    if crs is None:
        return None
    code = crs.to_epsg()
    return f"EPSG:{code}" if code else crs.name


def _overlaps(bounds: dict, footprint_bounds: tuple) -> bool:
    minx, miny, maxx, maxy = footprint_bounds
    return not (
        maxx < bounds["min_x"]
        or minx > bounds["max_x"]
        or maxy < bounds["min_y"]
        or miny > bounds["max_y"]
    )


def _add_warnings(report: PreflightReport) -> None:
    """Signale ce qui empêcherait une dérivation honnête."""
    ground = report.footprint_classes.get(GROUND)
    building = report.footprint_classes.get(BUILDING)
    unclassified = report.footprint_classes.get(UNCLASSIFIED)

    if not ground or ground.count == 0:
        report.warnings.append(
            "aucun point de classe 2 dans l'empreinte — TERRAIN_MAIN non dérivable ici"
        )
    if not building or building.count == 0:
        report.warnings.append(
            "aucun point de classe 6 dans l'empreinte — ROOFLINE_MAIN non dérivable"
        )
    elif (report.roof_cell_coverage or 0) < 0.5:
        report.warnings.append(
            f"couverture de toiture fragmentaire "
            f"({(report.roof_cell_coverage or 0) * 100:.0f} % des cellules) — "
            "une enveloppe en serait esquissée, non mesurée"
        )

    if unclassified and building and unclassified.count and building.count:
        # Si la classe 1 culmine au niveau du toit, elle en porte probablement
        # des éléments ; si elle reste basse, c'est de la végétation ou du sol.
        if unclassified.z_median is not None and building.z_median is not None:
            if unclassified.z_median >= building.z_median:
                report.warnings.append(
                    "classe 1 aussi haute que la classe 6 — examiner avant de "
                    "l'exclure du MNS, elle peut porter des superstructures"
                )
