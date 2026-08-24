"""Hauteurs mesurées depuis le nDSM LiDAR, quand il en couvre l'emprise.

Le module remplace une hypothèse d'animation par une mesure — **là où la
mesure existe**, et nulle part ailleurs. Sur le pilote, la tuile ne couvre que
le bâtiment cible : les 27 volumes voisins restent en hauteur supposée, et le
disent.

Deux choix méritent d'être explicités.

**Le percentile plutôt que le maximum.** Le maximum d'un nDSM sur une emprise
capte les superstructures — cheminées, édicules d'ascenseur, unités de
ventilation — et un arbre qui déborde. Le p90 donne la hauteur du corps du
bâtiment, qui est ce qu'une silhouette doit rendre.

**Un plancher de couverture.** Une emprise dont trois cellules sur mille
portent une valeur ne donne pas une hauteur : elle donne un artefact de bord.
En deçà du plancher, le module refuse de mesurer plutôt que de mesurer mal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..logging import get_logger

log = get_logger("conditioning-heights")

#: Percentile retenu : le corps du bâti, superstructures écartées.
HEIGHT_PERCENTILE = 90.0

#: Fraction minimale de cellules valides sur l'emprise pour croire à la mesure.
MIN_COVERAGE = 0.30

#: Nombre minimal de cellules valides, indépendamment de la fraction.
MIN_CELLS = 20

#: En deçà, la « hauteur » relève du bruit du modèle de terrain.
MIN_HEIGHT_M = 2.0

#: Classes ASPRS utiles : 2 pour le sol, 6 pour le bâti. Un point non classé
#: peut être une superstructure comme une branche, et n'atteste rien seul.
GROUND_CLASS = 2
BUILDING_CLASS = 6

#: Points de classe bâtiment requis dans une emprise pour croire à la mesure.
MIN_BUILDING_POINTS = 60

#: Écart d'altitude au-delà duquel deux cellules voisines n'appartiennent pas
#: à la même toiture. Mesuré sur ce pilote : l'auvent d'entrée est à quatre
#: mètres et le corps principal à dix, et les relier produisait des facettes
#: verticales en dents de scie le long du décrochement.
ROOF_STEP_BREAK_M = 2.5

#: Pas d'échantillonnage du toit, en cellules du raster. Le nDSM est à 0,5 m ;
#: un pas de 1 échantillonne le nDSM à sa pleine résolution — cinquante
#: centimètres ici. Un pas plus large jetait une cellule sur deux, et les
#: décrochements fins — le porche d'entrée, ses avancées — s'arrondissaient
#: avant même d'être rendus. Un pas plus large laissait la
#: surface s'arrêter à deux mètres du bord de l'emprise : les murs montaient,
#: le toit ne les rejoignait pas, et le volume se creusait en cuvette bordée
#: de dents. Le coût reste modeste — quelques milliers de triangles pour un
#: bâtiment de mille huit cents mètres carrés.
ROOF_STEP = 1


class RasterUnavailable(RuntimeError):
    """Le nDSM manque ou n'est pas lisible : aucune mesure n'est possible."""


@dataclass
class HeightMeasurement:
    """Ce que le nDSM établit sur l'emprise d'un volume."""

    feature_id: str
    height_m: float
    cells: int
    coverage: float
    percentile: float
    source: str

    def as_dict(self) -> dict:
        return {
            "feature_id": self.feature_id,
            "height_m": round(self.height_m, 2),
            "cells": self.cells,
            "coverage": round(self.coverage, 3),
            "percentile": self.percentile,
            "source": self.source,
        }


def _vectorized_grid_surface(
    values: np.ndarray,
    polygon,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Découpe une grille sur le contour vectoriel exact de l'emprise.

    Les anciennes surfaces ne gardaient que les centres de cellule situés dans
    le bâtiment. Leur bord suivait donc la maille en escalier et restait en
    retrait des murs. Ici, chaque cellule est intersectée avec le polygone
    régularisé : l'intérieur conserve la résolution du raster, tandis que le
    bord partage exactement les arêtes XY de l'emprise.
    """
    from scipy.ndimage import map_coordinates
    from shapely.geometry import Polygon
    from shapely.ops import triangulate

    rows, cols = values.shape
    if len(x_edges) != cols + 1 or len(y_edges) != rows + 1:
        raise ValueError("les arêtes de grille ne correspondent pas aux valeurs")
    if cols < 1 or rows < 1:
        return None

    dx = float(x_edges[1] - x_edges[0])
    dy = float(y_edges[1] - y_edges[0])
    if abs(dx) <= 1e-12 or abs(dy) <= 1e-12:
        return None

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    vertex_lookup: dict[tuple[float, float], int] = {}

    def vertex(x: float, y: float) -> int:
        key = (round(float(x), 9), round(float(y), 9))
        found = vertex_lookup.get(key)
        if found is not None:
            return found
        # Les valeurs sont portées par les centres de cellule. Les coordonnées
        # continues de scipy sont donc décalées d'une demi-cellule par rapport
        # aux arêtes utilisées pour découper le contour.
        col = (float(x) - float(x_edges[0])) / dx - 0.5
        row = (float(y) - float(y_edges[0])) / dy - 0.5
        height = float(
            map_coordinates(
                values,
                np.asarray([[row], [col]], dtype=np.float64),
                order=1,
                mode="nearest",
            )[0]
        )
        index = len(vertices)
        vertex_lookup[key] = index
        vertices.append([float(x), float(y), height])
        return index

    def add_triangle(coords) -> None:  # noqa: ANN001
        indices = [vertex(float(x), float(y)) for x, y in coords]
        if len(set(indices)) != 3:
            return
        a, b, c = (np.asarray(vertices[index][:2]) for index in indices)
        if abs(float(np.cross(b - a, c - a))) <= 1e-10:
            return
        faces.append(indices)

    for row in range(rows):
        for col in range(cols):
            corners = [
                (float(x_edges[col]), float(y_edges[row])),
                (float(x_edges[col + 1]), float(y_edges[row])),
                (float(x_edges[col + 1]), float(y_edges[row + 1])),
                (float(x_edges[col]), float(y_edges[row + 1])),
            ]
            cell = Polygon(corners)
            if not polygon.intersects(cell):
                continue
            if polygon.covers(cell):
                indices = [vertex(x, y) for x, y in corners]
                heights = [vertices[index][2] for index in indices]
                # Aux ruptures, la diagonale relie les deux coins les plus
                # proches en altitude. La marche reste nette sans ouvrir le
                # toit, comme dans l'ancienne grille intérieure.
                if (
                    max(heights) - min(heights) > ROOF_STEP_BREAK_M
                    and abs(heights[0] - heights[2])
                    > abs(heights[1] - heights[3])
                ):
                    faces.extend(
                        [
                            [indices[0], indices[1], indices[3]],
                            [indices[1], indices[2], indices[3]],
                        ]
                    )
                else:
                    faces.extend(
                        [
                            [indices[0], indices[1], indices[2]],
                            [indices[0], indices[2], indices[3]],
                        ]
                    )
                continue
            clipped = polygon.intersection(cell)
            if clipped.is_empty or clipped.area <= 1e-10:
                continue
            pieces = (
                [clipped]
                if clipped.geom_type == "Polygon"
                else [part for part in clipped.geoms if part.geom_type == "Polygon"]
            )
            for piece in pieces:
                # L'intersection d'une cellule et d'une emprise peut être
                # concave. Shapely propose une Delaunay, puis ce filtre conserve
                # seulement les triangles réellement couverts par la pièce.
                for triangle in triangulate(piece):
                    if piece.covers(triangle):
                        add_triangle(list(triangle.exterior.coords)[:-1])

    if not faces:
        return None
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def measure_footprint(
    raster, footprint: np.ndarray, feature_id: str
) -> HeightMeasurement | None:
    """Mesure la hauteur d'une emprise, ou rend None si le raster n'y dit rien."""
    from rasterio.mask import mask
    from shapely.geometry import Polygon

    polygon = Polygon(footprint)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        return None

    try:
        window, _ = mask(
            raster,
            [polygon.__geo_interface__],
            crop=True,
            filled=True,
            nodata=raster.nodata,
        )
    except ValueError:
        # L'emprise ne rencontre pas la tuile : ce n'est pas une erreur, c'est
        # une absence de donnée, et elle se traite comme telle.
        return None

    band = window[0]
    valid = band[
        (band != raster.nodata) & np.isfinite(band) & (band > MIN_HEIGHT_M)
    ]
    coverage = valid.size / max(band.size, 1)
    if valid.size < MIN_CELLS or coverage < MIN_COVERAGE:
        log.info(
            "%s : couverture insuffisante (%d cellules, %.0f %%) — hauteur non mesurée",
            feature_id,
            valid.size,
            coverage * 100,
        )
        return None

    height = float(np.percentile(valid, HEIGHT_PERCENTILE))
    return HeightMeasurement(
        feature_id=feature_id,
        height_m=height,
        cells=int(valid.size),
        coverage=float(coverage),
        percentile=HEIGHT_PERCENTILE,
        source="ndsm_lidar",
    )


def build_roof_surface(
    raster, footprint: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Triangule la surface réelle du toit depuis le nDSM.

    Sans mesure, un prisme se ferme par un cône vers son centre : une
    commodité géométrique qui n'observe rien. Le nDSM donne la forme réelle —
    mesuré ici, un toit dont l'altitude varie de 9,4 à 10,9 m d'une aile à
    l'autre — et cette forme remplace la fermeture inventée.

    Les sommets sont produits en coordonnées absolues du CRS de travail, comme
    le reste de la scène, pour qu'aucune conversion ne traîne en aval.
    """
    from rasterio.mask import mask
    from shapely.geometry import Polygon

    polygon = Polygon(footprint)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        return None

    try:
        window, transform = mask(
            raster,
            [polygon.__geo_interface__],
            crop=True,
            filled=True,
            nodata=raster.nodata,
        )
    except ValueError:
        return None

    band = window[0]
    rows, cols = band.shape
    step = max(1, ROOF_STEP)

    valid = (band != raster.nodata) & np.isfinite(band) & (band > MIN_HEIGHT_M)
    if valid.sum() < MIN_CELLS:
        return None

    # Le nDSM couvre l'emprise à quelques pour cent près — 3 % de cellules
    # manquantes sur ce pilote, dispersées. Échantillonner la grille telle
    # quelle rejetait tout quadrilatère touchant un trou, et le toit sortait
    # dentelé, troué en son milieu. Les manques sont donc comblés par le plus
    # proche voisin mesuré, ce qui ne crée pas de hauteur : cela prolonge la
    # surface observée sur des lacunes de quelques décimètres.
    filled = band.astype(np.float64, copy=True)
    if not valid.all():
        from scipy.ndimage import distance_transform_edt

        _, nearest = distance_transform_edt(~valid, return_indices=True)
        filled = filled[tuple(nearest)]

    # Les cellules sont conservées à leur résolution native puis découpées sur
    # le polygone vectoriel. Le bord du maillage rejoint ainsi exactement les
    # murs, au lieu de suivre les centres de pixels en escalier.
    sampled = filled[::step, ::step]
    x_edges = np.asarray(
        [(transform * (col * step, 0))[0] for col in range(sampled.shape[1] + 1)],
        dtype=np.float64,
    )
    y_edges = np.asarray(
        [(transform * (0, row * step))[1] for row in range(sampled.shape[0] + 1)],
        dtype=np.float64,
    )
    return _vectorized_grid_surface(sampled, polygon, x_edges, y_edges)


def apply_laz_heights(scene, laz_path) -> dict:
    """Complète les hauteurs supposées par une mesure dans le nuage brut.

    Le nDSM reste prioritaire là où il existe : il est qualifié et porte une
    surface de toit exploitable. Le nuage sert pour tout ce qu'il ne couvre
    pas — sur ce pilote, les vingt-sept volumes voisins, que le raster dérivé
    ignorait alors que la tuile source les contient.
    """
    # Plusieurs tuiles peuvent être fournies : un site déborde souvent de
    # celle qui porte son bâtiment, et chaque volume est mesuré dans la tuile
    # qui le contient.
    # Rien à compléter : la question de la tuile ne se pose pas. Lever ici
    # ferait échouer une scène déjà entièrement mesurée au seul motif qu'aucun
    # nuage n'est fourni.
    pending = [p for p in scene.prisms if p.height_assumed]
    if not pending:
        return {"measured": 0, "total": len(scene.prisms), "still_assumed": 0,
                "measurements": []}

    tiles = [Path(laz_path)] if isinstance(laz_path, (str, Path)) else list(laz_path)
    tiles = [t for t in tiles if Path(t).is_file()]
    if not tiles:
        raise RasterUnavailable(f"tuile LiDAR absente : {laz_path}")

    import laspy

    bounds = []
    for tile in tiles:
        with laspy.open(str(tile)) as reader:
            header = reader.header
            bounds.append((
                float(header.mins[0]), float(header.mins[1]),
                float(header.maxs[0]), float(header.maxs[1]),
            ))
    tile_bounds = (
        min(b[0] for b in bounds), min(b[1] for b in bounds),
        max(b[2] for b in bounds), max(b[3] for b in bounds),
    )

    found: dict[str, HeightMeasurement] = {}
    for tile in tiles:
        remaining = [p for p in pending if p.feature_id not in found]
        if not remaining:
            break
        found.update(measure_from_laz(tile, remaining))
    for prism in pending:
        measurement = found.get(prism.feature_id)
        if measurement is None:
            # Dire *pourquoi* la mesure manque : « hors couverture » est un
            # motif, « pas mesuré » n'en est pas un. Sur ce pilote, vingt
            # volumes sur vingt-sept sortent simplement de la tuile obtenue.
            x, y = prism.footprint[:, 0], prism.footprint[:, 1]
            outside = (
                x.min() < tile_bounds[0]
                or x.max() > tile_bounds[2]
                or y.min() < tile_bounds[1]
                or y.max() > tile_bounds[3]
            )
            prism.height_source = (
                "hypothèse — emprise hors de la tuile LiDAR obtenue "
                "(élargir la découverte avec `geo discover --scene`)"
                if outside
                else "hypothèse — trop peu de points classés bâtiment dans l'emprise"
            )
            continue
        prism.height_m = measurement.height_m
        prism.height_assumed = False
        prism.height_source = (
            f"nuage LiDAR, p{measurement.percentile:.0f} sur "
            f"{measurement.cells} points de bâti"
        )

    still = sum(1 for p in scene.prisms if p.height_assumed)
    out_of_tile = sum(
        1 for p in scene.prisms if p.height_assumed and "hors de la tuile" in p.height_source
    )
    scene.provenance["laz"] = [str(t) for t in tiles]
    log.info("hauteurs complétées au nuage : %d volume(s)", len(found))
    return {
        "laz": [str(t) for t in tiles],
        "tile_bounds": [round(v, 1) for v in tile_bounds],
        "measured": len(found),
        "total": len(scene.prisms),
        "still_assumed": still,
        "outside_tile": out_of_tile,
        "measurements": [m.as_dict() for m in found.values()],
    }


def apply_measured_heights(scene, ndsm_path: Path) -> dict:
    """Remplace, dans la scène, les hauteurs supposées par les hauteurs mesurées.

    Les volumes que le raster ne couvre pas gardent leur hauteur d'hypothèse et
    restent marqués comme telle : une tuile partielle ne doit pas transformer
    en mesure ce qu'elle n'a pas vu.
    """
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - dépend de l'installation
        raise RasterUnavailable(
            "rasterio est requis pour lire le nDSM — installer l'extra 'geo'"
        ) from exc

    ndsm_path = Path(ndsm_path)
    if not ndsm_path.is_file():
        raise RasterUnavailable(f"nDSM absent : {ndsm_path}")

    measured: list[HeightMeasurement] = []
    with rasterio.open(ndsm_path) as raster:
        raster_crs = str(raster.crs)
        if scene.crs != "unknown" and raster_crs and raster_crs != scene.crs:
            raise RasterUnavailable(
                f"le nDSM est en {raster_crs} et la scène en {scene.crs} : "
                "aucune mesure n'est reprojetée en silence"
            )
        for prism in scene.prisms:
            found = measure_footprint(raster, prism.footprint, prism.feature_id)
            if found is None:
                continue
            prism.height_m = found.height_m
            prism.height_assumed = False
            prism.height_source = (
                f"nDSM LiDAR, p{found.percentile:.0f} sur {found.cells} cellules "
                f"({found.coverage:.0%} de l'emprise)"
            )
            surface = build_roof_surface(raster, prism.footprint)
            if surface is not None:
                prism.roof_vertices, prism.roof_faces = surface
                log.info(
                    "%s : toit mesuré, %d triangles",
                    prism.feature_id,
                    len(prism.roof_faces),
                )
            measured.append(found)

    scene.provenance["ndsm"] = str(ndsm_path)
    scene.provenance["measured_heights"] = [m.as_dict() for m in measured]
    log.info(
        "hauteurs mesurées : %d/%d volumes", len(measured), len(scene.prisms)
    )
    return {
        "ndsm": str(ndsm_path),
        "measured": len(measured),
        "total": len(scene.prisms),
        "still_assumed": len(scene.prisms) - len(measured),
        "measurements": [m.as_dict() for m in measured],
    }


def measure_from_laz(
    laz_path: Path, prisms: list, buffer_m: float = 1.0
) -> dict[str, HeightMeasurement]:
    """Mesure les hauteurs directement dans le nuage LiDAR classifié.

    Le nDSM dérivé ne couvre que l'emprise du bâtiment cible — 72 × 77 m sur ce
    pilote — alors que la tuile source couvre un kilomètre carré et contient
    les voisins. Les mesurer ne demande donc aucune donnée nouvelle : il suffit
    de lire le nuage là où le raster n'a pas été dérivé.

    La hauteur est la différence entre le toit et le sol **local**, tous deux
    lus dans la même emprise : un terrain en pente fausserait toute référence
    d'altitude commune.
    """
    try:
        import laspy
    except ImportError as exc:  # pragma: no cover - dépend de l'installation
        raise RasterUnavailable(
            "laspy est requis pour lire la tuile LiDAR — installer l'extra 'geo'"
        ) from exc
    import shapely
    from shapely.geometry import Polygon

    laz_path = Path(laz_path)
    if not laz_path.is_file():
        raise RasterUnavailable(f"tuile LiDAR absente : {laz_path}")

    shapes = {}
    for prism in prisms:
        polygon = Polygon(prism.footprint)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty:
            continue
        shapes[prism.feature_id] = (polygon, polygon.bounds)

    if not shapes:
        return {}

    roof: dict[str, list[float]] = {k: [] for k in shapes}
    ground: dict[str, list[float]] = {k: [] for k in shapes}

    with laspy.open(str(laz_path)) as reader:
        for chunk in reader.chunk_iterator(2_000_000):
            x = np.asarray(chunk.x)
            y = np.asarray(chunk.y)
            z = np.asarray(chunk.z)
            classes = np.asarray(chunk.classification)
            useful = (classes == BUILDING_CLASS) | (classes == GROUND_CLASS)
            if not useful.any():
                continue
            x, y, z, classes = x[useful], y[useful], z[useful], classes[useful]

            for feature_id, (polygon, bounds) in shapes.items():
                minx, miny, maxx, maxy = bounds
                # Pré-filtre par boîte englobante : tester le polygone sur des
                # millions de points serait inutilement coûteux.
                box = (
                    (x >= minx - buffer_m)
                    & (x <= maxx + buffer_m)
                    & (y >= miny - buffer_m)
                    & (y <= maxy + buffer_m)
                )
                if not box.any():
                    continue
                # Le test point-dans-polygone est vectorisé : l'appeler point
                # par point sur vingt-trois millions de points prendrait des
                # heures là où `contains_xy` traite le lot d'un coup.
                inside = shapely.contains_xy(polygon, x[box], y[box])
                if not inside.any():
                    continue
                zin, cin = z[box][inside], classes[box][inside]
                roof[feature_id].extend(zin[cin == BUILDING_CLASS].tolist())
                ground[feature_id].extend(zin[cin == GROUND_CLASS].tolist())

    found: dict[str, HeightMeasurement] = {}
    for feature_id in shapes:
        tops, floors = roof[feature_id], ground[feature_id]
        if len(tops) < MIN_BUILDING_POINTS:
            log.info(
                "%s : %d point(s) de bâti — hauteur non mesurée",
                feature_id,
                len(tops),
            )
            continue
        # Le sol de l'emprise même est souvent rare — le bâtiment le masque.
        # À défaut, la base des points de bâti donne le niveau d'assise.
        base = (
            float(np.median(floors))
            if len(floors) >= 10
            else float(np.percentile(tops, 2))
        )
        height = float(np.percentile(tops, HEIGHT_PERCENTILE)) - base
        if height < MIN_HEIGHT_M:
            continue
        found[feature_id] = HeightMeasurement(
            feature_id=feature_id,
            height_m=height,
            cells=len(tops),
            coverage=1.0,
            percentile=HEIGHT_PERCENTILE,
            source="laz_lidar",
        )
    return found


def find_laz(workspace, centre: tuple[float, float] | None = None) -> Path | None:  # noqa: ANN001
    """Tuile LiDAR qui contient un point, ou la première à défaut.

    Le tri alphabétique ne dit rien de la géographie : dès qu'un site a
    plusieurs tuiles, la première du nom peut être celle du coin opposé. Sur ce
    pilote, l'acquisition des tuiles voisines avait fait basculer la lecture
    sur une emprise ne contenant pas le bâtiment, et la végétation ressortait
    vide. Le centre du site départage.
    """
    tiles = find_laz_tiles(workspace)
    if not tiles:
        return None
    if centre is None:
        return tiles[0]

    import laspy

    cx, cy = centre
    for tile in tiles:
        with laspy.open(str(tile)) as reader:
            header = reader.header
            if (
                header.mins[0] <= cx <= header.maxs[0]
                and header.mins[1] <= cy <= header.maxs[1]
            ):
                return tile
    return tiles[0]


def find_laz_tiles(workspace) -> list[Path]:  # noqa: ANN001
    """Toutes les tuiles LiDAR du workspace.

    Un site déborde souvent de la tuile qui contient son bâtiment : sur ce
    pilote, sept obstacles sur vingt-sept tenaient dans la première, contre
    vingt et un une fois les quatre tuiles voisines acquises. Ne lire que la
    première laissait donc la moitié du bâti en hauteur conventionnelle.
    """
    raw = workspace.path("06_geo", "lidar_raw")
    if not raw.is_dir():
        return []
    return sorted(raw.glob("*.LAZ")) + sorted(raw.glob("*.laz"))


def find_ndsm(workspace) -> Path | None:  # noqa: ANN001
    """Cherche le nDSM qualifié le plus récent du workspace."""
    derived = workspace.path("06_geo", "derived")
    if not derived.is_dir():
        return None
    candidates = sorted(
        (p for p in derived.glob("*/ndsm.tif") if "SUPERSEDED" not in str(p)),
        key=lambda p: p.parent.name,
    )
    return candidates[-1] if candidates else None


def build_roof_surface_from_cloud(
    points: np.ndarray,
    footprint: np.ndarray,
    cell_m: float = 0.5,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Triangule une toiture directement depuis le nuage, sans raster dérivé.

    Le nDSM du pilote ne couvre que le bâtiment cible : ses vingt-sept voisins
    n'avaient donc aucune surface de toit, et se fermaient par un cône. Or les
    quatre tuiles acquises les contiennent tous.

    La grille est construite à la volée depuis les retours de classe bâtiment —
    même maille, même fermeture aux ruptures que `build_roof_surface`, sans
    passer par un raster intermédiaire qu'il faudrait produire et qualifier.
    """
    import shapely
    from shapely.geometry import Polygon

    polygon = Polygon(footprint)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty or len(points) < MIN_CELLS:
        return None

    inside = shapely.contains_xy(polygon, points[:, 0], points[:, 1])
    if inside.sum() < MIN_CELLS:
        return None
    selected = points[inside]

    minx, miny, maxx, maxy = polygon.bounds
    cols = int(np.ceil((maxx - minx) / cell_m)) + 1
    rows = int(np.ceil((maxy - miny) / cell_m)) + 1
    if cols < 3 or rows < 3:
        return None

    # Hauteur maximale par cellule : la toiture, non ce qui passe dessous.
    # Le maximum s'accumule sur `-inf`, non sur NaN : `max(nan, x)` reste NaN
    # et la grille ressortait vide malgré quinze mille retours.
    grid = np.full((rows, cols), -np.inf)
    col = np.clip(((selected[:, 0] - minx) / cell_m).astype(int), 0, cols - 1)
    row = np.clip(((selected[:, 1] - miny) / cell_m).astype(int), 0, rows - 1)
    np.maximum.at(grid, (row, col), selected[:, 2])

    filled = np.isfinite(grid)
    grid = np.where(filled, grid, np.nan)
    if filled.sum() < MIN_CELLS:
        return None

    # Les trous sont comblés par le plus proche voisin mesuré, comme pour le
    # nDSM : une cellule vide au milieu d'un toit n'est pas un trou du toit.
    if not filled.all():
        from scipy.ndimage import distance_transform_edt

        _, nearest = distance_transform_edt(~filled, return_indices=True)
        grid = grid[tuple(nearest)]

    x_edges = minx + np.arange(cols + 1, dtype=np.float64) * cell_m
    y_edges = miny + np.arange(rows + 1, dtype=np.float64) * cell_m
    return _vectorized_grid_surface(grid, polygon, x_edges, y_edges)


#: Maille de toiture au premier plan, en mètres. C'est la finesse que le nuage
#: soutient réellement : en deçà, on interpole entre des retours absents.
ROOF_CELL_NEAR_M = 0.5

#: Maille au-delà de laquelle affiner ne se voit plus. Un toit à six cents
#: mètres n'occupe que quelques dizaines de pixels : le décrire au demi-mètre
#: revient à calculer des triangles d'un tiers de pixel.
ROOF_CELL_FAR_M = 4.0

#: Distance à partir de laquelle la maille commence à s'élargir.
ROOF_DETAIL_RADIUS_M = 60.0


def _roof_cell_for(prism, centre: tuple[float, float]) -> float:  # noqa: ANN001
    """Finesse de toiture méritée par un volume, selon son éloignement.

    La cible garde la maille fine quoi qu'il arrive : c'est le sujet du plan.
    Les autres s'élargissent avec la distance, parce qu'un triangle projeté
    plus petit qu'un pixel coûte un appel de rastérisation entier pour ne rien
    dessiner de visible. Mesuré sur ce pilote, les voisins portaient
    quatre-vingt-dix-huit pour cent des triangles de la scène.
    """
    if prism.is_target:
        return ROOF_CELL_NEAR_M

    footprint = prism.footprint
    distance = float(
        np.hypot(footprint[:, 0] - centre[0], footprint[:, 1] - centre[1]).min()
    )
    if distance <= ROOF_DETAIL_RADIUS_M:
        return ROOF_CELL_NEAR_M

    # La taille apparente décroît comme l'inverse de la distance : la maille
    # suit la même loi, pour que le triangle projeté garde son ordre de
    # grandeur en pixels d'un bout à l'autre de la scène.
    scaled = ROOF_CELL_NEAR_M * distance / ROOF_DETAIL_RADIUS_M
    return float(min(scaled, ROOF_CELL_FAR_M))


def apply_cloud_roofs(scene, laz_path, ground_z: float) -> dict:  # noqa: ANN001
    """Donne une toiture triangulée aux volumes que le nDSM ne couvre pas.

    Plusieurs tuiles sont acceptées : un site déborde de celle qui porte son
    bâtiment, et se limiter à elle laissait les voisins sans toiture — le cas
    même que cette fonction doit résoudre.
    """
    from .laz_cache import read_window

    tiles = (
        [Path(laz_path)]
        if isinstance(laz_path, (str, Path))
        else [Path(t) for t in laz_path]
    )
    tiles = [t for t in tiles if t.is_file()]
    if not tiles:
        return {"built": 0, "total": len(scene.prisms)}

    pending = [p for p in scene.prisms if not p.roof_measured]
    if not pending:
        return {"built": 0, "total": len(scene.prisms)}

    span = 0.0
    for prism in pending:
        centre = prism.footprint.mean(axis=0)
        span = max(
            span,
            float(np.hypot(centre[0] - scene.centre[0], centre[1] - scene.centre[1]))
            + float(
                np.hypot(
                    prism.footprint[:, 0] - centre[0],
                    prism.footprint[:, 1] - centre[1],
                ).max()
            ),
        )

    built = 0
    remaining = list(pending)
    for tile in tiles:
        if not remaining:
            break
        window = read_window(tile, scene.centre, span + 15.0)
        if window is None:
            continue
        built_mask = window.classification == BUILDING_CLASS
        if not built_mask.any():
            continue

        cloud = np.c_[
            window.x[built_mask], window.y[built_mask], window.z[built_mask] - ground_z
        ]

        still: list = []
        for prism in remaining:
            surface = build_roof_surface_from_cloud(
                cloud, prism.footprint, cell_m=_roof_cell_for(prism, scene.centre)
            )
            if surface is None:
                still.append(prism)
                continue
            prism.roof_vertices, prism.roof_faces = surface
            built += 1
        remaining = still

    log.info("toitures depuis le nuage : %d volume(s)", built)
    return {"built": built, "total": len(scene.prisms)}
