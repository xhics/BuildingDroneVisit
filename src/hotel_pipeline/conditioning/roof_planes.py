"""Segmentation de la toiture en plans, à la manière des modèles LOD2.

La surface de toit était triangulée telle quelle depuis le nDSM : chaque
cellule devenait un sommet, et le bruit du relevé se retrouvait dans la
géométrie. Surtout, rien ne reconnaissait qu'un toit est fait de **plans** —
des versants qui se rencontrent sur un faîtage.

Mesuré sur ce pilote : la pente locale médiane atteint vingt-trois degrés, et
la croissance de région isole des versants nets entre huit et douze mètres
d'altitude. Le bâtiment a bien deux pans, ce que la photographie confirme et
que la triangulation directe rendait comme une nappe irrégulière.

La méthode suit l'état de l'art en reconstruction LOD2 : normales estimées par
analyse en composantes principales locale, puis croissance de région sous
double contrainte — orientation voisine et coplanarité.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..logging import get_logger

log = get_logger("conditioning-roof-planes")

#: Voisins retenus pour estimer une normale locale. Douze lissent le bruit du
#: relevé sans effacer une arête de faîtage.
NORMAL_NEIGHBOURS = 12

#: Écart d'orientation toléré dans un même plan, en degrés.
NORMAL_TOLERANCE_DEG = 12.0

#: Écart au plan toléré, en mètres. Une toiture réelle n'est jamais
#: parfaitement plane : tuiles, ondulation, bruit du capteur.
PLANE_DISTANCE_M = 0.35

#: Rayon de propagation d'une région, en mètres.
GROWTH_RADIUS_M = 1.2

#: En deçà, un groupe de points ne décrit pas un pan de toiture.
MIN_PLANE_POINTS = 40

#: Pas de sous-échantillonnage, en mètres. La segmentation ne gagne rien à
#: travailler sur des retours redondants.
SAMPLE_STEP_M = 0.5


@dataclass
class RoofPlane:
    """Un pan de toiture, décrit par son orientation et son étendue."""

    points: np.ndarray
    normal: np.ndarray
    origin: np.ndarray

    @property
    def slope_deg(self) -> float:
        return float(np.degrees(np.arccos(abs(float(self.normal[2])))))

    @property
    def area_m2(self) -> float:
        """Emprise au sol du pan, en mètres carrés."""
        if len(self.points) < 3:
            return 0.0
        from shapely.geometry import MultiPoint

        return float(MultiPoint(self.points[:, :2]).convex_hull.area)

    def height_at(self, x: float, y: float) -> float:
        """Altitude du plan à l'aplomb d'un point."""
        if abs(self.normal[2]) < 1e-6:
            return float(self.origin[2])
        return float(
            self.origin[2]
            - (
                self.normal[0] * (x - self.origin[0])
                + self.normal[1] * (y - self.origin[1])
            )
            / self.normal[2]
        )

    def as_dict(self) -> dict:
        return {
            "points": int(len(self.points)),
            "slope_deg": round(self.slope_deg, 1),
            "area_m2": round(self.area_m2, 1),
            "z_min": round(float(self.points[:, 2].min()), 2),
            "z_max": round(float(self.points[:, 2].max()), 2),
        }


@dataclass
class RoofDecomposition:
    """Les pans d'une toiture, et ce que la segmentation n'a pas expliqué."""

    feature_id: str
    planes: list[RoofPlane] = field(default_factory=list)
    unassigned: int = 0
    total: int = 0

    @property
    def explained(self) -> float:
        return 0.0 if self.total == 0 else 1.0 - self.unassigned / self.total

    @property
    def pitched(self) -> list[RoofPlane]:
        """Pans en pente franche : ce sont eux qui font la forme du toit."""
        return [p for p in self.planes if p.slope_deg >= 8.0]

    def as_dict(self) -> dict:
        return {
            "feature_id": self.feature_id,
            "planes": len(self.planes),
            "pitched_planes": len(self.pitched),
            "explained": round(self.explained, 3),
            "detail": [p.as_dict() for p in sorted(
                self.planes, key=lambda p: -p.area_m2
            )[:8]],
            "caveats": [
                "un pan est ajusté sur des retours aériens : sa pente est "
                "mesurée, son contour reste celui du nuage, pas une arête "
                "vectorisée",
                "les points inexpliqués sont ceux qu'aucun plan n'absorbe — "
                "superstructures, bordures, bruit — et non une erreur",
            ],
        }


def _normals(points: np.ndarray, neighbours: int) -> np.ndarray:
    """Normales locales par analyse en composantes principales."""
    from scipy.spatial import cKDTree

    tree = cKDTree(points[:, :2])
    _, index = tree.query(points[:, :2], k=min(neighbours, len(points)))
    normals = np.zeros((len(points), 3))
    for position, group in enumerate(np.atleast_2d(index)):
        cloud = points[group] - points[group].mean(axis=0)
        normals[position] = np.linalg.svd(cloud, full_matrices=False)[2][2]
    # Orientées vers le haut : un toit se regarde du ciel.
    return normals * np.sign(normals[:, 2])[:, None]


def segment(
    points: np.ndarray,
    feature_id: str = "unknown",
    sample_step_m: float = SAMPLE_STEP_M,
) -> RoofDecomposition:
    """Décompose un nuage de toiture en pans coplanaires.

    Les points sont d'abord ramenés à une grille : segmenter des retours
    redondants coûte sans rien préciser. La croissance part ensuite des zones
    les plus planes, qui donnent les germes les plus fiables.
    """
    from scipy.spatial import cKDTree

    result = RoofDecomposition(feature_id=feature_id, total=len(points))
    if len(points) < MIN_PLANE_POINTS * 2:
        return result

    # Sous-échantillonnage : un point par cellule de grille.
    keys = (
        np.floor(points[:, 0] / sample_step_m).astype(np.int64) * 1_000_003
        + np.floor(points[:, 1] / sample_step_m).astype(np.int64)
    )
    _, first = np.unique(keys, return_index=True)
    sampled = points[first]
    if len(sampled) < MIN_PLANE_POINTS * 2:
        return result

    normals = _normals(sampled, NORMAL_NEIGHBOURS)
    tree = cKDTree(sampled[:, :2])
    labels = np.full(len(sampled), -1)
    cosine = np.cos(np.radians(NORMAL_TOLERANCE_DEG))

    # Les zones planes d'abord : leurs normales sont les mieux estimées.
    for seed in np.argsort(-normals[:, 2]):
        if labels[seed] >= 0:
            continue
        reference_normal = normals[seed]
        reference_point = sampled[seed]
        stack = [int(seed)]
        members = [int(seed)]
        labels[seed] = -2  # en cours

        while stack:
            current = stack.pop()
            for candidate in tree.query_ball_point(
                sampled[current, :2], GROWTH_RADIUS_M
            ):
                if labels[candidate] != -1:
                    continue
                if abs(float(np.dot(normals[candidate], reference_normal))) < cosine:
                    continue
                gap = abs(
                    float(np.dot(sampled[candidate] - reference_point, reference_normal))
                )
                if gap > PLANE_DISTANCE_M:
                    continue
                labels[candidate] = -2
                stack.append(candidate)
                members.append(candidate)

        if len(members) < MIN_PLANE_POINTS:
            labels[members] = -1
            labels[seed] = -3  # germe épuisé, ne pas reprendre
            continue

        selected = sampled[members]
        normal = selected - selected.mean(axis=0)
        fitted = np.linalg.svd(normal, full_matrices=False)[2][2]
        fitted *= np.sign(fitted[2]) if fitted[2] != 0 else 1.0
        labels[members] = len(result.planes)
        result.planes.append(
            RoofPlane(
                points=selected,
                normal=fitted,
                origin=selected.mean(axis=0),
            )
        )

    result.unassigned = int((labels < 0).sum())
    result.total = len(sampled)
    log.info(
        "%s : %d pan(s) dont %d en pente, %.0f %% des points expliqués",
        feature_id,
        len(result.planes),
        len(result.pitched),
        result.explained * 100,
    )
    return result


def apply_to_roof(prism, decomposition: RoofDecomposition) -> int:
    """Réajuste la surface de toit d'un volume sur les pans détectés.

    Chaque sommet du maillage prend l'altitude du pan qui le couvre. Les
    versants deviennent alors francs — ce qu'une triangulation directe du
    modèle de hauteur rendait comme une nappe irrégulière, faîtage compris.

    Les sommets qu'aucun pan ne couvre gardent leur altitude d'origine : la
    segmentation corrige ce qu'elle explique, elle n'invente rien ailleurs.
    """
    if prism.roof_vertices is None or not decomposition.planes:
        return 0

    from scipy.spatial import cKDTree

    vertices = prism.roof_vertices
    adjusted = 0

    # Chaque pan est cherché parmi ceux dont l'emprise couvre le sommet ; à
    # défaut, le plus proche en projection horizontale l'emporte.
    trees = [
        (plane, cKDTree(plane.points[:, :2])) for plane in decomposition.planes
    ]

    for index, (x, y, z) in enumerate(vertices):
        best_plane = None
        best_distance = 2.0  # au-delà, le pan ne décrit plus ce point
        for plane, tree in trees:
            distance, _ = tree.query([x, y])
            if distance < best_distance:
                best_plane, best_distance = plane, float(distance)
        if best_plane is None:
            continue
        fitted = best_plane.height_at(float(x), float(y))
        # Un ajustement démesuré signale que le pan ne couvre pas ce sommet.
        if abs(fitted - z) > 2.5:
            continue
        vertices[index, 2] = fitted
        adjusted += 1

    log.info(
        "%s : %d/%d sommet(s) de toit réajusté(s) sur les pans",
        decomposition.feature_id,
        adjusted,
        len(vertices),
    )
    return adjusted


def apply_to_scene(scene, laz_path, ground_z: float) -> dict:  # noqa: ANN001
    """Segmente et réajuste la toiture de chaque volume qui en porte une.

    La fenêtre du nuage est lue une seule fois pour toute la scène : une
    lecture par volume coûterait plusieurs minutes pour le même résultat.
    """
    from pathlib import Path as _Path

    import shapely
    from shapely.geometry import Polygon

    from .laz_cache import read_window

    tiles = (
        [_Path(laz_path)]
        if isinstance(laz_path, (str, _Path))
        else [_Path(t) for t in laz_path]
    )
    tiles = [t for t in tiles if t.is_file()]
    if not tiles:
        return {"segmented": 0, "total": len(scene.prisms)}

    span = 0.0
    for prism in scene.prisms:
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

    segmented = 0
    adjusted_total = 0
    pitched_total = 0
    done: set[str] = set()

    for tile in tiles:
        window = read_window(tile, scene.centre, span + 15.0)
        if window is None:
            continue
        built = window.classification == 6
        if not built.any():
            continue
        xs, ys, zs = window.x[built], window.y[built], window.z[built] - ground_z
        segmented, adjusted_total, pitched_total = _segment_pass(
            scene, xs, ys, zs, done, segmented, adjusted_total, pitched_total
        )

    log.info(
        "toitures segmentées : %d volume(s), %d sommet(s) réajusté(s)",
        segmented,
        adjusted_total,
    )
    return {
        "segmented": segmented,
        "total": len(scene.prisms),
        "vertices_adjusted": adjusted_total,
        "pitched_planes": pitched_total,
    }


def _segment_pass(scene, xs, ys, zs, done, segmented, adjusted_total, pitched_total):  # noqa: ANN001
    """Segmente les volumes qu'une tuile couvre, en ignorant les faits."""
    import shapely
    from shapely.geometry import Polygon

    for prism in scene.prisms:
        if prism.roof_vertices is None or prism.feature_id in done:
            continue
        polygon = Polygon(prism.footprint)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty:
            continue
        inside = shapely.contains_xy(polygon, xs, ys)
        if inside.sum() < MIN_PLANE_POINTS * 2:
            continue

        # Le bâtiment cible mérite la maille fine ; ses voisins n'occupent
        # que quelques pour cent de l'image et se contentent d'un mètre. Sans
        # cette distinction, la segmentation de la scène passait de vingt
        # secondes à près de sept minutes.
        decomposition = segment(
            np.c_[xs[inside], ys[inside], zs[inside]],
            prism.feature_id,
            sample_step_m=SAMPLE_STEP_M if prism.is_target else 1.0,
        )
        if not decomposition.planes:
            continue
        adjusted = apply_to_roof(prism, decomposition)
        if adjusted:
            segmented += 1
            adjusted_total += adjusted
            pitched_total += len(decomposition.pitched)
            prism.roof_planes = decomposition
            done.add(prism.feature_id)

    return segmented, adjusted_total, pitched_total


#: Angle minimal entre deux pans pour qu'ils forment une arête franche.
#: En deçà, ils appartiennent au même versant vu sous deux germes.
RIDGE_MIN_ANGLE_DEG = 15.0

#: Distance en deçà de laquelle deux pans sont tenus pour adjacents, en mètres.
RIDGE_ADJACENCY_M = 2.0

#: Longueur minimale d'une arête retenue, en mètres.
RIDGE_MIN_LENGTH_M = 3.0


@dataclass
class RoofRidge:
    """Arête entre deux pans : faîtage, noue ou brisis."""

    start: np.ndarray
    end: np.ndarray
    angle_deg: float
    kind: str
    #: Indices des deux pans dont l'intersection produit cette arête. Ils
    #: disent quelles arêtes se rencontrent : deux arêtes partageant un pan
    #: sont voisines sur le toit, ce qu'aucune mesure de longueur ne révèle.
    plane_indices: tuple[int, int] = (-1, -1)

    @property
    def length_m(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "length_m": round(self.length_m, 2),
            "angle_deg": round(self.angle_deg, 1),
            "z_start": round(float(self.start[2]), 2),
            "z_end": round(float(self.end[2]), 2),
        }


def _intersect_planes(a: RoofPlane, b: RoofPlane) -> tuple[np.ndarray, np.ndarray] | None:
    """Droite d'intersection de deux plans, si elle existe."""
    direction = np.cross(a.normal, b.normal)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return None
    direction = direction / norm

    # Point de la droite : résolution du système des deux plans, complété par
    # une troisième équation orthogonale pour lever l'indétermination.
    matrix = np.stack([a.normal, b.normal, direction])
    offsets = np.array([
        float(np.dot(a.normal, a.origin)),
        float(np.dot(b.normal, b.origin)),
        0.0,
    ])
    try:
        point = np.linalg.solve(matrix, offsets)
    except np.linalg.LinAlgError:
        return None
    return point, direction


def ridges(decomposition: RoofDecomposition) -> list[RoofRidge]:
    """Arêtes de toiture, obtenues en intersectant les pans adjacents.

    C'est le pas que la reconstruction LOD2 ajoute à la segmentation : une
    arête n'est pas cherchée dans le nuage, où elle est floue, mais **déduite**
    de la rencontre de deux plans. Le faîtage devient alors une droite nette,
    là où un maillage ajusté l'arrondit toujours un peu.

    Les extrémités sont bornées par l'étendue commune des deux pans : la droite
    d'intersection est infinie, l'arête ne l'est pas.
    """
    from scipy.spatial import cKDTree

    planes = [p for p in decomposition.planes if len(p.points) >= MIN_PLANE_POINTS]
    found: list[RoofRidge] = []
    if len(planes) < 2:
        return found

    trees = [cKDTree(p.points[:, :2]) for p in planes]

    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            first, second = planes[i], planes[j]
            cosine = abs(float(np.dot(first.normal, second.normal)))
            angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
            if angle < RIDGE_MIN_ANGLE_DEG:
                continue

            # Adjacence : les deux pans doivent se toucher quelque part.
            close = trees[j].query(first.points[:, :2])[0]
            touching = close <= RIDGE_ADJACENCY_M
            if touching.sum() < 8:
                continue

            line = _intersect_planes(first, second)
            if line is None:
                continue
            point, direction = line

            # Bornes : projection des points de contact sur la droite.
            contact = first.points[touching]
            t = (contact - point) @ direction
            start = point + direction * float(t.min())
            end = point + direction * float(t.max())

            ridge = RoofRidge(
                plane_indices=(i, j),
                start=start,
                end=end,
                angle_deg=angle,
                kind=(
                    "faitage"
                    if abs(direction[2]) < 0.2 and first.slope_deg > 8
                    else "brisis"
                ),
            )
            if ridge.length_m >= RIDGE_MIN_LENGTH_M:
                found.append(ridge)

    found.sort(key=lambda r: -r.length_m)
    log.info("%s : %d arête(s) de toiture", decomposition.feature_id, len(found))
    return found
