"""Environnement d'un site : végétation, sol et bâtiments liés.

Un plan aérien d'hôtel montre rarement le seul bâtiment. Les arbres qui bordent
l'allée, la haie du stationnement, l'aile annexe reliée au corps principal : ce
sont eux qui donnent l'échelle et masquent une partie de la façade. Les ignorer
produit un volume flottant dans le vide, que le générateur habillera d'une
végétation inventée — et donc fausse aux mauvais endroits.

Deux sources, et une leçon tirée du pilote pour chacune.

**Le LiDAR.** La tuile ne porte aucune classe de végétation : les classes 3, 4
et 5 sont absentes, et vingt-six pour cent des points restent « non classé ».
La végétation y est pourtant — six cent quatre-vingt mille points entre trente
centimètres et quinze mètres au-dessus du sol. Elle se retrouve par la hauteur,
non par l'étiquette.

**OpenStreetMap.** La requête du pipeline ne demande ni `natural`, ni
`landuse`, ni `leisure` : aucun massif, aucune pelouse n'entre donc dans le
manifeste. Les emprises existent en amont, elles ne sont simplement jamais
réclamées.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..logging import get_logger

log = get_logger("conditioning-environment")

#: Hauteur minimale au-dessus du sol pour qu'un point compte comme végétation.
#: En deçà, c'est du bruit de terrain ou du mobilier bas.
MIN_VEGETATION_M = 0.4

#: Au-delà, un point isolé est un artefact : ni arbre ni bâtiment plausible.
MAX_VEGETATION_M = 40.0

#: Strates de végétation, en mètres. Elles ne décrivent pas des espèces mais
#: des rôles visuels : ce qui borde une allée, ce qui masque un étage, ce qui
#: dépasse la toiture.
STRATA: tuple[tuple[str, float, float], ...] = (
    ("arbustes", 0.4, 2.0),
    ("petits_arbres", 2.0, 6.0),
    ("arbres_matures", 6.0, 15.0),
    ("arbres_hauts", 15.0, 40.0),
)

#: Taille de cellule pour agréger les points en amas, en mètres. Deux mètres
#: séparent deux arbres voisins sans éclater une couronne en morceaux.
CLUSTER_CELL_M = 2.0

#: En deçà, un amas de cellules relève du bruit plutôt que d'un massif.
MIN_CLUSTER_CELLS = 3

#: Diamètre maximal d'un massif, en mètres. Sans ce plafond, la fusion des
#: cellules contiguës relie de proche en proche toute une bande arborée : un
#: premier essai rendait un massif de cent soixante-treize mètres de rayon,
#: qui n'occulte rien de précis et ne borne donc rien d'utile. Un massif trop
#: large est redécoupé en cellules de cette taille.
MAX_CLUSTER_SPAN_M = 24.0

#: Rayon minimal de classification du sol, en mètres. Il dépasse celui de la
#: végétation parce qu'un sol absent se voit — le regard porte loin à l'horizon
#: — alors qu'un arbre manquant à cent mètres ne manque à personne.
GROUND_RADIUS_M = 160.0

#: Distance en deçà de laquelle un bâtiment voisin est tenu pour *lié* au site
#: — aile annexe, dépendance, abri d'entrée — plutôt que simple voisin.
#: Réglé à trente-cinq mètres après mesure : sur ce pilote, le bâti le plus
#: proche est à trente mètres, de l'autre côté du stationnement mais dans le
#: même ensemble visuel, tandis que le suivant est à cent mètres. La frontière
#: naturelle du site tombe entre les deux.
LINKED_BUILDING_M = 35.0


@dataclass
class VegetationPatch:
    """Un massif végétal, décrit par son emprise et sa hauteur."""

    stratum: str
    centre: tuple[float, float]
    radius_m: float
    height_m: float
    points: int
    #: Allure déduite des images au sol : conique, etale, colonnaire, arbustif.
    #: `None` tant qu'aucune vue n'a été lue — le rendu pose alors un cylindre.
    shape: str | None = None
    #: Anneaux LiDAR du houppier. Préférés au profil d'allure lorsqu'ils
    #: existent : l'image peut qualifier la forme, mais ne remplace pas la
    #: distribution 3D mesurée.
    envelope: list[list[tuple[float, float, float]]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "stratum": self.stratum,
            "shape": self.shape,
            "centre": [round(c, 2) for c in self.centre],
            "radius_m": round(self.radius_m, 2),
            "height_m": round(self.height_m, 2),
            "points": self.points,
            "envelope": [
                [[round(x, 2), round(y, 2), round(z, 2)] for x, y, z in ring]
                for ring in self.envelope
            ],
        }


@dataclass
class LinkedBuilding:
    """Un bâtiment tenu pour lié au site, et ce qui justifie ce lien."""

    feature_id: str
    distance_m: float
    shares_parcel: bool
    reason: str

    def as_dict(self) -> dict:
        return {
            "feature_id": self.feature_id,
            "distance_m": round(self.distance_m, 2),
            "shares_parcel": self.shares_parcel,
            "reason": self.reason,
        }


@dataclass
class StreetFurniture:
    """Un objet vertical fin : lampadaire, mât d'enseigne, panneau.

    Il n'est pas de la végétation et ne doit pas être rendu comme telle, mais
    il occulte bel et bien : un mât devant une façade se voit sur un plan.
    """

    centre: tuple[float, float]
    radius_m: float
    height_m: float

    def as_dict(self) -> dict:
        return {
            "centre": [round(c, 2) for c in self.centre],
            "radius_m": round(self.radius_m, 2),
            "height_m": round(self.height_m, 2),
        }


@dataclass
class SiteEnvironment:
    """Ce qui entoure le bâtiment, et d'où chaque élément vient."""

    hotel_id: str
    patches: list[VegetationPatch] = field(default_factory=list)
    furniture: list[StreetFurniture] = field(default_factory=list)
    linked: list[LinkedBuilding] = field(default_factory=list)
    #: Cellules de sol classées par nature, quand le nuage le permet.
    ground_cells: list = field(default_factory=list)
    #: Plages de sol en polygones, préférées aux cellules pour le rendu :
    #: elles donnent des bordures nettes là où un damier montre des créneaux.
    ground_patches: list = field(default_factory=list)
    ground_cell_m: float = 1.0
    #: Relief du terrain, quand un modèle le porte. `None` = sol plat.
    terrain: object | None = None
    ground_z: float | None = None
    provenance: dict = field(default_factory=dict)

    def by_stratum(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for patch in self.patches:
            counts[patch.stratum] = counts.get(patch.stratum, 0) + 1
        return counts

    def as_dict(self) -> dict:
        return {
            "hotel_id": self.hotel_id,
            "vegetation_count": len(self.patches),
            "furniture_count": len(self.furniture),
            "ground_cells": len(self.ground_cells),
            "ground_patches": len(self.ground_patches),
            "terrain_relief_m": (
                None if self.terrain is None else round(self.terrain.relief_m, 2)
            ),
            "by_stratum": self.by_stratum(),
            "linked_buildings": [b.as_dict() for b in self.linked],
            "ground_z": None if self.ground_z is None else round(self.ground_z, 2),
            "provenance": self.provenance,
            "patches": [p.as_dict() for p in self.patches],
            "furniture": [f.as_dict() for f in self.furniture],
            "caveats": [
                "la végétation est déduite de la hauteur des points non "
                "classés : cette tuile ne porte aucune classe végétation",
                "les objets fins, hauts et continus du sol au sommet sont "
                "classés comme mobilier — lampadaire, mât, panneau — et non "
                "comme végétation, sur leur seule signature géométrique",
                "un massif décrit un volume occupé, pas une espèce ni un "
                "feuillage : il borne ce que le générateur peut y placer",
                "la végétation est saisonnière — un relevé d'hiver ne montre "
                "pas la même occultation qu'un feuillage d'été",
            ],
        }


def _stratum_of(height: float) -> str | None:
    for name, low, high in STRATA:
        if low <= height < high:
            return name
    return None


def _cluster_cells(
    xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, cell: float
) -> list[tuple[tuple[float, float], float, float, int]]:
    """Regroupe des points en cellules, puis les cellules voisines en amas.

    Un simple découpage en grille suffit ici : on ne cherche pas à isoler des
    arbres individuels, mais des volumes occupés. Une couronne étalée sur
    quatre cellules donne un massif, non quatre arbustes.
    """
    if xs.size == 0:
        return []

    ix = np.floor(xs / cell).astype(np.int64)
    iy = np.floor(ys / cell).astype(np.int64)
    keys = {}
    for index in range(xs.size):
        keys.setdefault((int(ix[index]), int(iy[index])), []).append(index)

    occupied = {k: v for k, v in keys.items() if len(v) >= 4}
    if not occupied:
        return []

    # Fusion des cellules contiguës, en huit-connexité.
    seen: set[tuple[int, int]] = set()
    clusters: list[tuple[tuple[float, float], float, float, int]] = []
    for start in occupied:
        if start in seen:
            continue
        stack = [start]
        members: list[tuple[int, int]] = []
        seen.add(start)
        while stack:
            cx, cy = stack.pop()
            members.append((cx, cy))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbour = (cx + dx, cy + dy)
                    if neighbour in occupied and neighbour not in seen:
                        seen.add(neighbour)
                        stack.append(neighbour)

        if len(members) < MIN_CLUSTER_CELLS:
            continue

        indices = [i for m in members for i in occupied[m]]
        px, py, pz = xs[indices], ys[indices], zs[indices]
        span = max(px.max() - px.min(), py.max() - py.min())

        # Un amas trop étendu est redécoupé : la connexité seule relierait une
        # haie continue et un boisé voisin en un unique volume, sans frontière
        # utile au rendu.
        if span > MAX_CLUSTER_SPAN_M:
            step = MAX_CLUSTER_SPAN_M
            bx = np.floor(px / step).astype(np.int64)
            by = np.floor(py / step).astype(np.int64)
            for tile in {(int(a), int(b)) for a, b in zip(bx, by)}:
                sel = (bx == tile[0]) & (by == tile[1])
                if sel.sum() < 8:
                    continue
                clusters.append(_describe(px[sel], py[sel], pz[sel], cell))
            continue

        clusters.append(_describe(px, py, pz, cell))
    return clusters


def _describe(
    px: np.ndarray, py: np.ndarray, pz: np.ndarray, cell: float
) -> tuple[tuple[float, float], float, float, int]:
    """Centre, rayon et hauteur d'un groupe de points.

    Le rayon est celui du **disque de même aire** que la surface réellement
    occupée, non celui du cercle circonscrit. Mesuré sur ce pilote, le cercle
    circonscrit d'amas allongés — une haie, une rangée d'arbres — couvrait deux
    fois et demie la surface de la zone étudiée : les massifs se chevauchaient
    partout et noyaient le bâtiment sous un tapis de volumes.
    """
    centre = (float(px.mean()), float(py.mean()))

    # Aire occupée : nombre de cellules distinctes réellement remplies.
    ix = np.floor(px / cell).astype(np.int64)
    iy = np.floor(py / cell).astype(np.int64)
    occupied_cells = len({(int(a), int(b)) for a, b in zip(ix, iy)})
    equivalent = float(np.sqrt(occupied_cells * cell * cell / np.pi))

    radius = float(max(equivalent, cell * 0.5))
    # Le p90 plutôt que le maximum : un point aberrant au-dessus d'une
    # couronne étirerait tout le massif vers le ciel.
    height = float(np.percentile(pz, 90))
    return centre, radius, height, int(px.size)


def extract_vegetation(
    laz_path: Path,
    centre: tuple[float, float],
    radius_m: float = 150.0,
    cell_m: float = CLUSTER_CELL_M,
    footprints: list | None = None,
    keep_radius_m: float | None = None,
) -> tuple[list[VegetationPatch], float | None, list[StreetFurniture]]:
    """Retrouve les massifs végétaux autour d'un site, par leur hauteur.

    La classification de la tuile ne distingue pas la végétation : elle est
    donc reconstruite depuis les points non classés, mesurés au-dessus du sol
    local. C'est une déduction, et le rapport la présente comme telle.
    """
    from .laz_cache import read_window

    laz_path = Path(laz_path)
    if not laz_path.is_file():
        log.info("tuile LiDAR absente : %s", laz_path)
        return [], None, []

    window = read_window(laz_path, centre, radius_m)
    if window is None:
        log.info("aucun point dans l'emprise")
        return [], None, []

    # Classe 2 : le sol. Classe 1 : non classé, où vit la végétation faute
    # d'étiquette dédiée dans cette tuile.
    ground_mask = window.classification == 2
    loose_mask = window.classification == 1
    ground_z = [window.z[ground_mask]] if ground_mask.any() else []
    unclassified_x = [window.x[loose_mask]] if loose_mask.any() else []
    unclassified_y = [window.y[loose_mask]] if loose_mask.any() else []
    unclassified_z = [window.z[loose_mask]] if loose_mask.any() else []

    if not ground_z:
        log.info("aucun point de sol dans l'emprise : hauteurs indéterminables")
        return [], None, []

    base = float(np.median(np.concatenate(ground_z)))
    if not unclassified_x:
        return [], base, []

    xs = np.concatenate(unclassified_x)
    ys = np.concatenate(unclassified_y)
    zs = np.concatenate(unclassified_z) - base

    usable = (zs >= MIN_VEGETATION_M) & (zs <= MAX_VEGETATION_M)
    if keep_radius_m is not None and keep_radius_m < radius_m:
        # La fenêtre lue est plus large que le rayon voulu : elle sert aussi au
        # sol. La végétation, elle, se limite aux abords du site.
        usable &= (np.abs(xs - centre[0]) <= keep_radius_m) & (
            np.abs(ys - centre[1]) <= keep_radius_m
        )
    xs, ys, zs = xs[usable], ys[usable], zs[usable]

    # Ce qui se dresse **sur** un toit n'est pas de la végétation : cheminées,
    # édicules d'ascenseur, unités de ventilation. Mesuré sur ce pilote, vingt
    # et un objets tombaient dans l'emprise du bâtiment cible et sortaient en
    # « arbres matures » jusqu'à quatorze mètres — c'est-à-dire des cheminées
    # plantées sur la toiture.
    if footprints:
        import shapely
        from shapely.geometry import Polygon

        on_roof = np.zeros(xs.size, dtype=bool)
        for ring in footprints:
            polygon = Polygon(ring)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if polygon.is_empty:
                continue
            on_roof |= shapely.contains_xy(polygon, xs, ys)
        if on_roof.any():
            log.info(
                "%d point(s) écarté(s) : superstructures de toiture, non végétation",
                int(on_roof.sum()),
            )
            xs, ys, zs = xs[~on_roof], ys[~on_roof], zs[~on_roof]

    # La segmentation par couronnes remplace le regroupement par connexité :
    # celui-ci fusionnait une rangée d'arbres en un bloc de mille mètres
    # carrés, rendu comme un pavé vert sans rapport avec un arbre.
    from .canopy import segment

    patches: list[VegetationPatch] = []
    furniture: list[StreetFurniture] = []
    for item in segment(xs, ys, zs):
        if item.kind == "poteau":
            furniture.append(
                StreetFurniture(
                    centre=item.centre,
                    radius_m=item.radius_m,
                    height_m=item.height_m,
                )
            )
            continue
        stratum = _stratum_of(item.height_m) or "arbustes"
        patches.append(
            VegetationPatch(
                stratum=stratum,
                centre=item.centre,
                radius_m=item.radius_m,
                height_m=item.height_m,
                points=item.points,
                envelope=item.envelope,
            )
        )

    log.info(
        "végétation : %d couronne(s), %d objet(s) de mobilier",
        len(patches),
        len(furniture),
    )
    return patches, base, furniture


def find_linked_buildings(scene, max_distance_m: float = LINKED_BUILDING_M) -> list[LinkedBuilding]:
    """Distingue les bâtiments *liés* au site de ses simples voisins.

    Un obstacle collé au bâtiment cible — aile annexe, abri d'entrée, garage —
    ne joue pas le même rôle qu'un immeuble de bureaux à cent mètres : il
    appartient visuellement à l'établissement, et une vidéo qui l'omet montre un
    autre lieu. La distinction se fait sur la distance entre emprises, non sur
    un tag : le manifeste ne porte aucune notion d'appartenance.
    """
    from shapely.geometry import Polygon

    target = scene.target
    if target is None:
        return []

    target_shape = Polygon(target.footprint)
    if not target_shape.is_valid:
        target_shape = target_shape.buffer(0)

    linked: list[LinkedBuilding] = []
    for prism in scene.prisms:
        if prism.is_target:
            continue
        shape = Polygon(prism.footprint)
        if not shape.is_valid:
            shape = shape.buffer(0)
        distance = float(target_shape.distance(shape))
        if distance > max_distance_m:
            continue
        touching = distance < 0.5
        linked.append(
            LinkedBuilding(
                feature_id=prism.feature_id,
                distance_m=distance,
                shares_parcel=touching,
                reason=(
                    "emprise jointive au bâtiment cible : aile ou annexe"
                    if touching
                    else f"à {distance:.1f} m du bâtiment cible, dans son ensemble bâti"
                ),
            )
        )

    linked.sort(key=lambda b: b.distance_m)
    log.info("bâtiments liés : %d", len(linked))
    return linked


def build(
    scene,
    laz_path: Path | None = None,
    radius_m: float = 60.0,
    dtm_path: Path | None = None,
) -> SiteEnvironment:
    """Compose l'environnement d'un site : végétation, sol, bâti lié.

    Le rayon par défaut couvre les abords immédiats, non le quartier. Un premier
    essai à cent vingt mètres remplissait le cadre de massifs lointains qui
    n'occultent jamais la façade mais bouchent l'avant-plan : ce qui compte pour
    un plan d'établissement, c'est ce qui pousse **sur** le site.
    """
    environment = SiteEnvironment(hotel_id=scene.hotel_id)
    environment.linked = find_linked_buildings(scene)

    if laz_path is not None:
        # La végétation et le sol lisent la même fenêtre : demander la plus
        # large des deux emprises permet de ne parcourir la tuile qu'une fois.
        # La végétation reste filtrée à son propre rayon juste après.
        window_radius = max(radius_m, GROUND_RADIUS_M)
        patches, ground, furniture = extract_vegetation(
            laz_path,
            scene.centre,
            window_radius,
            footprints=[p.footprint for p in scene.prisms],
            keep_radius_m=radius_m,
        )
        environment.patches = patches
        environment.furniture = furniture
        environment.ground_z = ground
        # Le sol : où finit la pelouse, où commence l'asphalte. OpenStreetMap
        # ne le dit pas aux abords de ce site, le retour LiDAR si.
        from .surface_lidar import classify_ground

        # Le sol est classé plus loin que la végétation : la caméra orbite à
        # une distance comparable au rayon du bâtiment, et son champ balaie le
        # terrain bien au-delà. Mesuré sur ce pilote, un rayon de soixante
        # mètres laissait le sol s'arrêter à soixante-dix-huit mètres du centre
        # alors que le regard portait à plus de cent cinquante : le vide au
        # sol se lisait comme des trous dans la chaussée.
        surface = classify_ground(
            laz_path,
            scene.centre,
            radius_m=max(radius_m, GROUND_RADIUS_M),
            footprints=[p.footprint for p in scene.prisms],
        )
        environment.ground_cells = surface.cells
        environment.ground_cell_m = surface.cell_m

        from .ground_polygons import from_cells

        environment.ground_patches = from_cells(
            surface.cells, surface.cell_m, scene.hotel_id
        ).patches
        environment.provenance["ground_by_kind"] = surface.by_kind()
        environment.provenance["laz"] = str(laz_path)
        environment.provenance["vegetation_method"] = (
            "points non classés, hauteur au-dessus du sol médian local — "
            "cette tuile ne porte aucune classe végétation"
        )
    else:
        environment.provenance["vegetation_method"] = (
            "aucune tuile LiDAR : la végétation n'est pas relevée"
        )

    if dtm_path is not None:
        from .terrain import load as load_terrain

        environment.terrain = load_terrain(dtm_path, scene.centre, GROUND_RADIUS_M)

    environment.provenance["radius_m"] = radius_m
    return environment
