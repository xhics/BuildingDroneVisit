"""Rasteriseur z-buffer autonome, sans dépendance graphique.

Le rendu ne cherche pas la beauté : il cherche la justesse géométrique. Ce que
le générateur consomme, c'est la profondeur, la normale, la silhouette et le
crédit accordé à chaque pixel — pas une texture.

Tout tient en numpy pour que le harnais tourne sur une machine sans OpenGL, et
que la comparaison A/B ne dépende pas d'un environnement graphique.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .scene import ConditioningScene, Prism


#: Crédit accordé au sol classé. La nature vient d'une déduction sur
#: l'intensité du retour, non d'un relevé de matériau : elle oriente le
#: générateur sans prétendre décrire la surface.
GROUND_CONFIDENCE = 0.40

#: Crédit accordé au mobilier urbain. Sa forme est mesurée mais grossière : un
#: cylindre à la place d'un lampadaire borne l'occultation, pas l'apparence.
FURNITURE_CONFIDENCE = 0.30

#: Crédit accordé à un massif végétal. Un encombrement mesuré vaut mieux que
#: rien, mais il ne décrit pas la forme réelle d'un feuillage.
VEGETATION_CONFIDENCE = 0.35


@dataclass
class Camera:
    """Pose de prise de vue, en CRS projeté, axe Z vers le haut."""

    position: np.ndarray
    target: np.ndarray
    fov_deg: float = 60.0
    width: int = 512
    height: int = 288

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Repère caméra : avant, droite, haut.

        Mémorisé : il ne dépend que de la pose, et il était recalculé une fois
        par triangle — quinze mille fois par image sur ce pilote, pour trois
        produits vectoriels toujours identiques.
        """
        cached = getattr(self, "_basis", None)
        if cached is not None:
            return cached
        found = self._compute_basis()
        object.__setattr__(self, "_basis", found)
        return found

    def axes(self) -> np.ndarray:
        """Matrice (3, 3) dont les colonnes sont droite, haut et avant."""
        cached = getattr(self, "_axes", None)
        if cached is not None:
            return cached
        forward, right, up = self.basis()
        matrix = np.stack([right, up, forward], axis=1)
        object.__setattr__(self, "_axes", matrix)
        return matrix

    def _compute_basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        forward = self.target - self.position
        norm = np.linalg.norm(forward)
        if norm < 1e-9:
            raise ValueError("caméra et cible confondues : direction indéfinie")
        forward = forward / norm
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(forward, world_up))) > 0.999:
            world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        return forward, right, up


@dataclass
class RenderedFrame:
    """Ce qu'une pose établit, canal par canal."""

    depth: np.ndarray
    normal: np.ndarray
    silhouette: np.ndarray
    confidence: np.ndarray
    #: Fraction de pixels couverts par le bâtiment cible.
    target_coverage: float
    #: Fraction de pixels dont la géométrie repose sur une hauteur supposée.
    assumed_fraction: float
    hit_any: bool

    def stats(self) -> dict:
        finite = self.depth[np.isfinite(self.depth)]
        return {
            "target_coverage": round(self.target_coverage, 4),
            "assumed_fraction": round(self.assumed_fraction, 4),
            "geometry_hit": bool(self.hit_any),
            "depth_min_m": round(float(finite.min()), 2) if finite.size else None,
            "depth_max_m": round(float(finite.max()), 2) if finite.size else None,
        }


def _edge_heights(
    prism: Prism,
    a: np.ndarray,
    b: np.ndarray,
    ha: float,
    hb: float,
    steps: int,
    edge_index: int = -1,
    relief=None,  # noqa: ANN001
) -> np.ndarray:
    """Altitude du haut du mur, échantillonnée le long d'une arête.

    Le relief relevé dans le nuage prime : il décrit le mur lui-même, quand la
    surface de toiture n'en donne que le pourtour vu de dessus.
    """
    if relief is not None and edge_index >= 0:
        profile = relief.profiles.get(edge_index)
        if profile is not None:
            out = np.empty(steps + 1, dtype=np.float64)
            for k in range(steps + 1):
                found = relief.height_along(edge_index, k / steps)
                out[k] = found if found is not None else (
                    ha + (hb - ha) * k / steps
                )
            return out

    if steps <= 1 or not prism.roof_measured:
        return np.array([ha, hb], dtype=np.float64)

    # Toute l'arête est interrogée d'un coup : l'arbre rend les huit voisins
    # de chaque échantillon en une passe, là où la boucle relisait la nappe
    # entière à chaque pas.
    ts = np.linspace(0.0, 1.0, steps + 1)
    points = np.c_[a[0] + (b[0] - a[0]) * ts, a[1] + (b[1] - a[1]) * ts]
    return _roof_heights_at(prism, points)


def _roof_tree(prism: Prism):  # noqa: ANN001
    """Arbre des sommets de toiture, construit une fois par volume."""
    tree = getattr(prism, "_roof_tree_cache", None)
    if tree is None:
        tree = cKDTree(prism.roof_vertices[:, :2])
        prism._roof_tree_cache = tree
    return tree


def _roof_heights_at(prism: Prism, points: np.ndarray) -> np.ndarray:
    """Altitude du toit à l'aplomb d'un lot de points.

    Même règle que `_local_roof_height` — le groupe d'altitudes le mieux
    représenté l'emporte sur la valeur la plus haute — mais appliquée à tous
    les points en parallèle.
    """
    vertices = prism.roof_vertices
    count = min(8, len(vertices))
    _distances, indices = _roof_tree(prism).query(points, k=count)
    if indices.ndim == 1:
        indices = indices[:, None]

    heights = np.sort(vertices[indices, 2], axis=1)
    gaps = np.diff(heights, axis=1)
    if gaps.size == 0:
        return heights[:, 0]

    split = np.argmax(gaps, axis=1) + 1
    # Sous le décrochement, les deux groupes ; le plus fourni gagne. `split`
    # compte les points du groupe bas, la colonne restante ceux du haut.
    take_lower = split >= (heights.shape[1] - split)
    ranks = np.arange(heights.shape[1])[None, :]
    lower_mask = ranks < split[:, None]
    mask = np.where(take_lower[:, None], lower_mask, ~lower_mask)

    grouped = np.where(mask, heights, np.nan)
    split_heights = np.nanmedian(grouped, axis=1)
    # Sans décrochement franc, le voisinage décrit une seule toiture.
    return np.where(gaps.max(axis=1) > 2.0, split_heights, np.median(heights, axis=1))


def _local_roof_height(vertices: np.ndarray, x: float, y: float) -> float:
    """Altitude du toit à l'aplomb d'un point, décrochements respectés.

    Le maximum du voisinage refermait bien le volume, mais écrasait les
    décrochements réels : mesuré sur ce pilote, l'auvent d'entrée forme une
    zone compacte de deux cent trente mètres carrés à moins de quatre mètres,
    et ses murs montaient tout de même à dix — le corps principal l'avalait.

    Le voisinage est donc lu en deux temps. Les altitudes proches sont
    regroupées, et c'est le **groupe le mieux représenté** qui l'emporte, non
    la valeur la plus haute : un mur suit la toiture qui le surmonte
    réellement, même quand une toiture plus haute passe à trois mètres de là.
    """
    # `argpartition` plutôt qu'un tri complet : seuls les huit plus proches
    # comptent, et les trier tous coûtait un temps proportionnel à n log n sur
    # des nappes de plusieurs dizaines de milliers de sommets — depuis que
    # chaque volume porte une toiture triangulée, cette ligne dominait le rendu.
    distances = np.hypot(vertices[:, 0] - x, vertices[:, 1] - y)
    count = min(8, len(distances))
    nearest = np.argpartition(distances, count - 1)[:count]
    heights = np.sort(vertices[nearest, 2])

    # Séparation au plus grand saut : deux toitures distinctes sont séparées
    # d'un décrochement franc, deux points d'une même toiture ne le sont pas.
    gaps = np.diff(heights)
    if gaps.size and gaps.max() > 2.0:
        split = int(np.argmax(gaps)) + 1
        lower, upper = heights[:split], heights[split:]
        chosen = lower if lower.size >= upper.size else upper
        return float(np.median(chosen))
    return float(np.median(heights))


def _wall_heights(prism: Prism) -> np.ndarray:
    """Hauteur du mur à l'aplomb de chaque sommet de l'emprise.

    Sans toit mesuré, tous les murs montent à la hauteur unique du prisme.
    Avec un toit mesuré, chaque sommet prend l'altitude de la surface la plus
    proche : le volume épouse alors les décrochements réels — avancées, ailes
    plus basses, auvent d'entrée — au lieu d'une boîte à couvercle plat.
    """
    footprint = prism.footprint
    if not prism.roof_measured:
        return np.full(len(footprint), prism.height_m, dtype=np.float64)

    vertices = prism.roof_vertices
    # Le **maximum** du voisinage, non sa moyenne. Un sommet d'emprise se situe
    # au bord du toit, là où le nDSM mêle des cellules du toit et du sol
    # adjacent : moyenner les deux plaçait le haut du mur sous la surface, et
    # le volume rendu s'ouvrait — murs plongeant sous un toit qui flottait.
    # Prendre le plus haut du voisinage referme le volume.
    count = min(6, len(vertices))
    _distances, indices = _roof_tree(prism).query(footprint, k=count)
    if indices.ndim == 1:
        indices = indices[:, None]
    return vertices[indices, 2].max(axis=1)


def _prism_faces(prism: Prism) -> list[tuple[np.ndarray, bool]]:
    """Murs et toit d'un prisme, en triangles. Le drapeau marque le toit.

    Le toit est distingué parce qu'aucune source au sol ne l'atteste : c'est
    la surface que le masque de confiance déclasse le plus fort.

    Depuis le maillage canonique, ce module ne construit plus rien : il lit
    les triangles de `prism.canonical_mesh`, la même instance que le
    textureur, la collision et l'export consomment. L'extrusion locale n'est
    conservée que pour un prisme qui n'a pas encore reçu son maillage.
    """
    mesh = getattr(prism, "canonical_mesh", None)
    if mesh is not None:
        return [
            (mesh.vertices[face], mesh.face_kind[index] in ("roof", "roof_step"))
            for index, face in enumerate(mesh.faces)
        ]

    faces: list[tuple[np.ndarray, bool]] = []
    fp = prism.footprint
    n = len(fp)
    h = prism.height_m

    # Hauteur du mur au droit de chaque sommet. Sur un volume mesuré, elle suit
    # la surface du toit : un auvent d'entrée descend à quatre mètres quand le
    # corps du bâtiment en fait douze, et une hauteur unique l'effaçait — le
    # nDSM du pilote porte pourtant près d'un millier de cellules entre deux et
    # sept mètres, toutes écrasées au profil du corps principal.
    heights = _wall_heights(prism)

    # Le haut du mur est échantillonné le long de l'arête, non seulement à ses
    # extrémités. Une arête de vingt mètres joignant deux sommets bas passait
    # sous un toit plus haut en son milieu, et le volume s'ouvrait vu de biais.
    # Le relief de façade, quand il est relevé, commande la finesse : une
    # arête de quatre-vingt-huit mètres portant un pignon se découpe en
    # vingt-neuf segments, là où six lissaient le décrochement.
    relief = getattr(prism, "facade_relief", None)
    for i in range(n):
        a = fp[i]
        b = fp[(i + 1) % n]
        profile = relief.profiles.get(i) if relief is not None else None
        steps = (
            max(int(profile.heights.size), 2) if profile is not None
            else (6 if prism.roof_measured else 1)
        )
        edge = _edge_heights(
            prism, a, b, heights[i], heights[(i + 1) % n], steps, i, relief
        )
        for k in range(steps):
            t0, t1 = k / steps, (k + 1) / steps
            x0, y0 = a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0
            x1, y1 = a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1
            h0, h1 = edge[k], edge[k + 1]
            p0 = np.array([x0, y0, 0.0])
            p1 = np.array([x1, y1, 0.0])
            p2 = np.array([x1, y1, h1])
            p3 = np.array([x0, y0, h0])
            faces.append((np.stack([p0, p1, p2]), False))
            faces.append((np.stack([p0, p2, p3]), False))

    # Toit mesuré : la surface réelle du nDSM remplace la fermeture inventée.
    if prism.roof_measured:
        vertices = prism.roof_vertices
        for tri in prism.roof_faces:
            faces.append((vertices[tri], True))

        # La surface échantillonnée s'arrête à la dernière cellule entière —
        # quatre mètres du bord en moyenne sur ce pilote — et laissait une
        # ouverture entre elle et le haut des murs. Une jupe referme le volume
        # en reliant chaque arête de l'emprise au point de toit le plus proche.
        return faces

    # Sans mesure, le prisme se ferme par un cône vers son centre. Ce n'est pas
    # une observation : la carte de confiance déclasse ces faces en
    # conséquence, via `roof_confidence`.
    centre = fp.mean(axis=0)
    apex = np.array([centre[0], centre[1], h])
    for i in range(n):
        a = fp[i]
        b = fp[(i + 1) % n]
        tri = np.stack([
            np.array([a[0], a[1], h]),
            np.array([b[0], b[1], h]),
            apex,
        ])
        faces.append((tri, True))
    return faces


#: Largeur relative d'un volume végétal à huit niveaux, du pied au sommet.
#: Le relevé aérien donne la position et la hauteur ; la lecture des images au
#: sol dit si la masse est effilée, étalée ou colonnaire. Sans cette lecture,
#: un cylindre uniforme tient lieu de tout arbre.
VEGETATION_PROFILES: dict[str, tuple[float, ...]] = {
    "conique": (0.95, 1.00, 1.00, 0.90, 0.75, 0.55, 0.35, 0.15),
    "etale": (0.15, 0.20, 0.45, 0.85, 1.00, 1.00, 0.90, 0.55),
    "colonnaire": (0.75, 0.85, 0.95, 1.00, 1.00, 1.00, 0.90, 0.70),
    "arbustif": (1.00, 1.00, 1.00, 1.00, 0.85, 0.55, 0.25, 0.10),
    "cylindre": (1.0,) * 8,
}


def _vegetation_faces(patch) -> list[np.ndarray]:
    """Volume d'un massif, profilé selon la forme lue dans les images.

    Le rendu ne figure pas un feuillage, qu'aucune donnée n'atteste. Il pose un
    volume occupé — à la bonne place, à la bonne hauteur, et désormais avec la
    bonne allure — pour que profondeur et occultations soient justes. Un
    conifère effilé et un érable étalé occupent le même disque vu du ciel : la
    lecture au sol est le seul moyen de les distinguer.
    """
    cx, cy = patch.centre
    radius = patch.radius_m
    top = patch.height_m
    base = max(0.0, top * 0.15)

    shape = getattr(patch, "shape", None) or "cylindre"
    profile = VEGETATION_PROFILES.get(shape, VEGETATION_PROFILES["cylindre"])

    # Le maillage suit la taille du massif : un arbuste de deux mètres n'a pas
    # besoin de huit niveaux de profil pour borner son encombrement. Chaque
    # massif coûtait cent vingt triangles, tous niveaux confondus — plus de la
    # moitié du budget d'une image pour des volumes que le générateur ne fait
    # qu'habiller.
    levels = len(profile) if patch.height_m >= 6.0 else 4
    if levels < len(profile):
        step = (len(profile) - 1) / (levels - 1)
        profile = tuple(profile[round(i * step)] for i in range(levels))

    sides = 8 if patch.height_m >= 4.0 else 6
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    cosines, sines = np.cos(angles), np.sin(angles)
    levels = len(profile)

    def ring_at(level: int) -> list[tuple[float, float, float]]:
        z = base + (top - base) * level / (levels - 1)
        scale = max(profile[level], 0.05) * radius
        return [
            (cx + scale * cosines[i], cy + scale * sines[i], z) for i in range(sides)
        ]

    faces: list[np.ndarray] = []
    rings = [ring_at(level) for level in range(levels)]
    for level in range(levels - 1):
        lower, upper = rings[level], rings[level + 1]
        for i in range(sides):
            j = (i + 1) % sides
            p0 = np.array(lower[i])
            p1 = np.array(lower[j])
            p2 = np.array(upper[j])
            p3 = np.array(upper[i])
            faces.append(np.stack([p0, p1, p2]))
            faces.append(np.stack([p0, p2, p3]))

    apex = np.array([cx, cy, top])
    for i in range(sides):
        j = (i + 1) % sides
        faces.append(np.stack([np.array(rings[-1][i]), np.array(rings[-1][j]), apex]))
    return faces


def _patch_faces(patch, terrain=None) -> list[np.ndarray]:  # noqa: ANN001
    """Triangule une plage de sol depuis son contour.

    Le sol était rendu en carreaux de la taille de la maille, et ses bordures
    montraient des marches d'escalier. Un contour simplifié donne des bords
    nets avec bien moins de triangles — soixante et une plages de neuf sommets
    médians remplacent ici vingt-sept mille carreaux.
    """
    ring = patch.ring
    if len(ring) < 4:
        return []

    # Le contour est fermé : le dernier sommet répète le premier.
    points = ring[:-1] if ring[0] == ring[-1] else list(ring)
    if len(points) < 3:
        return []

    centre = (
        float(np.mean([p[0] for p in points])),
        float(np.mean([p[1] for p in points])),
    )
    # Le sol épouse le terrain quand il est connu : posé à plat, il faisait
    # flotter les volumes sur un plan idéal que le relevé dément.
    def _z(x: float, y: float) -> float:
        return 0.0 if terrain is None else terrain.height_at(x, y)

    apex = np.array([centre[0], centre[1], _z(*centre)])

    # Éventail depuis le centroïde : une plage de sol reste assez convexe pour
    # que ce découpage suffise, et il évite d'embarquer un triangulateur.
    faces: list[np.ndarray] = []
    for index in range(len(points)):
        a = points[index]
        b = points[(index + 1) % len(points)]
        faces.append(
            np.stack([
                np.array([a[0], a[1], _z(a[0], a[1])]),
                np.array([b[0], b[1], _z(b[0], b[1])]),
                apex,
            ])
        )
    return faces


def _furniture_faces(item) -> list[np.ndarray]:  # noqa: ANN001
    """Colonne verticale d'un mât : un prisme fin, du sol à son sommet."""
    cx, cy = item.centre
    radius = max(item.radius_m, 0.25)
    top = item.height_m

    sides = 6
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    ring = [(cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles]

    faces: list[np.ndarray] = []
    for i in range(sides):
        ax, ay = ring[i]
        bx, by = ring[(i + 1) % sides]
        p0 = np.array([ax, ay, 0.0])
        p1 = np.array([bx, by, 0.0])
        p2 = np.array([bx, by, top])
        p3 = np.array([ax, ay, top])
        faces.append(np.stack([p0, p1, p2]))
        faces.append(np.stack([p0, p2, p3]))
    return faces


def _project(points: np.ndarray, camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    """Passe du monde à l'écran. Retourne les pixels et la profondeur.

    Les trois axes du repère caméra sont regroupés en une seule matrice,
    mémorisée avec la pose : un produit matriciel remplace trois produits
    scalaires et l'assemblage qui suivait. Appelée une fois par triangle,
    cette fonction faisait à elle seule vingt-huit mille `stack` par image.
    """
    axes = camera.axes()
    local = (points - camera.position) @ axes
    x, y, z = local[:, 0], local[:, 1], local[:, 2]

    f = 1.0 / math.tan(math.radians(camera.fov_deg) * 0.5)
    aspect = camera.width / camera.height
    safe_z = np.where(np.abs(z) < 1e-6, 1e-6, z)

    screen = np.empty((len(points), 2), dtype=np.float64)
    screen[:, 0] = ((f / aspect) * x / safe_z + 1.0) * 0.5 * camera.width
    screen[:, 1] = (1.0 - f * y / safe_z) * 0.5 * camera.height
    return screen, z


def _rasterise(
    tri: np.ndarray,
    camera: Camera,
    depth: np.ndarray,
    normal: np.ndarray,
    silhouette: np.ndarray,
    confidence: np.ndarray,
    silhouette_value: int,
    confidence_value: float,
    projected: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    """Projette un triangle et l'inscrit s'il est plus proche que le z-buffer.

    Extrait de la boucle des volumes bâtis pour que la végétation emprunte
    exactement le même chemin : deux rasteriseurs divergents finiraient par
    produire des occultations incohérentes entre un massif et un mur.
    """
    h, w = depth.shape
    if projected is not None:
        # Le pré-filtre a déjà projeté ce triangle et validé sa profondeur
        # comme son aire : refaire les deux ne changerait aucun pixel.
        screen, zcam = projected
        a, b, c = screen
        area = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
    else:
        screen, zcam = _project(tri, camera)
        if np.any(zcam <= 0.05):
            return
        a, b, c = screen
        area = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
        if abs(area) < 1e-9:
            return

    min_x = max(int(np.floor(screen[:, 0].min())), 0)
    max_x = min(int(np.ceil(screen[:, 0].max())), w - 1)
    min_y = max(int(np.floor(screen[:, 1].min())), 0)
    max_y = min(int(np.ceil(screen[:, 1].max())), h - 1)
    if min_x > max_x or min_y > max_y:
        return

    # Les coordonnées barycentriques sont séparables : chaque poids est une
    # fonction affine de x et de y, et se compose par diffusion de deux
    # vecteurs. Construire une grille de points par triangle — quinze mille
    # `mgrid` et autant de `stack` par image — coûtait le tiers du rendu pour
    # un résultat identique.
    px = np.arange(min_x, max_x + 1, dtype=np.float64) + 0.5
    py = np.arange(min_y, max_y + 1, dtype=np.float64) + 0.5
    dx = px - a[0]
    dy = (py - a[1])[:, None]

    inv_area = 1.0 / area
    w0 = ((b[0] - a[0]) * dy - dx * (b[1] - a[1])) * inv_area
    w1 = (dx * (c[1] - a[1]) - (c[0] - a[0]) * dy) * inv_area
    w2 = 1.0 - w0 - w1
    inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
    if not inside.any():
        return

    # Interpolation perspective-correcte de la profondeur.
    inv_z = w2 / zcam[0] + w1 / zcam[1] + w0 / zcam[2]
    inv_z = np.where(np.abs(inv_z) < 1e-12, 1e-12, inv_z)
    tri_depth = 1.0 / inv_z

    # `np.cross` sur des vecteurs de trois composantes passe par un mécanisme
    # générique coûteux — il pesait un tiers du temps de rendu, appelé une fois
    # par triangle. Le produit écrit à la main donne le même résultat.
    e1 = tri[1] - tri[0]
    e2 = tri[2] - tri[0]
    nx = e1[1] * e2[2] - e1[2] * e2[1]
    ny = e1[2] * e2[0] - e1[0] * e2[2]
    nz = e1[0] * e2[1] - e1[1] * e2[0]
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length < 1e-12:
        return
    nx, ny, nz = nx / length, ny / length, nz / length
    to_camera = camera.position - tri[0]
    if nx * to_camera[0] + ny * to_camera[1] + nz * to_camera[2] < 0:
        nx, ny, nz = -nx, -ny, -nz
    face_normal = np.array([nx, ny, nz])

    # Le découpage numpy rend une vue, non une copie : écrire dedans écrit
    # dans la carte. Les quatre recopies qui suivaient étaient sans effet.
    rows, cols = slice(min_y, max_y + 1), slice(min_x, max_x + 1)
    region_depth = depth[rows, cols]
    closer = inside & (tri_depth < region_depth) & (tri_depth > 0)
    if not closer.any():
        return

    region_depth[closer] = tri_depth[closer]
    silhouette[rows, cols][closer] = silhouette_value
    normal[rows, cols][closer] = face_normal
    confidence[rows, cols][closer] = confidence_value


def _keep_mask(faces: list, camera: Camera):
    """Quels triangles peuvent toucher l'image, décidé en une seule passe.

    Le rastériseur rejette déjà ce qui sort du cadre — mais il le fait un
    triangle à la fois, après un appel de fonction et une projection. Les
    projeter tous ensemble et n'entrer dans la boucle que pour les survivants
    remplace des dizaines de milliers d'appels Python par trois opérations
    numpy.

    La projection est rendue avec le verdict : `_rasterise` la reprendrait
    sinon à l'identique, et la calculer deux fois annulait le gain.
    """
    if not faces:
        return np.zeros(0, dtype=bool), None, None

    corners = np.asarray([tri for tri, *_rest in faces], dtype=np.float64)
    flat = corners.reshape(-1, 3)
    screen, zcam = _project(flat, camera)
    screen = screen.reshape(len(faces), 3, 2)
    zcam = zcam.reshape(len(faces), 3)

    # Un sommet derrière la caméra suffit à écarter : c'est exactement la
    # condition que `_rasterise` applique ensuite.
    keep = (zcam > 0.05).all(axis=1)

    # Boîte du triangle entièrement hors cadre : rien à inscrire.
    keep &= screen[:, :, 0].max(axis=1) >= 0.0
    keep &= screen[:, :, 0].min(axis=1) <= camera.width - 1
    keep &= screen[:, :, 1].max(axis=1) >= 0.0
    keep &= screen[:, :, 1].min(axis=1) <= camera.height - 1

    # Triangle dégénéré à l'écran : aire nulle, aucun pixel couvert.
    a, b, c = screen[:, 0], screen[:, 1], screen[:, 2]
    area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (
        b[:, 1] - a[:, 1]
    )
    keep &= np.abs(area) >= 1e-9
    return keep, screen, zcam


def _prism_bounds(prism: Prism) -> tuple[np.ndarray, float]:
    """Sphère englobant un volume : centre au sol, et rayon.

    Mémorisée sur le prisme : elle ne dépend que de la géométrie, que la pose
    de caméra ne change pas.
    """
    cached = getattr(prism, "_bounds_cache", None)
    if cached is not None:
        return cached

    footprint = prism.footprint
    centre = footprint.mean(axis=0)
    spread = float(
        np.hypot(footprint[:, 0] - centre[0], footprint[:, 1] - centre[1]).max()
    )
    top = prism.height_m
    if prism.roof_measured:
        top = max(top, float(prism.roof_vertices[:, 2].max()))

    middle = np.array([centre[0], centre[1], top * 0.5])
    found = (middle, math.hypot(spread, top * 0.5))
    prism._bounds_cache = found
    return found


def _patch_bounds(patch) -> tuple[np.ndarray, float]:  # noqa: ANN001
    """Sphère englobant un massif, un mobilier ou une parcelle de sol."""
    cached = getattr(patch, "_bounds_cache", None)
    if cached is not None:
        return cached

    centre = np.asarray(getattr(patch, "centre", (0.0, 0.0)), dtype=np.float64)
    radius = float(getattr(patch, "radius_m", 0.0) or 0.0)
    height = float(getattr(patch, "height_m", 0.0) or 0.0)
    outline = getattr(patch, "outline", None)
    if outline is not None and len(outline):
        outline = np.asarray(outline, dtype=np.float64)
        centre = outline[:, :2].mean(axis=0)
        radius = max(
            radius,
            float(np.hypot(outline[:, 0] - centre[0], outline[:, 1] - centre[1]).max()),
        )

    middle = np.array([centre[0], centre[1], height * 0.5])
    found = (middle, math.hypot(max(radius, 1.0), height * 0.5))
    try:
        patch._bounds_cache = found
    except AttributeError:
        # Un objet figé refuse le cache ; il paiera le calcul à chaque image.
        pass
    return found


def _visible(bounds: tuple[np.ndarray, float], camera: Camera) -> bool:
    """L'objet peut-il toucher l'image, même partiellement ?

    Le test est délibérément permissif : il écarte ce qui est franchement
    derrière ou franchement à côté, et laisse passer tout le reste. Un doute
    coûte quelques triangles rastérisés pour rien ; un rejet à tort ferait
    disparaître un bâtiment de la vidéo.
    """
    centre, radius = bounds
    forward, right, up = camera.basis()
    offset = centre - camera.position

    depth = float(offset @ forward)
    # Derrière la caméra, et pas seulement de justesse.
    if depth + radius <= 0.05:
        return False

    # Demi-champ élargi du rayon de l'objet : un massif dont le centre sort du
    # cadre peut très bien y déborder.
    half_v = math.radians(camera.fov_deg) * 0.5
    half_h = math.atan(math.tan(half_v) * camera.width / camera.height)
    slack = radius / max(depth, 1e-6)

    if abs(float(offset @ right)) / max(depth, 1e-6) > math.tan(half_h) + slack:
        return False
    if abs(float(offset @ up)) / max(depth, 1e-6) > math.tan(half_v) + slack:
        return False
    return True


def render_frame(
    scene: ConditioningScene,
    camera: Camera,
    environment=None,  # noqa: ANN001
) -> RenderedFrame:
    """Rasterise la scène pour une pose, en z-buffer.

    L'environnement est optionnel : sans lui, le bâtiment flotte dans le vide et
    le générateur invente la végétation qui l'entoure, y compris devant une
    façade que rien ne masque en réalité.
    """
    w, h = camera.width, camera.height
    depth = np.full((h, w), np.inf, dtype=np.float64)
    normal = np.zeros((h, w, 3), dtype=np.float64)
    silhouette = np.zeros((h, w), dtype=np.uint8)
    confidence = np.zeros((h, w), dtype=np.float64)

    for prism in scene.prisms:
        if not _visible(_prism_bounds(prism), camera):
            continue
        base_conf = prism.confidence
        faces = _prism_faces(prism)
        keep, screens, zcams = _keep_mask(faces, camera)
        for slot, ((tri, is_roof), wanted) in enumerate(zip(faces, keep)):
            if not wanted:
                continue
            _rasterise(
                tri,
                camera,
                depth,
                normal,
                silhouette,
                confidence,
                silhouette_value=2 if prism.is_target else 1,
                # Le toit porte son propre crédit : effondré quand il n'est
                # qu'une fermeture géométrique, égal aux murs quand un nDSM
                # aérien l'atteste directement.
                confidence_value=prism.roof_confidence if is_roof else base_conf,
                projected=(screens[slot], zcams[slot]),
            )

    # La végétation est rendue après les volumes bâtis : elle les occulte
    # parfois, et le z-buffer tranche. Son crédit est délibérément bas — un
    # massif borne un encombrement, il ne décrit ni espèce ni feuillage.
    if environment is not None:
        for patch in environment.patches:
            if not _visible(_patch_bounds(patch), camera):
                continue
            for tri in _vegetation_faces(patch):
                _rasterise(
                    tri, camera, depth, normal, silhouette, confidence,
                    silhouette_value=3, confidence_value=VEGETATION_CONFIDENCE,
                )
        # Le mobilier porte sa propre silhouette : un mât devant une façade
        # occulte réellement, mais le générateur ne doit pas y peindre un
        # arbre. Les deux natures sont donc distinguées jusqu'au rendu.
        # Le sol est posé avant tout le reste : il n'occulte rien, mais sans lui
        # le bâtiment flotte et le générateur choisit seul entre pelouse et
        # bitume — souvent l'un à la place de l'autre.
        for patch in getattr(environment, "ground_patches", ()) or ():
            for tri in _patch_faces(patch, getattr(environment, "terrain", None)):
                _rasterise(
                    tri, camera, depth, normal, silhouette, confidence,
                    silhouette_value=(
                        5 if patch.kind == "vegetal"
                        else 7 if patch.kind == "indetermine_pose"
                        else 6
                    ),
                    confidence_value=GROUND_CONFIDENCE,
                )

        for item in getattr(environment, "furniture", ()):
            if not _visible(_patch_bounds(item), camera):
                continue
            for tri in _furniture_faces(item):
                _rasterise(
                    tri, camera, depth, normal, silhouette, confidence,
                    silhouette_value=4, confidence_value=FURNITURE_CONFIDENCE,
                )

    hit = np.isfinite(depth)
    total = float(w * h)
    target_px = int((silhouette == 2).sum())
    assumed_px = int(((confidence > 0) & (confidence < 0.5)).sum())

    return RenderedFrame(
        depth=depth,
        normal=normal,
        silhouette=silhouette,
        confidence=confidence,
        target_coverage=target_px / total,
        assumed_fraction=assumed_px / total,
        hit_any=bool(hit.any()),
    )
