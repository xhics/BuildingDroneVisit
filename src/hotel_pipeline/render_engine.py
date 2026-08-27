"""Le moteur de rendu : un seul chemin du maillage canonique aux buffers.

Le contrat est celui du plan de correction :

    CanonicalCamera + CanonicalSceneMesh
        │
        ▼
    Rasterizer / BVH
        ├── depth_z          (Z espace caméra, jamais distance euclidienne)
        ├── triangle_id      (le pixel voit CE triangle)
        ├── surface_id       (donc CETTE façade — deux surfaces à profondeur
        │                     quasi égale ne se confondent plus)
        ├── normal
        └── semantic_id

Trois règles structurelles :

1. **aucun clamp au bord** : un polygone qui déborde de l'image est découpé
   polygonalement (Sutherland–Hodgman) avant rastérisation — une façade à
   moitié hors champ ne laisse aucune bande collée au bord ;
2. **clip near-plane en espace caméra**, avant projection : un triangle qui
   traverse le plan proche devient zéro, un ou deux triangles, et ne produit
   jamais de coordonnée gigantesque ni de NaN ;
3. **interpolation perspective** par 1/z partout : la profondeur rastérisée
   d'un triangle très oblique rejoint l'intersection analytique du rayon.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------
# Découpage polygonal
# ----------------------------------------------------------------------
def clip_polygon_half_plane(
    polygon: np.ndarray,
    axis: int,
    threshold: float,
    keep_positive: bool,
) -> np.ndarray:
    """Découpe un polygone (N,2 ou N,3) par un demi-plan, sommets interpolés."""
    if len(polygon) == 0:
        return polygon

    def _inside(point) -> bool:
        value = point[axis] - threshold
        return value >= 0.0 if keep_positive else value <= 0.0

    output: list[np.ndarray] = []
    count = len(polygon)
    for index in range(count):
        current = polygon[index]
        previous = polygon[index - 1]
        current_inside = _inside(current)
        previous_inside = _inside(previous)
        if current_inside:
            if not previous_inside:
                t = (threshold - previous[axis]) / (current[axis] - previous[axis])
                output.append(previous + t * (current - previous))
            output.append(current)
        elif previous_inside:
            t = (threshold - previous[axis]) / (current[axis] - previous[axis])
            output.append(previous + t * (current - previous))
    return np.asarray(output, dtype=np.float64).reshape((-1, polygon.shape[1]))


def clip_triangle_near(
    tri_camera: np.ndarray, z_near: float
) -> list[np.ndarray]:
    """Clippe un triangle espace caméra contre z = z_near.

    Retourne 0 triangle (entièrement derrière), 1 (devant, ou un sommet
    devant et deux derrière), ou 2 (deux sommets devant, un derrière).
    Aucune projection n'a lieu avant ce découpage : c'est ce qui empêche
    l'explosion des coordonnées.
    """
    tri = np.asarray(tri_camera, dtype=np.float64).reshape((3, 3))
    in_front = tri[:, 2] >= z_near
    if in_front.all():
        return [tri]
    if not in_front.any():
        return []

    def _clip_point(p_front, p_back):
        t = (z_near - p_front[2]) / (p_back[2] - p_front[2])
        return p_front + t * (p_back - p_front)

    if int(in_front.sum()) == 1:
        # Un sommet devant : le triangle coupé reste UN triangle.
        a = tri[int(np.argmax(in_front))]
        others = [i for i in range(3) if i != int(np.argmax(in_front))]
        ab = _clip_point(a, tri[others[0]])
        ac = _clip_point(a, tri[others[1]])
        return [np.stack([a, ab, ac])]

    # Deux sommets devant : le quadrilatère coupé donne DEUX triangles.
    back_index = int(np.argmin(in_front))
    front_indices = [i for i in range(3) if i != back_index]
    a, b = tri[front_indices[0]], tri[front_indices[1]]
    c = tri[back_index]
    bc = _clip_point(b, c)
    ac = _clip_point(a, c)
    first = np.stack([a, b, bc])
    second = np.stack([a, bc, ac])
    area1 = abs(
        (first[1][0] - first[0][0]) * (first[2][1] - first[0][1])
        - (first[1][1] - first[0][1]) * (first[2][0] - first[0][0])
    )
    pieces = []
    if area1 > 1e-12:
        pieces.append(first)
        area2 = abs(
            (second[1][0] - second[0][0]) * (second[2][1] - second[0][1])
            - (second[1][1] - second[0][1]) * (second[2][0] - second[0][0])
        )
        if area2 > 1e-12:
            pieces.append(second)
    return pieces or [np.stack([a, b, ac])]


def clip_polygon_to_image(
    polygon_screen: np.ndarray, width: int, height: int
) -> np.ndarray:
    """Découpe un polygone écran contre le rectangle image.

    Jamais ``x = clamp(x, w)`` : les sommets hors champ sont remplacés par
    des intersections exactes avec les bords, si bien qu'aucune bande fausse
    ne peut être collée au bord.
    """
    poly = np.asarray(polygon_screen, dtype=np.float64)
    for axis, threshold, keep in (
        (0, 0.0, True),
        (0, float(width), False),
        (1, 0.0, True),
        (1, float(height), False),
    ):
        if len(poly) == 0:
            break
        poly = clip_polygon_half_plane(poly, axis, threshold, keep)
    return poly.reshape((-1, 2))


# ----------------------------------------------------------------------
# Buffers de frame
# ----------------------------------------------------------------------
@dataclass
class FrameBuffers:
    """Ce qu'une frame établit, canal par canal."""

    depth_z: np.ndarray                 # Z espace caméra ; inf = rien
    triangle_id: np.ndarray             # index global de face ; -1 = vide
    surface_id: np.ndarray              # identité logique par face ; -1
    normal: np.ndarray                  # normale monde par pixel
    semantic_id: np.ndarray             # classe sémantique ; -1
    width: int
    height: int
    camera: object = None               # CanonicalCamera
    input_mesh_digest: str | None = None
    surface_id_lookup: dict[int, str] = field(default_factory=dict)

    def depth_at(self, x: int, y: int) -> float:
        if 0 <= y < self.height and 0 <= x < self.width:
            return float(self.depth_z[y, x])
        return float("inf")

    def hit(self, x: int, y: int) -> tuple[float, int | None]:
        """Compatibility view used by texture visibility: metric Z + face id."""
        if 0 <= y < self.height and 0 <= x < self.width:
            return float(self.depth_z[y, x]), int(self.triangle_id[y, x])
        return float("inf"), None

    def physical_surface_at(self, x: int, y: int) -> str | None:
        if not (0 <= y < self.height and 0 <= x < self.width):
            return None
        return self.surface_id_lookup.get(int(self.surface_id[y, x]))

    def visible(
        self,
        point_world: np.ndarray,
        surface_id: int,
        tolerance_m: float = 0.35,
    ) -> bool:
        """La question unique du textureur : ce pixel voit-il cette surface ?

        Le verdict croise la profondeur **et** l'identité de surface : deux
        murs à quelques centimètres d'écart ne se substituent plus l'un à
        l'autre.
        """
        camera = self.camera
        screen, z = camera.project(np.asarray(point_world, dtype=float).reshape((1, 3)))
        x, y = int(round(float(screen[0, 0]))), int(round(float(screen[0, 1])))
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        if float(z[0]) <= camera.near_m:
            return False
        hit_z = float(self.depth_z[y, x])
        if not np.isfinite(hit_z):
            return False
        hit_surface = int(self.surface_id[y, x])
        return hit_surface == int(surface_id) and abs(hit_z - float(z[0])) <= tolerance_m


# ----------------------------------------------------------------------
# Rasterisation du maillage canonique
# ----------------------------------------------------------------------
def rasterize_mesh(
    mesh,  # noqa: ANN001 - CanonicalSceneMesh
    camera,  # noqa: ANN001 - CanonicalCamera
    width: int | None = None,
    height: int | None = None,
    near_m: float | None = None,
    surface_ids: np.ndarray | list[int] | None = None,
    semantic_ids: np.ndarray | list[int] | None = None,
) -> FrameBuffers:
    """Rastérise LE maillage canonique dans les buffers complets.

    Une seule source de triangles : le CanonicalSceneMesh. Aucun mur ou toit
    reconstruit séparément ne peut doubler une surface dans le z-buffer.
    """
    from .reality_contract import require_canonical_mesh

    receipt = require_canonical_mesh(mesh, "renderer")
    width = int(width or camera.width)
    height = int(height or camera.height)
    near = float(near_m or camera.near_m)

    depth = np.full((height, width), np.inf, dtype=np.float64)
    triangle_id = np.full((height, width), -1, dtype=np.int32)
    surface_buf = np.full((height, width), -1, dtype=np.int32)
    normal_buf = np.zeros((height, width, 3), dtype=np.float64)
    semantic_buf = np.full((height, width), -1, dtype=np.int32)

    faces = mesh.faces
    vertices = mesh.vertices
    physical_surface_lookup: dict[int, str] = {}
    if surface_ids is None:
        semantic_surfaces = sorted(set(mesh.surface_ids))
        encoded = {surface_id: index for index, surface_id in enumerate(semantic_surfaces)}
        surface_ids = np.asarray([encoded[value] for value in mesh.surface_ids], dtype=np.int32)
        physical_surface_lookup = {value: key for key, value in encoded.items()}
    if semantic_ids is None:
        semantic_ids = np.full(len(faces), -1, dtype=np.int32)

    for face_index, face in enumerate(faces):
        tri_world = vertices[face]
        tri_cam = tri_world @ camera.R.T + camera.t
        clipped = clip_triangle_near(tri_cam, near)
        for piece in clipped:
            # Projection directe du morceau déjà en espace caméra : aucun
            # sommet derrière la caméra ne peut atteindre cette étape.
            screen, zcam = _project_camera_space(piece, camera)
            if len(screen) < 3:
                continue
            polygon = clip_polygon_to_image(screen, width, height)
            if len(polygon) < 3:
                continue
            _rasterise_polygon(
                polygon,
                piece,
                screen,
                zcam,
                camera,
                depth,
                triangle_id,
                surface_buf,
                normal_buf,
                semantic_buf,
                int(face_index),
                int(surface_ids[face_index]),
                int(semantic_ids[face_index]),
                tri_world,
            )

    return FrameBuffers(
        depth_z=depth,
        triangle_id=triangle_id,
        surface_id=surface_buf,
        normal=normal_buf,
        semantic_id=semantic_buf,
        width=width,
        height=height,
        camera=camera,
        input_mesh_digest=receipt.input_mesh_digest,
        surface_id_lookup=physical_surface_lookup,
    )


def _project_camera_space(points_cam: np.ndarray, camera):  # noqa: ANN001
    """Projection directe de points déjà en espace caméra."""
    points_cam = np.asarray(points_cam, dtype=np.float64)
    depth = points_cam[:, 2]
    safe_depth = np.where(np.abs(depth) < 1e-9, 1e-9, depth)
    xn = points_cam[:, 0] / safe_depth
    yn = points_cam[:, 1] / safe_depth
    ud, vd = camera._distort(xn, yn)
    if camera.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        f = float(camera.params[0])
        fx = fy = f
    else:
        fx, fy = float(camera.params[0]), float(camera.params[1])
    cx, cy = camera.principal
    return np.column_stack([fx * ud + cx, fy * vd + cy]), depth


def _polygon_from_points(points: np.ndarray):
    """Construit un polygone shapely fermé depuis une liste de points."""
    from shapely.geometry import Polygon

    return Polygon(np.asarray(points, dtype=np.float64))


def _rasterise_polygon(
    polygon_screen,      # noqa: ANN001 - polygone clippé (M,2)
    tri_cam,             # noqa: ANN001 - morceau espace caméra
    screen_full,         # noqa: ANN001 - projection complète (3+,2)
    zcam_full,           # noqa: ANN001 - profondeurs complètes (3+)
    camera,              # noqa: ANN001
    depth, triangle_id, surface_buf, normal_buf, semantic_buf,  # noqa: ANN001
    face_index: int,
    surface_id: int,
    semantic_id: int,
    tri_world,           # noqa: ANN001
) -> None:
    """Rastérise un polygone clippé : Z perspective-exact par rayon."""
    height, width = depth.shape
    min_x = max(int(np.floor(polygon_screen[:, 0].min())), 0)
    max_x = min(int(np.ceil(polygon_screen[:, 0].max())), width - 1)
    min_y = max(int(np.floor(polygon_screen[:, 1].min())), 0)
    max_y = min(int(np.ceil(polygon_screen[:, 1].max())), height - 1)
    if min_x > max_x or min_y > max_y:
        return

    import shapely

    path_polygon = _polygon_from_points(polygon_screen)
    xs = np.arange(min_x, max_x + 1, dtype=np.float64) + 0.5
    ys = np.arange(min_y, max_y + 1, dtype=np.float64) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)
    candidates = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    # `contains_xy` exclut la frontière. Sur l'arête commune de deux
    # triangles, cela créait une fissure d'un pixel et rendait le résultat
    # dépendant de la triangulation. `intersects_xy` applique la fermeture du
    # polygone : une arête partagée est couverte, puis le z-buffer départage.
    inside = shapely.intersects_xy(
        path_polygon, grid_x.ravel(), grid_y.ravel()
    ).reshape(grid_x.shape)
    if not inside.any():
        return

    # Profondeur exacte par rayon, calculée **intégralement en espace
    # caméra** : le pixel définit la droite (xn, yn, 1), son intersection
    # avec le plan du triangle clippé donne le Z perspective-exact — c'est
    # la forme analytique de l'interpolation par 1/z, y compris sur un
    # triangle très oblique.
    e1 = tri_cam[1] - tri_cam[0]
    e2 = tri_cam[2] - tri_cam[0]
    normal_cam = np.cross(e1, e2)
    norm_len = float(np.linalg.norm(normal_cam))
    if norm_len < 1e-12:
        return
    normal_cam /= norm_len

    rays_world = np.asarray([
        camera.ray_from_pixel(u, v) for u, v in candidates
    ])
    rays_cam = rays_world @ camera.R.T
    rays_cam /= np.maximum(rays_cam[:, 2:3], 1e-12)
    denom = rays_cam @ normal_cam
    safe_denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    tt = float(normal_cam @ tri_cam[0]) / safe_denom
    z_pixels = tt.reshape(grid_x.shape)

    region_depth = depth[min_y:max_y + 1, min_x:max_x + 1]
    closer = inside & (z_pixels > camera.near_m) & (z_pixels < region_depth)
    if not closer.any():
        return

    rows, cols = np.where(closer)
    region_depth[rows, cols] = z_pixels[closer]
    # Normale monde pour le buffer : déduite du triangle monde d'origine,
    # orientée vers la caméra.
    world_e1 = tri_world[1] - tri_world[0]
    world_e2 = tri_world[2] - tri_world[0]
    world_normal = np.cross(world_e1, world_e2)
    world_len = float(np.linalg.norm(world_normal))
    if world_len > 1e-12:
        world_normal /= world_len
        to_camera = camera.position() - tri_world[0]
        if float(world_normal @ to_camera) < 0.0:
            world_normal = -world_normal
    else:
        world_normal = np.zeros(3)
    triangle_id[min_y:max_y + 1, min_x:max_x + 1][closer] = face_index
    surface_buf[min_y:max_y + 1, min_x:max_x + 1][closer] = surface_id
    normal_buf[min_y:max_y + 1, min_x:max_x + 1][closer] = world_normal
    semantic_buf[min_y:max_y + 1, min_x:max_x + 1][closer] = semantic_id


def _undistort_pixels(pixels: np.ndarray, camera) -> np.ndarray:  # noqa: ANN001
    """Inverse approché de la distorsion par itération de Newton (3 pas).

    Retourne les coordonnées **normalisées** (xn, yn) du rayon : suffisant
    pour poser des rayons, la profondeur écrite reste issue du vrai
    triangle, la distorsion ne sert qu'à choisir le pixel.
    """
    return np.asarray(
        [camera.undistort_pixel(u, v) for u, v in pixels], dtype=float
    )


# ----------------------------------------------------------------------
# BVH ray-triangle sur le maillage canonique
# ----------------------------------------------------------------------
class BVH:
    """Accélérateur rayon/triangle : toute visibilité devient un raycast.

    Nœuds médians sur l'axe le plus étendu ; feuilles Möller–Trumbore. Un
    rayon sous un auvent atteint le mur derrière ; un rayon à travers le toit
    est bloqué — la 2,5D est terminée.
    """

    MAX_LEAF = 8

    def __init__(self, mesh, surface_ids=None) -> None:  # noqa: ANN001
        mesh_vertices = np.asarray(mesh.vertices, dtype=np.float64)
        self.vertices = mesh_vertices
        self.faces = np.asarray(mesh.faces, dtype=np.int64)
        if surface_ids is None:
            surface_ids = np.arange(len(self.faces), dtype=np.int64)
        self.surface_ids = np.asarray(surface_ids, dtype=np.int64)
        self.triangles = mesh_vertices[self.faces]
        centroids = self.triangles.mean(axis=1)
        indices = list(range(len(self.faces)))
        self._nodes: list[dict] = []
        self._build(indices, centroids)

    def _bounds_of(self, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        pts = self.triangles[indices].reshape((-1, 3))
        return pts.min(axis=0), pts.max(axis=0)

    def _build(self, indices: list[int], centroids: np.ndarray) -> int:
        lo, hi = self._bounds_of(indices)
        node_index = len(self._nodes)
        self._nodes.append({"lo": lo, "hi": hi, "left": -1, "right": -1, "items": []})
        if len(indices) <= self.MAX_LEAF:
            self._nodes[node_index]["items"] = indices
            return node_index
        extents = hi - lo
        axis = int(np.argmax(extents))
        order = sorted(indices, key=lambda i: centroids[i, axis])
        middle = len(order) // 2
        left = self._build(order[:middle], centroids)
        right = self._build(order[middle:], centroids)
        self._nodes[node_index]["left"] = left
        self._nodes[node_index]["right"] = right
        return node_index

    @staticmethod
    def _ray_aabb(origin, inv_dir, lo, hi) -> bool:  # noqa: ANN001
        t0 = (lo - origin) * inv_dir
        t1 = (hi - origin) * inv_dir
        tmin = np.minimum(t0, t1)
        tmax = np.maximum(t0, t1)
        return bool(tmin.max() <= max(tmax.max(), 0.0))

    def raycast(
        self, origin: np.ndarray, direction: np.ndarray, max_distance_m: float = 5e3
    ) -> tuple[float, int] | None:
        """Premier triangle touché : (distance, index de face), sinon None."""
        origin = np.asarray(origin, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            return None
        direction = direction / norm
        inv_dir = 1.0 / np.where(np.abs(direction) < 1e-12, 1e-12, direction)

        best_t = max_distance_m
        best_face = -1
        stack = [0]
        while stack:
            node = self._nodes[stack.pop()]
            if not self._ray_aabb(origin, inv_dir, node["lo"], node["hi"]):
                continue
            if node["items"]:
                for face_index in node["items"]:
                    hit = _moller_trumbore(
                        origin, direction, self.triangles[face_index], best_t
                    )
                    if hit is not None:
                        best_t, _ = hit
                        best_face = face_index
                continue
            if node["left"] >= 0:
                stack.append(node["left"])
            if node["right"] >= 0:
                stack.append(node["right"])
        if best_face < 0:
            return None
        return best_t, best_face

    def occludes(self, start: np.ndarray, end: np.ndarray, margin_m: float = 0.01) -> bool:
        """Un obstacle sur le segment start→end ?"""
        delta = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
        length = float(np.linalg.norm(delta))
        if length < margin_m:
            return False
        found = self.raycast(start, delta, max_distance_m=length - margin_m)
        return found is not None

    def surface_at(self, origin: np.ndarray, direction: np.ndarray) -> int | None:
        found = self.raycast(origin, direction)
        if found is None:
            return None
        return int(self.surface_ids[found[1]])


def _moller_trumbore(
    origin: np.ndarray, direction: np.ndarray, tri: np.ndarray, max_distance: float
) -> tuple[float, np.ndarray] | None:
    edge1 = tri[1] - tri[0]
    edge2 = tri[2] - tri[0]
    pvec = np.cross(direction, edge2)
    det = float(edge1 @ pvec)
    if abs(det) < 1e-12:
        return None
    inv_det = 1.0 / det
    tvec = origin - tri[0]
    u = float(tvec @ pvec) * inv_det
    if u < -1e-9 or u > 1.0 + 1e-9:
        return None
    qvec = np.cross(tvec, edge1)
    v = float(direction @ qvec) * inv_det
    if v < -1e-9 or u + v > 1.0 + 1e-9:
        return None
    t = float(edge2 @ qvec) * inv_det
    if t < 1e-6 or t > max_distance:
        return None
    return t, np.array([u, v])


def build_bvh(mesh, surface_ids=None) -> BVH:  # noqa: ANN001
    """Construit le BVH du maillage canonique (helper lisible)."""
    return BVH(mesh, surface_ids=surface_ids)
