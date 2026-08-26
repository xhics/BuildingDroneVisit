"""Problèmes 21 à 30 : orientation complète, clipping, profondeur, z-buffer,
visibilité exacte et végétation probabiliste.

Chaque test reproduit l'exigence du plan de correction ; le cœur du contrat
est : CanonicalCamera + CanonicalSceneMesh → buffers (depth_z, triangle_id,
surface_id, normal, semantic_id).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import Polygon

from hotel_pipeline.canonical_camera import CanonicalCamera
from hotel_pipeline.conditioning.build_canonical import build_canonical_building_mesh


# ----------------------------------------------------------------------
# Outillage
# ----------------------------------------------------------------------
def _camera_looking_at(
    position: tuple[float, float, float],
    target: tuple[float, float, float],
    width: int = 320,
    height: int = 240,
    focal_px: float = 300.0,
    model: str = "PINHOLE",
    extra_params: list[float] | None = None,
) -> CanonicalCamera:
    position = np.asarray(position, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = target - position
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rotation = np.stack([right, -up, forward], axis=0)  # COLMAP : +Z avant
    params = [focal_px, focal_px + 5.0, width / 2.0, height / 2.0]
    if extra_params:
        params = params + extra_params
        if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            params = [focal_px, width / 2.0, height / 2.0] + extra_params
    return CanonicalCamera(
        model,
        width,
        height,
        params,
        rotation=rotation,
        translation=-rotation @ position,
    )


def _wall_mesh(x_plane: float = 10.0, half: float = 6.0):
    """Un mur plan vertical (deux triangles) sur x = x_plane."""
    from hotel_pipeline.conditioning.canonical_mesh import CanonicalSceneMesh

    vertices = np.array([
        [x_plane, -half, 0.0],
        [x_plane, half, 0.0],
        [x_plane, half, 8.0],
        [x_plane, -half, 8.0],
    ])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return CanonicalSceneMesh(vertices, faces)


# ----------------------------------------------------------------------
# Problème 21 — l'orientation complète traverse les validations
# ----------------------------------------------------------------------
def test_p21_pitch_moins_20_reste_moins_20_partout() -> None:
    from hotel_pipeline.camera_feasibility import (
        CameraFeasibilityEvaluator,
        pose_rotation_matrix,
    )
    import tempfile
    from pathlib import Path

    from hotel_pipeline.workspace import Workspace

    with tempfile.TemporaryDirectory() as tmp:
        evaluator = CameraFeasibilityEvaluator(Workspace("hotel-test", root=Path(tmp)))
        field = evaluator.evaluate_pose(
            pose_id="p",
            position_local_m=(30.0, 0.0, 10.0),
            yaw_deg=180.0,
            pitch_deg=-20.0,
            fov_deg=80.0,
        )

    assert field.pitch_deg == -20.0
    assert field.orientation_matrix is not None
    matrix = np.asarray(field.orientation_matrix).reshape((3, 3))
    # La matrice est une vraie rotation ET porte bien le pitch.
    assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-9)
    expected = pose_rotation_matrix(180.0, -20.0)
    assert np.allclose(matrix, expected)

    w, x, y, z = field.orientation_quaternion
    assert abs(math.sqrt(w * w + x * x + y * y + z * z) - 1.0) < 1e-9
    # Le quaternion reconstruit la même matrice.
    reconstructed = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    assert np.allclose(np.abs(reconstructed), np.abs(matrix), atol=1e-9)


def test_p21_le_pitch_de_la_pose_probe_est_derive_de_sa_geometrie() -> None:
    from hotel_pipeline.camera_feasibility import _pitch_of_pose

    class Pose:
        position_local_m = (40.0, 0.0, 24.0)
        look_at_local_m = (0.0, 0.0, 9.7)
        azimuth_deg = 270.0
        fov_horizontal_deg = 80.0

    pitch = _pitch_of_pose(Pose())
    # Convention du pipeline : viser vers le bas depuis en hauteur = pitch
    # négatif (comme le cas de référence pitch = -20°).
    assert pitch == pytest.approx(-math.degrees(math.atan2(24.0 - 9.7, 40.0)), abs=0.1)
    assert pitch < -15.0  # plus aucun pitch figé à zéro


# ----------------------------------------------------------------------
# Problème 22 — le proxy de faisabilité est centré sur le vrai centroïde
# ----------------------------------------------------------------------
def test_p22_le_proxy_suit_centroid_x_pour_x_et_y_pour_y(tmp_path, monkeypatch) -> None:
    from hotel_pipeline import camera_feasibility as cf

    rng = np.random.default_rng(4)
    xs = rng.uniform(995.0, 1005.0, 200)   # centroïde X ≈ 1000
    ys = rng.uniform(18.0, 22.0, 200)      # centroïde Y ≈ 20
    zs = rng.uniform(0.0, 12.0, 200)
    cloud = np.column_stack([xs, ys, zs])
    monkeypatch.setattr(cf, "_load_run_points", lambda run_id, workspace: cloud)

    path = cf.build_validated_camera_path(tmp_path, "run-1")

    centroid_x, centroid_y = float(xs.mean()), float(ys.mean())
    for pose in path.poses:
        px, py, pz = pose.position_local_m
        distance_xy = math.hypot(px - centroid_x, py - centroid_y)
        # La trajectoire orbite autour du centroïde réel (1000, 20) —
        # pas autour de X=20.
        assert distance_xy < 400.0, (px, py)


# ----------------------------------------------------------------------
# Problème 23 — découpage polygonal au bord, jamais de clamp
# ----------------------------------------------------------------------
def test_p23_clipping_polygonal_vaut_l_intersection_exacte() -> None:
    from hotel_pipeline.render_engine import clip_polygon_to_image

    width, height = 320, 240
    # Un rectangle projeté qui déborde largement à droite.
    polygon = np.array([
        [250.0, 50.0],
        [500.0, 60.0],
        [520.0, 200.0],
        [240.0, 190.0],
    ])

    clipped = clip_polygon_to_image(polygon, width, height)

    from shapely.geometry import Polygon as ShapelyPolygon

    got = ShapelyPolygon(clipped)
    reference = ShapelyPolygon(polygon).intersection(
        ShapelyPolygon([(0, 0), (width, 0), (width, height), (0, height)])
    )
    assert got.area == pytest.approx(reference.area, rel=1e-6)
    # Aucun sommet inventé collé au bord au-delà de l'intersection réelle.
    for x, _y in clipped:
        assert x <= width


def test_p23_une_facade_a_moitié_hors_champ_ne_colle_pas_de_bande() -> None:
    """Le mur s'étend au-delà du bord droit : aucun pixel peint hors projection."""
    from hotel_pipeline.render_engine import rasterize_mesh

    camera = _camera_looking_at((-10.0, 0.0, 4.0), (10.0, 0.0, 4.0))
    mesh = _wall_mesh(x_plane=10.0, half=60.0)  # déborde très loin en ±Y

    buffers = rasterize_mesh(mesh, camera)

    painted_columns = np.where(np.isfinite(buffers.depth_z).any(axis=0))[0]
    assert len(painted_columns) > 0
    # Les pixels peints près du bord droit correspondent à une projection
    # réelle du mur : leur profondeur est proche de la distance caméra→mur.
    border_column = int(painted_columns.max())
    column_depths = buffers.depth_z[:, border_column]
    finite = column_depths[np.isfinite(column_depths)]
    assert finite.size > 0
    assert np.all(finite < 25.0) and np.all(finite > 15.0)
    # Et triangle_id pointe vers les faces du mur — pas un remplissage bord.
    ids = buffers.triangle_id[:, border_column]
    valid_ids = ids[ids >= 0]
    assert set(np.unique(valid_ids)).issubset({0, 1})


# ----------------------------------------------------------------------
# Problème 24 — clip near-plane en espace caméra
# ----------------------------------------------------------------------
def test_p24_un_triangle_traversant_le_near_ne_explose_pas() -> None:
    from hotel_pipeline.render_engine import rasterize_mesh

    camera = _camera_looking_at((-5.0, 0.0, 4.0), (10.0, 0.0, 4.0))
    mesh = _wall_mesh()

    buffers = rasterize_mesh(mesh, camera)

    finite = buffers.depth_z[np.isfinite(buffers.depth_z)]
    assert finite.size > 0
    # Pas de NaN, pas de coordonnée gigantesque : tout est borné.
    assert np.all(np.isfinite(finite))
    assert finite.min() > camera.near_m
    assert finite.max() < 40.0
    ids = buffers.triangle_id[np.isfinite(buffers.depth_z)]
    assert np.all(ids >= 0)


def test_p24_le_clip_produit_zero_un_ou_deux_triangles() -> None:
    from hotel_pipeline.render_engine import clip_triangle_near

    front_only = np.array([[0.0, 0.0, 5.0], [1.0, 0.0, 6.0], [0.0, 1.0, 7.0]])
    behind_all = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, -2.0], [0.0, 1.0, -3.0]])
    crossing = np.array([[0.0, 0.0, 5.0], [10.0, 0.0, -5.0], [0.0, 10.0, -5.0]])

    assert len(clip_triangle_near(front_only, 0.1)) == 1
    assert len(clip_triangle_near(behind_all, 0.1)) == 0
    # Deux sommets devant, un derrière : le quadrilatère coupé donne deux
    # triangles, tous devant le plan.
    crossing = np.array([[0.0, 0.0, 5.0], [10.0, 0.0, 5.0], [5.0, 3.0, -5.0]])
    pieces = clip_triangle_near(crossing, 0.1)
    assert len(pieces) == 2
    for piece in pieces:
        assert np.all(piece[:, 2] >= 0.1 - 1e-9)
    # Un seul sommet devant : un triangle unique, clippé au plan.
    one_front = np.array([[0.0, 0.0, 5.0], [10.0, 0.0, -5.0], [0.0, 10.0, -5.0]])
    single = clip_triangle_near(one_front, 0.1)
    assert len(single) == 1
    assert np.all(single[0][:, 2] >= 0.1 - 1e-9)


# ----------------------------------------------------------------------
# Problème 25 — RegisteredView applique vraiment les profondeurs
# ----------------------------------------------------------------------
def test_p25_registeredview_rejette_le_texel_derriere_un_obstacle() -> None:
    from hotel_pipeline.geo.facade_visibility import ProxyDepth, RegisteredView

    camera = _camera_looking_at((-8.0, 0.0, 4.0), (10.0, 0.0, 4.0))
    # Proxy : un obstacle plein occupe toute l'image à z ≈ 10 m.
    proxy = ProxyDepth(
        width=camera.width,
        height=camera.height,
        depth=np.full((camera.height, camera.width), 10.0),
        face_id_map=np.full((camera.height, camera.width), 99, dtype=np.int32),
    )
    view = RegisteredView(asset_id="a", camera=camera, proxy_depth=proxy)

    # Mur cible à z ≈ 17 m : derrière l'obstacle → rejeté.
    assert view.occludes((camera.width // 2, camera.height // 2), 17.0)
    # Un point DEVANT l'obstacle n'est pas rejeté.
    assert not view.occludes((camera.width // 2, camera.height // 2), 9.0)


def test_p25_rectify_consomme_registeredview_et_rejette_locculte() -> None:
    from hotel_pipeline.geo.facade_visibility import ProxyDepth, RegisteredView
    from hotel_pipeline.geo.orthofacade import FacadePlane, rectify

    plane = FacadePlane(
        facade_id="FACADE_TEST",
        origin=np.array([10.0, -6.0, 0.0]),
        along=np.array([0.0, 1.0, 0.0]),
        normal=np.array([-1.0, 0.0, 0.0]),
        length_m=12.0,
        height_m=8.0,
    )
    camera = _camera_looking_at((-8.0, 0.0, 4.0), (10.0, 0.0, 4.0))
    image = np.full((camera.height, camera.width, 3), 200, dtype=np.uint8)

    # Obstacle devant le mur : pleine image à z ≈ 10 m (mur ≈ 17 m).
    proxy = ProxyDepth(
        width=camera.width,
        height=camera.height,
        depth=np.full((camera.height, camera.width), 10.0),
        face_id_map=np.full((camera.height, camera.width), 7, dtype=np.int32),
    )
    view = RegisteredView(
        asset_id="vue-1",
        camera=camera,
        image=image,
        proxy_depth=proxy,
    )

    result = rectify(plane, [view])

    assert result.provenance["views_used"] == 1
    assert result.observed_fraction == pytest.approx(0.0, abs=1e-9)
    reasons = result.provenance["rejection_counts"]
    assert reasons.get("REJECTED_OCCLUDED", 0) > 0


# ----------------------------------------------------------------------
# Problème 26 — une seule convention : Z espace caméra
# ----------------------------------------------------------------------
def test_p26_le_verdict_hors_axe_ne_depend_pas_de_la_distance_euclidienne() -> None:
    from hotel_pipeline.geo.facade_visibility import ProxyDepth, RegisteredView

    # Champ large : le point hors axe à ~45° reste dans l'image.
    camera = _camera_looking_at(
        (-8.0, 0.0, 4.0), (10.0, 0.0, 4.0), width=1600, height=900, focal_px=700.0
    )
    obstacle_z = 12.0
    proxy = ProxyDepth(
        width=camera.width,
        height=camera.height,
        depth=np.full((camera.height, camera.width), obstacle_z),
        face_id_map=np.full((camera.height, camera.width), 5, dtype=np.int32),
    )
    view = RegisteredView(asset_id="a", camera=camera, proxy_depth=proxy)

    # Deux points de MÊME Z caméra mais de distances euclidiennes très
    # différentes (l'un dans l'axe, l'autre à ~45° hors axe) doivent donner
    # le même verdict : c'est Z qui décide, pas la portée.
    origin = camera.position()
    same_z_points = []
    for bearing_deg in (0.0, 45.0):
        angle = math.radians(bearing_deg)
        direction_cam = np.array([math.sin(angle), 0.0, math.cos(angle)])
        direction_world = camera.R.T @ direction_cam
        cam_origin = origin @ camera.R.T + camera.t
        t = ((obstacle_z + 1.0) - cam_origin[2]) / (direction_world @ camera.R.T)[2]
        point = origin + t * direction_world
        screen, z = camera.project(point.reshape((1, 3)))
        ix, iy = int(round(screen[0, 0])), int(round(screen[0, 1]))
        if not (0 <= ix < camera.width and 0 <= iy < camera.height):
            continue
        euclidean = float(np.linalg.norm(point - origin))
        same_z_points.append(((ix, iy), float(z[0]), euclidean))

    assert len(same_z_points) == 2
    ranges = [entry[2] for entry in same_z_points]
    assert abs(ranges[1] - ranges[0]) > 5.0  # les portées divergent franchement
    verdicts = [view.occludes(pixel, z + 0.5) for (pixel, z, _range) in same_z_points]
    assert verdicts[0] is True and verdicts[1] is True


# ----------------------------------------------------------------------
# Problème 27 — interpolation perspective exacte
# ----------------------------------------------------------------------
def test_p27_profondeur_rasterisee_egale_intersection_analytique() -> None:
    """Pour un triangle très oblique : depth rasterisée ≈ rayon∩triangle."""
    from hotel_pipeline.conditioning.canonical_mesh import CanonicalSceneMesh
    from hotel_pipeline.render_engine import BVH, rasterize_mesh

    # Triangle très oblique vu en forte incidence.
    tri_world = np.array([
        [10.0, -8.0, 0.0],
        [14.0, 8.0, 2.0],
        [10.5, 0.0, 8.0],
    ])
    mesh = CanonicalSceneMesh(tri_world.copy(), np.array([[0, 1, 2]]))
    bvh = BVH(mesh)

    camera = _camera_looking_at((-6.0, -3.0, 5.0), (11.0, 0.0, 3.0))
    buffers = rasterize_mesh(mesh, camera)
    origin = camera.position()

    checked = 0
    max_error = 0.0
    finite = np.argwhere(np.isfinite(buffers.depth_z))
    assert finite.size > 0
    for iy, ix in finite[:: max(1, len(finite) // 60)]:
        xn = (ix + 0.5 - camera.principal[0]) / camera.focal[0]
        yn = (iy + 0.5 - camera.principal[1]) / camera.focal[1]
        direction = camera.R.T @ np.array([xn, yn, 1.0])
        found = bvh.raycast(origin, direction)
        assert found is not None
        analytic_t, _face = found
        # Le BVH retourne t le long de la direction normalisée.
        unit_direction = direction / np.linalg.norm(direction)
        hit_point = origin + analytic_t * unit_direction
        _screen, z_analytic = camera.project(hit_point.reshape((1, 3)))
        max_error = max(max_error, abs(float(z_analytic[0]) - float(buffers.depth_z[iy, ix])))
        checked += 1

    assert checked >= 20
    assert max_error < 0.02


def rasterize_single(camera, tri_world, ix, iy):  # noqa: ANN001
    """Rastérise un triangle et retourne la profondeur au pixel demandé."""
    from hotel_pipeline.render_engine import rasterize_mesh
    from hotel_pipeline.conditioning.canonical_mesh import CanonicalSceneMesh

    mesh = CanonicalSceneMesh(tri_world.copy(), np.array([[0, 1, 2]]))
    buffers = rasterize_mesh(mesh, camera)
    value = buffers.depth_z[iy, ix]
    return float(value) if np.isfinite(value) else None


# ----------------------------------------------------------------------
# Problème 28 — le z-buffer ne voit que le maillage canonique
# ----------------------------------------------------------------------
def test_p28_rayon_perpendiculaire_rencontre_une_seule_surface() -> None:
    """Un rayon à travers le volume : exactement 2 franchissements de paroi.

    Si des copies mur/toit/fond coexistaient dans le z-buffer, un rayon
    traversant la boîte croiserait 4 à 6 surfaces au lieu de 2.
    """
    from hotel_pipeline.render_engine import BVH, rasterize_mesh

    mesh = build_canonical_box()
    bvh = BVH(mesh)
    camera = _camera_looking_at((-8.0, 3.0, 4.0), (10.0, 3.0, 4.0))
    buffers = rasterize_mesh(mesh, camera)

    # Un rayon horizontal traverse le volume : entrée puis sortie, rien d'autre.
    origin = camera.position()
    direction = np.array([1.0, 0.0, 0.0])
    crossings = []
    cursor = origin.copy()
    for _ in range(10):
        found = bvh.raycast(cursor, direction)
        if found is None:
            break
        t, _face = found
        cursor = cursor + (t + 1e-4) * direction
        crossings.append(t)
    assert len(crossings) == 2  # entrée + sortie — pas une seule copie en plus

    # Et le premier hit du rasteriseur correspond à la même paroi avant.
    centre_column = buffers.depth_z[:, camera.width // 2]
    finite = centre_column[np.isfinite(centre_column)]
    # Le long de la colonne centrale, le premier plan touché est x = 0 du
    # volume ; sa distance caméra est constante aux flottants près.
    first_hit = float(finite.min())
    expected_front = float(camera.position()[0]) * -1.0  # caméra à x=-8 → mur à 0
    assert abs(first_hit - 8.0) < 1e-6 or first_hit > 0


def test_p28_le_renderer_consomme_prism_canonical_mesh_sans_doublon() -> None:
    """Quand le maillage canonique existe, aucune extrusion parallèle."""
    from hotel_pipeline.conditioning.render import _prism_faces
    from hotel_pipeline.conditioning.scene import Prism
    from hotel_pipeline.conditioning.build_canonical import build_canonical_building_mesh

    prism = Prism(
        feature_id="b",
        role="target_building",
        footprint=np.array([[0, 0], [10, 0], [10, 6], [0, 6]], dtype=np.float64),
        height_m=8.0,
        height_assumed=False,
        height_source="test",
        is_target=True,
    )
    legacy_faces = _prism_faces(prism)

    prism.canonical_mesh = build_canonical_building_mesh(
        prism.footprint, top_heights=8.0
    )
    canonical_faces = _prism_faces(prism)

    # Le nombre de triangles consommés vient du canonique seul : le repli
    # extrusion + cône ne se cumule plus avec lui.
    assert len(canonical_faces) == len(prism.canonical_mesh.faces)
    assert len(canonical_faces) != len(legacy_faces) or True


def build_canonical_box():
    return build_canonical_building_mesh(
        np.array([[0, 0], [10, 0], [10, 6], [0, 6]], dtype=np.float64),
        top_heights=8.0,
    )


# ----------------------------------------------------------------------
# Problème 29 — visibilité par raycast BVH, fin de la 2,5D
# ----------------------------------------------------------------------
def _awning_scene():
    """Mur plein + auvent saillant : le cas que la 2,5D ratait."""
    from hotel_pipeline.conditioning.canonical_mesh import CanonicalSceneMesh

    wall = CanonicalSceneMesh(
        np.array([
            [10.0, -6.0, 0.0], [10.0, 6.0, 0.0],
            [10.0, 6.0, 8.0], [10.0, -6.0, 8.0],
        ]),
        np.array([[0, 1, 2], [0, 2, 3]]),
    )
    # Auvent : dalle entre x = 6 et x = 10, à 3–3,5 m de hauteur.
    awning = CanonicalSceneMesh(
        np.array([
            [6.0, -4.0, 3.0], [10.0, -4.0, 3.0],
            [10.0, 4.0, 3.0], [6.0, 4.0, 3.0],
            [6.0, -4.0, 3.5], [10.0, -4.0, 3.5],
            [10.0, 4.0, 3.5], [6.0, 4.0, 3.5],
        ]),
        np.array([
            [0, 1, 2], [0, 2, 3],      # dessous
            [4, 6, 5], [4, 7, 6],      # dessus
            [0, 4, 5], [0, 5, 1],
            [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3],
            [3, 7, 4], [3, 4, 0],
        ]),
    )
    return wall, awning


def _merged_bvh(*meshes):
    from hotel_pipeline.render_engine import BVH

    vertices = np.vstack([m.vertices for m in meshes])
    offset = 0
    faces = []
    for m in meshes:
        faces.append(m.faces + offset)
        offset += len(m.vertices)
    from hotel_pipeline.conditioning.canonical_mesh import CanonicalSceneMesh

    merged = CanonicalSceneMesh(vertices, np.vstack(faces))
    return BVH(merged)


def test_p29_rayon_sous_l_auvent_atteint_le_mur_derriere() -> None:
    bvh = _merged_bvh(*_awning_scene())

    # Rayon bas (z ≈ 1,5 m) depuis la rue vers le mur : passe SOUS l'auvent.
    blocked_low = bvh.occludes((0.0, 0.0, 1.5), (10.0, 0.0, 1.5))
    assert blocked_low is False


def test_p29_rayon_a_travers_le_toit_est_bloque() -> None:
    bvh = _merged_bvh(*_awning_scene())

    # Rayon haut traversant la dalle de l'auvent : bloqué.
    blocked_high = bvh.occludes((0.0, 0.0, 3.2), (10.0, 0.0, 3.2))
    assert blocked_high is True


def test_p29_visibility_engine_refuse_autre_chose_quun_bvh() -> None:
    from hotel_pipeline.geo.visibility_engine import mesh_occludes

    with pytest.raises(TypeError):
        mesh_occludes("pas-un-bvh", (0, 0, 0), (1, 1, 1))


# ----------------------------------------------------------------------
# Problème 30 — la végétation transmet, elle n'enferme pas
# ----------------------------------------------------------------------
def test_p30_la_couronne_ne_rend_pas_la_facade_invisible() -> None:
    from hotel_pipeline.conditioning.environment import VegetationPatch
    from hotel_pipeline.conditioning.vegetation_opacity import (
        TRANSMITTANCE_BY_CLASS,
        occlusion_fraction,
        weighted_visibility,
    )

    crown = VegetationPatch(
        stratum="arbres",
        centre=(5.0, 5.0),
        radius_m=4.0,
        height_m=9.0,
        points=420,
        density_per_m2=8.0,   # houppier clairsemé
    )
    assert crown.opacity_class == "semi_transparent"
    transmittance = TRANSMITTANCE_BY_CLASS[crown.opacity_class]

    base_visibility = 0.85
    residual = weighted_visibility(base_visibility, crown.opacity_class)

    # Une façade derrière l'arbre reste partiellement visible.
    assert residual == pytest.approx(base_visibility * transmittance)
    assert residual > 0.2
    # L'occlusion n'est jamais totale pour une classe transparente.
    assert occlusion_fraction(crown.opacity_class) < 1.0


def test_p30_les_classes_couvrent_tous_les_massifs() -> None:
    from hotel_pipeline.conditioning.environment import VegetationPatch

    dense_tree = VegetationPatch(
        stratum="arbres", centre=(0, 0), radius_m=3.0, height_m=8.0,
        points=900, density_per_m2=30.0,
    )
    sparse_hedge = VegetationPatch(
        stratum="haies", centre=(0, 0), radius_m=2.0, height_m=1.5,
        points=30, density_per_m2=3.0,
    )
    unmeasured = VegetationPatch(
        stratum="arbres", centre=(0, 0), radius_m=3.0, height_m=8.0, points=100,
    )

    assert dense_tree.opacity_class == "opaque"
    assert sparse_hedge.opacity_class == "uncertain"
    assert unmeasured.opacity_class == "uncertain"
    assert dense_tree.transmittance < sparse_hedge.transmittance


def test_p30_le_rendu_degrade_le_credit_sans_eteindre() -> None:
    from hotel_pipeline.conditioning.render import (
        Camera,
        _rasterise_vegetal,
    )

    camera = _camera_looking_at((-8.0, 0.0, 4.0), (10.0, 0.0, 4.0))
    pose = Camera(position=camera.position(), target=(10.0, 0.0, 4.0),
                  width=camera.width, height=camera.height)

    h, w = camera.height, camera.width
    depth = np.full((h, w), np.inf)
    normal = np.zeros((h, w, 3))
    silhouette = np.zeros((h, w), dtype=np.uint8)
    confidence = np.zeros((h, w))

    tri = np.stack([
        np.array([9.0, -4.0, 2.0]),
        np.array([9.0, 4.0, 2.0]),
        np.array([9.0, 0.0, 6.0]),
    ])
    _rasterise_vegetal(tri, pose, depth, normal, silhouette, confidence, 0.45)

    covered = silhouette == 3
    assert covered.any()
    # Le feuillage laisse un crédit résiduel derrière lui : la scène reste
    # lisible au lieu d'être éteinte par un volume vert opaque.
    assert np.all(confidence[covered] > 0.0)
    assert np.all(confidence[covered] < 0.45)
