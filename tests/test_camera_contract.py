"""Problèmes 11 à 20 : cours intérieures, provenance, contrat caméra.

Chaque test est l'exigence du plan de correction, pas un reflet de
l'implémentation. Les tests 15 à 20 verrouillent le contrat caméra :
CanonicalCamera est la seule définition partagée par COLMAP, les textures,
le z-buffer et le viewer.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

pycolmap = pytest.importorskip("pycolmap")

from hotel_pipeline.canonical_camera import (  # noqa: E402
    CanonicalCamera,
    asset_identity,
    camera_group_key,
    canonize_image,
    group_by_camera,
    transform_mask,
)
from hotel_pipeline.conditioning.build_canonical import build_canonical_building_mesh  # noqa: E402


# ----------------------------------------------------------------------
# Problème 11 — les cours intérieures restent vides
# ----------------------------------------------------------------------
def _courtyard_mesh():
    footprint = np.array([[0, 0], [20, 0], [20, 16], [0, 16]], dtype=np.float64)
    courtyard = np.array([[6, 6], [14, 6], [14, 10], [6, 10]], dtype=np.float64)
    return build_canonical_building_mesh(footprint, top_heights=9.0, interiors=[courtyard])


def test_p11_la_cour_reste_vide_au_centre() -> None:
    mesh = _courtyard_mesh()

    centre = Point(10.0, 8.0)
    covering = [
        index
        for index, face in enumerate(mesh.faces)
        if Polygon(mesh.vertices[face][:, :2]).covers(centre)
    ]
    assert covering == [], "un triangle referme la cour"


def test_p11_le_maillage_de_cour_est_etanche_et_traversable() -> None:
    mesh = _courtyard_mesh()

    audit = mesh.audit()
    assert audit["boundary_edges"] == 0

    # Le raycast traverse le mur intérieur : depuis la cour vers l'extérieur,
    # la première intersection est le mur de cour lui-même (x = 14).
    hit = mesh.raycast(np.array([10.0, 8.0, 8.0]), np.array([1.0, 0.0, 0.0]))
    assert hit == pytest.approx(4.0, abs=1e-3)


# ----------------------------------------------------------------------
# Problème 12 — toit inconnu : aucune architecture inventée
# ----------------------------------------------------------------------
def test_p12_sans_mesure_le_toit_est_une_enveloppe_minimale_inconnue() -> None:
    from hotel_pipeline.conditioning.render import _prism_faces
    from hotel_pipeline.conditioning.scene import Prism

    prism = Prism(
        feature_id="b",
        role="target_building",
        footprint=np.array([[0, 0], [12, 0], [12, 8], [0, 8]], dtype=np.float64),
        height_m=10.0,
        height_assumed=True,
        height_source="hypothèse",
        is_target=True,
    )

    faces = _prism_faces(prism)
    roof_faces = [tri for tri, is_roof in faces if is_roof]

    assert roof_faces, "une enveloppe minimale ferme encore le volume"
    # Toutes les faces de toit sont coplanaires et horizontales à h : aucun
    # cône, aucune pente fictive.
    for tri in roof_faces:
        assert np.allclose(tri[:, 2], 10.0)

    # La provenance dit UNKNOWN, jamais "measured".
    assert prism.roof_provenance_class == "UNKNOWN_MINIMAL_ENVELOPE"

    # Et côté maillage canonique, même promesse : le repli reste un couvercle
    # plat, jamais une forme architecturale.
    mesh = build_canonical_building_mesh(
        np.array([[0, 0], [12, 0], [12, 8], [0, 8]], dtype=np.float64),
        top_heights=10.0,
    )
    roof_indices = [i for i, kind in enumerate(mesh.face_kind) if kind == "roof"]
    for i in roof_indices:
        assert np.allclose(mesh.vertices[mesh.faces[i]][:, 2], 10.0)


# ----------------------------------------------------------------------
# Problème 13 — les trous LiDAR ne deviennent pas des surfaces inventées
# ----------------------------------------------------------------------
def test_p13_un_gros_trou_lidar_reste_unknown() -> None:
    from hotel_pipeline.conditioning.heights import (
        build_roof_surface_from_cloud,
        _gate_hole_filling,
        MAX_FILL_DISTANCE_M,
    )

    rng = np.random.default_rng(2)
    left = np.column_stack([
        rng.uniform(0.0, 4.5, 800),
        rng.uniform(0.0, 10.0, 800),
        np.full(800, 10.0),
    ])
    right = np.column_stack([
        rng.uniform(9.5, 14.0, 800),
        rng.uniform(0.0, 10.0, 800),
        np.full(800, 10.0),
    ])
    points = np.vstack([left, right])  # trou de ~5 m au milieu

    surface = build_roof_surface_from_cloud(points, np.array([[0, 0], [14, 0], [14, 10], [0, 10]], dtype=float))
    assert surface is not None
    vertices, faces = surface

    # Aucun sommet dans la zone inconnue : le trou n'est pas recousu.
    centre = Point(7.0, 5.0)
    inside_hole = sum(
        1 for face in faces if Polygon(vertices[face][:, :2]).covers(centre)
    )
    assert inside_hole == 0

    # Et le garde-fou bas niveau expose bien ses comptes UNKNOWN.
    valid = np.ones((5, 5), dtype=bool)
    values = np.where(valid, 10.0, np.nan)
    filled, audit = _gate_hole_filling(valid, values, cell_m=1.0)
    assert MAX_FILL_DISTANCE_M <= 1.0


def test_p13_les_petits_trous_sont_combles_avec_confiance_degradee() -> None:
    from hotel_pipeline.conditioning.heights import _gate_hole_filling

    grid = np.full((7, 7), np.nan)
    grid[::2, ::2] = 10.0  # mesures éparses : trous ≤ ~1,4 m
    valid = np.isfinite(grid)

    filled, audit = _gate_hole_filling(valid, grid, cell_m=1.0)
    assert audit["filled_cells"] > 0
    assert 0.0 < (audit["confidence_min"] or 1.0) < 1.0


# ----------------------------------------------------------------------
# Problème 14 — triangulation contrainte, concavités et trous
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,polygon",
    [
        ("L", Polygon([(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)])),
        ("U", Polygon([(0, 0), (12, 0), (12, 8), (9, 8), (9, 3), (3, 3), (3, 8), (0, 8)])),
        (
            "cour",
            Polygon([(0, 0), (20, 0), (20, 16), (0, 16)], holes=[[(6, 6), (14, 6), (14, 10), (6, 10)]]),
        ),
        (
            "deux-cours",
            Polygon(
                [(0, 0), (30, 0), (30, 24), (0, 24)],
                holes=[[(8, 8), (22, 8), (22, 16), (8, 16)], [(2, 2), (4, 2), (4, 4), (2, 4)]],
            ),
        ),
    ],
)
def test_p14_triangulation_contrainte_sans_debordement_ni_manque(name, polygon) -> None:
    from hotel_pipeline.conditioning.constrained_triangulation import (
        triangulate_constrained,
    )

    triangles = triangulate_constrained(polygon)

    total = sum(
        abs(
            (t[1, 0] - t[0, 0]) * (t[2, 1] - t[0, 1])
            - (t[1, 1] - t[0, 1]) * (t[2, 0] - t[0, 0])
        )
        / 2.0
        for t in triangles
    )
    assert total == pytest.approx(polygon.area, rel=1e-6)

    for hole in polygon.interiors:
        probe = hole.representative_point()
        for t in triangles:
            assert not Polygon(t).contains(probe)


# ----------------------------------------------------------------------
# Problème 15 — EXIF réellement normalisé
# ----------------------------------------------------------------------
def _write_image(path, size=(8, 6), orientation=None):
    from PIL import Image

    suffix = path.suffix.lower()
    image = Image.new("RGB", size)
    for x in range(size[0]):
        for y in range(size[1]):
            image.putpixel((x * 1, y), ((x * 31) % 256, (y * 17) % 256, (x + y) % 256))
    if orientation is not None:
        exif = image.getexif()
        exif[274] = orientation
        image.save(path, exif=exif.tobytes())
    else:
        image.save(path)
    return path


def test_p15_exif_rotation_et_photo_tournee_convergent(tmp_path) -> None:
    """Même photo avec EXIF rotation et photo physiquement tournée : mêmes pixels.

    Le PNG évite la perte JPEG : l'égalité doit être exacte au pixel.
    """
    from PIL import Image, ImageOps

    plain_path = tmp_path / "plain.png"
    tagged_path = tmp_path / "tagged.png"
    _write_image(plain_path)
    _write_image(tagged_path, orientation=6)

    canonical_tagged = canonize_image(tagged_path)
    assert canonical_tagged.lineage.steps[0]["orientation_before"] == 6

    # La photo physiquement tournée (pixels déjà droits, sans EXIF) doit
    # donner exactement les mêmes pixels que la photo redressée par EXIF.
    rotated_physically = Image.open(tagged_path).transpose(Image.Transpose.ROTATE_270)
    straight_path = tmp_path / "straight.png"
    rotated_physically.save(straight_path)
    canonical_straight = canonize_image(straight_path)

    a = np.asarray(canonical_tagged.image.convert("RGB"))
    b = np.asarray(canonical_straight.image.convert("RGB"))
    assert a.shape == b.shape
    assert np.array_equal(a, b)


def test_p15_l_image_canonique_porte_orientation_1_et_sa_transform(tmp_path) -> None:
    path = tmp_path / "photo.png"
    _write_image(path, orientation=6)
    canonical = canonize_image(path)

    payload = canonical.as_dict()
    assert payload["lineage"]["steps"][0]["orientation_after"] == 1
    matrix = np.asarray(payload["lineage"]["transform_original_to_canonical"])
    assert matrix.shape == (2, 3)


def test_p15_toutes_les_orientations_redressent_comme_pil(tmp_path) -> None:
    """Les transform EXIF 1–8 coïncident pixel pour pixel avec PIL."""
    from PIL import Image, ImageOps

    source = tmp_path / "src.png"
    _write_image(source)

    for orientation in range(2, 9):
        path = tmp_path / f"o{orientation}.png"
        image = Image.open(source)
        exif = image.getexif()
        exif[274] = orientation
        image.save(path, exif=exif.tobytes())

        canonical = canonize_image(path)
        got = np.asarray(canonical.image.convert("RGB"))
        expected = np.asarray(
            ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        )
        assert got.shape == expected.shape, orientation
        assert np.array_equal(got, expected), orientation


# ----------------------------------------------------------------------
# Problème 16 — identité par contenu, pas par nom de fichier
# ----------------------------------------------------------------------
def test_p16_deux_img001_differentes_restant_distinctes(tmp_path) -> None:
    (tmp_path / "autre").mkdir()
    first = _write_image(tmp_path / "IMG_001.png")
    second = _write_image(tmp_path / "autre" / "IMG_001.png", size=(10, 8))

    id_first = asset_identity(first)
    id_second = asset_identity(second)

    # L'empreinte de contenu (préfixe) sépare deux fichiers distincts,
    # même baptisés IMG_001 tous les deux.
    assert id_first.split("_")[0] != id_second.split("_")[0]
    assert "IMG_001" in id_first and "IMG_001" in id_second
    # Même contenu, autre nom : la partie empreinte reste identique.
    copy = tmp_path / "copie.png"
    copy.write_bytes(first.read_bytes())
    assert asset_identity(copy).split("_")[0] == id_first.split("_")[0]


# ----------------------------------------------------------------------
# Problème 17 — groupes de vraies caméras
# ----------------------------------------------------------------------
def test_p17_iphone_drone_street_view_ne_partagent_jamais_les_intrinseques() -> None:
    iphone = {
        "sensor": "Apple/iPhone 13", "width": 4032, "height": 3024,
        "focal_length_mm": 5.7, "source": "ground",
    }
    drone = {
        "sensor": "DJI/FC3411", "width": 5472, "height": 3648,
        "focal_length_mm": 8.8, "source": "aerial",
    }
    street = {
        "sensor": "Google/street_view", "width": 8192, "height": 4096,
        "focal_length_mm": 0.0, "source": "street_view", "panorama": True,
    }

    groups = group_by_camera([(iphone, "a"), (drone, "b"), (street, "c"), (iphone, "d")])

    assert len(groups) == 3
    assert camera_group_key(iphone) == camera_group_key({**iphone, "source": "ground"})
    # Un crop déclaré change la vraie caméra : nouveau groupe.
    cropped = {**iphone, "crop": [100, 100, 3800, 2800]}
    assert camera_group_key(cropped) != camera_group_key(iphone)


def test_p17_single_camera_ne_fusionne_pas_des_capteurs_differents() -> None:
    """L'option ne peut pas produire une clé commune à deux matériels."""
    a = {"sensor": "A", "width": 4000, "height": 3000}
    b = {"sensor": "B", "width": 1920, "height": 1080}

    assert camera_group_key(a) != camera_group_key(b)


# ----------------------------------------------------------------------
# Problème 18 — distorsion exacte COLMAP
# ----------------------------------------------------------------------
def _compare_with_colmap(model_name: str, params: list[float]) -> None:
    camera = pycolmap.Camera(
        model=model_name, width=1920, height=1080, params=params, camera_id=1
    )
    contract = CanonicalCamera(model_name, 1920, 1080, params)

    rng = np.random.default_rng(0)
    points = np.column_stack([
        rng.uniform(-30.0, 30.0, 200),
        rng.uniform(-20.0, 20.0, 200),
        rng.uniform(5.0, 60.0, 200),
    ])

    ours, _depth = contract.project(points)

    reference = np.asarray(camera.img_from_cam(points), dtype=float)

    assert np.abs(ours - reference).max() < 0.1, model_name


@pytest.mark.parametrize(
    "model,params",
    [
        ("SIMPLE_PINHOLE", [1500.0, 960.0, 540.0]),
        ("PINHOLE", [1500.0, 1520.0, 960.0, 540.0]),
        ("SIMPLE_RADIAL", [1500.0, 960.0, 540.0, -0.21]),
        ("RADIAL", [1500.0, 960.0, 540.0, -0.21, 0.06]),
        ("OPENCV", [1500.0, 1510.0, 960.0, 540.0, -0.21, 0.08, 1e-4, -3e-5]),
        (
            "FULL_OPENCV",
            [1500.0, 1510.0, 960.0, 540.0, -0.2, 0.07, 1e-4, -3e-5, -0.01, 0.005, -0.002, 0.001],
        ),
        ("OPENCV_FISHEYE", [1200.0, 1210.0, 960.0, 540.0, -0.1, 0.02, -0.003, 0.0004]),
    ],
)
def test_p18_projection_colmap_a_moins_dun_dixieme_de_pixel(model, params) -> None:
    _compare_with_colmap(model, params)


def test_p18_les_coefficients_sont_conserves_bord_compris() -> None:
    params = [1500.0, 1510.0, 960.0, 540.0, -0.21, 0.08, 1e-4, -3e-5]
    contract = CanonicalCamera("OPENCV", 1920, 1080, params)
    restored = CanonicalCamera.from_dict(contract.as_dict())

    assert restored.model == "OPENCV"
    assert np.allclose(restored.params, params)

    # Un point proche du bord subit bien la distorsion : il s'écarte du pinhole.
    edge = np.array([[1900.0, 20.0, 1.0]])
    undistorted = np.column_stack([
        params[0] * edge[:, 0] / edge[:, 2] + params[2],
        params[1] * edge[:, 1] / edge[:, 2] + params[3],
    ])
    projected, _ = contract.project(edge)
    assert np.abs(projected - undistorted).max() > 1.0


# ----------------------------------------------------------------------
# Problème 19 — chaîne original → EXIF → crop → resize propagée
# ----------------------------------------------------------------------
def test_p19_une_image_reduite_de_moitie_garde_la_meme_geometrie() -> None:
    fx, fy, cx, cy = 1800.0, 1810.0, 960.0, 540.0
    camera = CanonicalCamera("PINHOLE", 1920, 1080, [fx, fy, cx, cy])

    point_world = np.array([[3.0, 1.5, 12.0]])
    u, _depth = camera.project(point_world)

    half = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
    reduced = camera.adapt_to(half, 960, 540)

    u2, _ = reduced.project(point_world)

    # La projection physique se conserve : fx/fy/cx/cy tous divisés par deux.
    assert reduced.focal[0] == pytest.approx(fx * 0.5)
    assert reduced.principal[0] == pytest.approx(cx * 0.5)
    assert u2 == pytest.approx(u * 0.5)


def test_p19_le_crop_decale_le_point_principal_exactement() -> None:
    camera = CanonicalCamera("PINHOLE", 1920, 1080, [1800.0, 1810.0, 960.0, 540.0])
    crop = np.array([[1.0, 0.0, -120.0], [0.0, 1.0, -40.0]])

    cropped = camera.adapt_to(crop, 1600, 900)

    assert cropped.principal == pytest.approx((960.0 - 120.0, 540.0 - 40.0))
    assert cropped.focal == pytest.approx(camera.focal)


def test_p19_un_masque_suivit_la_lignee_pixel(tmp_path) -> None:
    mask = np.zeros((6, 8), dtype=np.uint8)
    mask[0, 0] = 1  # coin haut-gauche
    lineage = canonize_image(_write_image(tmp_path / "p.png")).lineage
    transformed = transform_mask(mask, lineage)  # orientation 1 : identité

    assert np.array_equal(transformed, mask)


def test_p19_exif_transpose_est_propage_aux_intrinseques(tmp_path) -> None:
    """Une photo stockée tournée voit sa focale suivre la transposition."""
    path = tmp_path / "rot.png"
    _write_image(path, orientation=6)
    canonical = canonize_image(path)

    camera = CanonicalCamera("PINHOLE", 8, 6, [700.0, 710.0, 4.0, 3.0])
    adapted = canonical.lineage.apply_intrinsics(camera)

    # Orientation 6 : w/h échangés, focales suivent la rotation.
    assert (adapted.width, adapted.height) == (6, 8)
    assert adapted.focal[0] == pytest.approx(710.0)
    assert adapted.focal[1] == pytest.approx(700.0)


# ----------------------------------------------------------------------
# Problème 20 — viewer et pipeline partagent la même caméra
# ----------------------------------------------------------------------
def test_p20_le_zbuffer_projette_via_la_meme_canonical_camera() -> None:
    from hotel_pipeline.conditioning.render import Camera, _project

    contract = CanonicalCamera(
        "OPENCV",
        1280,
        720,
        [1100.0, 1105.0, 640.0, 360.0, -0.18, 0.05, 2e-4, -1e-4],
        rotation=np.eye(3),
        translation=np.array([0.0, 0.0, -25.0]),
    )
    pose = Camera.from_canonical(contract)

    point = np.array([[2.0, 1.0, 0.0]])
    through_contract, depth_contract = contract.project(point)
    through_render, depth_render = _project(point, pose)

    assert np.abs(through_contract - through_render).max() < 1e-9
    assert np.allclose(depth_contract, depth_render)


def test_distorted_pixel_round_trips_through_unique_camera_ray() -> None:
    camera = CanonicalCamera(
        "SIMPLE_RADIAL", 1200, 800, [900.0, 600.0, 400.0, 0.18],
        translation=np.array([2.0, -3.0, 1.0]),
    )
    pixel = np.array([1040.0, 690.0])
    ray = camera.ray_from_pixel(*pixel, world=False)
    point_camera = ray * (12.0 / ray[2])
    world = camera.R.T @ (point_camera - camera.t)
    projected, _ = camera.project(world[None, :])
    np.testing.assert_allclose(projected[0], pixel, atol=1e-5)


def test_exif_axis_rotation_transforms_translation_and_preserves_centre() -> None:
    camera = CanonicalCamera(
        "PINHOLE", 800, 600, [700.0, 710.0, 400.0, 300.0],
        translation=np.array([4.0, -2.0, 7.0]),
    )
    transform = np.array([[0.0, -1.0, 600.0], [1.0, 0.0, 0.0]])
    adapted = camera.adapt_to(transform, 600, 800)
    np.testing.assert_allclose(adapted.position(), camera.position(), atol=1e-12)
    q3 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(adapted.t, q3 @ camera.t)


def test_p20_le_contrat_survit_a_la_serialisation_viewer() -> None:
    contract = CanonicalCamera(
        "SIMPLE_RADIAL",
        1024,
        768,
        [900.0, 512.0, 384.0, -0.15],
        near_m=0.5,
        far_m=500.0,
        camera_id="cam-7",
        group="iphone",
    )
    restored = CanonicalCamera.from_dict(contract.as_dict())

    point = np.array([[1.0, -2.0, 10.0]])
    a, _ = contract.project(point)
    b, _ = restored.project(point)
    assert np.abs(a - b).max() < 1e-9
    assert restored.near_m == 0.5 and restored.far_m == 500.0


def test_p20_le_payload_viewer_porte_le_contrat_et_projette_a_l_identique() -> None:
    from hotel_pipeline.conditioning.canonical import viewer_payload
    from hotel_pipeline.conditioning.render import Camera, _project

    payload = viewer_payload({
        "hotel_id": "x",
        "volumes": [
            {
                "id": "t",
                "target": True,
                "fp": [[0, 0], [20, 0], [20, 12], [0, 12]],
                "h": 10.0,
            }
        ],
    })

    embedded = payload["canonical_camera"]
    assert embedded["contract"] == "canonical_camera/1"
    contract = CanonicalCamera.from_dict(embedded)
    pose = Camera.from_canonical(contract)

    point = np.array([[5.0, 6.0, 4.0]])
    through_contract, depth_contract = contract.project(point)
    through_render, depth_render = _project(point, pose)

    # Même point 3D → écart nul côté Python et côté viewer.
    assert np.abs(through_contract - through_render).max() < 1e-9
    assert np.allclose(depth_contract, depth_render)
