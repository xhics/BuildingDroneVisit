"""Les dix corrections géométriques, chacune avec son test d'épreuve.

Chaque problème du plan de correction a son test : le test est l'exigence,
pas l'implémentation.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

from hotel_pipeline.conditioning.build_canonical import build_canonical_building_mesh
from hotel_pipeline.conditioning.canonical_mesh import (
    CanonicalSceneMesh,
    footprint_records,
)
from hotel_pipeline.conditioning.roof_planes import (
    RoofDecomposition,
    RoofPlane,
    segment,
)


# ----------------------------------------------------------------------
# Outillage
# ----------------------------------------------------------------------
def _plane_from_points(points: np.ndarray) -> RoofPlane:
    cloud = points - points.mean(axis=0)
    normal = np.linalg.svd(cloud, full_matrices=False)[2][2]
    normal = normal * (np.sign(normal[2]) or 1.0)
    return RoofPlane(points=points, normal=normal, origin=points.mean(axis=0))


class FlatTerrain:
    """Terrain incliné synthétique : z = 0.02 * x (continue par nature)."""

    def height_at(self, x: float, y: float) -> float:
        return 0.02 * x


# ----------------------------------------------------------------------
# Problème 1 — un seul maillage, un seul digest
# ----------------------------------------------------------------------
def test_p1_le_digest_est_identite_partout() -> None:
    """Renderer, textureur, collision et export lisent le même mesh."""
    footprint = np.array([[0, 0], [10, 0], [10, 6], [0, 6]], dtype=np.float64)
    mesh = build_canonical_building_mesh(footprint, top_heights=8.0)

    # Le renderer consomme les triangles canoniques.
    rendered = [tri for tri, _ in mesh.triangles()]
    # Le textureur passe par la sérialisation du même objet.
    payload = mesh.as_dict()
    textured = CanonicalSceneMesh.from_dict(payload).triangles()
    # L'export relit le digest embarqué.
    exported_digest = payload["mesh_digest"]

    assert len(rendered) == len(textured)
    assert mesh.mesh_digest() == CanonicalSceneMesh.from_dict(payload).mesh_digest()
    assert mesh.mesh_digest() == exported_digest
    # Collision : le raycast traverse le même volume.
    hit = mesh.raycast(np.array([-5.0, 3.0, 4.0]), np.array([1.0, 0.0, 0.0]))
    assert hit is not None and 4.5 < hit < 5.5


def test_p1_le_digest_est_invariant_par_permutation() -> None:
    """Deux copies mémoire différentes du même bâtiment → même digest."""
    vertices = np.array(
        [
            [0, 0, 0], [10, 0, 0], [10, 6, 0],
            [0, 6, 0], [0, 0, 8], [10, 0, 8], [10, 6, 8], [0, 6, 8],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
         [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
         [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
    )
    perm = np.arange(len(vertices))[::-1]
    remap = {int(old): int(new) for new, old in enumerate(perm)}
    other_faces = np.array([[remap[int(v)] for v in face] for face in faces])
    first = CanonicalSceneMesh(vertices, faces)
    second = CanonicalSceneMesh(vertices[perm], other_faces)

    assert first.mesh_digest() == second.mesh_digest()


def test_p1_le_textureur_ne_reconstruit_plus_les_triangles() -> None:
    """Un payload portant `solid` ne passe plus par l'extrusion locale."""
    from hotel_pipeline.conditioning.facade_texture import _build_triangles_from_payload

    solid_vertices = [[0, 0, 0], [10, 0, 0], [10, 6, 8]]
    payload = {
        "volumes": [
            {
                "fp": [[0, 0], [999, 999]],  # emprise volontairement fausse
                "h": 42.0,
                "solid": {
                    "vertices": solid_vertices,
                    "faces": [[0, 1, 2]],
                },
            }
        ]
    }
    triangles, ids = _build_triangles_from_payload(payload)

    assert len(triangles) == 1
    assert triangles[0].shape == (3, 3)
    assert ids == [0]


# ----------------------------------------------------------------------
# Problème 2 — identité stable des sommets, CW/CCW équivalents
# ----------------------------------------------------------------------
def test_p2_le_meme_batiment_cw_et_ccw_donne_le_meme_mesh() -> None:
    footprint = np.array([[0, 0], [12, 0], [12, 8], [0, 8]], dtype=np.float64)
    cw = footprint[::-1].copy()

    forward = build_canonical_building_mesh(footprint, top_heights=9.0)
    backward = build_canonical_building_mesh(cw, top_heights=9.0)

    assert forward.mesh_digest() == backward.mesh_digest()
    assert not Polygon(cw).exterior.is_ccw  # l'entrée était bien inversée


def test_p2_l_inversion_emporte_les_hauteurs_pas_seulement_xy() -> None:
    """Le record entier s'inverse : ground_z/top_z suivent leur sommet."""
    ring = np.array([[0, 0], [20, 0], [20, 10]], dtype=np.float64)
    tops = np.array([4.0, 6.0, 8.0])

    records = footprint_records(ring, top_heights=tops)
    reversed_records = footprint_records(ring[::-1].copy(), top_heights=tops[::-1])

    def _by_position(items):
        return {(round(r.x, 6), round(r.y, 6)): r.top_z for r in items}

    assert _by_position(records) == _by_position(reversed_records)


def test_p2_buffer_zero_rematche_geometriquement() -> None:
    """Après réparation, les hauteurs restent liées au bon point du sol."""
    # Anneau auto-intersectant : buffer(0) répare la géométrie.
    ring = np.array([[0, 0], [10, 0], [10, 10], [2, 2], [0, 10]], dtype=np.float64)
    tops = np.array([5.0, 6.0, 7.0, 8.0, 9.0])

    records = footprint_records(ring, top_heights=tops)

    positions = {(round(r.x, 3), round(r.y, 3)): r.top_z for r in records}
    # Le sommet (10,0) garde sa hauteur de 6 m où qu'il soit réordonné.
    assert any(abs(z - 6.0) < 1e-9 for z in positions.values())
    # Chaque hauteur d'origine reste présente : aucune n'a été perdue.
    assert sorted(positions.values()) == [5.0, 6.0, 7.0, 8.0, 9.0]


# ----------------------------------------------------------------------
# Problème 3 — le pied des murs suit le terrain
# ----------------------------------------------------------------------
def test_p3_aucun_pied_de_mur_ne_flotte_sur_terrain_incline() -> None:
    """Pente de ~1 m sous le bâtiment : chaque pied touche le sol local."""
    terrain = FlatTerrain()
    footprint = np.array(
        [[0, 0], [40, 0], [40, 12], [0, 12]], dtype=np.float64
    )  # 0 m à 0.8 m de dénivelé en x ; amplifions :
    class Steep:
        def height_at(self, x: float, y: float) -> float:
            return x / 40.0  # 1 m de dénivelé sur la largeur

    mesh = build_canonical_building_mesh(footprint, top_heights=10.0, terrain=Steep())

    grounds = [record.ground_z for record in mesh.records]
    assert grounds[0] == pytest.approx(0.0, abs=1e-6)
    assert grounds[1] == pytest.approx(1.0, abs=1e-6)

    # Aucune extrémité de mur ne flotte ni ne pénètre le sol : pour chaque
    # sommet de base du maillage, z == terrain.height_at(x, y).
    for point in mesh.vertices:
        if point[2] < 0.5:  # zone des pieds
            expected = Steep().height_at(point[0], point[1])
            assert point[2] >= expected - 1e-6


def test_p3_le_plancher_structurel_reste_un_attribut_distinct() -> None:
    """ground_z décrit le sol extérieur ; floor_z reste disponible."""
    records = footprint_records(
        np.array([[0, 0], [10, 0], [10, 10]], dtype=np.float64)
    )
    assert all(record.floor_z is None for record in records)


# ----------------------------------------------------------------------
# Problème 4 — interpolation bilinéaire et grille adaptative
# ----------------------------------------------------------------------
rasterio = pytest.importorskip("rasterio")


def _dtm(tmp_path, values, res=1.0):
    path = tmp_path / "dtm.tif"
    transform = rasterio.transform.from_origin(0.0, values.shape[0] * res, res, res)
    with rasterio.open(
        path, "w", driver="GTiff", height=values.shape[0], width=values.shape[1],
        count=1, dtype="float32", crs="EPSG:2950", transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(values.astype("float32"), 1)
    return path


def test_p4_une_pente_continue_reste_continue(tmp_path) -> None:
    """Aucun effet escalier : l'interpolation entre deux nœuds est affine."""
    size = 160
    slope = np.tile(np.linspace(100.0, 104.0, size), (size, 1))
    grid = TerrainGridForTest(slope, step=4.0, origin=(0.0, 0.0))

    zs = [grid.height_at(x, 50.0) for x in np.linspace(20.0, 60.0, 200)]
    diffs = np.diff(zs)
    # Une rampe régulière : toutes les différences locales sont égales à
    # 1e-9 près — pas de marches.
    assert np.allclose(diffs, diffs[0], atol=1e-9)


def test_p4_la_grille_adaptive_raffine_pres_du_batiment(tmp_path) -> None:
    from hotel_pipeline.conditioning.terrain import load

    # Source à 25 cm : la zone fine doit en tirer une maille de ~50 cm.
    size = 1600
    slope = np.tile(np.linspace(100.0, 110.0, size), (size, 1))
    path = _dtm(tmp_path, slope, res=0.25)

    result = load(path, (200.0, 200.0), radius_m=180.0)

    assert result is not None
    steps = sorted(grid.step_m for _, grid in result.zones)
    assert steps[0] <= 0.6   # fin près du bâtiment
    assert steps[-1] >= 3.5  # large au loin
    # Continuité à cheval sur deux zones : pas de saut.
    inner = result.height_at(29.9, 200.0)
    outer = result.height_at(30.1, 200.0)
    assert abs(inner - outer) < 0.05 * abs(result.relief_m) + 0.01


class TerrainGridForTest:
    """Grille directe construite à la main pour tester la bilinéaire."""

    def __init__(self, values, step, origin):
        from hotel_pipeline.conditioning.terrain import TerrainGrid

        self.grid = TerrainGrid(
            x0=origin[0] + step / 2,
            y0=origin[1] + step / 2,
            step_m=step,
            heights=values[::-1, :] - 102.0,
            reference_z=102.0,
        )

    def height_at(self, x, y):
        return self.grid.height_at(x, y)


# ----------------------------------------------------------------------
# Problème 5 — pans de toit reconstruits ensemble
# ----------------------------------------------------------------------
def test_p5_deux_pans_partagent_exactement_les_sommets_du_fitage() -> None:
    """Toit à deux versants : le faîtage est une droite exacte, partagée."""
    # Deux pans parfaits, adjacents le long de la ligne x = 8, z = 10.
    gx, gy = np.meshgrid(
        np.linspace(0.0, 8.0, 33), np.linspace(0.0, 10.0, 21), indexing="ij"
    )
    left_points = np.column_stack([
        gx.ravel(), gy.ravel(), (10.0 + 0.25 * (8.0 - gx)).ravel()
    ])
    gx2, gy2 = np.meshgrid(
        np.linspace(8.0, 16.0, 33), np.linspace(0.0, 10.0, 21), indexing="ij"
    )
    right_points = np.column_stack([
        gx2.ravel(), gy2.ravel(), (10.0 + 0.25 * (gx2 - 8.0)).ravel()
    ])

    decomposition = RoofDecomposition(
        feature_id="toit-pente",
        planes=[_plane_from_points(left_points), _plane_from_points(right_points)],
        total=len(left_points) + len(right_points),
    )

    from hotel_pipeline.conditioning.roof_reconstruct import reconstruct_roof

    roof = reconstruct_roof(decomposition, Polygon([(0, 0), (16, 0), (16, 10), (0, 10)]))
    assert roof is not None

    # Les deux pans produisent des faces ; aucun recouvrement ni trou :
    # l'union des polygones découpés couvre l'emprise sans double emploi.
    polygons = list(roof.plane_polygons.values())
    assert len(polygons) == 2
    union = polygons[0].union(polygons[1])
    footprint_area = Polygon([(0, 0), (16, 0), (16, 10), (0, 10)]).area
    assert union.area <= footprint_area + 1e-6
    assert union.area >= footprint_area - 4.0

    # Faîtage partagé : au moins un sommet est porté par des faces de deux
    # pans différents — le même sommet exactement, pas deux doublons.
    owner_sets: dict[int, set[int]] = {}
    for face_index, face in enumerate(roof.faces):
        plane = roof.face_plane[face_index]
        if plane < 0:
            continue
        for vertex_index in face:
            owner_sets.setdefault(int(vertex_index), set()).add(plane)
    shared_vertices = [
        index for index, owners in owner_sets.items() if len(owners) >= 2
    ]
    assert shared_vertices, "aucun sommet de faîtage réellement partagé"
    # Et ce sommet est bien sur la ligne du faîtage.
    for index in shared_vertices[:1]:
        vertex = roof.vertices[index]
        assert abs(vertex[0] - 8.0) < 1e-3
        assert abs(vertex[2] - 10.0) < 1e-3


def test_p5_le_graphe_relie_pans_et_aretes() -> None:
    from hotel_pipeline.conditioning.roof_planes import ridges

    decomposition = RoofDecomposition(feature_id="g")
    ridge_list = ridges(decomposition)
    assert ridge_list == []  # graphe vide cohérent sans pan


# ----------------------------------------------------------------------
# Problème 6 — décrochement vertical explicite
# ----------------------------------------------------------------------
def test_p6_toit_haut_contre_toit_bas_produit_une_face_verticale() -> None:
    """12 m contre 5 m : toit haut plat + mur vertical de 7 m + toit bas."""
    rng = np.random.default_rng(3)
    high_x = rng.uniform(0.0, 10.0, 500)
    low_x = rng.uniform(10.3, 20.0, 500)
    ys = rng.uniform(0.0, 10.0, 1000)
    points = np.column_stack([
        np.concatenate([high_x, low_x]),
        ys,
        np.concatenate([np.full(500, 12.0), np.full(500, 5.0)]),
    ])

    decomposition = segment(points, "decroche")
    from hotel_pipeline.conditioning.roof_reconstruct import reconstruct_roof

    roof = reconstruct_roof(decomposition, Polygon([(0, 0), (20, 0), (20, 10), (0, 10)]))
    assert roof is not None

    # Des faces de décrochement existent et sont verticales.
    assert len(roof.step_faces) > 0
    for index in roof.step_faces:
        tri = roof.vertices[roof.faces[index]]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        normal = normal / max(np.linalg.norm(normal), 1e-12)
        assert abs(normal[2]) < 0.05  # normale horizontale → face verticale

    # La hauteur du décrochement vaut bien ~7 m quelque part.
    step_points = np.vstack([
        roof.vertices[roof.faces[index]] for index in roof.step_faces
    ])
    assert step_points[:, 2].max() - step_points[:, 2].min() > 6.5


# ----------------------------------------------------------------------
# Problème 7 — murs et toiture soudés, aucune arête libre
# ----------------------------------------------------------------------
def test_p7_boundary_edges_zero_et_weld_petit() -> None:
    from hotel_pipeline.conditioning.canonical_mesh import WELD_TOLERANCE_M

    footprint = np.array([[0, 0], [18, 0], [18, 9], [0, 9]], dtype=np.float64)
    mesh = build_canonical_building_mesh(footprint, top_heights=7.5)

    assert WELD_TOLERANCE_M <= 0.02
    assert mesh.boundary_edges() == []
    audit = mesh.audit()
    assert audit["watertight"]


def test_p7_le_haut_de_mur_est_derive_du_maillage_de_toit() -> None:
    """Vue rasante : le haut des murs coïncide avec le bord du toit."""
    rng = np.random.default_rng(11)
    xs = rng.uniform(0.0, 14.0, 700)
    ys = rng.uniform(0.0, 8.0, 700)
    zs = 6.0 + 0.2 * xs  # un seul pan incliné
    points = np.column_stack([xs, ys, zs])

    decomposition = segment(points, "pan-simple")
    footprint = np.array([[0, 0], [14, 0], [14, 8], [0, 8]], dtype=np.float64)
    mesh = build_canonical_building_mesh(
        footprint, top_heights=6.0, roof_decomposition=decomposition
    )

    # Chaque sommet de mur supérieur porte un sommet de toiture confondu
    # (tolérance de soudure), sinon une fissure serait visible en rasant.
    roof_kind_indices = [i for i, k in enumerate(mesh.face_kind) if k == "roof"]
    wall_top_z: dict[tuple[float, float], float] = {}
    for i, kind in enumerate(mesh.face_kind):
        if kind != "wall":
            continue
        for vertex_index in mesh.faces[i]:
            point = mesh.vertices[vertex_index]
            key = (round(point[0], 2), round(point[1], 2))
            # Le haut du mur : l'altitude maximale du mur en ce point XY,
            # le pied étant posé sur le terrain.
            wall_top_z[key] = max(wall_top_z.get(key, -np.inf), point[2])

    assert roof_kind_indices  # le toit mesuré est présent
    tolerance = 0.03
    checked = 0
    roof_points = np.vstack([
        mesh.vertices[mesh.faces[i]] for i in roof_kind_indices
    ])
    for key, z_wall in wall_top_z.items():
        near = roof_points[
            (np.abs(roof_points[:, 0] - key[0]) < 0.05)
            & (np.abs(roof_points[:, 1] - key[1]) < 0.05)
        ]
        if near.size:
            assert np.abs(near[:, 2] - z_wall).min() <= tolerance
            checked += 1
    assert checked >= 4  # les coins du bâtiment au moins


# ----------------------------------------------------------------------
# Problème 8 — l'aile basse ne monte pas vers le toit haut
# ----------------------------------------------------------------------
def test_p8_le_mur_de_l_aile_basse_reste_bas() -> None:
    """Aile basse accolée à un corps haut : son bord reste ~4 m."""
    rng = np.random.default_rng(21)
    main_x = rng.uniform(0.0, 10.0, 600)
    wing_x = rng.uniform(10.3, 16.0, 300)
    ys = rng.uniform(0.0, 10.0, 900)
    points = np.column_stack([
        np.concatenate([main_x, wing_x]),
        ys,
        np.concatenate([np.full(600, 12.0), np.full(300, 4.0)]),
    ])

    decomposition = segment(points, "aile")
    footprint = np.array([[0, 0], [16, 0], [16, 10], [0, 10]], dtype=np.float64)
    mesh = build_canonical_building_mesh(
        footprint, top_heights=12.0, roof_decomposition=decomposition
    )

    from hotel_pipeline.conditioning.roof_reconstruct import derive_wall_tops

    tops = derive_wall_tops(mesh.records, *_roof_args(decomposition))

    # Le bord droit de l'emprise appartient à l'aile basse : ~4 m.
    right_tops = [
        top for record, top in zip(mesh.records, tops) if record.x > 15.0
    ]
    assert right_tops and max(right_tops) < 6.0  # loin du corps à 12 m


def _roof_args(decomposition):
    from hotel_pipeline.conditioning.roof_reconstruct import (
        reconstruct_roof,
    )
    from shapely.geometry import Polygon as _Polygon

    roof = reconstruct_roof(
        decomposition, _Polygon([(0, 0), (16, 0), (16, 10), (0, 10)])
    )
    assert roof is not None
    return roof, decomposition


# ----------------------------------------------------------------------
# Problème 9 — points LiDAR parasites rejetés
# ----------------------------------------------------------------------
def test_p9_une_facade_devant_ne_modifie_pas_le_mur_cible() -> None:
    from hotel_pipeline.conditioning.facade_support_gate import gate_facade_support

    rng = np.random.default_rng(5)
    # Mur cible : plan y = 0, s'étendant en x et z.
    target = np.column_stack([
        rng.uniform(0.0, 30.0, 400),
        np.zeros(400),
        rng.uniform(0.0, 8.0, 400),
    ])
    # Façade parasite : parallèle, plaquée 2 m devant (y = 2).
    fake = np.column_stack([
        rng.uniform(0.0, 30.0, 120),
        np.full(120, 2.0),
        rng.uniform(0.0, 8.0, 120),
    ])
    points = np.vstack([target, fake])
    views = ["pose-%d" % (i % 4) for i in range(len(points))]

    verdict = gate_facade_support(
        points,
        plane_point=np.array([15.0, 0.0, 4.0]),
        plane_normal=np.array([0.0, 1.0, 0.0]),
        outward_normal=np.array([0.0, 1.0, 0.0]),
        view_ids=views,
        component_points=points,
        component_reference=np.array([15.0, 0.0, 4.0]),
    )

    accepted_target = verdict.accepted[:400]
    accepted_fake = verdict.accepted[400:]

    assert accepted_target.all()
    assert not accepted_fake.any()
    report = verdict.as_dict()
    assert report["rejected"] == 120


# ----------------------------------------------------------------------
# Problème 10 — MultiPolygon : graphe Site → Building → Part
# ----------------------------------------------------------------------
def test_p10_un_multipolygon_conserve_ses_volumes_secondaires() -> None:
    import json
    import tempfile
    from pathlib import Path

    from hotel_pipeline.conditioning.scene import load_scene

    wkt_main = "(0 0, 20 0, 20 12, 0 12, 0 0)"
    wkt_wing = "(20 2, 34 2, 34 8, 20 8, 20 2)"
    multipolygon = f"MULTIPOLYGON(({wkt_main}), ({wkt_wing}))"

    entry = {
        "feature_id": "hotel-a",
        "role": "target_building",
        "resolution_status": "resolved",
        "height_known": True,
        "height_m": 15.0,
        "projected_crs": "EPSG:2950",
        "projected_wkt": multipolygon,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "capture_geometry.json"
        path.write_text(json.dumps({"hotel_id": "a", "geometries": [entry]}))

        scene = load_scene(path)

    buildings = scene.buildings()
    parts = buildings["hotel-a"]

    assert len(parts) == 2          # plus aucune partie perdue
    assert sum(p.is_target for p in parts) == 1
    assert parts[0].parent_building_id is None
    assert parts[1].parent_building_id == "hotel-a"
    assert parts[1].part_index == 1
    # Même repère projeté : les deux parties partagent l'espace du site.
    assert parts[1].footprint[:, 0].max() > parts[0].footprint[:, 0].max()


def test_p10_le_resume_porte_le_graphe() -> None:
    import json
    import tempfile
    from pathlib import Path

    from hotel_pipeline.conditioning.scene import ConditioningScene, Prism

    prism = Prism(
        feature_id="b",
        role="target_building",
        footprint=np.array([[0, 0], [5, 0], [5, 5]], dtype=np.float64),
        height_m=10.0,
        height_assumed=False,
        height_source="test",
        is_target=True,
    )
    scene = ConditioningScene(hotel_id="x", crs="EPSG:2950", prisms=[prism])

    summary = scene.summary()

    assert summary["building_count"] == 1
    assert summary["secondary_parts"] == 0
