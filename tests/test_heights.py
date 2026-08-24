"""Une hauteur mesurée doit remplacer l'hypothèse — et seulement là où on mesure."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from hotel_pipeline.conditioning.heights import (  # noqa: E402
    MIN_CELLS,
    RasterUnavailable,
    apply_measured_heights,
    build_roof_surface,
    measure_footprint,
)
from hotel_pipeline.conditioning.scene import ConditioningScene, Prism  # noqa: E402


def _square(half: float, cx: float = 50.0, cy: float = 50.0) -> np.ndarray:
    return np.array(
        [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ],
        dtype=np.float64,
    )


def _raster(tmp_path: Path, values: np.ndarray, nodata: float = -9999.0) -> Path:
    path = tmp_path / "ndsm.tif"
    transform = rasterio.transform.from_origin(0.0, 100.0, 0.5, 0.5)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:2950",
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(values.astype("float32"), 1)
    return path


def _flat(height: float = 9.0, size: int = 200) -> np.ndarray:
    return np.full((size, size), height, dtype=np.float64)


def _prism(footprint: np.ndarray, target: bool = True) -> Prism:
    return Prism(
        feature_id="TARGET_BUILDING" if target else "OBST",
        role="target_building" if target else "obstacle_building",
        footprint=footprint,
        height_m=12.0,
        height_assumed=True,
        height_source="hypothèse",
        is_target=target,
    )


def _scene(prisms: list[Prism]) -> ConditioningScene:
    return ConditioningScene(
        hotel_id="t", crs="EPSG:2950", prisms=prisms, centre=(50.0, 50.0)
    )


def test_une_hauteur_mesuree_remplace_l_hypothese(tmp_path: Path) -> None:
    path = _raster(tmp_path, _flat(9.0))
    scene = _scene([_prism(_square(10.0))])

    summary = apply_measured_heights(scene, path)

    prism = scene.target
    assert prism is not None
    assert prism.height_assumed is False
    assert prism.height_m == pytest.approx(9.0, abs=0.2)
    assert "nDSM" in prism.height_source
    assert summary["measured"] == 1
    assert summary["still_assumed"] == 0


def test_un_volume_hors_couverture_reste_une_hypothese(tmp_path: Path) -> None:
    """Une tuile partielle ne doit pas transformer en mesure ce qu'elle n'a pas vu."""
    path = _raster(tmp_path, _flat(9.0))
    covered = _prism(_square(10.0))
    far_away = _prism(_square(10.0, cx=5000.0, cy=5000.0), target=False)
    scene = _scene([covered, far_away])

    summary = apply_measured_heights(scene, path)

    assert covered.height_assumed is False
    assert far_away.height_assumed is True
    assert far_away.height_m == 12.0
    assert summary["measured"] == 1
    assert summary["still_assumed"] == 1


def test_la_mesure_ecarte_les_superstructures(tmp_path: Path) -> None:
    """Le p90 rend le corps du bâtiment, pas la cheminée qui le dépasse."""
    values = _flat(9.0)
    values[90:96, 90:96] = 25.0  # édicule très haut, faible emprise
    path = _raster(tmp_path, values)
    scene = _scene([_prism(_square(12.0))])

    apply_measured_heights(scene, path)

    assert scene.target.height_m < 12.0


def test_une_couverture_trop_faible_ne_produit_pas_de_mesure(tmp_path: Path) -> None:
    values = np.full((200, 200), -9999.0)
    values[100:103, 100:103] = 9.0  # quelques cellules isolées
    path = _raster(tmp_path, values)
    prism = _prism(_square(15.0))
    scene = _scene([prism])

    summary = apply_measured_heights(scene, path)

    assert prism.height_assumed is True
    assert summary["measured"] == 0


def test_un_crs_different_arrete_la_mesure(tmp_path: Path) -> None:
    """Aucune reprojection en silence : un désaccord de référentiel arrête tout."""
    path = _raster(tmp_path, _flat(9.0))
    scene = _scene([_prism(_square(10.0))])
    scene.crs = "EPSG:32188"

    with pytest.raises(RasterUnavailable, match="EPSG"):
        apply_measured_heights(scene, path)


def test_un_ndsm_absent_est_signale(tmp_path: Path) -> None:
    scene = _scene([_prism(_square(10.0))])
    with pytest.raises(RasterUnavailable, match="absent"):
        apply_measured_heights(scene, tmp_path / "rien.tif")


def test_le_toit_mesure_remplace_la_fermeture_inventee(tmp_path: Path) -> None:
    path = _raster(tmp_path, _flat(9.0))
    prism = _prism(_square(12.0))
    scene = _scene([prism])

    apply_measured_heights(scene, path)

    assert prism.roof_measured is True
    assert prism.roof_vertices.shape[1] == 3
    assert len(prism.roof_faces) > 0
    # Le crédit du toit rejoint celui des murs dès qu'il est attesté.
    assert prism.roof_confidence == pytest.approx(prism.confidence)


def test_un_toit_non_mesure_reste_declasse() -> None:
    prism = _prism(_square(10.0))
    assert prism.roof_measured is False
    assert prism.roof_confidence < prism.confidence


def test_les_trous_du_raster_sont_combles(tmp_path: Path) -> None:
    """Trois pour cent de cellules manquantes ne doivent pas trouer le toit."""
    values = _flat(9.0)
    rng = np.random.default_rng(0)
    holes = rng.random(values.shape) < 0.03
    values[holes] = -9999.0
    path = _raster(tmp_path, values)
    prism = _prism(_square(12.0))

    with rasterio.open(path) as raster:
        surface = build_roof_surface(raster, prism.footprint)

    assert surface is not None
    vertices, faces = surface
    assert len(faces) > 0
    # Aucun sommet ne doit porter la valeur de nodata.
    assert vertices[:, 2].min() > 0


def test_le_toit_suit_le_relief_reel(tmp_path: Path) -> None:
    """Deux ailes d'altitudes différentes doivent rester distinctes."""
    values = _flat(8.0)
    values[:, 100:] = 12.0
    path = _raster(tmp_path, values)
    prism = _prism(_square(15.0))

    with rasterio.open(path) as raster:
        vertices, _ = build_roof_surface(raster, prism.footprint)

    assert vertices[:, 2].min() == pytest.approx(8.0, abs=0.3)
    assert vertices[:, 2].max() == pytest.approx(12.0, abs=0.3)


def test_le_contour_du_toit_rejoint_exactement_l_emprise_vectorielle(
    tmp_path: Path,
) -> None:
    """Le bord ne doit plus suivre les centres de pixels en escalier."""
    from shapely.geometry import Point, Polygon

    footprint = np.array(
        [[38.2, 41.3], [62.7, 43.1], [59.4, 61.8], [40.1, 58.9]],
        dtype=np.float64,
    )
    path = _raster(tmp_path, _flat(9.0))

    with rasterio.open(path) as raster:
        vertices, faces = build_roof_surface(raster, footprint)

    counts: Counter[tuple[int, int]] = Counter()
    for face in faces:
        counts.update(
            tuple(sorted((int(face[index]), int(face[(index + 1) % 3]))))
            for index in range(3)
        )
    boundary_vertices = {
        index
        for edge, owners in counts.items()
        if owners == 1
        for index in edge
    }
    outline = Polygon(footprint).boundary
    assert boundary_vertices
    assert max(
        outline.distance(Point(vertices[index, :2]))
        for index in boundary_vertices
    ) < 1e-7


def test_la_mesure_porte_sa_provenance(tmp_path: Path) -> None:
    path = _raster(tmp_path, _flat(9.0))
    prism = _prism(_square(10.0))

    with rasterio.open(path) as raster:
        found = measure_footprint(raster, prism.footprint, "X")

    assert found is not None
    assert found.cells >= MIN_CELLS
    assert found.source == "ndsm_lidar"
    assert 0.0 < found.coverage <= 1.0
    assert "height_m" in found.as_dict()


# --- mesure directe dans le nuage -------------------------------------------


def test_le_nuage_complete_ce_que_le_raster_ne_couvre_pas(tmp_path: Path) -> None:
    """Le nDSM pilote ne couvre que la cible ; la tuile source a les voisins."""
    laspy = pytest.importorskip("laspy")
    import numpy as np

    from hotel_pipeline.conditioning.heights import apply_laz_heights

    # Deux emprises : sol à 100 m, toits à 112 m et 108 m.
    rng = np.random.default_rng(0)
    xs, ys, zs, cls = [], [], [], []
    for cx, roof in ((50.0, 112.0), (200.0, 108.0)):
        px = rng.uniform(cx - 9, cx + 9, 400)
        py = rng.uniform(41.0, 59.0, 400)
        xs.append(px); ys.append(py); zs.append(np.full(400, roof)); cls.append(np.full(400, 6))
        gx = rng.uniform(cx - 9, cx + 9, 100)
        gy = rng.uniform(41.0, 59.0, 100)
        xs.append(gx); ys.append(gy); zs.append(np.full(100, 100.0)); cls.append(np.full(100, 2))

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = [0, 0, 0]
    header.scales = [0.01, 0.01, 0.01]
    las = laspy.LasData(header)
    las.x = np.concatenate(xs)
    las.y = np.concatenate(ys)
    las.z = np.concatenate(zs)
    las.classification = np.concatenate(cls).astype(np.uint8)
    path = tmp_path / "tile.laz"
    las.write(path)

    target = _prism(_square(9.0, cx=50.0, cy=50.0))
    neighbour = _prism(_square(9.0, cx=200.0, cy=50.0), target=False)
    scene = _scene([target, neighbour])

    summary = apply_laz_heights(scene, path)

    assert summary["measured"] == 2
    assert target.height_assumed is False
    assert neighbour.height_assumed is False
    assert target.height_m == pytest.approx(12.0, abs=0.5)
    assert neighbour.height_m == pytest.approx(8.0, abs=0.5)
    assert "nuage LiDAR" in neighbour.height_source


def test_le_nuage_ne_touche_pas_aux_hauteurs_deja_mesurees(tmp_path: Path) -> None:
    """Le nDSM reste prioritaire : il est qualifié et porte une surface de toit."""
    from hotel_pipeline.conditioning.heights import apply_laz_heights

    prism = _prism(_square(10.0))
    prism.height_m = 27.5
    prism.height_assumed = False
    prism.height_source = "nDSM LiDAR"
    scene = _scene([prism])

    summary = apply_laz_heights(scene, tmp_path / "absente.laz")

    assert summary["measured"] == 0
    assert prism.height_m == 27.5
    assert prism.height_source == "nDSM LiDAR"


def test_une_hauteur_non_mesuree_dit_pourquoi(tmp_path: Path) -> None:
    """Distinguer « hors de la tuile » de « trop peu de points » oriente l'action."""
    laspy = pytest.importorskip("laspy")
    import numpy as np

    from hotel_pipeline.conditioning.heights import apply_laz_heights

    # Les points débordent franchement l'emprise testée : les bornes du fichier
    # doivent la contenir, sans quoi le volume serait lui-même « hors tuile ».
    rng = np.random.default_rng(1)
    px = rng.uniform(30.0, 70.0, 400)
    py = rng.uniform(30.0, 70.0, 400)

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = [0, 0, 0]
    header.scales = [0.01, 0.01, 0.01]
    las = laspy.LasData(header)
    las.x = px
    las.y = py
    las.z = np.full(400, 110.0)
    las.classification = np.full(400, 6, dtype=np.uint8)
    path = tmp_path / "tile.laz"
    las.write(path)

    inside = _prism(_square(9.0, cx=50.0, cy=50.0))
    outside = _prism(_square(9.0, cx=5000.0, cy=5000.0), target=False)
    scene = _scene([inside, outside])

    summary = apply_laz_heights(scene, path)

    assert summary["outside_tile"] == 1
    assert "hors de la tuile" in outside.height_source
    assert "geo discover --scene" in outside.height_source
