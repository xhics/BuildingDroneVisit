"""Le relief des murs se lit dans le nuage, non dans la vue de dessus."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hotel_pipeline.conditioning.facade import (
    EdgeProfile,
    FacadeRelief,
    read_relief,
)


class _Prism:
    def __init__(self, footprint: np.ndarray) -> None:
        self.feature_id = "TARGET"
        self.footprint = footprint
        self.roof_faces = []
        self.facade_relief = None

    @property
    def roof_measured(self) -> bool:
        return False


def _tile(tmp_path: Path, points) -> Path:
    """Tuile de bâti : (x, y, z) en classe 6."""
    laspy = pytest.importorskip("laspy")

    xs, ys, zs = zip(*points)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = [0, 0, 0]
    header.scales = [0.01, 0.01, 0.01]
    las = laspy.LasData(header)
    las.x = np.array(xs, dtype=float)
    las.y = np.array(ys, dtype=float)
    las.z = np.array(zs, dtype=float)
    las.classification = np.full(len(xs), 6, dtype=np.uint8)

    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "tile.laz"
    las.write(path)
    return path


def _wall(y: float, x0: float, x1: float, height, count: int = 900):
    """Retours le long d'un mur, dont la hauteur peut varier avec x."""
    rng = np.random.default_rng(0)
    xs = rng.uniform(x0, x1, count)
    return [
        (float(x), float(y + rng.normal(0, 0.4)), float(height(x)))
        for x in xs
    ]


# --- lecture du relief ------------------------------------------------------


def test_un_mur_de_hauteur_constante_ne_montre_aucun_relief(tmp_path: Path) -> None:
    footprint = np.array([[0.0, 0.0], [40.0, 0.0], [40.0, 20.0], [0.0, 20.0]])
    path = _tile(tmp_path, _wall(0.0, 0.0, 40.0, lambda x: 110.0))

    relief = read_relief(path, _Prism(footprint), ground_z=100.0)

    assert relief.profiles
    assert relief.max_relief_m < 0.6


def test_un_pignon_ressort_du_profil(tmp_path: Path) -> None:
    """C'est le cas mesuré : un pignon fait varier le mur de trois mètres."""
    footprint = np.array([[0.0, 0.0], [40.0, 0.0], [40.0, 20.0], [0.0, 20.0]])
    # Le mur monte de trois mètres au milieu de l'arête.
    path = _tile(
        tmp_path,
        _wall(0.0, 0.0, 40.0, lambda x: 110.0 + (3.0 if 17 < x < 23 else 0.0)),
    )

    relief = read_relief(path, _Prism(footprint), ground_z=100.0)

    assert relief.max_relief_m > 2.0


def test_une_arete_trop_courte_est_ignoree(tmp_path: Path) -> None:
    footprint = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
    path = _tile(tmp_path, _wall(0.0, 0.0, 4.0, lambda x: 110.0))

    assert read_relief(path, _Prism(footprint), ground_z=100.0).profiles == {}


def test_une_tuile_absente_ne_produit_aucun_profil(tmp_path: Path) -> None:
    footprint = np.array([[0.0, 0.0], [40.0, 0.0], [40.0, 20.0], [0.0, 20.0]])

    relief = read_relief(tmp_path / "rien.laz", _Prism(footprint), ground_z=100.0)

    assert relief.profiles == {}
    assert relief.max_relief_m == 0.0


def test_la_hauteur_est_rapportee_au_sol(tmp_path: Path) -> None:
    footprint = np.array([[0.0, 0.0], [40.0, 0.0], [40.0, 20.0], [0.0, 20.0]])
    path = _tile(tmp_path, _wall(0.0, 0.0, 40.0, lambda x: 112.0))

    relief = read_relief(path, _Prism(footprint), ground_z=100.0)
    profile = next(iter(relief.profiles.values()))
    heights = profile.heights[np.isfinite(profile.heights)]

    assert heights.mean() == pytest.approx(12.0, abs=0.6)


# --- interpolation ----------------------------------------------------------


def test_la_hauteur_s_interpole_le_long_de_l_arete() -> None:
    relief = FacadeRelief(
        feature_id="t",
        profiles={0: EdgeProfile(0, np.array([10.0, 12.0, 14.0]), 100)},
    )

    assert relief.height_along(0, 0.0) == pytest.approx(10.0)
    assert relief.height_along(0, 0.5) == pytest.approx(12.0)
    assert relief.height_along(0, 1.0) == pytest.approx(14.0)


def test_un_segment_sans_retour_emprunte_a_son_voisin() -> None:
    """Un trou dans le relevé ne doit pas creuser le mur."""
    relief = FacadeRelief(
        feature_id="t",
        profiles={0: EdgeProfile(0, np.array([10.0, np.nan, 14.0]), 50)},
    )

    milieu = relief.height_along(0, 0.5)
    assert milieu is not None
    assert 9.0 <= milieu <= 15.0


def test_une_arete_non_profilee_ne_rend_rien() -> None:
    assert FacadeRelief(feature_id="t").height_along(3, 0.5) is None


def test_le_rapport_resume_le_relief() -> None:
    relief = FacadeRelief(
        feature_id="t",
        profiles={0: EdgeProfile(0, np.array([10.0, 13.0]), 200)},
    )
    payload = relief.as_dict()

    assert payload["edges_profiled"] == 1
    assert payload["max_relief_m"] == pytest.approx(3.0)


# --- rendu ------------------------------------------------------------------


def test_le_mur_rendu_suit_le_relief_releve() -> None:
    """Sans relief, le mur était dressé entre deux hauteurs interpolées."""
    from hotel_pipeline.conditioning.render import _edge_heights

    class _Bare:
        roof_measured = False
        roof_vertices = None

    relief = FacadeRelief(
        feature_id="t",
        profiles={0: EdgeProfile(0, np.array([10.0, 13.0, 10.0]), 300)},
    )

    plat = _edge_heights(_Bare(), np.zeros(2), np.array([40.0, 0.0]), 10.0, 10.0, 4)
    profile = _edge_heights(
        _Bare(), np.zeros(2), np.array([40.0, 0.0]), 10.0, 10.0, 4, 0, relief
    )

    assert plat.max() == pytest.approx(10.0)
    assert profile.max() > 12.0
