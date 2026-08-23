"""L'environnement borne un encombrement mesuré, il n'invente pas un jardin."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hotel_pipeline.conditioning.environment import (
    LINKED_BUILDING_M,
    MAX_CLUSTER_SPAN_M,
    SiteEnvironment,
    VegetationPatch,
    extract_vegetation,
    find_linked_buildings,
    _cluster_cells,
)


def _square(half: float, cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    return np.array(
        [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ],
        dtype=np.float64,
    )


class _Prism:
    def __init__(self, feature_id: str, footprint: np.ndarray, target: bool) -> None:
        self.feature_id = feature_id
        self.footprint = footprint
        self.is_target = target


class _Scene:
    def __init__(self, prisms: list[_Prism]) -> None:
        self.prisms = prisms
        self.hotel_id = "t"
        self.centre = (0.0, 0.0)

    @property
    def target(self):
        return next((p for p in self.prisms if p.is_target), None)


# --- amas -------------------------------------------------------------------


def test_un_amas_isole_devient_un_massif() -> None:
    rng = np.random.default_rng(0)
    xs = rng.uniform(0.0, 6.0, 200)
    ys = rng.uniform(0.0, 6.0, 200)
    zs = rng.uniform(4.0, 5.0, 200)

    clusters = _cluster_cells(xs, ys, zs, cell=2.0)

    assert len(clusters) == 1
    (centre, radius, height, count) = clusters[0]
    assert centre[0] == pytest.approx(3.0, abs=1.0)
    assert 0 < radius < 8.0
    assert 4.0 <= height <= 5.1
    assert count == 200


def test_le_rayon_ne_deborde_pas_la_surface_occupee() -> None:
    """Un cercle circonscrit sur une haie couvrirait deux fois trop de terrain."""
    rng = np.random.default_rng(1)
    xs = rng.uniform(0.0, 20.0, 600)
    ys = rng.uniform(0.0, 3.0, 600)  # bande étroite et longue
    zs = np.full(600, 3.0)

    clusters = _cluster_cells(xs, ys, zs, cell=2.0)
    radii = [c[1] for c in clusters]
    circumscribed = float(np.hypot(20.0, 3.0) / 2)

    assert max(radii) < circumscribed


def test_un_amas_trop_etendu_est_redecoupe() -> None:
    """La connexité seule relierait un boisé entier en un unique volume."""
    rng = np.random.default_rng(2)
    span = MAX_CLUSTER_SPAN_M * 4
    xs = rng.uniform(0.0, span, 2000)
    ys = rng.uniform(0.0, 6.0, 2000)
    zs = np.full(2000, 8.0)

    clusters = _cluster_cells(xs, ys, zs, cell=2.0)

    assert len(clusters) > 1
    for _, radius, _, _ in clusters:
        assert radius < MAX_CLUSTER_SPAN_M


def test_quelques_points_epars_ne_font_pas_un_massif() -> None:
    xs = np.array([0.0, 40.0, 80.0])
    ys = np.array([0.0, 40.0, 80.0])
    zs = np.array([3.0, 3.0, 3.0])

    assert _cluster_cells(xs, ys, zs, cell=2.0) == []


# --- bâtiments liés ---------------------------------------------------------


def test_une_annexe_jointive_est_liee_au_site() -> None:
    target = _Prism("TARGET_BUILDING", _square(10.0), True)
    annexe = _Prism("ANNEXE", _square(5.0, cx=15.0), False)

    linked = find_linked_buildings(_Scene([target, annexe]))

    assert len(linked) == 1
    assert linked[0].shares_parcel is True
    assert "aile ou annexe" in linked[0].reason


def test_un_immeuble_lointain_reste_un_voisin() -> None:
    target = _Prism("TARGET_BUILDING", _square(10.0), True)
    far = _Prism("VOISIN", _square(10.0, cx=400.0), False)

    assert find_linked_buildings(_Scene([target, far])) == []


def test_le_seuil_de_liaison_est_respecte() -> None:
    target = _Prism("TARGET_BUILDING", _square(10.0), True)
    just_inside = _Prism("PROCHE", _square(5.0, cx=10.0 + LINKED_BUILDING_M - 1 + 5), False)
    just_outside = _Prism("LOIN", _square(5.0, cx=10.0 + LINKED_BUILDING_M + 10 + 5), False)

    linked = find_linked_buildings(_Scene([target, just_inside, just_outside]))

    assert [b.feature_id for b in linked] == ["PROCHE"]


def test_sans_cible_aucun_batiment_n_est_lie() -> None:
    assert find_linked_buildings(_Scene([_Prism("A", _square(5.0), False)])) == []


# --- extraction -------------------------------------------------------------


def test_une_tuile_absente_ne_produit_aucune_vegetation(tmp_path: Path) -> None:
    patches, ground, furniture = extract_vegetation(tmp_path / "rien.laz", (0.0, 0.0))
    assert patches == []
    assert ground is None
    assert furniture == []


def test_la_vegetation_se_deduit_de_la_hauteur(tmp_path: Path) -> None:
    """Cette tuile ne porte aucune classe végétation : la hauteur en tient lieu."""
    laspy = pytest.importorskip("laspy")

    rng = np.random.default_rng(3)
    # Sol à 100 m, plus un bouquet d'arbres à 106 m, en classe « non classé ».
    gx = rng.uniform(-40.0, 40.0, 800)
    gy = rng.uniform(-40.0, 40.0, 800)
    tx = rng.uniform(8.0, 16.0, 400)
    ty = rng.uniform(8.0, 16.0, 400)

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = [0, 0, 0]
    header.scales = [0.01, 0.01, 0.01]
    las = laspy.LasData(header)
    las.x = np.concatenate([gx, tx])
    las.y = np.concatenate([gy, ty])
    las.z = np.concatenate([np.full(800, 100.0), np.full(400, 106.0)])
    las.classification = np.concatenate(
        [np.full(800, 2), np.full(400, 1)]
    ).astype(np.uint8)
    path = tmp_path / "tile.laz"
    las.write(path)

    patches, ground, furniture = extract_vegetation(path, (0.0, 0.0), radius_m=60.0)

    assert ground == pytest.approx(100.0, abs=0.5)
    assert patches
    # La strate suit la hauteur mesurée au-dessus du sol.
    assert all(p.stratum in {"arbres_matures", "petits_arbres"} for p in patches)
    assert patches[0].height_m == pytest.approx(6.0, abs=1.0)
    # Un bouquet étalé n'est pas du mobilier.
    assert furniture == []


def test_le_rapport_porte_ses_reserves() -> None:
    payload = SiteEnvironment(
        hotel_id="h",
        patches=[VegetationPatch("arbustes", (0.0, 0.0), 2.0, 1.5, 40)],
    ).as_dict()

    assert payload["vegetation_count"] == 1
    assert payload["by_stratum"] == {"arbustes": 1}
    joined = " ".join(payload["caveats"])
    assert "aucune classe végétation" in joined
    assert "saisonnière" in joined


def test_les_superstructures_de_toiture_ne_sont_pas_de_la_vegetation(
    tmp_path: Path,
) -> None:
    """Cheminées, édicules et unités de ventilation se dressent sur un toit.

    Mesuré sur le pilote : vingt et un objets tombaient dans l'emprise du
    bâtiment cible et sortaient en « arbres matures » jusqu'à quatorze mètres.
    """
    laspy = pytest.importorskip("laspy")

    rng = np.random.default_rng(7)
    # Une cheminée au centre du bâtiment, un arbre à l'écart.
    chimney = 300
    tree = 400
    cx, cy = 0.0, 0.0
    xs = np.concatenate([
        rng.uniform(cx - 1.0, cx + 1.0, chimney),
        rng.uniform(40.0, 48.0, tree),
    ])
    ys = np.concatenate([
        rng.uniform(cy - 1.0, cy + 1.0, chimney),
        rng.uniform(-4.0, 4.0, tree),
    ])
    ground = 400
    gx = rng.uniform(-60.0, 60.0, ground)
    gy = rng.uniform(-60.0, 60.0, ground)

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = [0, 0, 0]
    header.scales = [0.01, 0.01, 0.01]
    las = laspy.LasData(header)
    las.x = np.concatenate([xs, gx])
    las.y = np.concatenate([ys, gy])
    las.z = np.concatenate([
        np.full(chimney, 114.0),
        np.full(tree, 108.0),
        np.full(ground, 100.0),
    ])
    las.classification = np.concatenate([
        np.full(chimney + tree, 1),
        np.full(ground, 2),
    ]).astype(np.uint8)
    path = tmp_path / "tile.laz"
    las.write(path)

    footprint = np.array([[-8.0, -8.0], [8.0, -8.0], [8.0, 8.0], [-8.0, 8.0]])

    patches, _, furniture = extract_vegetation(
        path, (0.0, 0.0), radius_m=80.0, footprints=[footprint]
    )

    on_roof = [
        p for p in patches + furniture if abs(p.centre[0]) < 8 and abs(p.centre[1]) < 8
    ]
    assert on_roof == []
