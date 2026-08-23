"""Le sol se déduit du retour LiDAR, il ne se suppose pas."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hotel_pipeline.conditioning.surface_lidar import (
    INTENSITY_MARGIN,
    SurfaceCell,
    SurfaceMap,
    classify_ground,
)


def _tile(tmp_path: Path, blocks: list[tuple[float, float, int]]) -> Path:
    """Tuile de sol : chaque bloc porte sa propre intensité."""
    laspy = pytest.importorskip("laspy")

    rng = np.random.default_rng(0)
    xs, ys, intensity = [], [], []
    for cx, cy, value in blocks:
        count = 600
        xs.append(rng.uniform(cx - 5, cx + 5, count))
        ys.append(rng.uniform(cy - 5, cy + 5, count))
        intensity.append(np.full(count, value))

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = [0, 0, 0]
    header.scales = [0.01, 0.01, 0.01]
    px = np.concatenate(xs)
    py = np.concatenate(ys)
    las = laspy.LasData(header)
    las.x = px
    las.y = py
    las.z = np.full(px.size, 100.0)
    las.intensity = np.concatenate(intensity).astype(np.uint16)
    las.classification = np.full(px.size, 2, dtype=np.uint8)

    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "ground.laz"
    las.write(path)
    return path


def test_deux_surfaces_de_reflectance_distincte_sont_separees(tmp_path: Path) -> None:
    """L'asphalte renvoie moins que le gazon : c'est ce qui les distingue."""
    # Trois niveaux : le seuil est pris à la médiane du relevé, donc deux
    # blocs seulement placeraient la coupure sur l'un d'eux et rien ne
    # dépasserait par le haut.
    path = _tile(
        tmp_path,
        [(0.0, 0.0, 25000), (40.0, 0.0, 40000), (80.0, 0.0, 55000)],
    )

    surface = classify_ground(path, (40.0, 0.0), radius_m=90.0)
    kinds = surface.by_kind()

    assert kinds.get("mineral", 0) > 0
    assert kinds.get("vegetal", 0) > 0


def test_une_surface_uniforme_ne_se_scinde_pas_en_deux(tmp_path: Path) -> None:
    """Sans contraste, il n'y a rien à trancher : tout reste indéterminé."""
    path = _tile(tmp_path, [(0.0, 0.0, 40000), (30.0, 0.0, 40000)])

    surface = classify_ground(path, (15.0, 0.0), radius_m=60.0)

    assert surface.by_kind().get("indetermine", 0) > 0
    assert surface.by_kind().get("vegetal", 0) == 0


def test_le_seuil_se_derive_du_releve(tmp_path: Path) -> None:
    """Une valeur absolue échouerait sur la tuile suivante."""
    low = classify_ground(
        _tile(tmp_path / "a", [(0.0, 0.0, 20000), (40.0, 0.0, 30000)]),
        (20.0, 0.0),
        radius_m=60.0,
    )
    high = classify_ground(
        _tile(tmp_path / "b", [(0.0, 0.0, 50000), (40.0, 0.0, 60000)]),
        (20.0, 0.0),
        radius_m=60.0,
    )

    assert low.threshold < high.threshold
    assert low.by_kind().get("mineral", 0) > 0
    assert high.by_kind().get("mineral", 0) > 0


def test_le_sol_sous_un_batiment_est_ecarte(tmp_path: Path) -> None:
    """Ce qui est couvert par une emprise bâtie ne décrit pas le sol du site."""
    path = _tile(tmp_path, [(0.0, 0.0, 30000), (40.0, 0.0, 50000)])
    footprint = np.array([[-6.0, -6.0], [6.0, -6.0], [6.0, 6.0], [-6.0, 6.0]])

    without = classify_ground(path, (20.0, 0.0), radius_m=60.0)
    with_mask = classify_ground(
        path, (20.0, 0.0), radius_m=60.0, footprints=[footprint]
    )

    assert len(with_mask.cells) < len(without.cells)


def test_une_tuile_absente_ne_produit_aucune_carte(tmp_path: Path) -> None:
    assert classify_ground(tmp_path / "rien.laz", (0.0, 0.0)).cells == []


def test_les_aires_suivent_la_taille_de_cellule() -> None:
    cells = [SurfaceCell(0.0, 0.0, "vegetal", 5.0, None, 4)]
    surface = SurfaceMap(hotel_id="t", cells=cells, cell_m=2.0)

    assert surface.area_by_kind()["vegetal"] == pytest.approx(4.0)


def test_le_rapport_porte_ses_reserves() -> None:
    joined = " ".join(SurfaceMap(hotel_id="t").as_dict()["caveats"])

    assert "intensité" in joined
    assert "indetermine" in joined
    assert "hivernal" in joined


# --- continuité spatiale ----------------------------------------------------


def _cells(rows: list[str], cell_m: float = 4.0) -> list[SurfaceCell]:
    """Construit une grille depuis un damier textuel : v=végétal, m=minéral."""
    kinds = {"v": "vegetal", "m": "mineral", "?": "indetermine"}
    out = []
    for row, line in enumerate(rows):
        for col, char in enumerate(line):
            out.append(
                SurfaceCell(
                    x=(col + 0.5) * cell_m,
                    y=(len(rows) - 1 - row + 0.5) * cell_m,
                    kind=kinds[char],
                    intensity=40000.0,
                    greenness=None,
                    points=8,
                )
            )
    return out


def test_une_cellule_isolee_est_absorbee() -> None:
    """Un point de gazon au milieu d'un stationnement relève du bruit."""
    from hotel_pipeline.conditioning.surface_lidar import _smooth

    cells = _cells([
        "mmmmm",
        "mmmmm",
        "mmvmm",
        "mmmmm",
        "mmmmm",
    ])
    changed = _smooth(cells, 4.0)

    assert changed > 0
    assert all(c.kind == "mineral" for c in cells)


def test_une_frontiere_franche_survit_au_lissage() -> None:
    """Le bord d'une allée est une vraie limite, pas du bruit."""
    from hotel_pipeline.conditioning.surface_lidar import _smooth

    cells = _cells([
        "mmmvvv",
        "mmmvvv",
        "mmmvvv",
        "mmmvvv",
        "mmmvvv",
        "mmmvvv",
    ])
    _smooth(cells, 4.0)

    kinds = {c.kind for c in cells}
    assert kinds == {"mineral", "vegetal"}
    # Chaque moitié garde l'essentiel de ses cellules.
    assert sum(1 for c in cells if c.kind == "vegetal") >= 12


def test_une_plage_trop_petite_est_absorbee() -> None:
    """Deux cellules perdues ne décrivent pas une pelouse."""
    from hotel_pipeline.conditioning.surface_lidar import _absorb_small_patches

    cells = _cells([
        "mmmmmm",
        "mmmmmm",
        "mmvvmm",
        "mmmmmm",
        "mmmmmm",
    ])
    absorbed = _absorb_small_patches(cells, 4.0)

    assert absorbed == 2
    assert all(c.kind == "mineral" for c in cells)


def test_une_grande_plage_est_conservee() -> None:
    from hotel_pipeline.conditioning.surface_lidar import _absorb_small_patches

    cells = _cells([
        "mmmmmm",
        "mvvvvm",
        "mvvvvm",
        "mvvvvm",
        "mmmmmm",
    ])
    absorbed = _absorb_small_patches(cells, 4.0)

    assert absorbed == 0
    assert sum(1 for c in cells if c.kind == "vegetal") == 12


def test_l_indetermine_ne_gagne_jamais_un_vote() -> None:
    """Un doute marque une absence de nature, il n'en impose aucune."""
    from hotel_pipeline.conditioning.surface_lidar import _smooth

    cells = _cells([
        "?????",
        "?????",
        "??v??",
        "?????",
        "?????",
    ])
    _smooth(cells, 4.0)

    centre = [c for c in cells if c.kind == "vegetal"]
    assert len(centre) == 1


def test_le_lissage_rend_les_plages_continues() -> None:
    """Mesuré sur le pilote : 60 % des plages ne faisaient qu'une ou deux cellules."""
    from hotel_pipeline.conditioning.surface_lidar import (
        _absorb_small_patches,
        _smooth,
    )

    noisy = _cells([
        "mvmmvm",
        "mmvmmm",
        "vmmmvm",
        "mmvmmm",
        "mvmmmv",
        "mmmvmm",
    ])
    _smooth(noisy, 4.0)
    _absorb_small_patches(noisy, 4.0)

    assert sum(1 for c in noisy if c.kind == "vegetal") == 0
