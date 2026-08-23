"""Les bordures de sol se décrivent en contours, non en damier de cellules."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("skimage")

from hotel_pipeline.conditioning.ground_polygons import (  # noqa: E402
    MIN_PATCH_AREA_M2,
    GroundPatch,
    GroundPatches,
    from_cells,
)
from hotel_pipeline.conditioning.surface_lidar import SurfaceCell  # noqa: E402


def _cells(rows: list[str], cell_m: float = 1.0) -> list[SurfaceCell]:
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


def _block(size: int = 20, inset: int = 5) -> list[str]:
    """Un carré de gazon au milieu d'une surface minérale."""
    rows = []
    for row in range(size):
        line = "".join(
            "v" if inset <= row < size - inset and inset <= col < size - inset else "m"
            for col in range(size)
        )
        rows.append(line)
    return rows


def test_une_plage_devient_un_polygone() -> None:
    patches = from_cells(_cells(_block()), 1.0)

    vegetal = [p for p in patches.patches if p.kind == "vegetal"]
    assert vegetal
    assert len(vegetal[0].ring) >= 4


def test_le_contour_compte_moins_de_sommets_que_de_cellules() -> None:
    """C'est tout l'intérêt : décrire une bordure sans la pixelliser."""
    cells = _cells(_block(size=30, inset=8))
    patches = from_cells(cells, 1.0)

    vertices = sum(len(p.ring) for p in patches.patches)
    assert vertices < len(cells) / 5


def test_le_contour_est_ferme() -> None:
    patches = from_cells(_cells(_block()), 1.0)

    for patch in patches.patches:
        assert patch.ring[0] == patch.ring[-1]


def test_l_aire_reste_du_bon_ordre() -> None:
    """Le lissage arrondit la bordure, il ne redimensionne pas la plage."""
    patches = from_cells(_cells(_block(size=40, inset=10)), 1.0)

    vegetal = [p for p in patches.patches if p.kind == "vegetal"]
    assert vegetal
    # Le carré intérieur fait 20 × 20 cellules d'un mètre.
    assert 250.0 < vegetal[0].area_m2() < 600.0


def test_une_plage_minuscule_est_ecartee() -> None:
    """Quelques mètres carrés isolés relèvent du bruit du capteur."""
    rows = ["m" * 20 for _ in range(20)]
    rows[10] = "m" * 9 + "vv" + "m" * 9
    patches = from_cells(_cells(rows), 1.0)

    assert [p for p in patches.patches if p.kind == "vegetal"] == []


def test_sans_cellule_aucun_polygone() -> None:
    assert from_cells([], 1.0).patches == []


def test_une_grille_trop_petite_ne_produit_rien() -> None:
    assert from_cells(_cells(["vm", "mv"]), 1.0).patches == []


def test_les_deux_natures_sont_traitees_separement() -> None:
    patches = from_cells(_cells(_block(size=24, inset=6)), 1.0)
    kinds = {p.kind for p in patches.patches}

    assert "vegetal" in kinds
    assert "mineral" in kinds


def test_l_aire_par_nature_est_cumulee() -> None:
    patch = GroundPatch(
        kind="vegetal",
        ring=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)],
    )
    totals = GroundPatches(hotel_id="t", patches=[patch]).by_kind()

    assert totals["vegetal"] == pytest.approx(100.0)


def test_le_rapport_porte_ses_reserves() -> None:
    joined = " ".join(GroundPatches(hotel_id="t").as_dict()["caveats"])

    assert "lisse" in joined
    assert "bruit" in joined


def test_le_rendu_triangule_une_plage() -> None:
    from hotel_pipeline.conditioning.render import _patch_faces

    patch = GroundPatch(
        kind="vegetal",
        ring=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)],
    )
    faces = _patch_faces(patch)

    assert len(faces) == 4
    for triangle in faces:
        assert triangle.shape == (3, 3)
        # Le sol est plat, au niveau du terrain.
        assert np.allclose(triangle[:, 2], 0.0)


def test_un_contour_degenere_ne_produit_aucune_face() -> None:
    from hotel_pipeline.conditioning.render import _patch_faces

    assert _patch_faces(GroundPatch(kind="vegetal", ring=[(0.0, 0.0)])) == []


def test_une_cellule_indecise_ne_troue_pas_la_surface() -> None:
    """La chaussée continue sous le doute : un trou se voit comme un défaut.

    Mesuré sur le pilote : quatre pour cent du sol restait indéterminé, en
    petits groupes dispersés — autant de taches noires au milieu d'une route.
    """
    # Une plage minérale bordée de végétal, percée d'un doute en son centre :
    # sans bordure, aucun contour n'existe et le cas ne prouverait rien.
    size, inset = 40, 10
    rows = []
    for row in range(size):
        line = "".join(
            "m" if inset <= row < size - inset and inset <= col < size - inset else "v"
            for col in range(size)
        )
        rows.append(line)
    middle = size // 2
    rows[middle] = rows[middle][:middle] + "??" + rows[middle][middle + 2 :]

    patches = from_cells(_cells(rows), 1.0)
    mineral = [p for p in patches.patches if p.kind == "mineral"]

    assert mineral
    # La plage englobe le doute : elle couvre l'essentiel du carré intérieur,
    # trou compris, au lieu de s'y interrompre.
    assert mineral[0].area_m2() > 250.0


def test_le_comblement_n_efface_pas_la_classification() -> None:
    """Le doute est comblé pour le rendu, jamais dans la mesure."""
    from hotel_pipeline.conditioning.ground_polygons import _fill_undecided

    rows = ["mmmmm", "mm?mm", "mmmmm"]
    cells = _cells(rows)

    filled = _fill_undecided(cells)

    assert filled == 1
    undecided = [c for c in cells if c.kind == "indetermine"]
    assert len(undecided) == 1
    # La nature d'origine reste lisible, le comblement vit à côté.
    assert undecided[0].filled_kind == "mineral"


def test_un_doute_sans_voisin_reste_indecis() -> None:
    from hotel_pipeline.conditioning.ground_polygons import _fill_undecided

    cells = _cells(["???", "???", "???"])
    _fill_undecided(cells)

    assert all(c.filled_kind is None for c in cells)


def test_un_sol_uniforme_est_pose_sans_nature() -> None:
    """Un relevé sans contraste laissait le sol disparaître entièrement.

    Le seuil d'intensité se dérive du relevé : sur un terrain de réflectance
    homogène, il tombe au milieu d'une population unique et rien ne le
    franchit. Poser le sol sans nature vaut mieux que ne rien poser — un
    terrain existe même quand on ignore son revêtement.
    """
    rows = ["?" * 20 for _ in range(20)]
    patches = from_cells(_cells(rows), 1.0, kinds=("vegetal", "mineral"))

    assert patches.patches
    assert all(p.kind == "indetermine_pose" for p in patches.patches)


def test_un_sol_contraste_garde_ses_natures() -> None:
    """Le repli ne doit pas effacer une classification qui marche."""
    patches = from_cells(_cells(_block(size=24, inset=6)), 1.0)
    kinds = {p.kind for p in patches.patches}

    assert "indetermine_pose" not in kinds
    assert kinds <= {"vegetal", "mineral"}
