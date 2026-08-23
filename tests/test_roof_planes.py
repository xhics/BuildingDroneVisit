"""Une toiture est faite de pans, et leurs rencontres forment des arêtes."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from hotel_pipeline.conditioning.roof_planes import (  # noqa: E402
    MIN_PLANE_POINTS,
    RIDGE_MIN_ANGLE_DEG,
    RoofDecomposition,
    RoofPlane,
    ridges,
    segment,
)


def _slope(x0, x1, y0, y1, base, pitch, count=1200, seed=0):
    """Nuage d'un versant : altitude croissant avec x."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(x0, x1, count)
    ys = rng.uniform(y0, y1, count)
    zs = base + (xs - x0) * pitch + rng.normal(0, 0.03, count)
    return np.c_[xs, ys, zs]


def _flat(x0, x1, y0, y1, height, count=1200, seed=1):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(x0, x1, count)
    ys = rng.uniform(y0, y1, count)
    return np.c_[xs, ys, np.full(count, height) + rng.normal(0, 0.03, count)]


# --- segmentation -----------------------------------------------------------


def test_un_toit_plat_donne_un_seul_pan() -> None:
    decomposition = segment(_flat(0, 20, 0, 20, 10.0), "plat")

    assert len(decomposition.planes) >= 1
    assert decomposition.planes[0].slope_deg < 5.0
    assert decomposition.explained > 0.7


def test_deux_versants_donnent_deux_pans() -> None:
    """C'est le cas mesuré : le pilote a un toit à deux pentes."""
    montant = _slope(0, 10, 0, 20, 8.0, 0.4, seed=2)
    descendant = _slope(10, 20, 0, 20, 12.0, -0.4, seed=3)

    decomposition = segment(np.vstack([montant, descendant]), "deux_pans")
    pentus = decomposition.pitched

    assert len(pentus) >= 2
    for plane in pentus[:2]:
        assert 15.0 < plane.slope_deg < 30.0


def test_la_pente_d_un_pan_est_mesuree() -> None:
    # Une pente de 0,5 en x correspond à environ 26,6 degrés.
    decomposition = segment(_slope(0, 20, 0, 20, 8.0, 0.5), "pente")

    assert decomposition.planes
    assert decomposition.planes[0].slope_deg == pytest.approx(26.6, abs=3.0)


def test_un_nuage_trop_maigre_ne_produit_aucun_pan() -> None:
    points = _flat(0, 2, 0, 2, 10.0, count=MIN_PLANE_POINTS // 2)

    assert segment(points, "maigre").planes == []


def test_les_points_inexpliques_sont_comptes() -> None:
    """Superstructures et bordures ne sont pas une erreur, mais un reste."""
    decomposition = segment(_flat(0, 20, 0, 20, 10.0), "plat")

    assert 0.0 <= decomposition.explained <= 1.0
    assert decomposition.unassigned >= 0


def test_l_altitude_d_un_pan_s_evalue_hors_de_ses_points() -> None:
    plane = RoofPlane(
        points=_slope(0, 10, 0, 10, 8.0, 0.5),
        normal=np.array([-0.447, 0.0, 0.894]),
        origin=np.array([5.0, 5.0, 10.5]),
    )

    montant = plane.height_at(9.0, 5.0)
    descendant = plane.height_at(1.0, 5.0)

    assert montant > descendant


# --- arêtes -----------------------------------------------------------------


def _plane(normal, origin, points):
    return RoofPlane(
        points=points,
        normal=np.asarray(normal, dtype=float),
        origin=np.asarray(origin, dtype=float),
    )


def test_deux_pans_opposes_forment_un_faitage() -> None:
    """L'arête est déduite de l'intersection, non cherchée dans le nuage."""
    montant = _slope(0, 10, 0, 20, 8.0, 0.4, seed=4)
    descendant = _slope(10, 20, 0, 20, 12.0, -0.4, seed=5)
    decomposition = RoofDecomposition(
        feature_id="t",
        planes=[
            _plane([-0.371, 0.0, 0.928], [5.0, 10.0, 10.0], montant),
            _plane([0.371, 0.0, 0.928], [15.0, 10.0, 10.0], descendant),
        ],
        total=len(montant) + len(descendant),
    )

    found = ridges(decomposition)

    assert found
    faitages = [r for r in found if r.kind == "faitage"]
    assert faitages
    # Le faîtage est horizontal : ses deux extrémités sont à la même altitude.
    assert faitages[0].start[2] == pytest.approx(faitages[0].end[2], abs=0.3)


def test_deux_pans_paralleles_ne_forment_pas_d_arete() -> None:
    a = _flat(0, 10, 0, 10, 10.0, seed=6)
    b = _flat(12, 22, 0, 10, 10.0, seed=7)
    decomposition = RoofDecomposition(
        feature_id="t",
        planes=[
            _plane([0.0, 0.0, 1.0], [5.0, 5.0, 10.0], a),
            _plane([0.0, 0.0, 1.0], [17.0, 5.0, 10.0], b),
        ],
    )

    assert ridges(decomposition) == []


def test_des_pans_eloignes_ne_sont_pas_adjacents() -> None:
    """Deux versants de bâtiments différents ne partagent pas d'arête."""
    a = _slope(0, 10, 0, 10, 8.0, 0.4, seed=8)
    b = _slope(100, 110, 0, 10, 8.0, -0.4, seed=9)
    decomposition = RoofDecomposition(
        feature_id="t",
        planes=[
            _plane([-0.371, 0.0, 0.928], [5.0, 5.0, 10.0], a),
            _plane([0.371, 0.0, 0.928], [105.0, 5.0, 10.0], b),
        ],
    )

    assert ridges(decomposition) == []


def test_l_angle_minimal_ecarte_les_pans_quasi_confondus() -> None:
    from hotel_pipeline.conditioning.roof_planes import RIDGE_MIN_ANGLE_DEG as seuil

    assert seuil >= 10.0


def test_le_rapport_resume_la_decomposition() -> None:
    decomposition = segment(_flat(0, 20, 0, 20, 10.0), "plat")
    payload = decomposition.as_dict()

    assert payload["planes"] >= 1
    joined = " ".join(payload["caveats"])
    assert "vectorisée" in joined
    assert "inexpliqués" in joined


# --- application au maillage ------------------------------------------------


def test_le_maillage_est_replace_sur_les_pans() -> None:
    """Un sommet bruité doit rejoindre le plan qui le couvre."""
    from hotel_pipeline.conditioning.roof_planes import apply_to_roof

    class _Prism:
        feature_id = "t"

    prism = _Prism()
    points = _slope(0, 20, 0, 20, 8.0, 0.5, seed=10)
    prism.roof_vertices = np.array([[10.0, 10.0, 13.5]])  # 0,5 m trop haut
    decomposition = RoofDecomposition(
        feature_id="t",
        planes=[_plane([-0.447, 0.0, 0.894], [10.0, 10.0, 13.0], points)],
    )

    adjusted = apply_to_roof(prism, decomposition)

    assert adjusted == 1
    assert prism.roof_vertices[0, 2] == pytest.approx(13.0, abs=0.4)


def test_un_sommet_hors_de_tout_pan_reste_intact() -> None:
    """La segmentation corrige ce qu'elle explique, jamais le reste."""
    from hotel_pipeline.conditioning.roof_planes import apply_to_roof

    class _Prism:
        feature_id = "t"

    prism = _Prism()
    prism.roof_vertices = np.array([[500.0, 500.0, 9.0]])
    decomposition = RoofDecomposition(
        feature_id="t",
        planes=[
            _plane([0.0, 0.0, 1.0], [5.0, 5.0, 10.0], _flat(0, 10, 0, 10, 10.0))
        ],
    )

    apply_to_roof(prism, decomposition)

    assert prism.roof_vertices[0, 2] == pytest.approx(9.0)
