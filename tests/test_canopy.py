"""Un poteau n'est pas un arbre, et une rangée d'arbres n'est pas un bloc."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("skimage")

from hotel_pipeline.conditioning.canopy import (  # noqa: E402
    POLE_MAX_FOOTPRINT_M2,
    CanopyObject,
    _classify,
    segment,
)


def _crown(cx: float, cy: float, radius: float, height: float, count: int = 400):
    """Nuage en forme de houppier : dense, étalé, culminant au centre."""
    rng = np.random.default_rng(int(cx * 7 + cy))
    angle = rng.uniform(0, 2 * np.pi, count)
    spread = radius * np.sqrt(rng.uniform(0, 1, count))
    xs = cx + spread * np.cos(angle)
    ys = cy + spread * np.sin(angle)
    zs = height * (1.0 - 0.5 * (spread / radius)) * rng.uniform(0.75, 1.0, count)
    return xs, ys, zs


def _pole(cx: float, cy: float, height: float, count: int = 90):
    """Nuage en colonne : étroit, continu du sol au sommet."""
    rng = np.random.default_rng(int(cx * 13 + cy))
    xs = cx + rng.normal(0, 0.12, count)
    ys = cy + rng.normal(0, 0.12, count)
    zs = rng.uniform(0.2, height, count)
    return xs, ys, zs


def _merge(*clouds):
    xs = np.concatenate([c[0] for c in clouds])
    ys = np.concatenate([c[1] for c in clouds])
    zs = np.concatenate([c[2] for c in clouds])
    return xs, ys, zs


# --- classification ---------------------------------------------------------


def test_un_objet_fin_et_haut_est_du_mobilier() -> None:
    assert _classify(footprint_m2=1.0, height_m=6.0, spread_ratio=0.1) == "poteau"


def test_un_objet_fin_mais_etale_en_haut_reste_un_arbre() -> None:
    """Une couronne s'élargit vers le sommet, un mât non."""
    assert _classify(footprint_m2=4.0, height_m=6.0, spread_ratio=0.8) == "couronne"


def test_un_objet_bas_est_un_buisson() -> None:
    assert _classify(footprint_m2=3.0, height_m=1.2, spread_ratio=0.5) == "buisson"


def test_un_objet_large_est_une_couronne() -> None:
    assert _classify(footprint_m2=40.0, height_m=9.0, spread_ratio=0.4) == "couronne"


# --- segmentation -----------------------------------------------------------


def test_deux_arbres_voisins_ne_fusionnent_pas() -> None:
    """La connexité seule reliait une rangée entière en un bloc unique."""
    xs, ys, zs = _merge(_crown(0, 0, 4.0, 9.0), _crown(11, 0, 4.0, 9.0))

    crowns = [o for o in segment(xs, ys, zs) if o.kind == "couronne"]

    assert len(crowns) == 2
    for crown in crowns:
        assert crown.radius_m < 8.0


def test_une_couronne_a_un_rayon_d_arbre() -> None:
    xs, ys, zs = _crown(0, 0, 4.0, 10.0, count=800)

    crowns = [o for o in segment(xs, ys, zs) if o.kind == "couronne"]

    assert crowns
    assert 1.0 < crowns[0].radius_m < 7.0
    assert 7.0 < crowns[0].height_m < 11.0


def test_un_lampadaire_n_est_pas_rendu_comme_vegetation() -> None:
    """Quatorze pour cent des amas du pilote étaient du mobilier."""
    xs, ys, zs = _merge(_crown(0, 0, 4.0, 9.0), _pole(14.0, 0.0, 6.0))

    objects = segment(xs, ys, zs)
    poles = [o for o in objects if o.kind == "poteau"]

    assert len(poles) == 1
    assert poles[0].footprint_m2 <= POLE_MAX_FOOTPRINT_M2
    assert poles[0].height_m > 4.0


def test_le_mobilier_est_retire_du_nuage_vegetal() -> None:
    """Laissé dedans, un mât est absorbé par le bassin d'une couronne voisine."""
    xs, ys, zs = _merge(_crown(0, 0, 4.0, 9.0), _pole(6.0, 0.0, 7.0))

    objects = segment(xs, ys, zs)

    assert any(o.kind == "poteau" for o in objects)
    # Le mât ne doit pas gonfler la couronne voisine.
    crowns = [o for o in objects if o.kind == "couronne"]
    assert all(c.radius_m < 8.0 for c in crowns)


def test_une_branche_suspendue_n_est_pas_un_poteau() -> None:
    """Un mât part du sol ; une branche isolée n'a de retours qu'en altitude."""
    rng = np.random.default_rng(5)
    xs = rng.normal(0, 0.15, 60)
    ys = rng.normal(0, 0.15, 60)
    zs = rng.uniform(6.0, 7.0, 60)  # rien au sol

    assert [o for o in segment(xs, ys, zs) if o.kind == "poteau"] == []


def test_un_nuage_vide_ne_produit_aucun_objet() -> None:
    empty = np.array([])
    assert segment(empty, empty, empty) == []


def test_chaque_objet_porte_sa_mesure() -> None:
    xs, ys, zs = _crown(0, 0, 4.0, 9.0)
    objects = segment(xs, ys, zs)

    assert objects
    payload = objects[0].as_dict()
    assert set(payload) == {
        "kind",
        "centre",
        "radius_m",
        "height_m",
        "footprint_m2",
        "points",
        "envelope",
    }
    assert payload["points"] > 0
    assert len(payload["envelope"]) >= 4
    assert all(len(ring) >= 8 for ring in payload["envelope"])


def test_enveloppe_conserve_une_couronne_asymetrique() -> None:
    """Le viewer doit recevoir une forme mesurée, pas un rayon à extruder."""
    rng = np.random.default_rng(17)
    count = 900
    angle = rng.uniform(0, 2 * np.pi, count)
    radius = np.where(np.cos(angle) > 0, 5.0, 2.2) * np.sqrt(
        rng.uniform(0, 1, count)
    )
    xs = radius * np.cos(angle)
    ys = radius * np.sin(angle)
    zs = 9.0 * (1.0 - 0.35 * radius / 5.0) * rng.uniform(0.75, 1.0, count)

    crown = next(o for o in segment(xs, ys, zs) if o.kind == "couronne")
    middle = crown.envelope[len(crown.envelope) // 2]
    east = max(point[0] for point in middle) - crown.centre[0]
    west = crown.centre[0] - min(point[0] for point in middle)

    assert east > west * 1.2


def test_un_arbuste_n_est_pas_plus_large_qu_un_grand_arbre() -> None:
    """Le rayon vient de l'aire du bassin et ignorait la hauteur.

    Mesuré sur le pilote : les arbustes ressortaient à 1,26 de rapport
    rayon/hauteur contre 0,46 pour les arbres matures — un buisson paraissait
    donc plus massif à l'écran qu'un grand arbre voisin.
    """
    from hotel_pipeline.conditioning.canopy import MAX_RADIUS_RATIO

    rng = np.random.default_rng(11)
    # Masse très étalée mais peu haute : le rayon tiré de l'aire du bassin
    # dépasserait la hauteur si rien ne le plafonnait. La hauteur reste
    # au-dessus du seuil de détection d'un sommet, sinon rien n'est segmenté.
    count = 1200
    xs = rng.uniform(-9.0, 9.0, count)
    ys = rng.uniform(-9.0, 9.0, count)
    zs = rng.uniform(2.6, 3.2, count)

    objects = segment(xs, ys, zs)

    assert objects
    for item in objects:
        limit = MAX_RADIUS_RATIO.get(item.kind, 0.55) * item.height_m
        assert item.radius_m <= max(limit, 0.5) + 1e-6


def test_le_plafond_de_rayon_depend_de_la_nature() -> None:
    """Un arbuste reste ramassé, une couronne s'étale davantage."""
    from hotel_pipeline.conditioning.canopy import MAX_RADIUS_RATIO

    assert MAX_RADIUS_RATIO["buisson"] > MAX_RADIUS_RATIO["couronne"]
    assert MAX_RADIUS_RATIO["couronne"] > MAX_RADIUS_RATIO["poteau"]
