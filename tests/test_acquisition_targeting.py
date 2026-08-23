"""On ne paie que pour les vues susceptibles de montrer le sujet."""

from __future__ import annotations

import math

import pytest

from hotel_pipeline.acquisition_targeting import (
    MIN_SUBTENDED_DEG,
    NEIGHBOUR_SUCCESS_RATIO,
    Vantage,
    evaluate,
    neighbourhood,
    select,
    subtended_angle,
)

#: Bâtiment de 70 × 40 m centré sur l'origine.
FOOTPRINT = [(-35.0, -20.0), (35.0, -20.0), (35.0, 20.0), (-35.0, 20.0)]
CENTROID = (0.0, 0.0)


def test_angle_sous_tendu_decroit_avec_la_distance():
    proche = subtended_angle((0.0, -50.0), FOOTPRINT)
    loin = subtended_angle((0.0, -300.0), FOOTPRINT)

    assert proche > loin
    assert loin > 0


def test_angle_sous_tendu_depend_de_la_face_regardee():
    """L'arrière d'un bâtiment allongé se voit de plus loin que son pignon."""
    long_cote = subtended_angle((0.0, -100.0), FOOTPRINT)
    pignon = subtended_angle((100.0, 0.0), FOOTPRINT)

    assert long_cote > pignon


def test_empreinte_vide_ne_sous_tend_rien():
    assert subtended_angle((0.0, -50.0), []) == 0.0


def test_voisinage_sans_mesure_est_inconnu_non_defavorable():
    ratio, count = neighbourhood((0.0, -60.0), [])

    assert ratio is None
    assert count == 0


def test_voisinage_favorable_quand_les_voisins_reussissent():
    measured = [((5.0, -60.0), 0.9), ((-5.0, -62.0), 0.8), ((0.0, -58.0), 0.1)]

    ratio, count = neighbourhood((0.0, -60.0), measured)

    assert count == 3
    assert ratio == pytest.approx(2 / 3)
    assert ratio >= NEIGHBOUR_SUCCESS_RATIO


def test_voisins_hors_rayon_sont_ignores():
    measured = [((0.0, -60.0), 0.9), ((0.0, 300.0), 0.9)]

    _ratio, count = neighbourhood((0.0, -60.0), measured, radius_m=50.0)

    assert count == 1


def test_vue_proche_avec_bons_voisins_est_recommandee():
    measured = [((5.0, -50.0), 0.95), ((-5.0, -52.0), 0.90)]

    vantages = evaluate([("p", (0.0, -50.0))], FOOTPRINT, CENTROID, measured)

    assert vantages[0].tier == "recommended"
    assert vantages[0].geometry_favourable
    assert vantages[0].neighbourhood_favourable is True


def test_vue_lointaine_avec_mauvais_voisins_est_peu_prometteuse():
    measured = [((5.0, -300.0), 0.02), ((-5.0, -302.0), 0.01)]

    vantages = evaluate([("p", (0.0, -300.0))], FOOTPRINT, CENTROID, measured)

    assert vantages[0].tier == "unpromising"


def test_voisinage_inconnu_ne_disqualifie_pas():
    """L'absence de mesure n'est pas une mauvaise mesure."""
    vantages = evaluate([("p", (0.0, -300.0))], FOOTPRINT, CENTROID, [])

    assert vantages[0].tier == "plausible"
    assert vantages[0].neighbourhood_favourable is None


def test_le_tri_place_le_plus_prometteur_en_tete():
    measured = [((5.0, -50.0), 0.95), ((-5.0, -52.0), 0.90)]
    candidates = [("loin", (0.0, -300.0)), ("proche", (0.0, -50.0))]

    vantages = evaluate(candidates, FOOTPRINT, CENTROID, measured)

    assert vantages[0].panorama_id == "proche"


def test_le_voisinage_pese_plus_que_la_geometrie():
    """Il porte ce que la géométrie ne peut pas voir : occlusions réelles."""
    proche_mauvais_voisins = Vantage(
        "a", (0.0, -50.0), subtended_deg=60.0, distance_m=50.0,
        neighbour_ratio=0.0, neighbours=4,
    )
    loin_bons_voisins = Vantage(
        "b", (0.0, -200.0), subtended_deg=20.0, distance_m=200.0,
        neighbour_ratio=1.0, neighbours=4,
    )

    assert loin_bons_voisins.priority > proche_mauvais_voisins.priority


def test_rien_n_est_definitivement_rejete():
    """Un panorama écarté reste candidat, plus bas dans la file."""
    measured = [((5.0, -300.0), 0.0), ((-5.0, -302.0), 0.0)]
    vantages = evaluate([("p", (0.0, -300.0))], FOOTPRINT, CENTROID, measured)

    assert vantages[0].tier == "unpromising"
    assert select(vantages, budget=1) == vantages, "le budget doit pouvoir y aller"


def test_select_respecte_le_budget():
    candidates = [(f"p{i}", (float(i * 10), -50.0)) for i in range(10)]

    vantages = evaluate(candidates, FOOTPRINT, CENTROID, [])

    assert len(select(vantages, budget=3)) == 3


def test_select_sert_d_abord_les_recommandes():
    measured = [((0.0, -50.0), 0.95), ((2.0, -50.0), 0.95)]
    candidates = [("loin", (0.0, -400.0)), ("proche", (1.0, -50.0))]

    vantages = evaluate(candidates, FOOTPRINT, CENTROID, measured)

    assert select(vantages, budget=1)[0].panorama_id == "proche"


def test_seuil_geometrique_est_celui_mesure():
    """45,1° médians pour les exploitables, 26,9° pour les perdues."""
    assert 26.9 < MIN_SUBTENDED_DEG < 45.1


def test_pose_non_attestee_retrograde_sans_exclure():
    """Une vue superbe à une position fausse vaut moins — mais reste candidate.

    Mesuré ailleurs dans le pipeline : une dérive de 40 m suffit à projeter les
    cellules de pelouse sur le stationnement.
    """
    measured = [((5.0, -50.0), 0.95), ((-5.0, -52.0), 0.90)]

    attestee = evaluate(
        [("p", (0.0, -50.0))], FOOTPRINT, CENTROID, measured,
        attested={"p": True},
    )[0]
    douteuse = evaluate(
        [("p", (0.0, -50.0))], FOOTPRINT, CENTROID, measured,
        attested={"p": False},
    )[0]

    assert douteuse.priority < attestee.priority
    assert douteuse.tier != "unpromising", "rétrograder n'est pas exclure"
    assert any("non attestée" in r for r in douteuse.reasons)


def test_provenance_inconnue_ne_penalise_pas():
    measured = [((5.0, -50.0), 0.95), ((-5.0, -52.0), 0.90)]

    sans = evaluate([("p", (0.0, -50.0))], FOOTPRINT, CENTROID, measured)[0]
    avec = evaluate(
        [("p", (0.0, -50.0))], FOOTPRINT, CENTROID, measured, attested={},
    )[0]

    assert sans.priority == avec.priority
    assert sans.pose_attested is None
