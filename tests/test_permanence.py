"""La permanence borne ce qu'on produit — et l'ignorance ne promeut jamais."""

from __future__ import annotations

import pytest

from hotel_pipeline.permanence import (
    MIN_DATES_FOR_VARIANCE,
    SEASONAL_ABOVE,
    STABLE_BELOW,
    Permanence,
    Production,
    classify_kind,
    infer_from_variance,
    resolve,
    summarise,
)


def test_batiment_autorise_une_geometrie():
    obj = resolve("b1", kind="building")

    assert obj.permanence is Permanence.PERMANENT
    assert obj.production is Production.MESH_3D
    assert not obj.season_dependent


def test_arbre_n_autorise_qu_un_volume_approche():
    obj = resolve("t1", kind="tree")

    assert obj.permanence is Permanence.SEASONAL_STRUCTURE
    assert obj.production is Production.APPROXIMATE_VOLUME
    assert obj.season_dependent


def test_plate_bande_n_autorise_qu_une_surface():
    obj = resolve("f1", kind="flowerbed")

    assert obj.production is Production.SURFACE_2D


def test_fleurs_ne_produisent_aucune_geometrie():
    obj = resolve("p1", kind="flowers")

    assert obj.permanence is Permanence.EPHEMERAL
    assert obj.production is Production.APPEARANCE_ONLY


def test_objet_inconnu_reste_ephemere():
    """L'ignorance penche du côté qui n'invente rien."""
    obj = resolve("x1", kind="quelque chose d'inconnu")

    assert obj.permanence is Permanence.EPHEMERAL
    assert obj.production is Production.APPEARANCE_ONLY
    assert any("éphémère" in e for e in obj.evidence)


def test_objet_sans_type_ni_variance_reste_ephemere():
    obj = resolve("x2")

    assert obj.permanence is Permanence.EPHEMERAL


def test_classify_kind_est_insensible_a_la_casse():
    assert classify_kind("  Tree ") is Permanence.SEASONAL_STRUCTURE
    assert classify_kind(None) is Permanence.EPHEMERAL


# --- Consensus multi-dates --------------------------------------------------

def test_apparence_stable_sur_plusieurs_dates_vaut_permanence():
    permanence, why = infer_from_variance(0.03, 4)

    assert permanence is Permanence.PERMANENT
    assert "stable" in why


def test_apparence_variable_vaut_saisonnier():
    permanence, why = infer_from_variance(0.60, 4)

    assert permanence is Permanence.SEASONAL_SURFACE
    assert "variable" in why


def test_trop_peu_de_dates_ne_tranche_pas():
    """Deux photos ne disent rien de la stabilité dans le temps."""
    permanence, why = infer_from_variance(0.02, 2)

    assert permanence is None
    assert str(MIN_DATES_FOR_VARIANCE) in why


def test_zone_grise_ne_tranche_pas():
    milieu = (STABLE_BELOW + SEASONAL_ABOVE) / 2

    permanence, why = infer_from_variance(milieu, 5)

    assert permanence is None
    assert "ni stable" in why


def test_variance_non_mesuree_ne_tranche_pas():
    assert infer_from_variance(None, 9)[0] is None


def test_la_variance_ne_promeut_pas_un_massif_en_batiment():
    """Une apparence stable ne suffit pas à autoriser un maillage.

    Le type déclaré prime : il vient d'une source, la variance d'une inférence.
    """
    obj = resolve("f2", kind="flowerbed", variance=0.01,
                  dates={"2024-06", "2024-07", "2024-08"})

    assert obj.permanence is Permanence.SEASONAL_SURFACE
    assert obj.production is Production.SURFACE_2D


def test_une_variance_forte_retrograde_un_objet_declare_permanent():
    """Un mur couvert de vigne n'est pas rendu par un maillage sans saison."""
    obj = resolve("w1", kind="wall", variance=SEASONAL_ABOVE + 0.3,
                  dates={"2024-01", "2024-06", "2024-10"})

    assert obj.permanence is Permanence.SEASONAL_STRUCTURE
    assert any("rétrogradé" in e for e in obj.evidence)


def test_retrogradation_exige_assez_de_dates():
    """Deux photos ne suffisent pas non plus à rétrograder."""
    obj = resolve("w2", kind="wall", variance=0.9, dates={"2024-01", "2024-06"})

    assert obj.permanence is Permanence.PERMANENT


def test_objet_saisonnier_vu_dans_une_seule_saison_est_signale():
    obj = resolve("t2", kind="tree", seasons={"summer"})

    assert obj.seasons_missing


def test_objet_saisonnier_vu_dans_deux_saisons_est_complet():
    obj = resolve("t3", kind="tree", seasons={"summer", "winter"})

    assert not obj.seasons_missing


def test_objet_permanent_n_a_pas_besoin_de_saisons():
    obj = resolve("b2", kind="building", seasons={"summer"})

    assert not obj.seasons_missing


def test_summarise_compte_ce_qui_est_modelisable():
    out = summarise([
        resolve("b", kind="building"),
        resolve("t", kind="tree", seasons={"summer"}),
        resolve("x"),
    ])

    assert out["total"] == 3
    assert out["geometry_eligible"] == 1
    assert out["by_permanence"]["permanent"] == 1
    assert "t" in out["single_season_objects"]
