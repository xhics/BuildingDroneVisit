"""Le volume rendu est confronté aux photographies, ronde par ronde."""

from __future__ import annotations

import numpy as np
import pytest

from hotel_pipeline.conditioning.rounds import (
    MIN_PROFILE_MASS,
    ComparisonReport,
    RoundResult,
    _agreement,
    _profile_from_render,
)


class _Frame:
    def __init__(self, silhouette: np.ndarray) -> None:
        self.silhouette = silhouette
        self.target_coverage = float((silhouette == 2).mean())


# --- accord de profils ------------------------------------------------------


def test_deux_profils_identiques_s_accordent() -> None:
    profile = np.array([0.0, 0.1, 0.4, 0.3, 0.1, 0.0, 0.0, 0.0])
    assert _agreement(profile, profile) == pytest.approx(1.0)


def test_un_profil_decale_s_accorde_moins() -> None:
    """C'est le cas mesuré : bon cadrage 0,88, mauvais 0,50."""
    photo = np.array([0.0, 0.0, 0.3, 0.4, 0.2, 0.0, 0.0, 0.0])
    proche = np.array([0.0, 0.1, 0.35, 0.3, 0.15, 0.0, 0.0, 0.0])
    loin = np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.4, 0.1])

    assert _agreement(photo, proche) > _agreement(photo, loin)


def test_un_profil_vide_ne_s_accorde_avec_rien() -> None:
    """Deux images vides corrèlent parfaitement : le piège à éviter."""
    photo = np.array([0.0, 0.3, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0])
    vide = np.zeros(8)

    assert _agreement(photo, vide) == 0.0
    assert _agreement(vide, vide) == 0.0


def test_un_profil_trop_maigre_est_ecarte() -> None:
    photo = np.array([0.0, 0.3, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0])
    maigre = np.full(8, MIN_PROFILE_MASS / 4)

    assert _agreement(photo, maigre) == 0.0


def test_des_profils_de_longueurs_differentes_se_comparent() -> None:
    """La photo et le rendu n'ont pas la même hauteur en pixels."""
    photo = np.array([0.0, 0.2, 0.5, 0.3, 0.1, 0.0])
    rendu = np.array([0.0, 0.0, 0.2, 0.4, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0])

    assert _agreement(photo, rendu) > 0.0


def test_un_profil_constant_ne_prouve_rien() -> None:
    plat = np.full(8, 0.3)
    assert _agreement(plat, plat) == 0.0


# --- extraction depuis un rendu ---------------------------------------------


def test_le_profil_du_rendu_suit_les_bandes() -> None:
    silhouette = np.zeros((4, 10), dtype=int)
    silhouette[1, :] = 2  # bâtiment cible sur une bande
    silhouette[2, :5] = 1  # obstacle sur une demi-bande

    profile = _profile_from_render(_Frame(silhouette), (1, 2))

    assert profile[0] == pytest.approx(0.0)
    assert profile[1] == pytest.approx(1.0)
    assert profile[2] == pytest.approx(0.5)


def test_la_vegetation_n_entre_pas_dans_le_profil_du_bati() -> None:
    silhouette = np.full((3, 4), 3, dtype=int)  # tout en végétation
    assert _profile_from_render(_Frame(silhouette), (1, 2)).sum() == 0.0


# --- verdict ----------------------------------------------------------------


def _round(correlation: float) -> RoundResult:
    return RoundResult("a", 200.0, 40.0, 60.0, correlation, 0.1)


def test_un_bon_accord_declare_la_silhouette_conforme() -> None:
    report = ComparisonReport("t", best=[_round(0.9), _round(0.8)])

    assert report.verdict() == "silhouette_conforme"
    assert report.mean_correlation > 0.8


def test_un_accord_moyen_reste_approchant() -> None:
    assert ComparisonReport("t", best=[_round(0.55)]).verdict() == "silhouette_approchante"


def test_un_accord_faible_dement_la_silhouette() -> None:
    """C'est l'intérêt de la comparaison : pouvoir dire que le volume est faux."""
    assert ComparisonReport("t", best=[_round(0.2)]).verdict() == "silhouette_dementie"


def test_sans_vue_comparable_aucun_verdict() -> None:
    report = ComparisonReport("t")

    assert report.verdict() == "non_comparable"
    assert report.mean_correlation == 0.0


def test_le_rapport_porte_ses_reserves() -> None:
    joined = " ".join(ComparisonReport("t").as_dict()["caveats"])

    assert "apparence" in joined
    assert "recherchée" in joined
    assert "presque vide" in joined
