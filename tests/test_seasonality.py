"""La saison se lit sur le calendrier **et** sur les pixels."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from hotel_pipeline.seasonality import (
    GREEN_ACTIVE,
    SNOW_SUSPECT,
    SeasonReading,
    ground_mask,
    parse_capture_date,
    read,
    season_of_month,
    summarise,
)


def _scene(ground_bgr, size=640):
    """Image : ciel en haut, sol d'une couleur donnée en bas."""
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[:, :] = (200, 140, 90)
    canvas[int(size * 0.5):, :] = ground_bgr
    return canvas


def test_saison_du_mois():
    assert season_of_month(1) == "winter"
    assert season_of_month(5) == "spring"
    assert season_of_month(7) == "summer"
    assert season_of_month(10) == "autumn"


def test_mois_inconnu_ne_recoit_pas_de_saison_par_defaut():
    """Supposer l'été ferait entrer un jardin fleuri dans une scène d'hiver."""
    assert season_of_month(None) is None
    assert season_of_month(13) is None


def test_parse_capture_date():
    assert parse_capture_date("2025-05") == (2025, 5)
    assert parse_capture_date("2016") == (2016, None)
    assert parse_capture_date("") == (None, None)
    assert parse_capture_date(None) == (None, None)
    assert parse_capture_date("pas-une-date") == (None, None)


def test_mois_hors_bornes_ignore():
    assert parse_capture_date("2025-19") == (2025, None)


def test_gazon_vert_est_vu_comme_vegetation_active():
    reading = read(_scene((60, 170, 70)), "2025-07")

    assert reading.measured
    assert reading.green_fraction > GREEN_ACTIVE
    assert reading.vegetation_active is True
    assert reading.snow_possible is False


def test_asphalte_n_est_ni_vert_ni_neige():
    reading = read(_scene((90, 90, 92)), "2025-07")

    assert reading.measured
    assert reading.vegetation_active is False
    assert reading.snow_possible is False


def test_surface_blanche_lisse_leve_l_indice_de_neige():
    reading = read(_scene((240, 240, 240)), "2025-01")

    assert reading.snow_index > SNOW_SUSPECT
    assert reading.snow_possible is True


def test_indice_de_neige_reste_un_indice():
    """Une chaussée claire et lisse le lève aussi — le module ne le cache pas.

    Mesuré sur le pilote : 16 % d'indice sur une vue d'avril sans aucune neige,
    dont le candidat était l'asphalte clair d'un cul-de-sac. D'où le conflit
    signalé quand le mois contredit l'indice.
    """
    reading = read(_scene((240, 240, 240)), "2025-07")

    assert reading.snow_possible is True
    assert reading.conflicts, "un mois d'été avec de la neige doit être signalé"
    assert "confondue" in reading.conflicts[0]


def test_printemps_precoce_est_signale_non_rejete():
    """Avril vert n'est pas une erreur : c'est un printemps précoce.

    Cas réel du pilote — une vue d'avril 2025 montre un arbre en fleurs et un
    gazon vert. La référence reste utilisable pour une scène printanière.
    """
    reading = read(_scene((60, 170, 70)), "2025-04")

    assert reading.vegetation_active is True
    assert reading.foliage_expected is False
    assert reading.conflicts
    assert reading.measured, "un conflit n'invalide pas la mesure"


def test_cadrage_sans_sol_ne_conclut_pas():
    ciel = np.zeros((640, 640, 3), np.uint8)
    ciel[:, :] = (200, 140, 90)
    horizon = np.full(640, 639.0)  # le ciel descend jusqu'en bas

    reading = read(ciel, "2025-07", horizon=horizon)

    assert reading.status == "no_ground"
    assert not reading.measured
    assert reading.green_fraction is None


def test_masque_de_sol_exclut_le_dessus_de_la_ligne_de_toit():
    image = np.zeros((640, 640, 3), np.uint8)
    horizon = np.full(640, 400.0)

    mask = ground_mask(image, horizon)

    assert not mask[:400].any()
    assert mask[500].all()


def test_image_sans_date_reste_mesurable():
    """Les pixels parlent même quand le calendrier se tait."""
    reading = read(_scene((60, 170, 70)), None)

    assert reading.declared_season is None
    assert reading.measured
    assert reading.vegetation_active is True


def test_summarise_signale_les_saisons_absentes():
    """Le plus utile : dire quelles scènes ne peuvent pas être référencées."""
    readings = [
        SeasonReading(declared_season="spring", capture_month=5),
        SeasonReading(declared_season="summer", capture_month=7),
        SeasonReading(declared_season=None),
    ]

    out = summarise(readings)

    assert out["total"] == 3
    assert out["undated"] == 1
    assert out["by_season"] == {"spring": 1, "summer": 1}
    assert out["missing_seasons"] == ["autumn", "winter"]
