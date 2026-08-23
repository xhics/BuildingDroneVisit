"""Ce qui se ressemble à toutes les dates est permanent ; ce qui change ne l'est pas."""

from __future__ import annotations

import pytest

from hotel_pipeline.permanence import Permanence
from hotel_pipeline.temporal_consensus import (
    MIN_DATES,
    CellObservation,
    build,
    cell_of,
    summarise,
    to_scene_objects,
    verdict_for,
)

CELL = (3, 7)


def _obs(date, season, green, snow=0.0, brightness=0.5):
    return CellObservation(
        cell=CELL, date=date, season=season,
        green=green, snow=snow, brightness=brightness,
    )


def _verdict(observations, **kwargs):
    kwargs.setdefault("stable_below", 0.10)
    kwargs.setdefault("seasonal_above", 0.25)
    return verdict_for(observations, **kwargs)


def test_cellule_du_maillage():
    assert cell_of(0.0, 0.0, size_m=5.0) == (0, 0)
    assert cell_of(7.5, 12.0, size_m=5.0) == (1, 2)
    assert cell_of(-1.0, -1.0, size_m=5.0) == (-1, -1)


def test_asphalte_identique_partout_est_stable():
    verdict = _verdict([
        _obs("2024-06", "summer", 0.02, brightness=0.40),
        _obs("2024-10", "autumn", 0.03, brightness=0.42),
        _obs("2025-04", "spring", 0.02, brightness=0.41),
    ])

    assert verdict.status == "stable"
    assert verdict.decided


def test_gazon_qui_change_avec_les_saisons_est_saisonnier():
    verdict = _verdict([
        _obs("2024-06", "summer", 0.85, brightness=0.55),
        _obs("2024-10", "autumn", 0.30, brightness=0.45),
        _obs("2025-01", "winter", 0.02, snow=0.90, brightness=0.90),
    ])

    assert verdict.status == "seasonal"
    assert verdict.decided


def test_une_seule_date_ne_tranche_pas():
    verdict = _verdict([_obs("2024-06", "summer", 0.85)])

    assert verdict.status == "insufficient_dates"
    assert not verdict.decided
    assert str(MIN_DATES) in (verdict.reason or "")


def test_variation_dans_une_seule_saison_n_est_pas_de_la_saisonnalite():
    """Une voiture garée là ce jour-là n'est pas un changement de saison."""
    verdict = _verdict([
        _obs("2024-06", "summer", 0.85, brightness=0.55),
        _obs("2024-07", "summer", 0.10, brightness=0.20),
    ])

    assert verdict.status == "unstable_same_season"
    assert not verdict.decided
    assert "non saisonnalité" in (verdict.reason or "")


def test_zone_grise_ne_tranche_pas():
    # La dispersion est la moyenne des étendues des trois composantes : il
    # faut donc les faire varier ensemble pour viser la zone grise, un écart
    # sur le seul vert étant dilué par les deux autres.
    verdict = _verdict([
        _obs("2024-06", "summer", 0.50, snow=0.10, brightness=0.50),
        _obs("2024-10", "autumn", 0.68, snow=0.28, brightness=0.68),
    ])

    assert verdict.status == "undecided"
    assert not verdict.decided


def test_deux_vues_du_meme_jour_ne_gonflent_pas_la_variance():
    """Deux angles du même jour mesurent l'angle, pas la saison."""
    verdict = _verdict([
        _obs("2024-06", "summer", 0.90, brightness=0.70),
        _obs("2024-06", "summer", 0.10, brightness=0.20),
        _obs("2024-10", "autumn", 0.50, brightness=0.45),
    ])

    # Les deux vues de juin sont moyennées : reste juin contre octobre.
    assert verdict.status in {"stable", "undecided"}
    assert len(verdict.dates) == 2


def test_build_regroupe_par_cellule():
    observations = [
        CellObservation((0, 0), "2024-06", "summer", 0.9, 0.0, 0.6),
        CellObservation((0, 0), "2025-01", "winter", 0.0, 0.9, 0.9),
        CellObservation((1, 1), "2024-06", "summer", 0.05, 0.0, 0.40),
        CellObservation((1, 1), "2025-01", "winter", 0.04, 0.0, 0.41),
    ]

    verdicts = build(observations)

    assert verdicts[(0, 0)].status == "seasonal"
    assert verdicts[(1, 1)].status == "stable"


def test_cellules_non_tranchees_ne_deviennent_pas_des_objets():
    """Produire une surface pour ce qu'on n'a pas classé serait inventer."""
    verdicts = build([
        CellObservation((0, 0), "2024-06", "summer", 0.5, 0.0, 0.5),
    ])

    assert to_scene_objects(verdicts) == []


def test_conversion_en_objets_de_scene():
    verdicts = build([
        CellObservation((0, 0), "2024-06", "summer", 0.9, 0.0, 0.6),
        CellObservation((0, 0), "2025-01", "winter", 0.0, 0.9, 0.9),
        CellObservation((1, 1), "2024-06", "summer", 0.05, 0.0, 0.40),
        CellObservation((1, 1), "2025-01", "winter", 0.04, 0.0, 0.41),
    ])

    objects = {o.object_id: o for o in to_scene_objects(verdicts)}

    assert objects["CELL_0_0"].permanence is Permanence.SEASONAL_SURFACE
    assert objects["CELL_1_1"].permanence is Permanence.PERMANENT
    assert objects["CELL_0_0"].observed_seasons == {"summer", "winter"}


def test_summarise_rapporte_la_part_tranchee():
    verdicts = build([
        CellObservation((0, 0), "2024-06", "summer", 0.05, 0.0, 0.40),
        CellObservation((0, 0), "2025-01", "winter", 0.04, 0.0, 0.41),
        CellObservation((2, 2), "2024-06", "summer", 0.5, 0.0, 0.5),
    ])

    out = summarise(verdicts)

    assert out["cells"] == 2
    assert out["decided_fraction"] == pytest.approx(0.5)


# --- Projection des cellules et échantillonnage ------------------------------

import numpy as np  # noqa: E402

from hotel_pipeline.temporal_consensus import (  # noqa: E402
    MIN_PATCH_PIXELS,
    project_cell,
    sample_patch,
)


def test_cellule_devant_la_camera_est_projetee():
    # Caméra à 60 m au sud, visant le nord ; cellule à l'origine.
    box = project_cell((0, 0), (0.0, -60.0), 0.0, 60.0)

    assert box is not None
    left, top, right, bottom = box
    assert 0 <= left < right <= 640
    assert 0 <= top < bottom <= 640
    # Le sol est sous l'horizon : la cellule est dans la moitié basse.
    assert top > 320


def test_cellule_derriere_la_camera_n_est_pas_projetee():
    assert project_cell((0, 0), (0.0, -60.0), 180.0, 60.0) is None


def test_cellule_hors_cadre_n_est_pas_projetee():
    """Une cellule non vue n'est pas une cellule vide."""
    assert project_cell((40, 0), (0.0, -60.0), 0.0, 20.0) is None


def test_cellule_lointaine_retrecit():
    proche = project_cell((0, 0), (0.0, -30.0), 0.0, 60.0)
    lointaine = project_cell((0, 0), (0.0, -150.0), 0.0, 60.0)

    assert proche is not None and lointaine is not None
    aire_proche = (proche[2] - proche[0]) * (proche[3] - proche[1])
    aire_lointaine = (lointaine[2] - lointaine[0]) * (lointaine[3] - lointaine[1])
    assert aire_lointaine < aire_proche


def test_echantillon_de_gazon_est_vert():
    image = np.zeros((640, 640, 3), np.uint8)
    image[:, :] = (60, 170, 70)

    descriptor = sample_patch(image, (100, 100, 200, 200))

    assert descriptor is not None
    green, snow, brightness = descriptor
    assert green > 0.8
    assert snow < 0.1


def test_echantillon_de_neige_est_neutre_et_lumineux():
    image = np.zeros((640, 640, 3), np.uint8)
    image[:, :] = (240, 240, 240)

    green, snow, brightness = sample_patch(image, (100, 100, 200, 200))

    assert snow > 0.8
    assert green < 0.1
    assert brightness > 0.8


def test_surface_claire_texturee_n_est_pas_de_la_neige():
    """La neige est lisse ; un damier clair ne l'est pas."""
    image = np.zeros((640, 640, 3), np.uint8)
    image[:, :] = (240, 240, 240)
    image[::4, :] = (120, 120, 120)

    _green, snow, _brightness = sample_patch(image, (100, 100, 200, 200))

    assert snow == 0.0


def test_emprise_trop_petite_ne_donne_pas_de_descripteur():
    image = np.zeros((640, 640, 3), np.uint8)

    assert sample_patch(image, (10, 10, 12, 12)) is None


def test_seuil_de_taille_est_respecte():
    image = np.zeros((640, 640, 3), np.uint8)
    image[:, :] = (60, 170, 70)
    cote = int(MIN_PATCH_PIXELS ** 0.5)

    assert sample_patch(image, (0, 0, cote - 1, cote - 1)) is None
    assert sample_patch(image, (0, 0, cote + 2, cote + 2)) is not None


def test_cellule_derriere_un_obstacle_n_est_pas_visible():
    """Sans ce test, une cellule au sol se décrit avec des pixels de façade.

    Mesuré sur le pilote : 43 % des observations traversaient le bâtiment.
    """
    from shapely.geometry import box as shapely_box

    from hotel_pipeline.temporal_consensus import cell_is_visible

    batiment = shapely_box(-20.0, -10.0, 20.0, 10.0)
    camera = (0.0, -100.0)

    # Cellule derrière le bâtiment, vue depuis le sud.
    assert not cell_is_visible((0, 6), camera, [batiment])
    # Cellule devant : la vue est dégagée.
    assert cell_is_visible((0, -6), camera, [batiment])


def test_sans_obstacle_tout_est_visible():
    from hotel_pipeline.temporal_consensus import cell_is_visible

    assert cell_is_visible((0, 6), (0.0, -100.0), [])
    assert cell_is_visible((0, 6), (0.0, -100.0), None)
