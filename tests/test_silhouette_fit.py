"""L'accord silhouette/contours doit retrouver une pose — ou se déclarer aveugle."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from hotel_pipeline.silhouette_fit import (
    HeightEstimate,
    estimate_height,
    skyline,
    skyline_is_plausible,
    skyline_cost,
    MIN_CONTRAST,
    edge_distance_map,
    fit_cost,
    measure,
    project_roofline,
    search,
)

FOOTPRINT = [(-35.0, -20.0), (35.0, -20.0), (35.0, 20.0), (-35.0, 20.0)]
RIDGE = 13.0


def _render(camera, heading, fov, size=640):
    """Image de synthèse où le faîtage est la seule arête présente.

    La vérité terrain est ainsi connue exactement : ce que le module doit
    retrouver est ce qui a servi à dessiner.
    """
    canvas = np.zeros((size, size, 3), np.uint8)
    points = project_roofline(
        camera, FOOTPRINT, RIDGE, heading, fov, width_px=size, height_px=size
    )
    for u, v in points:
        cv2.circle(canvas, (int(u), int(v)), 1, (255, 255, 255), -1)
    return canvas


def test_projection_borne_au_cadre():
    points = project_roofline((0.0, -150.0), FOOTPRINT, RIDGE, 0.0, 60.0)
    assert points
    assert all(0 <= u < 640 and 0 <= v < 640 for u, v in points)


def test_faitage_hors_champ_ne_rend_rien():
    assert project_roofline((0.0, -150.0), FOOTPRINT, RIDGE, 180.0, 60.0) == []


def test_cout_minimal_a_la_pose_qui_a_dessine():
    """Le coût doit être plus bas à la vérité qu'à côté."""
    truth = (0.0, -120.0)
    dmap = edge_distance_map(_render(truth, 0.0, 60.0))

    exact, _ = fit_cost(dmap, truth, FOOTPRINT, RIDGE, 0.0, 60.0)
    shifted, _ = fit_cost(dmap, (15.0, -120.0), FOOTPRINT, RIDGE, 0.0, 60.0)

    # Le plancher n'est pas zéro : le trait dessiné a une épaisseur et Canny
    # le discrétise. Ce qui compte est l'écart entre la vérité et le décalage.
    assert exact < 2.0
    assert shifted > exact * 3


def test_recherche_retrouve_la_pose_synthetique():
    truth = (8.0, -118.0)
    dmap = edge_distance_map(_render(truth, 0.0, 60.0))
    declared = (truth[0] - 14.0, truth[1] + 10.0)

    found, cost, background = search(
        dmap, declared, FOOTPRINT, RIDGE, [(0.0, 60.0)], radius_m=30.0, step_m=1.0
    )

    assert math.dist(found, truth) < 3.0
    assert cost < background


def test_image_sans_contour_ne_conclut_pas():
    """Un ciel uni n'atteste rien : le coût y est partout identique."""
    blank = np.full((640, 640, 3), 128, np.uint8)
    dmap = edge_distance_map(blank)

    result = measure(dmap, (0.0, -120.0), FOOTPRINT, RIDGE, 0.0, 60.0)

    assert result.status == "ambiguous"
    assert not result.identified


def test_batiment_hors_cadre_est_declare_non_visible():
    dmap = edge_distance_map(_render((0.0, -120.0), 0.0, 60.0))

    result = measure(dmap, (0.0, -120.0), FOOTPRINT, RIDGE, 180.0, 60.0)

    assert result.status == "not_visible"
    assert not result.identified


def test_contraste_mesure_le_detachement_du_fond():
    truth = (0.0, -120.0)
    dmap = edge_distance_map(_render(truth, 0.0, 60.0))

    result = measure(dmap, truth, FOOTPRINT, RIDGE, 0.0, 60.0)

    assert result.identified
    assert result.contrast > MIN_CONTRAST


def test_contraste_nul_quand_le_fond_egale_l_optimum():
    from hotel_pipeline.silhouette_fit import FitResult

    assert FitResult(cost_px=10.0, background_px=10.0).contrast == pytest.approx(0.0)
    assert FitResult(cost_px=10.0, background_px=None).contrast is None


def test_l_optimum_ne_fuit_pas_vers_le_bord_du_domaine():
    """Le coût ne doit pas récompenser l'éloignement.

    Défaut mesuré sur le pilote avant normalisation : le coût décroissait de
    12,6 à 1,6 px à mesure qu'on élargissait le rayon exploré, et l'optimum
    se retrouvait toujours collé au bord — 10 m pour un rayon de 10 m, 79 m
    pour un rayon de 80 m. Aucun de ces minima n'était réel. Le contraste au
    voisinage ne le détectait pas, le biais agissant aussi sur le voisinage.
    """
    truth = (0.0, -120.0)
    dmap = edge_distance_map(_render(truth, 0.0, 60.0))

    distances = []
    for radius in (10.0, 30.0, 60.0):
        found, _cost, _background = search(
            dmap, truth, FOOTPRINT, RIDGE, [(0.0, 60.0)],
            radius_m=radius, step_m=2.0,
        )
        distances.append(math.dist(found, truth))

    # La solution doit rester près de la vérité quel que soit le domaine, et
    # surtout ne pas grandir avec lui.
    assert all(d < 6.0 for d in distances), distances
    assert distances[-1] < distances[0] + 5.0


def test_cout_stable_quand_la_silhouette_retrecit():
    """Deux poses également fausses doivent coûter comparablement.

    Sans normalisation, la plus éloignée coûtait deux fois moins cher pour un
    alignement identique — c'est ce qui faisait fuir l'optimum.
    """
    truth = (0.0, -120.0)
    dmap = edge_distance_map(_render(truth, 0.0, 60.0))

    near, _ = fit_cost(dmap, (10.0, -120.0), FOOTPRINT, RIDGE, 0.0, 60.0)
    far, _ = fit_cost(dmap, (10.0, -220.0), FOOTPRINT, RIDGE, 0.0, 60.0)

    assert near is not None and far is not None
    # La pose lointaine ne doit pas paraître meilleure que la proche.
    assert far > near


def _render_scene(camera, heading, fov, size=640, ground_clutter=True):
    """Scène avec ciel, bâtiment, et un sol encombré de contours.

    Reproduit la structure qui piégeait la mesure : une bande basse saturée de
    contours (voitures, clôture, végétation) vers laquelle le faîtage glisse
    quand la caméra s'éloigne.
    """
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[:, :] = (200, 140, 90)  # ciel bleu, en BGR
    # Un sol, sans quoi le ciel descendrait jusqu'au bas de l'image sur les
    # colonnes que le bâtiment n'occupe pas, et la frontière de ciel y
    # suivrait le bord du cadre au lieu d'une ligne de toit.
    canvas[int(size * 0.62):, :] = (70, 90, 80)

    points = project_roofline(
        camera, FOOTPRINT, RIDGE, heading, fov, width_px=size, height_px=size
    )
    if points:
        upper = {}
        for u, v in points:
            column = int(u)
            if column not in upper or v < upper[column]:
                upper[column] = v
        # Le bâti est **plein** : on interpole entre colonnes échantillonnées,
        # sans quoi les colonnes intermédiaires laissent voir le sol et la
        # frontière de ciel y plonge de 133 px — un artefact de rendu, non un
        # défaut de mesure.
        occupied = sorted(upper)
        for left, right in zip(occupied, occupied[1:]):
            for column in range(left, right + 1):
                ratio = (column - left) / max(1, right - left)
                row = upper[left] + (upper[right] - upper[left]) * ratio
                canvas[int(row):, column] = (60, 70, 110)

    if ground_clutter:
        for row in range(int(size * 0.62), size, 6):
            cv2.line(canvas, (0, row), (size, row), (20, 20, 20), 1)
        for column in range(0, size, 9):
            cv2.line(canvas, (column, int(size * 0.62)), (column, size), (30, 30, 30), 1)
    return canvas


def test_skyline_suit_la_frontiere_du_ciel():
    image = _render_scene((0.0, -120.0), 0.0, 60.0)
    horizon = skyline(image)

    assert horizon is not None
    assert np.isfinite(horizon).sum() > 300


def test_skyline_absente_sur_image_sans_ciel():
    assert skyline(np.zeros((640, 640, 3), np.uint8)) is None


def test_skyline_cost_minimal_a_la_verite():
    truth = (0.0, -120.0)
    horizon = skyline(_render_scene(truth, 0.0, 60.0))

    exact, _ = skyline_cost(horizon, truth, FOOTPRINT, RIDGE, 0.0, 60.0)
    shifted, _ = skyline_cost(horizon, (0.0, -180.0), FOOTPRINT, RIDGE, 0.0, 60.0)

    assert exact < 5.0
    assert shifted > exact * 2


def test_skyline_cost_ne_fuit_pas_dans_le_bruit_du_sol():
    """Le défaut central : sans contrainte de ciel, l'optimum partait au bord.

    Mesuré sur le pilote — la bande basse de l'image a une distance médiane au
    contour de 1 à 3 px contre 66 à 152 px dans le ciel, et le faîtage y
    glissait en s'éloignant. Avec la frontière de ciel comme référence,
    l'optimum se stabilise : 44,4 m pour un rayon exploré de 45, 60 ou 80 m.
    """
    truth = (0.0, -120.0)
    horizon = skyline(_render_scene(truth, 0.0, 60.0))

    def best_within(radius, step=3.0):
        best = None
        span = int(radius / step)
        for i in range(-span, span + 1):
            for j in range(-span, span + 1):
                if math.hypot(i * step, j * step) > radius:
                    continue
                camera = (truth[0] + i * step, truth[1] + j * step)
                cost, _ = skyline_cost(
                    horizon, camera, FOOTPRINT, RIDGE, 0.0, 60.0
                )
                if cost is not None and (best is None or cost < best[0]):
                    best = (cost, math.dist(camera, truth))
        return best

    distances = [best_within(r)[1] for r in (20.0, 40.0, 60.0)]

    # Ce qui est garanti est la **stabilité** : élargir le domaine ne doit
    # plus déplacer la solution. Avant la contrainte de ciel, elle suivait le
    # bord — 10 m pour 10 m explorés, 79 m pour 80 m.
    assert distances[1] == pytest.approx(distances[2], abs=1e-6), distances
    assert distances[2] < 40.0, distances


def test_une_vue_unique_laisse_la_profondeur_indeterminee():
    """La skyline seule ne fixe pas la distance, et c'est structurel.

    Reculer le long de l'axe de visée ne change presque pas la ligne de toit
    d'un bâtiment plat : mesuré ici, le coût vaut 4,42 à 27 m de la vérité
    contre 4,81 à la vérité même — 8 % d'écart, sous le bruit. C'est ce que
    la contrainte multi-vues de `pose_refine` est faite de lever ; ce module
    ne doit pas prétendre y suffire seul.
    """
    truth = (0.0, -120.0)
    horizon = skyline(_render_scene(truth, 0.0, 60.0))

    at_truth, _ = skyline_cost(horizon, truth, FOOTPRINT, RIDGE, 0.0, 60.0)
    behind, _ = skyline_cost(
        horizon, (truth[0], truth[1] - 27.0), FOOTPRINT, RIDGE, 0.0, 60.0
    )

    assert at_truth is not None and behind is not None
    # Les deux sont indiscernables : moins de 20 % d'écart.
    assert abs(behind - at_truth) / at_truth < 0.20


# --- Estimation de hauteur --------------------------------------------------

def _observation(camera, heading, fov, true_height, label):
    """Une vue synthétique d'un bâtiment de hauteur connue."""
    scene = _render_scene_at(camera, heading, fov, true_height)
    return (skyline(scene), camera, heading, fov, label)


def _render_scene_at(camera, heading, fov, height, size=640):
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[:, :] = (200, 140, 90)
    canvas[int(size * 0.62):, :] = (70, 90, 80)
    points = project_roofline(
        camera, FOOTPRINT, height, heading, fov, width_px=size, height_px=size
    )
    if points:
        upper = {}
        for u, v in points:
            column = int(u)
            if column not in upper or v < upper[column]:
                upper[column] = v
        occupied = sorted(upper)
        for left, right in zip(occupied, occupied[1:]):
            for column in range(left, right + 1):
                ratio = (column - left) / max(1, right - left)
                row = upper[left] + (upper[right] - upper[left]) * ratio
                canvas[int(row):, column] = (60, 70, 110)
    return canvas


def test_estime_une_hauteur_connue():
    """Trois positions distinctes, hauteur retrouvée."""
    truth = 8.5
    observations = [
        _observation((0.0, -120.0), 0.0, 60.0, truth, "sud"),
        _observation((90.0, 0.0), 270.0, 60.0, truth, "est"),
        _observation((-60.0, -90.0), 30.0, 60.0, truth, "sud-ouest"),
    ]

    estimate = estimate_height(observations, FOOTPRINT)

    assert estimate.measured, estimate.reason
    assert estimate.height_m == pytest.approx(truth, abs=1.0)
    assert estimate.votes >= 2


def test_une_seule_vue_ne_suffit_pas():
    """Le minimum de votes protège d'une mesure non corroborée."""
    observations = [_observation((0.0, -120.0), 0.0, 60.0, 8.5, "sud")]

    estimate = estimate_height(observations, FOOTPRINT)

    assert not estimate.measured
    assert estimate.status == "insufficient_votes"


def test_optimum_au_bord_du_domaine_est_rejete():
    """Une recherche qui bute sur sa borne n'a pas convergé."""
    observations = [
        _observation((0.0, -120.0), 0.0, 60.0, 8.5, "sud"),
        _observation((90.0, 0.0), 270.0, 60.0, 8.5, "est"),
    ]

    # Domaine dont la vérité est exclue : tous les votes butent sur une borne.
    estimate = estimate_height(
        observations, FOOTPRINT, minimum_m=12.0, maximum_m=16.0
    )

    assert not estimate.measured
    assert "bord du domaine" in (estimate.reason or "")


def test_vues_incoherentes_sont_signalees():
    """Deux hauteurs franchement différentes ne font pas une mesure."""
    observations = [
        _observation((0.0, -120.0), 0.0, 60.0, 6.0, "basse"),
        _observation((90.0, 0.0), 270.0, 60.0, 13.0, "haute"),
    ]

    estimate = estimate_height(observations, FOOTPRINT)

    assert estimate.status == "inconsistent"
    assert not estimate.measured


def test_skyline_verticale_est_ecartee():
    """Une arête de bâtiment proche n'est pas une ligne de toit.

    Cas réel du pilote : une vue notée 0,94 par CLIP dont la moitié du cadre
    est occupée par une concession automobile ; la frontière de ciel y suivait
    son arête verticale, oscillant de v=59 à v=469.
    """
    horizon = np.array([60.0] * 320 + [470.0] * 320)

    ok, reason = skyline_is_plausible(horizon, [250.0, 260.0])

    assert not ok
    assert "verticale" in (reason or "") or "toit" in (reason or "")


def test_skyline_horizontale_est_acceptee():
    horizon = np.array([250.0 + (i % 7) for i in range(640)])

    ok, reason = skyline_is_plausible(horizon, [245.0, 265.0])

    assert ok, reason
