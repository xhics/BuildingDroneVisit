"""Le raffinement doit retrouver un décalage connu — ou refuser de conclure."""

from __future__ import annotations

import math

import pytest

from hotel_pipeline.pose_refine import (
    MAX_RESIDUAL_PX,
    RefinedPose,
    SilhouetteObservation,
    project_bounds,
    refine,
    summarise,
)

#: Bâtiment rectangulaire de 70 × 40 m, haut de 13 m, centré sur l'origine.
BUILDING = [
    (x, y, z)
    for x in (-35.0, 35.0)
    for y in (-20.0, 20.0)
    for z in (0.0, 13.0)
]


def _observe(camera, heading, fov, **kwargs):
    """Silhouette qu'une caméra placée là produirait — la vérité terrain."""
    bounds = project_bounds(camera, BUILDING, heading, fov)
    assert bounds is not None
    return SilhouetteObservation(
        heading_deg=heading, fov_deg=fov, u_min=bounds[0], u_max=bounds[1], **kwargs
    )


def test_projection_centre_le_batiment_vu_de_face():
    """Caméra plein sud visant le nord : le bâtiment est centré."""
    bounds = project_bounds((0.0, -150.0), BUILDING, 0.0, 60.0)
    assert bounds is not None
    assert (bounds[0] + bounds[1]) / 2 == pytest.approx(320.0, abs=1.0)


def test_projection_rend_none_si_le_batiment_est_derriere():
    assert project_bounds((0.0, -150.0), BUILDING, 180.0, 60.0) is None


def test_retrouve_un_decalage_connu():
    """Le cas du pilote : position déclarée fausse, deux vues la corrigent."""
    truth = (12.0, -95.0)
    declared = (truth[0] - 12.0, truth[1] + 12.0)  # dérive de ~17 m
    observations = [
        _observe(truth, 5.0, 64.0),
        _observe(truth, 20.0, 64.0),
    ]

    result = refine("pano", declared, BUILDING, observations)

    assert result.converged
    assert result.residual_px < 2.0
    assert result.refined is not None
    assert math.dist(result.refined, truth) < 2.0


def test_ne_bouge_pas_une_pose_deja_juste():
    truth = (0.0, -120.0)
    observations = [_observe(truth, 0.0, 60.0), _observe(truth, 15.0, 60.0)]

    result = refine("pano", truth, BUILDING, observations)

    assert result.converged
    assert result.shift_m < 2.0


def test_une_seule_vue_ne_suffit_pas():
    """Position et cap y sont indiscernables : refuser vaut mieux qu'ajuster."""
    truth = (0.0, -120.0)
    result = refine("pano", (10.0, -110.0), BUILDING, [_observe(truth, 0.0, 60.0)])

    assert result.status == "insufficient_views"
    assert result.refined is None


def test_deux_vues_trop_proches_ne_contraignent_pas():
    truth = (0.0, -120.0)
    observations = [_observe(truth, 0.0, 60.0), _observe(truth, 2.0, 60.0)]

    result = refine("pano", (10.0, -110.0), BUILDING, observations)

    assert result.status == "insufficient_baseline"
    assert result.refined is None


def test_bornes_tronquees_ne_contraignent_pas():
    """Une silhouette coupée par le cadre mesure le cadre, non le bâtiment."""
    truth = (0.0, -120.0)
    observations = [
        _observe(truth, 0.0, 60.0, left_truncated=True, right_truncated=True),
        _observe(truth, 15.0, 60.0, left_truncated=True),
    ]

    result = refine("pano", (10.0, -110.0), BUILDING, observations)

    assert result.status == "insufficient_bounds"


def test_observations_incoherentes_ne_convergent_pas():
    """Aucune position n'explique des bornes contradictoires : le dire."""
    observations = [
        SilhouetteObservation(0.0, 60.0, u_min=10.0, u_max=60.0),
        SilhouetteObservation(20.0, 60.0, u_min=580.0, u_max=630.0),
    ]

    result = refine("pano", (0.0, -120.0), BUILDING, observations)

    assert not result.converged
    assert result.status == "not_converged"
    assert result.residual_px > MAX_RESIDUAL_PX


def test_refus_hors_rayon_de_recherche():
    """Une erreur de 200 m n'est pas une dérive de saisie : ne pas téléporter."""
    truth = (0.0, -120.0)
    observations = [_observe(truth, 0.0, 60.0), _observe(truth, 15.0, 60.0)]

    result = refine("pano", (0.0, -320.0), BUILDING, observations)

    assert not result.converged
    if result.refined is not None:
        assert result.shift_m <= 40.0 + 1e-6


def test_summarise_compte_les_statuts():
    out = summarise([
        RefinedPose("a", (0, 0), refined=(3, 4), residual_px=2.0, status="refined"),
        RefinedPose("b", (0, 0), status="insufficient_views"),
    ])
    assert out["total"] == 2
    assert out["refined"] == 1
    assert out["median_shift_m"] == pytest.approx(5.0)
