"""La provenance décide de ce qui peut ancrer une projection."""

from __future__ import annotations

import math

import pytest
from shapely.geometry import Point

from hotel_pipeline.panorama_provenance import (
    BEARING_TOLERANCE_DEG,
    PoseProvenance,
    bearing_error,
    classify,
    summarise,
)


def test_capture_vehiculee_est_attestee():
    p = PoseProvenance("pano", "© Google")
    assert p.surveyed
    assert p.pose_status == "attested"
    assert p.usable_as_anchor


def test_photosphere_utilisateur_demande_un_raffinement():
    """Mesuré sur le pilote : 16,5° d'écart, ~17 m de dérive en position."""
    p = PoseProvenance("pano", "© Marc Durand - Panosphere360")
    assert not p.surveyed
    assert p.pose_status == "needs_refinement"
    assert not p.usable_as_anchor


def test_provenance_inconnue_n_est_jamais_attestee():
    """Le défaut ne doit pas être la famille la plus favorable.

    Supposer « Google Car » faute de preuve reproduirait exactement l'erreur
    que ce module corrige : la vue désalignée était indiscernable d'une vue
    saine jusqu'à ce qu'on lise son `copyright`.
    """
    p = PoseProvenance("pano", None)
    assert p.pose_status == "unknown_provenance"
    assert not p.usable_as_anchor


def test_capture_vehiculee_qui_derive_est_rattrapee_par_la_mesure():
    """La famille oriente, la mesure tranche."""
    p = PoseProvenance("pano", "© Google", bearing_error_deg=BEARING_TOLERANCE_DEG + 5)
    assert p.surveyed
    assert p.pose_status == "needs_refinement"


def test_ecart_sous_tolerance_reste_attestee():
    p = PoseProvenance("pano", "© Google", bearing_error_deg=BEARING_TOLERANCE_DEG - 1)
    assert p.pose_status == "attested"


def test_classify_n_invente_pas_de_copyright_absent():
    out = classify(["a", "b"], copyrights={"a": "© Google"})
    assert out["a"].pose_status == "attested"
    assert out["b"].pose_status == "unknown_provenance"


def test_bearing_error_est_circulaire():
    """359° et 1° sont distants de 2°, non de 358°."""
    centroid = Point(0.0, 100.0)  # plein nord de l'origine → cap géométrique 0°
    assert bearing_error((0.0, 0.0), centroid, 359.0) == pytest.approx(1.0)
    assert bearing_error((0.0, 0.0), centroid, 1.0) == pytest.approx(1.0)


def test_bearing_error_en_crs_projete():
    """`atan2(dx, dy)` : est pur vaut 90°, non 0°."""
    centroid = Point(100.0, 0.0)
    assert bearing_error((0.0, 0.0), centroid, 90.0) == pytest.approx(0.0)


def test_summarise_compte_par_statut():
    out = summarise(
        classify(
            ["a", "b", "c", "d"],
            copyrights={"a": "© Google", "b": "© Google", "c": "© Virtuo 360"},
        )
    )
    assert out["total"] == 4
    assert out["by_status"]["attested"] == 2
    assert out["by_status"]["needs_refinement"] == 1
    assert out["by_status"]["unknown_provenance"] == 1
    assert out["attested_fraction"] == pytest.approx(0.5)


def test_cache_illisible_ne_promeut_personne():
    """Un cache absent rend l'inconnu, non un défaut favorable."""
    out = classify(["a"], copyrights={})
    assert out["a"].pose_status == "unknown_provenance"
