"""Tests de l'alignement Sim(3) canonique.

Ces tests portent sur le **comportement numérique**, pas sur la forme des
schémas. C'est précisément ce qui manquait : trois implémentations d'Umeyama
divergentes passaient toute la suite parce qu'aucun test ne vérifiait qu'une
transformation connue était retrouvée.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hotel_pipeline.geometry_align import (
    align_by_correspondence,
    alignment_rmse,
    apply_sim3,
    umeyama_sim3,
)


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def test_recovers_known_similarity_exactly() -> None:
    """Une Sim(3) connue doit être retrouvée à la précision machine."""
    rng = np.random.default_rng(12345)
    for _ in range(200):
        n = int(rng.integers(3, 40))
        source = rng.normal(size=(n, 3)) * float(rng.uniform(0.1, 50.0))
        R_true = _random_rotation(rng)
        s_true = float(rng.uniform(0.05, 20.0))
        t_true = rng.normal(size=3) * 10.0
        target = apply_sim3(source, R_true, t_true, s_true)

        R, t, s = umeyama_sim3(source, target)

        assert s == pytest.approx(s_true, rel=1e-8)
        assert np.allclose(R, R_true, atol=1e-8)
        assert np.allclose(t, t_true, atol=1e-6)
        assert alignment_rmse(apply_sim3(source, R, t, s), target) < 1e-8


def test_rotation_is_proper_even_on_reflected_input() -> None:
    """La correction de réflexion garantit det(R) = +1, jamais -1."""
    rng = np.random.default_rng(7)
    source = rng.normal(size=(12, 3))
    reflected = source.copy()
    reflected[:, 2] *= -1.0

    R, _, _ = umeyama_sim3(source, reflected)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)


def test_direction_is_source_to_target() -> None:
    """`umeyama_sim3(a, b)` amène a sur b — et non l'inverse.

    C'est la confusion qui faisait rapporter 6,8 m d'écart entre deux runs
    pourtant parfaitement superposables.
    """
    rng = np.random.default_rng(3)
    source = rng.normal(size=(10, 3)) * 4.0
    R_true = _random_rotation(rng)
    target = apply_sim3(source, R_true, np.array([5.0, -2.0, 1.0]), 3.0)

    R, t, s = umeyama_sim3(source, target)
    assert alignment_rmse(apply_sim3(source, R, t, s), target) < 1e-9

    # Appliquer la transformation au mauvais nuage ne doit pas coïncider.
    assert alignment_rmse(apply_sim3(target, R, t, s), source) > 1.0


def test_identical_clouds_have_zero_rmse() -> None:
    """Deux reconstructions identiques ne dérivent pas."""
    rng = np.random.default_rng(99)
    pts = rng.normal(size=(15, 3)) * 7.0
    R, t, s = umeyama_sim3(pts, pts)
    assert alignment_rmse(apply_sim3(pts, R, t, s), pts) < 1e-9


def test_degenerate_inputs_return_identity() -> None:
    """Une entrée insuffisante retourne l'identité, pas une transformation inventée."""
    pts = np.zeros((2, 3))
    R, t, s = umeyama_sim3(pts, pts)
    assert np.allclose(R, np.eye(3))
    assert np.allclose(t, np.zeros(3))
    assert s == 1.0

    # Formes incompatibles.
    R, t, s = umeyama_sim3(np.zeros((5, 3)), np.zeros((4, 3)))
    assert np.allclose(R, np.eye(3))
    assert s == 1.0


def test_correspondence_is_by_key_not_position() -> None:
    """Un sous-corpus mélangé s'aligne parfaitement sur le corpus plein.

    Apparier par position comparerait des caméras sans rapport et produirait
    une dérive fantôme — en plus de planter sur des tailles différentes.
    """
    rng = np.random.default_rng(21)
    ids = [f"asset-{i}" for i in range(12)]
    full = {aid: rng.normal(size=3) * 6.0 for aid in ids}

    # Sous-ensemble dans un ordre arbitraire, taille différente.
    subset_ids = ["asset-9", "asset-2", "asset-7", "asset-0", "asset-11"]
    subset = {aid: full[aid] for aid in subset_ids}

    rmse, n_common = align_by_correspondence(subset, full)
    assert n_common == 5
    assert rmse < 1e-9


def test_correspondence_survives_rigid_transform_of_subset() -> None:
    """Un sous-corpus transformé reste aligné : la dérive mesurée est nulle."""
    rng = np.random.default_rng(5)
    ids = [f"a{i}" for i in range(10)]
    full = {aid: rng.normal(size=3) * 5.0 for aid in ids}

    R = _random_rotation(rng)
    moved = {
        aid: (2.5 * (R @ full[aid]) + np.array([1.0, 2.0, 3.0]))
        for aid in ids[:6]
    }

    rmse, n_common = align_by_correspondence(moved, full)
    assert n_common == 6
    assert rmse < 1e-9


def test_too_few_common_ids_is_inconclusive_not_zero() -> None:
    """Moins de 3 identifiants communs : Sim(3) sous-déterminée, donc inf."""
    a = {"x": np.zeros(3), "y": np.ones(3)}
    b = {"x": np.zeros(3), "y": np.ones(3)}
    rmse, n_common = align_by_correspondence(a, b)
    assert n_common == 2
    assert math.isinf(rmse)


def test_alignment_rmse_is_per_point_distance() -> None:
    """L'RMSE est une distance par point, pas une moyenne par coordonnée."""
    a = np.zeros((1, 3))
    b = np.array([[3.0, 4.0, 0.0]])  # distance 5
    assert alignment_rmse(a, b) == pytest.approx(5.0)
