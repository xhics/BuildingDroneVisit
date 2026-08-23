"""Alignement Sim(3) — implémentation canonique unique (Umeyama 1991).

Ce module est la **seule** source de vérité pour l'alignement de similitude.
Les copies divergentes dans `reconstruction_consensus.py` et `stability.py`
produisaient deux échelles différentes pour la même entrée ; elles délèguent
désormais ici.

Convention explicite, parce que c'est là que les bugs se logent :
`umeyama_sim3(source, target)` retourne `(R, t, s)` tel que

    apply_sim3(source, R, t, s) ≈ target

c'est-à-dire la transformation qui amène **source sur target**. Le premier
argument est celui qui bouge.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "umeyama_sim3",
    "apply_sim3",
    "alignment_rmse",
    "align_by_correspondence",
]


def umeyama_sim3(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Estime la Sim(3) amenant `source` sur `target` (Umeyama 1991).

    Args:
        source: points (N, 3) à transformer.
        target: points (N, 3) de destination, en correspondance indice à indice.

    Returns:
        `(R, t, s)` avec `R` rotation 3x3 (det = +1), `t` translation (3,),
        `s` échelle scalaire > 0, tels que ``s * R @ source_i + t ≈ target_i``.

    Une entrée dégénérée (formes incompatibles, moins de 3 points, variance
    nulle) retourne l'identité plutôt qu'une transformation inventée.
    """
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        return np.eye(3), np.zeros(3), 1.0
    if source.shape[0] < 3:
        return np.eye(3), np.zeros(3), 1.0

    n = source.shape[0]
    mu_src = source.mean(axis=0)
    mu_tgt = target.mean(axis=0)

    src_c = source - mu_src
    tgt_c = target - mu_tgt

    # Variance de la SOURCE : c'est elle qu'on met à l'échelle.
    var_src = float((src_c**2).sum()) / n
    if var_src < 1e-12:
        return np.eye(3), mu_tgt - mu_src, 1.0

    # Covariance target x source (ordre important pour la direction).
    sigma = (tgt_c.T @ src_c) / n

    U, d, Vt = np.linalg.svd(sigma)

    # Correction de réflexion : garantit det(R) = +1.
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt

    # Échelle d'Umeyama : trace(D @ S) / var_source.
    scale = float((d * np.diag(S)).sum()) / var_src
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    t = mu_tgt - scale * (R @ mu_src)
    return R, t, scale


def apply_sim3(
    points: np.ndarray, R: np.ndarray, t: np.ndarray, s: float
) -> np.ndarray:
    """Applique ``s * R @ p + t`` à chaque ligne de `points` (N, 3)."""
    return s * (points @ R.T) + t


def alignment_rmse(a: np.ndarray, b: np.ndarray) -> float:
    """RMSE euclidienne par point entre deux nuages en correspondance.

    Contrairement à une moyenne sur toutes les coordonnées aplaties, on prend
    la norme par point : le résultat est une distance en mètres, pas une
    quantité par axe.
    """
    if a.shape != b.shape or a.size == 0:
        return float("inf")
    per_point = np.linalg.norm(a - b, axis=1)
    return float(np.sqrt((per_point**2).mean()))


def align_by_correspondence(
    source: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
) -> tuple[float, int]:
    """Aligne deux jeux de centres indexés par identifiant.

    La correspondance est établie par **clé partagée**, jamais par position
    dans un tableau : deux sous-corpus échantillonnés séparément n'ont aucune
    raison d'être dans le même ordre.

    Returns:
        `(rmse_m, n_common)`. Avec moins de 3 identifiants communs, l'alignement
        Sim(3) est sous-déterminé et on retourne `(inf, n_common)` — une
        impossibilité de conclure, pas une dérive nulle.
    """
    common = sorted(set(source) & set(target))
    if len(common) < 3:
        return float("inf"), len(common)

    src = np.array([source[k] for k in common], dtype=float)
    tgt = np.array([target[k] for k in common], dtype=float)

    R, t, s = umeyama_sim3(src, tgt)
    return alignment_rmse(apply_sim3(src, R, t, s), tgt), len(common)
