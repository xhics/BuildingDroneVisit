"""Normalisation photométrique inter-vues avant la fusion de texture.

Problème 39 — différences d'exposition et de balance des blancs entre
photos créent des coutures dans l'atlas. Avant la fusion robuste, chaque
vue est alignée photométriquement sur une vue de référence sur les zones
de recouvrement : gain/bias par canal, estimés de façon robuste (médiane
des ratios puis médiane des résidus), bornés pour rester une correction
*légère* — jamais une réinterprétation des couleurs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Texels communs minimum entre deux vues avant d'estimer un modèle.
MIN_OVERLAP_TEXELS = 32

#: Résidu médian maximal toléré **après** correction : au-delà, le modèle
#: écraserait un désaccord réel (contenu différent, pas exposition) et il
#: n'est pas appliqué.
MAX_RESIDUAL_RGB = 24.0

GAIN_RANGE = (0.5, 2.0)
BIAS_RANGE = (-64.0, 64.0)


@dataclass(frozen=True)
class GainBias:
    """Transformation affine légère par canal : out = gain * in + bias."""

    gain: tuple[float, float, float]
    bias: tuple[float, float, float]

    def apply(self, colour: np.ndarray) -> np.ndarray:
        values = np.asarray(colour, dtype=np.float64)
        return np.clip(
            values * np.asarray(self.gain, dtype=np.float64)
            + np.asarray(self.bias, dtype=np.float64),
            0.0,
            255.0,
        )

    def is_identity(self) -> bool:
        return (
            tuple(self.gain) == (1.0, 1.0, 1.0)
            and tuple(self.bias) == (0.0, 0.0, 0.0)
        )

    def as_dict(self) -> dict:
        return {
            "gain": [round(float(g), 6) for g in self.gain],
            "bias": [round(float(b), 4) for b in self.bias],
        }


def estimate_gain_bias(reference: np.ndarray, other: np.ndarray) -> GainBias | None:
    """Gain/bias par canal ramenant ``other`` vers ``reference``.

    Robuste : médiane des ratios pour le gain, médiane des résidus pour le
    bias. Retourne ``None`` si le recouvrement est vide ou dégénéré, ou si
    le résidu après correction dépasse :data:`MAX_RESIDUAL_RGB` — signe que
    les deux zones diffèrent par leur contenu et non par leur exposition.
    """
    reference = np.asarray(reference, dtype=np.float64).reshape(-1, 3)
    other = np.asarray(other, dtype=np.float64).reshape(-1, 3)
    if reference.shape[0] == 0:
        return None
    gain = np.ones(3, dtype=np.float64)
    bias = np.zeros(3, dtype=np.float64)
    for channel in range(3):
        ref_ch = reference[:, channel]
        other_ch = other[:, channel]
        usable = np.abs(other_ch) > 8.0
        if int(usable.sum()) >= 1:
            ratios = ref_ch[usable] / other_ch[usable]
            finite = ratios[np.isfinite(ratios)]
            if finite.size:
                gain[channel] = float(np.clip(np.median(finite), *GAIN_RANGE))
        bias[channel] = float(
            np.clip(np.median(ref_ch - gain[channel] * other_ch), *BIAS_RANGE)
        )
    corrected = np.clip(other * gain[None, :] + bias[None, :], 0.0, 255.0)
    residual = float(np.median(np.abs(reference - corrected)))
    if residual > MAX_RESIDUAL_RGB:
        return None
    return GainBias(tuple(float(g) for g in gain), tuple(float(b) for b in bias))


def fit_view_normalizations(
    samples_by_view: dict[int, dict[int, np.ndarray]],
) -> dict[int, GainBias]:
    """Aligne chaque vue sur la référence (la vue couvrant le plus de texels).

    ``samples_by_view`` associe à chaque vue un dict ``{texel_slot: couleur}``.
    La vue de référence n'est pas transformée ; les autres reçoivent un
    :class:`GainBias` estimé sur les texels communs.
    """
    if not samples_by_view:
        return {}
    reference_index = max(samples_by_view, key=lambda v: len(samples_by_view[v]))
    normalizations: dict[int, GainBias] = {
        reference_index: GainBias((1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    }
    reference_samples = samples_by_view[reference_index]
    for view_index, samples in sorted(samples_by_view.items()):
        if view_index == reference_index:
            continue
        shared = [
            (reference_samples[slot], colour)
            for slot, colour in samples.items()
            if slot in reference_samples
        ]
        if len(shared) < MIN_OVERLAP_TEXELS:
            continue
        reference_pixels = np.asarray([pair[0] for pair in shared], dtype=np.float64)
        other_pixels = np.asarray([pair[1] for pair in shared], dtype=np.float64)
        model = estimate_gain_bias(reference_pixels, other_pixels)
        if model is not None:
            normalizations[view_index] = model
    return normalizations


__all__ = [
    "BIAS_RANGE",
    "GAIN_RANGE",
    "MAX_RESIDUAL_RGB",
    "MIN_OVERLAP_TEXELS",
    "GainBias",
    "estimate_gain_bias",
    "fit_view_normalizations",
]
