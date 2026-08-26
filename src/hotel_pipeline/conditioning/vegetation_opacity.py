"""Opacité de la végétation : trois classes, jamais un mur vert.

Un houppier n'est pas une cloison. La classe d'opacité — opaque,
semi_transparent, uncertain — et sa transmittance sont la seule monnaie
d'échange acceptée partout : le rendu dégrade le crédit des pixels masqués
au lieu de les éteindre, la visibilité pondère l'occlusion au lieu de la
décréter totale.
"""

from __future__ import annotations

#: Transmittance par classe d'opacité. Une couronne semi-transparente laisse
#: passer plus qu'elle ne bloque : une façade derrière un arbre reste
#: partiellement visible, ce qu'un volume opaque interdisait.
TRANSMITTANCE_BY_CLASS: dict[str, float] = {
    "opaque": 0.05,
    "semi_transparent": 0.45,
    "uncertain": 0.70,
}

#: Classes fermées : tout consommateur doit tomber dans l'une d'elles.
OPACITY_CLASSES = frozenset(TRANSMITTANCE_BY_CLASS)


def occlusion_fraction(opacity_class: str) -> float:
    """Fraction d'occlusion d'un massif : 1 − transmittance."""
    if opacity_class not in OPACITY_CLASSES:
        raise ValueError(
            f"classe d'opacité inconnue : {opacity_class!r} — "
            f"attendues : {sorted(OPACITY_CLASSES)}"
        )
    return round(1.0 - TRANSMITTANCE_BY_CLASS[opacity_class], 3)


def weighted_visibility(base_visibility: float, opacity_class: str) -> float:
    """Visibilité résiduelle derrière un massif de la classe donnée.

    Une couronne d'arbre ne rend pas automatiquement 100 % d'une façade
    invisible : la visibilité est multipliée par la transmittance.
    """
    return round(max(0.0, min(1.0, base_visibility)) * TRANSMITTANCE_BY_CLASS[opacity_class], 4)
