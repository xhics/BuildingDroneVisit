"""Conditionnement géométrique d'un générateur vidéo.

Le paquet ne génère aucune vidéo. Il produit, le long d'une trajectoire, les
cartes que le générateur consomme — profondeur, normales, silhouette, masque de
confiance — et il dit, frame par frame, ce que la géométrie atteste réellement.

L'invariant du dépôt tient ici aussi : une surface sans preuve n'est pas
peinte d'une valeur plausible. Elle sort en `unknown` dans le masque de
confiance, et le générateur reçoit l'autorisation explicite d'y improviser.
"""

from .scene import ConditioningScene, Prism, load_scene
from .render import RenderedFrame, render_frame
from .sequence import SequenceResult, render_sequence

__all__ = [
    "ConditioningScene",
    "Prism",
    "load_scene",
    "RenderedFrame",
    "render_frame",
    "SequenceResult",
    "render_sequence",
]
