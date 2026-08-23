"""Identité visuelle du bâtiment : est-ce bien *cet* établissement ?

Le pipeline savait déjà reconnaître un bâtiment d'une piscine. Il ne savait pas
reconnaître **ce** bâtiment-ci du pavillon d'en face, et c'est ce qui a laissé
des maisons résidentielles dans un lot nommé « façade ».

Aucune règle sur les métadonnées ne pouvait l'attraper : la distinction est
purement visuelle. Elle est donc confiée à des modèles — similarité d'embedding
contre des ancres confirmées, lecture d'enseigne, attributs décrits en langage
naturel — et non à des seuils écrits à la main sur des champs du manifeste.
"""

from .embedding import EmbeddingIndex, ImageEmbedder
from .anchors import AnchorSet, load_anchors
from .verdict import IdentityVerdict, IdentityStatus, judge
from .screen import ScreeningResult, screen_assets

__all__ = [
    "AnchorSet",
    "EmbeddingIndex",
    "IdentityStatus",
    "IdentityVerdict",
    "ImageEmbedder",
    "ScreeningResult",
    "judge",
    "load_anchors",
    "screen_assets",
]
