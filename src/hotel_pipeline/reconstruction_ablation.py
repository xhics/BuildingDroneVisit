"""Mesurer l'effet d'un masquage plutôt que de le supposer bénéfique.

Masquer les objets mobiles avant l'extraction de features est une bonne
pratique : une voiture qui bouge entre deux passages contamine la géométrie,
et corriger après coup est trop tard. Mais sur un corpus pauvre, retirer des
pixels retire aussi des correspondances — et une voiture **stationnée** entre
deux passages du même véhicule de captation est un point fixe utile.

Le masquage ne doit donc pas être adopté sur principe. Ce module compare des
variantes sur ce qui décide réellement d'un solve :

- combien de paires restent valides ;
- combien de correspondances les soutiennent ;
- la taille de la composante connexe, seule à dire si le graphe tient.

**Ce que la comparaison ne dit pas.** Elle mesure l'appariement, non la
reconstruction : plus de correspondances ne garantit pas un meilleur solve, et
un masque peut améliorer la justesse des poses en réduisant leur nombre. La
composante connexe est le meilleur indicateur disponible sans lancer COLMAP,
pas une preuve.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .logging import get_logger

log = get_logger("reconstruction-ablation")

#: Part masquée d'une image au-delà de laquelle on refuse de l'utiliser. Un
#: masque couvrant l'essentiel du cadre ne protège plus la géométrie : il la
#: supprime.
MAX_MASKED_FRACTION = 0.75

#: Inliers minimaux pour tenir une paire pour valide. Aligné sur le seuil de
#: `view_graph`, pour que les deux comptages disent la même chose.
MIN_INLIERS = 8


@dataclass
class VariantResult:
    """Ce qu'une variante de masquage donne, mesuré et non supposé."""

    name: str
    pairs_attempted: int = 0
    pairs_valid: int = 0
    inliers_total: int = 0
    #: Taille de la plus grande composante connexe du graphe.
    largest_component: int = 0
    nodes: int = 0
    masked_fraction_mean: float = 0.0
    images_refused: int = 0

    @property
    def valid_fraction(self) -> float:
        return self.pairs_valid / max(self.pairs_attempted, 1)

    @property
    def connected_fraction(self) -> float:
        return self.largest_component / max(self.nodes, 1)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "pairs_attempted": self.pairs_attempted,
            "pairs_valid": self.pairs_valid,
            "valid_fraction": round(self.valid_fraction, 3),
            "inliers_total": self.inliers_total,
            "largest_component": self.largest_component,
            "nodes": self.nodes,
            "connected_fraction": round(self.connected_fraction, 3),
            "masked_fraction_mean": round(self.masked_fraction_mean, 3),
            "images_refused": self.images_refused,
        }


@dataclass
class AblationReport:
    """Les variantes comparées, et celle que la mesure désigne."""

    variants: list[VariantResult] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    def best(self) -> VariantResult | None:
        """Variante retenue : la connexité d'abord, les inliers pour départager.

        La connexité prime parce qu'un graphe fragmenté ne reconstruit rien,
        quel que soit le nombre de correspondances. À connexité égale, plus de
        correspondances vaut mieux.
        """
        if not self.variants:
            return None
        return max(
            self.variants,
            key=lambda v: (v.largest_component, v.inliers_total),
        )

    def as_dict(self) -> dict:
        chosen = self.best()
        baseline = next((v for v in self.variants if v.name == "sans_masque"), None)
        verdict = "aucune variante mesurée"
        if chosen is not None and baseline is not None:
            if chosen.name == baseline.name:
                verdict = (
                    "le masquage ne compense pas ce qu'il retire : "
                    "aucune variante ne dépasse l'absence de masque"
                )
            elif chosen.largest_component > baseline.largest_component:
                verdict = (
                    f"{chosen.name} élargit la composante connexe "
                    f"({baseline.largest_component} → {chosen.largest_component})"
                )
            else:
                verdict = (
                    f"{chosen.name} conserve la connexité et gagne "
                    f"{chosen.inliers_total - baseline.inliers_total} inlier(s)"
                )
        return {
            "variants": [v.as_dict() for v in self.variants],
            "chosen": chosen.name if chosen else None,
            "verdict": verdict,
            "provenance": self.provenance,
            "caveats": [
                "la mesure porte sur l'appariement, non sur la reconstruction : "
                "plus de correspondances ne garantit pas un meilleur solve",
                "un masque appliqué au rendu seulement arrive trop tard — la "
                "géométrie est déjà contaminée",
            ],
        }


def largest_component(nodes: list[str], edges: list[tuple[str, str]]) -> int:
    """Taille de la plus grande composante connexe, par union-find."""
    parent = {node: node for node in nodes}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for left, right in edges:
        if left not in parent or right not in parent:
            continue
        a, b = find(left), find(right)
        if a != b:
            parent[a] = b

    sizes: dict[str, int] = {}
    for node in nodes:
        root = find(node)
        sizes[root] = sizes.get(root, 0) + 1
    return max(sizes.values()) if sizes else 0


def apply_mask(image, mask, dilate_px: int = 0):  # noqa: ANN001
    """Applique un masque à une image, éventuellement dilaté.

    La dilatation traite les bords : la silhouette d'un objet mobile déborde
    de son masque, et les descripteurs y mêlent l'objet et son fond.
    """
    import cv2

    if mask is None:
        return image, 0.0
    working = mask
    if dilate_px > 0:
        kernel = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
        working = cv2.dilate(mask, kernel)

    covered = float((working > 0).mean())
    masked = image.copy()
    masked[working > 0] = 0
    return masked, covered


def compare(variants: dict, matcher, nodes: list[str]) -> AblationReport:  # noqa: ANN001
    """Compare des variantes de masquage sur le même corpus.

    `variants` associe un nom à une fonction rendant, pour chaque image, son
    masque — ou `None` pour la variante sans masque. `matcher` rend la liste
    des paires valides sous forme `(a, b, inliers)`.
    """
    report = AblationReport()
    for name, produce in variants.items():
        result = VariantResult(name=name, nodes=len(nodes))
        pairs = matcher(produce, result)
        result.pairs_attempted = max(result.pairs_attempted, len(pairs))
        valid = [(a, b) for a, b, inliers in pairs if inliers >= MIN_INLIERS]
        result.pairs_valid = len(valid)
        result.inliers_total = sum(
            inliers for _a, _b, inliers in pairs if inliers >= MIN_INLIERS
        )
        result.largest_component = largest_component(nodes, valid)
        report.variants.append(result)
        log.info(
            "%s : %d paire(s) valide(s), composante %d/%d",
            name,
            result.pairs_valid,
            result.largest_component,
            result.nodes,
        )
    return report


__all__ = [
    "MAX_MASKED_FRACTION",
    "MIN_INLIERS",
    "AblationReport",
    "VariantResult",
    "apply_mask",
    "compare",
    "largest_component",
]
