"""Sélection des volumes qui comptent : ceux qui peuvent masquer la cible.

Le manifeste retient comme obstacle tout bâtiment du voisinage cartographique.
C'est le bon critère pour une étude de visibilité, mais pas pour un rendu : sur
ce pilote, vingt-sept volumes sont ainsi retenus et **aucun** ne masque jamais
la cible. Ils occupent un peu plus d'un pour cent de l'image, doublent le coût
du rendu, et surtout encombrent la scène de constructions dont le générateur
n'a que faire.

Le tri se fait sur la seule question qui vaille : **ce volume peut-il, depuis
un point quelconque de la trajectoire, se glisser entre la caméra et la
cible ?** Un volume qui ne le peut pas est retiré de la géométrie — non
supprimé du manifeste, où il reste un fait cartographique.

**Ce que retirer un voisin ne fait pas.** Il ne vide pas l'arrière-plan : ce
qu'on voit derrière un bâtiment, une image de référence le montre mieux qu'un
prisme gris, et le paquet de conditionnement transporte ces images. La
géométrie ne sert qu'à contraindre parallaxe et occultation ; ce qui n'occulte
rien n'a rien à y contraindre.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..logging import get_logger

log = get_logger("conditioning-occlusion")

#: Marge angulaire, en degrés, ajoutée au secteur occupé par la cible. Elle
#: couvre l'imprécision des emprises et le fait qu'une orbite n'est pas un
#: cercle parfait : un volume qui frôle le secteur est gardé.
ANGULAR_MARGIN_DEG = 12.0

#: Marge de distance, en mètres. Un volume légèrement plus loin que la cible
#: peut encore dépasser derrière elle et compter dans la silhouette.
DEPTH_MARGIN_M = 15.0


@dataclass
class OcclusionVerdict:
    """Ce qu'un volume peut masquer, et pourquoi il est gardé ou non."""

    feature_id: str
    kept: bool
    reason: str
    min_distance_m: float = 0.0
    #: Part de la trajectoire, en degrés, depuis laquelle il s'interpose.
    blocking_arc_deg: float = 0.0

    def as_dict(self) -> dict:
        return {
            "feature_id": self.feature_id,
            "kept": self.kept,
            "reason": self.reason,
            "min_distance_m": round(self.min_distance_m, 1),
            "blocking_arc_deg": round(self.blocking_arc_deg, 1),
        }


@dataclass
class OcclusionReport:
    verdicts: list[OcclusionVerdict] = field(default_factory=list)
    orbit_radius_m: float = 0.0

    @property
    def kept(self) -> list[str]:
        return [v.feature_id for v in self.verdicts if v.kept]

    @property
    def dropped(self) -> list[str]:
        return [v.feature_id for v in self.verdicts if not v.kept]

    def as_dict(self) -> dict:
        return {
            "kept_count": len(self.kept),
            "dropped_count": len(self.dropped),
            "orbit_radius_m": round(self.orbit_radius_m, 1),
            "verdicts": [v.as_dict() for v in self.verdicts],
            "caveats": [
                "un volume retiré reste au manifeste : il n'est pas nié, il "
                "ne contraint simplement aucune occultation",
                "l'arrière-plan vient des images de référence, non de la "
                "géométrie — un prisme gris ne le décrit pas mieux",
            ],
        }


def _angular_span(footprint: np.ndarray, centre: tuple[float, float]) -> tuple[float, float]:
    """Secteur angulaire occupé par une emprise, vu du centre de la scène.

    Rendu comme (milieu, demi-largeur) en degrés. Les angles sont recentrés
    autour du premier sommet avant d'être moyennés : sans cela, une emprise à
    cheval sur le nord verrait ses angles s'annuler entre 1° et 359°.
    """
    dx = footprint[:, 0] - centre[0]
    dy = footprint[:, 1] - centre[1]
    angles = np.degrees(np.arctan2(dy, dx))
    pivot = angles[0]
    folded = (angles - pivot + 180.0) % 360.0 - 180.0
    middle = (pivot + folded.mean()) % 360.0
    # Un secteur centré sur l'est donne un angle d'un cheveu négatif, que le
    # modulo renvoie à 360° — le même azimut, mais illisible dans un rapport.
    if middle > 360.0 - 1e-6:
        middle = 0.0
    return float(middle), float((folded.max() - folded.min()) * 0.5)


def select(
    scene,  # noqa: ANN001
    orbit_radius_m: float | None = None,
    angular_margin_deg: float = ANGULAR_MARGIN_DEG,
    depth_margin_m: float = DEPTH_MARGIN_M,
) -> OcclusionReport:
    """Retire de la scène les volumes qui ne peuvent jamais masquer la cible.

    Un volume s'interpose s'il est **plus proche du centre que la caméra** et
    s'il partage un secteur angulaire avec la cible. Au-delà du rayon de
    l'orbite, il est derrière la caméra pour ce secteur ; hors du secteur, la
    cible n'est pas dans son axe.
    """
    target = scene.target
    report = OcclusionReport()
    if target is None:
        return report

    centre = scene.centre
    if orbit_radius_m is None:
        # Le cadrage de `render_sequence` : rayon de la cible, facteur 1,35.
        orbit_radius_m = max(scene.radius_m() * 1.35, scene.radius_m() + 10.0)
    report.orbit_radius_m = float(orbit_radius_m)

    # La cible entoure le centre de la scène : son « secteur vu du centre »
    # n'a pas de sens — il vaut presque 360°. Ce qui compte est son rayon, car
    # la caméra orbite autour d'elle et la vise depuis toutes les directions.
    target_reach = float(
        np.hypot(
            target.footprint[:, 0] - centre[0],
            target.footprint[:, 1] - centre[1],
        ).max()
    )
    reach = orbit_radius_m + depth_margin_m

    keep: list = []
    for prism in scene.prisms:
        if prism.is_target:
            keep.append(prism)
            report.verdicts.append(
                OcclusionVerdict(prism.feature_id, True, "bâtiment cible")
            )
            continue

        distance = float(
            np.hypot(
                prism.footprint[:, 0] - centre[0],
                prism.footprint[:, 1] - centre[1],
            ).min()
        )
        if distance > reach:
            report.verdicts.append(
                OcclusionVerdict(
                    prism.feature_id,
                    False,
                    (
                        f"à {distance:.0f} m du centre, au-delà de l'orbite "
                        f"({reach:.0f} m) : il est derrière la caméra, "
                        "jamais entre elle et la cible"
                    ),
                    min_distance_m=distance,
                )
            )
            continue

        # Un volume s'interpose s'il pénètre le disque que la caméra balaie
        # en visant la cible : entre le bord de la cible et l'orbite. Il n'y a
        # pas de secteur à comparer — l'orbite couvre tous les azimuts, et un
        # volume dans cet anneau finit forcément devant la cible à un moment.
        _middle, half = _angular_span(prism.footprint, centre)
        # Arc de l'orbite depuis lequel il masque : sa largeur apparente vue
        # du centre, élargie de la marge.
        blocking = min(2.0 * half + angular_margin_deg, 360.0)
        keep.append(prism)
        report.verdicts.append(
            OcclusionVerdict(
                prism.feature_id,
                True,
                (
                    f"à {distance:.0f} m, entre le bord de la cible "
                    f"({target_reach:.0f} m) et l'orbite ({reach:.0f} m) : "
                    "il peut s'interposer"
                ),
                min_distance_m=distance,
                blocking_arc_deg=blocking,
            )
        )

    scene.prisms = keep
    log.info(
        "occultation : %d volume(s) gardé(s) sur %d",
        len(report.kept),
        len(report.verdicts),
    )
    return report


__all__ = [
    "ANGULAR_MARGIN_DEG",
    "DEPTH_MARGIN_M",
    "OcclusionReport",
    "OcclusionVerdict",
    "select",
]
