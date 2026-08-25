"""Rectifier les images dans le plan d'un mur, et voir ce qu'on a vraiment.

Le pipeline mesure la couverture en cellules et en fractions. Il ne montre pas
que deux images du même mur **se superposent** — or c'est la seule preuve que
les poses sont mutuellement compatibles. Une carte de couverture peut être
excellente sur des poses fausses ; une orthofaçade, non : les structures y
apparaissent doubles.

Le principe est celui des relevés de façade : chaque image est ramenée dans le
plan métrique du mur par une homographie, et les images ainsi rectifiées se
comparent pixel à pixel. Une paire dominée par une homographie — faible pour la
profondeur, et que `view_graph` refuse à juste titre de poser — devient ici
pleinement utile : elle confirme le plan observé.

**Ce que l'orthofaçade est.** Une mosaïque probatoire, où chaque texel porte ce
qui l'atteste : combien d'images le voient, laquelle domine, avec quelle
incidence, et si elles s'accordent. C'est le `SupportType` du dépôt appliqué à
la texture.

**Ce qu'elle n'est pas.** Ni une reconstruction, ni une texture de production.
Le plan de façade vient de l'extrusion d'une emprise — c'est un proxy mesuré au
sol, non un mur relevé. Un décrochement réel s'y projettera de travers, et le
désaccord entre images le dira plutôt que de le lisser.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..logging import get_logger

log = get_logger("geo-orthofacade")

#: Taille d'un texel, en mètres. Cinq centimètres suffisent à lire une fenêtre
#: sans produire des atlas que rien dans le corpus ne justifie.
TEXEL_M = 0.05

#: Résolution minimale, en pixels par mètre, qu'une vue doit offrir sur le mur.
#: En deçà, la rectifier revient à agrandir du bruit : mesuré sur ce pilote,
#: un mur de dix-huit mètres vu à cent quatre-vingt-quinze mètres n'occupe que
#: trente-cinq pixels, et la mosaïque produite ne montre que du ciel étiré.
#:
#: Deux pixels par mètre est le seuil en deçà duquel une fenêtre — un mètre de
#: large — ne peut plus être distinguée d'un mur plein.
#:
#: Ce seuil n'est pas calibré : sur ce pilote il n'écarte aucune vue, alors que
#: la mosaïque produite reste illisible. Le corriger demande de comparer
#: plusieurs sites, non de le remonter au jugé — un seuil réglé pour rendre un
#: résultat acceptable ici échouerait ailleurs. Voir `benchmark`.
MIN_PIXELS_PER_M = 2.0

#: Incidence maximale, en degrés depuis la normale du mur, au-delà de laquelle
#: une image n'est pas retenue : quelques pixels y décrivent plusieurs mètres.
MAX_INCIDENCE_DEG = 65.0

#: Écart type des couleurs, sur 255, au-delà duquel les images qui voient un
#: texel sont tenues pour en désaccord. Un désaccord franc signale une pose
#: fausse, un objet mobile, ou un décrochement que le plan proxy ignore.
DISAGREEMENT_LEVEL = 42.0


@dataclass
class FacadePlane:
    """Le plan métrique d'un mur, et le repère où l'orthofaçade se construit."""

    facade_id: str
    #: Point du mur au niveau du sol, en CRS projeté.
    origin: np.ndarray
    #: Direction horizontale le long du mur, unitaire.
    along: np.ndarray
    #: Normale extérieure horizontale, unitaire.
    normal: np.ndarray
    length_m: float
    height_m: float

    def point(self, u: float, v: float) -> np.ndarray:
        """Point 3D à l'abscisse `u` le long du mur, à la hauteur `v`."""
        return self.origin + self.along * u + np.array([0.0, 0.0, v])

    @property
    def normal_deg(self) -> float:
        return math.degrees(math.atan2(self.normal[1], self.normal[0])) % 360.0


@dataclass
class TexelSupport:
    """Ce qui atteste un texel, et ce qui le met en doute."""

    contributing: int = 0
    best_asset: str | None = None
    best_incidence_deg: float | None = None
    best_distance_m: float | None = None
    disagreement: float = 0.0

    @property
    def status(self) -> str:
        if self.contributing == 0:
            return "non_observe"
        if self.disagreement >= DISAGREEMENT_LEVEL:
            return "desaccord"
        if self.contributing == 1:
            return "vue_unique"
        return "accorde"


@dataclass
class Orthofacade:
    """Une façade rectifiée, et la provenance de chacun de ses texels."""

    facade_id: str
    width_px: int
    height_px: int
    #: Mosaïque RGB, ou `None` si aucune image n'a contribué.
    image: np.ndarray | None = None
    support: list[TexelSupport] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    def by_status(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for texel in self.support:
            counts[texel.status] = counts.get(texel.status, 0) + 1
        return counts

    @property
    def observed_fraction(self) -> float:
        if not self.support:
            return 0.0
        return sum(1 for t in self.support if t.contributing) / len(self.support)

    def as_dict(self) -> dict:
        counts = self.by_status()
        return {
            "facade_id": self.facade_id,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "texel_m": TEXEL_M,
            "observed_fraction": round(self.observed_fraction, 3),
            "by_status": counts,
            "disagreement_fraction": round(
                counts.get("desaccord", 0) / max(len(self.support), 1), 3
            ),
            "provenance": self.provenance,
            "caveats": [
                "le plan vient de l'extrusion d'une emprise, non d'un mur "
                "relevé : un décrochement réel s'y projette de travers",
                "un désaccord entre images signale une pose fausse, un objet "
                "mobile ou un décrochement — il ne dit pas lequel",
                "une mosaïque probatoire n'est pas une texture de production",
            ],
        }


def plane_from_edge(
    start: np.ndarray, end: np.ndarray, height_m: float, facade_id: str = "FACADE"
) -> FacadePlane:
    """Construit le plan d'un mur depuis une arête d'emprise et une hauteur."""
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    delta = end[:2] - start[:2]
    length = float(np.hypot(*delta))
    along = np.array([delta[0] / length, delta[1] / length, 0.0])
    # La normale extérieure est choisie par l'appelant : ici on prend celle
    # qui tourne à droite de la direction du mur, et le sens du contour décide.
    normal = np.array([along[1], -along[0], 0.0])
    return FacadePlane(
        facade_id=facade_id,
        origin=np.array([start[0], start[1], float(start[2]) if len(start) > 2 else 0.0]),
        along=along,
        normal=normal,
        length_m=length,
        height_m=float(height_m),
    )


def rectify(
    plane: FacadePlane,
    views,  # noqa: ANN001 - (asset_id, image, camera)
    texel_m: float = TEXEL_M,
) -> Orthofacade:
    """Projette chaque vue dans le plan du mur et fusionne les contributions.

    Le texel retient la contribution de **meilleure incidence** : c'est celle
    dont les pixels décrivent le moins de mètres. Les autres ne sont pas
    jetées — elles servent à mesurer le désaccord, qui dit si les poses sont
    compatibles.
    """
    cols = max(int(round(plane.length_m / texel_m)), 1)
    rows = max(int(round(plane.height_m / texel_m)), 1)
    found = Orthofacade(facade_id=plane.facade_id, width_px=cols, height_px=rows)
    found.support = [TexelSupport() for _ in range(rows * cols)]

    if not views:
        found.provenance = {"views": 0, "reason": "aucune vue fournie"}
        return found

    canvas = np.zeros((rows, cols, 3), dtype=np.float64)
    best_incidence = np.full((rows, cols), np.inf)
    samples: dict[int, list] = {}
    used = 0
    skipped_resolution = 0

    us = (np.arange(cols) + 0.5) * texel_m
    vs = (np.arange(rows) + 0.5) * texel_m

    for view in views:
        asset_id, image, camera = view[:3]
        visibility_mask = view[3] if len(view) >= 4 else None
        # Incidence : l'angle entre la normale du mur et la direction de vue.
        # Elle ne dépend pas du texel, à cette distance.
        centre = plane.point(plane.length_m * 0.5, plane.height_m * 0.5)
        to_camera = np.asarray(camera.position, dtype=np.float64) - centre
        span = float(np.linalg.norm(to_camera))
        if span < 1e-6:
            continue
        cosine = float(np.dot(plane.normal, to_camera / span))
        incidence = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        if incidence > MAX_INCIDENCE_DEG:
            continue

        # Résolution offerte sur le mur : c'est elle qui décide si rectifier a
        # un sens. Une vue lointaine passe l'incidence et n'apporte pourtant
        # que quelques pixels pour des mètres de façade.
        focal = getattr(camera, "f", None)
        if focal:
            pixels_per_m = float(focal) / max(span, 1e-6)
            if pixels_per_m < MIN_PIXELS_PER_M:
                skipped_resolution += 1
                continue

        used += 1
        for row, v in enumerate(vs):
            points = np.array([plane.point(u, v) for u in us])
            screen, depth = camera.project(points)
            if screen is None:
                continue
            for col in range(cols):
                x, y = screen[col]
                if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
                    continue
                if depth is not None and depth[col] <= 0.5:
                    continue
                if visibility_mask is not None and not visibility_mask[int(y), int(x)]:
                    continue
                colour = image[int(y), int(x)].astype(np.float64)
                slot = row * cols + col
                samples.setdefault(slot, []).append(colour)
                texel = found.support[slot]
                texel.contributing += 1
                if incidence < (texel.best_incidence_deg or math.inf):
                    texel.best_incidence_deg = incidence
                    texel.best_distance_m = span
                    texel.best_asset = asset_id
                if incidence < best_incidence[row, col]:
                    best_incidence[row, col] = incidence
                    canvas[row, col] = colour

    # Le désaccord se mesure sur les texels vus par plusieurs images : c'est
    # là, et seulement là, que la compatibilité des poses est testable.
    for slot, colours in samples.items():
        if len(colours) < 2:
            continue
        found.support[slot].disagreement = float(
            np.mean(np.std(np.asarray(colours), axis=0))
        )

    found.image = canvas.astype(np.uint8) if used else None
    found.provenance = {
        "views_supplied": len(views),
        "views_used": used,
        "views_too_far": skipped_resolution,
        "texel_m": texel_m,
        "min_pixels_per_m": MIN_PIXELS_PER_M,
        "max_incidence_deg": MAX_INCIDENCE_DEG,
        "disagreement_level": DISAGREEMENT_LEVEL,
    }
    log.info(
        "%s : %d vue(s) rectifiée(s), %.0f%% du mur observé",
        plane.facade_id,
        used,
        100 * found.observed_fraction,
    )
    return found


__all__ = [
    "DISAGREEMENT_LEVEL",
    "MAX_INCIDENCE_DEG",
    "MIN_PIXELS_PER_M",
    "TEXEL_M",
    "FacadePlane",
    "Orthofacade",
    "TexelSupport",
    "plane_from_edge",
    "rectify",
]
