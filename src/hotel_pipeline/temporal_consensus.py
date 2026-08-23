"""Séparer le permanent du saisonnier par la variation entre dates (Lot 2).

`permanence` sait ce qu'autorise chaque classe d'objet, mais quelqu'un doit
attribuer la classe. Sur le pilote, personne ne le fait : le manifeste de site
porte 14 objets, tous bâtis ou minéraux, et pas un seul végétal. Le jardin
existe sur les photographies et nulle part dans le modèle.

Le déclarer à la main ne passe pas à l'échelle — le pipeline doit s'appliquer à
un nouvel hôtel sans assistance. Ce module l'infère, en exploitant ce que le
corpus contient déjà : **plusieurs dates**.

Le principe
-----------
Ce qui se ressemble à toutes les dates est permanent ; ce qui change est
saisonnier. Une zone verte en juin, brune en avril et blanche en janvier est du
gazon ; une zone identique partout est de l'asphalte ou du bâti. **La variance
temporelle est le signal** — aucun classifieur botanique n'intervient, et c'est
justement pourquoi la méthode tient à 100 m de distance, là où une
identification d'espèce serait de la fiction.

Sur le pilote, le corpus acquis couvre quatre dates — août 2016, octobre 2023,
avril 2025, mai 2025 — soit trois saisons, dont deux vues à moins de 80 m.

Ce que le module exige
----------------------
La comparaison n'a de sens qu'entre vues **du même endroit** : deux
photographies prises de points différents cadrent des choses différentes, et
leur écart mesure le déplacement, non la saison. On compare donc par cellule
d'un maillage au sol, chaque cellule n'étant retenue que si plusieurs dates
l'observent.

Ce que la méthode hérite des poses
----------------------------------
Une cellule est placée dans l'image par la **pose** de la vue. Une pose fausse
déplace donc toutes ses cellules, et leur apparence est lue au mauvais endroit.
Sur le pilote, un photosphère dont la position dérive d'une quarantaine de
mètres projette ses cellules de pelouse sur le stationnement et jusque sur la
façade.

D'où l'enchaînement obligé : `panorama_provenance` d'abord — pour savoir quelles
poses sont attestées —, `silhouette_fit` ensuite pour la hauteur, et le
consensus au sol en dernier. Alimenter ce module avec des poses non attestées
produit une carte du sol convaincante et fausse.

Mesuré sur le pilote, l'effet du seul test d'occlusion :

```text
                observations   cellules   tranchées   saisonnières
sans occlusion       3564         786       30,9 %         24
avec occlusion       1989         637       33,9 %         12
```

Le filtre retire 44 % des observations et fait pourtant **monter** la part
tranchée : ce qu'il supprime empêchait de conclure plus qu'il n'aidait. La
moitié des cellules « saisonnières » étaient des façades vues à travers le
bâtiment.

Ce que le module ne fait pas
----------------------------
Il ne dit pas *ce qu'est* une zone saisonnière — massif, gazon, parterre. Il
dit qu'elle **change**, ce qui suffit à lui refuser un maillage 3D et à exiger
une palette par saison. Nommer la plante demanderait des pixels que la
distance ne donne pas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .logging import get_logger

log = get_logger("temporal-consensus")

#: Côté d'une cellule du maillage au sol, en mètres. Assez large pour qu'une
#: cellule soit vue par plusieurs panoramas, assez fin pour séparer une
#: plate-bande d'un stationnement.
CELL_SIZE_M = 5.0

#: Dates distinctes nécessaires pour qu'une cellule reçoive un verdict.
MIN_DATES = 2

#: Sous cette taille, une emprise de cellule ne porte pas assez de pixels pour
#: décrire quoi que ce soit. Une cellule de 5 m vue à 150 m avec un champ de
#: 25° couvre environ 20 × 20 px ; au-delà, elle devient illisible.
MIN_PATCH_PIXELS = 64

#: Saisons distinctes au-delà desquelles la variance devient interprétable
#: comme saisonnalité et non comme simple bruit d'éclairage.
MIN_SEASONS_FOR_SEASONALITY = 2


@dataclass
class CellObservation:
    """Ce qu'une vue dit d'une cellule au sol."""

    cell: tuple[int, int]
    date: str
    season: str | None
    #: Descripteur d'apparence, dans [0, 1] par composante : vert, neige,
    #: luminance. Trois nombres suffisent — on cherche un changement, pas une
    #: signature fine.
    green: float
    snow: float
    brightness: float

    def descriptor(self) -> tuple[float, float, float]:
        return (self.green, self.snow, self.brightness)


@dataclass
class CellVerdict:
    """Verdict de permanence pour une cellule au sol."""

    cell: tuple[int, int]
    dates: set[str] = field(default_factory=set)
    seasons: set[str] = field(default_factory=set)
    variance: float | None = None
    status: str = "unobserved"
    reason: str | None = None

    @property
    def decided(self) -> bool:
        return self.status in {"stable", "seasonal"}

    def as_dict(self) -> dict:
        return {
            "cell": list(self.cell),
            "status": self.status,
            "reason": self.reason,
            "variance": round(self.variance, 4) if self.variance is not None else None,
            "dates": sorted(self.dates),
            "seasons": sorted(self.seasons),
        }


def cell_of(x: float, y: float, *, size_m: float = CELL_SIZE_M) -> tuple[int, int]:
    """Cellule du maillage contenant un point projeté."""
    return (int(math.floor(x / size_m)), int(math.floor(y / size_m)))


def _spread(values: list[tuple[float, float, float]]) -> float:
    """Dispersion d'un jeu de descripteurs, dans [0, 1].

    Moyenne des étendues par composante plutôt qu'un écart-type : avec deux ou
    trois dates, l'écart-type est trop instable, et l'étendue dit directement
    « de combien cette cellule a changé au maximum ».
    """
    if len(values) < 2:
        return 0.0
    spans = []
    for index in range(3):
        column = [v[index] for v in values]
        spans.append(max(column) - min(column))
    return sum(spans) / len(spans)


def project_cell(
    cell: tuple[int, int],
    camera: tuple[float, float],
    heading_deg: float,
    fov_deg: float,
    *,
    size_m: float = CELL_SIZE_M,
    width_px: int = 640,
    height_px: int = 640,
    pitch_deg: float = 0.0,
    camera_height_m: float = 2.5,
) -> tuple[int, int, int, int] | None:
    """Emprise pixel d'une cellule au sol dans une vue donnée.

    La cellule est à l'altitude du sol (z = 0) ; c'est ce qui la distingue du
    faîtage projeté par `silhouette_fit`. Rend `None` si elle tombe derrière la
    caméra ou hors du cadre — une cellule non vue n'est pas une cellule vide.
    """
    focal = (width_px / 2.0) / math.tan(math.radians(fov_deg / 2.0))
    heading = math.radians(heading_deg)
    pitch = math.radians(pitch_deg)

    fx = math.sin(heading) * math.cos(pitch)
    fy = math.cos(heading) * math.cos(pitch)
    fz = math.sin(pitch)
    norm = math.hypot(fx, fy)
    if norm < 1e-9:
        return None
    rx, ry = fy / norm, -fx / norm
    ux = ry * fz
    uy = -rx * fz
    uz = rx * fy - ry * fx

    columns: list[float] = []
    rows: list[float] = []
    for corner_x in (cell[0], cell[0] + 1):
        for corner_y in (cell[1], cell[1] + 1):
            dx = corner_x * size_m - camera[0]
            dy = corner_y * size_m - camera[1]
            dz = -camera_height_m
            depth = dx * fx + dy * fy + dz * fz
            if depth <= 0.1:
                return None
            columns.append(focal * (dx * rx + dy * ry) / depth + width_px / 2.0)
            rows.append(focal * -(dx * ux + dy * uy + dz * uz) / depth + height_px / 2.0)

    left, right = int(min(columns)), int(math.ceil(max(columns)))
    top, bottom = int(min(rows)), int(math.ceil(max(rows)))
    if right < 0 or left >= width_px or bottom < 0 or top >= height_px:
        return None
    return (
        max(0, left), max(0, top),
        min(width_px, max(left + 1, right)), min(height_px, max(top + 1, bottom)),
    )


def cell_is_visible(
    cell: tuple[int, int],
    camera: tuple[float, float],
    occluders,  # noqa: ANN001 — géométries projetées bloquant la vue
    *,
    size_m: float = CELL_SIZE_M,
) -> bool:
    """La cellule est-elle vue, ou masquée par un volume interposé ?

    `project_cell` répond à « où tomberait cette cellule dans l'image », non à
    « la voit-on ». Sans ce test, une cellule située derrière le bâtiment se
    projette **sur sa façade**, et son apparence est décrite par des pixels de
    brique. Mesuré sur le pilote : 43 % des observations traversaient ainsi le
    bâtiment cible, décrivant du mur comme s'il s'agissait de sol.

    Le test est en 2D et volontairement grossier — un mur bloque la vue au sol
    quelle que soit sa hauteur. Il n'attrape pas les occlusions basses
    (haies, véhicules), qui restent une source de bruit assumée.
    """
    from shapely.geometry import LineString, Point

    centre = Point((cell[0] + 0.5) * size_m, (cell[1] + 0.5) * size_m)
    ray = LineString([camera, (centre.x, centre.y)])
    for occluder in occluders or ():
        if occluder is None:
            continue
        # `crosses` et non `intersects` : une cellule au bord de l'empreinte
        # touche le polygone sans être masquée par lui.
        if ray.crosses(occluder):
            return False
    return True


def sample_patch(image, box: tuple[int, int, int, int]) -> tuple[float, float, float] | None:  # noqa: ANN001
    """Descripteur d'apparence d'une zone d'image : vert, neige, luminance.

    Les seuils sont ceux de `seasonality`, pour que « vert » veuille dire la
    même chose partout. Une zone trop petite rend `None` : quelques pixels ne
    décrivent rien de fiable.
    """
    import cv2
    import numpy as np

    from .seasonality import (
        GREEN_HUE_MAX, GREEN_HUE_MIN, GREEN_MIN_SATURATION, GREEN_MIN_VALUE,
        SNOW_MAX_CHANNEL_SPREAD, SNOW_MAX_SATURATION, SNOW_MAX_TEXTURE,
        SNOW_MIN_VALUE,
    )

    left, top, right, bottom = box
    patch = image[top:bottom, left:right]
    if patch.size == 0 or patch.shape[0] < 2 or patch.shape[1] < 2:
        return None
    if patch.shape[0] * patch.shape[1] < MIN_PATCH_PIXELS:
        return None

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(int)
    saturation = hsv[:, :, 1].astype(int)
    value = hsv[:, :, 2].astype(int)
    blue = patch[:, :, 0].astype(int)
    green_channel = patch[:, :, 1].astype(int)
    red = patch[:, :, 2].astype(int)

    grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    deviation = float(grey.std())

    verdant = (
        (hue > GREEN_HUE_MIN) & (hue < GREEN_HUE_MAX)
        & (saturation > GREEN_MIN_SATURATION) & (value > GREEN_MIN_VALUE)
    )
    neutral = (
        (np.abs(red - green_channel) < SNOW_MAX_CHANNEL_SPREAD)
        & (np.abs(green_channel - blue) < SNOW_MAX_CHANNEL_SPREAD)
        & (np.abs(red - blue) < SNOW_MAX_CHANNEL_SPREAD)
    )
    snowy = neutral & (value > SNOW_MIN_VALUE) & (saturation < SNOW_MAX_SATURATION)
    snow_fraction = float(snowy.mean()) if deviation < SNOW_MAX_TEXTURE else 0.0

    return (
        float(verdant.mean()),
        snow_fraction,
        float(value.mean() / 255.0),
    )


def verdict_for(
    observations: list[CellObservation],
    *,
    stable_below: float,
    seasonal_above: float,
    min_dates: int = MIN_DATES,
) -> CellVerdict:
    """Verdict d'une cellule à partir de ses observations datées."""
    if not observations:
        return CellVerdict(cell=(0, 0), status="unobserved", reason="aucune observation")

    verdict = CellVerdict(cell=observations[0].cell)
    verdict.dates = {o.date for o in observations}
    verdict.seasons = {o.season for o in observations if o.season}

    if len(verdict.dates) < min_dates:
        verdict.status = "insufficient_dates"
        verdict.reason = (
            f"{len(verdict.dates)} date(s) : il en faut {min_dates} pour "
            "distinguer le permanent du saisonnier"
        )
        return verdict

    # Une seule observation par date : sinon deux vues du même jour, prises
    # sous des angles différents, gonfleraient la variance sans qu'aucune
    # saison ait changé.
    by_date: dict[str, list[tuple[float, float, float]]] = {}
    for observation in observations:
        by_date.setdefault(observation.date, []).append(observation.descriptor())
    per_date = [
        tuple(sum(c[i] for c in group) / len(group) for i in range(3))
        for group in by_date.values()
    ]

    verdict.variance = _spread(per_date)

    if verdict.variance < stable_below:
        verdict.status = "stable"
        verdict.reason = (
            f"apparence constante sur {len(verdict.dates)} dates "
            f"(variation {verdict.variance:.2f})"
        )
        return verdict

    if verdict.variance > seasonal_above:
        if len(verdict.seasons) < MIN_SEASONS_FOR_SEASONALITY:
            # Varier entre deux dates d'une même saison n'est pas de la
            # saisonnalité : c'est de l'éclairage, de l'ombre, ou un véhicule
            # garé là ce jour-là.
            verdict.status = "unstable_same_season"
            verdict.reason = (
                f"variation {verdict.variance:.2f} mais une seule saison "
                "observée : éclairage ou objet mobile, non saisonnalité"
            )
            return verdict
        verdict.status = "seasonal"
        verdict.reason = (
            f"apparence variable sur {len(verdict.seasons)} saisons "
            f"(variation {verdict.variance:.2f})"
        )
        return verdict

    verdict.status = "undecided"
    verdict.reason = (
        f"variation {verdict.variance:.2f} entre {stable_below} et "
        f"{seasonal_above} : la mesure ne tranche pas"
    )
    return verdict


def build(
    observations: list[CellObservation],
    *,
    stable_below: float | None = None,
    seasonal_above: float | None = None,
    min_dates: int = MIN_DATES,
) -> dict[tuple[int, int], CellVerdict]:
    """Verdict pour chaque cellule observée.

    Les seuils par défaut viennent de `permanence`, pour que les deux modules
    ne divergent pas : une cellule tenue pour stable ici doit l'être là-bas.
    """
    from .permanence import SEASONAL_ABOVE, STABLE_BELOW

    stable_below = STABLE_BELOW if stable_below is None else stable_below
    seasonal_above = SEASONAL_ABOVE if seasonal_above is None else seasonal_above

    grouped: dict[tuple[int, int], list[CellObservation]] = {}
    for observation in observations:
        grouped.setdefault(observation.cell, []).append(observation)

    verdicts = {
        cell: verdict_for(
            items, stable_below=stable_below,
            seasonal_above=seasonal_above, min_dates=min_dates,
        )
        for cell, items in grouped.items()
    }
    decided = sum(1 for v in verdicts.values() if v.decided)
    log.info(
        "consensus temporel : %d cellule(s), %d tranchée(s)",
        len(verdicts), decided,
    )
    return verdicts


def to_scene_objects(verdicts: dict[tuple[int, int], CellVerdict]) -> list:
    """Convertit les verdicts en objets de scène, avec leur permanence.

    Une cellule non tranchée ne devient **pas** un objet : produire une surface
    pour ce qu'on n'a pas su classer reviendrait à inventer du terrain.
    """
    from .permanence import Permanence, SceneObject

    objects: list = []
    for cell, verdict in sorted(verdicts.items()):
        if not verdict.decided:
            continue
        permanence = (
            Permanence.PERMANENT if verdict.status == "stable"
            else Permanence.SEASONAL_SURFACE
        )
        objects.append(
            SceneObject(
                object_id=f"CELL_{cell[0]}_{cell[1]}",
                kind=None,
                permanence=permanence,
                observed_dates=set(verdict.dates),
                observed_seasons=set(verdict.seasons),
                temporal_variance=verdict.variance,
                evidence=[verdict.reason] if verdict.reason else [],
            )
        )
    return objects


def summarise(verdicts: dict[tuple[int, int], CellVerdict]) -> dict:
    """Bilan par statut, et couverture réellement tranchée."""
    counts: dict[str, int] = {}
    for verdict in verdicts.values():
        counts[verdict.status] = counts.get(verdict.status, 0) + 1
    total = len(verdicts) or 1
    return {
        "cells": len(verdicts),
        "by_status": dict(sorted(counts.items())),
        "decided_fraction": round(
            sum(1 for v in verdicts.values() if v.decided) / total, 3
        ),
    }


__all__ = [
    "CELL_SIZE_M",
    "MIN_DATES",
    "MIN_SEASONS_FOR_SEASONALITY",
    "CellObservation",
    "CellVerdict",
    "build",
    "MIN_PATCH_PIXELS",
    "cell_is_visible",
    "cell_of",
    "project_cell",
    "sample_patch",
    "summarise",
    "to_scene_objects",
    "verdict_for",
]
