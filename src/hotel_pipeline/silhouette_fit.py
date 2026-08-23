"""Accord entre la silhouette prédite et les contours de l'image (Lot 2).

`pose_refine` sait chercher une position à partir de **bornes** lues dans
l'image. Restait à les produire, et aucune lecture générique n'y parvenait sur
le corpus réel : le masque de ciel donne une ligne de toit propre mais aucune
borne latérale — le bâtiment se fond dans les arbres au même niveau —, et un
détecteur de brique par teinte répond aux voitures rouges et aux toitures
voisines. Mesuré sur un recadrage du pilote, il retenait les colonnes 0 à 575
là où le bâtiment occupe 150 à 520.

Ce module renverse la question. On ne cherche pas « où est le bâtiment ? » —
question difficile et sans réponse fiable ici. On demande : **quelle position
de caméra fait coïncider la silhouette du modèle mesuré avec les contours de
l'image ?** La forme attendue est connue, exactement ; seule la pose manque.

C'est le même retournement qui a débloqué la reconstruction : la géométrie est
déjà mesurée — empreinte cadastrale, hauteur LiDAR — et n'a pas à être
redécouverte dans des photographies à 90 m. Ne reste qu'à l'y poser.

Le signal : la frontière du ciel, non « un contour »
----------------------------------------------------
La ligne de faîtage est l'arête la plus fiable d'un bâtiment vu de loin : elle
se découpe sur le ciel, là où les arêtes verticales se perdent dans la
végétation et les véhicules.

Mais il ne suffit pas de la rapprocher de *contours*. Mesuré sur un recadrage
du pilote, la carte de contours est fortement structurée en bandes :

```text
bande v      distance médiane au contour
  0-240        66 à 152 px   (ciel, vide)
320-480         1 à 3 px     (sol, voitures, clôture, végétation)
```

En éloignant la caméra, le faîtage projeté **descend** — v moyen 256 à 0 m,
284 à 80 m — et glisse dans la bande saturée du bas. Il n'y trouve pas le toit,
il y trouve du bruit, et le coût s'effondre sans qu'aucun alignement
s'améliore. L'optimum fuyait alors jusqu'au bord du domaine, quel qu'il soit :

```text
rayon exploré    10 m   30 m   60 m   80 m
déplacement      10,0   29,7   58,5   79,1     ← toujours le bord
```

Normaliser par l'étendue de la silhouette n'y suffit pas : le biais est
vertical, non d'échelle.

La correction est une **contrainte structurelle**. On ne compare pas le
faîtage à n'importe quel contour, mais à la frontière basse de la plus grande
région de ciel — la seule courbe de l'image qui soit, par construction, une
ligne de toit. Le faîtage ne peut alors plus se réfugier dans le bruit du sol,
et l'optimum se stabilise :

```text
rayon exploré    30 m   45 m   60 m   80 m
déplacement      29,7   44,4   44,4   44,4     ← minimum réel
```

Mesuré sur le recadrage `99h_64f` du photosphère du pilote, en balayant autour
de la position déclarée :

```text
          dy=-10   dy=0   dy=+10  dy=+20
dx=-20     15,5    13,1     7,2    16,1
dx=-10     20,1    16,0    13,1    19,2
dx=  0     24,3    20,8    24,8    20,6
dx=+10     27,8    32,6    29,5    28,9
```

Le minimum tombe vers (−20, +10) m, cohérent avec le (−12, +12) obtenu
indépendamment par lecture visuelle des bornes. Deux méthodes, même verdict :
la position déclarée de ce photosphère est fausse d'une dizaine de mètres.

Le biais d'éloignement, et sa correction
----------------------------------------
Une distance moyenne aux contours est **structurellement biaisée** : plus la
caméra s'éloigne, plus la silhouette rétrécit, plus ses points se concentrent,
et plus ils tombent près d'un contour par simple compacité. Mesuré sur le
pilote — le faîtage passe de 427 px de large à 191 px pour 80 m de recul, à
nombre de points constant — et l'optimum fuyait alors jusqu'au bord du domaine
de recherche :

```text
rayon exploré    coût     déplacement retenu
     10 m      12,55 px        10,0 m
     20 m       7,83 px        19,7 m
     30 m       5,49 px        29,7 m
     60 m       2,75 px        58,5 m
     80 m       1,58 px        79,1 m
```

Aucun de ces minima n'est réel : chacun est le bord du domaine. Le contraste
au voisinage ne le détectait pas — il valait 0,75, excellent — parce que le
biais agit *aussi* sur le voisinage.

La correction est de normaliser par l'étendue de la silhouette : on compare
des distances rapportées à la taille apparente du bâtiment, non des pixels
bruts. Un coût de 5 px sur une silhouette de 400 px de large n'est pas
comparable au même 5 px sur 190 px.

Hauteur et distance sont couplées
---------------------------------
Rabaisser le toit et reculer la caméra produisent la même image. La vallée de
coût est donc **diagonale**, et une base angulaire même large ne la redresse
pas — deux cadrages d'un même panorama n'offrent aucune parallaxe. Mesuré sur
le pilote, à 42° de base :

```text
hauteur    déplacement radial  →  coût
  6,5 m          −10 m              11,1
  7,5 m            0 m               9,2
  8,5 m          +10 m               7,7
 10,3 m          +20 m               6,9
 13,0 m          +30 m              11,5
```

Conséquence pratique, et c'est la leçon la plus coûteuse de ce module : un
raffinement de position lancé avec une hauteur fausse ne corrige pas la
position — **il compense la hauteur en déplaçant la caméra**. Sur le pilote,
un panorama véhiculé (position fiable) semblait dériver de 58 m ; c'était le
modèle qui était trop haut de 4,75 m. Avec la hauteur recalée, son coût
d'origine passe de 43,0 à 10,0 px et le déplacement retenu tombe à 18 m — dont
une part reste une compensation résiduelle, non une dérive avérée.

D'où la discipline : **estimer la hauteur d'abord, sur des positions
attestées, puis raffiner les positions.** Jamais l'inverse, et jamais les deux
en même temps sur une seule vue.

Ce que la skyline ne fixe pas : la profondeur
---------------------------------------------
Reculer le long de l'axe de visée ne change presque pas la ligne de toit d'un
bâtiment à toiture plate. Mesuré en test : le coût vaut 4,42 à 27 m derrière
la vérité contre 4,81 à la vérité même — 8 % d'écart, sous le bruit. Le
minimum est donc **réel mais plat en profondeur**.

Une vue unique ne suffit donc pas, et ce module ne prétend pas y suffire :
c'est `pose_refine` et ses vues à caps écartés qui lèvent l'indétermination.
Ce qui est garanti ici est la stabilité — élargir le domaine exploré ne
déplace plus la solution.

Ce que le coût ne dit pas
-------------------------
Un coût faible signifie « le faîtage tombe près de contours », non « le
faîtage tombe sur *ce* toit ». Une clôture, un fil électrique ou une ligne de
toiture voisine attirent la mesure aussi bien que la bonne. Le module rend donc
le coût **et** la marge qui le sépare du fond, et refuse de conclure quand
l'optimum ne se détache pas — un minimum plat est un minimum non identifié.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .logging import get_logger

log = get_logger("silhouette-fit")

#: Seuils de Canny. Bas, pour ne pas perdre une arête de toit contre un ciel
#: clair ; le bruit qui en résulte est absorbé par la moyenne.
CANNY_LOW = 40
CANNY_HIGH = 120

#: Points échantillonnés par arête du faîtage.
SAMPLES_PER_EDGE = 60

#: Bornes de teinte, saturation et valeur du ciel en HSV OpenCV.
SKY_HUE_MIN = 90
SKY_HUE_MAX = 130
SKY_MIN_SATURATION = 40
SKY_MIN_VALUE = 110

#: Bornes de recherche de hauteur, en mètres. Un optimum atteignant l'une
#: d'elles n'est pas retenu : la recherche n'a pas convergé.
MIN_HEIGHT_M = 4.0
MAX_HEIGHT_M = 16.0
HEIGHT_STEP_M = 0.25

#: Un vote de hauteur doit être bien recalé et porter sur assez d'image.
MAX_VOTE_COST_PX = 20.0
MIN_VOTE_COLUMNS = 100

#: Votes indépendants minimaux, et dispersion au-delà de laquelle ils ne
#: décrivent pas le même bâtiment.
MIN_HEIGHT_VOTES = 2
MAX_HEIGHT_SPREAD_M = 3.0

#: Amplitude maximale d'une frontière de ciel tenue pour une ligne de toit.
#: Au-delà, elle longe une arête verticale — un bâtiment de premier plan, un
#: mur proche — et non une toiture.
MAX_SKYLINE_AMPLITUDE_PX = 260.0

#: Écart au-delà duquel la frontière de ciel ne décrit manifestement pas ce
#: toit — un arbre au premier plan, un bâtiment voisin plus haut. Comptés
#: comme aberrants, ils pénalisent sans dominer la moyenne.
MAX_SKYLINE_GAP_PX = 80.0

#: Étendue de référence, en pixels : le coût est exprimé comme s'il portait
#: sur une silhouette de cette largeur. Purement une échelle — elle rend les
#: coûts lisibles en pixels sans changer l'ordre des candidats.
REFERENCE_EXTENT_PX = 400.0

#: Nombre minimal de points du faîtage tombant dans le cadre pour que la
#: mesure ait un sens. En deçà, on note un fragment, non une silhouette.
MIN_VISIBLE_POINTS = 20

#: Marge relative minimale entre le meilleur coût et le coût médian du
#: voisinage exploré. Un optimum qui ne se détache pas du fond n'identifie
#: rien : l'accepter donnerait une position arbitraire avec l'autorité d'une
#: mesure.
MIN_CONTRAST = 0.25


@dataclass
class FitResult:
    """Accord mesuré entre une silhouette prédite et une image."""

    cost_px: float | None = None
    visible_points: int = 0
    #: Coût médian du voisinage exploré, quand un balayage a eu lieu.
    background_px: float | None = None
    status: str = "not_measured"
    reason: str | None = None

    @property
    def contrast(self) -> float | None:
        """Part du fond que l'optimum retranche. 0 = indiscernable du fond."""
        if self.cost_px is None or not self.background_px:
            return None
        return max(0.0, 1.0 - self.cost_px / self.background_px)

    @property
    def identified(self) -> bool:
        return self.status == "fitted"

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "cost_px": round(self.cost_px, 2) if self.cost_px is not None else None,
            "background_px": (
                round(self.background_px, 2) if self.background_px is not None else None
            ),
            "contrast": round(self.contrast, 3) if self.contrast is not None else None,
            "visible_points": self.visible_points,
        }


def edge_distance_map(image):  # noqa: ANN001
    """Distance de chaque pixel au contour le plus proche.

    Une transformée de distance plutôt qu'un simple comptage de contours : elle
    rend le coût **continu**, donc optimisable. Compter les superpositions
    exactes donnerait une fonction en escalier, plate partout sauf sur des
    coïncidences ponctuelles, où aucune descente ne progresse.
    """
    import cv2

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    edges = cv2.Canny(cv2.GaussianBlur(grey, (5, 5), 0), CANNY_LOW, CANNY_HIGH)
    return cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)


def project_roofline(
    camera: tuple[float, float],
    footprint_coords: list[tuple[float, float]],
    ridge_height_m: float,
    heading_deg: float,
    fov_deg: float,
    *,
    width_px: int = 640,
    height_px: int = 640,
    pitch_deg: float = 0.0,
    camera_height_m: float = 2.5,
    samples_per_edge: int = SAMPLES_PER_EDGE,
) -> list[tuple[float, float]]:
    """Points du faîtage projetés, bornés au cadre."""
    focal = (width_px / 2.0) / math.tan(math.radians(fov_deg / 2.0))
    heading = math.radians(heading_deg)
    pitch = math.radians(pitch_deg)

    fx = math.sin(heading) * math.cos(pitch)
    fy = math.cos(heading) * math.cos(pitch)
    fz = math.sin(pitch)
    norm = math.hypot(fx, fy)
    if norm < 1e-9:
        return []
    rx, ry = fy / norm, -fx / norm
    # Vertical image = droite ∧ avant, pour rester orthonormé sous tangage.
    ux, uy, uz = (
        ry * fz - 0.0 * fy,
        0.0 * fx - rx * fz,
        rx * fy - ry * fx,
    )

    points: list[tuple[float, float]] = []
    count = len(footprint_coords)
    for index in range(count):
        ax, ay = footprint_coords[index]
        bx, by = footprint_coords[(index + 1) % count]
        for step in range(samples_per_edge + 1):
            ratio = step / samples_per_edge
            px = ax + (bx - ax) * ratio
            py = ay + (by - ay) * ratio
            dx, dy, dz = px - camera[0], py - camera[1], ridge_height_m - camera_height_m
            depth = dx * fx + dy * fy + dz * fz
            if depth <= 0.1:
                continue
            u = focal * (dx * rx + dy * ry) / depth + width_px / 2.0
            v = focal * -(dx * ux + dy * uy + dz * uz) / depth + height_px / 2.0
            if 0.0 <= u < width_px and 0.0 <= v < height_px:
                points.append((u, v))
    return points


def skyline(image, min_columns: int = 64):  # noqa: ANN001
    """Frontière basse de la plus grande région de ciel, colonne par colonne.

    C'est la seule courbe de l'image qui soit par construction une ligne de
    toit : tout ce qui est sous elle est du bâti, de la végétation ou du sol.
    Comparer le faîtage à *elle* plutôt qu'à des contours quelconques est ce
    qui empêche l'optimum de fuir dans le bruit du bas de l'image.

    La plus grande composante connexe seule est retenue : un morceau de ciel
    aperçu entre deux arbres n'est pas l'horizon, et le suivre déplacerait la
    ligne de plusieurs centaines de pixels.

    Returns:
        Un tableau de hauteurs par colonne, `NaN` là où aucun ciel n'est vu,
        ou `None` si l'image n'en montre pas assez pour conclure.
    """
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(int)
    saturation = hsv[:, :, 1].astype(int)
    value = hsv[:, :, 2].astype(int)
    mask = (
        (hue > SKY_HUE_MIN) & (hue < SKY_HUE_MAX)
        & (saturation > SKY_MIN_SATURATION) & (value > SKY_MIN_VALUE)
    ).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == largest).astype(np.uint8)

    height, width = mask.shape
    horizon = np.full(width, np.nan)
    for column in range(width):
        rows = np.where(mask[:, column] == 1)[0]
        if len(rows):
            horizon[column] = float(rows[-1])

    import math as _math

    defined = int(np.sum([not _math.isnan(v) for v in horizon]))
    if defined < min_columns:
        return None
    return horizon


def skyline_is_plausible(
    horizon,  # noqa: ANN001
    predicted_rows: list[float],
) -> tuple[bool, str | None]:
    """La frontière de ciel peut-elle être la ligne de toit cherchée ?

    Un score de prominence élevé ne garantit rien ici : CLIP répond « bâtiment
    d'hôtel » à une concession automobile occupant la moitié du cadre, et la
    frontière de ciel y suit l'arête **verticale** de ce bâtiment de premier
    plan. Mesuré sur le pilote — une vue notée 0,94 dont la skyline oscillait
    entre v=59 et v=469, soit 410 px d'amplitude, là où un toit vu à 176 m
    tient dans quelques dizaines.

    Deux garde-fous, tous deux sur la forme :

    - **amplitude** : une ligne de toit est presque horizontale à distance ;
    - **accord d'échelle** : sa dispersion doit ressembler à celle que le
      modèle prédit, sinon la courbe décrit autre chose.
    """
    import math as _math

    values = [v for v in horizon if not _math.isnan(v)]
    if len(values) < MIN_VISIBLE_POINTS:
        return False, "frontière de ciel trop fragmentaire"

    amplitude = max(values) - min(values)
    if amplitude > MAX_SKYLINE_AMPLITUDE_PX:
        return False, (
            f"amplitude {amplitude:.0f} px : la frontière de ciel suit une "
            "arête verticale (bâtiment de premier plan), non une ligne de toit"
        )

    if predicted_rows:
        predicted_amplitude = max(predicted_rows) - min(predicted_rows)
        # Une skyline dix fois plus accidentée que la silhouette attendue ne
        # décrit pas le même objet.
        if amplitude > max(60.0, predicted_amplitude * 6.0):
            return False, (
                f"amplitude {amplitude:.0f} px contre {predicted_amplitude:.0f} "
                "px attendus : la courbe ne décrit pas ce toit"
            )
    return True, None


def skyline_cost(
    horizon,  # noqa: ANN001
    camera: tuple[float, float],
    footprint_coords: list[tuple[float, float]],
    ridge_height_m: float,
    heading_deg: float,
    fov_deg: float,
    **kwargs,
) -> tuple[float | None, int]:
    """Écart moyen entre le faîtage projeté et la frontière du ciel.

    Pour chaque colonne, seule l'arête **supérieure** du faîtage compte : c'est
    elle qui borde le ciel, tandis que les arêtes plus basses appartiennent aux
    murs arrière et n'ont pas à coïncider avec l'horizon.
    """
    import math as _math

    width_px = len(horizon)
    points = project_roofline(
        camera, footprint_coords, ridge_height_m, heading_deg, fov_deg,
        width_px=width_px, **kwargs,
    )
    if len(points) < MIN_VISIBLE_POINTS:
        return None, len(points)

    upper: dict[int, float] = {}
    for u, v in points:
        column = int(u)
        if column not in upper or v < upper[column]:
            upper[column] = v

    # Seules les colonnes où le modèle prédit du bâtiment sont comparées. Là
    # où il n'en prédit pas, la frontière de ciel décrit le sol, un arbre ou le
    # bord du cadre : l'y confronter mesurerait le décor, non le bâtiment.
    # Constaté en test : sans ce filtre, des colonnes latérales vides tiraient
    # l'écart de 5 à 72 px alors que la pose était exacte.
    deviations: list[float] = []
    outliers = 0
    for column, v in upper.items():
        if not (0 <= column < width_px) or _math.isnan(horizon[column]):
            continue
        gap = abs(v - horizon[column])
        # Un écart énorme signale que la frontière de ciel décrit autre chose
        # que ce toit — un arbre devant, un bâtiment voisin. On le compte
        # comme aberrant plutôt que de le laisser dominer la moyenne.
        if gap > MAX_SKYLINE_GAP_PX:
            outliers += 1
            continue
        deviations.append(gap)

    if len(deviations) < MIN_VISIBLE_POINTS:
        return None, len(deviations)
    # Les aberrants ne disparaissent pas sans trace : ils pénalisent, sinon
    # une pose qui n'aligne que trois colonnes paraîtrait parfaite.
    penalty = MAX_SKYLINE_GAP_PX * outliers / (len(deviations) + outliers)
    return sum(deviations) / len(deviations) + penalty, len(deviations)


def fit_cost(
    distance_map,  # noqa: ANN001
    camera: tuple[float, float],
    footprint_coords: list[tuple[float, float]],
    ridge_height_m: float,
    heading_deg: float,
    fov_deg: float,
    **kwargs,
) -> tuple[float | None, int]:
    """Distance moyenne du faîtage projeté aux contours de l'image."""
    height_px, width_px = distance_map.shape[:2]
    points = project_roofline(
        camera, footprint_coords, ridge_height_m, heading_deg, fov_deg,
        width_px=width_px, height_px=height_px, **kwargs,
    )
    if len(points) < MIN_VISIBLE_POINTS:
        return None, len(points)

    total = 0.0
    for u, v in points:
        total += float(distance_map[int(v), int(u)])
    mean = total / len(points)

    # Normalisation par l'étendue apparente : sans elle, s'éloigner rétrécit
    # la silhouette, concentre ses points et fait chuter la distance moyenne
    # sans qu'aucun alignement ne s'améliore. L'optimum fuyait alors jusqu'au
    # bord du domaine de recherche.
    columns = [u for u, _ in points]
    rows = [v for _, v in points]
    extent = max(
        1.0, math.hypot(max(columns) - min(columns), max(rows) - min(rows))
    )
    return mean / extent * REFERENCE_EXTENT_PX, len(points)


def measure(
    distance_map,  # noqa: ANN001
    camera: tuple[float, float],
    footprint_coords: list[tuple[float, float]],
    ridge_height_m: float,
    heading_deg: float,
    fov_deg: float,
    *,
    background_radius_m: float = 25.0,
    background_samples: int = 5,
    **kwargs,
) -> FitResult:
    """Mesure l'accord, et dit s'il se détache du fond.

    Le fond est estimé en déplaçant la caméra autour de la position testée :
    si le coût y est comparable, la silhouette n'identifie aucune position en
    particulier et le résultat ne vaut rien.
    """
    cost, visible = fit_cost(
        distance_map, camera, footprint_coords, ridge_height_m,
        heading_deg, fov_deg, **kwargs,
    )
    if cost is None:
        return FitResult(
            visible_points=visible, status="not_visible",
            reason=(
                f"{visible} point(s) du faîtage dans le cadre, "
                f"minimum {MIN_VISIBLE_POINTS}"
            ),
        )

    neighbours: list[float] = []
    for index in range(background_samples):
        angle = 2.0 * math.pi * index / background_samples
        probe = (
            camera[0] + background_radius_m * math.cos(angle),
            camera[1] + background_radius_m * math.sin(angle),
        )
        probe_cost, _ = fit_cost(
            distance_map, probe, footprint_coords, ridge_height_m,
            heading_deg, fov_deg, **kwargs,
        )
        if probe_cost is not None:
            neighbours.append(probe_cost)

    background = (
        sorted(neighbours)[len(neighbours) // 2] if neighbours else None
    )
    result = FitResult(
        cost_px=cost, visible_points=visible, background_px=background,
        status="fitted",
    )
    if result.contrast is not None and result.contrast < MIN_CONTRAST:
        result.status = "ambiguous"
        result.reason = (
            f"contraste {result.contrast:.2f} sous {MIN_CONTRAST} : l'optimum "
            "ne se détache pas du voisinage et n'identifie pas de position"
        )
    return result


@dataclass
class HeightEstimate:
    """Hauteur de bâti lue sur les images, et ce qui la soutient."""

    height_m: float | None = None
    votes: int = 0
    spread_m: float | None = None
    status: str = "not_measured"
    reason: str | None = None
    #: `(hauteur, coût, colonnes, identifiant de vue)` de chaque vote retenu.
    detail: list[tuple[float, float, int, str]] = field(default_factory=list)

    @property
    def measured(self) -> bool:
        return self.status == "measured"

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "height_m": round(self.height_m, 2) if self.height_m is not None else None,
            "votes": self.votes,
            "spread_m": round(self.spread_m, 2) if self.spread_m is not None else None,
            "detail": [
                {
                    "view": view,
                    "height_m": round(height, 2),
                    "cost_px": round(cost, 1),
                    "columns": columns,
                }
                for height, cost, columns, view in self.detail
            ],
        }


def estimate_height(
    observations,  # noqa: ANN001 — [(horizon, camera, heading, fov, label)]
    footprint_coords: list[tuple[float, float]],
    *,
    minimum_m: float = MIN_HEIGHT_M,
    maximum_m: float = MAX_HEIGHT_M,
    step_m: float = HEIGHT_STEP_M,
    max_cost_px: float = MAX_VOTE_COST_PX,
    min_columns: int = MIN_VOTE_COLUMNS,
    **kwargs,
) -> HeightEstimate:
    """Hauteur de bâti qui explique le mieux les lignes de toit observées.

    **À n'employer que sur des positions attestées.** Hauteur et distance
    radiale sont indiscernables depuis un point de vue unique : reculer la
    caméra et rabaisser le toit produisent la même image. Mesuré sur le pilote,
    la vallée de coût est diagonale — (6,5 m ; −10 m), (7,5 m ; 0 m),
    (8,5 m ; +10 m), (10,3 m ; +20 m) coûtent tous entre 7 et 11 px, et 42° de
    base angulaire n'y changent rien, deux cadrages d'un même panorama
    n'offrant aucune parallaxe. Seule une position connue lève l'ambiguïté.

    Chaque vue vote séparément ; les votes butant sur une borne, trop coûteux
    ou portant sur trop peu de colonnes sont écartés. Un optimum au bord du
    domaine n'est pas un optimum : c'est une recherche qui n'a pas convergé.
    """
    import statistics

    votes: list[tuple[float, float, int, str]] = []
    rejected: list[str] = []

    for horizon, camera, heading_deg, fov_deg, label in observations:
        if horizon is None:
            rejected.append(f"{label}: pas de frontière de ciel")
            continue

        best: tuple[float, float, int] | None = None
        height = minimum_m
        while height <= maximum_m + 1e-9:
            cost, columns = skyline_cost(
                horizon, camera, footprint_coords, height,
                heading_deg, fov_deg, **kwargs,
            )
            if cost is not None and (best is None or cost < best[1]):
                best = (height, cost, columns)
            height += step_m

        if best is None:
            rejected.append(f"{label}: aucune hauteur mesurable")
            continue

        height, cost, columns = best
        if height <= minimum_m + 1e-9 or height >= maximum_m - 1e-9:
            rejected.append(
                f"{label}: optimum au bord du domaine ({height:.1f} m), non convergé"
            )
            continue
        if cost > max_cost_px:
            rejected.append(f"{label}: coût {cost:.0f} px trop élevé")
            continue
        if columns < min_columns:
            rejected.append(f"{label}: {columns} colonnes comparées, trop peu")
            continue
        votes.append((height, cost, columns, label))

    if len(votes) < MIN_HEIGHT_VOTES:
        return HeightEstimate(
            votes=len(votes), status="insufficient_votes", detail=votes,
            reason=(
                f"{len(votes)} vue(s) exploitable(s) sur {len(observations)}, "
                f"minimum {MIN_HEIGHT_VOTES} — " + " ; ".join(rejected[:4])
            ),
        )

    heights = [h for h, _, _, _ in votes]
    spread = max(heights) - min(heights)
    estimate = HeightEstimate(
        height_m=statistics.median(heights), votes=len(votes),
        spread_m=spread, status="measured", detail=votes,
    )
    if spread > MAX_HEIGHT_SPREAD_M:
        estimate.status = "inconsistent"
        estimate.reason = (
            f"les vues divergent de {spread:.1f} m : aucune hauteur unique "
            "ne les explique"
        )
    log.info(
        "hauteur estimée : %.2f m sur %d vue(s), dispersion %.2f m",
        estimate.height_m, estimate.votes, spread,
    )
    return estimate


def search(
    distance_map,  # noqa: ANN001
    origin: tuple[float, float],
    footprint_coords: list[tuple[float, float]],
    ridge_height_m: float,
    views: list[tuple[float, float]],
    *,
    radius_m: float = 30.0,
    step_m: float = 2.0,
    **kwargs,
) -> tuple[tuple[float, float], float, float]:
    """Position minimisant le coût cumulé sur plusieurs cadrages.

    `views` porte les `(cap, champ)` des recadrages disponibles. Comme dans
    `pose_refine`, plusieurs vues à caps écartés valent mieux qu'une : une vue
    unique laisse position et cap indiscernables.

    Returns:
        `(position, coût, coût médian exploré)` — le troisième terme sert à
        juger si l'optimum se détache.
    """
    best_camera = origin
    best_cost = float("inf")
    explored: list[float] = []

    span = int(radius_m / step_m)
    for i in range(-span, span + 1):
        for j in range(-span, span + 1):
            candidate = (origin[0] + i * step_m, origin[1] + j * step_m)
            if math.hypot(candidate[0] - origin[0], candidate[1] - origin[1]) > radius_m:
                continue
            total = 0.0
            usable = 0
            for heading_deg, fov_deg in views:
                cost, _ = fit_cost(
                    distance_map if not isinstance(distance_map, dict)
                    else distance_map[(heading_deg, fov_deg)],
                    candidate, footprint_coords, ridge_height_m,
                    heading_deg, fov_deg, **kwargs,
                )
                if cost is not None:
                    total += cost
                    usable += 1
            if usable == len(views) and usable > 0:
                mean = total / usable
                explored.append(mean)
                if mean < best_cost:
                    best_cost, best_camera = mean, candidate

    background = sorted(explored)[len(explored) // 2] if explored else float("nan")
    log.info(
        "recherche silhouette : coût %.1f px (fond %.1f px) à %.1f m de l'origine",
        best_cost, background,
        math.hypot(best_camera[0] - origin[0], best_camera[1] - origin[1]),
    )
    return best_camera, best_cost, background


__all__ = [
    "CANNY_HIGH",
    "CANNY_LOW",
    "HEIGHT_STEP_M",
    "HeightEstimate",
    "MAX_HEIGHT_M",
    "MAX_HEIGHT_SPREAD_M",
    "MAX_SKYLINE_AMPLITUDE_PX",
    "MAX_VOTE_COST_PX",
    "MIN_HEIGHT_M",
    "MIN_HEIGHT_VOTES",
    "MIN_VOTE_COLUMNS",
    "estimate_height",
    "MAX_SKYLINE_GAP_PX",
    "skyline_is_plausible",
    "REFERENCE_EXTENT_PX",
    "SKY_HUE_MAX",
    "SKY_HUE_MIN",
    "SKY_MIN_SATURATION",
    "SKY_MIN_VALUE",
    "skyline",
    "skyline_cost",
    "MIN_CONTRAST",
    "MIN_VISIBLE_POINTS",
    "SAMPLES_PER_EDGE",
    "FitResult",
    "edge_distance_map",
    "fit_cost",
    "measure",
    "project_roofline",
    "search",
]
