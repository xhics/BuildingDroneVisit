"""Raffinement de pose par recalage de silhouette (Lot 2).

`panorama_provenance` sépare les poses attestées de celles qui ne le sont pas.
Ce module s'occupe des secondes : il cherche la position de caméra qui fait
coïncider la **silhouette projetée du modèle mesuré** avec le bâtiment
réellement visible dans l'image.

Pourquoi pas un PnP classique
-----------------------------
`solvePnP` demande des correspondances point-à-point — « ce coin du modèle est
ce pixel ». Rien ici ne les fournit de façon fiable : détecter les coins d'un
bâtiment de brique à 90 m, partiellement masqué par des arbres et des voitures,
est précisément le problème que la mise en correspondance dense n'arrivait pas
à résoudre.

Le recalage de silhouette demande moins. Il ne compare pas des points mais des
**bornes** : jusqu'où s'étend le bâtiment à gauche, à droite, et où passe sa
ligne de toit. Ces bornes survivent aux occlusions partielles qui détruisent
l'appariement de points.

Ce qui contraint la solution
----------------------------
Une vue unique ne suffit pas : un décalage latéral de la caméra et une erreur
de cap produisent la même image, et rien dans un seul cadre ne les distingue.
Deux recadrages du même panorama à des caps différents lèvent l'ambiguïté —
mesuré sur le pilote, deux vues à 85,5° et 99,1° contraignent la position à
~15 px de résidu, révélant une dérive de 17 m sur un photosphère utilisateur.

D'où la règle : **un panorama à vue unique n'est pas raffiné.** Il reste
`needs_refinement`, ce qui est honnête, plutôt que de recevoir une position
ajustée sur une contrainte insuffisante.

Précision, et ce qu'elle suppose
--------------------------------
Mesurée sur 50 configurations synthétiques par niveau, en bruitant les bornes
lues (deux vues, base ≥ 8°, bâtiment de 70 × 40 m à ~120 m) :

```text
bruit de lecture   erreur de position   p90
      ±2 px              0,4 m         1,3 m
      ±5 px              1,1 m         3,2 m
     ±10 px              2,1 m         6,3 m
     ±20 px              4,3 m        12,3 m
```

Soit environ **0,2 m d'erreur de position par pixel d'erreur de lecture**. Une
dérive de 17 m se corrige donc à ~2 m près avec une lecture à ±10 px — et 2 m
à 90 m ne pèsent qu'une dizaine de pixels à la reprojection.

Attention à la limite : à ±20 px, l'erreur médiane atteint 4,3 m sans qu'aucun
cas soit déclaré non concluant. `MAX_RESIDUAL_PX` filtre les observations
*contradictoires*, pas les observations *imprécises* — un bruit symétrique
laisse un résidu faible tout en déplaçant l'optimum. La qualité de la lecture
de silhouette borne donc la précision, et rien ici ne la mesure à sa place.

Ce que le module ne fait pas
----------------------------
Il ne corrige pas le cap. Sur le pilote l'erreur mesurée était bien en
position — deux caps différents déviaient du même côté, signature d'une
translation. Ajouter le cap aux inconnues rendrait le problème sous-déterminé
avec deux vues seulement, et l'optimiseur compenserait une erreur par l'autre.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .logging import get_logger

log = get_logger("pose-refine")

#: Nombre minimal de vues distinctes d'un même panorama pour tenter un
#: raffinement. En deçà, position et cap sont indiscernables.
MIN_VIEWS = 2

#: Écart angulaire minimal entre deux vues pour qu'elles contraignent
#: réellement. Deux caps quasi identiques rendent la même image et
#: n'apportent qu'une seule équation.
MIN_BASELINE_DEG = 8.0

#: Rayon de recherche autour de la position déclarée, en mètres. Au-delà, il
#: ne s'agit plus d'une dérive de saisie mais d'un panorama mal apparié, que
#: le raffinement ne doit pas « sauver » en le téléportant.
SEARCH_RADIUS_M = 40.0

#: Pas final de la recherche, en mètres.
SEARCH_STEP_M = 1.0

#: Résidu au-delà duquel le raffinement est déclaré non concluant, en pixels.
#: Un optimum trouvé mais mauvais reste un échec : l'accepter donnerait une
#: fausse position avec l'autorité d'une mesure.
MAX_RESIDUAL_PX = 40.0


@dataclass
class SilhouetteObservation:
    """Bornes du bâtiment lues dans une image, et le cadrage qui l'a produite."""

    heading_deg: float
    fov_deg: float
    #: Colonnes extrêmes du bâtiment observé, en pixels.
    u_min: float
    u_max: float
    width_px: int = 640
    pitch_deg: float = 0.0
    #: Confiance de la lecture. Une borne touchant le bord du cadre est
    #: tronquée : le bâtiment continue hors champ, et la borne ne mesure
    #: alors que le cadre.
    left_truncated: bool = False
    right_truncated: bool = False

    @property
    def usable_bounds(self) -> int:
        """Combien de bornes portent réellement de l'information."""
        return int(not self.left_truncated) + int(not self.right_truncated)


@dataclass
class RefinedPose:
    """Résultat d'un raffinement, concluant ou non."""

    panorama_id: str
    #: Position d'origine, en CRS projeté.
    origin: tuple[float, float]
    refined: tuple[float, float] | None = None
    residual_px: float | None = None
    views_used: int = 0
    baseline_deg: float = 0.0
    status: str = "not_attempted"
    reason: str | None = None

    @property
    def shift_m(self) -> float | None:
        if self.refined is None:
            return None
        return math.hypot(
            self.refined[0] - self.origin[0], self.refined[1] - self.origin[1]
        )

    @property
    def converged(self) -> bool:
        return self.status == "refined"

    def as_dict(self) -> dict:
        return {
            "panorama_id": self.panorama_id,
            "status": self.status,
            "reason": self.reason,
            "views_used": self.views_used,
            "baseline_deg": round(self.baseline_deg, 1),
            "origin": [round(c, 2) for c in self.origin],
            "refined": (
                [round(c, 2) for c in self.refined] if self.refined else None
            ),
            "shift_m": round(self.shift_m, 1) if self.shift_m is not None else None,
            "residual_px": (
                round(self.residual_px, 1) if self.residual_px is not None else None
            ),
        }


def project_bounds(
    camera: tuple[float, float],
    vertices: list[tuple[float, float, float]],
    heading_deg: float,
    fov_deg: float,
    *,
    width_px: int = 640,
    pitch_deg: float = 0.0,
    camera_height_m: float = 2.5,
) -> tuple[float, float] | None:
    """Colonnes extrêmes qu'occuperait le modèle vu depuis `camera`.

    Rend `None` si aucun sommet n'est devant la caméra — la vue regarde
    ailleurs, et prétendre des bornes serait inventer.
    """
    focal = (width_px / 2.0) / math.tan(math.radians(fov_deg / 2.0))
    heading = math.radians(heading_deg)
    pitch = math.radians(pitch_deg)

    forward = (
        math.sin(heading) * math.cos(pitch),
        math.cos(heading) * math.cos(pitch),
        math.sin(pitch),
    )
    # Droite = avant ∧ vertical, renormalisée : à fort tangage la norme chute.
    right = (forward[1], -forward[0], 0.0)
    norm = math.hypot(right[0], right[1])
    if norm < 1e-9:
        return None
    right = (right[0] / norm, right[1] / norm, 0.0)

    columns: list[float] = []
    for x, y, z in vertices:
        dx, dy, dz = x - camera[0], y - camera[1], z - camera_height_m
        depth = dx * forward[0] + dy * forward[1] + dz * forward[2]
        if depth <= 0.1:
            continue
        lateral = dx * right[0] + dy * right[1]
        columns.append(focal * lateral / depth + width_px / 2.0)

    if not columns:
        return None
    return min(columns), max(columns)


def _residual(
    camera: tuple[float, float],
    vertices: list[tuple[float, float, float]],
    observations: list[SilhouetteObservation],
) -> float | None:
    """Écart quadratique moyen entre bornes prédites et bornes observées.

    Seules les bornes non tronquées comptent : une silhouette coupée par le
    bord du cadre mesure le cadre, non le bâtiment.
    """
    total = 0.0
    count = 0
    for observation in observations:
        predicted = project_bounds(
            camera, vertices, observation.heading_deg, observation.fov_deg,
            width_px=observation.width_px, pitch_deg=observation.pitch_deg,
        )
        if predicted is None:
            return None
        if not observation.left_truncated:
            total += (predicted[0] - observation.u_min) ** 2
            count += 1
        if not observation.right_truncated:
            total += (predicted[1] - observation.u_max) ** 2
            count += 1
    if count == 0:
        return None
    return math.sqrt(total / count)


def _baseline(observations: list[SilhouetteObservation]) -> float:
    """Étendue angulaire des caps observés — ce qui contraint la position."""
    if len(observations) < 2:
        return 0.0
    headings = [o.heading_deg for o in observations]
    spread = 0.0
    for i, a in enumerate(headings):
        for b in headings[i + 1:]:
            spread = max(spread, abs((a - b + 180.0) % 360.0 - 180.0))
    return spread


def refine(
    panorama_id: str,
    origin: tuple[float, float],
    vertices: list[tuple[float, float, float]],
    observations: list[SilhouetteObservation],
    *,
    search_radius_m: float = SEARCH_RADIUS_M,
    search_step_m: float = SEARCH_STEP_M,
) -> RefinedPose:
    """Cherche la position qui explique le mieux les silhouettes observées.

    Recherche par grille décroissante : le résidu n'est pas convexe — un
    bâtiment allongé produit des minima locaux le long de son axe — et une
    descente de gradient s'y piège. La grille reste abordable parce que
    l'espace est borné à ±40 m.
    """
    result = RefinedPose(
        panorama_id=panorama_id, origin=origin,
        views_used=len(observations), baseline_deg=_baseline(observations),
    )

    if len(observations) < MIN_VIEWS:
        result.status = "insufficient_views"
        result.reason = (
            f"{len(observations)} vue(s) : position et cap restent "
            "indiscernables sous " + str(MIN_VIEWS)
        )
        return result

    if result.baseline_deg < MIN_BASELINE_DEG:
        result.status = "insufficient_baseline"
        result.reason = (
            f"caps écartés de {result.baseline_deg:.1f}° seulement : les vues "
            "montrent la même chose et n'apportent qu'une contrainte"
        )
        return result

    if sum(o.usable_bounds for o in observations) < 3:
        result.status = "insufficient_bounds"
        result.reason = (
            "trop de bornes tronquées par le bord du cadre pour contraindre "
            "deux inconnues"
        )
        return result

    baseline_residual = _residual(origin, vertices, observations)

    best_camera = origin
    best_residual = baseline_residual if baseline_residual is not None else float("inf")

    # Grille décroissante : chaque passe explore un voisinage deux fois plus
    # fin autour du meilleur point trouvé. Le rayon local doit rester d'au
    # moins deux pas — le réduire davantage arrête la descente avant que le
    # pas fin ait servi, et la solution reste à quelques mètres de l'optimum.
    step = max(search_step_m, search_radius_m / 8.0)
    radius = search_radius_m
    while True:
        centre = best_camera
        span = max(1, int(round(radius / step)))
        for i in range(-span, span + 1):
            for j in range(-span, span + 1):
                candidate = (centre[0] + i * step, centre[1] + j * step)
                if math.hypot(
                    candidate[0] - origin[0], candidate[1] - origin[1]
                ) > search_radius_m:
                    continue
                residual = _residual(candidate, vertices, observations)
                if residual is not None and residual < best_residual:
                    best_residual, best_camera = residual, candidate
        if step <= search_step_m + 1e-9:
            break
        step = max(search_step_m, step / 2.0)
        radius = step * 2.0

    # La grille ne descend pas sous son pas : à 95 m avec un champ de 64°, un
    # mètre de caméra déplace la silhouette d'environ 5 px, et le résidu
    # plafonne là sans que la position soit fausse pour autant. Un polissage
    # continu, parti d'un optimum déjà localisé, retire ce plancher.
    polished = _polish(best_camera, vertices, observations, origin, search_radius_m)
    if polished is not None:
        candidate_residual = _residual(polished, vertices, observations)
        if candidate_residual is not None and candidate_residual < best_residual:
            best_camera, best_residual = polished, candidate_residual

    if best_residual > MAX_RESIDUAL_PX:
        result.status = "not_converged"
        result.residual_px = best_residual
        result.reason = (
            f"meilleur résidu {best_residual:.0f} px au-delà de "
            f"{MAX_RESIDUAL_PX:.0f} px : aucune position n'explique ces vues"
        )
        return result

    result.refined = best_camera
    result.residual_px = best_residual
    result.status = "refined"
    log.info(
        "%s : position déplacée de %.1f m, résidu %.0f px (%d vues, base %.0f°)",
        panorama_id, result.shift_m or 0.0, best_residual,
        result.views_used, result.baseline_deg,
    )
    return result


def _polish(
    start: tuple[float, float],
    vertices: list[tuple[float, float, float]],
    observations: list[SilhouetteObservation],
    origin: tuple[float, float],
    search_radius_m: float,
) -> tuple[float, float] | None:
    """Descente locale continue depuis un optimum déjà localisé par la grille.

    Nelder-Mead plutôt qu'un gradient : le résidu n'est pas dérivable partout —
    un sommet qui passe derrière la caméra le fait sauter — et la simplexe s'en
    accommode. Le résultat reste borné au rayon de recherche : le polissage
    affine, il ne relocalise pas.
    """
    try:
        from scipy.optimize import minimize
    except ImportError:  # scipy absent : la grille seule fait le travail
        return None

    def cost(vector) -> float:  # noqa: ANN001
        camera = (float(vector[0]), float(vector[1]))
        if math.hypot(
            camera[0] - origin[0], camera[1] - origin[1]
        ) > search_radius_m:
            return 1e9
        residual = _residual(camera, vertices, observations)
        return 1e9 if residual is None else residual

    outcome = minimize(
        cost, [start[0], start[1]], method="Nelder-Mead",
        options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 800},
    )
    if not getattr(outcome, "success", False):
        return None
    return (float(outcome.x[0]), float(outcome.x[1]))


def summarise(results: list[RefinedPose]) -> dict:
    """Bilan d'une campagne de raffinement."""
    by_status: dict[str, int] = {}
    for item in results:
        by_status[item.status] = by_status.get(item.status, 0) + 1
    shifts = [r.shift_m for r in results if r.shift_m is not None]
    return {
        "total": len(results),
        "by_status": by_status,
        "refined": by_status.get("refined", 0),
        "median_shift_m": (
            round(sorted(shifts)[len(shifts) // 2], 1) if shifts else None
        ),
    }


__all__ = [
    "MAX_RESIDUAL_PX",
    "MIN_BASELINE_DEG",
    "MIN_VIEWS",
    "SEARCH_RADIUS_M",
    "SEARCH_STEP_M",
    "RefinedPose",
    "SilhouetteObservation",
    "project_bounds",
    "refine",
    "summarise",
]
