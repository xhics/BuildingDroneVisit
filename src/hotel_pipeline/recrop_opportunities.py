"""Cadrages récupérables depuis les panoramas déjà acquis (Lot 2).

Deux questions étaient confondues, et la confusion a coûté deux façades :

- **« cette vue visait-elle la façade ? »** — propriété du cadrage que *nous*
  avons demandé ;
- **« cette façade est-elle visible depuis cette position ? »** — propriété du
  lieu, indépendante de tout cadrage.

Pour une sphère Street View, la seconde seule compte : le panorama existe en
entier, et `resolve_url` sait en extraire n'importe quel cap depuis le même
`pano_id`, sans nouvelle découverte. Le cap stocké est un artefact de notre
propre requête, non une propriété de l'imagerie.

Filtrer sur `heading_is_measured` écartait donc 124 panoramas du pilote et
rendait `FACADE_RIGHT` et `FACADE_REAR` « jamais vues ». Mesurée depuis les
positions seules, la réalité est autre :

```text
façade            strict   position   segments observables
FACADE_PRIMARY        20        157   44/44
FACADE_LEFT           17        110   33/33
FACADE_RIGHT           0         91   22/22
FACADE_REAR            0         62   11/11
```

Second piège, distinct : un seuil de distance unique servait deux usages
incompatibles. 150 m est raisonnable pour une **texture** ; c'est absurde pour
la **géométrie**, alors que l'arrière d'un bâtiment de 72 × 77 m se voit
nécessairement de plus loin que sa façade avant. D'où deux portées déclarées.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

#: Portée pour la **géométrie** : au-delà, la parallaxe reste exploitable même
#: si le détail ne l'est plus. L'arrière d'un bâtiment se voit de loin.
GEOMETRY_RANGE_M = 400.0

#: Portée pour l'**apparence** : au-delà, un mur n'occupe plus assez de pixels
#: pour porter une texture. C'est l'ancien seuil unique, rendu à son seul
#: usage légitime.
TEXTURE_RANGE_M = 150.0

#: Champ demandé, en multiples de l'angle réellement sous-tendu par le sujet.
#:
#: **Calibré**, non choisi. Une sphère a trois degrés de liberté — cap, champ,
#: inclinaison — et le balayage n'en faisait varier qu'un, demandant toujours
#: 70°. À 150 m, un bâtiment de 60 m sous-tend 22,6° : un cadre de 70° est
#: alors à 96 % du stationnement, et la façade arrière du pilote était rejetée
#: à 0,396 alors qu'elle est parfaitement lisible.
#:
#: Mesuré sur ce même panorama :
#:
#: ```text
#: champ  ×sous-tendu  prominence
#:   15°       0,66       0,295   trop serré : la silhouette sort du cadre
#:   20°       0,88       0,996
#:   25°       1,11       0,997   <- optimum
#:   40°       1,77       0,907
#:   70°       3,09       0,396   contexte, non sujet
#: ```
FOV_MARGIN = 1.1

#: Bornes de l'API Street View. Demander hors de cette plage échoue.
FOV_MIN_DEG = 12.0
FOV_MAX_DEG = 110.0

#: Largeur supposée du sujet quand rien ne la mesure, en mètres.
DEFAULT_SUBJECT_WIDTH_M = 60.0


def fov_for(distance_m: float, subject_width_m: float = DEFAULT_SUBJECT_WIDTH_M) -> float:
    """Champ à demander pour qu'un sujet remplisse le cadre sans le déborder.

    Le sujet sous-tend `2·atan(w / 2d)`. On demande un peu plus, pour garder
    la silhouette et un peu de contexte — trop serré, la lecture pixel perd le
    bâtiment autant que trop large.
    """
    import math

    if distance_m <= 0 or subject_width_m <= 0:
        return FOV_MAX_DEG
    subtended = 2.0 * math.degrees(math.atan(subject_width_m / (2.0 * distance_m)))
    return float(min(FOV_MAX_DEG, max(FOV_MIN_DEG, subtended * FOV_MARGIN)))


def pitch_for(distance_m: float, subject_height_m: float = 12.0) -> float:
    """Inclinaison à demander pour centrer un bâtiment de plain-pied.

    À distance, le milieu d'un bâtiment est au-dessus de l'horizon de la
    caméra. Demander `pitch=0` cadre le stationnement ; quelques degrés
    suffisent à recentrer la façade.
    """
    import math

    if distance_m <= 0:
        return 0.0
    # Centre du bâtiment à mi-hauteur, caméra à ~2,5 m du sol.
    rise = (subject_height_m / 2.0) - 2.5
    return float(max(0.0, min(20.0, math.degrees(math.atan(rise / distance_m)))))


@dataclass
class RecropOpportunity:
    """Un cadrage à demander, sur un panorama déjà acquis."""

    panorama_id: str
    asset_id: str
    facade_id: str
    #: Cap à demander, en degrés — dirigé vers les points réellement visibles.
    heading_deg: float
    distance_m: float
    #: Indices des points de mur que ce cadrage montrerait.
    covers: list[int] = field(default_factory=list)
    #: Portée qui a permis de le retenir : `geometry` ou `texture`.
    purpose: str = "geometry"
    #: Champ et inclinaison à demander — dérivés de la distance, non fixes.
    fov_deg: float = 70.0
    pitch_deg: float = 0.0
    #: Prominence **lue sur les pixels** d'un recadrage déjà acquis, si connue.
    #: `None` = jamais vérifié ; ce n'est pas un score nul.
    verified_prominence: float | None = None

    def as_dict(self) -> dict:
        return {
            "panorama_id": self.panorama_id,
            "asset_id": self.asset_id,
            "facade_id": self.facade_id,
            "heading_deg": round(self.heading_deg, 1),
            "distance_m": round(self.distance_m, 1),
            "fov_deg": round(self.fov_deg, 1),
            "verified_prominence": (
                round(self.verified_prominence, 4)
                if self.verified_prominence is not None else None
            ),
            "pitch_deg": round(self.pitch_deg, 1),
            "covers": list(self.covers),
            "purpose": self.purpose,
        }


def _bearing_to(origin: tuple[float, float], x: float, y: float) -> float:
    """Azimut projeté, 0° au nord. `atan2(dx, dy)` en CRS projeté."""
    return math.degrees(math.atan2(x - origin[0], y - origin[1])) % 360.0


def opportunities_for_facade(
    facade_id: str,
    samples: list,
    positions,  # noqa: ANN001 — itérable de (asset_id, panorama_id, origin)
    footprint,  # noqa: ANN001
    obstacles: list,
    *,
    purpose: str = "geometry",
    fov_deg: float = 70.0,
) -> list[RecropOpportunity]:
    """Cadrages récupérables couvrant une façade.

    La visibilité est évaluée **sans cap** : on demande ce que la position
    permet de voir, puis on calcule le cap qu'il faudrait demander pour le
    cadrer. C'est l'inverse de la démarche précédente, qui jugeait un cap
    subi.
    """
    from .geo.facade_coverage import visible_points

    max_distance = GEOMETRY_RANGE_M if purpose == "geometry" else TEXTURE_RANGE_M

    found: list[RecropOpportunity] = []
    for asset_id, panorama_id, origin in positions:
        seen, report = visible_points(
            samples, origin, footprint, obstacles, None, None,
            max_distance_m=max_distance,
        )
        if not seen:
            continue

        # Cap dirigé vers le milieu de ce qui est réellement visible — non vers
        # le centroïde de la façade, dont une moitié peut être masquée.
        bearings = [
            _bearing_to(origin, samples[i].x, samples[i].y) for i in seen
        ]
        heading = _mean_bearing(bearings)

        # Ce que la position permet de voir n'est pas ce qu'**une image**
        # contient : un cadrage a un champ fini. On repasse donc la visibilité
        # avec le cap **et le champ réellement demandé**, pour ne promettre que
        # ce qui tiendra dans la vignette. Le champ suit la distance : figé à
        # 70°, il annonçait une couverture qu'un télé-objectif ne rend pas, et
        # inversement rejetait des façades lointaines parfaitement lisibles.
        provisional_distance = report.distance_m or max_distance
        requested_fov = fov_for(provisional_distance)
        framed, framed_report = visible_points(
            samples, origin, footprint, obstacles, heading, requested_fov,
            max_distance_m=max_distance,
        )
        if not framed:
            continue

        distance = framed_report.distance_m or report.distance_m or max_distance
        found.append(
            RecropOpportunity(
                panorama_id=panorama_id,
                asset_id=asset_id,
                facade_id=facade_id,
                heading_deg=heading,
                distance_m=distance,
                covers=sorted(framed),
                purpose=purpose,
                # Le cadrage à demander suit la distance : un champ fixe de
                # 70° remplit l'image de stationnement dès 120 m.
                fov_deg=requested_fov,
                pitch_deg=pitch_for(distance),
            )
        )
    return found


def _mean_bearing(bearings: list[float]) -> float:
    """Moyenne circulaire : 350° et 10° ont pour moyenne 0°, non 180°."""
    if not bearings:
        return 0.0
    x = sum(math.cos(math.radians(b)) for b in bearings)
    y = sum(math.sin(math.radians(b)) for b in bearings)
    if abs(x) < 1e-12 and abs(y) < 1e-12:
        return bearings[0]
    return math.degrees(math.atan2(y, x)) % 360.0


def select_minimal(
    opportunities: list[RecropOpportunity],
    total_samples: int,
    *,
    max_requests: int | None = None,
) -> list[RecropOpportunity]:
    """Sous-ensemble couvrant le plus de mur avec le moins de requêtes.

    Couverture gloutonne : à chaque tour, le cadrage qui apporte le plus de
    points **non encore couverts**. Demander les 91 cadrages possibles d'une
    façade paierait 91 images pour la couvrir trois fois.
    """
    remaining = set(range(total_samples))
    chosen: list[RecropOpportunity] = []
    pool = list(opportunities)

    while remaining and pool:
        if max_requests is not None and len(chosen) >= max_requests:
            break
        # À gain égal, la **prominence vérifiée** départage — et seulement à
        # défaut la proximité. Classer par distance seule ramenait la rue
        # résidentielle arrière, où des pavillons absents du modèle
        # d'obstacles bouchent la vue : les six meilleurs candidats du pilote
        # montraient tous des maisons, tandis que les vues exploitables
        # venaient de plus loin, à travers les stationnements.
        best = max(
            pool,
            key=lambda o: (
                len(remaining & set(o.covers)),
                o.verified_prominence if o.verified_prominence is not None else -1.0,
                -o.distance_m,
            ),
        )
        gain = remaining & set(best.covers)
        if not gain:
            break
        chosen.append(best)
        remaining -= gain
        pool.remove(best)

    return chosen


def coverage_of(
    opportunities: list[RecropOpportunity], total_samples: int
) -> float:
    """Part du mur que l'union des cadrages retenus couvrirait."""
    if total_samples <= 0:
        return 0.0
    union: set[int] = set()
    for opportunity in opportunities:
        union |= set(opportunity.covers)
    return len(union) / total_samples


__all__ = [
    "FOV_MARGIN",
    "GEOMETRY_RANGE_M",
    "fov_for",
    "pitch_for",
    "TEXTURE_RANGE_M",
    "RecropOpportunity",
    "coverage_of",
    "opportunities_for_facade",
    "select_minimal",
]
