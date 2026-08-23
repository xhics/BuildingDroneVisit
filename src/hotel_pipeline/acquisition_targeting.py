"""Ne demander que les recadrages susceptibles de montrer le sujet (Lot 2).

Chaque recadrage Street View est une **requête facturée**, et le registre du
pilote en porte 74 pour 14 vues exploitables : **73 % de perte**. Le module
`recrop_opportunities` propose déjà bien mieux qu'un balayage aveugle — il
choisit le cap et le champ depuis la géométrie — mais il ne sait pas quels
*points de vue* méritent d'être payés.

Ce module répond à cette seule question : **parmi les panoramas disponibles,
lesquels demander ?** Il ne remplace pas la lecture pixel qui suit ; il évite
de payer pour des vues que cette lecture rejettera.

Ce qui prédit le succès, mesuré
-------------------------------
Sur les 62 recadrages du pilote dont la géométrie est reconstituable :

```text
variable                    exploitables   perdus
angle sous-tendu (°)            45,1        26,9    ← discriminant
distance (m)                      90         141
écart cap-centroïde (°)          2,9         3,6
ratio champ / sous-tendu         1,11        1,08    ← ne discrimine pas
```

Deux enseignements. D'abord, **l'angle sous-tendu domine** : ce n'est pas la
distance en soi qui compte, c'est la place que le bâtiment occupe vu d'ici.
Ensuite, le ratio champ/sous-tendu ne sépare rien — `fov_for` fait déjà
correctement son travail. Ce qui échoue est le **choix du point de vue**, non
le cadrage.

Le second signal est gratuit : les vues déjà payées
---------------------------------------------------
Un filtre géométrique plafonne à 33 % de précision, et c'est structurel : il ne
peut pas savoir qu'une concession automobile bouche la vue, ni qu'une rangée
d'arbres masque la façade. Cette information-là n'existe que dans les pixels
déjà achetés.

D'où le second critère : **un panorama dont les voisins ont donné de bonnes
vues en donnera probablement aussi.** Validé en croisé sur le pilote, chaque
point étant prédit par ses voisins seuls :

```text
rayon    prédictibles   précision   rappel   requêtes perdues évitées
 30 m         36           33 %      33 %            21
 50 m         49           31 %      36 %            29
 80 m         56           44 %      58 %            35     ← retenu
120 m         59           33 %      25 %            41
```

Combiné
-------
```text
stratégie                              requêtes   exploitables   rendement
aucun filtre                              62           13          21 %
angle sous-tendu ≥ 35°                    24            8          33 %
voisinage 80 m                            16            7          44 %
voisinage 80 m ET angle ≥ 35°             13            6          46 %
```

Le rendement plus que double. Le prix est un rappel partiel : on renonce à des
vues exploitables pour éviter beaucoup de vues perdues. C'est le bon arbitrage
quand chaque requête coûte, et l'appelant peut le desserrer.

Validé en laisser-un-de-côté
----------------------------
Chaque recadrage du pilote prédit par les 61 autres, sans jamais s'utiliser
lui-même :

```text
tier            requêtes   exploitables   rendement
recommended        13            6           46 %
plausible          20            4           20 %
unpromising        29            3           10 %
sans ciblage       62           13           21 %
```

Les trois niveaux se séparent et s'ordonnent correctement. Payer les seuls
`recommended` double le rendement (21 % → 46 %) en divisant les requêtes par
cinq. Et `unpromising` n'est pas vide — 3 vues exploitables s'y trouvent —, ce
qui justifie de les garder en file plutôt que de les interdire.

Ce que ces chiffres ne disent pas
---------------------------------
Ils viennent d'**un seul site**, et les seuils y sont calibrés. `MIN_SUBTENDED_DEG`
tient entre les deux médianes observées (26,9° et 45,1°) ; rien ne garantit
qu'un bâtiment de forme très différente les reproduise. Le rayon de 80 m
dépend de la densité du réseau Street View local.

Le module reste utile ailleurs parce que sa **structure** ne dépend pas du
site — géométrie plus voisinage mesuré, sans rejet définitif — mais ses
constantes demanderaient d'être revérifiées sur un second hôtel.

Ce que le module refuse de faire
--------------------------------
Il ne **rejette** rien définitivement. Un panorama écarté reste candidat : il
est seulement placé plus bas dans la file. Une campagne qui a du budget peut
descendre la liste ; une campagne serrée s'arrête en haut. Confondre « peu
prometteur » et « inutile » ferait perdre des vues que le voisinage n'a pas su
prédire faute de mesures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .logging import get_logger

log = get_logger("targeting")

#: Angle sous-tendu par le bâtiment sous lequel une vue est peu prometteuse.
#: Médiane mesurée : 45,1° pour les vues exploitables, 26,9° pour les perdues.
MIN_SUBTENDED_DEG = 35.0

#: Rayon de voisinage, en mètres. Calibré : 80 m maximise la précision (44 %)
#: et le rappel (58 %) en validation croisée sur le pilote.
NEIGHBOUR_RADIUS_M = 80.0

#: Voisins mesurés nécessaires pour que leur avis compte.
MIN_NEIGHBOURS = 2

#: Part de voisins exploitables au-delà de laquelle le voisinage est favorable.
NEIGHBOUR_SUCCESS_RATIO = 0.34

#: Facteur appliqué à la priorité d'un panorama dont la pose n'est pas
#: attestée. Rétrograder plutôt qu'exclure : la vue reste exploitable pour
#: l'apparence, et `pose_refine` peut la récupérer.
POSE_UNATTESTED_PENALTY = 0.6

#: Score de prominence à partir duquel une vue est tenue pour exploitable.
#: Aligné sur `recrop_sweep.ACCEPT_THRESHOLD`.
USEFUL_SCORE = 0.60


@dataclass
class Vantage:
    """Un point de vue candidat, et ce qui le recommande ou non."""

    panorama_id: str
    origin: tuple[float, float]
    subtended_deg: float
    distance_m: float
    #: Part de voisins exploitables, `None` si aucun voisin mesuré.
    neighbour_ratio: float | None = None
    neighbours: int = 0
    #: La pose de ce panorama est-elle attestée ? `None` si non renseigné.
    #: Une vue superbe à une position fausse ne sert ni à projeter le modèle,
    #: ni à situer le sol : `temporal_consensus` a montré qu'une dérive de
    #: 40 m suffit à poser les cellules de pelouse sur le stationnement.
    pose_attested: bool | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def geometry_favourable(self) -> bool:
        return self.subtended_deg >= MIN_SUBTENDED_DEG

    @property
    def neighbourhood_favourable(self) -> bool | None:
        """`None` quand aucun voisin n'a été mesuré — inconnu, non défavorable."""
        if self.neighbour_ratio is None:
            return None
        return self.neighbour_ratio >= NEIGHBOUR_SUCCESS_RATIO

    @property
    def priority(self) -> float:
        """Rang dans la file. Plus haut = à demander en premier.

        La composition est délibérément simple — deux signaux, une somme
        pondérée. Un modèle appris sur 62 points surajusterait, et le pilote
        n'a pas de quoi le valider.
        """
        score = min(1.0, self.subtended_deg / 60.0)
        if self.neighbour_ratio is not None:
            # Le voisinage pèse plus que la géométrie : il porte ce que la
            # géométrie ne peut pas voir — occlusions réelles, végétation.
            score = 0.35 * score + 0.65 * self.neighbour_ratio
        if self.pose_attested is False:
            # Rétrogradé, non exclu : la vue reste utile pour l'apparence, et
            # un raffinement de pose peut la récupérer.
            score *= POSE_UNATTESTED_PENALTY
        return round(score, 4)

    @property
    def tier(self) -> str:
        """`recommended` / `plausible` / `unpromising`. Jamais `rejected`."""
        if self.geometry_favourable and self.neighbourhood_favourable:
            return "recommended"
        if self.neighbourhood_favourable is False and not self.geometry_favourable:
            return "unpromising"
        return "plausible"

    def as_dict(self) -> dict:
        return {
            "panorama_id": self.panorama_id,
            "tier": self.tier,
            "priority": self.priority,
            "subtended_deg": round(self.subtended_deg, 1),
            "distance_m": round(self.distance_m, 1),
            "neighbours": self.neighbours,
            "pose_attested": self.pose_attested,
            "neighbour_ratio": (
                round(self.neighbour_ratio, 3)
                if self.neighbour_ratio is not None else None
            ),
            "reasons": list(self.reasons),
        }


def subtended_angle(
    origin: tuple[float, float], footprint_coords: list[tuple[float, float]]
) -> float:
    """Angle horizontal que le bâtiment occupe vu depuis `origin`, en degrés.

    C'est le prédicteur le plus fort mesuré : il dit la place du sujet dans le
    cadre, là où la distance seule ignore la taille et l'orientation du
    bâtiment. L'arrière d'un bâtiment de 72 × 77 m se voit de bien plus loin
    que sa façade étroite.
    """
    if not footprint_coords:
        return 0.0
    bearings = [
        math.degrees(math.atan2(x - origin[0], y - origin[1])) % 360.0
        for x, y in footprint_coords
    ]
    reference = bearings[0]
    relative = sorted(
        ((bearing - reference + 180.0) % 360.0 - 180.0) for bearing in bearings
    )
    return relative[-1] - relative[0]


def neighbourhood(
    origin: tuple[float, float],
    measured: list[tuple[tuple[float, float], float]],
    *,
    radius_m: float = NEIGHBOUR_RADIUS_M,
    useful_score: float = USEFUL_SCORE,
) -> tuple[float | None, int]:
    """Part de vues exploitables déjà mesurées autour d'un point.

    `measured` porte `[(position, score), ...]` des recadrages déjà payés. Un
    point sans voisin mesuré rend `(None, 0)` : l'absence de voisinage n'est
    pas un mauvais voisinage.
    """
    scores = [
        score for position, score in measured
        if math.hypot(position[0] - origin[0], position[1] - origin[1]) <= radius_m
    ]
    if len(scores) < MIN_NEIGHBOURS:
        return None, len(scores)
    useful = sum(1 for score in scores if score >= useful_score)
    return useful / len(scores), len(scores)


def evaluate(
    candidates: list[tuple[str, tuple[float, float]]],
    footprint_coords: list[tuple[float, float]],
    centroid: tuple[float, float],
    measured: list[tuple[tuple[float, float], float]] | None = None,
    *,
    radius_m: float = NEIGHBOUR_RADIUS_M,
    attested: dict[str, bool] | None = None,
) -> list[Vantage]:
    """Classe les points de vue candidats, du plus prometteur au moins.

    Aucun n'est écarté : le tri suffit, et une campagne bien dotée descend
    plus bas dans la file.
    """
    measured = measured or []
    vantages: list[Vantage] = []
    for panorama_id, origin in candidates:
        subtended = subtended_angle(origin, footprint_coords)
        ratio, count = neighbourhood(origin, measured, radius_m=radius_m)
        vantage = Vantage(
            panorama_id=panorama_id, origin=origin, subtended_deg=subtended,
            distance_m=math.hypot(origin[0] - centroid[0], origin[1] - centroid[1]),
            neighbour_ratio=ratio, neighbours=count,
            pose_attested=(attested or {}).get(panorama_id),
        )
        if vantage.pose_attested is False:
            vantage.reasons.append(
                "pose non attestée : la vue servira l'apparence, non la géométrie"
            )
        if vantage.geometry_favourable:
            vantage.reasons.append(
                f"le bâtiment occupe {subtended:.0f}° du champ"
            )
        else:
            vantage.reasons.append(
                f"le bâtiment n'occupe que {subtended:.0f}° "
                f"(seuil {MIN_SUBTENDED_DEG:.0f}°)"
            )
        if ratio is None:
            vantage.reasons.append(
                f"{count} voisin(s) mesuré(s) : voisinage inconnu, non défavorable"
            )
        else:
            vantage.reasons.append(
                f"{ratio:.0%} des {count} voisins mesurés sont exploitables"
            )
        vantages.append(vantage)

    vantages.sort(key=lambda v: (-v.priority, v.distance_m))
    log.info(
        "ciblage : %d candidat(s), %d recommandé(s), %d peu prometteur(s)",
        len(vantages),
        sum(1 for v in vantages if v.tier == "recommended"),
        sum(1 for v in vantages if v.tier == "unpromising"),
    )
    return vantages


def select(
    vantages: list[Vantage], *, budget: int, include_plausible: bool = True
) -> list[Vantage]:
    """Les `budget` premiers points de vue à demander.

    Les `unpromising` ne sont servis que si le budget dépasse tout le reste :
    payer pour eux avant d'avoir épuisé les autres serait gaspiller, mais les
    interdire ferait perdre les vues que le voisinage n'a pas su prédire.
    """
    order = ["recommended"] + (["plausible"] if include_plausible else [])
    chosen = [v for v in vantages if v.tier in order][:budget]
    if len(chosen) < budget:
        rest = [v for v in vantages if v not in chosen]
        chosen.extend(rest[: budget - len(chosen)])
    return chosen


def expected_yield(vantages: list[Vantage], measured_baseline: float) -> dict:
    """Rendement attendu d'une sélection, comparé au rendement observé.

    `measured_baseline` est la part de vues exploitables déjà constatée sans
    ciblage — 21 % sur le pilote. Le gain n'a de sens que rapporté à elle.
    """
    tiers: dict[str, int] = {}
    for vantage in vantages:
        tiers[vantage.tier] = tiers.get(vantage.tier, 0) + 1
    return {
        "candidates": len(vantages),
        "by_tier": dict(sorted(tiers.items())),
        "baseline_yield": round(measured_baseline, 3),
        "recommended": tiers.get("recommended", 0),
    }


__all__ = [
    "MIN_NEIGHBOURS",
    "MIN_SUBTENDED_DEG",
    "NEIGHBOUR_RADIUS_M",
    "NEIGHBOUR_SUCCESS_RATIO",
    "POSE_UNATTESTED_PENALTY",
    "USEFUL_SCORE",
    "Vantage",
    "evaluate",
    "expected_yield",
    "neighbourhood",
    "select",
    "subtended_angle",
]
