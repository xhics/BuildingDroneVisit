"""Carte des observations manquantes : ce qu'il faudrait photographier.

Mesurer ce qu'on a photographié ne dit pas ce qu'il faut aller chercher. Une
façade vue dix fois depuis le même trottoir reste **non reconstructible** : dix
vues quasi confondues ne donnent aucune parallaxe, et la triangulation y est
aussi mal conditionnée qu'avec une seule image.

Ce module répond donc à une autre question que la couverture d'apparence :
**chaque portion de mur est-elle observée deux fois sous un angle exploitable ?**
C'est la condition de la reconstruction multivue, et elle se mesure avant toute
acquisition — non après un solve qui échoue sans dire pourquoi.

Le résultat est une carte par cellule de mur, portant :

- combien de vues indépendantes l'observent ;
- le meilleur angle de triangulation disponible ;
- la distance et l'incidence des vues qui la couvrent ;
- ce qui manque, dit en direction et en distance plutôt qu'en score.

**Ce que la carte ne dit pas.** Elle décrit la géométrie de l'observation, non
la qualité des images : une cellule bien triangulée par deux vues floues sera
comptée comme observée. La netteté relève d'`appearance_quality`, et le fait
qu'une vue montre bien le bâtiment relève d'`in_frame`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..logging import get_logger

log = get_logger("geo-observation-map")

#: Angle de triangulation, en degrés, en deçà duquel deux vues sont trop
#: proches pour se compléter. En dessous de cinq degrés, l'incertitude en
#: profondeur explose : c'est le seuil usuel des travaux SfM, et celui que
#: COLMAP applique par défaut pour accepter un point triangulé.
MIN_TRIANGULATION_DEG = 5.0

#: Angle au-delà duquel deux vues se ressemblent trop peu pour que la mise en
#: correspondance tienne : la façade y change d'aspect, et les descripteurs
#: cessent de s'apparier.
MAX_TRIANGULATION_DEG = 60.0

#: Incidence maximale, en degrés depuis la normale du mur. Au-delà, la façade
#: est vue en enfilade : quelques pixels décrivent plusieurs mètres, et la
#: texture n'y est plus exploitable.
MAX_INCIDENCE_DEG = 65.0

#: Distance au-delà de laquelle une façade n'occupe plus assez de pixels pour
#: porter de la structure. Mesuré sur ce pilote, les vues de rue exploitables
#: sont à moins de cent mètres.
MAX_USEFUL_DISTANCE_M = 120.0

#: Nombre de vues indépendantes qu'une cellule doit recevoir. Deux suffisent à
#: trianguler ; une troisième donne de quoi écarter une correspondance fausse.
TARGET_VIEWS = 3


@dataclass
class CellObservation:
    """Ce qu'une portion de mur reçoit comme observations."""

    facade_id: str
    #: Position du point de mur, en CRS projeté.
    x: float
    y: float
    #: Direction vers laquelle ce mur regarde, en degrés (0 = est).
    normal_deg: float
    view_count: int = 0
    #: Meilleur angle entre deux vues qui l'observent, en degrés.
    best_triangulation_deg: float = 0.0
    nearest_distance_m: float | None = None
    best_incidence_deg: float | None = None
    contributing: list[str] = field(default_factory=list)

    @property
    def triangulable(self) -> bool:
        """Deux vues suffisamment écartées la couvrent-elles ?"""
        return (
            self.view_count >= 2
            and MIN_TRIANGULATION_DEG <= self.best_triangulation_deg <= MAX_TRIANGULATION_DEG
        )

    @property
    def status(self) -> str:
        if self.view_count == 0:
            return "aucune_vue"
        if self.view_count == 1:
            return "vue_unique"
        if self.best_triangulation_deg < MIN_TRIANGULATION_DEG:
            return "parallaxe_insuffisante"
        if self.best_triangulation_deg > MAX_TRIANGULATION_DEG:
            return "vues_trop_ecartees"
        if self.view_count < TARGET_VIEWS:
            return "triangulable_sans_marge"
        return "observe"

    def as_dict(self) -> dict:
        return {
            "facade_id": self.facade_id,
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "normal_deg": round(self.normal_deg, 1),
            "view_count": self.view_count,
            "best_triangulation_deg": round(self.best_triangulation_deg, 1),
            "nearest_distance_m": (
                round(self.nearest_distance_m, 1)
                if self.nearest_distance_m is not None
                else None
            ),
            "best_incidence_deg": (
                round(self.best_incidence_deg, 1)
                if self.best_incidence_deg is not None
                else None
            ),
            "status": self.status,
            "triangulable": self.triangulable,
        }


@dataclass
class MissingObservation:
    """Une prise de vue qui comblerait un manque, dite en termes d'acquisition."""

    facade_id: str
    #: Azimut depuis lequel photographier, en degrés.
    bearing_deg: float
    #: Distance recommandée, en mètres.
    distance_m: float
    #: Position à atteindre, en CRS projeté.
    x: float
    y: float
    cells_gained: int
    reason: str
    #: Cellules partagées avec une vue déjà acquise. Sans recouvrement, la
    #: nouvelle prise forme un îlot que le solve ne rattachera pas.
    shared_with_existing: int = 0
    #: Recommandations voisines avec lesquelles elle partage des cellules.
    linked_to: list[int] = field(default_factory=list)

    @property
    def connected(self) -> bool:
        """Cette prise se rattachera-t-elle à quelque chose ?"""
        return self.shared_with_existing > 0 or bool(self.linked_to)

    def as_dict(self) -> dict:
        return {
            "facade_id": self.facade_id,
            "bearing_deg": round(self.bearing_deg, 1),
            "distance_m": round(self.distance_m, 1),
            "position": [round(self.x, 2), round(self.y, 2)],
            "cells_gained": self.cells_gained,
            "shared_with_existing": self.shared_with_existing,
            "linked_to": self.linked_to,
            "connected": self.connected,
            "reason": self.reason,
        }


@dataclass
class ObservationMap:
    """Ce qui est observable, ce qui ne l'est pas, et ce qu'il faudrait acquérir."""

    cells: list[CellObservation] = field(default_factory=list)
    missing: list[MissingObservation] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    def by_status(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for cell in self.cells:
            counts[cell.status] = counts.get(cell.status, 0) + 1
        return counts

    def by_facade(self) -> dict[str, dict]:
        grouped: dict[str, dict] = {}
        for cell in self.cells:
            entry = grouped.setdefault(
                cell.facade_id, {"total": 0, "triangulable": 0, "unseen": 0}
            )
            entry["total"] += 1
            if cell.triangulable:
                entry["triangulable"] += 1
            if cell.view_count == 0:
                entry["unseen"] += 1
        for entry in grouped.values():
            entry["fraction"] = round(entry["triangulable"] / max(entry["total"], 1), 3)
        return grouped

    def as_dict(self) -> dict:
        triangulable = sum(1 for c in self.cells if c.triangulable)
        return {
            "cell_count": len(self.cells),
            "triangulable_count": triangulable,
            "triangulable_fraction": round(
                triangulable / max(len(self.cells), 1), 3
            ),
            "by_status": self.by_status(),
            "by_facade": self.by_facade(),
            "cells": [c.as_dict() for c in self.cells],
            "missing": [m.as_dict() for m in self.missing],
            "provenance": self.provenance,
            "caveats": [
                "la carte décrit la géométrie de l'observation, non la qualité "
                "des images : deux vues floues bien écartées comptent comme "
                "observées",
                "une position recommandée suppose qu'on puisse y accéder — "
                "rien ici ne vérifie qu'il s'agit d'un lieu public",
            ],
        }


def _bearing(origin: tuple[float, float], point: tuple[float, float]) -> float:
    """Azimut du point vu depuis l'origine, en degrés dans [0, 360)."""
    return math.degrees(math.atan2(point[1] - origin[1], point[0] - origin[0])) % 360.0


def _angular_gap(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def _incidence(normal_deg: float, view_bearing_deg: float) -> float:
    """Écart entre la normale du mur et la direction d'où on le regarde."""
    return _angular_gap(normal_deg, (view_bearing_deg + 180.0) % 360.0)


def build(
    samples,  # noqa: ANN001 - list[FacadeSample]
    observations: list[tuple[str, tuple[float, float], list[int]]],
    facade_id: str = "FACADE",
) -> ObservationMap:
    """Croise les points de mur et les vues qui les atteignent.

    `observations` porte, pour chaque vue retenue, son identifiant, sa position
    et les indices de points qu'elle voit — ce que `visible_points` calcule
    déjà, occultations comprises. Ce module n'y ajoute que la question de la
    **parallaxe** : deux vues qui voient le même point, mais depuis le même
    endroit, n'apportent qu'une observation.
    """
    cells = [
        CellObservation(
            facade_id=facade_id,
            x=sample.x,
            y=sample.y,
            normal_deg=math.degrees(math.atan2(sample.normal[1], sample.normal[0])) % 360.0,
        )
        for sample in samples
    ]

    # Azimut de chaque vue, point par point : c'est lui qui porte la parallaxe.
    seen_by: list[list[tuple[str, float, float]]] = [[] for _ in cells]
    for asset_id, origin, indices in observations:
        for index in indices:
            if not 0 <= index < len(cells):
                continue
            cell = cells[index]
            bearing = _bearing(origin, (cell.x, cell.y))
            distance = math.hypot(cell.x - origin[0], cell.y - origin[1])
            if distance > MAX_USEFUL_DISTANCE_M:
                continue
            incidence = _incidence(cell.normal_deg, bearing)
            if incidence > MAX_INCIDENCE_DEG:
                continue
            seen_by[index].append((asset_id, bearing, distance))
            if cell.nearest_distance_m is None or distance < cell.nearest_distance_m:
                cell.nearest_distance_m = distance
            if cell.best_incidence_deg is None or incidence < cell.best_incidence_deg:
                cell.best_incidence_deg = incidence

    for cell, views in zip(cells, seen_by):
        cell.view_count = len(views)
        cell.contributing = [asset_id for asset_id, _b, _d in views]
        # Meilleure paire : l'écart angulaire le plus proche de l'utile, non le
        # plus grand. Deux vues à cent quatre-vingts degrés voient des faces
        # opposées et ne s'apparient pas davantage que deux vues confondues.
        best = 0.0
        for i in range(len(views)):
            for j in range(i + 1, len(views)):
                gap = _angular_gap(views[i][1], views[j][1])
                if gap > MAX_TRIANGULATION_DEG:
                    continue
                best = max(best, gap)
        if best == 0.0 and len(views) >= 2:
            # Toutes les paires dépassent l'écart utile : on retient la plus
            # petite, pour que le statut dise « trop écartées » et non « aucune ».
            best = min(
                _angular_gap(views[i][1], views[j][1])
                for i in range(len(views))
                for j in range(i + 1, len(views))
            )
        cell.best_triangulation_deg = best

    found = ObservationMap(cells=cells)
    found.provenance = {
        "min_triangulation_deg": MIN_TRIANGULATION_DEG,
        "max_triangulation_deg": MAX_TRIANGULATION_DEG,
        "max_incidence_deg": MAX_INCIDENCE_DEG,
        "max_useful_distance_m": MAX_USEFUL_DISTANCE_M,
        "target_views": TARGET_VIEWS,
        "observations_supplied": len(observations),
    }
    log.info(
        "carte d'observation : %d cellule(s), %d triangulable(s)",
        len(cells),
        sum(1 for c in cells if c.triangulable),
    )
    return found


#: Demi-angle, en degrés, du secteur qu'une prise couvre depuis sa position.
#: Volontairement plus large que le secteur de regroupement : une caméra voit
#: au-delà du mur qui l'a motivée, et c'est ce débord qui crée le recouvrement.
REACH_HALF_ANGLE_DEG = 55.0


def _cells_reached(
    cells: list[CellObservation],
    position: tuple[float, float],
    normal_deg: float,
    sector_deg: float,
) -> set[int]:
    """Cellules qu'une prise atteindrait depuis cette position.

    Le critère est celui de `build` : distance utile et incidence acceptable.
    Reprendre les mêmes seuils évite qu'une prise soit jugée connectée sur des
    cellules que la mesure, elle, ne compterait pas.
    """
    reached: set[int] = set()
    for index, cell in enumerate(cells):
        distance = math.hypot(cell.x - position[0], cell.y - position[1])
        if distance > MAX_USEFUL_DISTANCE_M or distance < 1e-6:
            continue
        bearing = _bearing(position, (cell.x, cell.y))
        if _incidence(cell.normal_deg, bearing) > MAX_INCIDENCE_DEG:
            continue
        # Le mur doit tomber dans le champ de la prise, orientée vers le
        # secteur qui l'a motivée.
        aim = (normal_deg + 180.0) % 360.0
        if _angular_gap(bearing, aim) > REACH_HALF_ANGLE_DEG:
            continue
        reached.add(index)
    return reached


def _bridge_positions(
    cells: list[CellObservation],
    missing: list[MissingObservation],
    covered: list[set[int]],
    distance_m: float,
    sector_deg: float,
) -> list[MissingObservation]:
    """Prises intermédiaires reliant un îlot au reste du réseau.

    Une position isolée comble de vraies cellules, mais le solve ne saura pas
    où la placer : rien de ce qu'elle voit n'a été vu ailleurs. La liaison se
    cherche donc à mi-chemin — un azimut intermédiaire, d'où le champ couvre à
    la fois le secteur orphelin et un secteur déjà relié.

    Quand aucun azimut intermédiaire ne couvre les deux, aucune liaison n'est
    proposée : mieux vaut dire que l'îlot reste isolé que suggérer une prise
    qui ne relierait rien.
    """
    bridges: list[MissingObservation] = []
    orphans = [slot for slot, entry in enumerate(missing) if not entry.connected]
    if not orphans:
        return bridges

    anchored = [slot for slot, entry in enumerate(missing) if entry.connected]
    seen_now = {index for index, cell in enumerate(cells) if cell.view_count > 0}

    for orphan in orphans:
        target = missing[orphan]
        aim = (target.bearing_deg + 180.0) % 360.0
        best: tuple[int, float, set[int]] | None = None

        for other in anchored:
            neighbour = (missing[other].bearing_deg + 180.0) % 360.0
            # Azimut à mi-chemin, en tenant compte du repliement à 360°.
            delta = (neighbour - aim + 180.0) % 360.0 - 180.0
            middle = (aim + delta * 0.5) % 360.0
            centre_x = sum(c.x for c in cells) / len(cells)
            centre_y = sum(c.y for c in cells) / len(cells)
            radians = math.radians(middle)
            position = (
                centre_x + math.cos(radians) * distance_m,
                centre_y + math.sin(radians) * distance_m,
            )
            reach = _cells_reached(cells, position, middle, sector_deg)
            # La liaison doit toucher l'îlot **et** quelque chose de déjà relié.
            if not (reach & covered[orphan]):
                continue
            if not (reach & (seen_now | covered[other])):
                continue
            score = len(reach)
            if best is None or score > best[0]:
                best = (score, middle, reach)

        if best is None:
            continue

        _score, middle, reach = best
        radians = math.radians(middle)
        centre_x = sum(c.x for c in cells) / len(cells)
        centre_y = sum(c.y for c in cells) / len(cells)
        bridges.append(
            MissingObservation(
                facade_id=target.facade_id,
                bearing_deg=(middle + 180.0) % 360.0,
                distance_m=distance_m,
                x=centre_x + math.cos(radians) * distance_m,
                y=centre_y + math.sin(radians) * distance_m,
                cells_gained=len(reach & covered[orphan]),
                reason="liaison : rattache un secteur autrement isolé",
                shared_with_existing=len(reach & seen_now),
                linked_to=[orphan],
            )
        )
        # L'îlot cesse de l'être : la liaison le rattache.
        target.linked_to.append(len(missing) + len(bridges) - 1)

    return bridges


def recommend(
    found: ObservationMap, distance_m: float = 45.0, sector_deg: float = 30.0
) -> ObservationMap:
    """Propose les prises de vue qui combleraient le plus de cellules.

    La recommandation est délibérément grossière : un azimut, une distance, un
    point. Elle ne prétend pas désigner un emplacement précis — la voirie, les
    clôtures et les droits d'accès ne figurent nulle part ici — mais indiquer
    la **direction** d'où le manque se comble.

    Les cellules sont regroupées par orientation de mur : celles qui regardent
    dans la même direction se comblent depuis le même secteur, et proposer une
    position par cellule noierait la carte sous des redondances.
    """
    gaps = [c for c in found.cells if not c.triangulable]
    if not gaps:
        found.missing = []
        return found

    buckets: dict[tuple[str, int], list[CellObservation]] = {}
    for cell in gaps:
        key = (cell.facade_id, int(cell.normal_deg // sector_deg))
        buckets.setdefault(key, []).append(cell)

    missing: list[MissingObservation] = []
    covered: list[set[int]] = []
    for (facade_id, sector), group in sorted(
        buckets.items(), key=lambda item: -len(item[1])
    ):
        # Se placer dans l'axe du mur : c'est l'incidence nulle, la meilleure
        # pour la texture comme pour la mise en correspondance.
        normal = (sector + 0.5) * sector_deg
        centre_x = sum(c.x for c in group) / len(group)
        centre_y = sum(c.y for c in group) / len(group)
        radians = math.radians(normal)
        reasons = {c.status for c in group}

        # Ce que cette prise verrait : son propre secteur, plus les cellules
        # voisines que sa position atteindrait. C'est le recouvrement qui
        # décide si elle se rattache, non le nombre de cellules comblées.
        position = (
            centre_x + math.cos(radians) * distance_m,
            centre_y + math.sin(radians) * distance_m,
        )
        reach = _cells_reached(found.cells, position, normal, sector_deg)

        # Une cellule déjà observée et revue par la nouvelle prise fait le
        # lien : c'est là que le graphe se referme.
        shared = sum(
            1
            for index in reach
            if found.cells[index].view_count > 0
        )
        linked = [
            slot for slot, previous in enumerate(covered) if reach & previous
        ]

        missing.append(
            MissingObservation(
                facade_id=facade_id,
                bearing_deg=(normal + 180.0) % 360.0,
                distance_m=distance_m,
                x=position[0],
                y=position[1],
                cells_gained=len(group),
                reason=", ".join(sorted(reasons)),
                shared_with_existing=shared,
                linked_to=linked,
            )
        )
        covered.append(reach)

    # La réciprocité : si B se rattache à A, A se rattache à B. Sans elle, la
    # première recommandation paraîtrait isolée alors qu'une suivante la relie.
    for slot, entry in enumerate(missing):
        for other in entry.linked_to:
            if slot not in missing[other].linked_to:
                missing[other].linked_to.append(slot)

    # Un îlot ne se rattache pas en reculant : ses cellules ne sont vues par
    # personne, quelle que soit la distance. Il faut une prise **entre** lui et
    # le reste, dont le champ couvre les deux.
    bridges = _bridge_positions(found.cells, missing, covered, distance_m, sector_deg)
    missing.extend(bridges)

    found.missing = missing
    orphans = [m for m in missing if not m.connected]
    if orphans:
        log.warning(
            "%d prise(s) recommandée(s) sans recouvrement : elles combleraient "
            "des cellules mais formeraient un îlot que le solve ne rattache pas",
            len(orphans),
        )
    if bridges:
        log.info("%d prise(s) de liaison ajoutée(s)", len(bridges))
    log.info("%d prise(s) de vue recommandée(s)", len(missing))
    return found


__all__ = [
    "MAX_INCIDENCE_DEG",
    "MAX_TRIANGULATION_DEG",
    "MAX_USEFUL_DISTANCE_M",
    "MIN_TRIANGULATION_DEG",
    "REACH_HALF_ANGLE_DEG",
    "TARGET_VIEWS",
    "CellObservation",
    "MissingObservation",
    "ObservationMap",
    "build",
    "recommend",
]
