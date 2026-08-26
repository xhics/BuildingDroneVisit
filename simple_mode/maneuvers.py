"""Figures de vol stylisées pour l'aperçu pédagogique du trajet caméra.

Ces trajectoires ne sont **pas mesurées** sur la géométrie réelle du
bâtiment : elles illustrent des figures de vol standard en tournage
immobilier/hôtelier, centrées sur l'adresse géocodée et mises à l'échelle de
l'image satellite. Pour un trajet ancré sur des vantages vérifiés (imagerie
réelle, LiDAR), voir ``hotel_pipeline.camera_path_real`` dans le pipeline
principal — un système différent, avec des garanties différentes.

Repère local : mètres est/nord autour du centre géocodé, altitude en mètres
au-dessus du sol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace


@dataclass
class Waypoint:
    east_m: float
    north_m: float
    altitude_m: float


@dataclass
class Maneuver:
    id: str
    name_fr: str
    color: tuple[int, int, int]
    waypoints: list[Waypoint]
    #: Ce que la figure exige du pilote — la source de la difficulté.
    skill_fr: str
    #: Ce qu'elle apporte à la vidéo finale.
    purpose_fr: str

    @property
    def radius_range_m(self) -> tuple[float, float]:
        radii = [math.hypot(w.east_m, w.north_m) for w in self.waypoints]
        return min(radii), max(radii)

    @property
    def altitude_range_m(self) -> tuple[float, float]:
        altitudes = [w.altitude_m for w in self.waypoints]
        return min(altitudes), max(altitudes)

    @property
    def path_length_m(self) -> float:
        length = 0.0
        for a, b in zip(self.waypoints, self.waypoints[1:]):
            length += math.dist((a.east_m, a.north_m), (b.east_m, b.north_m))
        return length


def _circle(
    radius_m: float,
    altitude_m: float,
    *,
    turns: float = 1.0,
    start_deg: float = 0.0,
    clockwise: bool = True,
    samples: int = 72,
) -> list[Waypoint]:
    sign = 1.0 if clockwise else -1.0
    n = max(2, int(samples * turns))
    points = []
    for i in range(n + 1):
        t = i / n
        angle = math.radians(start_deg + sign * 360.0 * turns * t)
        points.append(Waypoint(radius_m * math.sin(angle), radius_m * math.cos(angle), altitude_m))
    return points


def _spiral(
    r0: float,
    r1: float,
    a0: float,
    a1: float,
    *,
    turns: float,
    start_deg: float = 0.0,
    clockwise: bool = True,
    samples: int = 90,
) -> list[Waypoint]:
    sign = 1.0 if clockwise else -1.0
    points = []
    for i in range(samples + 1):
        t = i / samples
        angle = math.radians(start_deg + sign * 360.0 * turns * t)
        radius = r0 + (r1 - r0) * t
        altitude = a0 + (a1 - a0) * t
        points.append(Waypoint(radius * math.sin(angle), radius * math.cos(angle), altitude))
    return points


def _line(
    p0: tuple[float, float, float], p1: tuple[float, float, float], samples: int = 40
) -> list[Waypoint]:
    points = []
    for i in range(samples + 1):
        t = i / samples
        east = p0[0] + (p1[0] - p0[0]) * t
        north = p0[1] + (p1[1] - p0[1]) * t
        alt = p0[2] + (p1[2] - p0[2]) * t
        points.append(Waypoint(east, north, alt))
    return points


def default_maneuvers() -> list[Maneuver]:
    """Séquence pédagogique par défaut, dans l'ordre d'un montage vidéo type.

    Reconnaissance -> approche -> figure principale -> passage final. Les
    rayons et altitudes sont des gabarits pour un bâtiment de taille
    "hôtel de centre-ville" ; ajuste-les une fois la taille réelle connue.
    """
    return [
        Maneuver(
            id="orbit_reco",
            name_fr="Orbite de reconnaissance",
            color=(255, 159, 10),  # orange
            waypoints=_circle(radius_m=55.0, altitude_m=45.0, turns=1.0, clockwise=True),
            skill_fr=(
                "vol en orbite à rayon et altitude constants — repose sur un mode "
                "POI (point of interest) GPS-assisté ou, en manuel, sur une "
                "coordination fine du roulis et du lacet pour garder le bâtiment "
                "centré sur toute la boucle"
            ),
            purpose_fr="établit le contexte : le bâtiment et ses abords, vus d'ensemble",
        ),
        Maneuver(
            id="reveal_push",
            name_fr="Approche en travelling avant",
            color=(64, 156, 255),  # bleu
            waypoints=_line((0.0, -70.0, 26.0), (0.0, -12.0, 9.0)),
            skill_fr=(
                "translation rectiligne avec descente simultanée — exige une "
                "gestion continue du tangage et du gaz pour que la vitesse "
                "d'approche décélère à l'approche du sujet, sans à-coup"
            ),
            purpose_fr="rapproche le regard du spectateur vers l'entrée du bâtiment",
        ),
        Maneuver(
            id="orbit_descent",
            name_fr="Orbite descendante en spirale",
            color=(255, 62, 87),  # rouge cramoisi
            waypoints=_spiral(r0=38.0, r1=14.0, a0=40.0, a1=10.0, turns=1.25, clockwise=True),
            skill_fr=(
                "figure la plus exigeante de la séquence : rayon, altitude et cap "
                "varient tous les trois en continu — nécessite un pilote confirmé "
                "en vol manuel, ou un logiciel de mission gérant une orbite à "
                "rayon variable, sous peine de dérive ou d'à-coups visibles"
            ),
            purpose_fr="figure signature : révèle les façades sous plusieurs angles en se rapprochant",
        ),
        Maneuver(
            id="flyover",
            name_fr="Passage rasant final",
            color=(70, 214, 140),  # vert
            waypoints=_line((-60.0, 6.0, 8.0), (60.0, 2.0, 6.0)),
            skill_fr=(
                "passage à basse altitude et vitesse stabilisée, sous la hauteur "
                "de toit — la proximité du bâti dégrade souvent la réception GPS, "
                "ce qui impose un pilotage assisté visuellement (mode ATTI) plutôt "
                "qu'un vol tout automatique"
            ),
            purpose_fr="clôt la séquence sur un mouvement dynamique et rasant",
        ),
    ]


def chain_maneuvers(maneuvers: list[Maneuver]) -> list[Maneuver]:
    """Recale chaque figure pour qu'elle démarre exactement où la précédente finit.

    Les gabarits sont écrits indépendamment les uns des autres (chacun
    autour de son propre centre local) : juxtaposés tels quels, le tracé
    saute d'une figure à l'autre. Cette fonction translate chaque figure
    (est, nord *et* altitude) pour que son premier point coïncide avec le
    dernier point de la précédente — condition nécessaire pour qu'une seule
    vidéo continue puisse être générée sans coupure. Fonctionne aussi bien
    sur le gabarit par défaut que sur un plan conçu par IA (même structure
    ``Maneuver``).
    """
    if not maneuvers:
        return maneuvers

    chained = [maneuvers[0]]
    previous_end = maneuvers[0].waypoints[-1]

    for maneuver in maneuvers[1:]:
        current_start = maneuver.waypoints[0]
        d_east = previous_end.east_m - current_start.east_m
        d_north = previous_end.north_m - current_start.north_m
        d_alt = previous_end.altitude_m - current_start.altitude_m

        shifted_waypoints = [
            Waypoint(w.east_m + d_east, w.north_m + d_north, w.altitude_m + d_alt)
            for w in maneuver.waypoints
        ]
        shifted = replace(maneuver, waypoints=shifted_waypoints)
        chained.append(shifted)
        previous_end = shifted.waypoints[-1]

    return chained


def scale_maneuvers(
    maneuvers: list[Maneuver], *, radius_scale: float = 1.0, altitude_scale: float = 1.0
) -> list[Maneuver]:
    """Met les figures à l'échelle d'un site plus grand qu'un bâtiment isolé.

    ``radius_scale`` agrandit les distances au centre (est/nord) ;
    ``altitude_scale`` agrandit séparément l'altitude, en général plus
    modérément (survoler un grand complexe ne justifie pas de monter aussi
    haut que son diamètre le suggérerait). Voir ``places.fetch_viewport_extent_m``
    pour dériver ces facteurs depuis la taille réelle du site.
    """
    scaled = []
    for maneuver in maneuvers:
        waypoints = [
            Waypoint(w.east_m * radius_scale, w.north_m * radius_scale, w.altitude_m * altitude_scale)
            for w in maneuver.waypoints
        ]
        scaled.append(replace(maneuver, waypoints=waypoints))
    return scaled


__all__ = [
    "Maneuver",
    "Waypoint",
    "chain_maneuvers",
    "default_maneuvers",
    "scale_maneuvers",
]
