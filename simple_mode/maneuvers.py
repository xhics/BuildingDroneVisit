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
    #: Où pointe la caméra. ``"centre"`` la garde braquée sur le sujet, ce
    #: qui convient aux orbites ; ``"trajet"`` la fait regarder dans le sens
    #: du vol, indispensable pour une traversée — sinon, une fois le
    #: bâtiment franchi, la caméra se retournerait pour le viser à nouveau
    #: au lieu de continuer vers l'avant.
    heading_mode: str = "centre"

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


def cinematic_maneuvers() -> list[Maneuver]:
    """Séquence conçue pour le rendu 3D réel (``cesium_render``).

    ``default_maneuvers`` vise des rayons serrés (jusqu'à 4 m) qui ont du sens
    pour une illustration ou un prompt, mais que le rendu sur tuiles
    photogrammétriques doit écarter : sous ~45 m de distance ou ~30 m
    d'altitude, la reconstruction se délite. Les bornes de
    ``cesium_render`` repousseraient donc chaque pose, déformant la figure.

    Cette séquence reste d'emblée dans l'enveloppe nette — aucune pose n'a
    besoin d'être corrigée — et privilégie des mouvements amples, plus
    lisibles en vue aérienne qu'un passage rasant.
    """
    return [
        Maneuver(
            id="etablissement",
            name_fr="Orbite d'établissement",
            color=(255, 159, 10),
            waypoints=_circle(radius_m=115.0, altitude_m=100.0, turns=0.55, clockwise=True),
            skill_fr=(
                "orbite haute à rayon constant — la vitesse angulaire doit rester "
                "lente pour éviter le flou de mouvement à cette distance"
            ),
            purpose_fr="situe le bâtiment dans son quartier, vue large",
        ),
        Maneuver(
            id="descente_approche",
            name_fr="Spirale d'approche descendante",
            color=(64, 156, 255),
            waypoints=_spiral(
                r0=115.0, r1=72.0, a0=100.0, a1=58.0, turns=0.7, start_deg=198.0, clockwise=True
            ),
            skill_fr=(
                "rayon et altitude décroissent ensemble pendant que le cap tourne — "
                "la figure la plus exigeante, elle demande une coordination continue "
                "des trois axes"
            ),
            purpose_fr="resserre progressivement le cadre sur le bâtiment",
        ),
        Maneuver(
            id="revele_facades",
            name_fr="Révélé des façades",
            color=(255, 62, 87),
            waypoints=_circle(
                radius_m=72.0, altitude_m=52.0, turns=0.75, start_deg=450.0, clockwise=True
            ),
            skill_fr=(
                "orbite basse et lente à hauteur de toiture — c'est le plan qui "
                "porte le sujet, la stabilité y prime sur la vitesse"
            ),
            purpose_fr="fait défiler les façades sous un angle valorisant",
        ),
        Maneuver(
            id="degagement",
            name_fr="Dégagement ascendant",
            color=(70, 214, 140),
            waypoints=_spiral(
                r0=72.0, r1=135.0, a0=52.0, a1=115.0, turns=0.5, start_deg=720.0, clockwise=True
            ),
            skill_fr=(
                "montée combinée à un éloignement et une rotation — exige un gaz "
                "progressif pour une sortie fluide, sans à-coup en fin de course"
            ),
            purpose_fr="clôt la séquence en rendant le contexte, effet de recul",
        ),
    ]


def _helix(
    r0: float, r1: float, a0: float, a1: float, *, turns: float, start_deg: float = 0.0,
    clockwise: bool = True, samples: int = 110,
) -> list[Waypoint]:
    """Orbite qui monte ou descend en spirale — la « helix » des cadreurs.

    Distincte de ``_spiral`` par son usage : ici le rayon varie peu et
    l'altitude beaucoup, ce qui fait défiler la hauteur du sujet plutôt que
    de s'en approcher.
    """
    return _spiral(
        r0, r1, a0, a1, turns=turns, start_deg=start_deg, clockwise=clockwise, samples=samples
    )


def _ease(t: float) -> float:
    """Accélération puis décélération douces (cosinus).

    Une figure échantillonnée linéairement démarre et s'arrête sèchement.
    Les cadreurs appellent cela un *speed ramp* : c'est ce qui distingue un
    mouvement habité d'un déplacement mécanique.
    """
    return 0.5 - 0.5 * math.cos(math.pi * max(0.0, min(1.0, t)))


def apply_speed_ramp(maneuver: Maneuver, *, samples: int | None = None) -> Maneuver:
    """Ré-échantillonne une figure avec entrée et sortie adoucies."""
    points = maneuver.waypoints
    if len(points) < 3:
        return maneuver
    count = samples or len(points)

    eased: list[Waypoint] = []
    for index in range(count):
        position = _ease(index / (count - 1)) * (len(points) - 1)
        low = min(int(position), len(points) - 2)
        frac = position - low
        a, b = points[low], points[low + 1]
        eased.append(
            Waypoint(
                a.east_m + (b.east_m - a.east_m) * frac,
                a.north_m + (b.north_m - a.north_m) * frac,
                a.altitude_m + (b.altitude_m - a.altitude_m) * frac,
            )
        )
    return replace(maneuver, waypoints=eased)


def artistic_maneuvers() -> list[Maneuver]:
    """Séquence bâtie sur les figures nommées du tournage par drone.

    ``cinematic_maneuvers`` enchaîne des orbites de rayons différents : lisible,
    mais monotone — c'est le reproche fait au premier montage. On alterne ici
    des figures de natures distinctes, comme le ferait un cadreur :

    - **reveal** : on monte derrière un obstacle pour découvrir le sujet ;
    - **parallax orbit** : orbite basse et proche, où l'arrière-plan défile
      vite derrière le sujet et donne la profondeur ;
    - **helix** : orbite qui descend en spirale, faisant défiler la hauteur ;
    - **push-in** : rapprochement franc, décéléré à l'arrivée ;
    - **fly-over** : passage au-dessus, qui bascule le regard vers l'arrière.

    Chaque figure reçoit une entrée et une sortie adoucies : sans cela, le
    raccord entre deux figures se voit comme un à-coup.
    """
    figures = [
        Maneuver(
            id="reveal",
            name_fr="Révélé ascendant",
            color=(255, 159, 10),
            waypoints=_line((0.0, -150.0, 35.0), (0.0, -95.0, 105.0), samples=70),
            skill_fr=(
                "montée verticale pendant l'avance, le sujet se dévoile derrière "
                "la ligne d'horizon proche — la caméra doit monter plus vite "
                "qu'elle n'avance"
            ),
            purpose_fr="ouvre la séquence en dévoilant progressivement le bâtiment",
        ),
        Maneuver(
            id="orbite_parallaxe",
            name_fr="Orbite en parallaxe",
            color=(64, 156, 255),
            waypoints=_circle(radius_m=78.0, altitude_m=62.0, turns=0.85, start_deg=180.0),
            skill_fr=(
                "orbite serrée et basse : l'arrière-plan défile vite derrière le "
                "sujet, ce qui crée la profondeur — exige un rayon constant au mètre"
            ),
            purpose_fr="donne le volume et la profondeur du bâtiment",
        ),
        Maneuver(
            id="helix",
            name_fr="Hélice descendante",
            color=(255, 62, 87),
            waypoints=_helix(88.0, 74.0, 105.0, 48.0, turns=0.9, start_deg=486.0),
            skill_fr=(
                "l'altitude chute pendant que l'orbite se poursuit : la façade "
                "défile de haut en bas, révélant les étages un à un"
            ),
            purpose_fr="fait défiler la hauteur de la façade, étage par étage",
        ),
        Maneuver(
            id="push_in",
            name_fr="Rapprochement décéléré",
            color=(255, 214, 10),
            waypoints=_line((0.0, 96.0, 48.0), (0.0, 58.0, 40.0), samples=60),
            skill_fr=(
                "avance franche puis décélération marquée à l'approche — c'est la "
                "décélération, pas la vitesse, qui donne le poids au plan"
            ),
            purpose_fr="resserre l'attention sur l'entrée avant la traversée",
        ),
        Maneuver(
            id="survol",
            name_fr="Survol basculant",
            color=(70, 214, 140),
            waypoints=_line((0.0, -70.0, 70.0), (0.0, 120.0, 92.0), samples=80),
            skill_fr=(
                "passage au-dessus du sujet, le regard bascule progressivement "
                "vers l'arrière pendant que le drone poursuit sa route"
            ),
            purpose_fr="conclut en survolant le site et en ouvrant sur les abords",
        ),
    ]
    return [apply_speed_ramp(figure) for figure in figures]


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
    "apply_speed_ramp",
    "artistic_maneuvers",
    "chain_maneuvers",
    "cinematic_maneuvers",
    "default_maneuvers",
    "scale_maneuvers",
]
