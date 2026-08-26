"""Génère le texte pédagogique décrivant la séquence de trajectoires."""

from __future__ import annotations

from .maneuvers import Maneuver

_INTRO = """# Trajectoire caméra — {address}

Coordonnées : {lat:.6f}, {lon:.6f}
Image satellite : zoom {zoom}, échelle approximative {mpp:.2f} m/pixel

Cette page décrit une séquence de {count} figures de vol, illustrées sur
l'image satellite jointe. Les trajectoires sont **stylisées** : centrées sur
l'adresse géocodée et mises à l'échelle de l'image, elles ne sont pas
mesurées sur la géométrie réelle du bâtiment. Chaque figure combine une
distance (rayon), une plage d'altitude et une technique de pilotage — c'est
cette combinaison qui définit le niveau de difficulté du plan, autrement dit
le « skill » de drone qu'il demande.
"""

_MANEUVER = """## {index}. {name}

- **Rôle dans la vidéo** : {purpose}
- **Rayon** : {r_min:.0f} à {r_max:.0f} m du centre
- **Altitude** : {a_min:.0f} à {a_max:.0f} m
- **Longueur du tracé** : environ {length:.0f} m
- **Technique de pilotage** : {skill}
"""

_OUTRO = """
---

Ces figures sont des gabarits par défaut, pas un plan de vol mesuré. Pour les
ajuster à un bâtiment particulier une fois sa taille connue, modifie les
rayons et altitudes dans `simple_mode/maneuvers.py`.
"""


def build_report(
    *,
    address: str,
    lat: float,
    lon: float,
    zoom: int,
    mpp: float,
    maneuvers: list[Maneuver],
) -> str:
    """Compose le rapport Markdown décrivant la séquence de figures de vol."""
    parts = [
        _INTRO.format(address=address, lat=lat, lon=lon, zoom=zoom, mpp=mpp, count=len(maneuvers))
    ]
    for index, maneuver in enumerate(maneuvers, start=1):
        r_min, r_max = maneuver.radius_range_m
        a_min, a_max = maneuver.altitude_range_m
        parts.append(
            _MANEUVER.format(
                index=index,
                name=maneuver.name_fr,
                purpose=maneuver.purpose_fr,
                r_min=r_min,
                r_max=r_max,
                a_min=a_min,
                a_max=a_max,
                length=maneuver.path_length_m,
                skill=maneuver.skill_fr,
            )
        )
    parts.append(_OUTRO)
    return "\n".join(parts)


__all__ = ["build_report"]
