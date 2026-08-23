"""Trajectoire ancrée sur des points de vue **réels** (Lot 2).

L'orbite virtuelle de `scene_package` tourne autour d'une boîte proxy : elle
décrit un cadrage souhaitable, sans se demander si une image existe pour le
soutenir. Demander à un générateur vidéo un angle que personne n'a jamais
photographié revient à lui demander d'inventer — et c'est précisément ce que la
porte de fidélité cherche à interdire.

Ce module renverse la dépendance : **la trajectoire suit ce qu'on a vu.** Chaque
pose est adossée à un panorama réel, à un cap et à un champ dont la lecture
pixel a confirmé qu'ils montrent le bâtiment. Ce qui manque se dit — un arc sans
point de vue vérifié reste un trou déclaré, non une pose interpolée.

Le mouvement demandé — descente de 40 m vers le sol en tournant — n'est pas
réalisable par de l'imagerie de rue, qui vit à ~2,5 m. La descente est donc une
**consigne d'animation** portée par la trajectoire, tandis que les références
d'apparence viennent des vantages réels. Le générateur reçoit les deux, et le
paquet dit clairement lequel est mesuré et lequel est demandé.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .logging import get_logger

log = get_logger("camera-path")

#: Altitude de départ, en mètres au-dessus du sol.
START_ALTITUDE_M = 40.0

#: Altitude d'arrivée : hauteur d'œil, cohérente avec l'imagerie de rue.
END_ALTITUDE_M = 2.5

#: Écart angulaire au-delà duquel deux poses ne se recouvrent plus : le
#: mouvement y devient un saut, et le générateur doit inventer la transition.
CONTINUITY_GAP_DEG = 45.0


@dataclass
class PathPose:
    """Une pose de la trajectoire, et ce qui l'atteste."""

    index: int
    bearing_deg: float
    altitude_m: float
    distance_m: float
    #: Panorama qui atteste ce point de vue, quand il en existe un.
    panorama_id: str | None = None
    heading_deg: float | None = None
    fov_deg: float | None = None
    pitch_deg: float | None = None
    prominence: float | None = None
    facade_id: str | None = None

    @property
    def anchored(self) -> bool:
        """La pose s'appuie-t-elle sur une image réelle et vérifiée ?"""
        return self.panorama_id is not None and self.prominence is not None

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "bearing_deg": round(self.bearing_deg, 1),
            "altitude_m": round(self.altitude_m, 1),
            "distance_m": round(self.distance_m, 1),
            "anchored": self.anchored,
            "panorama_id": self.panorama_id,
            "heading_deg": round(self.heading_deg, 1) if self.heading_deg is not None else None,
            "fov_deg": round(self.fov_deg, 1) if self.fov_deg is not None else None,
            "pitch_deg": round(self.pitch_deg, 1) if self.pitch_deg is not None else None,
            "prominence": round(self.prominence, 4) if self.prominence is not None else None,
            "facade_id": self.facade_id,
        }


@dataclass
class RealCameraPath:
    """Trajectoire descendante, ancrée sur des vantages vérifiés."""

    hotel_id: str
    poses: list[PathPose] = field(default_factory=list)
    gaps: list[tuple[float, float]] = field(default_factory=list)
    start_altitude_m: float = START_ALTITUDE_M
    end_altitude_m: float = END_ALTITUDE_M

    @property
    def anchored_fraction(self) -> float:
        if not self.poses:
            return 0.0
        return sum(1 for p in self.poses if p.anchored) / len(self.poses)

    @property
    def arc_deg(self) -> float:
        """Étendue angulaire réellement parcourue par les poses ancrées."""
        bearings = sorted(p.bearing_deg for p in self.poses if p.anchored)
        if len(bearings) < 2:
            return 0.0
        gaps = [
            (bearings[(i + 1) % len(bearings)] - bearings[i]) % 360.0
            for i in range(len(bearings))
        ]
        return max(0.0, 360.0 - max(gaps))

    def verdict(self) -> str:
        if not self.poses:
            return "unsupported"
        if self.anchored_fraction < 0.5:
            return "mostly_unsupported"
        if self.gaps:
            return "discontinuous"
        return "continuous"

    def as_dict(self) -> dict:
        return {
            "hotel_id": self.hotel_id,
            "movement": "descending_orbit",
            "start_altitude_m": self.start_altitude_m,
            "end_altitude_m": self.end_altitude_m,
            "verdict": self.verdict(),
            "anchored_fraction": round(self.anchored_fraction, 3),
            "arc_deg": round(self.arc_deg, 1),
            "gaps_deg": [
                [round(a, 1), round(b, 1)] for a, b in self.gaps
            ],
            "poses": [p.as_dict() for p in self.poses],
            "caveats": [
                "l'altitude est une consigne d'animation : l'imagerie de rue "
                "vit à ~2,5 m et n'atteste aucune vue à 40 m",
                "une pose non ancrée n'a aucune référence réelle : le "
                "générateur y invente",
            ],
        }


def _bearing(origin: tuple[float, float], centre) -> float:  # noqa: ANN001
    return math.degrees(
        math.atan2(origin[0] - centre.x, origin[1] - centre.y)
    ) % 360.0


def build(
    hotel_id: str,
    scene,  # noqa: ANN001
    verifications: list[dict],
    *,
    poses: int = 24,
    minimum_prominence: float = 0.15,
) -> RealCameraPath:
    """Compose une orbite descendante à partir des vantages vérifiés.

    Les poses sont réparties régulièrement en azimut ; chacune adopte le
    vantage vérifié le plus proche en cap. Aucun vantage à portée laisse la
    pose **non ancrée** — elle reste dans la trajectoire pour que le trou soit
    visible, jamais comblée par une interpolation qui ferait croire à une
    référence.
    """
    centre = scene.footprint.centroid
    positions = {provider: origin for _asset, provider, origin in scene.viewpoints}

    anchors: list[tuple[float, dict, tuple[float, float]]] = []
    for row in verifications:
        score = row.get("score")
        if score is None or score < minimum_prominence:
            continue
        origin = positions.get(row.get("panorama_id"))
        if origin is None:
            continue
        anchors.append((_bearing(origin, centre), row, origin))

    if not anchors:
        log.warning("aucun vantage vérifié résolu en position : trajectoire non ancrée")

    path = RealCameraPath(hotel_id=hotel_id)
    step = 360.0 / max(1, poses)
    for index in range(poses):
        bearing = (index * step) % 360.0
        ratio = index / max(1, poses - 1)
        # Descente linéaire de 40 m vers la hauteur d'œil, en tournant.
        altitude = START_ALTITUDE_M + (END_ALTITUDE_M - START_ALTITUDE_M) * ratio

        best = None
        for anchor_bearing, row, origin in anchors:
            delta = abs((anchor_bearing - bearing + 180.0) % 360.0 - 180.0)
            if best is None or delta < best[0]:
                best = (delta, row, origin)

        if best is None or best[0] > CONTINUITY_GAP_DEG:
            path.poses.append(
                PathPose(
                    index=index, bearing_deg=bearing, altitude_m=altitude,
                    distance_m=0.0,
                )
            )
            continue

        _delta, row, origin = best
        path.poses.append(
            PathPose(
                index=index,
                bearing_deg=bearing,
                altitude_m=altitude,
                distance_m=math.hypot(origin[0] - centre.x, origin[1] - centre.y),
                panorama_id=row.get("panorama_id"),
                heading_deg=row.get("heading_deg"),
                fov_deg=row.get("fov_deg"),
                pitch_deg=row.get("pitch_deg"),
                prominence=row.get("score"),
                facade_id=row.get("facade_id"),
            )
        )

    path.gaps = _gaps(path.poses)
    log.info(
        "trajectoire %s : %d pose(s), %.0f%% ancrée(s), arc %.0f°",
        hotel_id, len(path.poses), 100 * path.anchored_fraction, path.arc_deg,
    )
    return path


def _gaps(poses: list[PathPose]) -> list[tuple[float, float]]:
    """Intervalles d'azimut sans aucune pose ancrée."""
    gaps: list[tuple[float, float]] = []
    start: float | None = None
    for pose in poses:
        if pose.anchored:
            if start is not None:
                gaps.append((start, pose.bearing_deg))
                start = None
        elif start is None:
            start = pose.bearing_deg
    if start is not None and poses:
        gaps.append((start, poses[0].bearing_deg))
    return gaps


def reference_requests(path: RealCameraPath) -> list[dict]:
    """Recadrages à télécharger pour servir de références au générateur.

    Dédupliqués par `(panorama, cap, champ)` : deux poses voisines partagent
    souvent le même vantage, et le payer deux fois n'ajoute aucune référence.
    """
    seen: set[tuple] = set()
    requests: list[dict] = []
    for pose in path.poses:
        if not pose.anchored:
            continue
        key = (
            pose.panorama_id,
            round((pose.heading_deg or 0.0) / 5.0),
            round((pose.fov_deg or 70.0) / 5.0),
        )
        if key in seen:
            continue
        seen.add(key)
        requests.append(
            {
                "panorama_id": pose.panorama_id,
                "heading_deg": pose.heading_deg,
                "fov_deg": pose.fov_deg,
                "pitch_deg": pose.pitch_deg or 0.0,
                "facade_id": pose.facade_id,
                "prominence": pose.prominence,
                "for_pose": pose.index,
            }
        )
    return requests


__all__ = [
    "CONTINUITY_GAP_DEG",
    "END_ALTITUDE_M",
    "START_ALTITUDE_M",
    "PathPose",
    "RealCameraPath",
    "build",
    "reference_requests",
]
