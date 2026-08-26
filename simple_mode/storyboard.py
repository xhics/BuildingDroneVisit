"""Assemble un story-board continu : un seul trajet + plusieurs références réelles.

Un trajet chaîné (voir ``maneuvers.chain_maneuvers``) est échantillonné en
``N`` points régulièrement espacés ; chacun reçoit une image réelle
d'ancrage (Street View si le point est bas et proche d'un panorama, sinon la
photo satellite). Le résultat est pensé pour **un seul appel** de génération
vidéo à plusieurs images de référence (ex : Seedance 2.5 sur Sogni accepte
jusqu'à 30 images de référence par appel), avec un unique prompt décrivant
tout le survol — pas des clips séparés à recoller ensuite. La continuité se
construit ici, dans la trajectoire et dans l'appel, pas au montage.

Le dernier point, la « traversée du bâtiment », n'a par construction aucune
photo réelle possible (aucune image ne montre l'intérieur d'un mur) : il
réutilise la dernière référence réelle comme simple ancrage de style, marqué
``reference_kind="generatif"`` plutôt que présenté comme une observation.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .geo_utils import offset_to_latlon
from .maneuvers import Maneuver, Waypoint
from .street_view import Panorama, download_image, find_panoramas, nearest_to

#: Au-delà, un panorama piéton n'est pas une référence pertinente : on
#: retombe sur la photo satellite.
STREET_VIEW_MAX_ALTITUDE_M = 12.0

#: Marge sous la limite de 30 images de référence du modèle Seedance 2.5.
MAX_KEYFRAMES = 20

ASSUMED_SPEED_MPS = 5.0
MIN_TOTAL_DURATION_S = 15.0
#: Plafond du modèle vidéo visé (Seedance 2.5 : 4-30 s par clip, confirmé
#: dans le code source public de sogni-agent) — voir sogni_cli.py.
MAX_TOTAL_DURATION_S = 30.0
FLYTHROUGH_DURATION_S = 4.0

THROUGH_BUILDING_ID = "traversee_finale"


@dataclass
class Keyframe:
    order: int
    maneuver_id: str
    east_m: float
    north_m: float
    altitude_m: float
    reference_image: str
    reference_kind: str  # "street_view" | "satellite" | "generatif"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ContinuousStoryboard:
    address: str
    keyframes: list[Keyframe]
    master_prompt_fr: str
    total_duration_s: float

    def save(self, path: str | Path) -> None:
        payload = {
            "address": self.address,
            "master_prompt_fr": self.master_prompt_fr,
            "total_duration_s": self.total_duration_s,
            "keyframes": [k.as_dict() for k in self.keyframes],
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _flatten_segments(maneuvers: list[Maneuver]) -> list[tuple[Maneuver, Waypoint, Waypoint, float, float]]:
    """Liste plate des segments du trajet, avec la distance cumulée à leur début."""
    flat = []
    cumulative = 0.0
    for maneuver in maneuvers:
        waypoints = maneuver.waypoints
        for a, b in zip(waypoints, waypoints[1:]):
            seg_len = math.dist((a.east_m, a.north_m), (b.east_m, b.north_m))
            if seg_len <= 0:
                continue
            flat.append((maneuver, a, b, cumulative, seg_len))
            cumulative += seg_len
    return flat


def _sample_positions(maneuvers: list[Maneuver], *, count: int) -> list[tuple[Maneuver, Waypoint]]:
    """``count`` points régulièrement espacés le long du trajet (déjà chaîné).

    Le premier point échantillonné est le tout début du trajet, le dernier
    en est la toute fin — jamais tronqués par l'espacement.
    """
    if not maneuvers:
        return []
    if count <= 1:
        return [(maneuvers[0], maneuvers[0].waypoints[0])]

    flat = _flatten_segments(maneuvers)
    if not flat:
        return [(maneuvers[0], maneuvers[0].waypoints[0])]

    total_length = flat[-1][3] + flat[-1][4]
    targets = [i * total_length / (count - 1) for i in range(count)]

    samples: list[tuple[Maneuver, Waypoint]] = []
    seg_idx = 0
    for target in targets:
        while seg_idx < len(flat) - 1 and target > flat[seg_idx][3] + flat[seg_idx][4]:
            seg_idx += 1
        maneuver, a, b, start_cum, seg_len = flat[seg_idx]
        t = 0.0 if seg_len <= 0 else max(0.0, min(1.0, (target - start_cum) / seg_len))
        waypoint = Waypoint(
            a.east_m + (b.east_m - a.east_m) * t,
            a.north_m + (b.north_m - a.north_m) * t,
            a.altitude_m + (b.altitude_m - a.altitude_m) * t,
        )
        samples.append((maneuver, waypoint))
    return samples


def _resolve_reference(
    waypoint: Waypoint,
    lat: float,
    lon: float,
    panoramas: list[Panorama],
    api_key_maps: str,
    out_dir: Path,
    downloaded: dict[str, Path],
    satellite_path: Path,
) -> tuple[str, str]:
    if waypoint.altitude_m <= STREET_VIEW_MAX_ALTITUDE_M and panoramas:
        target_lat, target_lon = offset_to_latlon(lat, lon, waypoint.east_m, waypoint.north_m)
        nearest = nearest_to(panoramas, target_lat, target_lon, max_distance_m=45.0)
        if nearest is not None:
            if nearest.pano_id not in downloaded:
                dest = out_dir / f"streetview_{nearest.pano_id}.jpg"
                if not dest.exists():
                    download_image(nearest, api_key_maps, dest)
                downloaded[nearest.pano_id] = dest
            return str(downloaded[nearest.pano_id]), "street_view"
    return str(satellite_path), "satellite"


def _master_prompt(address: str, maneuvers: list[Maneuver], *, include_flythrough: bool) -> str:
    steps_fr = "; ".join(f"{i + 1}) {m.purpose_fr}" for i, m in enumerate(maneuvers))
    prompt = (
        f"Vol de drone continu et ininterrompu autour de « {address} ». "
        f"Une seule prise, sans coupure ni arrêt entre les mouvements : {steps_fr}."
    )
    if include_flythrough:
        prompt += (
            " Le vol se termine en traversant la façade du bâtiment comme si "
            "les murs devenaient transparents, révélant un intérieur cohérent "
            "avec le style architectural extérieur, avant de ressortir de "
            "l'autre côté."
        )
    prompt += (
        " Mouvement fluide et stabilisé de bout en bout, vitesse constante, "
        "aucune coupure de montage, cohérent avec les images de référence fournies."
    )
    return prompt


def build_continuous_storyboard(
    *,
    address: str,
    lat: float,
    lon: float,
    satellite_path: str | Path,
    maneuvers: list[Maneuver],
    api_key_maps: str,
    out_dir: str | Path,
    max_keyframes: int = MAX_KEYFRAMES,
    include_building_flythrough: bool = True,
) -> ContinuousStoryboard:
    """Construit un trajet continu et ses images de référence.

    ``maneuvers`` doit déjà être chaîné (``maneuvers.chain_maneuvers``) :
    ce module échantillonne le trajet tel quel, il ne referme pas les sauts.
    """
    out_dir = Path(out_dir)
    satellite_path = Path(satellite_path)
    panoramas = find_panoramas(lat, lon, api_key_maps)
    downloaded: dict[str, Path] = {}

    reserved = 1 if include_building_flythrough else 0
    samples = _sample_positions(maneuvers, count=max(2, max_keyframes - reserved))

    keyframes: list[Keyframe] = []
    for order, (maneuver, waypoint) in enumerate(samples):
        reference_image, reference_kind = _resolve_reference(
            waypoint, lat, lon, panoramas, api_key_maps, out_dir, downloaded, satellite_path
        )
        keyframes.append(
            Keyframe(
                order=order,
                maneuver_id=maneuver.id,
                east_m=round(waypoint.east_m, 1),
                north_m=round(waypoint.north_m, 1),
                altitude_m=round(waypoint.altitude_m, 1),
                reference_image=reference_image,
                reference_kind=reference_kind,
            )
        )

    if include_building_flythrough and keyframes:
        last = keyframes[-1]
        keyframes.append(
            Keyframe(
                order=len(keyframes),
                maneuver_id=THROUGH_BUILDING_ID,
                east_m=last.east_m,
                north_m=last.north_m,
                altitude_m=last.altitude_m,
                reference_image=last.reference_image,
                reference_kind="generatif",
            )
        )

    total_length = sum(m.path_length_m for m in maneuvers)
    total_duration = total_length / ASSUMED_SPEED_MPS
    if include_building_flythrough:
        total_duration += FLYTHROUGH_DURATION_S
    total_duration = max(MIN_TOTAL_DURATION_S, min(MAX_TOTAL_DURATION_S, total_duration))

    return ContinuousStoryboard(
        address=address,
        keyframes=keyframes,
        master_prompt_fr=_master_prompt(address, maneuvers, include_flythrough=include_building_flythrough),
        total_duration_s=round(total_duration, 1),
    )


__all__ = [
    "MAX_KEYFRAMES",
    "THROUGH_BUILDING_ID",
    "ContinuousStoryboard",
    "Keyframe",
    "build_continuous_storyboard",
]
