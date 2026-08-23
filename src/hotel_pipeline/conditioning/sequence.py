"""Trajectoire orbitale et export de la séquence de conditionnement.

Le module produit ce qu'un générateur vidéo conditionné consomme : une suite de
cartes cohérentes entre elles, plus le rapport qui dit où la géométrie tient et
où elle cède la main.

C'est là que se décide le repli discuté au niveau produit : chaque frame porte
son `guidance_mode`, et une frame dont la géométrie ne vaut rien le déclare
plutôt que d'imposer une contrainte fausse au générateur.
"""

from __future__ import annotations

import os

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..logging import get_logger
from .png import (
    confidence_to_png,
    depth_to_png,
    normal_to_png,
    silhouette_to_png,
    write_png,
)
from .render import Camera, RenderedFrame, render_frame
from .support import SupportMap
from .scene import ConditioningScene

log = get_logger("conditioning-sequence")

#: En deçà, la géométrie ne contraint plus rien d'utile : mieux vaut le dire.
MIN_TARGET_COVERAGE = 0.02
#: Au delà, l'image est dominée par des volumes de hauteur supposée.
MAX_ASSUMED_FRACTION = 0.60

#: En deçà, aucune référence photographique ne montre cette face du bâtiment :
#: le générateur y inventera la façade, quelle que soit la qualité de la
#: géométrie. Les deux appuis sont distincts et le second ne se déduit pas du
#: premier.
MIN_PHOTO_SUPPORT = 0.25

#: Marge d'ouverture au-dessus du toit, en fraction de la hauteur de la cible.
#: Un multiple simple ne convient pas aux deux échelles : à 3,6×, un motel de
#: 12 m ouvre correctement à 43 m tandis qu'une tour de 38 m partirait à 137 m,
#: d'où la cible ne pèse plus que 7 % de l'image. La marge est donc additive et
#: dégressive — elle compte surtout pour les bâtiments bas.
START_CLEARANCE_M = 30.0
START_CLEARANCE_RATIO = 0.35

#: Altitude d'arrivée, à hauteur de regard sur la façade plutôt qu'au sol.
END_ALTITUDE_RATIO = 0.5


@dataclass
class FrameRecord:
    """Ce qu'une frame établit, et ce qu'elle autorise au générateur."""

    index: int
    bearing_deg: float
    altitude_m: float
    distance_m: float
    stats: dict
    guidance_mode: str
    guidance_reason: str
    #: Appui photographique de cet angle, et la référence qui l'explique.
    photo_support: float = 1.0
    nearest_reference: str | None = None
    files: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "bearing_deg": round(self.bearing_deg, 2),
            "altitude_m": round(self.altitude_m, 2),
            "distance_m": round(self.distance_m, 2),
            "guidance_mode": self.guidance_mode,
            "guidance_reason": self.guidance_reason,
            "photo_support": round(self.photo_support, 3),
            "nearest_reference": self.nearest_reference,
            **self.stats,
            "files": self.files,
        }


@dataclass
class SequenceResult:
    """La séquence produite, et son verdict d'ensemble."""

    hotel_id: str
    output_dir: Path
    frames: list[FrameRecord]
    scene_summary: dict
    parameters: dict
    support_summary: dict | None = None
    environment_summary: dict | None = None

    @property
    def strong_fraction(self) -> float:
        if not self.frames:
            return 0.0
        strong = sum(1 for f in self.frames if f.guidance_mode == "geometry_strong")
        return strong / len(self.frames)

    @property
    def unreferenced_fraction(self) -> float:
        """Part du plan qu'aucune photographie n'appuie."""
        if not self.frames:
            return 0.0
        return sum(1 for f in self.frames if f.guidance_mode == "unreferenced") / len(
            self.frames
        )

    def verdict(self) -> str:
        """Ce que la géométrie autorise, pour l'ensemble du plan."""
        if not self.frames:
            return "unusable"
        # Un plan majoritairement non appuyé se dit tel quel, même quand sa
        # géométrie est solide : c'est l'apparence qui sera inventée.
        if self.unreferenced_fraction > 0.5:
            return "unreferenced_arc"
        if self.strong_fraction >= 0.8:
            return "condition_strongly"
        if self.strong_fraction >= 0.4:
            return "condition_partially"
        return "prefer_ungrounded"

    def as_dict(self) -> dict:
        return {
            "hotel_id": self.hotel_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "verdict": self.verdict(),
            "strong_fraction": round(self.strong_fraction, 3),
            "unreferenced_fraction": round(self.unreferenced_fraction, 3),
            "frame_count": len(self.frames),
            "support": self.support_summary,
            "environment": self.environment_summary,
            "parameters": self.parameters,
            "scene": self.scene_summary,
            "frames": [f.as_dict() for f in self.frames],
            "caveats": [
                "la hauteur des volumes vient d'une hypothèse quand aucune "
                "mesure ne la couvre : la silhouette verticale est alors "
                "indicative, l'emprise au sol seule étant attestée. Les "
                "volumes mesurés au nDSM portent leur source dans la scène.",
                "aucune source au sol n'atteste un toit : ces pixels sont "
                "déclassés dans le masque de confiance",
                "une frame 'prefer_ungrounded' ne doit pas être imposée au "
                "générateur : une géométrie fausse contraint pire qu'aucune",
                "une frame 'unreferenced' a une géométrie solide mais aucune "
                "photographie de cette face : l'apparence y sera inventée, "
                "même si la profondeur est juste",
                "les massifs végétaux bornent un encombrement mesuré, non une "
                "espèce ni un feuillage : ils disent où de la végétation existe "
                "et jusqu'à quelle hauteur, pas à quoi elle ressemble",
            ],
        }


def _grade(frame: RenderedFrame, photo_support: float | None) -> tuple[str, str]:
    """Décide ce que cette frame autorise au générateur.

    Deux appuis sont pesés séparément : la géométrie contraint la profondeur et
    les occultations, la photographie atteste l'apparence. Une frame peut être
    géométriquement irréprochable et n'être appuyée par aucune image — le
    générateur y inventera la façade, et c'est ce que le mode doit dire.
    """
    if not frame.hit_any or frame.target_coverage < MIN_TARGET_COVERAGE:
        return (
            "prefer_ungrounded",
            f"la cible n'occupe que {frame.target_coverage:.1%} de l'image : "
            "la géométrie ne contraint rien d'utile",
        )
    if frame.assumed_fraction > MAX_ASSUMED_FRACTION:
        return (
            "geometry_weak",
            f"{frame.assumed_fraction:.1%} des pixels reposent sur une hauteur "
            "supposée : contraindre la profondeur imposerait une erreur",
        )
    if photo_support is not None and photo_support < MIN_PHOTO_SUPPORT:
        return (
            "unreferenced",
            f"géométrie solide mais appui photographique de {photo_support:.0%} : "
            "aucune référence ne montre cette face, le générateur l'inventera",
        )
    return (
        "geometry_strong",
        f"cible sur {frame.target_coverage:.1%} de l'image, géométrie attestée "
        "dominante",
    )


def orbit_camera(
    scene: ConditioningScene,
    bearing_deg: float,
    altitude_m: float,
    distance_m: float,
    fov_deg: float,
    width: int,
    height: int,
) -> Camera:
    """Place la caméra sur l'orbite, visant le haut du bâtiment cible."""
    cx, cy = scene.centre
    theta = math.radians(bearing_deg)
    position = np.array([
        cx + distance_m * math.sin(theta),
        cy + distance_m * math.cos(theta),
        altitude_m,
    ])
    target_prism = scene.target
    look_z = (target_prism.height_m * 0.55) if target_prism else 5.0
    return Camera(
        position=position,
        target=np.array([cx, cy, look_z]),
        fov_deg=fov_deg,
        width=width,
        height=height,
    )


@dataclass
class _FrameJob:
    """Ce qu'une image a besoin de savoir, identique d'une pose à l'autre.

    Regroupé pour n'être transmis qu'une fois par processus : la scène et son
    environnement pèsent lourd, et les réexpédier à chaque image annulerait le
    gain de la répartition.
    """

    scene: ConditioningScene
    environment: object
    support: SupportMap | None
    output_dir: Path
    distance: float
    fov_deg: float
    width: int
    height: int
    near: float
    far: float
    write_images: bool


#: Job du processus courant, posé une fois à l'ouverture du pool.
_WORKER_JOB: "_FrameJob | None" = None


def _worker_init(job: "_FrameJob") -> None:
    global _WORKER_JOB
    _WORKER_JOB = job


def _worker_render(pose: tuple[int, float, float]) -> "FrameRecord":
    assert _WORKER_JOB is not None
    return _render_one(_WORKER_JOB, pose)


def _render_one(job: "_FrameJob", pose: tuple[int, float, float]) -> "FrameRecord":
    """Rend une image et écrit ses cartes. Sans effet sur les autres poses."""
    index, bearing, altitude = pose
    camera = orbit_camera(
        job.scene, bearing, altitude, job.distance, job.fov_deg, job.width, job.height
    )
    frame = render_frame(job.scene, camera, job.environment)
    photo_support, nearest = (
        job.support.support_at(bearing) if job.support is not None else (None, None)
    )
    mode, reason = _grade(frame, photo_support)

    files: dict[str, str] = {}
    if job.write_images:
        stem = f"{index:04d}.png"
        write_png(
            job.output_dir / "depth" / stem,
            depth_to_png(frame.depth, job.near, job.far),
        )
        write_png(job.output_dir / "normal" / stem, normal_to_png(frame.normal))
        write_png(
            job.output_dir / "silhouette" / stem, silhouette_to_png(frame.silhouette)
        )
        write_png(
            job.output_dir / "confidence" / stem, confidence_to_png(frame.confidence)
        )
        files = {
            "depth": f"depth/{stem}",
            "normal": f"normal/{stem}",
            "silhouette": f"silhouette/{stem}",
            "confidence": f"confidence/{stem}",
        }

    return FrameRecord(
        index=index,
        bearing_deg=bearing % 360.0,
        altitude_m=altitude,
        distance_m=job.distance,
        stats=frame.stats(),
        guidance_mode=mode,
        guidance_reason=reason,
        photo_support=1.0 if photo_support is None else photo_support,
        nearest_reference=nearest,
        files=files,
    )


def _worker_count(requested: int | None, frames: int) -> int:
    """Combien de processus ouvrir, sans en ouvrir pour rien."""
    if requested is not None:
        return max(1, min(requested, frames))
    # Les cœurs physiques, non les logiques : le rastériseur sature déjà son
    # cœur, et deux fils sur le même n'y ajoutent rien.
    available = os.cpu_count() or 1
    return max(1, min(available // 2 or 1, frames))


def _render_poses(
    job: "_FrameJob", poses: list[tuple[int, float, float]], workers: int | None
) -> list["FrameRecord"]:
    """Rend toutes les poses, réparties sur les cœurs disponibles.

    Un seul processus demandé — ou une seule image — reste en direct : ouvrir
    un pool pour cela coûterait plus que le calcul lui-même. Un pool qui refuse
    de démarrer n'arrête pas le rendu : on retombe sur la boucle séquentielle,
    plus lente mais sûre.
    """
    count = _worker_count(workers, len(poses))
    if count <= 1 or len(poses) <= 1:
        return [_render_one(job, pose) for pose in poses]

    try:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(
            max_workers=count, initializer=_worker_init, initargs=(job,)
        ) as pool:
            records = list(pool.map(_worker_render, poses, chunksize=1))
    except Exception as exc:  # noqa: BLE001 - repli délibérément large
        log.warning(
            "rendu réparti indisponible (%s) : retour au rendu séquentiel", exc
        )
        return [_render_one(job, pose) for pose in poses]

    log.info("rendu réparti sur %d processus", count)
    # `pool.map` conserve l'ordre, mais l'index le dit sans avoir à le supposer.
    return sorted(records, key=lambda r: r.index)


def render_sequence(
    scene: ConditioningScene,
    output_dir: Path,
    frame_count: int = 90,
    arc_deg: float = 180.0,
    start_bearing_deg: float = 200.0,
    start_altitude_m: float | None = None,
    end_altitude_m: float | None = None,
    distance_factor: float = 1.35,
    fov_deg: float = 60.0,
    width: int = 512,
    height: int = 288,
    write_images: bool = True,
    support: SupportMap | None = None,
    environment=None,  # noqa: ANN001
    workers: int | None = None,
) -> SequenceResult:
    """Rend l'orbite descendante et écrit les cartes de conditionnement."""
    output_dir = Path(output_dir)
    for name in ("depth", "normal", "silhouette", "confidence"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    radius = scene.radius_m()
    # Le cadrage se règle sur le rayon, pas sur une distance absolue : un motel
    # bas et long et une tour doivent tous deux remplir l'image.
    distance = max(radius * distance_factor, radius + 10.0)

    # L'altitude aussi se dérive du bâtiment. Des valeurs absolues réglées sur
    # un motel de 12 m — départ à 45 m, arrivée à 6 m — décrivent une descente
    # tout autre sur une tour de 38 m : la caméra finirait sous le tiers de sa
    # hauteur et la cible saturerait l'image. La trajectoire est donc exprimée
    # en fractions de la hauteur de la cible.
    target = scene.target
    target_height = target.height_m if target else 12.0
    if start_altitude_m is None:
        start_altitude_m = (
            target_height * (1.0 + START_CLEARANCE_RATIO) + START_CLEARANCE_M
        )
    if end_altitude_m is None:
        end_altitude_m = max(target_height * END_ALTITUDE_RATIO, 3.0)
    near, far = max(distance - radius * 2.0, 1.0), distance + radius * 2.5

    poses = []
    for i in range(frame_count):
        t = i / max(frame_count - 1, 1)
        bearing = start_bearing_deg + arc_deg * t
        # Descente en ease-out : l'approche ralentit près du sol.
        altitude = end_altitude_m + (start_altitude_m - end_altitude_m) * (1.0 - t) ** 1.6
        poses.append((i, bearing, altitude))

    job = _FrameJob(
        scene=scene,
        environment=environment,
        support=support,
        output_dir=output_dir,
        distance=distance,
        fov_deg=fov_deg,
        width=width,
        height=height,
        near=near,
        far=far,
        write_images=write_images,
    )

    # Les images sont indépendantes : chacune part d'une pose et n'écrit que
    # ses propres fichiers. Les répartir sur les cœurs disponibles est le seul
    # gain qui reste sans toucher au rastériseur lui-même.
    records = _render_poses(job, poses, workers)

    result = SequenceResult(
        hotel_id=scene.hotel_id,
        output_dir=output_dir,
        frames=records,
        scene_summary=scene.summary(),
        support_summary=support.as_dict() if support is not None else None,
        environment_summary=(
            None if environment is None else {
                "vegetation_count": len(environment.patches),
                "by_stratum": environment.by_stratum(),
                "linked_buildings": len(environment.linked),
            }
        ),
        parameters={
            "frame_count": frame_count,
            "arc_deg": arc_deg,
            "start_bearing_deg": start_bearing_deg,
            "start_altitude_m": start_altitude_m,
            "end_altitude_m": end_altitude_m,
            "distance_m": round(distance, 2),
            "fov_deg": fov_deg,
            "resolution": [width, height],
            "depth_near_m": round(near, 2),
            "depth_far_m": round(far, 2),
        },
    )

    report = output_dir / "conditioning_report.json"
    report.write_text(
        json.dumps(result.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info(
        "séquence rendue : %d frames, verdict %s (%.0f%% fortement contraintes)",
        len(records),
        result.verdict(),
        result.strong_fraction * 100,
    )
    return result
