"""Rendu d'un vol de drone sur la géométrie 3D réelle (Google Photorealistic 3D Tiles).

Contrairement à la voie ``sogni_cli`` (deux photos + un prompt, où le modèle
invente tout l'intermédiaire), ici **la trajectoire calculée est réellement
appliquée** : chaque waypoint de ``maneuvers`` devient une pose de caméra
géoréférencée dans CesiumJS, qui diffuse les tuiles 3D photoréalistes de
Google. Le rendu est déterministe — même trajectoire, même vidéo — et la
géométrie des bâtiments est mesurée, pas hallucinée.

Chaîne : waypoints (est/nord/altitude) -> poses (lat/lon/hauteur, cap,
tangage) -> page ``cesium_scene.html`` pilotée par Playwright -> captures
PNG -> assemblage ffmpeg.

Prérequis ::

    pip install playwright && python -m playwright install chromium
    winget install --id Gyan.FFmpeg

La clé ``GOOGLE_MAPS_API_KEY`` doit avoir accès à l'API Map Tiles (tester
avec ``https://tile.googleapis.com/v1/3dtiles/root.json?key=...`` : un
HTTP 200 confirme l'accès).

Limite connue : les tuiles 3D n'ont **pas d'intérieur**. Traverser une
façade montre le maillage vu de l'envers, c'est-à-dire du vide — l'effet de
traversée doit rester une passe séparée (IA ou transition stylisée), pas un
simple waypoint qui rentre dans le bâtiment.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .geo_utils import offset_to_latlon
from .maneuvers import Maneuver, Waypoint

SCENE_HTML = Path(__file__).with_name("cesium_scene.html")

DEFAULT_FPS = 24
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720

#: Images rendues par seconde de vidéo finale ; le nombre de poses est dérivé
#: de la durée souhaitée, pas des waypoints bruts (qui sont irréguliers).
DEFAULT_DURATION_S = 20

#: Au-delà, on considère que les tuiles ne se chargeront pas pour cette pose.
TILE_TIMEOUT_MS = 20000

#: Enveloppe de vol utilisable sur des tuiles photogrammétriques.
#:
#: Ces tuiles sont reconstruites depuis des prises de vue aériennes : vues de
#: loin et d'en haut elles sont photoréalistes, mais de près la reconstruction
#: s'effondre — les arbres deviennent des masses informes et les façades se
#: délavent. Mesuré sur cette scène : net à 45 m d'altitude / 55 m de
#: distance, inexploitable à 27 m / 15 m. Ces bornes gardent la **forme** de
#: la trajectoire tout en la maintenant dans l'enveloppe où la géométrie
#: tient. Elles bornent aussi le cas dégénéré des altitudes négatives que le
#: chaînage (``maneuvers.chain_maneuvers``) peut produire — un drone ne vole
#: pas sous terre.
MIN_ALTITUDE_M = 30.0

#: Distance horizontale minimale au centre du sujet, même motif.
MIN_RADIUS_M = 45.0

#: L'altitude du sol est mesurée sur les tuiles 3D elles-mêmes, dans la
#: scène (voir `cesium_scene.html`). L'API Google Elevation a été écartée :
#: elle renvoie une altitude au-dessus du géoïde, quand Cesium attend une
#: hauteur au-dessus de l'ellipsoïde — un écart d'une trentaine de mètres
#: selon les régions, suffisant pour transformer un vol rasant en vue
#: d'horizon.


class CesiumRenderError(RuntimeError):
    """Le rendu 3D a échoué."""


@dataclass
class CameraPose:
    lat: float
    lon: float
    #: Hauteur **au-dessus du sol**. Le passage en hauteur absolue se fait
    #: dans la scène, à partir du sol mesuré sur les tuiles 3D — voir
    #: `cesium_scene.html`.
    agl: float
    heading: float
    pitch: float

    def as_dict(self) -> dict:
        return {
            "lat": self.lat,
            "lon": self.lon,
            "agl": round(self.agl, 2),
            "heading": round(self.heading, 2),
            "pitch": round(self.pitch, 2),
        }


def _flatten(maneuvers: list[Maneuver]) -> list[Waypoint]:
    points: list[Waypoint] = []
    for maneuver in maneuvers:
        for waypoint in maneuver.waypoints:
            # Les figures chaînées partagent leur point de jonction : le
            # garder deux fois figerait la caméra une image sur deux.
            if points and (
                abs(points[-1].east_m - waypoint.east_m) < 1e-6
                and abs(points[-1].north_m - waypoint.north_m) < 1e-6
                and abs(points[-1].altitude_m - waypoint.altitude_m) < 1e-6
            ):
                continue
            points.append(waypoint)
    return points


def _resample_path(points: list[Waypoint], count: int) -> list[Waypoint]:
    """``count`` positions régulièrement espacées **en distance parcourue**.

    Échantillonner les waypoints bruts donnerait une vitesse irrégulière :
    les figures n'ont pas la même densité de points par mètre.
    """
    if len(points) < 2 or count < 2:
        return points[:count] or points

    cumulative = [0.0]
    for a, b in zip(points, points[1:]):
        cumulative.append(cumulative[-1] + math.dist((a.east_m, a.north_m), (b.east_m, b.north_m)))
    total = cumulative[-1]
    if total <= 0:
        return [points[0]] * count

    sampled: list[Waypoint] = []
    segment = 0
    for i in range(count):
        target = total * i / (count - 1)
        while segment < len(points) - 2 and cumulative[segment + 1] < target:
            segment += 1
        span = cumulative[segment + 1] - cumulative[segment]
        t = 0.0 if span <= 0 else (target - cumulative[segment]) / span
        a, b = points[segment], points[segment + 1]
        sampled.append(
            Waypoint(
                a.east_m + (b.east_m - a.east_m) * t,
                a.north_m + (b.north_m - a.north_m) * t,
                a.altitude_m + (b.altitude_m - a.altitude_m) * t,
            )
        )
    return sampled


def build_poses(
    maneuvers: list[Maneuver],
    lat: float,
    lon: float,
    *,
    frame_count: int,
    target_height_m: float = 15.0,
) -> list[CameraPose]:
    """Convertit la trajectoire en poses caméra visant le bâtiment.

    Le cap pointe vers le centre géocodé et le tangage est calculé depuis la
    hauteur relative : la caméra regarde donc réellement le sujet à chaque
    image, au lieu de garder une orientation fixe.
    """
    points = _resample_path(_flatten(maneuvers), frame_count)

    poses: list[CameraPose] = []
    for waypoint in points:
        east, north = waypoint.east_m, waypoint.north_m

        # Repousse la pose hors du rayon minimal, en conservant son azimut :
        # la figure garde sa forme, seule son échelle est bornée.
        radius = math.hypot(east, north)
        if radius < MIN_RADIUS_M:
            if radius < 1e-6:
                east, north = 0.0, MIN_RADIUS_M
            else:
                scale = MIN_RADIUS_M / radius
                east, north = east * scale, north * scale

        p_lat, p_lon = offset_to_latlon(lat, lon, east, north)

        # Cap : depuis la position vers le centre (0,0) du repère local.
        heading = math.degrees(math.atan2(-east, -north)) % 360.0

        altitude = max(MIN_ALTITUDE_M, waypoint.altitude_m)

        ground_distance = math.hypot(east, north)
        rise = altitude - target_height_m
        pitch = -math.degrees(math.atan2(rise, max(1.0, ground_distance)))
        pitch = max(-89.0, min(10.0, pitch))

        poses.append(
            CameraPose(lat=p_lat, lon=p_lon, agl=altitude, heading=heading, pitch=pitch)
        )
    return poses


def render_flight(
    poses: list[CameraPose],
    api_key: str,
    *,
    centre: tuple[float, float],
    out_path: str | Path,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: int = DEFAULT_FPS,
    keep_frames: bool = False,
    progress=None,  # noqa: ANN001 — callable(index, total, note="") optionnel
) -> Path:
    """Rend la séquence de poses en une vidéo, via Chromium headless + ffmpeg."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - dépend de l'installation
        raise CesiumRenderError(
            "playwright manquant. Installer : pip install playwright "
            "&& python -m playwright install chromium"
        ) from exc

    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        raise CesiumRenderError(
            "ffmpeg introuvable — requis pour assembler les images. "
            "Installer : winget install --id Gyan.FFmpeg"
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = out_path.parent / f"{out_path.stem}_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            args=[
                # WebGL logiciel : indispensable en headless sans GPU, sinon
                # Cesium ne rend qu'un fond noir.
                "--use-gl=angle",
                "--use-angle=swiftshader",
                "--enable-unsafe-swiftshader",
                "--ignore-gpu-blocklist",
            ]
        )
        page = browser.new_page(viewport={"width": width, "height": height})
        page.goto(SCENE_HTML.as_uri())

        page.evaluate(
            "config => window.dronePrepare(config)",
            {
                "apiKey": api_key,
                "centre": {"lat": centre[0], "lon": centre[1]},
                "waypoints": [p.as_dict() for p in poses],
            },
        )
        page.wait_for_function("window.droneReady === true || window.droneError !== null", timeout=120000)
        error = page.evaluate("window.droneError")
        if error:
            browser.close()
            raise CesiumRenderError(f"initialisation de la scène 3D échouée : {error}")

        ground = page.evaluate("window.droneGroundHeight")
        source = page.evaluate("window.droneGroundSource")
        if progress is not None:
            progress(0, len(poses), f"sol mesuré à {ground:.1f} m ({source})")

        for index in range(len(poses)):
            page.evaluate(
                "args => window.droneSeek(args[0], args[1])", [index, TILE_TIMEOUT_MS]
            )
            # Le WebGL logiciel rend lentement en haute résolution : la valeur
            # par défaut de Playwright (30 s) expire sur des vues chargées.
            page.screenshot(path=str(frames_dir / f"frame_{index:05d}.png"), timeout=120000)
            if progress is not None:
                progress(index + 1, len(poses), "")

        browser.close()

    command = [
        ffmpeg, "-y", "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(out_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not out_path.exists():
        raise CesiumRenderError(f"assemblage ffmpeg échoué :\n{result.stderr[-2000:]}")

    if not keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)
    return out_path


def _find_ffmpeg() -> str | None:
    """Résout ffmpeg même absent du PATH (voir `sogni_cli._find_ffmpeg`)."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    import os

    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        packages = Path(local_appdata) / "Microsoft" / "WinGet" / "Packages"
        if packages.exists():
            matches = sorted(packages.glob("Gyan.FFmpeg*/**/bin/ffmpeg.exe"))
            if matches:
                return str(matches[0])
    return None


__all__ = [
    "DEFAULT_DURATION_S",
    "DEFAULT_FPS",
    "CameraPose",
    "CesiumRenderError",
    "build_poses",
    "render_flight",
]
