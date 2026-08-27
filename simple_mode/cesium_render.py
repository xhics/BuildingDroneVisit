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


def _flatten_with_mode(maneuvers: list[Maneuver]) -> list[tuple[Waypoint, str]]:
    """Points du trajet, chacun avec le mode de visée de sa figure."""
    out: list[tuple[Waypoint, str]] = []
    for maneuver in maneuvers:
        for waypoint in maneuver.waypoints:
            if out and (
                abs(out[-1][0].east_m - waypoint.east_m) < 1e-6
                and abs(out[-1][0].north_m - waypoint.north_m) < 1e-6
                and abs(out[-1][0].altitude_m - waypoint.altitude_m) < 1e-6
            ):
                continue
            out.append((waypoint, maneuver.heading_mode))
    return out


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
        cumulative.append(cumulative[-1] + math.dist(
            (a.east_m, a.north_m, a.altitude_m),
            (b.east_m, b.north_m, b.altitude_m),
        ))
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
    enforce_envelope: bool = True,
) -> list[CameraPose]:
    """Convertit la trajectoire en poses caméra.

    Le tangage est calculé depuis la hauteur relative, et le cap dépend du
    ``heading_mode`` de chaque figure : braqué sur le sujet pour les orbites,
    dans le sens du vol pour une traversée.

    ``enforce_envelope`` applique les bornes de qualité photogrammétrique ;
    les segments de traversée doivent s'en affranchir pour pouvoir cadrer une
    façade de près.
    """
    flat = _flatten_with_mode(maneuvers)
    source = [w for w, _m in flat]
    points = _resample_path(source, frame_count)

    # Le mode de visée suit la géométrie, pas l'indice : on retrouve pour
    # chaque point ré-échantillonné la figure d'origine la plus proche.
    def mode_at(index: int) -> str:
        if not flat:
            return "centre"
        target = points[index]
        best, best_d = 0, float("inf")
        for i, (w, _m) in enumerate(flat):
            d = (w.east_m - target.east_m) ** 2 + (w.north_m - target.north_m) ** 2 + (
                w.altitude_m - target.altitude_m
            ) ** 2
            if d < best_d:
                best, best_d = i, d
        return flat[best][1]

    poses: list[CameraPose] = []
    for index, waypoint in enumerate(points):
        east, north = waypoint.east_m, waypoint.north_m

        # Repousse la pose hors du rayon minimal, en conservant son azimut :
        # la figure garde sa forme, seule son échelle est bornée.
        radius = math.hypot(east, north)
        if enforce_envelope and radius < MIN_RADIUS_M:
            if radius < 1e-6:
                east, north = 0.0, MIN_RADIUS_M
            else:
                scale = MIN_RADIUS_M / radius
                east, north = east * scale, north * scale

        p_lat, p_lon = offset_to_latlon(lat, lon, east, north)

        if mode_at(index) == "trajet" and index < len(points) - 1:
            # Cap dans le sens du vol : sans cela, une fois le bâtiment
            # franchi, la caméra pivoterait pour le viser de nouveau.
            nxt = points[index + 1]
            heading = math.degrees(math.atan2(nxt.east_m - waypoint.east_m, nxt.north_m - waypoint.north_m)) % 360.0
        elif mode_at(index) == "trajet":
            prev = points[index - 1] if len(points) > 1 else waypoint
            heading = math.degrees(math.atan2(waypoint.east_m - prev.east_m, waypoint.north_m - prev.north_m)) % 360.0
        else:
            # Cap : depuis la position vers le centre (0,0) du repère local.
            heading = math.degrees(math.atan2(-east, -north)) % 360.0

        altitude = waypoint.altitude_m
        if enforce_envelope:
            altitude = max(MIN_ALTITUDE_M, altitude)
        altitude = max(2.0, altitude)

        if mode_at(index) == "trajet":
            # Vol de traversée : la caméra reste quasi horizontale pour
            # cadrer la façade, et non le toit — une visée plongeante donne
            # au générateur une vue de dessus, d'où l'« intérieur » qui
            # apparaît au-dessus du bâtiment au lieu de dedans.
            pitch = -4.0
        else:
            ground_distance = math.hypot(east, north)
            rise = altitude - target_height_m
            pitch = -math.degrees(math.atan2(rise, max(1.0, ground_distance)))
            pitch = max(-89.0, min(10.0, pitch))

        poses.append(
            CameraPose(lat=p_lat, lon=p_lon, agl=altitude, heading=heading, pitch=pitch)
        )
    return poses


#: Vitesse de la caméra, en m/s. Un survol cinématique posé tient plutôt
#: entre 8 et 15 m/s, mais un trajet de ~1,8 km y dure plus de deux minutes,
#: soit plusieurs heures de rendu. 25 m/s (90 km/h) reste dans le domaine
#: d'un drone rapide en ligne droite et divise la durée par deux ; au-delà,
#: le mouvement cesse d'être crédible et la photogrammétrie défile trop vite
#: pour être lue.
CRUISE_SPEED_MPS = 25.0


def path_length_m(maneuvers: list[Maneuver]) -> float:
    """Longueur du trajet, en 3D (les montées comptent)."""
    points = _flatten(maneuvers)
    return sum(
        math.dist(
            (a.east_m, a.north_m, a.altitude_m), (b.east_m, b.north_m, b.altitude_m)
        )
        for a, b in zip(points, points[1:])
    )


def allocate_frames(
    lengths: list[float], *, fps: int, cruise_speed_mps: float = CRUISE_SPEED_MPS
) -> list[int]:
    """Images à rendre par segment, pour une vitesse identique partout.

    Répartir les images à parts fixes (70/30, par exemple) alors que les
    segments n'ont pas la même longueur fait varier la vitesse au raccord —
    mesuré à ×1,5 entre l'avant et l'après d'une traversée. En dérivant le
    nombre d'images de la longueur et d'une vitesse commune, chaque segment
    parcourt la même distance par image, donc la caméra reprend exactement à
    l'allure où elle s'était arrêtée.
    """
    return [max(2, round(length / cruise_speed_mps * fps)) for length in lengths]


def render_depth(
    pose: CameraPose,
    api_key: str,
    *,
    centre: tuple[float, float],
    out_path: str | Path,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    tile_detail: float = 4.0,
) -> Path:
    """Rend une passe de profondeur pour une pose donnée.

    Sert à mesurer la couverture réelle du cadre : quels pixels reposent sur
    des tuiles fines, lesquels sur des tuiles lointaines, lesquels sur rien
    du tout (le ciel). Voir ``coverage.build_mask`` pour la suite.
    """
    from playwright.sync_api import sync_playwright

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            args=["--headless=new", "--enable-gpu", "--ignore-gpu-blocklist", "--enable-webgl"]
        )
        page = browser.new_page(viewport={"width": width, "height": height})
        page.goto(SCENE_HTML.as_uri())
        page.evaluate(
            "config => window.dronePrepare(config)",
            {
                "apiKey": api_key,
                "centre": {"lat": centre[0], "lon": centre[1]},
                "tileDetail": tile_detail,
                "supersample": 1.0,
                "waypoints": [pose.as_dict()],
            },
        )
        page.wait_for_function(
            "window.droneReady === true || window.droneError !== null", timeout=120000
        )
        error = page.evaluate("window.droneError")
        if error:
            browser.close()
            raise CesiumRenderError(f"passe de profondeur échouée : {error}")

        page.evaluate("args => window.droneSeek(args[0], args[1])", [0, TILE_TIMEOUT_MS])
        page.evaluate("() => window.droneDepthMode(true)")
        # Le shader force le retraitement des tuiles : capturer aussitôt ne
        # donnerait que l'arrière-plan, la géométrie ayant momentanément
        # disparu le temps de sa reconstruction.
        page.evaluate("() => window.droneSettle(160)")
        page.screenshot(path=str(out_path), timeout=120000)
        browser.close()

    return out_path


def probe_site(lat: float, lon: float, api_key: str, *, timeout_ms: int = 120000) -> dict:
    """Mesure sol et sommet du bâti sur les tuiles 3D, avant tout rendu.

    Sans cette mesure, l'altitude de traversée n'est qu'une supposition : si
    elle dépasse la hauteur du bâtiment, la caméra survole les toits et le
    générateur, n'ayant aucune façade devant lui, fabrique un intérieur
    flottant au-dessus du bâti.

    Renvoie ``{"ground": float, "top": float, "height": float}`` en mètres.
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            args=["--headless=new", "--enable-gpu", "--ignore-gpu-blocklist", "--enable-webgl"]
        )
        page = browser.new_page(viewport={"width": 640, "height": 360})
        page.goto(SCENE_HTML.as_uri())
        page.evaluate(
            "config => window.dronePrepare(config)",
            {"apiKey": api_key, "centre": {"lat": lat, "lon": lon}, "waypoints": []},
        )
        page.wait_for_function(
            "window.droneReady === true || window.droneError !== null", timeout=timeout_ms
        )
        error = page.evaluate("window.droneError")
        if error:
            browser.close()
            raise CesiumRenderError(f"sondage du site échoué : {error}")
        ground = float(page.evaluate("window.droneGroundHeight"))
        top = float(page.evaluate("window.droneTopHeight"))
        browser.close()

    return {"ground": ground, "top": top, "height": max(0.0, top - ground)}


def build_continuous_traverse(
    *,
    traverse_bearing_deg: float = 90.0,
    traverse_altitude_m: float = 30.0,
    entry_distance_m: float = 55.0,
    exit_distance_m: float = 120.0,
    samples: int = 40,
) -> tuple[list[Maneuver], list[Maneuver]]:
    """Découpe le vol en deux parties continues, séparées par la traversée.

    Renvoie ``(avant, apres)``. « Avant » enchaîne les figures d'orbite puis
    une approche frontale qui finit face à la façade ; « après » repart de
    l'autre côté du bâtiment, dans le même sens, et se dégage en altitude.

    Les deux parties se rendent en 3D réel et ne sont **pas** deux vols
    distincts : la seconde reprend exactement là où la première s'arrête, à
    l'épaisseur du bâtiment près — c'est précisément ce trou que
    l'interpolation IA vient combler. Sans cette construction, chaque
    segment démarrerait à un endroit arbitraire et la suite ne se lirait pas
    comme un seul vol.
    """
    from .maneuvers import _line, artistic_maneuvers, chain_maneuvers

    figures = artistic_maneuvers()
    # Tout sauf la dernière figure précède la traversée ; le survol final
    # conclut la séquence une fois le bâtiment franchi.
    orbits = chain_maneuvers(figures[:-1])
    end = orbits[-1].waypoints[-1]

    angle = math.radians(traverse_bearing_deg)
    de, dn = math.sin(angle), math.cos(angle)
    entry = (-de * entry_distance_m, -dn * entry_distance_m)

    # L'orbite se termine à un azimut arbitraire : sans ce raccord, la ligne
    # d'approche partirait de travers et n'attaquerait pas la façade de face
    # (cap mesuré à 191° au lieu de 90°). On rejoint d'abord l'axe de
    # traversée, en gardant le sujet à l'image, puis on fonce droit dessus.
    align_start = (-de * (entry_distance_m + 85.0), -dn * (entry_distance_m + 85.0))
    alignment = Maneuver(
        id="alignement_axe",
        name_fr="Alignement sur l'axe de traversée",
        color=(191, 90, 242),
        waypoints=_line(
            (end.east_m, end.north_m, end.altitude_m),
            (align_start[0], align_start[1], traverse_altitude_m + 12.0),
            samples=samples,
        ),
        skill_fr="repositionnement en descente douce pour se placer dans l'axe",
        purpose_fr="place le drone dans l'axe du bâtiment avant l'approche",
    )

    approach = Maneuver(
        id="approche_facade",
        name_fr="Approche frontale de la façade",
        color=(255, 214, 10),
        waypoints=_line(
            (align_start[0], align_start[1], traverse_altitude_m + 12.0),
            (entry[0], entry[1], traverse_altitude_m),
            samples=samples,
        ),
        skill_fr="rapprochement rectiligne à hauteur de façade, vitesse constante",
        purpose_fr="amène le regard face au bâtiment, juste avant la traversée",
        heading_mode="trajet",
    )

    exit_leg = Maneuver(
        id="sortie_traversee",
        name_fr="Sortie de la traversée",
        color=(70, 214, 140),
        waypoints=_line(
            (de * entry_distance_m, dn * entry_distance_m, traverse_altitude_m),
            (de * exit_distance_m, dn * exit_distance_m, traverse_altitude_m + 20.0),
            samples=samples,
        ),
        skill_fr="poursuite rectiligne après le franchissement, sans rupture de vitesse",
        purpose_fr="ressort de l'autre côté et poursuit le mouvement",
        heading_mode="trajet",
    )

    after = chain_maneuvers([exit_leg, figures[-1]])
    return [*orbits, alignment, approach], after


def build_traverse_poses(
    lat: float,
    lon: float,
    *,
    bearing_deg: float,
    altitude_m: float = 40.0,
    approach_from_m: float = 90.0,
    stop_before_m: float = 55.0,
    exit_to_m: float = 90.0,
    frames_per_leg: int = 24,
) -> tuple[list[CameraPose], list[CameraPose]]:
    """Poses d'approche puis de sortie de l'autre côté, pour l'effet de traversée.

    Renvoie ``(approche, sortie)`` : deux vols rectilignes opposés, alignés
    sur le même axe (``bearing_deg``, cap de l'approche vers le bâtiment).
    Le passage **à travers** le bâti n'est délibérément pas rendu ici : les
    tuiles 3D ne modélisent que l'extérieur, la caméra n'y verrait que le
    maillage vu de l'envers. Ce trou est destiné à être comblé par une
    interpolation IA entre la dernière image de l'approche et la première de
    la sortie — deux images **réelles** et proches en point de vue, ce qui
    est le cas d'usage où ces modèles sont fiables.
    """
    angle = math.radians(bearing_deg)
    # Vecteur unitaire du sens de vol (l'approche va vers le centre).
    de, dn = math.sin(angle), math.cos(angle)

    def leg(start_distance: float, end_distance: float) -> list[CameraPose]:
        poses: list[CameraPose] = []
        for i in range(frames_per_leg):
            t = i / (frames_per_leg - 1)
            distance = start_distance + (end_distance - start_distance) * t
            # Position en amont du centre le long de l'axe de vol.
            east, north = -de * distance, -dn * distance
            p_lat, p_lon = offset_to_latlon(lat, lon, east, north)
            pitch = -math.degrees(math.atan2(altitude_m - 15.0, max(1.0, abs(distance))))
            poses.append(
                CameraPose(
                    lat=p_lat, lon=p_lon, agl=max(MIN_ALTITUDE_M, altitude_m),
                    heading=bearing_deg % 360.0, pitch=max(-89.0, min(10.0, pitch)),
                )
            )
        return poses

    approach = leg(approach_from_m, stop_before_m)
    # La sortie repart de l'autre côté du bâtiment, même axe, même sens.
    exit_leg = leg(-stop_before_m, -exit_to_m)
    return approach, exit_leg


def render_flight(
    poses: list[CameraPose],
    api_key: str,
    *,
    centre: tuple[float, float],
    out_path: str | Path,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: int = DEFAULT_FPS,
    iso_time: str | None = None,
    tile_detail: float = 4.0,
    supersample: float = 2.0,
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
        # Le mode headless récent de Chromium sait utiliser un vrai GPU : sur
        # cette scène, ~13 s par image en 1080p avec SwiftShader (rendu
        # logiciel) contre une fraction de seconde via Direct3D11. Le repli
        # logiciel reste indispensable sur une machine sans GPU exploitable,
        # sinon Cesium ne rend qu'un fond noir.
        hardware_args = [
            "--headless=new",
            "--enable-gpu",
            "--ignore-gpu-blocklist",
            "--enable-webgl",
        ]
        software_args = [
            "--use-gl=angle",
            "--use-angle=swiftshader",
            "--enable-unsafe-swiftshader",
            "--ignore-gpu-blocklist",
        ]
        browser = None
        for args in (hardware_args, software_args):
            try:
                candidate = playwright.chromium.launch(args=args)
                probe = candidate.new_page()
                renderer = probe.evaluate(
                    "() => { const c = document.createElement('canvas');"
                    " const gl = c.getContext('webgl2') || c.getContext('webgl');"
                    " if (!gl) return ''; const d = gl.getExtension('WEBGL_debug_renderer_info');"
                    " return d ? gl.getParameter(d.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER); }"
                )
                probe.close()
                if renderer:
                    browser = candidate
                    if progress is not None:
                        progress(0, len(poses), f"rendu via {renderer[:70]}")
                    break
                candidate.close()
            except Exception:  # noqa: BLE001 — on essaie simplement le repli
                continue
        if browser is None:
            raise CesiumRenderError("aucun contexte WebGL disponible (ni matériel, ni logiciel)")
        page = browser.new_page(viewport={"width": width, "height": height})
        page.goto(SCENE_HTML.as_uri())

        page.evaluate(
            "config => window.dronePrepare(config)",
            {
                "apiKey": api_key,
                "centre": {"lat": centre[0], "lon": centre[1]},
                "isoTime": iso_time,
                "tileDetail": tile_detail,
                "supersample": supersample,
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
