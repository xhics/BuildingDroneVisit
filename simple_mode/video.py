"""Assemble le story-board continu d'un survol drone (une seule prise, sans coupure).

Usage :
    python -m simple_mode.video "123 rue Principale, Boucherville, QC"

La trajectoire est chaînée (chaque figure démarre où la précédente finit) et
mise à l'échelle de l'étendue réelle du lieu (détectée via l'API Places, ou
fournie avec ``--extent-m``) — un grand complexe reçoit des figures plus
amples qu'un bâtiment isolé. Plusieurs images de référence réelles (Street
View, satellite) sont échantillonnées tout le long du trajet, pas une seule
par figure.

Ne soumet **pas** de vidéo à Sogni par défaut — voir ``sogni_client.py`` pour
pourquoi le schéma exact de la requête n'est pas encore confirmé. Avec
``--probe-sogni``, interroge l'API Sogni pour observer la vraie forme de sa
réponse et finaliser ce schéma.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .cli import _find_dotenv
from .geocode import GeocodeError, geocode_address
from .maneuvers import chain_maneuvers, default_maneuvers, scale_maneuvers
from .places import fetch_viewport_extent_m
from .satellite import fetch_satellite_image
from .storyboard import MAX_KEYFRAMES, build_continuous_storyboard

#: Diagonale typique du viewport Places d'un bâtiment isolé (le viewport
#: Places est généreux — il inclut abords et stationnement, pas seulement
#: l'empreinte du bâtiment). Calibré sur des adresses réelles : ~360-390 m
#: pour un immeuble ou un hôtel de centre-ville isolé. Un grand complexe
#: (campus, centre commercial, parc à thème) dépasse largement ce seuil.
BASELINE_DIAGONAL_M = 350.0

#: Le rayon ne grandit pas indéfiniment avec la taille détectée — au-delà,
#: mieux vaut un `--extent-m` choisi à la main qu'une extrapolation aveugle.
MAX_RADIUS_SCALE = 4.0

#: L'altitude grandit plus lentement que le rayon : survoler un grand
#: complexe ne justifie pas de monter aussi haut que son diamètre le suggère.
MAX_ALTITUDE_SCALE = 1.6


def _detect_scale(address: str, places_key: str | None) -> tuple[float, float, float | None]:
    """Renvoie ``(radius_scale, altitude_scale, diagonal_m)``.

    ``diagonal_m`` est ``None`` si l'étendue n'a pas pu être détectée — le
    gabarit par défaut (bâtiment isolé) s'applique alors sans échelle.
    """
    if not places_key:
        return 1.0, 1.0, None
    extent = fetch_viewport_extent_m(address, places_key)
    if extent is None:
        return 1.0, 1.0, None
    diagonal = math.hypot(*extent)
    radius_scale = max(1.0, min(MAX_RADIUS_SCALE, diagonal / BASELINE_DIAGONAL_M))
    altitude_scale = min(radius_scale, MAX_ALTITUDE_SCALE)
    return radius_scale, altitude_scale, diagonal


def _restyle_flight(
    args, geocoded: dict, rendered: Path, out_dir: Path, label: str
) -> list[Path]:  # noqa: ANN001
    """Retexture le vol rendu en préservant sa géométrie, tronçon par tronçon.

    La régénération libre (Ref2VA) échouait ici : ne contraignant aucune
    image, le modèle reconstruisait le bâtiment à sa façon en cours de plan.
    Le conditionnement ControlNet contraint **chaque** image, si bien que le
    générateur ne peut plus agir que sur la matière — lumière, reflets, flou
    de mouvement — que le moteur 3D ne sait pas produire.

    Le vol est découpé avant traitement : LTX v2v ne traite que des plans
    courts.
    """
    from .production_plan import TIMES_OF_DAY
    from .references import build_aerial_references
    from .sogni_cli import MAX_V2V_SECONDS, restyle_video, split_video

    chunks = split_video(rendered, out_dir / f"tronçons_{label}", chunk_seconds=MAX_V2V_SECONDS)
    look = TIMES_OF_DAY.get(args.time_of_day)
    establishment = geocoded["formatted_address"].split(",")[0]

    # Nommer l'établissement et décrire ses matériaux pèse autant que le
    # conditionnement : à prompt générique, le modèle produit *un* bâtiment du
    # même genre plutôt que *celui-là*. La description vient du classement
    # visuel des photos réelles quand il est disponible.
    materials, appearance = _exterior_reference(args, geocoded, out_dir)
    street_views = sorted(out_dir.glob("streetview_*.jpg"))

    print(
        f"  {label} : {len(chunks)} tronçon(s) retexturés "
        f"({args.control_mode}, force {args.control_strength}) — facturé…"
    )
    produced: list[Path] = []
    for index, chunk in enumerate(chunks):
        references = build_aerial_references(
            rendered_chunk=chunk,
            exterior_photos=[appearance] if appearance else [],
            street_view_photos=street_views,
        )
        # Le rôle des images est énoncé dans le prompt : sans cela le moteur
        # peut traiter une référence comme un plan à insérer, et en recopier
        # le cadrage ou les inscriptions.
        prompt = (
            f"{establishment}, vue aérienne réelle filmée au drone. "
            "Conserver exactement l'architecture, la silhouette et les matériaux du "
            f"bâtiment de la vidéo source{materials}. Ne rien ajouter ni retirer à "
            "sa structure. "
            f"{references.prompt_clause_fr()} "
            f"{look.look_fr if look else 'lumière naturelle directionnelle'}. "
            "Rendu photographique de caméra : flou de mouvement naturel, profondeur "
            "de champ, reflets et matières réalistes, grain fin."
        )
        if index == 0:
            print(f"    {references.describe_fr()}")

        hidden = references.reference_only()
        produced.append(
            restyle_video(
                chunk, prompt,
                out_path=out_dir / f"ia_{label}_{index:02d}.mp4",
                control=args.control_mode, control_strength=args.control_strength,
                appearance_image=hidden[0].path if hidden else None,
            )
        )
        print(f"    tronçon {index + 1}/{len(chunks)}", flush=True)
    return produced


def _possessive(name: str) -> str:
    from .interior_journey import possessive_fr

    return possessive_fr(name)


def _exterior_reference(args, geocoded: dict, out_dir: Path) -> tuple[str, Path | None]:  # noqa: ANN001
    """Description de la façade et photo utilisable comme référence d'apparence.

    Le conditionnement structurel impose la forme, jamais la matière : sans
    référence, le modèle choisit lui-même et produit un bâtiment plausible
    mais différent.

    La photo doit être **propre**. Une vue chargée de texte ou d'un premier
    plan massif contamine la génération : essayée avec une photo à massif
    floral « UNESCO », le modèle en a recopié le lettrage dans le plan
    aérien. On ne retient donc qu'une photo signalée exploitable par le
    classement visuel — sinon la description seule, sans image.
    """
    sources = out_dir / "sources"
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not sources.exists() or not openai_key:
        return "", None

    from .hotel_sources import HotelSourceError, SourcePhoto, classify_photos

    photos = [SourcePhoto(p) for p in sorted(sources.glob("*.jpg"))[:6]]
    if not photos:
        return "", None
    try:
        classify_photos(photos, openai_key)
    except HotelSourceError:
        return "", None

    exteriors = [p for p in photos if p.category == "exterieur"]
    if not exteriors:
        return "", None

    description = ""
    for photo in exteriors:
        if photo.description_fr:
            description = f" — {photo.description_fr.rstrip('.').lower()}"
            break

    clean = next((p for p in exteriors if p.clean_reference), None)
    if clean is None:
        print(
            "  (aucune photo de façade exploitable comme référence : "
            "description seule, sans image)",
            file=sys.stderr,
        )
    return description, clean.path if clean else None


def _interior_stops(args, geocoded: dict, out_dir: Path) -> list:  # noqa: ANN001
    """Photos réelles de l'établissement, à utiliser comme étapes intérieures.

    Sans elles le passage n'a aucune matière et le générateur invente une
    enfilade générique. Un échec de sourcing n'est pas bloquant : on retombe
    sur un passage direct, plus court, plutôt que d'interrompre le rendu.
    """
    if args.interior_photos <= 0:
        return []
    places_key = os.environ.get("GOOGLE_PLACES_API_KEY") or os.environ.get("GOOGLE_MAPS_API_KEY")
    if not places_key:
        return []

    from .hotel_sources import HotelSourceError, fetch_photos, find_place

    try:
        place = find_place(args.address, places_key)
        photos = fetch_photos(
            place, places_key, out_dir / "sources", limit=args.interior_photos
        )
    except (HotelSourceError, Exception):  # noqa: BLE001 — repli assumé
        print("  (aucune photo d'établissement récupérée : passage direct)", file=sys.stderr)
        return []

    from .interior_journey import build_stops

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        from .hotel_sources import HotelSite, classify_photos

        try:
            classify_photos(photos, openai_key)
            site = HotelSite(args.address, "", "", 0, 0, photos)
            # Les vues de façade et les panoramas sont déjà couverts, et bien
            # mieux, par le survol 3D : les rejouer comme étape intérieure
            # ferait ressortir la caméra du bâtiment qu'elle vient de pénétrer.
            ordered = [p for p in site.journey() if p.category not in ("exterieur", "vue")]
            return build_stops(ordered)
        except HotelSourceError as exc:
            # Sans classement l'ordre reste celui de Places : moins narratif,
            # mais les références restent de vraies photos du lieu.
            print(f"  (classement indisponible, ordre brut : {exc})", file=sys.stderr)

    return build_stops(photos)


def _passage_prompts(count: int, anchored: bool) -> list[str]:
    """Un prompt par saut du passage intérieur."""
    if not anchored:
        return [
            "Vue subjective depuis un drone qui avance. La caméra franchit la façade "
            "droit devant elle et pénètre à l'intérieur du bâtiment, au même niveau, "
            "sans monter. Mouvement continu vers l'avant, caméra horizontale."
        ] * count

    prompts = []
    for index in range(count):
        if index == 0:
            prompts.append(
                "Vue subjective depuis un drone qui avance. La caméra franchit la "
                "façade droit devant elle et pénètre à l'intérieur du bâtiment, au "
                "même niveau, pour déboucher sur l'espace montré à l'image finale. "
                "Mouvement continu vers l'avant, caméra horizontale, sans coupure."
            )
        elif index == count - 1:
            prompts.append(
                "La caméra quitte l'espace intérieur en avançant et ressort du "
                "bâtiment par la façade, débouchant sur la vue extérieure finale. "
                "Mouvement continu vers l'avant, vitesse constante, sans recul."
            )
        else:
            prompts.append(
                "La caméra poursuit son avancée à l'intérieur de l'établissement et "
                "passe de l'espace montré au départ à celui montré à l'arrivée, en "
                "traversant le passage qui les relie. Mouvement fluide vers l'avant, "
                "hauteur constante, sans coupure."
            )
    return prompts


def _render_traverse(
    args, geocoded: dict, maps_key: str, out_dir: Path, width: int, height: int,
    *, base_flight: Path, progress,  # noqa: ANN001
) -> int:
    """Approche 3D + passage intérieur généré + sortie 3D, recollés.

    Les deux bornes du passage sont des images **réellement rendues** : le
    modèle n'invente que le court intérieur, qui n'existe dans aucune donnée
    (les tuiles 3D ne modélisent que l'extérieur).
    """
    from .cesium_render import (
        CesiumRenderError,
        build_continuous_traverse,
        build_poses,
        probe_site,
        render_flight,
    )
    from .sogni_cli import (
        SogniCliError,
        concat_videos,
        generate_from_references,
        generate_transition_chain,
    )

    sogni_key = os.environ.get("SOGNI_API_KEY")
    if not sogni_key:
        print("Erreur : --traverse nécessite SOGNI_API_KEY dans .env.", file=sys.stderr)
        return 1

    traverse_altitude = args.traverse_altitude
    entry_distance = 55.0
    if traverse_altitude <= 0:
        # Mesure du bâti avant de construire la trajectoire : viser au-dessus
        # du toit ferait survoler le bâtiment au lieu de lui faire face, et le
        # générateur, privé de façade, invente un intérieur en l'air.
        try:
            site = probe_site(geocoded["lat"], geocoded["lon"], maps_key)
        except CesiumRenderError as exc:
            print(f"Sondage du site échoué : {exc}", file=sys.stderr)
            return 1
        traverse_altitude = max(4.0, site["height"] * 0.55)
        # Un petit bâtiment doit être abordé de plus près, sinon sa façade
        # n'occupe qu'une fraction de l'image.
        entry_distance = max(20.0, site["height"] * 3.0)
        print(
            f"  bâti mesuré : {site['height']:.1f} m de haut -> traversée à "
            f"{traverse_altitude:.1f} m, entrée à {entry_distance:.0f} m"
        )

    # Un seul vol, coupé en deux par l'épaisseur du bâtiment — et non deux
    # vidéos indépendantes qui démarreraient chacune à un endroit arbitraire.
    before, after = build_continuous_traverse(
        traverse_bearing_deg=args.traverse_bearing,
        traverse_altitude_m=traverse_altitude,
        entry_distance_m=entry_distance,
    )
    # Les images sont réparties d'après la longueur de chaque segment, pas à
    # parts fixes : sinon la caméra change d'allure au raccord (×1,5 mesuré).
    from .cesium_render import allocate_frames, path_length_m

    lengths = [path_length_m(before), path_length_m(after)]
    frames_before, frames_after = allocate_frames(
        lengths, fps=args.render_3d_fps, cruise_speed_mps=args.cruise_speed
    )
    print(
        f"\nVol continu avec traversée (cap {args.traverse_bearing:.0f}°) — "
        f"vitesse {args.cruise_speed:.0f} m/s : {lengths[0]:.0f} m puis {lengths[1]:.0f} m, "
        f"soit {(sum(lengths) / args.cruise_speed):.0f} s de vol"
    )
    legs = {}
    try:
        for name, maneuvers_part, count in (
            ("avant", before, frames_before),
            ("apres", after, frames_after),
        ):
            poses = build_poses(
                maneuvers_part, geocoded["lat"], geocoded["lon"],
                frame_count=count, enforce_envelope=False,
            )
            legs[name] = render_flight(
                poses, maps_key, centre=(geocoded["lat"], geocoded["lon"]),
                out_path=out_dir / f"vol_{name}.mp4", width=width, height=height,
                fps=args.render_3d_fps, keep_frames=True, progress=progress,
                tile_detail=args.tile_detail, supersample=args.supersample,
            )
    except CesiumRenderError as exc:
        print(f"Rendu du vol échoué : {exc}", file=sys.stderr)
        return 1

    # Bornes du passage : dernière image avant le bâtiment, première après.
    last_approach = sorted((out_dir / "vol_avant_frames").glob("frame_*.png"))[-1]
    first_exit = sorted((out_dir / "vol_apres_frames").glob("frame_*.png"))[0]

    # Étapes intérieures : de vraies photos de l'établissement. Elles servent
    # d'ancrages intermédiaires, ce qui découpe le passage en sauts courts —
    # un long passage d'un seul tenant dérive, et atteignait ici le plafond de
    # 15 s du modèle, soit 15 % de vidéo entièrement inventée.
    # Mode « références libres » (Ref2VA) : les photos inspirent le plan au
    # lieu d'en verrouiller les extrémités. Le mode flf2v précédent épinglait
    # première et dernière image, d'où un rendu trop littéral où l'on
    # reconnaissait les photos plutôt qu'un mouvement de caméra.
    stops = _interior_stops(args, geocoded, out_dir)
    if not stops:
        print("  (aucune photo d'intérieur : passage direct entre deux images rendues)")
        anchors = [last_approach, first_exit]
        try:
            clips = generate_transition_chain(
                anchors, _passage_prompts(1, False), out_dir=out_dir / "passages",
                duration_s=args.passage_clip_seconds,
            )
        except SogniCliError as exc:
            print(f"Passage intérieur échoué : {exc}", file=sys.stderr)
            return 1
    else:
        from .interior_journey import build_ref2va_prompt
        from .production_plan import TIMES_OF_DAY

        look = TIMES_OF_DAY.get(args.time_of_day)
        prompt = build_ref2va_prompt(
            stops,
            establishment_fr=geocoded["formatted_address"].split(",")[0],
            time_of_day_fr=look.look_fr if look else "",
        )
        print(
            f"Passage intérieur : {len(stops)} espace(s) réel(s) en références libres "
            f"({' -> '.join(s.label_fr for s in stops)}) — facturé…"
        )
        try:
            clips = [
                generate_from_references(
                    [s.photo for s in stops], prompt,
                    out_path=out_dir / "vol_interieur.mp4",
                    duration_s=args.interior_seconds,
                )
            ]
        except SogniCliError as exc:
            print(f"Passage intérieur échoué : {exc}", file=sys.stderr)
            print(f"Les deux segments 3D restent disponibles : {legs}", file=sys.stderr)
            return 1

    # Le rendu 3D peut servir de livrable, ou seulement de référence : dans ce
    # dernier cas chaque plan aérien est régénéré, ce qui apporte le flou de
    # mouvement et la matière photographique absents du moteur 3D.
    aerial_before = [legs["avant"]]
    aerial_after = [legs["apres"]]
    if args.ai_final:
        # Un plan en échec ne doit pas emporter la série : les plans déjà
        # produits sont facturés, et le rendu 3D reste un repli exploitable.
        # On dégrade cette partie-là, sans interrompre le reste.
        for key in ("avant", "apres"):
            try:
                produced = _restyle_flight(
                    args, geocoded, legs[key], out_dir, key
                )
            except SogniCliError as exc:
                print(f"  plan(s) « {key} » en échec : {exc}", file=sys.stderr)
                print("  -> repli sur le rendu 3D pour cette partie", file=sys.stderr)
                continue
            if produced and key == "avant":
                aerial_before = produced
            elif produced:
                aerial_after = produced

    try:
        final = concat_videos(
            [*aerial_before, *clips, *aerial_after], out_dir / "vol_complet.mp4"
        )
    except SogniCliError as exc:
        print(f"Recollage échoué : {exc}", file=sys.stderr)
        return 1

    print(f"Vol complet (une seule séquence continue) : {final}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Story-board continu d'un survol drone à partir d'une adresse."
    )
    parser.add_argument("address", help="Adresse à géocoder")
    parser.add_argument("--out", default="video_out", help="Dossier de sortie (défaut: video_out/)")
    parser.add_argument("--zoom", type=int, default=19, help="Zoom de l'image satellite (défaut: 19)")
    parser.add_argument(
        "--extent-m",
        type=float,
        default=None,
        help="Diagonale approximative du site en mètres — sinon détectée via l'API Places",
    )
    parser.add_argument(
        "--max-keyframes",
        type=int,
        default=MAX_KEYFRAMES,
        help=f"Nombre d'images de référence le long du trajet (défaut: {MAX_KEYFRAMES})",
    )
    parser.add_argument(
        "--ai-plan-fixture",
        default=None,
        help="Charge un plan JSON local (même schéma que simple_mode.ai_plan) au lieu du gabarit par défaut",
    )
    parser.add_argument(
        "--render-3d",
        action="store_true",
        help=(
            "Rend un vrai vol de drone sur la géométrie 3D de Google (Cesium) : la trajectoire "
            "calculée est réellement appliquée, rendu déterministe, aucune invention. "
            "Nécessite playwright + chromium et ffmpeg."
        ),
    )
    parser.add_argument(
        "--render-3d-duration", type=float, default=15.0, help="Durée du vol 3D en secondes (défaut: 15)"
    )
    parser.add_argument("--render-3d-fps", type=int, default=12, help="Images par seconde du vol 3D (défaut: 12)")
    parser.add_argument(
        "--tile-detail",
        type=float,
        default=4.0,
        help=(
            "Erreur d'écran tolérée sur les tuiles 3D (défaut: 4). Cesium utilise 16 "
            "pour la navigation interactive ; abaisser charge des tuiles plus fines "
            "— netteté mesurée +31%%, rendu ~3x plus long."
        ),
    )
    parser.add_argument(
        "--supersample",
        type=float,
        default=2.0,
        help="Facteur de suréchantillonnage avant réduction (défaut: 2.0, lisse les arêtes)",
    )
    parser.add_argument(
        "--cruise-speed",
        type=float,
        default=25.0,
        help=(
            "Vitesse de la caméra en m/s (défaut: 25, drone rapide). C'est elle qui fixe "
            "la durée : la vidéo dure aussi longtemps que le trajet l'exige, ce qui "
            "garantit une vitesse identique d'un segment à l'autre. 8-15 pour un survol "
            "posé mais plus long à rendre ; au-delà de 25 le mouvement perd en crédibilité."
        ),
    )
    parser.add_argument(
        "--render-3d-size",
        default="1280x720",
        help="Résolution du vol 3D, format LARGEURxHAUTEUR (défaut: 1280x720)",
    )
    parser.add_argument(
        "--traverse",
        action="store_true",
        help=(
            "Ajoute un plan de traversée du bâtiment : approche et sortie rendues en 3D réel, "
            "passage intérieur comblé par Sogni entre les deux images réelles (facturé). "
            "S'utilise avec --render-3d."
        ),
    )
    parser.add_argument(
        "--traverse-bearing",
        type=float,
        default=90.0,
        help="Cap de l'axe de traversée en degrés (défaut: 90, soit d'ouest en est)",
    )
    parser.add_argument(
        "--traverse-altitude",
        type=float,
        default=0.0,
        help=(
            "Altitude de la traversée en mètres au-dessus du sol. Par défaut (0), "
            "elle est déduite de la hauteur réelle du bâtiment mesurée sur les tuiles 3D. "
            "Viser trop haut fait survoler le toit : le générateur, sans façade devant "
            "lui, fabrique alors un intérieur flottant au-dessus du bâtiment."
        ),
    )
    parser.add_argument(
        "--interior-photos",
        type=int,
        default=4,
        help=(
            "Nombre de photos réelles de l'établissement servant d'étapes intérieures "
            "(défaut: 4 ; 0 pour un passage direct). Chaque étape ajoute un saut facturé, "
            "mais raccourcit d'autant ce que le générateur doit inventer sans référence."
        ),
    )
    parser.add_argument(
        "--ai-final",
        action="store_true",
        help=(
            "Le rendu 3D ne sert plus que de référence : chaque plan est régénéré par "
            "l'IA à partir des images rendues (mode Ref2VA). Apporte flou de mouvement, "
            "24 i/s et matière photographique, que la 3D ne peut pas produire. Facturé "
            "par plan."
        ),
    )
    parser.add_argument(
        "--control-mode",
        default="canny",
        choices=["depth", "canny", "detailer"],
        help=(
            "Conditionnement structurel de la retexturation (défaut: depth). "
            "'depth' impose les volumes, 'canny' les contours, 'detailer' préserve "
            "tout sans changer le style. C'est ce qui empêche le modèle de "
            "reconstruire le bâtiment à sa façon."
        ),
    )
    parser.add_argument(
        "--control-strength",
        type=float,
        default=0.95,
        help=(
            "Force du conditionnement (défaut: 0.85). Plus haut, le modèle colle à "
            "la géométrie source ; plus bas, il réinterprète — sous 0.6 il commence "
            "à s'écarter du bâtiment réel."
        ),
    )
    parser.add_argument(
        "--ai-segment-seconds",
        type=int,
        default=15,
        help="Durée des tronçons envoyés à la retexturation (défaut: 15 s, maximum LTX v2v)",
    )
    parser.add_argument(
        "--interior-seconds",
        type=int,
        default=10,
        help=(
            "Durée du parcours intérieur en secondes (défaut: 10 ; le modèle accepte "
            "6 à 15). Généré en une passe à partir des photos en références libres."
        ),
    )
    parser.add_argument(
        "--time-of-day",
        default="doree",
        choices=["aube", "matin", "midi", "doree", "bleue", "nuit"],
        help=(
            "Moment de la journée (défaut: doree). Oriente le vocabulaire des prompts ; "
            "sans effet sur les tuiles 3D, dont les textures portent leur éclairage d'origine."
        ),
    )
    parser.add_argument(
        "--passage-clip-seconds",
        type=int,
        default=5,
        help=(
            "Durée de chaque saut du passage intérieur (défaut: 5 s). Le modèle dérive "
            "au-delà de quelques secondes : mieux vaut plusieurs sauts courts qu'un long."
        ),
    )
    parser.add_argument(
        "--probe-sogni",
        action="store_true",
        help="Sonde l'API Sogni (SOGNI_API_KEY requis) pour confirmer le schéma réel avant un envoi",
    )
    parser.add_argument(
        "--generate-sogni",
        action="store_true",
        help=(
            "Rend une chaîne de clips MiniMax H3 (un par paire d'images adjacentes) et les "
            "recolle en une VRAIE vidéo, facturée. Nécessite SOGNI_API_KEY, `sogni-agent` et "
            "`ffmpeg` installés."
        ),
    )
    parser.add_argument(
        "--sogni-model",
        default=None,
        help="Modèle sogni-agent pour chaque clip de transition (défaut: minimax-h3-flf2v-turbo)",
    )
    parser.add_argument(
        "--sogni-clip-duration",
        type=int,
        default=None,
        help="Durée en secondes de chaque clip de transition (défaut: 5)",
    )
    parser.add_argument(
        "--sogni-chain-keyframes",
        type=int,
        default=None,
        help="Nombre d'images-clés utilisées pour la chaîne (N images -> N-1 clips payants ; défaut: 6)",
    )
    args = parser.parse_args(argv)

    dotenv_path = _find_dotenv()
    if dotenv_path:
        load_dotenv(dotenv_path)

    maps_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not maps_key:
        print("Erreur : GOOGLE_MAPS_API_KEY manquant dans .env.", file=sys.stderr)
        return 1

    try:
        geocoded = geocode_address(args.address, maps_key)
    except GeocodeError as exc:
        print(f"Erreur de géocodage : {exc}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        image, _mpp = fetch_satellite_image(geocoded["lat"], geocoded["lon"], maps_key, zoom=args.zoom)
    except Exception as exc:  # noqa: BLE001 — remonté tel quel à l'utilisateur
        print(f"Erreur lors de la récupération de l'image satellite : {exc}", file=sys.stderr)
        return 1
    satellite_path = out_dir / "satellite.png"
    image.save(satellite_path)

    if args.ai_plan_fixture:
        from .ai_plan import AIPlanError, load_plan_fixture

        try:
            maneuvers = load_plan_fixture(args.ai_plan_fixture)
        except AIPlanError as exc:
            print(f"Erreur du plan (fixture) : {exc}", file=sys.stderr)
            return 1
    else:
        maneuvers = default_maneuvers()

    if args.extent_m is not None:
        diagonal = args.extent_m
        radius_scale = max(1.0, min(MAX_RADIUS_SCALE, diagonal / BASELINE_DIAGONAL_M))
        altitude_scale = min(radius_scale, MAX_ALTITUDE_SCALE)
    else:
        places_key = os.environ.get("GOOGLE_PLACES_API_KEY") or maps_key
        radius_scale, altitude_scale, diagonal = _detect_scale(geocoded["formatted_address"], places_key)

    if diagonal is not None:
        print(f"Étendue du site : ~{diagonal:.0f} m de diagonale -> échelle rayon x{radius_scale:.1f}")
    else:
        print("Étendue du site non détectée : gabarit par défaut (bâtiment isolé), sans mise à l'échelle")

    maneuvers = scale_maneuvers(maneuvers, radius_scale=radius_scale, altitude_scale=altitude_scale)
    maneuvers = chain_maneuvers(maneuvers)

    storyboard = build_continuous_storyboard(
        address=geocoded["formatted_address"],
        lat=geocoded["lat"],
        lon=geocoded["lon"],
        satellite_path=satellite_path,
        maneuvers=maneuvers,
        api_key_maps=maps_key,
        out_dir=out_dir,
        max_keyframes=args.max_keyframes,
    )
    storyboard_path = out_dir / "storyboard.json"
    storyboard.save(storyboard_path)

    street_view_count = sum(1 for k in storyboard.keyframes if k.reference_kind == "street_view")
    print(f"Adresse résolue : {geocoded['formatted_address']}")
    print(
        f"Story-board continu : {storyboard_path} — {len(storyboard.keyframes)} image(s) de "
        f"référence ({street_view_count} Street View, {len(storyboard.keyframes) - street_view_count} "
        f"satellite/générative), durée totale estimée {storyboard.total_duration_s:.0f}s"
    )

    if args.render_3d:
        from .cesium_render import CesiumRenderError, build_poses, render_flight
        from .maneuvers import artistic_maneuvers

        # Le gabarit par défaut vise des rayons serrés qui sortent de
        # l'enveloppe nette des tuiles photogrammétriques : plus de la moitié
        # de ses poses devraient être repoussées, ce qui déforme la figure.
        # La séquence artistique y tient d'emblée et alterne des figures de
        # natures distinctes plutôt que des orbites successives. Un plan conçu
        # par IA (--ai-plan-fixture) reste prioritaire : il est demandé explicitement.
        flight_maneuvers = maneuvers if args.ai_plan_fixture else chain_maneuvers(
            scale_maneuvers(artistic_maneuvers(), radius_scale=radius_scale, altitude_scale=1.0)
        )

        try:
            width, height = (int(v) for v in args.render_3d_size.lower().split("x"))
        except ValueError:
            print(f"Erreur : --render-3d-size invalide ({args.render_3d_size!r}), attendu LARGEURxHAUTEUR.", file=sys.stderr)
            return 1

        def _progress(index: int, total: int, note: str) -> None:
            if note:
                print(f"  {note}")
            elif index % 10 == 0 or index == total:
                print(f"  {index}/{total} images", flush=True)

        if args.traverse:
            # La traversée n'est pas un plan à part : c'est le même vol, coupé
            # par l'épaisseur du bâtiment. Le rendre en plus d'un vol complet
            # produirait deux séquences sans lien — le défaut relevé sur la
            # première version.
            code = _render_traverse(
                args, geocoded, maps_key, out_dir, width, height,
                base_flight=None, progress=_progress,
            )
            if code != 0:
                return code
        else:
            from .cesium_render import allocate_frames, path_length_m

            # Même règle que pour la traversée : la durée découle du trajet et
            # de la vitesse, pour que l'allure soit comparable d'un vol à l'autre.
            length = path_length_m(flight_maneuvers)
            (frame_count,) = allocate_frames(
                [length], fps=args.render_3d_fps, cruise_speed_mps=args.cruise_speed
            )
            poses = build_poses(flight_maneuvers, geocoded["lat"], geocoded["lon"], frame_count=frame_count)
            print(
                f"\nRendu 3D : {length:.0f} m à {args.cruise_speed:.0f} m/s -> "
                f"{frame_count} images ({frame_count / args.render_3d_fps:.0f}s à "
                f"{args.render_3d_fps} i/s, {width}x{height}) — géométrie réelle, sans facturation Sogni."
            )
            try:
                result = render_flight(
                    poses, maps_key, centre=(geocoded["lat"], geocoded["lon"]),
                    out_path=out_dir / "flight_3d.mp4", width=width, height=height,
                    fps=args.render_3d_fps, progress=_progress,
                    tile_detail=args.tile_detail, supersample=args.supersample,
                )
            except CesiumRenderError as exc:
                print(f"Rendu 3D échoué : {exc}", file=sys.stderr)
                return 1
            print(f"Vol 3D généré : {result}")

    if args.probe_sogni:
        sogni_key = os.environ.get("SOGNI_API_KEY")
        if not sogni_key:
            print("Erreur : --probe-sogni nécessite SOGNI_API_KEY dans .env.", file=sys.stderr)
            return 1
        from .sogni_client import SogniError, probe

        try:
            result = probe(sogni_key)
        except SogniError as exc:
            print(f"Sondage Sogni échoué : {exc}", file=sys.stderr)
            return 1
        print("\nRéponse Sogni brute (pour finaliser le schéma de sogni_client.py) :")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.generate_sogni:
        sogni_key = os.environ.get("SOGNI_API_KEY")
        if not sogni_key:
            print("Erreur : --generate-sogni nécessite SOGNI_API_KEY dans .env.", file=sys.stderr)
            return 1
        from .sogni_cli import (
            DEFAULT_CHAIN_KEYFRAMES,
            DEFAULT_CLIP_DURATION_S,
            DEFAULT_VIDEO_MODEL,
            SogniCliError,
            _dedupe_adjacent_images,
            _resample,
            generate_video_chain,
        )

        model = args.sogni_model or DEFAULT_VIDEO_MODEL
        clip_duration = args.sogni_clip_duration or DEFAULT_CLIP_DURATION_S
        chain_keyframes = args.sogni_chain_keyframes or DEFAULT_CHAIN_KEYFRAMES
        # Compté après le même ré-échantillonnage et la même déduplication que
        # `generate_video_chain` : chaque clip est facturé, l'estimation
        # affichée doit donc être le nombre réel, pas une borne haute.
        clip_count = len(_dedupe_adjacent_images(_resample(storyboard.keyframes, chain_keyframes))) - 1

        print(
            f"\nEnvoi à sogni-agent : {clip_count} clip(s) de transition ({model}, "
            f"{clip_duration}s chacun) puis recollage — VRAIE génération, facturée "
            f"sur le compte Sogni."
        )
        video_path = out_dir / "video.mp4"
        try:
            result_path = generate_video_chain(
                storyboard,
                out_path=video_path,
                video_model=model,
                clip_duration_s=clip_duration,
                chain_keyframes=chain_keyframes,
            )
        except SogniCliError as exc:
            print(f"Génération Sogni échouée : {exc}", file=sys.stderr)
            return 1
        print(f"Vidéo générée : {result_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
