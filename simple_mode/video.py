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
