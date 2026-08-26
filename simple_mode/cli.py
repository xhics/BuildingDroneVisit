"""Point d'entrée en ligne de commande du mode simple.

Usage :
    python -m simple_mode "123 rue Principale, Boucherville, QC"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .geocode import GeocodeError, geocode_address
from .maneuvers import chain_maneuvers, default_maneuvers
from .narrative import build_report
from .render import render
from .satellite import fetch_satellite_image


def _find_dotenv() -> Path | None:
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents]:
        candidate = parent / ".env"
        if candidate.exists():
            return candidate
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aperçu du trajet caméra drone à partir d'une adresse."
    )
    parser.add_argument("address", help="Adresse à géocoder, ex: '123 rue Principale, Boucherville, QC'")
    parser.add_argument("--zoom", type=int, default=19, help="Zoom de l'image satellite (défaut: 19)")
    parser.add_argument("--size", type=int, default=640, help="Taille de l'image en pixels (défaut: 640)")
    parser.add_argument(
        "--scale", type=int, default=2, choices=(1, 2), help="Échelle Google Static Maps (défaut: 2)"
    )
    parser.add_argument(
        "--out", default="out", help="Préfixe des fichiers de sortie (défaut: 'out' -> out.png, out.md)"
    )
    parser.add_argument("--api-key", default=None, help="Clé Google Maps ; sinon lue depuis GOOGLE_MAPS_API_KEY")
    parser.add_argument(
        "--ai-plan",
        action="store_true",
        help="Conçoit le plan de vol via OpenAI plutôt que le gabarit par défaut (nécessite OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--ai-plan-fixture",
        default=None,
        help="Charge un plan JSON local (même schéma que --ai-plan) au lieu d'appeler OpenAI — "
        "pratique pour tester sans clé API. Prioritaire sur --ai-plan.",
    )
    parser.add_argument(
        "--ai-illustration",
        action="store_true",
        help="Génère en plus une illustration artistique du trajet via OpenAI (nécessite OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--openai-model", default=None, help="Modèle OpenAI pour le plan (défaut: OPENAI_MODEL ou gpt-4o-mini)"
    )
    parser.add_argument(
        "--openai-image-model",
        default=None,
        help="Modèle OpenAI pour l'illustration (défaut: OPENAI_IMAGE_MODEL ou gpt-image-1)",
    )
    args = parser.parse_args(argv)

    dotenv_path = _find_dotenv()
    if dotenv_path:
        load_dotenv(dotenv_path)

    api_key = args.api_key or os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        print(
            "Erreur : aucune clé API. Définis GOOGLE_MAPS_API_KEY dans .env ou passe --api-key.",
            file=sys.stderr,
        )
        return 1

    try:
        geocoded = geocode_address(args.address, api_key)
    except GeocodeError as exc:
        print(f"Erreur de géocodage : {exc}", file=sys.stderr)
        return 1

    try:
        image, mpp = fetch_satellite_image(
            geocoded["lat"], geocoded["lon"], api_key,
            zoom=args.zoom, size=args.size, scale=args.scale,
        )
    except Exception as exc:  # noqa: BLE001 — remonté tel quel à l'utilisateur
        print(f"Erreur lors de la récupération de l'image satellite : {exc}", file=sys.stderr)
        return 1

    maneuvers = default_maneuvers()
    if args.ai_plan_fixture:
        from .ai_plan import AIPlanError, load_plan_fixture

        try:
            maneuvers = load_plan_fixture(args.ai_plan_fixture)
        except AIPlanError as exc:
            print(f"Erreur du plan (fixture) : {exc}", file=sys.stderr)
            return 1
        print(f"Plan de vol chargé depuis fixture : {len(maneuvers)} figure(s) ({args.ai_plan_fixture})")
    elif args.ai_plan:
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            print("Erreur : --ai-plan nécessite OPENAI_API_KEY dans .env.", file=sys.stderr)
            return 1
        from .ai_plan import AIPlanError, generate_plan

        model = args.openai_model or os.environ.get("OPENAI_MODEL") or None
        try:
            maneuvers = generate_plan(
                geocoded["formatted_address"], geocoded["lat"], geocoded["lon"], openai_key,
                **({"model": model} if model else {}),
            )
        except AIPlanError as exc:
            print(f"Erreur du plan IA : {exc}", file=sys.stderr)
            return 1
        print(f"Plan de vol conçu par OpenAI : {len(maneuvers)} figure(s)")

    # Chaîne les figures (chacune démarre où la précédente finit) : un tracé
    # continu se lit mieux qu'une juxtaposition de boucles disjointes, et
    # c'est la même condition qu'exige une vidéo non-stop (simple_mode.video).
    maneuvers = chain_maneuvers(maneuvers)

    overlay = render(image, mpp, maneuvers, title=geocoded["formatted_address"])

    out_png = Path(f"{args.out}.png")
    out_md = Path(f"{args.out}.md")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(out_png)

    report = build_report(
        address=geocoded["formatted_address"],
        lat=geocoded["lat"],
        lon=geocoded["lon"],
        zoom=args.zoom,
        mpp=mpp,
        maneuvers=maneuvers,
    )
    out_md.write_text(report, encoding="utf-8")

    print(f"Adresse résolue : {geocoded['formatted_address']}")
    print(f"Image (précise) : {out_png}")
    print(f"Texte           : {out_md}")

    if args.ai_illustration:
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            print("Avertissement : --ai-illustration ignoré, OPENAI_API_KEY absent.", file=sys.stderr)
        else:
            from .ai_image import AIIllustrationError, generate_illustration

            image_model = args.openai_image_model or os.environ.get("OPENAI_IMAGE_MODEL") or None
            try:
                illustration = generate_illustration(
                    image, maneuvers, geocoded["formatted_address"], openai_key,
                    **({"model": image_model} if image_model else {}),
                )
                out_illustration = Path(f"{args.out}.illustration.png")
                illustration.save(out_illustration)
                print(f"Illustration IA : {out_illustration} (indicative, non pixel-exacte)")
            except AIIllustrationError as exc:
                print(f"Avertissement : illustration IA échouée : {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
