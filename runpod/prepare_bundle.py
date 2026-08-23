"""Prépare le paquet minimal à téléverser sur une VM GPU.

La règle est de ne transférer que ce que le modèle consomme. Le workspace du
pilote pèse trois gigaoctets — rasters dérivés, tuile LiDAR, sept cent images
brutes, environnements virtuels de runs COLMAP passés — dont le modèle de forme
n'utilise rien. Le lot qualifié, lui, tient en quelques mégaoctets.

Transférer moins, c'est démarrer plus vite ; et sur une machine facturée à la
minute, le temps de téléversement est du temps payé.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from hotel_pipeline.conditioning.shape_input import build


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hotel_id")
    parser.add_argument("--out", type=Path, default=Path("runpod/bundle"))
    parser.add_argument(
        "--max-images", type=int, default=24, help="Plafond d'images du lot."
    )
    args = parser.parse_args()

    from hotel_pipeline.workspace import Workspace

    workspace = Workspace(args.hotel_id)
    screening = workspace.path("09_confidence", "identity_screening.json")
    manifest = workspace.path("00_manifest", "asset_manifest.json")
    for required in (screening, manifest):
        if not required.is_file():
            print(f"absent : {required}")
            print("lancer d'abord `identity screen`")
            return 1

    lot = build(screening, manifest, max_images=args.max_images)
    if not lot.images:
        print("aucune image qualifiée — rien à téléverser")
        return 1

    images_dir = args.out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for existing in images_dir.glob("*"):
        existing.unlink()

    total = 0
    for index, image in enumerate(lot.images):
        # Le nom porte l'azimut : le recalage ultérieur en a besoin, et un
        # dossier trié par angle se relit sans consulter le manifeste.
        bearing = "xxx" if image.bearing_deg is None else f"{image.bearing_deg:03.0f}"
        target = images_dir / f"{index:02d}_{bearing}deg_{image.path.name}"
        shutil.copy2(image.path, target)
        total += target.stat().st_size

    (args.out / "shape_input.json").write_text(
        json.dumps(lot.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"{len(lot.images)} image(s), {total / 1e6:.1f} Mo → {args.out}")
    print(f"  {len(lot.placed)} placée(s), arc {lot.angular_span():.0f}°")
    print(f"  écartées : {lot.rejected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
