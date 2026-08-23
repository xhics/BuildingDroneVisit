"""Inférence de forme depuis plusieurs vues, sur GPU.

Le script est autonome : il ne dépend pas du paquet `hotel_pipeline`, pour que
la VM n'ait à recevoir que les images et ce fichier. Il écrit un nuage de
points et les poses estimées, que le poste local recale ensuite sur l'emprise
géoréférencée.

Ce qui sort d'un modèle feed-forward vit dans un **repère arbitraire** : ni
échelle métrique, ni orientation connue. Le fichier de sortie le déclare, pour
qu'aucune mesure n'en soit tirée avant recalage.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _write_ply(path: Path, points, colours=None) -> None:
    """Nuage en PLY ASCII, lisible par MeshLab, CloudCompare ou Blender."""
    count = len(points)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {count}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if colours is not None:
        lines += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    lines.append("end_header")
    for index in range(count):
        x, y, z = points[index][:3]
        row = f"{x:.5f} {y:.5f} {z:.5f}"
        if colours is not None:
            r, g, b = colours[index][:3]
            row += f" {int(r)} {int(g)} {int(b)}"
        lines.append(row)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_vggt(image_paths: list[Path], out_dir: Path) -> dict:
    import numpy as np
    import torch
    from PIL import Image
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    images = load_and_preprocess_images([str(p) for p in image_paths]).to(device)

    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        predictions = model(images)

    points = predictions["world_points"].float().cpu().numpy().reshape(-1, 3)
    confidence = (
        predictions["world_points_conf"].float().cpu().numpy().reshape(-1)
        if "world_points_conf" in predictions
        else None
    )

    # Un nuage feed-forward porte beaucoup de points de faible confiance —
    # ciel, chaussée, arrière-plan lointain. Les garder noierait la façade.
    if confidence is not None:
        keep = confidence >= float(np.percentile(confidence, 50))
        points = points[keep]

    _write_ply(out_dir / "shape.ply", points)
    return {
        "backend": "vggt",
        "device": device,
        "images": len(image_paths),
        "points": int(len(points)),
    }


def run_mapanything(image_paths: list[Path], out_dir: Path) -> dict:
    import numpy as np
    import torch
    from mapanything.model import MapAnything

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MapAnything.from_pretrained("facebook/map-anything").to(device).eval()

    with torch.no_grad():
        result = model.run([str(p) for p in image_paths])

    points = np.asarray(result["points3d"]).reshape(-1, 3)
    _write_ply(out_dir / "shape.ply", points)
    return {
        "backend": "mapanything",
        "device": device,
        "images": len(image_paths),
        "points": int(len(points)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", default="vggt", choices=("vggt", "mapanything"))
    args = parser.parse_args()

    image_paths = sorted(
        p
        for p in args.images.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        print(f"aucune image dans {args.images}")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(image_paths)} image(s) — backend {args.backend}")

    runner = run_vggt if args.backend == "vggt" else run_mapanything
    summary = runner(image_paths, args.out)
    summary.update(
        {
            "produced_at": datetime.now(timezone.utc).isoformat(),
            "source_images": [p.name for p in image_paths],
            "caveats": [
                "repère arbitraire : ni échelle métrique ni orientation connue, "
                "le nuage doit être recalé sur l'emprise géoréférencée avant "
                "tout usage métrique",
                "un nuage feed-forward mêle le bâtiment et son environnement ; "
                "la façade doit être isolée avant comparaison",
            ],
        }
    )
    (args.out / "shape_run.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
