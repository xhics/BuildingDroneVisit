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
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requis pour le lot RunPod")
    device = "cuda"
    dtype = (
        torch.bfloat16
        if torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )

    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    images = load_and_preprocess_images([str(p) for p in image_paths]).to(device)

    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        predictions = model(images)

    world_points = predictions["world_points"].float().cpu().numpy()
    points = world_points.reshape(-1, 3)
    model_images = predictions["images"].float().cpu().numpy()
    if model_images.ndim == 5:
        model_images = model_images[0]
    colours = np.clip(
        np.transpose(model_images, (0, 2, 3, 1)).reshape(-1, 3) * 255.0,
        0,
        255,
    ).astype(np.uint8)
    confidence = (
        predictions["world_points_conf"].float().cpu().numpy().reshape(-1)
        if "world_points_conf" in predictions
        else None
    )

    # Un nuage feed-forward porte beaucoup de points de faible confiance —
    # ciel, chaussée, arrière-plan lointain. Les garder noierait la façade.
    finite = np.isfinite(points).all(axis=1)
    keep = finite
    if confidence is not None:
        finite_confidence = np.isfinite(confidence)
        keep &= finite_confidence
        if finite_confidence.any():
            threshold = float(np.percentile(confidence[finite_confidence], 50))
            keep &= confidence >= threshold
        else:
            keep &= False
    points_before_filter = int(len(points))
    nonfinite_removed = int((~finite).sum())
    points = points[keep]
    colours = colours[keep]

    _write_ply(out_dir / "shape.ply", points, colours)
    extrinsics, intrinsics = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    extrinsics = extrinsics.float().cpu().numpy()[0]
    intrinsics = intrinsics.float().cpu().numpy()[0]
    (out_dir / "cameras.json").write_text(
        json.dumps(
            {
                "convention": "opencv_camera_from_world",
                "image_size_hw": list(images.shape[-2:]),
                "cameras": [
                    {
                        "image": image_paths[index].name,
                        "extrinsic": extrinsics[index].tolist(),
                        "intrinsic": intrinsics[index].tolist(),
                    }
                    for index in range(len(image_paths))
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "backend": "vggt",
        "device": device,
        "images": len(image_paths),
        "points": int(len(points)),
        "points_before_filter": points_before_filter,
        "nonfinite_removed": nonfinite_removed,
        "confidence_percentile": 50,
        "camera_file": "cameras.json",
    }


def run_mapanything(image_paths: list[Path], out_dir: Path) -> dict:
    import numpy as np
    import torch
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requis pour le lot RunPod")
    device = "cuda"
    model = MapAnything.from_pretrained("facebook/map-anything").to(device).eval()
    views = load_images([str(path) for path in image_paths])

    with torch.no_grad():
        predictions = model.infer(
            views,
            memory_efficient_inference=True,
            minibatch_size=1,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=True,
            confidence_percentile=50,
        )

    point_sets = []
    for prediction in predictions:
        value = prediction["pts3d"]
        if torch.is_tensor(value):
            value = value.detach().float().cpu().numpy()
        points_for_view = np.asarray(value).reshape(-1, 3)
        point_sets.append(points_for_view)
    points = np.concatenate(point_sets, axis=0)
    points_before_filter = int(len(points))
    finite = np.isfinite(points).all(axis=1)
    # MapAnything encode les pixels masqués par [0, 0, 0]. Les exporter
    # fabrique un énorme point artificiel à l'origine et peut masquer une
    # géométrie par ailleurs exploitable.
    nonzero = np.linalg.norm(points, axis=1) > 1e-8
    keep = finite & nonzero
    points = points[keep]
    _write_ply(out_dir / "shape.ply", points)
    return {
        "backend": "mapanything",
        "device": device,
        "images": len(image_paths),
        "points": int(len(points)),
        "points_before_filter": points_before_filter,
        "nonfinite_removed": int((~finite).sum()),
        "masked_zero_removed": int((finite & ~nonzero).sum()),
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
