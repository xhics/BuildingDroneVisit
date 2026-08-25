from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from runpod.validate_batch import validate_input, validate_output


def _bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    images = bundle / "images"
    images.mkdir(parents=True)
    entries = []
    for index in range(8):
        path = images / f"{index:02d}.jpg"
        Image.new("RGB", (640, 480), (20 + index * 25, 40, 60)).save(path)
        entries.append({
            "asset_id": f"a{index}",
            "path": f"images/{path.name}",
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    (bundle / "shape_input.json").write_text(json.dumps({
        "hotel_id": "hotel-test",
        "count": 8,
        "placed": 8,
        "angular_span_deg": 180,
        "images": entries,
        "usage_policy": {
            "scope": "experimental_demo_only",
            "production_eligible": False,
        },
    }), "utf-8")
    return bundle


def test_le_lot_gpu_est_verifie_avant_facturation(tmp_path: Path) -> None:
    assert validate_input(_bundle(tmp_path))["status"] == "passed"


def test_une_image_modifiee_est_refusee(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    (bundle / "images" / "00.jpg").write_bytes(b"corrompu")
    report = validate_input(bundle)
    assert report["status"] == "failed"
    assert not next(c for c in report["checks"] if c["check_id"] == "image_integrity")["passed"]


def test_la_sortie_vggt_doit_porter_points_et_cameras(tmp_path: Path) -> None:
    root = tmp_path / "out" / "vggt"
    root.mkdir(parents=True)
    points = [f"{i / 1000:.3f} 0 0 120 80 40" for i in range(1001)]
    (root / "shape.ply").write_text(
        "\n".join([
            "ply", "format ascii 1.0", "element vertex 1001",
            "property float x", "property float y", "property float z",
            "property uchar red", "property uchar green", "property uchar blue",
            "end_header", *points, "",
        ]),
        "utf-8",
    )
    (root / "shape_run.json").write_text(json.dumps({
        "backend": "vggt", "device": "cuda", "images": 8, "points": 1001,
    }), "utf-8")
    (root / "cameras.json").write_text(json.dumps({
        "cameras": [{"image": f"{i}.jpg"} for i in range(8)],
    }), "utf-8")

    assert validate_output(tmp_path / "out", ["vggt"], 8)["status"] == "passed"


def test_une_sortie_ply_tronquee_est_refusee(tmp_path: Path) -> None:
    root = tmp_path / "out" / "vggt"
    root.mkdir(parents=True)
    (root / "shape.ply").write_text(
        "\n".join([
            "ply", "format ascii 1.0", "element vertex 1001",
            "property float x", "property float y", "property float z",
            "end_header", "0 0 0", "1 0 0", "",
        ]),
        "utf-8",
    )
    (root / "shape_run.json").write_text(json.dumps({
        "backend": "vggt", "device": "cuda", "images": 8, "points": 1001,
    }), "utf-8")
    (root / "cameras.json").write_text(json.dumps({
        "cameras": [{"image": f"{i}.jpg"} for i in range(8)],
    }), "utf-8")

    report = validate_output(tmp_path / "out", ["vggt"], 8)
    assert report["status"] == "failed"
    assert not next(c for c in report["checks"] if c["check_id"] == "vggt_points")["passed"]
