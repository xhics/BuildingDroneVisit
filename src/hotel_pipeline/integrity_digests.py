"""Content-addressed cache keys for reconstruction and semantic rasters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def _stable(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def reconstruction_digest(
    model_dir: Path,
    *,
    run_parameters: dict | None = None,
    critical_versions: dict | None = None,
) -> str:
    digest = hashlib.sha256()
    found = False
    for stem in ("cameras", "images", "points3D"):
        candidates = sorted(model_dir.glob(stem + ".*")) + ([model_dir / stem] if (model_dir / stem).is_file() else [])
        for path in candidates:
            found = True
            digest.update(path.name.encode())
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    if not found:
        raise FileNotFoundError(f"no COLMAP model files in {model_dir}")
    digest.update(_stable(run_parameters or {}))
    digest.update(_stable(critical_versions or {}))
    return digest.hexdigest()


def mask_raster_digest(
    pixels: np.ndarray,
    *,
    asset_id: str,
    pixel_transform: tuple[float, ...] | None,
    segmenter_version: str,
) -> str:
    array = np.ascontiguousarray(pixels)
    digest = hashlib.sha256()
    digest.update(array.tobytes())
    digest.update(_stable({
        "shape": array.shape, "dtype": str(array.dtype), "asset_id": asset_id,
        "pixel_transform": pixel_transform, "segmenter_version": segmenter_version,
    }))
    return digest.hexdigest()


__all__ = ["mask_raster_digest", "reconstruction_digest"]
