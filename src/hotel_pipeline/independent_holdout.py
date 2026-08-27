"""Independent TRAIN/HOLDOUT protocol established before reconstruction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Callable, Iterable


class HoldoutState(StrEnum):
    MEASURED = "MEASURED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class ImageSpatialRecord:
    image_id: str
    position: tuple[float, float, float]
    yaw_deg: float
    pitch_deg: float
    visible_surface_ids: tuple[str, ...] = ()

    @property
    def stratum(self) -> tuple[int, int]:
        return (int(self.yaw_deg % 360 // 45), int((self.pitch_deg + 90) // 30))


@dataclass(frozen=True)
class IndependentSplit:
    train_ids: tuple[str, ...]
    holdout_ids: tuple[str, ...]
    holdout_surface_coverage: float
    azimuth_bins: tuple[int, ...]
    elevation_bins: tuple[int, ...]
    digest: str


@dataclass(frozen=True)
class HoldoutResult:
    image_id: str
    state: HoldoutState
    localization_success: bool
    silhouette_iou: float | None = None
    boundary_error_px: float | None = None
    edge_f1: float | None = None
    reprojection_rmse_px: float | None = None
    depth_rmse_m: float | None = None
    appearance_score: float | None = None
    visible_surface_ids: tuple[str, ...] = ()
    failed_surface_ids: tuple[str, ...] = ()


def _split_digest(train: Iterable[str], holdout: Iterable[str]) -> str:
    payload = {"train": sorted(train), "holdout": sorted(holdout)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def spatial_train_holdout_split(
    records: list[ImageSpatialRecord], holdout_fraction: float = 0.2,
    *, min_train_per_stratum: int = 2,
) -> IndependentSplit:
    """Deterministic stratified split across pose bins and visible surfaces."""
    if not 0.0 <= holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in [0, 1)")
    ids = [record.image_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("image IDs must be unique")
    target = min(max(0, round(len(records) * holdout_fraction)), max(0, len(records) - 3))
    by_stratum: dict[tuple[int, int], list[ImageSpatialRecord]] = {}
    for record in records:
        by_stratum.setdefault(record.stratum, []).append(record)
    remaining = {key: len(value) for key, value in by_stratum.items()}
    selected: list[ImageSpatialRecord] = []
    covered_surfaces: set[str] = set()
    candidates = sorted(records, key=lambda item: item.image_id)
    while len(selected) < target:
        eligible = [
            item for item in candidates
            if item not in selected and remaining[item.stratum] > min_train_per_stratum
        ]
        if not eligible:
            break
        best = max(eligible, key=lambda item: (
            len(set(item.visible_surface_ids) - covered_surfaces),
            remaining[item.stratum],
            -sum(ord(char) for char in item.image_id),
        ))
        selected.append(best)
        remaining[best.stratum] -= 1
        covered_surfaces.update(best.visible_surface_ids)
    holdout = tuple(sorted(item.image_id for item in selected))
    train = tuple(sorted(set(ids) - set(holdout)))
    all_surfaces = {surface for item in records for surface in item.visible_surface_ids}
    coverage = len(covered_surfaces) / len(all_surfaces) if all_surfaces else 0.0
    return IndependentSplit(
        train, holdout, coverage,
        tuple(sorted({item.stratum[0] for item in selected})),
        tuple(sorted({item.stratum[1] for item in selected})),
        _split_digest(train, holdout),
    )


def assert_zero_holdout_leakage(
    split: IndependentSplit, stage_inputs: dict[str, Iterable[str]],
) -> None:
    holdout = set(split.holdout_ids)
    overlap = set(split.train_ids) & holdout
    leaks = {stage: sorted(holdout & set(ids)) for stage, ids in stage_inputs.items()}
    leaks = {stage: ids for stage, ids in leaks.items() if ids}
    if overlap or leaks:
        raise ValueError(f"TRAIN/HOLDOUT leakage: split={sorted(overlap)}, stages={leaks}")


def digest_tree(path: Path) -> str:
    """Content digest of a frozen model directory or canonical scene file."""
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    if not path.is_dir():
        return digest.hexdigest()
    for file in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file.relative_to(path).as_posix().encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def localize_against_frozen_model(
    model_path: Path, holdout_ids: Iterable[str],
    localizer: Callable[[str, Path], HoldoutResult | None],
) -> list[HoldoutResult]:
    """Localize holdouts without permitting a single model byte to change."""
    before = digest_tree(model_path)
    results = []
    for image_id in holdout_ids:
        result = localizer(image_id, model_path)
        results.append(result or HoldoutResult(
            image_id, HoldoutState.INSUFFICIENT_EVIDENCE, False,
        ))
    after = digest_tree(model_path)
    if before != after:
        raise RuntimeError("holdout localization modified the frozen TRAIN model")
    return results


def aggregate_by_surface(results: Iterable[HoldoutResult]) -> dict[str, dict]:
    buckets: dict[str, list[HoldoutResult]] = {}
    for result in results:
        if result.state is not HoldoutState.MEASURED:
            continue
        for surface_id in result.visible_surface_ids:
            buckets.setdefault(surface_id, []).append(result)
    output = {}
    for surface_id, rows in sorted(buckets.items()):
        scores = [row.silhouette_iou for row in rows if row.silhouette_iou is not None]
        output[surface_id] = {
            "images": len(rows),
            "novel_view_validation_score": sum(scores) / len(scores) if scores else None,
            "failed_images": sorted(row.image_id for row in rows if surface_id in row.failed_surface_ids),
        }
    return output


def result_as_dict(result: HoldoutResult) -> dict:
    payload = asdict(result)
    payload["state"] = result.state.value
    return payload


__all__ = [
    "HoldoutResult", "HoldoutState", "ImageSpatialRecord", "IndependentSplit",
    "aggregate_by_surface", "assert_zero_holdout_leakage", "digest_tree",
    "localize_against_frozen_model", "result_as_dict", "spatial_train_holdout_split",
]
