from __future__ import annotations

from pathlib import Path

import pytest

from hotel_pipeline.independent_holdout import (
    HoldoutResult,
    HoldoutState,
    ImageSpatialRecord,
    aggregate_by_surface,
    assert_zero_holdout_leakage,
    localize_against_frozen_model,
    spatial_train_holdout_split,
)


def _records() -> list[ImageSpatialRecord]:
    records = []
    for side, yaw in enumerate((0, 90, 180, 270)):
        for index in range(5):
            records.append(ImageSpatialRecord(
                f"side-{side}-{index}", (side * 10.0, index * 2.0, 4.0 + index),
                yaw, -10.0 + index * 5.0, (f"facade-{side}",),
            ))
    return records


def test_spatial_split_is_disjoint_deterministic_and_covers_surfaces():
    first = spatial_train_holdout_split(_records(), 0.2)
    second = spatial_train_holdout_split(_records(), 0.2)
    assert first == second
    assert set(first.train_ids).isdisjoint(first.holdout_ids)
    assert len(first.train_ids) == 16
    assert len(first.holdout_ids) == 4
    assert first.holdout_surface_coverage >= 0.75
    assert len(first.azimuth_bins) >= 3


def test_every_reconstruction_stage_is_audited_for_leakage():
    split = spatial_train_holdout_split(_records(), 0.2)
    clean = {stage: split.train_ids for stage in (
        "features", "matches", "sfm", "bundle", "dense", "texture", "roof",
    )}
    assert_zero_holdout_leakage(split, clean)
    dirty = {**clean, "matches": [*split.train_ids, split.holdout_ids[0]]}
    with pytest.raises(ValueError, match="matches"):
        assert_zero_holdout_leakage(split, dirty)


def test_holdout_localization_cannot_modify_frozen_model(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    geometry = model / "mesh.bin"
    geometry.write_bytes(b"frozen canonical geometry")

    def mutating_localizer(image_id: str, path: Path):
        geometry.write_bytes(b"changed by holdout")
        return HoldoutResult(image_id, HoldoutState.MEASURED, True)

    with pytest.raises(RuntimeError, match="modified"):
        localize_against_frozen_model(model, ["holdout-1"], mutating_localizer)


def test_unlocalizable_holdout_is_insufficient_evidence_not_fake_score(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "mesh.bin").write_bytes(b"fixed")
    results = localize_against_frozen_model(model, ["unknown"], lambda _id, _path: None)
    assert results[0].state is HoldoutState.INSUFFICIENT_EVIDENCE
    assert not results[0].localization_success
    assert results[0].silhouette_iou is None


def test_holdout_scores_are_aggregated_per_physical_surface():
    results = [
        HoldoutResult("h1", HoldoutState.MEASURED, True, silhouette_iou=0.9, visible_surface_ids=("east",)),
        HoldoutResult("h2", HoldoutState.MEASURED, True, silhouette_iou=0.7, visible_surface_ids=("east", "roof"), failed_surface_ids=("roof",)),
    ]
    surfaces = aggregate_by_surface(results)
    assert surfaces["east"]["novel_view_validation_score"] == pytest.approx(0.8)
    assert surfaces["roof"]["failed_images"] == ["h2"]

