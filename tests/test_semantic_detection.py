from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from hotel_pipeline.conditioning.semantic_detection import (
    canonical_class,
    mask_to_polygon,
    resolve_device,
    select_validated_images,
)
from hotel_pipeline.workspace import Workspace


def test_canonical_class_maps_architectural_vocabulary() -> None:
    assert canonical_class("structural beam") == "beam"
    assert canonical_class("hotel entrance door") == "door"
    assert canonical_class("road sign") == "road_sign"
    assert canonical_class("evergreen tree") == "tree_evergreen"
    assert canonical_class("deciduous tree") == "tree_deciduous"
    assert canonical_class("unknown fixture") == "architectural_object"


def test_resolve_device_can_be_forced_to_cpu() -> None:
    assert resolve_device("cpu") == "cpu"


def test_mask_to_polygon_keeps_an_explicit_method_and_score() -> None:
    mask = np.zeros((40, 50), dtype=bool)
    mask[10:30, 15:35] = True

    polygon = mask_to_polygon(mask, method="sam2-test", score=0.875)

    assert polygon is not None
    assert polygon["method"] == "sam2-test"
    assert polygon["mask_score"] == 0.875
    assert polygon["area_px2"] > 300
    assert len(polygon["points"]) >= 4


def test_select_validated_images_relocates_a_workspace_from_another_machine(
    tmp_path: Path,
) -> None:
    workspace = Workspace("hotel-test", root=tmp_path)
    image_path = workspace.path("02_images", "reference_only", "view.jpg")
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(image_path)

    confidence = workspace.path("09_confidence")
    confidence.mkdir(parents=True)
    (confidence / "identity_screening.json").write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "asset_id": "view-1",
                        "status": "match",
                        "path": (
                            "/old/mac/project/work/hotel-test/"
                            "02_images/reference_only/view.jpg"
                        ),
                    }
                ]
            }
        ),
        "utf-8",
    )
    localization = workspace.path("07_reconstruction", "localization")
    localization.mkdir(parents=True)
    (localization / "anchor-localization-test.json").write_text(
        json.dumps(
            {
                "poses": [
                    {
                        "asset_id": "view-1",
                        "decision": "accepted",
                        "evidence_class": "anchor_measured",
                    }
                ]
            }
        ),
        "utf-8",
    )

    selected = select_validated_images(workspace)

    assert len(selected) == 1
    assert selected[0].path == image_path
