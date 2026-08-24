from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from hotel_pipeline.conditioning.image_objects import detect, detect_cached


def test_detecte_un_panneau_rectangulaire_et_des_membres(tmp_path: Path) -> None:
    image = np.full((480, 640, 3), 230, dtype=np.uint8)
    cv2.rectangle(image, (170, 130), (470, 300), (20, 20, 20), 8)
    cv2.line(image, (80, 380), (560, 380), (10, 10, 10), 7)
    path = tmp_path / "facade.jpg"
    cv2.imwrite(str(path), image)

    found = detect(path)

    assert any(item["class"] == "panel_or_opening_candidate" for item in found)
    assert any(item["class"] == "linear_member_candidate" for item in found)


def test_cache_est_lie_a_empreinte_de_l_image(tmp_path: Path) -> None:
    image = np.full((240, 320, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (80, 60), (240, 170), (0, 0, 0), 5)
    path = tmp_path / "panel.jpg"
    cv2.imwrite(str(path), image)

    first = detect_cached(path, "asset/test", tmp_path / "cache")
    second = detect_cached(path, "asset/test", tmp_path / "cache")

    assert first == second
    assert list((tmp_path / "cache").glob("*.json"))
