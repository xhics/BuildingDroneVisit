"""Cache de lecture des silhouettes : ne relire que ce qui a changé."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hotel_pipeline.conditioning.silhouette import (
    SilhouetteMap,
    _cache_key,
    _cache_load,
    _cache_store,
)


@pytest.fixture()
def image(tmp_path: Path) -> Path:
    path = tmp_path / "vue.jpg"
    path.write_bytes(b"des octets qui font une image")
    return path


class TestCacheKey:
    def test_same_image_and_settings_give_the_same_key(self, image: Path):
        assert _cache_key(image, 32, "vit") == _cache_key(image, 32, "vit")

    def test_changed_content_changes_the_key(self, image: Path):
        before = _cache_key(image, 32, "vit")
        image.write_bytes(b"une autre photo, prise plus tard")
        assert _cache_key(image, 32, "vit") != before

    def test_tile_size_is_part_of_the_key(self, image: Path):
        assert _cache_key(image, 32, "vit") != _cache_key(image, 64, "vit")

    def test_model_is_part_of_the_key(self, image: Path):
        """Un autre encodeur donne une autre lecture : le cache doit le voir."""
        assert _cache_key(image, 32, "vit-b") != _cache_key(image, 32, "vit-l")

    def test_renaming_does_not_change_the_key(self, image: Path, tmp_path: Path):
        before = _cache_key(image, 32, "vit")
        moved = tmp_path / "autre_nom.jpg"
        moved.write_bytes(image.read_bytes())
        assert _cache_key(moved, 32, "vit") == before


class TestRoundTrip:
    def test_stored_reading_comes_back_identical(self, tmp_path: Path):
        found = SilhouetteMap(
            asset_id="A",
            labels=np.array([[0, 1], [2, 3]]),
            classes=["a", "b", "c", "d"],
            tile_px=32,
            bearing_deg=90.0,
        )
        path = tmp_path / "k.json"
        _cache_store(path, found)
        again = _cache_load(path, "A", 90.0)

        assert again is not None
        assert np.array_equal(again.labels, found.labels)
        assert again.classes == found.classes
        assert again.tile_px == 32
        assert again.bearing_deg == 90.0

    def test_missing_entry_is_a_miss_not_a_failure(self, tmp_path: Path):
        assert _cache_load(tmp_path / "absent.json", "A", None) is None

    def test_corrupt_entry_is_a_miss_not_a_failure(self, tmp_path: Path):
        """Un cache abîmé fait relire l'image, il ne fait pas tomber le rendu."""
        path = tmp_path / "k.json"
        path.write_text("{ceci n'est pas du json", encoding="utf-8")
        assert _cache_load(path, "A", None) is None

    def test_entry_missing_a_field_is_a_miss(self, tmp_path: Path):
        path = tmp_path / "k.json"
        path.write_text(json.dumps({"labels": [[0]]}), encoding="utf-8")
        assert _cache_load(path, "A", None) is None

    def test_bearing_comes_from_the_caller_not_the_cache(self, tmp_path: Path):
        """Le cap dépend de la pose demandée, pas de la photo lue."""
        path = tmp_path / "k.json"
        _cache_store(
            path,
            SilhouetteMap("A", np.array([[0]]), ["a"], 32, bearing_deg=10.0),
        )
        assert _cache_load(path, "A", 250.0).bearing_deg == 250.0

    def test_store_survives_an_unwritable_directory(self, tmp_path: Path):
        blocked = tmp_path / "fichier"
        blocked.write_text("", encoding="utf-8")
        _cache_store(
            blocked / "sous" / "k.json",
            SilhouetteMap("A", np.array([[0]]), ["a"], 32, None),
        )
