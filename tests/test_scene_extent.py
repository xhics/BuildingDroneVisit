"""Les sources doivent être interrogées à l'échelle de ce qui sera rendu."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from shapely import wkt as shapely_wkt

from hotel_pipeline.conditioning.scene_extent import (
    scene_envelope_wkt,
    uncovered_volumes,
)


def _entry(feature_id: str, role: str, wkt: str, status: str = "resolved") -> dict:
    return {
        "feature_id": feature_id,
        "role": role,
        "resolution_status": status,
        "wgs84_wkt": wkt,
        "projected_wkt": wkt,
    }


def _box(x0: float, y0: float, size: float = 0.001) -> str:
    pts = [
        (x0, y0),
        (x0 + size, y0),
        (x0 + size, y0 + size),
        (x0, y0 + size),
        (x0, y0),
    ]
    return "POLYGON ((" + ", ".join(f"{x} {y}" for x, y in pts) + "))"


def _manifest(tmp_path: Path, entries: list[dict]) -> Path:
    path = tmp_path / "capture_geometry.json"
    path.write_text(json.dumps({"geometries": entries}), encoding="utf-8")
    return path


def test_l_enveloppe_couvre_la_cible_et_ses_obstacles(tmp_path: Path) -> None:
    """C'est le défaut corrigé : la cible seule mesurait 72 m, la scène 1 km."""
    path = _manifest(
        tmp_path,
        [
            _entry("TARGET_BUILDING", "target_building", _box(0.0, 0.0)),
            _entry("OBST", "obstacle_building", _box(0.01, 0.01)),
        ],
    )
    envelope = shapely_wkt.loads(scene_envelope_wkt(path))
    target = shapely_wkt.loads(_box(0.0, 0.0))

    assert envelope.contains(target.centroid)
    assert envelope.bounds[2] > target.bounds[2]
    assert envelope.area > target.area * 10


def test_une_geometrie_dementie_n_elargit_pas_l_emprise(tmp_path: Path) -> None:
    """Demander des tuiles sur la foi d'une erreur reviendrait à la propager."""
    path = _manifest(
        tmp_path,
        [
            _entry("TARGET_BUILDING", "target_building", _box(0.0, 0.0)),
            _entry("OBST", "obstacle_building", _box(5.0, 5.0), status="stale"),
        ],
    )
    envelope = shapely_wkt.loads(scene_envelope_wkt(path))
    assert envelope.bounds[2] < 1.0


def test_les_voies_n_entrent_pas_dans_l_enveloppe(tmp_path: Path) -> None:
    """Seuls les volumes rendus comptent : une route ne se reconstruit pas."""
    path = _manifest(
        tmp_path,
        [
            _entry("TARGET_BUILDING", "target_building", _box(0.0, 0.0)),
            _entry("ROAD", "road_candidate", _box(2.0, 2.0)),
        ],
    )
    envelope = shapely_wkt.loads(scene_envelope_wkt(path))
    assert envelope.bounds[2] < 1.0


def test_sans_volume_rendu_aucune_enveloppe(tmp_path: Path) -> None:
    path = _manifest(tmp_path, [_entry("ROAD", "road_candidate", _box(0.0, 0.0))])
    assert scene_envelope_wkt(path) is None


def test_les_volumes_hors_couverture_sont_nommes() -> None:
    """« Hors couverture » est un motif ; « pas mesuré » n'en est pas un."""
    import numpy as np

    class _Prism:
        def __init__(self, feature_id: str, x: float) -> None:
            self.feature_id = feature_id
            self.footprint = np.array([[x, 0.0], [x + 5, 0.0], [x + 5, 5.0]])

    class _Scene:
        prisms = [_Prism("dedans", 10.0), _Prism("dehors", 900.0)]

    assert uncovered_volumes(_Scene(), (0.0, -10.0, 100.0, 100.0)) == ["dehors"]
