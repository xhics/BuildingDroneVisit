"""Tests de la fusion photographique multi-vues pour les façades.

Ces tests visent les cas réellement risqués que la couverture précédente ne
couvrait pas : registration refusée mais utilisée quand même, hauteur de mur
constante au lieu de ``wh``, bâtiment voisin confondu avec la cible, et
occulteur (voiture) non retiré du masque.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from pathlib import Path

from hotel_pipeline.conditioning import facade_texture as ft


def _proj_cam(width=640, height=480):
    class _Cam:
        position = np.array([0.0, -30.0, 2.5], dtype=float)
        f = 400.0

        def __init__(self):
            self.w, self.h = width, height

        def project(self, points):
            d = np.asarray(points, float) - self.position
            zc = d[:, 1]
            safe = np.maximum(zc, 1e-6)
            x = self.w / 2 + self.f * d[:, 0] / safe
            y = self.h / 2 - self.f * d[:, 2] / safe
            return np.stack([x, y], axis=1), zc

    return _Cam()


def test_texture_registration_refuses_a_rejected_registration() -> None:
    allowed, reason = ft._texture_registration_allowed(
        {"status": "refused", "metrics": {"fit": {"p90_m": 1.0}}}
    )
    assert not allowed
    assert "refus" in reason.lower()


def test_texture_registration_refuses_an_imprecise_accept() -> None:
    allowed, reason = ft._texture_registration_allowed(
        {"status": "accepted", "metrics": {"fit": {"p90_m": 3.0}}}
    )
    assert not allowed
    assert "imprécise" in reason.lower()


def test_texture_registration_allows_a_precise_accept() -> None:
    allowed, reason = ft._texture_registration_allowed(
        {"status": "accepted", "metrics": {"fit": {"p90_m": 2.15}}}
    )
    assert allowed
    assert reason == ""


def test_edge_height_uses_wall_top_per_edge() -> None:
    target = {"h": 10.0, "wh": [8.0, 12.0, 9.0, 11.0]}
    assert ft._edge_height(target, 0) == 10.0  # (8 + 12) / 2
    assert ft._edge_height(target, 1) == 10.5  # (12 + 9) / 2


def test_edge_height_falls_back_to_constant_h() -> None:
    assert ft._edge_height({"h": 9.0}, 3) == 9.0


def test_facade_polygon_mask_clips_to_the_target_wall() -> None:
    from hotel_pipeline.geo.orthofacade import plane_from_edge

    cam = _proj_cam()
    plane = plane_from_edge(
        np.array([-5.0, 0.0, 0.0]), np.array([5.0, 0.0, 0.0]), 6.0, "EDGE"
    )
    mask = ft._facade_polygon_mask(cam, plane, cam.w, cam.h)
    assert mask is not None
    # Le mur projeté couvre une région centrale, pas toute l'image.
    assert mask.any()
    assert not mask.all()
    # Un pixel bien à l'écart du mur en est exclu.
    assert not bool(mask[10, 10])
    # Le centre du mur projeté en fait partie.
    assert bool(mask[233, 320])


def test_semantic_building_mask_subtracts_a_car_occluder(tmp_path: Path) -> None:
    obs_dir = tmp_path / "11_conditioning"
    obs_dir.mkdir(parents=True)
    image = Image.new("RGB", (100, 100))
    image_path = tmp_path / "view.jpg"
    image.save(image_path)
    payload = {
        "inputs": [{"asset_id": "A", "path": str(image_path)}],
        "observations": [
            {
                "asset_id": "A",
                "class": "building",
                "segmentation_2d": {
                    "type": "polygon",
                    "points": [[0, 0], [100, 0], [100, 100], [0, 100]],
                },
            },
            {
                "asset_id": "A",
                "class": "car",
                "segmentation_2d": {
                    "type": "polygon",
                    "points": [[10, 10], [30, 10], [30, 30], [10, 30]],
                },
            },
        ],
    }
    (obs_dir / "semantic_observations.json").write_text(__import__("json").dumps(payload))

    masks = ft._semantic_building_masks(_workspace(tmp_path))
    assert masks, "un masque aurait dû être produit"
    mask = masks["A"]
    assert bool(mask[50, 50])   # centre du bâtiment conservé
    assert not bool(mask[20, 20])  # voiture retirée du masque


def _workspace(tmp_path: Path):
    class _WS:
        def __init__(self, root):
            self.root = Path(root)

        def path(self, *parts):
            return self.root.joinpath(*parts)

    return _WS(tmp_path)
