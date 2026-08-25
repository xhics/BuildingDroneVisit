from __future__ import annotations

import math

from hotel_pipeline.conditioning.viewpoint import optimal_camera


def _building(fp, h=12.0):
    return {"id": "T", "target": True, "h": h, "fp": fp, "wh": [h] * len(fp)}


def _payload(fp, textures=None, grammar=None):
    payload = {
        "volumes": [_building(fp)],
        "reference_fusion": {"textures": textures or []},
    }
    if grammar:
        payload["facade_grammar"] = grammar
    return payload


def test_camera_is_derived_not_hardcoded():
    # Carré simple : aucune texture, pas de grammaire -> borne générique.
    fp = [[0, 0], [10, 0], [10, 10], [0, 10]]
    cam = optimal_camera(_payload(fp))
    assert cam["source"] == "target_building_bounds"
    assert 0.0 <= cam["azimuth_deg"] < 360.0
    # Jamais la valeur magique figée de l'ancien code.
    assert cam["azimuth_deg"] != 210.0 or cam["source"] != "target_building_bounds"


def test_camera_favours_measured_textured_face():
    # L-forme : seule la face 1 (SE) porte une forte couverture.
    fp = [[0, 0], [10, 0], [10, 10], [0, 10], [-5, 5]]
    textures = [{"edge_index": 1, "observed_fraction": 0.9, "disagreement_fraction": 0.0}]
    cam = optimal_camera(_payload(fp, textures))
    assert cam["source"] == "measured_coverage_optimization"
    assert cam["azimuth_deg"] is not None
    # La face 1 pointe vers le SE (~135°) : la caméra doit la cadrer (<=90° d'écart).
    assert abs(((cam["azimuth_deg"] - 135.0 + 180) % 360) - 180) <= 90.0


def test_entrance_priority_overrides_low_coverage():
    # Face d'entrée (edge 0, N ~0°) peu couverte mais prioritaire face à une
    # face très couverte sur le côté opposé.
    fp = [[0, 0], [10, 0], [10, 10], [0, 10]]
    textures = [{"edge_index": 2, "observed_fraction": 0.95, "disagreement_fraction": 0.0}]
    grammar = {"entrance_tower_edge_index": 0, "main_edge_index": 0, "facade_edges": [0, 2]}
    cam = optimal_camera(_payload(fp, textures, grammar))
    # L'entrée (N) doit être visible : écart azimut/0° <= 90°.
    assert abs(((cam["azimuth_deg"] - 0.0 + 180) % 360) - 180) <= 90.0


def test_altitude_and_distance_framed_by_geometry():
    fp = [[0, 0], [40, 0], [40, 20], [0, 20]]
    cam = optimal_camera(_payload(fp))
    assert cam["target_distance_m"] >= 35.0
    assert 14.0 <= cam["altitude_deg"] <= 32.0
    assert cam["facade_azimuth_deg"] == cam["azimuth_deg"]
    assert math.isfinite(cam["facade_altitude_deg"])
