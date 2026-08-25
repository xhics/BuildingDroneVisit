from __future__ import annotations

import math

import numpy as np
import pytest

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
    fp = [[0, 0], [10, 0], [10, 10], [0, 10]]
    cam = optimal_camera(_payload(fp))
    assert cam["source"] == "target_building_bounds"
    assert 0.0 <= cam["azimuth_deg"] < 360.0
    assert cam["azimuth_deg"] != 210.0 or cam["source"] != "target_building_bounds"


def test_camera_favours_measured_textured_face():
    fp = [[0, 0], [10, 0], [10, 10], [0, 10], [-5, 5]]
    textures = [{"edge_index": 1, "observed_fraction": 0.9, "disagreement_fraction": 0.0}]
    cam = optimal_camera(_payload(fp, textures))
    assert cam["source"] == "measured_coverage"
    assert cam["azimuth_deg"] is not None
    assert abs(((cam["azimuth_deg"] - 0.0 + 180) % 360) - 180) <= 90.0


def test_entrance_priority_overrides_low_coverage():
    fp = [[0, 0], [10, 0], [10, 10], [0, 10]]
    textures = [{"edge_index": 2, "observed_fraction": 0.95, "disagreement_fraction": 0.0}]
    grammar = {"entrance_tower_edge_index": 0, "main_edge_index": 0, "facade_edges": [0, 2]}
    cam = optimal_camera(_payload(fp, textures, grammar))
    assert abs(((cam["azimuth_deg"] - 0.0 + 180) % 360) - 180) <= 90.0


def test_altitude_and_distance_framed_by_geometry():
    fp = [[0, 0], [40, 0], [40, 20], [0, 20]]
    cam = optimal_camera(_payload(fp))
    assert cam["target_distance_m"] >= 35.0
    assert 14.0 <= cam["altitude_deg"] <= 32.0
    assert cam["facade_azimuth_deg"] == cam["azimuth_deg"]
    assert math.isfinite(cam["facade_altitude_deg"])


def test_azimuth_convention_matches_viewer():
    fp = [[0, 0], [10, 0], [10, 10], [0, 10]]
    textures = [{"edge_index": 1, "observed_fraction": 0.9, "disagreement_fraction": 0.0}]
    cam = optimal_camera(_payload(fp, textures))
    focus = cam["focus"]
    az = math.radians(cam["azimuth_deg"])
    dist = cam["target_distance_m"]
    eye = [
        focus[0] + math.cos(az) * dist * math.cos(math.radians(cam["altitude_deg"])),
        focus[1] + math.sin(az) * dist * math.cos(math.radians(cam["altitude_deg"])),
    ]
    centre = [focus[0], focus[1]]
    outward = [eye[i] - centre[i] for i in range(2)]
    norm = math.hypot(*outward)
    if norm > 1e-6:
        outward = [outward[0] / norm, outward[1] / norm]
        edge = fp[1]
        to_edge = [edge[0] - centre[0], edge[1] - centre[1]]
        to_edge_norm = math.hypot(*to_edge)
        if to_edge_norm > 1e-6:
            to_edge = [to_edge[0] / to_edge_norm, to_edge[1] / to_edge_norm]
            dot = sum(outward[i] * to_edge[i] for i in range(2))
            assert dot > 0.5
