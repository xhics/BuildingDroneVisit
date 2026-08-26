"""Independent control-point checks; inverse round-trips are not evidence."""

from __future__ import annotations

import math
import numpy as np


def validate_control_points(
    transformed_xyz: np.ndarray,
    expected_xyz: np.ndarray,
    *,
    max_horizontal_error_m: float,
    max_vertical_error_m: float,
    max_azimuth_error_deg: float,
) -> dict:
    actual, expected = np.asarray(transformed_xyz, float), np.asarray(expected_xyz, float)
    if actual.shape != expected.shape or actual.ndim != 2 or actual.shape[1] != 3 or len(actual) < 2:
        raise ValueError("at least two matching independent XYZ control points are required")
    delta = actual - expected
    horizontal = np.linalg.norm(delta[:, :2], axis=1)
    vertical = np.abs(delta[:, 2])
    av, ev = actual[-1, :2] - actual[0, :2], expected[-1, :2] - expected[0, :2]
    aa, ea = math.degrees(math.atan2(av[0], av[1])), math.degrees(math.atan2(ev[0], ev[1]))
    azimuth_error = abs((aa - ea + 180) % 360 - 180)
    passed = (
        float(horizontal.max()) <= max_horizontal_error_m
        and float(vertical.max()) <= max_vertical_error_m
        and azimuth_error <= max_azimuth_error_deg
    )
    return {
        "status": "MEASURED", "passed": passed,
        "horizontal_error_max_m": float(horizontal.max()),
        "vertical_error_max_m": float(vertical.max()),
        "azimuth_error_deg": azimuth_error,
        "control_point_count": len(actual),
    }


__all__ = ["validate_control_points"]
