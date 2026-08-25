from __future__ import annotations

from hotel_pipeline.conditioning.facade_grammar import enrich
from hotel_pipeline.conditioning.facade_preview import render


def test_preview_rend_la_grammaire_sans_navigateur() -> None:
    payload = enrich(
        {
            "camera": {"focus": [5, 4, 4], "target_distance_m": 25},
            "volumes": [
                {
                    "target": True,
                    "h": 10,
                    "fp": [[0, 0], [10, 0], [10, 8], [0, 8]],
                    "rf": [],
                    "rv": [],
                    "topology": {"watertight": True},
                }
            ],
            "ground": [],
            "vegetation": [],
            "semantic_support_points": [],
            "semantic_surfaces": [],
        }
    )
    image = render(payload, azimuth_deg=210, width=320, height=180)
    assert image.size == (320, 180)
    assert len(image.getcolors(maxcolors=1_000_000) or []) > 8
