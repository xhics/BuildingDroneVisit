from __future__ import annotations

from hotel_pipeline.conditioning.facade_grammar import enrich


def _horizontal_distance(a: list[float], b: list[float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _payload() -> dict:
    return {
        "volumes": [
            {
                "id": "target",
                "target": True,
                "h": 12.0,
                "fp": [[0, 0], [32, 0], [32, 18], [0, 18]],
                "rf": [[0, 1, 2]],
                "topology": {"watertight": True},
            }
        ],
        "semantic_support_points": [
            {"instance_id": "door-1", "point3d_id": 1, "class": "door", "xyz": [15, -1, 1]},
            {"instance_id": "window-1", "point3d_id": 2, "class": "window", "xyz": [8, -1, 5]},
        ],
        "semantic_surfaces": [
            {
                "class": "road_sign",
                "surface": {"vertices": [[4, -8, 10], [7, -8, 10], [7, -8, 13]], "normal": [0, -1, 0]},
                "validation": {"extent_u_m": 3.0},
            }
        ],
        "ground": [{"kind": "road", "ring": [[0, 0], [1, 0], [1, 1]]}],
        "vegetation": [{"c": [1, 1], "r": 1, "h": 4}],
    }


def test_enrich_cree_une_grammaire_de_facade_auditable() -> None:
    payload = enrich(_payload())
    grammar = payload["facade_grammar"]

    assert grammar["status"] == "generated"
    assert grammar["floors"] == 3
    assert grammar["semantic_door_support"] == 1
    assert grammar["feature_counts"]["window"] >= 24
    assert grammar["feature_counts"]["canopy"] >= 1
    assert grammar["feature_counts"]["gable"] == 1
    assert grammar["feature_counts"]["sign"] == 1
    assert grammar["similarity"]["threshold_met"] is True
    assert grammar["similarity"]["photometric_claim"] is False


def test_enrich_ne_promet_rien_sans_empreinte_cible() -> None:
    payload = enrich({"volumes": []})
    assert payload["facade_features"] == []
    assert payload["facade_grammar"]["status"] == "blocked"


def test_canopy_box_faces_are_vertical_or_horizontal() -> None:
    payload = enrich(_payload())
    canopy = [
        feature
        for feature in payload["facade_features"]
        if feature["kind"] == "canopy"
    ]

    assert len(canopy) == 5
    for face in canopy[:4]:
        vertices = face["vertices"]
        assert _horizontal_distance(vertices[0], vertices[3]) < 1e-9
        assert _horizontal_distance(vertices[1], vertices[2]) < 1e-9
    assert len({vertex[2] for vertex in canopy[4]["vertices"]}) == 1
