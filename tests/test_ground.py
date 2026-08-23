"""Le sol se lit dans les tags, il ne se suppose pas."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hotel_pipeline.conditioning.ground import (
    GroundCover,
    GroundSurface,
    LoneTree,
    from_elements,
    kind_of,
    load,
)


def _way(way_id: int, tags: dict, lat: float = 45.5743, lon: float = -73.4437) -> dict:
    step = 0.0002
    return {
        "type": "way",
        "id": way_id,
        "tags": tags,
        "geometry": [
            {"lat": lat, "lon": lon},
            {"lat": lat + step, "lon": lon},
            {"lat": lat + step, "lon": lon + step},
            {"lat": lat, "lon": lon + step},
        ],
    }


# --- classification ---------------------------------------------------------


def test_les_tags_connus_donnent_une_nature() -> None:
    assert kind_of({"landuse": "grass"}) == "pelouse"
    assert kind_of({"natural": "wood"}) == "boise"
    assert kind_of({"natural": "water"}) == "eau"
    assert kind_of({"surface": "asphalt"}) == "mineral"


def test_un_tag_inconnu_ne_devient_pas_une_pelouse() -> None:
    """Là où la carte se tait, le sol reste indéterminé."""
    assert kind_of({"building": "yes"}) == "inconnu"
    assert kind_of({}) == "inconnu"


# --- conversion -------------------------------------------------------------


def test_une_emprise_devient_une_surface_projetee() -> None:
    cover = from_elements([_way(1, {"landuse": "grass"})], "EPSG:2950", "t")

    assert len(cover.surfaces) == 1
    surface = cover.surfaces[0]
    assert surface.kind == "pelouse"
    assert surface.area_m2() > 0
    # Le contour doit être fermé pour former un polygone.
    assert surface.ring[0] == surface.ring[-1]


def test_un_arbre_isole_est_un_noeud_avec_hauteur_supposee() -> None:
    element = {
        "type": "node",
        "id": 7,
        "tags": {"natural": "tree"},
        "lat": 45.5743,
        "lon": -73.4437,
    }
    cover = from_elements([element], "EPSG:2950", "t")

    assert len(cover.trees) == 1
    assert cover.trees[0].height_assumed is True


def test_une_emprise_lointaine_est_ecartee() -> None:
    """Un parc à trois cents mètres n'entre dans aucun plan d'établissement."""
    near = _way(1, {"landuse": "grass"})
    far = _way(2, {"landuse": "grass"}, lat=45.5900, lon=-73.4600)

    cover = from_elements(
        [near, far],
        "EPSG:2950",
        "t",
        centre=(309223.45, 5048238.20),
        radius_m=150.0,
    )

    assert len(cover.surfaces) == 1
    assert cover.provenance["skipped_out_of_range"] == 1


def test_une_emprise_incomplete_est_ignoree() -> None:
    broken = {"type": "way", "id": 3, "tags": {"landuse": "grass"}, "geometry": []}
    assert from_elements([broken], "EPSG:2950", "t").surfaces == []


# --- bilan ------------------------------------------------------------------


def test_les_aires_sont_cumulees_par_nature() -> None:
    cover = from_elements(
        [_way(1, {"landuse": "grass"}), _way(2, {"surface": "asphalt"})],
        "EPSG:2950",
        "t",
    )
    totals = cover.by_kind()

    assert set(totals) == {"pelouse", "mineral"}
    assert all(v > 0 for v in totals.values())


def test_le_bilan_de_proximite_ne_compte_que_le_visible() -> None:
    """Une superficie cumulée ne dit pas ce que la caméra verra."""
    close = GroundSurface("a", "pelouse", [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
    far = GroundSurface(
        "b", "pelouse", [(900, 900), (910, 900), (910, 910), (900, 910), (900, 900)]
    )
    cover = GroundCover(hotel_id="t", surfaces=[close, far])

    near = cover.near((5.0, 5.0), 100.0)

    assert [s.feature_id for s in near] == ["a"]
    # Le bilan global, lui, voit les deux : c'est bien pourquoi il ne suffit pas.
    assert cover.by_kind()["pelouse"] > close.area_m2()


def test_le_rapport_porte_ses_reserves() -> None:
    payload = GroundCover(hotel_id="t").as_dict()
    joined = " ".join(payload["caveats"])

    assert "ne dit pas ce que la caméra verra" in joined
    assert "hypothèse" in joined


def test_une_couverture_publiee_se_relit(tmp_path: Path) -> None:
    cover = GroundCover(
        hotel_id="t",
        surfaces=[GroundSurface("a", "pelouse", [(0, 0), (5, 0), (5, 5), (0, 5), (0, 0)])],
        trees=[LoneTree("n/1", (2.0, 2.0))],
    )
    payload = cover.as_dict()
    payload["surfaces"] = [
        {**s.as_dict(), "ring": [list(p) for p in s.ring]} for s in cover.surfaces
    ]
    path = tmp_path / "ground.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    reread = load(path)

    assert len(reread.surfaces) == 1
    assert reread.surfaces[0].kind == "pelouse"
    assert len(reread.trees) == 1
    assert reread.surfaces[0].area_m2() == pytest.approx(25.0)
