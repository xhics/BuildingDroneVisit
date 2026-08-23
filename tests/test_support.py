"""L'appui photographique est un second appui, distinct de la géométrie."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hotel_pipeline.conditioning.support import (
    FULL_SUPPORT_DEG,
    NO_SUPPORT_DEG,
    ReferenceView,
    SupportMap,
    from_screening,
)


def test_une_reference_appuie_pleinement_son_propre_angle() -> None:
    support = SupportMap([ReferenceView("a", 100.0)])
    score, nearest = support.support_at(100.0)
    assert score == pytest.approx(1.0)
    assert nearest == "a"


def test_l_appui_decroit_avec_l_ecart_angulaire() -> None:
    support = SupportMap([ReferenceView("a", 0.0)])
    near = support.support_at(FULL_SUPPORT_DEG - 5)[0]
    mid = support.support_at((FULL_SUPPORT_DEG + NO_SUPPORT_DEG) / 2)[0]
    far = support.support_at(NO_SUPPORT_DEG + 10)[0]
    assert near == pytest.approx(1.0)
    assert 0.0 < mid < 1.0
    assert far == 0.0


def test_l_appui_franchit_le_zero_du_cercle() -> None:
    """350° et 10° sont voisins : l'écart se mesure sur le cercle."""
    support = SupportMap([ReferenceView("a", 350.0)])
    assert support.support_at(10.0)[0] == pytest.approx(1.0)


def test_sans_reference_aucun_angle_n_est_appuye() -> None:
    support = SupportMap([])
    assert support.support_at(180.0) == (0.0, None)
    assert support.widest_gap() == 360.0


def test_le_plus_grand_trou_est_mesure() -> None:
    support = SupportMap(
        [ReferenceView("a", 0.0), ReferenceView("b", 90.0), ReferenceView("c", 120.0)]
    )
    assert support.widest_gap() == pytest.approx(240.0)


def test_une_reference_mediocre_appuie_moins() -> None:
    strong = SupportMap([ReferenceView("a", 0.0, quality=1.0)])
    weak = SupportMap([ReferenceView("a", 0.0, quality=0.3)])
    assert weak.support_at(0.0)[0] < strong.support_at(0.0)[0]


# --- lecture d'un dépistage -------------------------------------------------


def _write(tmp_path: Path, screening: dict, manifest: dict) -> tuple[Path, Path]:
    a = tmp_path / "screening.json"
    b = tmp_path / "manifest.json"
    a.write_text(json.dumps(screening), encoding="utf-8")
    b.write_text(json.dumps(manifest), encoding="utf-8")
    return a, b


def test_seules_les_images_identifiees_appuient(tmp_path: Path) -> None:
    """Une photographie du voisin n'appuie aucun angle de ce bâtiment-ci."""
    screening = {
        "assets": [
            {"asset_id": "ok", "status": "match", "reference_score": 0.8},
            {"asset_id": "ko", "status": "mismatch", "reference_score": 0.9},
        ]
    }
    manifest = {
        "assets": [
            {"id": "ok", "bearing_from_building_deg": 120.0},
            {"id": "ko", "bearing_from_building_deg": 300.0},
        ]
    }
    support = from_screening(*_write(tmp_path, screening, manifest))
    assert [r.asset_id for r in support.references] == ["ok"]


def test_une_reference_sans_azimut_n_est_pas_placee(tmp_path: Path) -> None:
    screening = {"assets": [{"asset_id": "x", "status": "match", "reference_score": 0.8}]}
    manifest = {"assets": [{"id": "x", "bearing_from_building_deg": None}]}
    support = from_screening(*_write(tmp_path, screening, manifest))
    assert len(support) == 0


def test_un_recrop_herite_de_l_azimut_de_son_asset_source(tmp_path: Path) -> None:
    """Cas réel : les meilleures références du pilote sont des recadrages.

    Sans cette résolution, aucune ne portait d'azimut et la carte d'appui
    restait vide alors que la donnée existait au manifeste.
    """
    screening = {
        "assets": [
            {
                "asset_id": "SECT225_zj6pG6EOemMZ7d_54h_51f",
                "status": "match",
                "reference_score": 0.6,
            }
        ]
    }
    manifest = {
        "assets": [
            {"id": "street_view-zj6pG6EOemMZ7dPlDXJeMA", "bearing_from_building_deg": 233.9}
        ]
    }
    support = from_screening(*_write(tmp_path, screening, manifest))
    assert len(support) == 1
    assert support.references[0].bearing_deg == pytest.approx(233.9)


def test_une_reference_trop_faible_est_ecartee(tmp_path: Path) -> None:
    screening = {"assets": [{"asset_id": "x", "status": "match", "reference_score": 0.05}]}
    manifest = {"assets": [{"id": "x", "bearing_from_building_deg": 10.0}]}
    support = from_screening(*_write(tmp_path, screening, manifest))
    assert len(support) == 0
