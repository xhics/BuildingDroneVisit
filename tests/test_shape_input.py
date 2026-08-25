"""Le lot soumis à un modèle de forme doit être qualifié, pas seulement gros."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hotel_pipeline.conditioning.shape_input import MIN_SIDE, ShapeInput, ShapeImage, build


def _image(tmp_path: Path, name: str, side: int = 640) -> Path:
    from PIL import Image

    path = tmp_path / name
    Image.new("RGB", (side, side), (120, 120, 120)).save(path)
    return path


def _write(tmp_path: Path, screening: dict, manifest: dict) -> tuple[Path, Path]:
    a, b = tmp_path / "s.json", tmp_path / "m.json"
    a.write_text(json.dumps(screening), encoding="utf-8")
    b.write_text(json.dumps(manifest), encoding="utf-8")
    return a, b


def test_seules_les_images_identifiees_entrent_dans_le_lot(tmp_path: Path) -> None:
    """Une photographie du voisin fausserait la forme reconstruite."""
    good, bad = _image(tmp_path, "ok.jpg"), _image(tmp_path, "ko.jpg")
    screening = {
        "assets": [
            {"asset_id": "ok", "status": "match", "reference_score": 0.5, "path": str(good)},
            {"asset_id": "ko", "status": "mismatch", "reference_score": 0.9, "path": str(bad)},
        ]
    }
    manifest = {"assets": [{"id": "ok", "bearing_from_building_deg": 10.0}]}
    result = build(*_write(tmp_path, screening, manifest))

    assert [i.asset_id for i in result.images] == ["ok"]
    assert result.rejected["identite"] == 1


def test_une_image_trop_petite_ne_contraint_aucune_forme(tmp_path: Path) -> None:
    small = _image(tmp_path, "small.jpg", side=MIN_SIDE - 100)
    screening = {
        "assets": [
            {"asset_id": "x", "status": "match", "reference_score": 0.5, "path": str(small)}
        ]
    }
    result = build(*_write(tmp_path, screening, {"assets": []}))

    assert result.images == []
    assert result.rejected["resolution"] == 1


def test_deux_vues_du_meme_angle_sont_redondantes(tmp_path: Path) -> None:
    """Trois degrés d'écart n'apportent aucune parallaxe nouvelle."""
    first, second = _image(tmp_path, "a.jpg"), _image(tmp_path, "b.jpg")
    screening = {
        "assets": [
            {"asset_id": "a", "status": "match", "reference_score": 0.6, "path": str(first)},
            {"asset_id": "b", "status": "match", "reference_score": 0.5, "path": str(second)},
        ]
    }
    manifest = {
        "assets": [
            {"id": "a", "bearing_from_building_deg": 100.0},
            {"id": "b", "bearing_from_building_deg": 103.0},
        ]
    }
    result = build(*_write(tmp_path, screening, manifest))

    assert [i.asset_id for i in result.images] == ["a"]
    assert result.rejected["redondance"] == 1


def test_deux_vues_bien_ecartees_sont_toutes_deux_retenues(tmp_path: Path) -> None:
    first, second = _image(tmp_path, "a.jpg"), _image(tmp_path, "b.jpg")
    screening = {
        "assets": [
            {"asset_id": "a", "status": "match", "reference_score": 0.6, "path": str(first)},
            {"asset_id": "b", "status": "match", "reference_score": 0.5, "path": str(second)},
        ]
    }
    manifest = {
        "assets": [
            {"id": "a", "bearing_from_building_deg": 100.0},
            {"id": "b", "bearing_from_building_deg": 160.0},
        ]
    }
    result = build(*_write(tmp_path, screening, manifest))

    assert len(result.images) == 2
    assert result.angular_span() == pytest.approx(60.0)


def test_un_recadrage_herite_de_l_azimut_de_son_asset_source(tmp_path: Path) -> None:
    crop = _image(tmp_path, "crop.jpg")
    screening = {
        "assets": [
            {
                "asset_id": "SECT225_zj6pG6EOemMZ7d_54h_51f",
                "status": "match",
                "reference_score": 0.6,
                "path": str(crop),
            }
        ]
    }
    manifest = {
        "assets": [
            {"id": "street_view-zj6pG6EOemMZ7dPlDXJeMA", "bearing_from_building_deg": 233.9}
        ]
    }
    result = build(*_write(tmp_path, screening, manifest))

    assert len(result.placed) == 1
    assert result.images[0].bearing_deg == pytest.approx(233.9)


def test_le_lot_est_plafonne(tmp_path: Path) -> None:
    """Un modèle feed-forward sature : mieux vaut peu de vues bien réparties."""
    entries, assets = [], []
    for index in range(30):
        path = _image(tmp_path, f"{index:02d}.jpg")
        entries.append(
            {
                "asset_id": f"a{index}",
                "status": "match",
                "reference_score": 0.5,
                "path": str(path),
            }
        )
        assets.append({"id": f"a{index}", "bearing_from_building_deg": index * 12.0})
    result = build(*_write(tmp_path, {"assets": entries}, {"assets": assets}), max_images=5)

    assert len(result.images) == 5


def test_le_lot_porte_ses_reserves() -> None:
    payload = ShapeInput(hotel_id="h", images=[]).as_dict()
    joined = " ".join(payload["caveats"])
    assert "repère arbitraire" in joined
    assert "recalée" in joined


def test_l_etendue_angulaire_ignore_les_images_non_placees(tmp_path: Path) -> None:
    lot = ShapeInput(
        hotel_id="h",
        images=[
            ShapeImage("a", Path("a.jpg"), 0.0, 0.5, 640),
            ShapeImage("b", Path("b.jpg"), None, 0.5, 640),
            ShapeImage("c", Path("c.jpg"), 90.0, 0.5, 640),
        ],
    )
    assert len(lot.placed) == 2
    assert lot.angular_span() == pytest.approx(90.0)


def test_un_chemin_absolu_d_une_autre_machine_est_relocalise(tmp_path: Path) -> None:
    workspace = tmp_path / "work" / "hotel-test"
    (workspace / "02_images" / "recrops").mkdir(parents=True)
    image = _image(workspace / "02_images" / "recrops", "facade.jpg")
    screening = workspace / "09_confidence" / "identity_screening.json"
    manifest = workspace / "00_manifest" / "asset_manifest.json"
    screening.parent.mkdir(parents=True)
    manifest.parent.mkdir(parents=True)
    screening.write_text(
        json.dumps({"assets": [{
            "asset_id": "facade",
            "status": "match",
            "reference_score": 0.7,
            "path": "/Users/old/work/hotel-test/02_images/recrops/facade.jpg",
        }]}),
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps({"hotel_id": "hotel-test", "assets": [{
            "id": "facade", "bearing_from_building_deg": 42.0,
        }]}),
        encoding="utf-8",
    )

    result = build(screening, manifest)

    assert result.images[0].path == image
