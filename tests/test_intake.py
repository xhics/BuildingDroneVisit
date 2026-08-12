"""Inventaire des médias, droits et versions d'entrée (§9, complément §4)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hotel_pipeline.intake import IntakeError, coverage, load_csv, promote, sha256_file
from hotel_pipeline.schemas import AssetManifest, EntranceVersion, ExteriorInterior, Rights

HEADER = "id,source,rights,category,exterior_or_interior,entrance_version\n"


def write_csv(tmp_path, body: str, header: str = HEADER):
    path = tmp_path / "inventaire.csv"
    path.write_text(header + body, encoding="utf-8")
    return path


class TestRightsAreMandatory:
    def test_missing_rights_column_value_rejected(self, tmp_path):
        path = write_csv(tmp_path, "img-1,site,,facade,exterior,after_renovation\n")
        with pytest.raises(IntakeError, match="'rights' est obligatoire"):
            load_csv(path)

    def test_invalid_rights_value_lists_allowed(self, tmp_path):
        path = write_csv(tmp_path, "img-1,site,peut-etre,facade,exterior,after_renovation\n")
        with pytest.raises(IntakeError, match="attendu l'un de"):
            load_csv(path)

    def test_unknown_column_rejected(self, tmp_path):
        path = write_csv(tmp_path, "img-1,site,owned\n", header="id,source,rights,inattendu\n")
        with pytest.raises(IntakeError, match="colonnes inconnues"):
            load_csv(path)


class TestImportDefaults:
    def test_nothing_is_production_eligible_on_import(self, tmp_path):
        """L'éligibilité production est une décision, jamais un défaut (§9)."""
        path = write_csv(
            tmp_path,
            "img-1,hotel,owned,facade,exterior,after_renovation\n"
            "img-2,tripadvisor,public_uncleared,facade,exterior,unknown\n",
        )
        assets = load_csv(path)
        assert len(assets) == 2
        assert all(not a.production_eligible for a in assets)
        assert all(not a.ai_eligible for a in assets)

    def test_enums_parsed(self, tmp_path):
        path = write_csv(tmp_path, "img-1,hotel,owned,facade,exterior,after_renovation\n")
        asset = load_csv(path)[0]
        assert asset.rights is Rights.OWNED
        assert asset.exterior_or_interior is ExteriorInterior.EXTERIOR
        assert asset.entrance_version is EntranceVersion.AFTER_RENOVATION

    def test_checksum_computed_when_file_present(self, tmp_path):
        images = tmp_path / "img"
        images.mkdir()
        (images / "a.jpg").write_bytes(b"contenu")
        path = write_csv(
            tmp_path,
            "img-1,hotel,owned,facade,exterior,after_renovation,a.jpg\n",
            header=HEADER.rstrip("\n") + ",file\n",
        )
        asset = load_csv(path, images_root=images)[0]
        assert asset.checksum == sha256_file(images / "a.jpg")

    def test_missing_file_is_an_error(self, tmp_path):
        path = write_csv(
            tmp_path,
            "img-1,hotel,owned,facade,exterior,after_renovation,absent.jpg\n",
            header=HEADER.rstrip("\n") + ",file\n",
        )
        with pytest.raises(IntakeError, match="fichier absent"):
            load_csv(path, images_root=tmp_path)


class TestPromotion:
    def test_uncleared_asset_cannot_be_promoted(self, tmp_path):
        """Le verrou du §9 tient au moment de la promotion, pas seulement à l'import."""
        path = write_csv(tmp_path, "img-1,tripadvisor,public_uncleared,facade,exterior,unknown\n")
        manifest = AssetManifest(hotel_id="h", assets=load_csv(path))
        with pytest.raises(ValidationError, match="production_eligible"):
            promote(manifest, ["img-1"])

    def test_owned_asset_promotes(self, tmp_path):
        path = write_csv(tmp_path, "img-1,hotel,owned,facade,exterior,after_renovation\n")
        manifest = AssetManifest(hotel_id="h", assets=load_csv(path))
        assert promote(manifest, ["img-1"]) == ["img-1"]
        assert manifest.production_eligible()[0].id == "img-1"

    def test_unknown_asset_rejected(self, tmp_path):
        path = write_csv(tmp_path, "img-1,hotel,owned,facade,exterior,after_renovation\n")
        manifest = AssetManifest(hotel_id="h", assets=load_csv(path))
        with pytest.raises(IntakeError, match="asset inconnu"):
            promote(manifest, ["img-absent"])


class TestCoverage:
    def test_counts_what_gates_the_pipeline(self, tmp_path):
        path = write_csv(
            tmp_path,
            "img-1,hotel,owned,facade,exterior,after_renovation\n"
            "img-2,hotel,owned,facade,exterior,unknown\n"
            "img-3,hotel,owned,interior,interior,unknown\n"
            "img-4,tripadvisor,public_uncleared,facade,exterior,after_renovation\n",
        )
        manifest = AssetManifest(hotel_id="h", assets=load_csv(path))
        promote(manifest, ["img-1", "img-2", "img-3"])

        counts = coverage(manifest)
        assert counts["total"] == 4
        assert counts["production_eligible"] == 3
        assert counts["exterior_eligible"] == 2
        assert counts["exterior_after_renovation"] == 1
        assert counts["entrance_version_unknown"] == 1
