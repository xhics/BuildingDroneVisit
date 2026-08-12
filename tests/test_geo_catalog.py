"""Routage territorial des sources géospatiales (Lot 1B §9).

Une source ouverte n'est pas une source disponible ici. GéoMont 2023 offre du
20 cm sur la Montérégie mais exclut le territoire de la CMM — dont Boucherville
fait partie. La retenir aurait produit un téléchargement inutile puis une
absence inexpliquée.
"""

from __future__ import annotations

import pytest

from hotel_pipeline.geo import SOURCES, route, territories_for

#: Bâtiment confirmé du WelcomINNS.
BOUCHERVILLE = (45.574128, -73.443289)

#: Montérégie hors CMM.
GRANBY = (45.400, -72.733)


class TestTerritories:
    def test_boucherville_belongs_to_the_cmm(self):
        assert "QC-CMM" in territories_for(*BOUCHERVILLE)

    def test_boucherville_is_also_monteregie(self):
        """Les deux appartenances coexistent : l'exclusion doit primer."""
        territories = territories_for(*BOUCHERVILLE)
        assert {"QC-CMM", "QC-MONTEREGIE"} <= territories

    def test_granby_is_monteregie_without_the_cmm(self):
        territories = territories_for(*GRANBY)
        assert "QC-MONTEREGIE" in territories
        assert "QC-CMM" not in territories


class TestRouting:
    def test_geomont_is_rejected_on_cmm_territory(self):
        routing = route(*BOUCHERVILLE)
        assert "geomont-ortho-2023" in routing.rejected
        assert "QC-CMM" in routing.rejected["geomont-ortho-2023"]

    def test_geomont_is_available_outside_the_cmm(self):
        routing = route(*GRANBY)
        assert any(s.source_id == "geomont-ortho-2023" for s in routing.available)

    def test_lidar_is_available_across_quebec(self):
        for position in (BOUCHERVILLE, GRANBY):
            assert any(s.source_id == "lidar-quebec" for s in route(*position).available)

    def test_cmm_orthophoto_is_available_in_boucherville(self):
        routing = route(*BOUCHERVILLE)
        assert any(s.source_id == "cmm-ortho" for s in routing.available)


class TestWhatEachSourceCanEstablish:
    def test_lidar_establishes_terrain_and_roofline(self):
        routing = route(*BOUCHERVILLE)
        assert [s.source_id for s in routing.for_object("TERRAIN_MAIN")] == ["lidar-quebec"]
        assert [s.source_id for s in routing.for_object("ROOFLINE_MAIN")] == ["lidar-quebec"]

    def test_lidar_never_establishes_the_parcel(self):
        """Une limite juridique ne se dérive pas d'un nuage de points."""
        lidar = next(s for s in SOURCES if s.source_id == "lidar-quebec")
        assert "PROPERTY_PARCEL" not in lidar.establishes
        assert "PROPERTY_PARCEL" in lidar.cannot_establish

    def test_orthophoto_never_establishes_the_parcel(self):
        for source_id in ("cmm-ortho", "geomont-ortho-2023"):
            source = next(s for s in SOURCES if s.source_id == source_id)
            assert "PROPERTY_PARCEL" in source.cannot_establish

    def test_only_the_cadastre_establishes_the_parcel(self):
        establishing = [s.source_id for s in SOURCES if "PROPERTY_PARCEL" in s.establishes]
        assert establishing == ["cadastre-quebec"]

    def test_five_metre_orthophoto_cannot_cut_a_roof(self):
        cmm = next(s for s in SOURCES if s.source_id == "cmm-ortho")
        assert cmm.resolution_m == 5.0
        assert "ROOFLINE_MAIN" in cmm.cannot_establish

    def test_no_source_both_establishes_and_denies_the_same_object(self):
        for source in SOURCES:
            assert not set(source.establishes) & set(source.cannot_establish), source.source_id


class TestCatalogueHygiene:
    def test_every_source_declares_a_licence(self):
        assert all(s.licence for s in SOURCES)

    def test_source_ids_are_unique(self):
        ids = [s.source_id for s in SOURCES]
        assert len(ids) == len(set(ids))
