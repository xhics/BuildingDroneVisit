"""Les schémas doivent refuser ce que le plan directeur interdit."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hotel_pipeline.schemas import (
    Asset,
    AssetCategory,
    AssetManifest,
    CriticalObject,
    CriticalObjectRegistry,
    ObjectState,
    Rights,
)


def make_asset(**overrides) -> Asset:
    fields = dict(
        id="img-001",
        source="site officiel",
        source_url_or_id="https://example.invalid/1.jpg",
        rights=Rights.OWNED,
        ai_eligible=False,
        confidence=0.9,
        category=AssetCategory.FACADE,
        checksum="a" * 64,
    )
    fields.update(overrides)
    return Asset(**fields)


class TestRightsGate:
    """Un asset aux droits non établis ne peut pas entrer en production (§9)."""

    def test_public_uncleared_cannot_be_production_eligible(self):
        with pytest.raises(ValidationError, match="production_eligible"):
            make_asset(rights=Rights.PUBLIC_UNCLEARED, production_eligible=True)

    def test_unknown_rights_cannot_be_ai_eligible(self):
        with pytest.raises(ValidationError, match="ai_eligible"):
            make_asset(rights=Rights.UNKNOWN, ai_eligible=True)

    def test_owned_asset_may_be_production_eligible(self):
        asset = make_asset(rights=Rights.OWNED, production_eligible=True)
        assert asset.production_eligible

    def test_uncleared_asset_stays_reference_only(self):
        asset = make_asset(rights=Rights.PUBLIC_UNCLEARED)
        manifest = AssetManifest(hotel_id="h", assets=[asset])
        assert manifest.production_eligible() == []
        assert manifest.reference_only() == [asset]


class TestAssetManifest:
    def test_duplicate_ids_rejected(self):
        with pytest.raises(ValidationError, match="dupliqué"):
            AssetManifest(hotel_id="h", assets=[make_asset(), make_asset()])

    def test_unknown_field_rejected(self):
        """Une métadonnée mal nommée ne doit pas passer en silence (§9)."""
        with pytest.raises(ValidationError):
            make_asset(unexpected_field="x")

    def test_confidence_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            make_asset(confidence=1.5)


class TestCriticalObjects:
    def test_confirmed_without_evidence_rejected(self):
        """'confirmed' sans source est une affirmation, pas une confirmation (§4)."""
        with pytest.raises(ValidationError, match="evidence_sources"):
            CriticalObject(id="BUILDING_MAIN", category="building", state=ObjectState.CONFIRMED)

    def test_confirmed_with_evidence_accepted(self):
        obj = CriticalObject(
            id="BUILDING_MAIN",
            category="building",
            state=ObjectState.CONFIRMED,
            evidence_sources=["osm_way_29382", "img-014"],
            confidence=0.91,
        )
        assert obj.state is ObjectState.CONFIRMED

    def test_missing_required_objects_reported(self):
        registry = CriticalObjectRegistry(hotel_id="h")
        missing = registry.missing_required()
        assert "BUILDING_MAIN" in missing
        assert "ENTRANCE_MAIN_CURRENT" in missing
        assert "PARKING_HOTEL" in missing

    def test_unresolved_objects_listed(self):
        registry = CriticalObjectRegistry(
            hotel_id="h",
            objects=[
                CriticalObject(id="A", category="c", state=ObjectState.UNRESOLVED),
                CriticalObject(
                    id="B", category="c", state=ObjectState.CONFIRMED, evidence_sources=["s"]
                ),
            ],
        )
        assert [o.id for o in registry.unresolved()] == ["A"]
