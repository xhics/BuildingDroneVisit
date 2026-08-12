"""Manifeste d'assets (plan directeur §9).

Un champ obligatoire absent ou mal typé doit produire une erreur explicite.
Aucun asset ne doit être routé silencieusement avec des métadonnées invalides.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import (
    AssetCategory,
    EntranceVersion,
    ExteriorInterior,
    PropertyMatchStatus,
    Rights,
)

#: Droits autorisant l'usage d'un asset en production (reconstruction).
PRODUCTION_RIGHTS = frozenset({Rights.OWNED, Rights.LICENSED, Rights.OPEN_DATA})


class GpsPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class Asset(BaseModel):
    """Un média source et sa qualification."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    id: str
    source: str
    source_url_or_id: str
    rights: Rights
    ai_eligible: bool
    confidence: float = Field(ge=0.0, le=1.0)
    category: AssetCategory
    capture_year: int | None = Field(default=None, ge=1900, le=2100)
    season: str | None = None
    device: str | None = None
    gps_if_available: GpsPoint | None = None
    checksum: str
    derived_from: list[str] = Field(default_factory=list)

    # Champs additionnels utiles au pilote (plan directeur §9).
    exterior_or_interior: ExteriorInterior = ExteriorInterior.UNKNOWN
    entrance_version: EntranceVersion = EntranceVersion.UNKNOWN
    property_match_status: PropertyMatchStatus = PropertyMatchStatus.UNCERTAIN
    duplicate_group: str | None = None
    production_eligible: bool = False

    @model_validator(mode="after")
    def _rights_gate_production(self) -> "Asset":
        """Un asset ne peut être éligible production que si ses droits le permettent.

        Verrou structurel du §9 : une image publique reste `reference_only` tant
        que ses droits ne permettent pas son usage en reconstruction.
        """
        if self.production_eligible and self.rights not in PRODUCTION_RIGHTS:
            raise ValueError(
                f"asset {self.id!r} marqué production_eligible avec rights={self.rights.value!r} ; "
                f"droits acceptés : {sorted(r.value for r in PRODUCTION_RIGHTS)}"
            )
        if self.ai_eligible and self.rights not in PRODUCTION_RIGHTS:
            raise ValueError(
                f"asset {self.id!r} marqué ai_eligible avec rights={self.rights.value!r}"
            )
        return self


class AssetManifest(BaseModel):
    """Registre des assets d'un hôtel. Source de vérité unique."""

    model_config = ConfigDict(extra="forbid")

    hotel_id: str
    assets: list[Asset] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids(self) -> "AssetManifest":
        seen: set[str] = set()
        for asset in self.assets:
            if asset.id in seen:
                raise ValueError(f"identifiant d'asset dupliqué : {asset.id!r}")
            seen.add(asset.id)
        return self

    def production_eligible(self) -> list[Asset]:
        return [a for a in self.assets if a.production_eligible]

    def reference_only(self) -> list[Asset]:
        return [a for a in self.assets if not a.production_eligible]
