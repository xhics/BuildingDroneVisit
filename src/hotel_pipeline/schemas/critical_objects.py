"""Objets critiques et leurs preuves (plan directeur §4)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import ObjectState

#: Objets à identifier et valider (plan directeur §4).
REQUIRED_OBJECTS: tuple[str, ...] = (
    "PROPERTY_WELCOMINNS",
    "BUILDING_MAIN",
    "ENTRANCE_MAIN_CURRENT",
    "ROOFLINE_MAIN",
    "FACADE_PRIMARY",
    "FACADE_LEFT",
    "FACADE_RIGHT",
    "FACADE_REAR",
    "HOTEL_SIGN",
    "DRIVEWAY_MAIN",
    "PARKING_HOTEL",
    "RUE_AMPERE",
    "TERRAIN_MAIN",
)

#: Objets à distinguer ou exclure (plan directeur §4).
EXCLUDED_OBJECTS: tuple[str, ...] = (
    "INDOOR_POOL",
    "PARK_AND_RIDE_DE_MORTAGNE",
    "NEIGHBOURING_HOTELS",
    "UNRELATED_COMMERCIAL_BUILDINGS",
    "AUTOROUTE_20_CONTEXT",
)


class SpatialRelation(BaseModel):
    """Relation qualitative entre deux objets.

    Produite par le Reference Reasoner (§10) : hypothèse, jamais vérité
    métrique. Les données géographiques et la reconstruction confirment
    ou rejettent.
    """

    model_config = ConfigDict(extra="forbid")

    subject: str
    relation: str
    object: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)


class HumanCorrection(BaseModel):
    """Correction humaine et sa justification (plan directeur §4)."""

    model_config = ConfigDict(extra="forbid")

    field: str
    previous_value: str | None = None
    new_value: str
    rationale: str
    author: str


class CriticalObject(BaseModel):
    """Un invariant du projet, avec ses preuves et son état."""

    model_config = ConfigDict(extra="forbid")

    id: str
    category: str
    geometry_ref: str | None = None  # chemin GeoJSON ou identifiant de feature
    location_hint: str | None = None
    evidence_sources: list[str] = Field(default_factory=list)
    period: str | None = None
    state: ObjectState = ObjectState.UNRESOLVED
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    relations: list[SpatialRelation] = Field(default_factory=list)
    corrections: list[HumanCorrection] = Field(default_factory=list)

    @model_validator(mode="after")
    def _confirmed_needs_evidence(self) -> "CriticalObject":
        """Un objet `confirmed` sans source n'est pas confirmé, il est affirmé."""
        if self.state is ObjectState.CONFIRMED and not self.evidence_sources:
            raise ValueError(
                f"objet {self.id!r} en état 'confirmed' sans evidence_sources"
            )
        return self


class CriticalObjectRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hotel_id: str
    objects: list[CriticalObject] = Field(default_factory=list)

    def by_id(self, object_id: str) -> CriticalObject | None:
        return next((o for o in self.objects if o.id == object_id), None)

    def unresolved(self) -> list[CriticalObject]:
        return [o for o in self.objects if o.state is not ObjectState.CONFIRMED]

    def missing_required(self) -> list[str]:
        present = {o.id for o in self.objects}
        return [oid for oid in REQUIRED_OBJECTS if oid not in present]
