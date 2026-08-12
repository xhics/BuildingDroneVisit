"""Manifeste spatial et candidats de propriété (plan directeur §14, §19).

Le risque principal du pilote est l'identification : ne pas confondre l'hôtel
avec le stationnement incitatif voisin ni avec un bâtiment commercial mitoyen
(plan directeur §3, §24). Ce manifeste enregistre les preuves de cette
identification, y compris le choix humain qui la clôt.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import ObjectState


def _now() -> datetime:
    return datetime.now(timezone.utc)


class GeocodeResult(BaseModel):
    """Position issue d'un géocodeur, avec sa provenance."""

    model_config = ConfigDict(extra="forbid")

    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    provider: str
    raw_label: str | None = None
    postcode: str | None = None
    retrieved_at: datetime = Field(default_factory=_now)


class BuildingCandidate(BaseModel):
    """Une empreinte candidate à `BUILDING_MAIN`.

    Le §3 du plan directeur prévient que l'empreinte n'est pas nommée « hôtel »
    et qu'un parc-o-bus voisin prête à confusion. Plusieurs candidats sont donc
    la situation normale, pas une anomalie.
    """

    model_config = ConfigDict(extra="forbid")

    feature_id: str                     # ex. "way/29382"
    source: str                         # ex. "overpass"
    tags: dict[str, str] = Field(default_factory=dict)
    centroid_lat: float = Field(ge=-90, le=90)
    centroid_lon: float = Field(ge=-180, le=180)
    area_m2: float = Field(ge=0)
    distance_to_geocode_m: float = Field(ge=0)
    wkt: str                            # géométrie, en WGS84
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    score_reasons: list[str] = Field(default_factory=list)


class GeometricAssertion(BaseModel):
    """Contrôle géométrique exécuté sur les objets retenus.

    Encode les séparations exigées par le §3 : le stationnement de l'hôtel est
    contigu au bâtiment, et distinct du parc-o-bus De Mortagne.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool
    detail: str


class SpatialManifest(BaseModel):
    """Vérité spatiale d'un hôtel et ses preuves."""

    model_config = ConfigDict(extra="forbid")

    hotel_id: str
    address: str
    geocode: GeocodeResult | None = None
    search_radius_m: int = 500
    candidates: list[BuildingCandidate] = Field(default_factory=list)

    #: Identifiant du candidat retenu comme BUILDING_MAIN. Décision humaine.
    confirmed_building_id: str | None = None
    confirmed_by: str | None = None
    confirmed_at: datetime | None = None
    confirmation_rationale: str | None = None

    parking_feature_id: str | None = None
    park_and_ride_feature_id: str | None = None

    #: Azimut de la façade avant, en degrés géographiques. Sans lui, un azimut
    #: d'observation ne peut pas être traduit en `front`, `left`, `right` ou
    #: `rear` : on saurait d'où l'on regarde, sans savoir quoi l'on regarde.
    front_azimuth_deg: float | None = Field(default=None, ge=0, lt=360)

    #: Comment cet azimut a été obtenu — la valeur seule ne dit pas si elle
    #: est mesurée, déduite ou décidée.
    front_azimuth_method: str | None = None
    assertions: list[GeometricAssertion] = Field(default_factory=list)

    @property
    def state(self) -> ObjectState:
        if self.confirmed_building_id:
            return ObjectState.CONFIRMED
        if len(self.candidates) == 1:
            return ObjectState.INFERRED
        if len(self.candidates) > 1:
            return ObjectState.CONFLICTED
        return ObjectState.UNRESOLVED

    def candidate(self, feature_id: str) -> BuildingCandidate | None:
        return next((c for c in self.candidates if c.feature_id == feature_id), None)

    def ranked(self) -> list[BuildingCandidate]:
        return sorted(self.candidates, key=lambda c: c.score, reverse=True)

    def failed_assertions(self) -> list[GeometricAssertion]:
        return [a for a in self.assertions if not a.passed]

    @model_validator(mode="after")
    def _confirmed_must_exist(self) -> "SpatialManifest":
        """Confirmer un bâtiment absent de la liste des candidats est une erreur."""
        if self.confirmed_building_id and not self.candidate(self.confirmed_building_id):
            raise ValueError(
                f"bâtiment confirmé {self.confirmed_building_id!r} absent des candidats"
            )
        if self.confirmed_building_id and not self.confirmed_by:
            raise ValueError("un bâtiment confirmé doit porter l'auteur de la confirmation")
        return self
