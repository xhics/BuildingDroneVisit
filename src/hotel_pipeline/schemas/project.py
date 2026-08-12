"""Manifeste de projet et état d'exécution."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import Phase1Status


def _now() -> datetime:
    return datetime.now(timezone.utc)


class BlockedState(BaseModel):
    """Verrou humain (complément d'implémentation §4).

    Une étape qui requiert une décision humaine écrit cet état, libère la VM et
    se reprend à la session suivante. Aucune attente interactive sur une machine
    facturée.
    """

    model_config = ConfigDict(extra="forbid")

    step: str
    awaiting: str
    expected_form: str
    created_at: datetime = Field(default_factory=_now)


class StepRecord(BaseModel):
    """Trace d'exécution d'une étape (plan directeur §18)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    completed_at: datetime = Field(default_factory=_now)
    parameters: dict[str, str] = Field(default_factory=dict)
    tool_versions: dict[str, str] = Field(default_factory=dict)


class ProjectManifest(BaseModel):
    """État d'un hôtel dans le pipeline."""

    model_config = ConfigDict(extra="forbid")

    hotel_id: str
    address: str

    #: Position fournie par l'humain, prioritaire sur tout géocodeur.
    #: Le géocodage de l'adresse officielle du WelcomINNS a rendu un code postal
    #: divergent (J4B 5M7 contre J4B 7M6) : quand la position exacte est connue,
    #: la donner supprime cette source d'erreur plutôt que de la corriger après.
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)

    created_at: datetime = Field(default_factory=_now)
    status: Phase1Status | None = None
    steps: list[StepRecord] = Field(default_factory=list)
    blocked: BlockedState | None = None

    def completed_steps(self) -> set[str]:
        return {s.name for s in self.steps}

    @model_validator(mode="after")
    def _coordinates_come_in_pairs(self) -> "ProjectManifest":
        if (self.lat is None) != (self.lon is None):
            raise ValueError("lat et lon doivent être fournis ensemble")
        return self

    def record(self, step: StepRecord) -> None:
        """Enregistre une étape, en remplaçant une exécution antérieure."""
        self.steps = [s for s in self.steps if s.name != step.name]
        self.steps.append(step)
