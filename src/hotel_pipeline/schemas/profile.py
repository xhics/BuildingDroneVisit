"""Profil d'établissement — ce qui distingue un hôtel d'un autre.

Tout ce que le code « savait » du WelcomINNS vit désormais ici : ses noms, ses
concurrents, ses travaux, sa taille, sa langue. Le code ne connaît plus aucun
établissement ; il lit un profil.

Trois spécificités ont motivé cette séparation :

- des identifiants nominatifs — `PROPERTY_WELCOMINNS`, `RUE_AMPERE` — dans le
  registre d'objets critiques ;
- une date de rénovation promue au rang de type, `PRE_2024` / `POST_2024` ;
- une plage d'emprise calibrée sur un hôtel de 116 chambres.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Emprise au sol plausible par chambre, en mètres carrés. Un hôtel bas et
#: étalé occupe davantage au sol qu'une tour à nombre de chambres égal : la
#: fourchette est donc large, et sert à écarter l'absurde, non à trancher.
FOOTPRINT_M2_PER_ROOM = (8.0, 40.0)

#: Bornes absolues, appliquées faute de toute indication de taille.
FOOTPRINT_FALLBACK_M2 = (300.0, 30_000.0)


class RenovationEvent(BaseModel):
    """Travaux datés affectant l'apparence.

    Remplace l'enum `pre_2024`/`post_2024` : un établissement peut n'avoir
    jamais été rénové, ou l'avoir été trois fois.
    """

    model_config = ConfigDict(extra="forbid")

    event_id: str
    occurred_on: date
    scope: str  # entrance, facade, roof, grounds, signage...
    evidence: list[str] = Field(default_factory=list)
    note: str | None = None


class PropertyProfile(BaseModel):
    """Identité et particularités d'un établissement."""

    model_config = ConfigDict(extra="forbid")

    property_id: str
    address: str
    official_name: str
    aliases: list[str] = Field(default_factory=list)

    #: Établissements voisins à ne pas confondre. Noms **complets** : exclure
    #: le jeton « Mortagne » avait disqualifié une page du WelcomINNS lui-même,
    #: dont les salles portent des noms de rues locales.
    competitor_names: list[str] = Field(default_factory=list)

    ocr_languages: list[str] = Field(default_factory=lambda: ["fr", "en"])
    renovation_events: list[RenovationEvent] = Field(default_factory=list)

    #: Indices de taille, facultatifs. Le nombre de chambres suffit à borner
    #: l'emprise sans coder une plage en dur.
    room_count: int | None = Field(default=None, gt=0)
    footprint_min_m2: float | None = Field(default=None, gt=0)
    footprint_max_m2: float | None = Field(default=None, gt=0)

    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    website_url: str | None = None
    place_query: str | None = None

    @model_validator(mode="after")
    def _coherent_bounds(self) -> "PropertyProfile":
        if (
            self.footprint_min_m2 is not None
            and self.footprint_max_m2 is not None
            and self.footprint_min_m2 >= self.footprint_max_m2
        ):
            raise ValueError("footprint_min_m2 doit être inférieur à footprint_max_m2")
        if (self.lat is None) != (self.lon is None):
            raise ValueError("lat et lon doivent être fournis ensemble")
        return self

    # -- identité --------------------------------------------------------

    def identity_terms(self) -> list[str]:
        """Termes dont la lecture confirme l'appartenance."""
        return [self.official_name, *self.aliases]

    def excluded_terms(self) -> list[str]:
        """Termes dont la lecture disqualifie une image."""
        return list(self.competitor_names)

    # -- taille ----------------------------------------------------------

    def footprint_range_m2(self) -> tuple[float, float]:
        """Emprise plausible, dérivée du profil et non d'une constante.

        Priorité aux bornes explicites, puis au nombre de chambres, puis à des
        bornes larges qui n'écartent que l'absurde.
        """
        if self.footprint_min_m2 is not None and self.footprint_max_m2 is not None:
            return self.footprint_min_m2, self.footprint_max_m2

        if self.room_count is not None:
            low, high = FOOTPRINT_M2_PER_ROOM
            return self.room_count * low, self.room_count * high

        return FOOTPRINT_FALLBACK_M2

    # -- temporalité -----------------------------------------------------

    def latest_event(self, scope: str | None = None) -> RenovationEvent | None:
        """Travaux les plus récents, éventuellement restreints à une portée."""
        candidates = [
            event
            for event in self.renovation_events
            if scope is None or event.scope == scope
        ]
        return max(candidates, key=lambda e: e.occurred_on, default=None)

    def is_after_latest_event(self, taken_on: date, scope: str | None = None) -> bool | None:
        """Une prise de vue est-elle postérieure aux derniers travaux ?

        Retourne `None` quand aucun événement n'est déclaré : sans travaux
        connus, la question n'a pas de sens, et supposer « à jour » serait une
        invention.
        """
        event = self.latest_event(scope)
        if event is None:
            return None
        return taken_on >= event.occurred_on
