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

#: Surface de plancher par chambre, circulation et services compris. C'est une
#: grandeur par **étage**, non par bâtiment : le nombre de chambres seul ne dit
#: rien de l'emprise au sol.
FLOOR_AREA_M2_PER_ROOM = (25.0, 70.0)

#: Nombre d'étages supposé quand le profil ne le déclare pas. La fourchette est
#: volontairement large : une tour de 600 chambres sur vingt niveaux occupe au
#: sol moins qu'un motel de 60 chambres de plain-pied, et une borne basse
#: calculée sans les étages l'aurait écartée.
ASSUMED_LEVELS = (1, 12)

#: Bornes absolues, appliquées faute de toute indication de taille.
FOOTPRINT_FALLBACK_M2 = (200.0, 40_000.0)


class RenovationEvent(BaseModel):
    """Travaux datés affectant l'apparence.

    Remplace l'enum `pre_2024`/`post_2024` : un établissement peut n'avoir
    jamais été rénové, ou l'avoir été trois fois.

    Trois dates distinctes, parce qu'elles ne disent pas la même chose. Le
    dossier municipal du WelcomINNS porte une date d'**approbation** — le
    23 septembre 2024 — qui ne prouve ni le début ni la fin des travaux. Une
    photographie postérieure à l'approbation peut parfaitement montrer
    l'ancienne entrée, ou un chantier.

    L'apparence n'est donc réputée actuelle qu'à partir de `completed_on`,
    et seulement si cette date est confirmée.
    """

    model_config = ConfigDict(extra="forbid")

    event_id: str
    scope: str  # entrance, facade, roof, grounds, signage...

    approved_on: date | None = None
    started_on: date | None = None
    completed_on: date | None = None

    #: `completed_on` est-il attesté, ou seulement estimé ?
    completion_confirmed: bool = False

    evidence: list[str] = Field(default_factory=list)
    note: str | None = None

    @model_validator(mode="after")
    def _at_least_one_date(self) -> "RenovationEvent":
        if not any((self.approved_on, self.started_on, self.completed_on)):
            raise ValueError(
                f"événement {self.event_id!r} sans aucune date : "
                "approved_on, started_on ou completed_on est requis"
            )
        if self.completion_confirmed and self.completed_on is None:
            raise ValueError(
                f"événement {self.event_id!r} : completion_confirmed sans completed_on — "
                "on ne confirme pas une date absente"
            )
        # Toutes les paires, y compris approbation/achèvement : une approbation
        # postérieure à l'achèvement était acceptée tant que started_on
        # manquait.
        for earlier, later, names in (
            (self.approved_on, self.started_on, "approved_on/started_on"),
            (self.started_on, self.completed_on, "started_on/completed_on"),
            (self.approved_on, self.completed_on, "approved_on/completed_on"),
        ):
            if earlier and later and earlier > later:
                raise ValueError(f"dates incohérentes : {names}")
        return self

    @property
    def reference_date(self) -> date:
        """Date la plus tardive connue, pour ordonner les événements."""
        return max(d for d in (self.approved_on, self.started_on, self.completed_on) if d)

    @property
    def establishes_current_appearance(self) -> bool:
        """Seule une fin de travaux confirmée fait référence d'apparence."""
        return self.completed_on is not None and self.completion_confirmed


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

    #: Nombre d'étages, s'il est connu. Sans lui, l'emprise se borne sur une
    #: plage d'étages supposée plutôt que sur un plain-pied implicite.
    expected_levels: int | None = Field(default=None, gt=0)
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

        Le nombre de chambres ne détermine pas l'emprise : il détermine une
        **surface de plancher**, que le nombre d'étages divise. Ignorer les
        niveaux revient à interdire les bâtiments hauts.

        Cette fourchette **n'élimine jamais** un candidat : elle sert à
        classer l'attention, et un bâtiment hors plage reste examinable.
        """
        if self.footprint_min_m2 is not None and self.footprint_max_m2 is not None:
            return self.footprint_min_m2, self.footprint_max_m2

        if self.room_count is None:
            return FOOTPRINT_FALLBACK_M2

        area_low, area_high = FLOOR_AREA_M2_PER_ROOM
        levels_low, levels_high = (
            (self.expected_levels, self.expected_levels)
            if self.expected_levels
            else ASSUMED_LEVELS
        )

        # Emprise minimale : beaucoup d'étages et des chambres compactes.
        # Emprise maximale : un seul niveau et des chambres généreuses.
        low = self.room_count * area_low / levels_high
        high = self.room_count * area_high / levels_low

        absolute_low, absolute_high = FOOTPRINT_FALLBACK_M2
        return max(low, absolute_low * 0.5), min(high, absolute_high)

    # -- temporalité -----------------------------------------------------

    def latest_event(self, scope: str | None = None) -> RenovationEvent | None:
        """Travaux les plus récents, éventuellement restreints à une portée."""
        candidates = [
            event
            for event in self.renovation_events
            if scope is None or event.scope == scope
        ]
        return max(candidates, key=lambda e: e.reference_date, default=None)

    def shows_current_appearance(self, taken_on: date, scope: str | None = None) -> bool | None:
        """Une prise de vue montre-t-elle l'état actuel ?

        Trois réponses distinctes, et l'indécision en fait partie :

        - `True`  : postérieure à une fin de travaux **confirmée** ;
        - `False` : antérieure au début des travaux ;
        - `None`  : entre les deux, ou fin de travaux non confirmée, ou aucun
          événement déclaré.

        La date d'approbation municipale ne suffit pas : approuver n'est ni
        commencer ni achever, et une photographie postérieure à l'approbation
        peut montrer l'ancienne entrée ou un chantier.
        """
        event = self.latest_event(scope)
        if event is None:
            return None

        # Seul un début de travaux attesté permet d'affirmer qu'une image lui
        # est antérieure. Une approbation ne prouve pas que le chantier a
        # commencé : il peut n'avoir jamais démarré.
        if event.started_on and taken_on < event.started_on:
            return False

        if event.establishes_current_appearance and taken_on >= event.completed_on:
            return True

        return None
