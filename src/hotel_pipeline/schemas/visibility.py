"""Visibilité multi-rayons (Lot 1B V2, étape 3).

Trois mesures y sont tenues séparées, parce qu'elles répondent à trois
questions différentes :

```text
la ligne de vue passe-t-elle ?     → VisibilityAssessment  (géométrie)
la cible entre-t-elle dans le cadre ? → FramingAssessment  (caméra)
que vaut cette voie ?               → CorridorVisibilityAssessment
```

Un corridor n'est pas une caméra : sans cap, champ, inclinaison ni dimensions,
`outside_frame` et la taille en pixels n'ont pas de sens. Les calculer quand
même reviendrait à inventer l'appareil qui n'y est pas encore.

Deux règles gouvernent le reste.

**Les rayons ne sont pas des pixels égaux.** Un échantillonnage adaptatif place
plus de rayons là où la silhouette varie ; les compter à l'unité donnerait donc
du poids à la finesse du maillage plutôt qu'à la réalité. Chaque cellule porte
sa largeur angulaire, et les fractions en sont pondérées.

**Une hauteur inconnue n'est pas une occultation.** Tant qu'une seule donnée
verticale manque, le rayon reste un risque. D'où deux bornes plutôt qu'une
fraction unique : ce qui est prouvé libre, et ce qui pourrait l'être.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RayPartition(StrEnum):
    """Catégorie d'une cellule angulaire. Exclusives et exhaustives.

    Purement géométriques : le cadrage n'y figure pas. Deux recadrages du même
    panorama voient la même scène — faire varier la visibilité avec le champ
    de vision reviendrait à dire que déplacer l'objectif déplace les murs.
    """

    #: Aucun obstacle en plan avant la cible.
    CLEAR_2D = "clear_2d"

    #: Un obstacle coupe la ligne de vue en plan, mais une donnée verticale
    #: manque : on ne peut pas dire s'il masque réellement.
    RISK_UNKNOWN_HEIGHT = "risk_unknown_height"

    #: Obstacle prouvé masquant, hauteurs et terrains connus des deux côtés.
    BLOCKED_2_5D = "blocked_2_5d"


class LineOfSightStatus(StrEnum):
    """Résumé opposable d'une évaluation."""

    CLEAR = "clear"
    PARTIAL = "partial"
    AT_RISK = "at_risk"
    BLOCKED = "blocked"
    #: Position, cible ou géométrie manquantes.
    INSUFFICIENT_DATA = "insufficient_data"


class VerticalVisibilityStatus(StrEnum):
    """Ce que l'on sait de la dimension verticale, à un rayon donné."""

    #: Terrains et hauteurs connus des deux côtés : le verdict est prouvable.
    FULLY_KNOWN = "fully_known"
    #: Au moins une donnée manque ; laquelle est dite dans `missing`.
    INCOMPLETE = "incomplete"
    #: Aucune donnée verticale du tout.
    UNKNOWN = "unknown"


class HitVerdict(StrEnum):
    """Ce qu'une intersection prouve, à elle seule."""

    #: Hauteurs et terrains connus : l'obstacle masque la cible sur ce rayon.
    BLOCKS = "blocks"
    #: Hauteurs et terrains connus : il ne la masque pas.
    PASSES_UNDER = "passes_under"
    #: Une donnée verticale manque : on ne peut rien conclure.
    UNDECIDABLE = "undecidable"


class ObstacleHit(BaseModel):
    """Une intersection, et ce qu'elle établit.

    Conserver les obstacles croisés sans leur verdict individuel faisait
    inscrire comme bloquants tous ceux d'un rayon dès qu'un seul l'était.
    """

    model_config = ConfigDict(extra="forbid")

    obstacle_ref: str
    distance_m: float = Field(ge=0)
    vertical_status: VerticalVisibilityStatus
    verdict: HitVerdict
    missing_vertical: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _verdict_requires_knowledge(self) -> "ObstacleHit":
        decided = self.verdict in (HitVerdict.BLOCKS, HitVerdict.PASSES_UNDER)
        if decided and self.vertical_status is not VerticalVisibilityStatus.FULLY_KNOWN:
            raise ValueError(
                f"{self.obstacle_ref} : verdict {self.verdict.value!r} sans données "
                "verticales complètes"
            )
        if self.verdict is HitVerdict.UNDECIDABLE and not self.missing_vertical:
            raise ValueError(
                f"{self.obstacle_ref} : indécidable sans dire ce qui manque"
            )
        return self


class RayAssessment(BaseModel):
    """Une cellule angulaire, sa largeur et son verdict.

    La largeur est la seule pondération admise : deux cellules de 0,1° et de
    2° ne disent pas la même chose de la façade.
    """

    model_config = ConfigDict(extra="forbid")

    bearing_deg: float = Field(ge=0, lt=360)
    angular_width_deg: float = Field(gt=0)
    partition: RayPartition

    #: Distance à la première intersection avec la cible, s'il y en a une.
    target_distance_m: float | None = Field(default=None, ge=0)

    #: Chaque intersection rencontrée **avant** la cible, avec son verdict
    #: propre. Un obstacle croisé n'est pas un obstacle responsable : les
    #: confondre inscrivait comme bloquants des voisins dont on ne savait rien.
    hits: list[ObstacleHit] = Field(default_factory=list)

    vertical_status: VerticalVisibilityStatus = VerticalVisibilityStatus.UNKNOWN
    missing_vertical: list[str] = Field(default_factory=list)

    @property
    def obstacles(self) -> list[str]:
        """Tous les obstacles croisés, du plus proche au plus lointain."""
        return [hit.obstacle_ref for hit in self.hits]

    @property
    def blocking(self) -> list[str]:
        return [hit.obstacle_ref for hit in self.hits if hit.verdict is HitVerdict.BLOCKS]

    @property
    def at_risk(self) -> list[str]:
        return [
            hit.obstacle_ref for hit in self.hits if hit.verdict is HitVerdict.UNDECIDABLE
        ]

    @model_validator(mode="after")
    def _verdict_matches_knowledge(self) -> "RayAssessment":
        if self.partition is RayPartition.BLOCKED_2_5D:
            if self.vertical_status is not VerticalVisibilityStatus.FULLY_KNOWN:
                raise ValueError(
                    "blocage prouvé sans données verticales complètes — une "
                    "hauteur inconnue reste un risque"
                )
            if not self.blocking:
                raise ValueError("blocage prouvé sans obstacle responsable")
        if self.partition is RayPartition.RISK_UNKNOWN_HEIGHT and not self.at_risk:
            raise ValueError(
                "risque annoncé sans obstacle dont une donnée manque"
            )
        return self


class VisibilityAssessment(BaseModel):
    """Ce que la géométrie dit d'une ligne de vue, cadrage exclu."""

    model_config = ConfigDict(extra="forbid")

    assessment_id: str
    subject_ref: str
    target_ref: str

    #: Position **projetée** de l'observateur, dans le référentiel des calculs.
    #: Les degrés n'ont pas de sens ici : tout se mesure en mètres.
    camera_x: float | None = None
    camera_y: float | None = None
    crs: str = "EPSG:2950"

    #: Intervalle angulaire réel occupé par la silhouette, et son passage
    #: éventuel par 0°.
    span_start_deg: float | None = Field(default=None, ge=0, lt=360)
    span_end_deg: float | None = Field(default=None, ge=0, lt=360)
    angular_span_deg: float | None = Field(default=None, ge=0, le=360)
    crosses_north: bool = False

    distance_m: float | None = Field(default=None, ge=0)

    #: Les trois fractions géométriques, pondérées par la largeur angulaire.
    #: Le cadrage n'en fait pas partie : il vit dans `FramingAssessment`.
    proven_clear_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_unknown_height_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    proven_blocked_fraction: float = Field(default=0.0, ge=0.0, le=1.0)

    #: Plus grand intervalle continu sans obstacle prouvé.
    largest_clear_span_deg: float = Field(default=0.0, ge=0)

    status: LineOfSightStatus = LineOfSightStatus.INSUFFICIENT_DATA
    obstacles_at_risk: list[str] = Field(default_factory=list)
    obstacles_blocking: list[str] = Field(default_factory=list)
    missing_vertical: list[str] = Field(default_factory=list)

    rays: list[RayAssessment] = Field(default_factory=list)

    @property
    def visible_lower_bound(self) -> float:
        """Ce qui est **prouvé** dégagé."""
        return self.proven_clear_fraction

    @property
    def visible_upper_bound(self) -> float:
        """Ce qui pourrait l'être si les hauteurs manquantes le permettaient.

        Publier une fraction unique laisserait choisir entre optimisme et
        pessimisme sans le dire.
        """
        return round(
            self.proven_clear_fraction + self.risk_unknown_height_fraction, 6
        )

    @model_validator(mode="after")
    def _fractions_partition_the_span(self) -> "VisibilityAssessment":
        total = (
            self.proven_clear_fraction
            + self.risk_unknown_height_fraction
            + self.proven_blocked_fraction
        )
        if self.rays and abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"les trois fractions totalisent {total:.6f} au lieu de 1 — "
                "une cellule est comptée deux fois ou pas du tout"
            )
        return self


class FramingAssessment(BaseModel):
    """Ce qu'une caméra précise verrait, cadrage compris.

    Exige un appareil : cap, champ, inclinaison et dimensions. Sans eux, la
    taille projetée reste inconnue — aucun champ générique n'est supposé.
    """

    model_config = ConfigDict(extra="forbid")

    assessment_id: str
    subject_ref: str

    heading_deg: float | None = Field(default=None, ge=0, lt=360)
    fov_deg: float | None = Field(default=None, gt=0, le=180)
    vertical_fov_deg: float | None = Field(default=None, gt=0, le=180)
    pitch_deg: float | None = Field(default=None, ge=-90, le=90)
    width_px: int | None = Field(default=None, gt=0)
    height_px: int | None = Field(default=None, gt=0)

    #: D'où viennent ces paramètres : demande explicite, intrinsèques du
    #: fournisseur, ou lecture d'une ancienne URL.
    parameters_source: str | None = None
    projection_model: str | None = None

    #: Portion de la silhouette réellement dans le cadre.
    target_in_frame_fraction: float | None = Field(default=None, ge=0.0, le=1.0)

    #: Largeur projetée sans écrêtage, puis après intersection avec le cadre.
    unclipped_width_fraction: float | None = Field(default=None, ge=0.0)
    clipped_width_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    expected_width_px: int | None = Field(default=None, ge=0)
    expected_height_px: int | None = Field(default=None, ge=0)

    #: La largeur et la hauteur ne s'établissent pas avec les mêmes données :
    #: la première demande cap, champ et dimensions ; la seconde exige en plus
    #: une inclinaison et une hauteur d'œil. Un seul drapeau publiait donc des
    #: pixels verticaux qu'aucune donnée ne soutenait.
    horizontal_computable: bool = False
    vertical_computable: bool = False
    horizontal_reason: str | None = None
    vertical_reason: str | None = None

    @model_validator(mode="after")
    def _computable_means_measured(self) -> "FramingAssessment":
        if self.horizontal_computable:
            missing = [
                name
                for name in ("heading_deg", "fov_deg", "width_px", "height_px")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(
                    f"largeur déclarée calculable sans {missing} — aucun champ "
                    "générique n'est supposé"
                )
            if not self.parameters_source:
                raise ValueError("cadrage calculable sans provenance de ses paramètres")
        elif not (self.horizontal_reason or "").strip():
            raise ValueError("largeur non calculable sans motif")

        if self.vertical_computable:
            if self.pitch_deg is None:
                raise ValueError(
                    "hauteur déclarée calculable sans inclinaison — une visée "
                    "supposée horizontale est une convention, pas une mesure"
                )
            if self.expected_height_px is None:
                raise ValueError("hauteur calculable sans hauteur mesurée")
        elif not (self.vertical_reason or "").strip():
            raise ValueError("hauteur non calculable sans motif")
        return self


class UsefulnessVerdict(StrEnum):
    USEFUL = "true"
    NOT_USEFUL = "false"
    UNKNOWN = "unknown"


class CorridorVisibilityAssessment(BaseModel):
    """Ce qu'une voie promet, sans supposer d'appareil.

    On y mesure une ligne de vue potentielle, jamais un cadrage : l'utilité
    géométrique ne vaut pas autorisation, et l'accessibilité reste ce que le
    manifeste géométrique en dit.
    """

    model_config = ConfigDict(extra="forbid")

    assessment_id: str
    corridor_id: str
    feature_id: str

    samples: int = Field(default=0, ge=0)
    sample_step_m: float | None = Field(default=None, gt=0)

    #: Segments continus d'échantillons. Vingt-cinq échantillons d'une même
    #: route ne font pas vingt-cinq points de vue — et les deux comptes sont
    #: séparés : un segment à risque n'est pas encore un segment utile.
    proven_clear_segments: int = Field(default=0, ge=0)
    potential_segments: int = Field(default=0, ge=0)

    best_sample_ids: list[str] = Field(default_factory=list)

    #: Secteurs du bâtiment réellement observables depuis cette voie. Sans eux,
    #: le Router sait qu'on voit « quelque chose », jamais quelle façade.
    observable_sectors: list[str] = Field(default_factory=list)

    #: Le meilleur échantillon, et **ses** mesures. Prendre le maximum de
    #: chaque grandeur séparément décrirait un emplacement qui n'existe pas.
    best_sample_id: str | None = None
    best_clear_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    best_risk_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    best_angular_span_deg: float | None = Field(default=None, ge=0, le=360)
    best_distance_m: float | None = Field(default=None, ge=0)

    #: Ouverture maximale rencontrée, quel qu'en soit l'échantillon : très
    #: grande, elle signale surtout une position très proche.
    max_angular_span_deg: float | None = Field(default=None, ge=0, le=360)

    #: Besoins que cette voie peut servir, et pour lesquels elle a été jugée.
    serves_demands: list[str] = Field(default_factory=list)
    obstacles_at_risk: list[str] = Field(default_factory=list)

    #: Géométriquement utile — ce qui ne rend rien autorisé. Ternaire : avec
    #: des obstacles tous de hauteur inconnue, conclure `false` serait aussi
    #: injustifié que conclure `true`.
    geometrically_useful: UsefulnessVerdict = UsefulnessVerdict.UNKNOWN
    access_status: str | None = None
    rationale: str = ""


class VisibilityRun(BaseModel):
    """Une exécution du moteur, et tout ce dont elle dépend."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    hotel_id: str
    computed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    engine_version: str = Field(min_length=1)
    method: str = Field(min_length=1)

    #: Réglages effectifs du moteur. Obligatoires et complets : un rapport dont
    #: on ignore le pas angulaire ne se rejoue pas.
    parameters: dict[str, str] = Field(min_length=1)

    #: Empreintes **obligatoires** : une mesure qu'on ne peut rattacher ni à
    #: une politique, ni à un corpus, ni à une géométrie ne se rejoue pas.
    capture_geometry_digest: str = Field(min_length=1)
    policy_digest: str = Field(min_length=1)
    site_manifest_digest: str = Field(min_length=1)
    assets_digest: str = Field(min_length=1)
    target_digest: str = Field(min_length=1)
    obstacles_digest: str = Field(min_length=1)
    road_geometry_digest: str = Field(min_length=1)

    elevation_artifacts: list[str] = Field(default_factory=list)

    assessments: list[VisibilityAssessment] = Field(default_factory=list)
    framings: list[FramingAssessment] = Field(default_factory=list)
    corridors: list[CorridorVisibilityAssessment] = Field(default_factory=list)

    @model_validator(mode="after")
    def _identifiers_are_unique(self) -> "VisibilityRun":
        ids = [a.assessment_id for a in self.assessments]
        if len(set(ids)) != len(ids):
            raise ValueError("identifiants d'évaluation dupliqués")

        framing_ids = [f.assessment_id for f in self.framings]
        if len(set(framing_ids)) != len(framing_ids):
            raise ValueError("cadrages dupliqués : deux mesures pour une même vue")
        unknown = sorted(set(framing_ids) - set(ids))
        if unknown:
            raise ValueError(f"cadrages sans évaluation correspondante : {unknown}")

        corridor_ids = [c.assessment_id for c in self.corridors]
        if len(set(corridor_ids)) != len(corridor_ids):
            raise ValueError("évaluations de corridor dupliquées")

        subjects = [a.subject_ref for a in self.assessments]
        if len(set(subjects)) != len(subjects):
            raise ValueError(
                "deux évaluations pour un même sujet — la correspondance avec "
                "l'asset ne serait plus univoque"
            )

        # Une mesure verticale employée sans dire d'où elle vient ne se
        # conteste pas : la provenance est exigée dès qu'un rayon tranche.
        decided = any(
            ray.vertical_status is VerticalVisibilityStatus.FULLY_KNOWN
            for assessment in self.assessments
            for ray in assessment.rays
        )
        if decided and not self.elevation_artifacts:
            raise ValueError(
                "des verdicts verticaux ont été rendus sans citer la moindre "
                "source d'élévation"
            )
        return self
