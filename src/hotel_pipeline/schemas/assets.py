"""Manifeste d'assets (plan directeur §9).

Un champ obligatoire absent ou mal typé doit produire une erreur explicite.
Aucun asset ne doit être routé silencieusement avec des métadonnées invalides.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from datetime import datetime, timezone

from .acquisition import AcquisitionProvenance
from .rights import RightsDecision
from .enums import (
    AssetCategory,
    Blinding,
    CaptureType,
    ClusterRole,
    EntranceVersion,
    ExteriorInterior,
    GeometrySuitability,
    PropertyMatchStatus,
    ReconstructionRole,
    ReviewDecision,
    ReviewStatus,
    Rights,
    Subject,
    TemporalStatus,
    ViewSector,
)

#: Droits autorisant l'usage d'un asset en production (reconstruction).
PRODUCTION_RIGHTS = frozenset({Rights.OWNED, Rights.LICENSED, Rights.OPEN_DATA})

#: Correspondances imposées entre une décision humaine et ce qu'elle emporte.
#: Les tenir dans le schéma les rend opposables à tout appelant, y compris à
#: un `model_copy` distrait : la cascade et la revue les réappliquent, le
#: manifeste les vérifie.
DECISION_STATUS: dict[ReviewDecision, ReviewStatus] = {
    ReviewDecision.CONFIRMED: ReviewStatus.HUMAN_ACCEPTED,
    ReviewDecision.REJECTED: ReviewStatus.REJECTED,
    # Examiné sans conclure : un état terminal, non une absence de revue. La
    # réouverture passe par une entrée ajoutée — preuve nouvelle ou
    # supersession — et jamais par un recalcul.
    ReviewDecision.UNRESOLVED: ReviewStatus.HUMAN_UNRESOLVED,
}

#: Statut porté par les manifestes antérieurs pour une revue non conclusive.
#: Conservé pour que la migration le reconnaisse, jamais écrit à nouveau.
LEGACY_UNRESOLVED_STATUS = ReviewStatus.NEEDS_REVIEW

VISIBILITY_OF: dict[ReviewDecision, bool | None] = {
    ReviewDecision.CONFIRMED: True,
    ReviewDecision.REJECTED: False,
    ReviewDecision.UNRESOLVED: None,
}

#: Champs écrits par `visibility apply`. Exclus de l'empreinte de base : leur
#: ajout ne doit pas périmer le run qui vient de les produire, alors qu'un cap
#: corrigé ou une revue humaine, eux, le doivent.
VISIBILITY_PROJECTED_FIELDS: frozenset[str] = frozenset(
    {
        "visibility_run_id",
        "visibility_run_digest",
        "visibility_assessment_id",
        "line_of_sight_status",
        "occlusion_risk_by",
        "occlusion_blocked_by",
        "target_in_frame_fraction",
        "occluded_by",
    }
)

#: Statuts qu'une personne seule peut poser.
_HUMAN_STATUSES = frozenset(
    {
        ReviewStatus.HUMAN_ACCEPTED,
        ReviewStatus.REJECTED,
        ReviewStatus.HUMAN_UNRESOLVED,
    }
)

#: Aptitudes autorisant un usage géométrique. `auxiliary` y figure : la vue
#: sert au raccord et à l'enregistrement, mais le Router la comptera à part —
#: un point de vue auxiliaire n'est pas une observation structurante.
_GEOMETRY_USABLE = frozenset(
    {GeometrySuitability.PRIMARY, GeometrySuitability.AUXILIARY}
)


class GpsPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class TemporalDecision(BaseModel):
    """Arbitrage humain de datation, pour une portée donnée.

    Prioritaire sur toute dérivation et jamais recalculé : remplacer le verrou
    humain par une déduction automatique ferait perdre la seule information
    qu'aucune date de fichier ne porte.
    """

    model_config = ConfigDict(extra="forbid")

    scope: str
    status: TemporalStatus
    decided_by: str
    decided_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    rationale: str
    evidence: list[str] = Field(default_factory=list)


class DecisionEntry(BaseModel):
    """Socle commun des arbitrages humains, quel qu'en soit l'objet.

    Immuable par convention : on n'édite jamais une entrée, on en ajoute une
    qui la corrige. L'empreinte de l'image jugée y figure — une décision porte
    sur ce qui a été vu, et si le fichier change, la décision ne le suit pas.
    """

    model_config = ConfigDict(extra="forbid")

    decided_by: str = Field(min_length=1)
    decided_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    rationale: str = Field(min_length=1)

    #: Au moins une preuve, obligatoire. Un motif dit ce que la personne a
    #: conclu ; la preuve dit sur quoi. Sans elle, une revue ne se rejoue pas :
    #: relire « c'est bien l'hôtel » six mois plus tard n'apprend rien.
    evidence: list[str] = Field(min_length=1)

    #: Empreinte de l'image au moment de la décision. Obligatoire : une
    #: décision qui ne dit pas sur quoi elle portait ne peut pas être opposée
    #: à un fichier modifié depuis.
    reviewed_checksum: str = Field(min_length=1)

    #: Ce que cette entrée corrige : l'index de l'entrée antérieure. La
    #: première n'en a pas.
    supersedes_index: int | None = Field(default=None, ge=0)

    #: Conditions dans lesquelles la décision a été prise. Portées ici plutôt
    #: que sur la seule visibilité : l'aptitude géométrique s'étiquette avec
    #: les mêmes précautions, et devait être traçable de la même façon.
    blinding: Blinding = Blinding.UNBLINDED

    #: Protocole d'étiquetage suivi, et empreinte du **protocole** — non de la
    #: file, que le nom précédent laissait croire. Une décision `blind` sans
    #: eux serait une déclaration invérifiable.
    review_protocol_id: str | None = None
    review_protocol_digest: str | None = None

    #: Empreinte de la file réellement présentée, quand elle est connue : deux
    #: files peuvent partager un protocole et différer par leur contenu.
    blind_queue_digest: str | None = None

    @model_validator(mode="after")
    def _blind_needs_a_protocol(self) -> "DecisionEntry":
        if self.blinding is Blinding.BLIND and not (
            self.review_protocol_id and self.review_protocol_digest
        ):
            raise ValueError(
                "décision déclarée aveugle sans protocole ni empreinte de file — "
                "l'aveuglement se prouve, il ne s'affirme pas"
            )
        return self

    @model_validator(mode="after")
    def _no_blank_text(self) -> "DecisionEntry":
        if not self.decided_by.strip():
            raise ValueError("revue sans auteur — une décision anonyme n'engage rien")
        if not self.rationale.strip():
            raise ValueError(
                "revue sans justification — le verdict seul ne s'audite pas"
            )
        blank = [e for e in self.evidence if not e.strip()]
        if blank or not self.evidence:
            raise ValueError(
                "revue sans preuve : une preuve vide n'en est pas une"
            )
        return self


class ReviewEntry(DecisionEntry):
    """Arbitrage d'identité : est-ce bien le bâtiment cible ?"""

    decision: ReviewDecision


class GeometryEntry(DecisionEntry):
    """Arbitrage d'aptitude : l'image apporte-t-elle de la structure ?

    Séparée de la visibilité parce que les deux réponses divergent
    couramment : le WelcomINNS est parfaitement identifiable à 117 m, sur
    40 % de la largeur du cadre, sans que sa façade y soit exploitable.
    """

    suitability: GeometrySuitability

    #: Mesures ayant fondé l'appréciation — fraction du cadre, dimensions en
    #: pixels, façade non masquée, netteté sur la cible. Conservées telles
    #: quelles : `quality_score` mesure le fichier entier, or un ciel net
    #: n'aide en rien la géométrie.
    measurements: dict[str, float] = Field(default_factory=dict)


def check_history(entries: list, label: str, asset_id: str) -> None:  # noqa: ANN001
    """Filiation d'un historique append-only, quelle qu'en soit la nature."""
    for position, entry in enumerate(entries):
        if position == 0:
            if entry.supersedes_index is not None:
                raise ValueError(
                    f"asset {asset_id!r} : la première décision {label} ne corrige "
                    f"rien, mais désigne l'entrée {entry.supersedes_index}"
                )
            continue
        if entry.supersedes_index is None:
            raise ValueError(
                f"asset {asset_id!r} : la décision {label} n° {position + 1} ne dit "
                "pas laquelle elle corrige"
            )
        if entry.supersedes_index >= position:
            raise ValueError(
                f"asset {asset_id!r} : la décision {label} n° {position + 1} corrige "
                f"l'entrée {entry.supersedes_index}, qui ne lui est pas antérieure"
            )


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

    # --- Lot 1B : vérité multidimensionnelle -----------------------------
    #: Origine véritable du média. Expedia, Hotels.com, Momondo et Kayak
    #: republient une même famille : sans ce champ, une republication gonfle
    #: artificiellement le nombre de vues.
    source_family: str | None = None

    exact_duplicate_group: str | None = None
    perceptual_duplicate_group: str | None = None
    viewpoint_cluster: str | None = None
    cluster_role: ClusterRole | None = None

    #: Dimensions et poids réels, servant à choisir le fichier canonique d'un
    #: groupe : une republication recompressée ne doit pas primer sur la source.
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    file_size_bytes: int | None = Field(default=None, ge=0)

    #: Azimut du bâtiment vers la caméra : de quel côté l'observateur se tient.
    #: Distinct de `heading_deg`, qui dit où la caméra regarde.
    bearing_from_building_deg: float | None = Field(default=None, ge=0, lt=360)

    #: Le cap est-il **observé** ou **choisi par nous** ?
    #:
    #: Mapillary rapporte le cap qu'un conducteur a réellement adopté : c'est
    #: une mesure, et la visibilité qui en découle vaut preuve. Street View rend
    #: un panorama sphérique dont nous extrayons la direction de notre choix :
    #: la visibilité n'y est qu'une intention de cadrage et ne dit rien du
    #: contenu de l'image.
    #:
    #: Confondre les deux revient à se donner raison — 105 vues « voyant le
    #: bâtiment » par construction, contre 20 réellement confirmées par le
    #: modèle sur le même lot.
    heading_is_measured: bool = True

    #: Multi-étiquette : une photo montre souvent bâtiment, parking et enseigne.
    subjects: list[Subject] = Field(default_factory=list)

    view_sector: ViewSector = ViewSector.UNKNOWN
    capture_type: CaptureType = CaptureType.UNKNOWN

    #: Défaut prudent : un asset ne devient source de géométrie que sur
    #: décision explicite, jamais par omission.
    reconstruction_role: ReconstructionRole = ReconstructionRole.REFERENCE_ONLY

    #: Statut agrégé, au plus restrictif. Ne sert qu'aux résumés : les
    #: décisions se prennent par portée.
    temporal_status: TemporalStatus = TemporalStatus.UNKNOWN

    #: Statut par portée — `entrance`, `facade`, `roof`, `signage`. Une photo
    #: peut montrer une entrée rénovée et une façade inchangée.
    temporal_by_scope: dict[str, TemporalStatus] = Field(default_factory=dict)
    temporal_method: str | None = None
    temporal_decisions: list[TemporalDecision] = Field(default_factory=list)

    #: Confiance et méthode de la qualification automatique. Conservées pour
    #: qu'une décision faible reste identifiable comme telle (Lot 1B §6).
    classification_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    classification_method: str | None = None
    review_status: ReviewStatus = ReviewStatus.NEEDS_REVIEW

    #: Usage assumé par l'opérateur malgré des droits non établis.
    #: Lève le verrou du §9 pour cet asset, mais reste inscrit dans le manifeste
    #: et propagé jusqu'au rapport final : l'option est tracée, pas dissoute.
    rights_encumbered: bool = False
    rights_note: str | None = None

    #: Décisions de droits, append-only. L'acquisition n'en produit aucune :
    #: elle constate un fait — ce fichier vient de là — et ne tranche rien.
    #: Une autorisation est une décision humaine, avec auteur, date, portée et
    #: preuves, et elle se corrige en ajoutant, jamais en réécrivant.
    rights_history: list["RightsDecision"] = Field(default_factory=list)

    #: Métadonnées de prise de vue, utiles au preflight et à la couverture.
    attribution: str | None = None
    heading_deg: float | None = Field(default=None, ge=0, lt=360)
    camera_lat: float | None = Field(default=None, ge=-90, le=90)
    camera_lon: float | None = Field(default=None, ge=-180, le=180)

    #: Texte lu sur l'image par OCR, conservé comme preuve du statut
    #: d'appartenance plutôt que comme simple verdict.
    sign_text: str | None = None

    #: Le bâtiment confirmé tombe-t-il dans le champ de la caméra, sans
    #: obstacle ? Critère géométrique, dont la valeur probante dépend de
    #: `heading_is_measured`.
    sees_building: bool | None = None

    #: Ancien verdict de cadrage, produit par l'annotateur mono-rayon
    #: supprimé. Conservé comme trace, **non probant** : la cascade ne s'en
    #: sert plus, seul un cadrage calculé fait foi.
    #:
    #: **Un** bâtiment quelconque est-il visible ? Réponse du modèle, qui ne
    #: distingue pas le WelcomINNS d'un concessionnaire Toyota.
    contains_building: bool | None = None

    #: **Le** bâtiment cible est-il réellement visible ?
    #:
    #: Distinction décisive : confondre les deux a produit 20 vues Street View
    #: classées porteuses de géométrie alors qu'elles montraient Boucherville
    #: Toyota, Rachelle Béry ou Tetra Tech — dont 17 avec `sees_building` faux
    #: et 15 avec un bâtiment interposé.
    #:
    #: Ne peut être établi que par une preuve : cap observé cadrant l'empreinte
    #: sans occlusion, enseigne lue, ou revue humaine.
    target_building_visible: bool | None = None
    target_evidence: str | None = None

    #: Arbitrage humain. Prioritaire sur toute déduction, et **jamais**
    #: recalculé : relancer la cascade ne doit pas effacer une décision prise
    #: par une personne.
    target_visibility_decision: ReviewDecision = ReviewDecision.UNRESOLVED
    reviewer: str | None = None
    reviewed_at: datetime | None = None
    review_rationale: str | None = None
    review_evidence: list[str] = Field(default_factory=list)

    #: Historique **append-only** des arbitrages. Les champs ci-dessus ne
    #: gardent que le dernier : une revue corrigée effacerait sans trace la
    #: décision qu'elle corrige, et l'on ne saurait plus ni ce qui avait été
    #: conclu, ni par qui, ni sur quelle preuve. Le dernier élément et les
    #: champs plats disent toujours la même chose.
    review_history: list["ReviewEntry"] = Field(default_factory=list)

    #: Aptitude géométrique, **indépendante** de l'identité. Une vue peut
    #: montrer la bonne façade sans porter de structure exploitable ; les
    #: confondre promouvait une vue lointaine au rang d'observation
    #: géométrique du seul fait qu'on y reconnaissait l'hôtel.
    geometry_suitability: GeometrySuitability = GeometrySuitability.UNASSESSED
    geometry_history: list["GeometryEntry"] = Field(default_factory=list)

    @property
    def has_been_assessed(self) -> bool:
        """Une personne s'est-elle prononcée sur l'aptitude géométrique ?"""
        return bool(self.geometry_history)

    @property
    def carries_geometry(self) -> bool:
        """L'aptitude autorise-t-elle un usage géométrique ?

        `unassessed` ne l'autorise pas : faire de l'absence d'examen une
        approbation est précisément ce qui a promu des vues lointaines.
        """
        return self.geometry_suitability in _GEOMETRY_USABLE

    @property
    def has_been_reviewed(self) -> bool:
        """Une personne a-t-elle regardé cet asset ?

        `target_visibility_decision` vaut `unresolved` par défaut : sans cette
        distinction, « jamais examiné » et « examiné sans conclusion » se
        confondraient, et le second perdrait toute valeur.
        """
        return bool(self.review_history)

    #: Score par sujet, conservé tel quel. Une confiance agrégée masquait la
    #: qualité réelle de la décision décisive.
    subject_scores: dict[str, float] = Field(default_factory=dict)

    #: Empreinte voisine coupant la ligne de visée, le cas échéant. Le champ de
    #: vision ne suffit pas : un pavillon interposé annule la vue.
    occluded_by: str | None = None
    target_distance_m: float | None = Field(default=None, ge=0)
    target_offset_deg: float | None = Field(default=None, ge=0, le=180)
    local_path: str | None = None
    #: Résultats projetés par `visibility apply`. Ils décrivent la géométrie,
    #: jamais le contenu : aucun d'eux ne dit que la caméra vise le bâtiment,
    #: ni qu'il entre dans l'image.
    visibility_run_id: str | None = None
    visibility_run_digest: str | None = None
    visibility_assessment_id: str | None = None
    line_of_sight_status: str | None = None

    #: Obstacles dont une donnée verticale manque, et obstacles prouvés
    #: masquants. Le premier n'est pas une occultation.
    occlusion_risk_by: list[str] = Field(default_factory=list)
    occlusion_blocked_by: list[str] = Field(default_factory=list)

    #: Part de la silhouette réellement dans le cadre, quand les paramètres de
    #: caméra permettent de la calculer. `None` signifie « non calculable »,
    #: jamais « hors cadre » : le corpus actuel n'en a aucune.
    target_in_frame_fraction: float | None = Field(default=None, ge=0.0, le=1.0)

    #: D'où vient réellement le fichier : identifiant fournisseur, positions
    #: interrogée et rendue, cadrage demandé, plan qui l'a retenu. Sans elle,
    #: `source_url_or_id` portait une URL de CDN, et l'identité durable de
    #: l'asset dépendait d'un lien qui expire.
    acquisition: "AcquisitionProvenance | None" = None

    phash: str | None = None
    #: Signature multi-segments utilisée uniquement pour comparer des
    #: republications plausibles. Elle résiste à un recadrage ou un filigrane,
    #: mais ne doit jamais rapprocher toutes les vues consécutives d'une rue.
    crop_resistant_hash: str | None = None
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0)

    @property
    def usable_in_production(self) -> bool:
        return self.rights in PRODUCTION_RIGHTS or self.rights_encumbered

    @model_validator(mode="after")
    def _rights_gate_production(self) -> "Asset":
        """Un asset ne peut être éligible production que si ses droits le permettent.

        Verrou structurel du §9 : une image publique reste `reference_only` tant
        que ses droits ne permettent pas son usage en reconstruction — sauf
        décision explicite de l'opérateur, qui doit alors être inscrite.
        """
        if self.production_eligible and not self.usable_in_production:
            raise ValueError(
                f"asset {self.id!r} marqué production_eligible avec "
                f"rights={self.rights.value!r} ; droits acceptés : "
                f"{sorted(r.value for r in PRODUCTION_RIGHTS)}, ou rights_encumbered explicite"
            )
        if self.ai_eligible and not self.usable_in_production:
            raise ValueError(
                f"asset {self.id!r} marqué ai_eligible avec rights={self.rights.value!r}"
            )
        if self.rights_encumbered and self.rights in PRODUCTION_RIGHTS:
            raise ValueError(
                f"asset {self.id!r} : rights_encumbered n'a pas de sens avec "
                f"des droits déjà suffisants ({self.rights.value!r})"
            )
        return self

    @model_validator(mode="after")
    def _review_history_is_append_only(self) -> "Asset":
        """L'historique de revue est append-only **par le schéma**, non par usage.

        Une convention tenue par un seul module cède au premier appelant qui
        l'ignore. Trois choses sont donc imposées ici : la filiation des
        entrées, leur accord avec les champs plats, et l'impossibilité
        d'afficher une décision humaine que rien n'atteste.
        """
        history = self.review_history
        check_history(history, "de visibilité", self.id)
        check_history(self.geometry_history, "d'aptitude géométrique", self.id)

        if self.geometry_history:
            if self.geometry_suitability is not self.geometry_history[-1].suitability:
                raise ValueError(
                    f"asset {self.id!r} : aptitude courante "
                    f"{self.geometry_suitability.value!r} ≠ dernière appréciation "
                    f"{self.geometry_history[-1].suitability.value!r}"
                )
        elif self.geometry_suitability is not GeometrySuitability.UNASSESSED:
            raise ValueError(
                f"asset {self.id!r} : aptitude {self.geometry_suitability.value!r} "
                "sans appréciation à l'appui"
            )

        if not history:
            # Aucune revue : les champs plats ne peuvent pas prétendre le
            # contraire. Sans cela, un statut humain se poserait sur du vide.
            # `human_unresolved` en fait partie : constater qu'on ne peut pas
            # trancher suppose d'avoir regardé.
            if self.review_status in _HUMAN_STATUSES:
                raise ValueError(
                    f"asset {self.id!r} porte le statut {self.review_status.value!r} "
                    "sans aucune revue à l'appui"
                )
            if self.target_visibility_decision is not ReviewDecision.UNRESOLVED:
                raise ValueError(
                    f"asset {self.id!r} porte la décision "
                    f"{self.target_visibility_decision.value!r} sans aucune revue"
                )
            if self.reviewer or self.review_rationale:
                raise ValueError(
                    f"asset {self.id!r} : auteur ou motif de revue sans historique"
                )
            return self

        last = history[-1]
        if self.target_visibility_decision is not last.decision:
            raise ValueError(
                f"asset {self.id!r} : décision courante "
                f"{self.target_visibility_decision.value!r} ≠ dernière revue "
                f"{last.decision.value!r}"
            )
        if self.review_status is not DECISION_STATUS[last.decision]:
            # Le cas le plus probable est un manifeste écrit avant que
            # « examiné sans conclure » cesse d'être « en attente de revue ».
            # Le dire évite de faire chercher une incohérence de données là où
            # il n'y a qu'une version antérieure.
            legacy = (
                last.decision is ReviewDecision.UNRESOLVED
                and self.review_status is LEGACY_UNRESOLVED_STATUS
            )
            hint = (
                " — manifeste antérieur au statut terminal ; "
                "« hotel-pipeline assets migrate-review-status <hotel_id> » "
                "le convertit sans toucher aux décisions"
                if legacy
                else ""
            )
            raise ValueError(
                f"asset {self.id!r} : statut {self.review_status.value!r} "
                f"incompatible avec la décision {last.decision.value!r} ; "
                f"attendu {DECISION_STATUS[last.decision].value!r}{hint}"
            )
        # Une personne qui n'a pas conclu n'interdit pas au système de constater
        # qu'il ne voit rien : après un `unresolved`, la déduction automatique
        # peut rendre `False` ou `None`. Elle ne peut pas rendre `True` — ce
        # serait établir ce que la revue a justement refusé d'établir.
        allowed = (
            (None, False)
            if last.decision is ReviewDecision.UNRESOLVED
            else (VISIBILITY_OF[last.decision],)
        )
        if self.target_building_visible not in allowed:
            raise ValueError(
                f"asset {self.id!r} : visibilité {self.target_building_visible!r} "
                f"incompatible avec la décision {last.decision.value!r} ; "
                f"attendu {allowed if len(allowed) > 1 else allowed[0]!r}"
            )
        if self.reviewer != last.decided_by or self.review_rationale != last.rationale:
            raise ValueError(
                f"asset {self.id!r} : auteur ou motif courant divergent de la "
                "dernière revue"
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

    def encumbered(self) -> list[Asset]:
        """Assets utilisés en production sous droits assumés par l'opérateur."""
        return [a for a in self.assets if a.production_eligible and a.rights_encumbered]

    def unique_photographs(self) -> int:
        """Photographies distinctes, republications fusionnées (Lot 1B §5)."""
        return len({a.perceptual_duplicate_group or a.exact_duplicate_group or a.id
                    for a in self.assets})

    def viewpoints(self) -> int:
        """Points de vue indépendants — l'unité que comptent les Gates."""
        return len({a.viewpoint_cluster for a in self.assets if a.viewpoint_cluster})
