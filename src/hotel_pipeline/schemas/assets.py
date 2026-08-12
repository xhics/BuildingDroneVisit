"""Manifeste d'assets (plan directeur §9).

Un champ obligatoire absent ou mal typé doit produire une erreur explicite.
Aucun asset ne doit être routé silencieusement avec des métadonnées invalides.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import (
    AssetCategory,
    CaptureType,
    ClusterRole,
    EntranceVersion,
    ExteriorInterior,
    PropertyMatchStatus,
    ReconstructionRole,
    ReviewStatus,
    Rights,
    Subject,
    TemporalStatus,
    ViewSector,
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

    temporal_status: TemporalStatus = TemporalStatus.UNKNOWN

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

    #: Score par sujet, conservé tel quel. Une confiance agrégée masquait la
    #: qualité réelle de la décision décisive.
    subject_scores: dict[str, float] = Field(default_factory=dict)

    #: Empreinte voisine coupant la ligne de visée, le cas échéant. Le champ de
    #: vision ne suffit pas : un pavillon interposé annule la vue.
    occluded_by: str | None = None
    target_distance_m: float | None = Field(default=None, ge=0)
    target_offset_deg: float | None = Field(default=None, ge=0, le=180)
    local_path: str | None = None
    phash: str | None = None
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
