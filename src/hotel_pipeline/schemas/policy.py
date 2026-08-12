"""Politique de pipeline — seuils et tolérances, versionnés (Lot 1B, généricité).

Ces valeurs ne décrivent pas un établissement : elles décrivent **notre
méthode**. Les placer dans le profil de chaque hôtel autoriserait un
recalibrage par site, et une calibration valable pour un seul corpus ne vaut
rien — c'est précisément ce qui a produit un seuil mesuré sur 36 images de
Boucherville puis appliqué comme s'il était universel.

Elles sont donc regroupées ici, versionnées, et destinées à être validées sur
plusieurs établissements avant d'être modifiées.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelPolicy(BaseModel):
    """Classifieur et ses seuils."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_name: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"

    subject_accept: float = Field(default=0.50, ge=0.0, le=1.0)
    subject_reject: float = Field(default=0.20, ge=0.0, le=1.0)

    #: Sur quoi ces seuils ont été mesurés. Sans cette trace, un seuil est un
    #: nombre sans autorité.
    #: En deçà, une décision automatique n'est pas acceptée sans revue.
    review_confidence_floor: float = Field(default=0.60, ge=0.0, le=1.0)

    calibration_id: str = "welcominns-2026-08-36-images"
    calibrated_on_sites: int = 1


class GeometryPolicy(BaseModel):
    """Visibilité et relations spatiales."""

    model_config = ConfigDict(extra="forbid")

    half_fov_deg: float = Field(default=45.0, gt=0, le=180)
    max_distance_m: float = Field(default=200.0, gt=0)

    #: Contiguïté franche puis association plausible entre bâtiment et parking.
    adjacency_strong_m: float = Field(default=8.0, gt=0)
    adjacency_max_m: float = Field(default=30.0, gt=0)


class DedupPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phash_hamming_threshold: int = Field(default=6, ge=0, le=64)
    position_tolerance_m: float = Field(default=10.0, gt=0)
    bearing_tolerance_deg: float = Field(default=25.0, gt=0, le=180)
    max_overlap_per_cluster: int = Field(default=2, ge=0)


class CollectionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    radius_m: int = Field(default=500, ge=25, le=2000)
    road_radius_m: int = Field(default=350, ge=25, le=2000)
    sample_spacing_m: float = Field(default=15.0, gt=0)
    snap_radius_m: int = Field(default=25, gt=0)
    max_panorama_distance_m: float = Field(default=220.0, gt=0)

    #: Champ de vision demandé aux vues Street View, et sa variante élargie
    #: pour les transitions route–entrée–stationnement.
    image_fov_deg: int = Field(default=80, gt=0, le=120)
    wide_fov_deg: int = Field(default=110, gt=0, le=120)


class TemporalPolicy(BaseModel):
    """Ce qu'une datation inconnue autorise, selon l'usage.

    La géométrie d'un volume change peu : une vue non datée reste exploitable
    pour la structure. L'apparence d'une entrée rénovée, non — une image
    antérieure aux travaux y introduirait une erreur invisible.
    """

    model_config = ConfigDict(extra="forbid")

    allow_unknown_for_geometry: bool = True
    allow_unknown_for_appearance: bool = False
    require_current_for_sensitive_zones: bool = True

    #: Portées dont l'apparence engage la fidélité du rendu. Une datation
    #: inconnue y interdit l'usage d'apparence — jamais l'usage géométrique.
    sensitive_scopes: list[str] = Field(default_factory=lambda: ["entrance", "signage"])


class PipelinePolicy(BaseModel):
    """Politique complète, identifiée par sa version."""

    model_config = ConfigDict(extra="forbid")

    version: str = "1.0.0"
    model: ModelPolicy = Field(default_factory=ModelPolicy)
    geometry: GeometryPolicy = Field(default_factory=GeometryPolicy)
    dedup: DedupPolicy = Field(default_factory=DedupPolicy)
    collection: CollectionPolicy = Field(default_factory=CollectionPolicy)
    temporal: TemporalPolicy = Field(default_factory=TemporalPolicy)


#: Politique par défaut. Toute fonction qui l'accepte doit la prendre en
#: paramètre, jamais lire une constante de module.
DEFAULT_POLICY = PipelinePolicy()
