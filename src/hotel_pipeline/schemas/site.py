"""Manifeste de site — les instances réelles d'un établissement (Lot 1B §4).

Séparation stricte :

- le **gabarit** (`REQUIRED_OBJECTS`, `EXCLUDED_KINDS`) définit des types
  génériques, valables partout ;
- le **manifeste de site** porte les instances propres à ce lieu, chacune avec
  son identifiant stable, sa nature, sa géométrie ou sa référence source, son
  état de confirmation, ses preuves et ses relations.

Deux règles gouvernent son contenu.

**Rien n'est inventé.** Un objet que les données ne permettent pas d'établir
existe quand même, à l'état `unresolved` : le supprimer ferait croire qu'il n'a
pas été cherché, et le remplir ferait croire qu'il est connu.

**Une exclusion est une instance, pas un mot.** Le parc-o-bus voisin n'est pas
un terme à filtrer : c'est un objet réel, géolocalisé, relié à la propriété par
une relation de distinction. C'est ce qui permet de vérifier la séparation au
lieu de l'espérer.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .critical_objects import EXCLUDED_KINDS, REQUIRED_OBJECTS
from .enums import ObjectState


class GeoSourceProvenance(BaseModel):
    """Origine d'une donnée géospatiale ayant servi à dériver un objet.

    Un objet dérivé sans provenance n'est pas vérifiable : on ignore de quel
    millésime il vient, dans quel référentiel vertical il est exprimé, et avec
    quel algorithme il a été produit. Une altitude sans référentiel vertical
    n'est pas une altitude — c'est un nombre.
    """

    model_config = ConfigDict(extra="forbid")

    source_id: str
    dataset: str            # ex. « LiDAR Québec », « Orthophotos CMM »
    vintage: str | None = None   # millésime ou année d'acquisition
    tile_id: str | None = None

    #: Référentiels horizontal **et** vertical, distincts et tous deux requis
    #: pour une donnée d'élévation.
    crs_horizontal: str | None = None
    crs_vertical: str | None = None

    #: Résolution d'un raster, ou densité de points d'un nuage.
    resolution_m: float | None = Field(default=None, gt=0)
    point_density_per_m2: float | None = Field(default=None, gt=0)

    file_digest: str | None = None
    licence: str | None = None
    retrieved_at: datetime | None = None
    notes: str | None = None

    #: La source porte-t-elle de l'élévation ? Un nuage de points et un MNT
    #: en portent ; une orthophoto, non.
    carries_elevation: bool = False

    @model_validator(mode="after")
    def _elevation_needs_both_datums(self) -> "GeoSourceProvenance":
        """Une élévation exige **les deux** référentiels.

        Un nuage de points avec densité et référentiel horizontal, mais sans
        référentiel vertical, était accepté : ses altitudes n'auraient référé
        à rien.
        """
        elevation = self.carries_elevation or self.point_density_per_m2 is not None

        if elevation and not self.crs_horizontal:
            raise ValueError(
                f"source {self.source_id!r} : source d'élévation sans référentiel "
                "horizontal — elle ne situe rien"
            )
        if elevation and not self.crs_vertical:
            raise ValueError(
                f"source {self.source_id!r} : source d'élévation sans référentiel "
                "vertical — ses altitudes ne réfèrent à rien"
            )
        if self.crs_vertical and not self.crs_horizontal:
            raise ValueError(
                f"source {self.source_id!r} : référentiel vertical sans référentiel "
                "horizontal"
            )
        return self

    def is_citable(self) -> list[str]:
        """Champs manquants pour qu'un objet dérivé puisse s'y référer.

        Une source consultée peut rester incomplète ; une source **citée** par
        une dérivation doit être identifiable et rejouable.
        """
        required = {
            "tile_id": self.tile_id,
            "vintage": self.vintage,
            "licence": self.licence,
            "retrieved_at": self.retrieved_at,
            "file_digest": self.file_digest,
        }
        return sorted(name for name, value in required.items() if not value)


class SiteRelation(BaseModel):
    """Lien entre deux instances du site.

    Les relations portent l'essentiel de la vérification : « le stationnement
    de l'hôtel est contigu au bâtiment » et « il est distinct du parc-o-bus »
    sont des assertions testables, là où une simple étiquette ne l'est pas.
    """

    model_config = ConfigDict(extra="forbid")

    predicate: str  # adjacent_to, distinct_from, serves, belongs_to, part_of
    target_id: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)


class SiteObject(BaseModel):
    """Une instance réelle, du type défini par le gabarit."""

    model_config = ConfigDict(extra="forbid")

    #: Identifiant stable, indépendant des sources : il survit à un changement
    #: d'identifiant OSM ou de fournisseur.
    object_id: str

    #: Type générique issu du gabarit — `BUILDING_MAIN`, `PARK_AND_RIDE`...
    kind: str

    state: ObjectState = ObjectState.UNRESOLVED

    #: D'où vient l'objet, tel que la source le nomme — `way/54581348`.
    source_ref: str | None = None
    geometry_wkt: str | None = None
    centroid_lat: float | None = Field(default=None, ge=-90, le=90)
    centroid_lon: float | None = Field(default=None, ge=-180, le=180)

    evidence: list[str] = Field(default_factory=list)
    relations: list[SiteRelation] = Field(default_factory=list)

    #: Sources géospatiales dont l'objet est dérivé, et par quel traitement.
    derived_from_sources: list[str] = Field(default_factory=list)
    derivation_method: str | None = None

    #: Pourquoi l'objet reste indéterminé, le cas échéant. Un état sans motif
    #: est une information perdue.
    unresolved_reason: str | None = None

    confirmed_by: str | None = None
    confirmed_at: datetime | None = None
    confirmation_rationale: str | None = None

    @model_validator(mode="after")
    def _confirmed_needs_evidence(self) -> "SiteObject":
        if self.state is ObjectState.CONFIRMED and not (self.evidence or self.source_ref):
            raise ValueError(
                f"objet {self.object_id!r} confirmé sans preuve ni référence source"
            )
        if self.state is ObjectState.UNRESOLVED and self.geometry_wkt:
            raise ValueError(
                f"objet {self.object_id!r} porte une géométrie mais reste 'unresolved'"
            )
        return self

    def relation_to(self, target_id: str) -> SiteRelation | None:
        return next((r for r in self.relations if r.target_id == target_id), None)


class SiteManifest(BaseModel):
    """Instances du site, requises comme exclues."""

    model_config = ConfigDict(extra="forbid")

    hotel_id: str
    objects: list[SiteObject] = Field(default_factory=list)
    built_at: datetime | None = None

    #: Sources géospatiales référencées par les objets dérivés.
    geo_sources: list[GeoSourceProvenance] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids(self) -> "SiteManifest":
        seen: set[str] = set()
        for obj in self.objects:
            if obj.object_id in seen:
                raise ValueError(f"identifiant d'objet dupliqué : {obj.object_id!r}")
            seen.add(obj.object_id)

        known = {o.object_id for o in self.objects}
        for obj in self.objects:
            for relation in obj.relations:
                if relation.target_id not in known:
                    raise ValueError(
                        f"relation de {obj.object_id!r} vers un objet absent : "
                        f"{relation.target_id!r}"
                    )

        source_ids = [s.source_id for s in self.geo_sources]
        duplicates = {i for i in source_ids if source_ids.count(i) > 1}
        if duplicates:
            raise ValueError(
                f"identifiants de source dupliqués : {sorted(duplicates)} — "
                "une référence de dérivation serait ambiguë"
            )

        sources = set(source_ids)
        by_id = {s.source_id: s for s in self.geo_sources}
        for obj in self.objects:
            missing = [s for s in obj.derived_from_sources if s not in sources]
            if missing:
                raise ValueError(
                    f"objet {obj.object_id!r} dérivé de sources non déclarées : {missing}"
                )
            if obj.derived_from_sources and not obj.derivation_method:
                raise ValueError(
                    f"objet {obj.object_id!r} dérivé sans méthode de dérivation"
                )
            for source_id in obj.derived_from_sources:
                incomplete = by_id[source_id].is_citable()
                if incomplete:
                    raise ValueError(
                        f"objet {obj.object_id!r} dérivé de {source_id!r}, dont "
                        f"la provenance est incomplète : {incomplete}"
                    )
        return self

    # -- accès ------------------------------------------------------------

    def by_id(self, object_id: str) -> SiteObject | None:
        return next((o for o in self.objects if o.object_id == object_id), None)

    def by_kind(self, kind: str) -> list[SiteObject]:
        return [o for o in self.objects if o.kind == kind]

    def confirmed(self) -> list[SiteObject]:
        return [o for o in self.objects if o.state is ObjectState.CONFIRMED]

    def unresolved(self) -> list[SiteObject]:
        return [o for o in self.objects if o.state is ObjectState.UNRESOLVED]

    def excluded_instances(self) -> list[SiteObject]:
        """Objets voisins à distinguer, réellement localisés."""
        return [o for o in self.objects if o.kind in EXCLUDED_KINDS]

    def missing_required(self) -> list[str]:
        """Types du gabarit qu'aucune instance ne représente.

        Différent d'un objet `unresolved` : celui-ci existe et porte son motif.
        Une absence ici signifie que le type n'a même pas été instancié.
        """
        present = {o.kind for o in self.objects}
        return [kind for kind in REQUIRED_OBJECTS if kind not in present]

    def source(self, source_id: str) -> GeoSourceProvenance | None:
        return next((s for s in self.geo_sources if s.source_id == source_id), None)

    def derived(self) -> list[SiteObject]:
        """Objets issus d'un traitement géospatial, par opposition aux relevés."""
        return [o for o in self.objects if o.derived_from_sources]

    def summary(self) -> dict[str, int]:
        return {
            "objects": len(self.objects),
            "confirmed": len(self.confirmed()),
            "unresolved": len(self.unresolved()),
            "excluded_instances": len(self.excluded_instances()),
            "missing_kinds": len(self.missing_required()),
            "relations": sum(len(o.relations) for o in self.objects),
            "geo_sources": len(self.geo_sources),
            "derived_objects": len(self.derived()),
        }
