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


def _first_cycle(parents: dict[str, list[str]]) -> list[str] | None:
    """Premier cycle rencontré dans la filiation, s'il en existe un.

    Une filiation circulaire rendrait la chaîne de dérivation impossible à
    rejouer : aucun artefact ne pourrait être produit en premier.
    """
    visiting: set[str] = set()
    done: set[str] = set()

    def walk(node: str, path: list[str]) -> list[str] | None:
        if node in done:
            return None
        if node in visiting:
            return [*path[path.index(node):], node]
        visiting.add(node)
        for parent in parents.get(node, []):
            found = walk(parent, [*path, node])
            if found:
                return found
        visiting.discard(node)
        done.add(node)
        return None

    for artifact_id in parents:
        found = walk(artifact_id, [])
        if found:
            return found
    return None


class DerivedArtifact(BaseModel):
    """Fichier produit par une dérivation — raster, TIN, nuage découpé.

    Un WKT ne peut pas représenter honnêtement une surface 2,5D : il dit où,
    jamais à quelle altitude, ni à quelle résolution, ni quelle part est
    mesurée plutôt qu'interpolée. L'artefact porte donc ce que la géométrie
    seule tait.
    """

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    role: str  # dtm, dsm_roof, ndsm, unclassified_roof_candidates...
    path: str
    format: str  # GeoTIFF, LAZ, GeoJSON...
    sha256: str

    crs_horizontal: str
    crs_vertical: str | None = None
    resolution_m: float = Field(gt=0)
    nodata: float | None = None

    #: Quel traitement, avec quels paramètres. Sans eux, l'artefact n'est pas
    #: reproductible — et un raster non reproductible n'est pas une mesure.
    algorithm_id: str
    parameters: dict[str, str] = Field(default_factory=dict)

    #: Parts respectives de cellules mesurées et interpolées. Leur somme peut
    #: être inférieure à 1 : le reste est sans donnée.
    measured_fraction: float = Field(ge=0.0, le=1.0)
    interpolated_fraction: float = Field(default=0.0, ge=0.0, le=1.0)

    #: **Dénominateur** de ces fractions. L'empreinte, l'anneau, la boîte du
    #: raster et le site entier donnent des chiffres très différents pour la
    #: même donnée ; la boîte englobante flatte systématiquement un bâtiment
    #: oblique. Jamais implicite.
    coverage_domain: str
    coverage_mask_artifact_id: str | None = None

    derived_from_sources: list[str] = Field(default_factory=list)

    #: Artefacts dont celui-ci procède. Le nDSM dérive du DTM, de la surface de
    #: toiture et de son masque de validité : ne citer que le LAZ masquerait la
    #: dérivation réelle et rendrait la chaîne irreproductible.
    derived_from_artifacts: list[str] = Field(default_factory=list)

    produced_at: datetime | None = None

    @model_validator(mode="after")
    def _fractions_and_datums(self) -> "DerivedArtifact":
        if self.measured_fraction + self.interpolated_fraction > 1.0 + 1e-9:
            raise ValueError(
                f"artefact {self.artifact_id!r} : mesuré + interpolé dépasse 1 "
                f"({self.measured_fraction} + {self.interpolated_fraction})"
            )
        if self.role in ELEVATION_ROLES and not self.crs_vertical:
            raise ValueError(
                f"artefact {self.artifact_id!r} porte des altitudes sans "
                "référentiel vertical"
            )
        if not self.derived_from_sources:
            raise ValueError(
                f"artefact {self.artifact_id!r} sans source : une dérivation "
                "sans origine n'est pas vérifiable"
            )
        if self.coverage_domain not in COVERAGE_DOMAINS:
            raise ValueError(
                f"artefact {self.artifact_id!r} : domaine de couverture "
                f"{self.coverage_domain!r} inconnu ; attendu l'un de "
                f"{sorted(COVERAGE_DOMAINS)}"
            )
        if self.coverage_domain == "mask" and not self.coverage_mask_artifact_id:
            raise ValueError(
                f"artefact {self.artifact_id!r} : domaine 'mask' sans artefact "
                "de masque référencé"
            )
        if self.artifact_id in self.derived_from_artifacts:
            raise ValueError(
                f"artefact {self.artifact_id!r} se cite lui-même comme parent"
            )
        return self

    @property
    def nodata_fraction(self) -> float:
        return round(1.0 - self.measured_fraction - self.interpolated_fraction, 6)


#: Domaines sur lesquels une fraction peut être calculée. Sans cette précision,
#: `measured_fraction=0.97` ne veut rien dire : 97 % de quoi ?
COVERAGE_DOMAINS: frozenset[str] = frozenset(
    {"footprint", "ring", "raster_box", "site", "mask"}
)

#: Rôles d'artefacts portant des altitudes, donc exigeant un datum vertical.
ELEVATION_ROLES: frozenset[str] = frozenset(
    {"dtm", "dsm", "dsm_roof", "ndsm", "tin", "unclassified_roof_candidates"}
)


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

    #: Artefacts produits qui portent la substance de l'objet — un raster
    #: d'altitude en dit davantage qu'un contour.
    artifact_ids: list[str] = Field(default_factory=list)

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

    #: Fichiers produits par les dérivations.
    artifacts: list[DerivedArtifact] = Field(default_factory=list)

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

        artifact_ids = [a.artifact_id for a in self.artifacts]
        duplicated = {i for i in artifact_ids if artifact_ids.count(i) > 1}
        if duplicated:
            raise ValueError(f"identifiants d'artefact dupliqués : {sorted(duplicated)}")

        for artifact in self.artifacts:
            unknown = [s for s in artifact.derived_from_sources if s not in sources]
            if unknown:
                raise ValueError(
                    f"artefact {artifact.artifact_id!r} dérivé de sources non "
                    f"déclarées : {unknown}"
                )
            # Même exigence que pour un objet : un artefact citant une
            # provenance incomplète serait tout aussi invérifiable.
            for source_id in artifact.derived_from_sources:
                incomplete = by_id[source_id].is_citable()
                if incomplete:
                    raise ValueError(
                        f"artefact {artifact.artifact_id!r} dérivé de "
                        f"{source_id!r}, dont la provenance est incomplète : "
                        f"{incomplete}"
                    )

        known_artifacts = set(artifact_ids)
        for artifact in self.artifacts:
            mask_id = artifact.coverage_mask_artifact_id
            if mask_id and mask_id not in known_artifacts:
                raise ValueError(
                    f"artefact {artifact.artifact_id!r} référence un masque "
                    f"absent : {mask_id!r}"
                )
            missing_parents = [
                a for a in artifact.derived_from_artifacts if a not in known_artifacts
            ]
            if missing_parents:
                raise ValueError(
                    f"artefact {artifact.artifact_id!r} dérive d'artefacts "
                    f"absents : {missing_parents}"
                )

        cycle = _first_cycle({a.artifact_id: a.derived_from_artifacts for a in self.artifacts})
        if cycle:
            raise ValueError(f"filiation d'artefacts cyclique : {' → '.join(cycle)}")

        for obj in self.objects:
            missing_artifacts = [a for a in obj.artifact_ids if a not in known_artifacts]
            if missing_artifacts:
                raise ValueError(
                    f"objet {obj.object_id!r} référence des artefacts absents : "
                    f"{missing_artifacts}"
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

    def artifact(self, artifact_id: str) -> DerivedArtifact | None:
        return next((a for a in self.artifacts if a.artifact_id == artifact_id), None)

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
            "artifacts": len(self.artifacts),
            "derived_objects": len(self.derived()),
        }
