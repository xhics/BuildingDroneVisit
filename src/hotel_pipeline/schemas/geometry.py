"""Registre des géométries de capture (Lot 1B V2, étape 2).

Sans géométries, un planificateur ne peut affirmer ni qu'une caméra est sur la
voie d'accès, ni qu'un voisin masque la façade : il ne dispose que de
centroïdes et de rayons. Ce registre les résout et les conserve, avec ce qui
permet d'en douter.

Deux distinctions gouvernent le fichier.

**La géométrie n'est pas l'objet.** `ACCESS_ROAD_MAIN` peut rester `inferred`
au SiteManifest tandis que sa géométrie est `unresolved` : ne pas retrouver un
tracé ne prouve pas que la voie n'existe pas. Ce registre ne modifie donc
jamais l'état des objets du site.

**Une absence n'est pas une panne.** `not_found` dit que la source a répondu et
n'a rien ; `discovery_error` dit qu'elle n'a pas répondu. Les confondre ferait
conclure « aucune route ici » d'un simple 504.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GeometryResolutionStatus(StrEnum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    #: Résolue autrefois, mais la source a changé depuis.
    STALE = "stale"


class SourceQueryStatus(StrEnum):
    SUCCESS = "success"
    #: La source a répondu, et ne connaît pas l'élément.
    NOT_FOUND = "not_found"
    #: La source n'a pas répondu, ou a répondu n'importe quoi. Une panne ne
    #: prouve aucune absence.
    DISCOVERY_ERROR = "discovery_error"


class GeometryRole(StrEnum):
    TARGET_BUILDING = "target_building"
    HOTEL_PARKING = "hotel_parking"
    ACCESS_ROAD = "access_road"
    ROAD_CANDIDATE = "road_candidate"
    OBSTACLE_BUILDING = "obstacle_building"
    CONTEXT_CORRIDOR = "context_corridor"
    FORBIDDEN_ZONE = "forbidden_zone"


class CorridorClass(StrEnum):
    ACCESS_MAIN = "access_main"
    PARKING_AISLE = "parking_aisle"
    #: Adjacente à la propriété, ce qui ne dit rien du droit d'y accéder :
    #: l'unique voie adjacente du WelcomINNS porte `access=customers`. La
    #: classe décrit la position, `access_status` décrit le droit.
    ADJACENT_ROAD = "adjacent_road"
    #: Non adjacente, éventuellement utile : elle le restera jusqu'à ce que la
    #: visibilité multi-rayons le démontre. Aucune acquisition ne s'y appuie
    #: aujourd'hui.
    NON_ADJACENT_POTENTIAL = "non_adjacent_potential"
    EXCLUDED = "excluded"


class AccessStatus(StrEnum):
    """Accessibilité **juridique**, distincte de la classe de corridor.

    L'absence de tag `access` chez OSM ne vaut pas autorisation : la plupart
    des voies publiques n'en portent aucun, et bien des allées privées non
    plus. Le silence reste donc un silence.
    """

    PUBLIC_CONFIRMED = "public_confirmed"
    PUBLIC_INFERRED = "public_inferred"

    #: Accès conditionnel — `customers`, `permit`, `delivery`, `destination`.
    #: Ce n'est pas une interdiction : une capture autorisée par
    #: l'établissement y reste possible, et confondre les deux fermait par
    #: avance l'allée qui longe l'hôtel.
    RESTRICTED = "restricted"

    #: Interdiction franche — `private`, `no`.
    PRIVATE = "private"
    UNKNOWN = "unknown"


#: Types géométriques admis par rôle.
GEOMETRY_TYPES: dict[GeometryRole, frozenset[str]] = {
    GeometryRole.TARGET_BUILDING: frozenset({"Polygon", "MultiPolygon"}),
    GeometryRole.HOTEL_PARKING: frozenset({"Polygon", "MultiPolygon"}),
    GeometryRole.OBSTACLE_BUILDING: frozenset({"Polygon", "MultiPolygon"}),
    GeometryRole.ACCESS_ROAD: frozenset({"LineString", "MultiLineString"}),
    GeometryRole.ROAD_CANDIDATE: frozenset({"LineString", "MultiLineString"}),
    GeometryRole.CONTEXT_CORRIDOR: frozenset({"Polygon", "MultiPolygon"}),
    GeometryRole.FORBIDDEN_ZONE: frozenset({"Polygon", "MultiPolygon"}),
}

#: Référentiel projeté des calculs. Les longueurs et les surfaces n'ont aucun
#: sens en degrés.
PROJECTED_CRS = "EPSG:2950"
GEOGRAPHIC_CRS = "EPSG:4326"

#: Écart maximal toléré, en mètres, entre la forme projetée conservée et la
#: reprojection de la forme géographique. Il couvre l'arrondi de sérialisation
#: à six décimales — environ 11 cm de longitude à cette latitude — sans laisser
#: passer une inversion d'axes, qui déplacerait la forme de milliers de
#: kilomètres.
CRS_TOLERANCE_M = 0.5

#: Décimales de la sérialisation canonique, fixées pour que deux empreintes de
#: la même forme coïncident.
CANONICAL_PRECISION = 6


class GeometrySourceSnapshot(BaseModel):
    """Une interrogation de source, telle qu'elle s'est passée."""

    model_config = ConfigDict(extra="forbid")

    snapshot_id: str
    source: str
    endpoint: str
    query: str
    queried_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: SourceQueryStatus
    element_count: int = Field(default=0, ge=0)

    #: Empreinte de la réponse. Obligatoire dès qu'elle a abouti : c'est elle
    #: qui rendra une géométrie `stale` quand la source aura changé.
    response_digest: str | None = None

    #: Ce que la requête a réellement demandé.
    search_radius_m: float | None = Field(default=None, gt=0)
    policy_digest: str | None = None
    error: str | None = None

    @model_validator(mode="after")
    def _status_matches_evidence(self) -> "GeometrySourceSnapshot":
        if self.status is SourceQueryStatus.SUCCESS:
            if not self.response_digest:
                raise ValueError(
                    f"instantané {self.snapshot_id!r} : succès sans empreinte de "
                    "réponse — rien ne dira qu'elle a changé"
                )
            if self.element_count == 0:
                raise ValueError(
                    f"instantané {self.snapshot_id!r} : succès sans élément ; une "
                    "réponse vide se déclare 'not_found'"
                )
        if self.status is SourceQueryStatus.DISCOVERY_ERROR:
            if not (self.error or "").strip():
                raise ValueError(
                    f"instantané {self.snapshot_id!r} : panne sans description"
                )
            if self.element_count:
                raise ValueError(
                    f"instantané {self.snapshot_id!r} : panne annoncée avec "
                    f"{self.element_count} élément(s) — une erreur ne rapporte rien"
                )
        return self


class ResolvedGeometry(BaseModel):
    """Une géométrie, son état de résolution et sa provenance."""

    model_config = ConfigDict(extra="forbid")

    feature_id: str
    role: GeometryRole
    resolution_status: GeometryResolutionStatus

    #: Pourquoi cette géométrie a cessé d'être rattachée au site. `stale` sans
    #: motif se relit comme une donnée périmée par le temps, alors qu'ici c'est
    #: une association qui a été démentie.
    stale_reason: str | None = None

    #: Référence telle que la source la nomme — `way/938806358`.
    source_ref: str | None = None
    snapshot_id: str | None = None

    #: Les deux représentations. La géographique s'échange, la projetée se
    #: calcule : conserver l'une sans l'autre obligerait à reprojeter à chaque
    #: usage, ou à mesurer des mètres en degrés.
    wgs84_wkt: str | None = None
    projected_wkt: str | None = None
    geometry_type: str | None = None

    horizontal_crs: str | None = None
    projected_crs: str | None = None
    transform_method: str | None = None
    #: Inscrit explicitement : `always_xy=False` inverse latitude et longitude
    #: sans rien signaler, et la forme part à des milliers de kilomètres.
    always_xy: bool | None = None
    pyproj_version: str | None = None

    #: Empreinte de la forme, sur une sérialisation canonique à précision
    #: fixée : sans elle, deux écritures de la même géométrie sembleraient
    #: différentes.
    geometry_digest: str | None = None

    derivation_method: str | None = None
    evidence: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)

    #: Motif obligatoire lorsqu'aucune forme n'a été obtenue.
    unresolved_reason: str | None = None

    #: Hauteur de l'obstacle, si elle est connue. Jamais inventée : une hauteur
    #: absente deviendra un `occlusion_risk`, pas une certitude.
    height_known: bool = False
    height_m: float | None = Field(default=None, gt=0)
    height_source: str | None = None

    #: Géométries dont celle-ci procède — un corridor cite ses routes.
    derived_from: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _shape_matches_status(self) -> "ResolvedGeometry":
        resolved = self.resolution_status is GeometryResolutionStatus.RESOLVED
        has_shape = bool(self.wgs84_wkt or self.projected_wkt)

        if not resolved and has_shape:
            raise ValueError(
                f"géométrie {self.feature_id!r} : forme présente sur un état "
                f"{self.resolution_status.value!r}"
            )
        if not resolved and not (self.unresolved_reason or "").strip():
            raise ValueError(
                f"géométrie {self.feature_id!r} : non résolue sans motif"
            )

        if resolved:
            missing = [
                name
                for name in (
                    "source_ref", "snapshot_id", "wgs84_wkt", "projected_wkt",
                    "geometry_type", "horizontal_crs", "projected_crs",
                    "transform_method", "geometry_digest", "derivation_method",
                )
                if not getattr(self, name)
            ]
            if missing:
                raise ValueError(
                    f"géométrie {self.feature_id!r} résolue sans provenance : {missing}"
                )
            if self.always_xy is not True:
                raise ValueError(
                    f"géométrie {self.feature_id!r} : `always_xy=True` doit être "
                    "explicite — l'inverse échange latitude et longitude en silence"
                )
            if not self.evidence:
                raise ValueError(
                    f"géométrie {self.feature_id!r} résolue sans preuve"
                )
            if self.horizontal_crs != GEOGRAPHIC_CRS:
                raise ValueError(
                    f"géométrie {self.feature_id!r} : référentiel géographique "
                    f"{self.horizontal_crs!r} attendu {GEOGRAPHIC_CRS}"
                )
            if not self.projected_crs:
                raise ValueError(
                    f"géométrie {self.feature_id!r} : référentiel projeté absent"
                )
            allowed = GEOMETRY_TYPES[self.role]
            if self.geometry_type not in allowed:
                raise ValueError(
                    f"géométrie {self.feature_id!r} de rôle {self.role.value!r} : "
                    f"type {self.geometry_type!r} ; attendu l'un de {sorted(allowed)}"
                )

        if self.height_known and self.height_m is None:
            raise ValueError(
                f"géométrie {self.feature_id!r} : hauteur déclarée connue sans valeur"
            )
        if self.height_m is not None and not self.height_known:
            raise ValueError(
                f"géométrie {self.feature_id!r} : hauteur fournie sans être déclarée connue"
            )
        return self


class RoadCorridor(BaseModel):
    """Une voie, ce qu'elle vaut pour la capture, et ce qu'on a le droit d'y faire.

    La classe dit l'usage envisageable, `access_status` dit le droit : une
    allée privée peut être la meilleure vue et rester inaccessible.
    """

    model_config = ConfigDict(extra="forbid")

    corridor_id: str
    feature_id: str
    corridor_class: CorridorClass
    access_status: AccessStatus = AccessStatus.UNKNOWN

    #: Distances **réelles**, calculées en projection depuis les formes, non
    #: depuis des centroïdes.
    distance_to_building_m: float | None = Field(default=None, ge=0)
    distance_to_parking_m: float | None = Field(default=None, ge=0)

    osm_tags: dict[str, str] = Field(default_factory=dict)
    rationale: str = Field(min_length=1)

    #: Paramètres du tampon, si un corridor surfacique en a été dérivé.
    buffer_m: float | None = Field(default=None, gt=0)
    derived_geometry_id: str | None = None

    #: Admissible pour une acquisition de bâtiment ? Une route non adjacente
    #: ne peut pas l'être avant la visibilité multi-rayons.
    admissible_for_building: bool = False

    @model_validator(mode="after")
    def _non_adjacent_stays_potential(self) -> "RoadCorridor":
        if (
            self.corridor_class is CorridorClass.NON_ADJACENT_POTENTIAL
            and self.admissible_for_building
        ):
            raise ValueError(
                f"corridor {self.corridor_id!r} : une route non adjacente reste "
                "potentielle tant que la visibilité multi-rayons ne l'a pas justifiée"
            )
        if self.corridor_class is CorridorClass.EXCLUDED and self.admissible_for_building:
            raise ValueError(
                f"corridor {self.corridor_id!r} : exclu et pourtant admissible"
            )
        return self


class CaptureGeometryManifest(BaseModel):
    """Les géométries dont dépend toute décision de capture."""

    model_config = ConfigDict(extra="forbid")

    #: Version du schéma. **Sans défaut** : un manifeste qui ne la porte pas
    #: n'est pas un manifeste de cette version, c'est un fichier antérieur, et
    #: le laisser recevoir « 1.0.0 » en silence lui prêterait des garanties
    #: qu'il n'a jamais eues. Le chargeur `load_capture_geometry` fait le tri.
    schema_version: str = Field(min_length=1)
    hotel_id: str
    built_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    snapshots: list[GeometrySourceSnapshot] = Field(default_factory=list)
    geometries: list[ResolvedGeometry] = Field(default_factory=list)
    corridors: list[RoadCorridor] = Field(default_factory=list)

    #: Référentiels du calcul. Obligatoires : un manifeste qui ne dit pas dans
    #: quel référentiel ses formes projetées vivent n'est pas exploitable, et
    #: rien ne permettrait de refuser de les confronter à un autre.
    source_crs: str = GEOGRAPHIC_CRS
    working_crs: str = Field(min_length=1)
    spatial_context_digest: str = Field(min_length=1)

    site_manifest_digest: str | None = None
    spatial_manifest_digest: str | None = None
    policy_digest: str | None = None
    overpass_elements_digest: str | None = None

    @model_validator(mode="after")
    def _invariants(self) -> "CaptureGeometryManifest":
        feature_ids = [g.feature_id for g in self.geometries]
        duplicated = sorted({i for i in feature_ids if feature_ids.count(i) > 1})
        if duplicated:
            raise ValueError(f"identifiants de géométrie dupliqués : {duplicated}")

        snapshot_ids = [s.snapshot_id for s in self.snapshots]
        if len(set(snapshot_ids)) != len(snapshot_ids):
            raise ValueError("identifiants d'instantané dupliqués")

        known_snapshots = set(snapshot_ids)
        known_features = set(feature_ids)

        for geometry in self.geometries:
            if geometry.snapshot_id and geometry.snapshot_id not in known_snapshots:
                raise ValueError(
                    f"géométrie {geometry.feature_id!r} : instantané "
                    f"{geometry.snapshot_id!r} absent du manifeste"
                )
            missing = [p for p in geometry.derived_from if p not in known_features]
            if missing:
                raise ValueError(
                    f"géométrie {geometry.feature_id!r} dérive de formes absentes : "
                    f"{missing}"
                )
            if (
                geometry.role is GeometryRole.CONTEXT_CORRIDOR
                and geometry.resolution_status is GeometryResolutionStatus.RESOLVED
                and not geometry.derived_from
            ):
                raise ValueError(
                    f"corridor {geometry.feature_id!r} sans filiation : on ne saurait "
                    "pas de quelles voies il procède"
                )

        # Le bâtiment cible ne peut pas se masquer lui-même. L'oublier ferait
        # rejeter toutes les vues pour occlusion par la cible.
        targets = {
            g.source_ref for g in self.geometries
            if g.role is GeometryRole.TARGET_BUILDING and g.source_ref
        }
        obstacles = {
            g.source_ref for g in self.geometries
            if g.role is GeometryRole.OBSTACLE_BUILDING and g.source_ref
        }
        overlap = sorted(targets & obstacles)
        if overlap:
            raise ValueError(
                f"le bâtiment cible figure parmi les obstacles : {overlap}"
            )

        mismatched = [
            geometry.feature_id
            for geometry in self.geometries
            if geometry.resolution_status is GeometryResolutionStatus.RESOLVED
            and geometry.projected_crs != self.working_crs
        ]
        if mismatched:
            raise ValueError(
                f"géométries hors du CRS de travail {self.working_crs!r} : {mismatched}"
            )

        corridor_ids = [c.corridor_id for c in self.corridors]
        if len(set(corridor_ids)) != len(corridor_ids):
            raise ValueError("identifiants de corridor dupliqués")
        for corridor in self.corridors:
            if corridor.feature_id not in known_features:
                raise ValueError(
                    f"corridor {corridor.corridor_id!r} : géométrie "
                    f"{corridor.feature_id!r} absente du manifeste"
                )
            if corridor.derived_geometry_id and corridor.derived_geometry_id not in known_features:
                raise ValueError(
                    f"corridor {corridor.corridor_id!r} : forme dérivée "
                    f"{corridor.derived_geometry_id!r} absente"
                )
        return self

    def by_role(self, role: GeometryRole) -> list[ResolvedGeometry]:
        return [g for g in self.geometries if g.role is role]

    def resolved(self, role: GeometryRole) -> list[ResolvedGeometry]:
        return [
            g for g in self.by_role(role)
            if g.resolution_status is GeometryResolutionStatus.RESOLVED
        ]

    def corridors_by_class(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for corridor in self.corridors:
            counts[corridor.corridor_class.value] = (
                counts.get(corridor.corridor_class.value, 0) + 1
            )
        return dict(sorted(counts.items()))
