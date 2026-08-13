"""Résolution des géométries de capture (Lot 1B V2, étape 2).

Le cache Overpass du site contient 56 éléments — 28 bâtiments, 28
stationnements — et **aucune route** : `way/938806358` n'y figure pas. Dériver
la voie d'accès de ce cache aurait donc produit une absence inventée. Deux
interrogations distinctes sont faites ici, et conservées telles quelles.

Rien n'y est déduit sans le dire : une forme résolue porte sa source, sa
méthode, ses deux référentiels et l'empreinte de la réponse qui l'a produite ;
une forme absente porte un motif ; une panne n'est jamais une absence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..logging import get_logger
from ..schemas.geometry import (
    CANONICAL_PRECISION,
    CRS_TOLERANCE_M,
    GEOGRAPHIC_CRS,
    PROJECTED_CRS,
    AccessStatus,
    CaptureGeometryManifest,
    CorridorClass,
    GeometryResolutionStatus,
    GeometryRole,
    GeometrySourceSnapshot,
    ResolvedGeometry,
    RoadCorridor,
    SourceQueryStatus,
)

log = get_logger("capture-geometry")

#: Interdiction franche.
_PRIVATE_ACCESS = frozenset({"private", "no"})

#: Accès conditionnel : l'usage est encadré, non fermé. `customers` dit que
#: les clients de l'établissement y passent — ce qui, pour l'hôtel lui-même,
#: est plutôt un argument.
_RESTRICTED_ACCESS = frozenset({"customers", "permit", "delivery", "destination"})

#: Types de voies dont l'usage public est établi par l'usage même du tag.
_PUBLIC_HIGHWAYS = frozenset(
    {
        "motorway", "trunk", "primary", "secondary", "tertiary",
        "unclassified", "residential", "living_street",
    }
)

#: Valeur de repli, employée seulement si aucune politique n'est fournie. Le
#: seuil réel vient de `policy.geometry.adjacency_max_m` : le tenir en double
#: ici garantissait qu'un jour les deux divergent.
DEFAULT_ADJACENCY_M = 30.0


def canonical_digest(geometry) -> str:  # noqa: ANN001 — shapely
    """Empreinte d'une forme, sur une écriture normalisée à précision fixée.

    Deux écritures de la même géométrie — ordre des anneaux, décimales
    superflues — donneraient sans cela deux empreintes, et toute comparaison
    ultérieure croirait à un changement.
    """
    from shapely import set_precision
    from shapely.ops import transform

    rounded = transform(
        lambda x, y, z=None: (
            round(x, CANONICAL_PRECISION), round(y, CANONICAL_PRECISION)
        ),
        geometry,
    )
    normalised = set_precision(rounded, 10**-CANONICAL_PRECISION).normalize()
    return hashlib.sha256(normalised.wkt.encode("utf-8")).hexdigest()[:16]


def response_digest(payload) -> str:  # noqa: ANN001
    """Empreinte d'une réponse source, stable à l'ordre des clés près."""
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def transformer():  # noqa: ANN201
    """Transformateur WGS84 → EPSG:2950, axes explicitement en x/y.

    `always_xy=False` échangerait latitude et longitude sans rien signaler, et
    la forme partirait à des milliers de kilomètres.
    """
    from pyproj import Transformer

    return Transformer.from_crs(GEOGRAPHIC_CRS, PROJECTED_CRS, always_xy=True)


def project(geometry):  # noqa: ANN001, ANN201
    from shapely.ops import transform

    forward = transformer()
    return transform(lambda xs, ys, zs=None: forward.transform(xs, ys), geometry)


def check_crs_pair(wgs84_wkt: str, projected_wkt: str) -> list[str]:
    """Reprojette la forme géographique et la confronte à la forme projetée.

    Conserver deux WKT indépendants n'établit rien : ils peuvent décrire deux
    endroits. La seule vérification utile est de refaire le calcul.
    """
    from shapely import wkt as shapely_wkt

    problems: list[str] = []
    try:
        geographic = shapely_wkt.loads(wgs84_wkt)
        projected = shapely_wkt.loads(projected_wkt)
    except Exception as exc:  # noqa: BLE001 — WKT illisible, quelle qu'en soit la cause
        return [f"WKT illisible : {exc}"]

    if not geographic.is_valid:
        problems.append("forme géographique invalide")
    if not projected.is_valid:
        problems.append("forme projetée invalide")
    if geographic.geom_type != projected.geom_type:
        problems.append(
            f"types incompatibles : {geographic.geom_type} et {projected.geom_type}"
        )
        return problems

    minx, miny, maxx, maxy = geographic.bounds
    if not (-180 <= minx <= 180 and -90 <= miny <= 90 and -180 <= maxx <= 180):
        problems.append(
            "coordonnées géographiques hors domaine"
        )
        return problems

    # Le domaine mondial ne suffit pas : une inversion latitude/longitude
    # produit ici (45,57 ; -73,44), deux valeurs parfaitement valides prises
    # séparément. Seule l'emprise du référentiel projeté la révèle — EPSG:2950
    # ne couvre que le Québec entre 75° et 72° ouest.
    if not _within_projected_area(minx, miny, maxx, maxy):
        problems.append(
            f"coordonnées hors de l'emprise de {PROJECTED_CRS} — latitude et "
            "longitude probablement inversées"
        )
        return problems

    recomputed = project(geographic)
    deviation = recomputed.hausdorff_distance(projected)
    if deviation > CRS_TOLERANCE_M:
        problems.append(
            f"les deux référentiels divergent de {deviation:.2f} m "
            f"(tolérance {CRS_TOLERANCE_M} m)"
        )
    return problems


def _within_projected_area(minx: float, miny: float, maxx: float, maxy: float) -> bool:
    from pyproj import CRS

    area = CRS.from_user_input(PROJECTED_CRS).area_of_use
    if area is None:  # pragma: no cover — tous les CRS utilisés en ont une
        return True
    west, south, east, north = area.bounds
    return west <= minx <= east and east >= maxx >= west and south <= miny <= north and south <= maxy <= north


def resolved_from(
    feature_id: str,
    role: GeometryRole,
    source_ref: str,
    snapshot_id: str,
    geometry,  # noqa: ANN001 — shapely en WGS84
    derivation_method: str,
    evidence: list[str],
    **extra,
) -> ResolvedGeometry:
    """Construit une géométrie résolue, ses deux référentiels compris."""
    import pyproj

    projected = project(geometry)
    return ResolvedGeometry(
        feature_id=feature_id,
        role=role,
        resolution_status=GeometryResolutionStatus.RESOLVED,
        source_ref=source_ref,
        snapshot_id=snapshot_id,
        wgs84_wkt=geometry.wkt,
        projected_wkt=projected.wkt,
        geometry_type=geometry.geom_type,
        horizontal_crs=GEOGRAPHIC_CRS,
        projected_crs=PROJECTED_CRS,
        transform_method=f"pyproj.Transformer.from_crs({GEOGRAPHIC_CRS}→{PROJECTED_CRS})",
        always_xy=True,
        pyproj_version=pyproj.__version__,
        geometry_digest=canonical_digest(geometry),
        derivation_method=derivation_method,
        evidence=evidence,
        **extra,
    )


def unresolved(
    feature_id: str, role: GeometryRole, reason: str, **extra
) -> ResolvedGeometry:
    """Une géométrie qu'on n'a pas obtenue — et pourquoi.

    L'objet, lui, garde son état au SiteManifest : ne pas retrouver un tracé ne
    prouve pas que la voie n'existe pas.
    """
    return ResolvedGeometry(
        feature_id=feature_id,
        role=role,
        resolution_status=GeometryResolutionStatus.UNRESOLVED,
        unresolved_reason=reason,
        **extra,
    )


# --- géométries Overpass ------------------------------------------------------


def shape_of(element: dict):  # noqa: ANN201
    """Forme d'un élément Overpass `out geom`.

    La fermeture ne suffit pas à faire une surface : `way/938806358` est une
    allée de stationnement en boucle, et l'interpréter comme un polygone la
    rendait incompatible avec son rôle de voie. Chez OSM, une voie reste
    linéaire tant qu'elle ne porte pas `area=yes` — c'est le tag qui décide,
    pas la géométrie.
    """
    from shapely.geometry import LineString, Polygon

    points = [(node["lon"], node["lat"]) for node in element.get("geometry") or []]
    if len(points) < 2:
        return None

    tags = element.get("tags") or {}
    linear_by_nature = "highway" in tags and (tags.get("area") or "").lower() != "yes"
    closed = len(points) >= 4 and points[0] == points[-1]

    if closed and not linear_by_nature:
        polygon = Polygon(points)
        return polygon if polygon.is_valid else polygon.buffer(0)
    return LineString(points)


def access_status_of(tags: dict[str, str]) -> tuple[AccessStatus, str]:
    """Accessibilité juridique d'une voie, et ce qui la fonde."""
    access = (tags.get("access") or "").strip().lower()
    highway = (tags.get("highway") or "").strip().lower()

    if access in _PRIVATE_ACCESS:
        return AccessStatus.PRIVATE, f"tag access={access!r}"
    if access in _RESTRICTED_ACCESS:
        return (
            AccessStatus.RESTRICTED,
            f"tag access={access!r} : accès conditionnel, non interdiction",
        )
    if access in {"yes", "public", "permissive"}:
        return AccessStatus.PUBLIC_CONFIRMED, f"tag access={access!r}"
    if highway in _PUBLIC_HIGHWAYS:
        # Déduit du type de voie, non affirmé : une résidentielle privée existe.
        return AccessStatus.PUBLIC_INFERRED, f"voie de type {highway!r}, sans tag access"
    return AccessStatus.UNKNOWN, "aucun tag access, type de voie non concluant"


def classify(
    element: dict,
    distance_to_building_m: float | None,
    distance_to_parking_m: float | None,
    access_ref: str | None,
    adjacency_max_m: float = DEFAULT_ADJACENCY_M,
) -> tuple[CorridorClass, str]:
    """Classe une voie, sans jamais la rendre admissible d'office.

    Le seuil d'adjacence est reçu, non lu dans une constante : une politique
    posée dans l'espace de travail doit changer le classement, faute de quoi
    elle décorerait un rapport sans rien décider.
    """
    tags = element.get("tags") or {}
    highway = (tags.get("highway") or "").lower()
    service = (tags.get("service") or "").lower()
    ref = f"way/{element.get('id')}"

    if access_ref and ref == access_ref:
        return CorridorClass.ACCESS_MAIN, "voie d'accès déclarée au manifeste de site"
    if service == "parking_aisle":
        return CorridorClass.PARKING_AISLE, "allée de stationnement (service=parking_aisle)"
    if distance_to_building_m is None:
        return CorridorClass.EXCLUDED, "distance au bâtiment non calculable"
    if distance_to_building_m <= adjacency_max_m:
        return (
            CorridorClass.ADJACENT_ROAD,
            f"à {distance_to_building_m:.0f} m de l'empreinte, sous le seuil "
            f"d'adjacence de {adjacency_max_m:.0f} m",
        )
    if highway in _PUBLIC_HIGHWAYS or highway == "service":
        return (
            CorridorClass.NON_ADJACENT_POTENTIAL,
            f"à {distance_to_building_m:.0f} m : au-delà du seuil de "
            f"{adjacency_max_m:.0f} m, hors adjacence ; utilité à établir par "
            "la visibilité multi-rayons",
        )
    return CorridorClass.EXCLUDED, f"type de voie {highway!r} sans usage de capture"


# --- rapport ------------------------------------------------------------------


@dataclass
class ResolutionReport:
    hotel_id: str = ""
    built_at: str = ""
    snapshots: list[dict] = field(default_factory=list)
    resolved: dict[str, int] = field(default_factory=dict)
    unresolved: list[dict] = field(default_factory=list)
    corridors: dict[str, int] = field(default_factory=dict)
    corridor_details: list[dict] = field(default_factory=list)
    invalid_geometries: list[str] = field(default_factory=list)

    #: Empreintes retirées des obstacles parce qu'elles décrivent la cible, et
    #: par quelle méthode. Une voisine supprimée par erreur serait sinon
    #: invisible.
    target_exclusions: list[dict] = field(default_factory=list)
    crs_problems: list[str] = field(default_factory=list)
    road_geometry_digest: str = ""
    obstacle_geometry_digest: str = ""
    manifest_digest: str = ""

    def as_dict(self) -> dict:
        return {
            "hotel_id": self.hotel_id,
            "built_at": self.built_at,
            "snapshots": self.snapshots,
            "resolved_by_role": self.resolved,
            "unresolved": self.unresolved,
            "corridors_by_class": self.corridors,
            "corridors": self.corridor_details,
            "invalid_geometries": self.invalid_geometries,
            "target_exclusions": self.target_exclusions,
            "crs_problems": self.crs_problems,
            # Deux empreintes distinctes : le plan d'acquisition les cite
            # séparément, et une route ajoutée ne doit pas se confondre avec un
            # obstacle ajouté.
            "road_geometry_digest": self.road_geometry_digest,
            "obstacle_geometry_digest": self.obstacle_geometry_digest,
            "manifest_digest": self.manifest_digest,
        }


def digest_of(geometries: list[ResolvedGeometry]) -> str:
    """Empreinte d'un ensemble de géométries, indépendante de leur ordre."""
    parts = sorted(
        f"{g.feature_id}:{g.geometry_digest or g.resolution_status.value}"
        for g in geometries
    )
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def snapshot(
    snapshot_id: str,
    source: str,
    endpoint: str,
    query: str,
    payload: list | dict | None,
    error: str | None = None,
    radius_m: float | None = None,
    policy_digest: str | None = None,
) -> GeometrySourceSnapshot:
    """Consigne une interrogation, en distinguant l'absence de la panne."""
    if error is not None:
        status = SourceQueryStatus.DISCOVERY_ERROR
        return GeometrySourceSnapshot(
            snapshot_id=snapshot_id, source=source, endpoint=endpoint, query=query,
            status=status, error=error, search_radius_m=radius_m,
            policy_digest=policy_digest,
        )

    elements = payload if isinstance(payload, list) else (payload or {}).get("elements", [])
    if not elements:
        return GeometrySourceSnapshot(
            snapshot_id=snapshot_id, source=source, endpoint=endpoint, query=query,
            status=SourceQueryStatus.NOT_FOUND, element_count=0,
            search_radius_m=radius_m, policy_digest=policy_digest,
            response_digest=response_digest(elements),
        )

    return GeometrySourceSnapshot(
        snapshot_id=snapshot_id, source=source, endpoint=endpoint, query=query,
        status=SourceQueryStatus.SUCCESS, element_count=len(elements),
        response_digest=response_digest(elements), search_radius_m=radius_m,
        policy_digest=policy_digest,
    )


def verify(manifest: CaptureGeometryManifest) -> list[str]:
    """Recalcule ce que le manifeste affirme : référentiels et empreintes."""
    problems: list[str] = []
    for geometry in manifest.geometries:
        if geometry.resolution_status is not GeometryResolutionStatus.RESOLVED:
            continue
        for problem in check_crs_pair(geometry.wgs84_wkt, geometry.projected_wkt):
            problems.append(f"{geometry.feature_id} : {problem}")

        from shapely import wkt as shapely_wkt

        recomputed = canonical_digest(shapely_wkt.loads(geometry.wgs84_wkt))
        if recomputed != geometry.geometry_digest:
            problems.append(
                f"{geometry.feature_id} : empreinte {geometry.geometry_digest} "
                f"≠ {recomputed} recalculée"
            )
    return problems


def mark_stale(manifest: CaptureGeometryManifest, digests: dict[str, str]) -> list[str]:
    """Passe en `stale` les formes dont la réponse source a changé.

    Une géométrie périmée n'est pas fausse : elle décrit ce que la source
    disait. Elle cesse simplement de pouvoir fonder une décision.
    """
    by_snapshot = {s.snapshot_id: s for s in manifest.snapshots}
    changed: list[str] = []

    for index, geometry in enumerate(manifest.geometries):
        if geometry.resolution_status is not GeometryResolutionStatus.RESOLVED:
            continue
        recorded = by_snapshot.get(geometry.snapshot_id or "")
        current = digests.get(geometry.snapshot_id or "")
        if not recorded or not current or recorded.response_digest == current:
            continue

        manifest.geometries[index] = geometry.model_copy(
            update={
                "resolution_status": GeometryResolutionStatus.STALE,
                "wgs84_wkt": None,
                "projected_wkt": None,
                "unresolved_reason": (
                    f"la source a changé depuis l'instantané {geometry.snapshot_id} "
                    f"({recorded.response_digest} → {current})"
                ),
            }
        )
        changed.append(geometry.feature_id)

    if changed:
        log.info("%d géométrie(s) périmée(s) : %s", len(changed), sorted(changed))
    return changed


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
