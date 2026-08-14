"""Assemblage du manifeste géométrique depuis les sources réelles.

Trois interrogations, consignées séparément : la voie d'accès nommée par le
manifeste de site, le réseau routier du périmètre, et le cache d'éléments déjà
collecté — ce dernier n'étant réutilisé que s'il est cité par son empreinte.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..logging import get_logger
from ..schemas.geometry import (
    AccessStatus,
    CaptureGeometryManifest,
    CorridorClass,
    GeometryResolutionStatus,
    GeometryRole,
    ResolvedGeometry,
    RoadCorridor,
    SourceQueryStatus,
)
from . import capture_geometry as cg

log = get_logger("resolve-geometry")

OVERPASS_MIRROR = "https://overpass-api.de/api/interpreter"


def _digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def resolve(
    hotel_id: str,
    building_wkt: str,
    access_road_ref: str | None,
    elements: list[dict],
    elements_digest: str,
    roads: list[dict] | None,
    roads_error: str | None,
    access_element: list[dict] | None,
    access_error: str | None,
    radius_m: float,
    parking_ref: str | None = None,
    policy_digest: str | None = None,
    adjacency_max_m: float = cg.DEFAULT_ADJACENCY_M,
    projection_service=None,
) -> tuple[CaptureGeometryManifest, cg.ResolutionReport]:
    # `projection_service` reste nommé pour la lisibilité des appels, mais il
    # est requis : sans lui, ce module projetait en EPSG:2950 littéral.
    """Construit le manifeste géométrique à partir de réponses déjà obtenues.

    La fonction ne fait aucun appel réseau : elle reçoit ce que les sources ont
    répondu, y compris leurs pannes. C'est ce qui la rend rejouable sur des
    réponses figées.
    """
    from shapely import wkt as shapely_wkt

    if projection_service is None:
        raise ValueError(
            "aucun service de projection : le référentiel de travail se résout "
            "depuis la position du site (« geo reference »), il ne se suppose pas"
        )

    from .geometry_loader import CURRENT_SCHEMA_VERSION

    manifest = CaptureGeometryManifest(
        schema_version=CURRENT_SCHEMA_VERSION,
        hotel_id=hotel_id,
        source_crs=projection_service.reference.source_crs,
        working_crs=projection_service.working_crs,
        # L'empreinte vient du contexte lui-même, définie une seule fois :
        # la recalculer ici ferait diverger le manifeste, le run et
        # l'application au premier changement de format.
        spatial_context_digest=projection_service.reference.context_digest(),
        policy_digest=policy_digest,
        overpass_elements_digest=elements_digest,
    )
    report = cg.ResolutionReport(hotel_id=hotel_id, built_at=cg.now_iso())

    # --- instantanés ---------------------------------------------------------
    cache_snapshot = cg.snapshot(
        "overpass-features-cache", "overpass", OVERPASS_MIRROR,
        "way[building] / way[amenity=parking] (cache de collecte)",
        elements, radius_m=radius_m, policy_digest=policy_digest,
    )
    roads_snapshot = cg.snapshot(
        "overpass-roads", "overpass", OVERPASS_MIRROR,
        f"way[highway](around:{radius_m:.0f})", roads, error=roads_error,
        radius_m=radius_m, policy_digest=policy_digest,
    )
    snapshots = [cache_snapshot, roads_snapshot]

    access_snapshot = None
    if access_road_ref:
        access_snapshot = cg.snapshot(
            "overpass-access-road", "overpass", OVERPASS_MIRROR,
            f"way({access_road_ref.split('/')[-1]}) — voie d'accès du site",
            access_element, error=access_error,
            policy_digest=policy_digest,
        )
        snapshots.append(access_snapshot)

    manifest.snapshots = snapshots
    report.snapshots = [json.loads(s.model_dump_json()) for s in snapshots]

    geometries: list[ResolvedGeometry] = []
    corridors: list[RoadCorridor] = []

    # --- bâtiment cible ------------------------------------------------------
    target = shapely_wkt.loads(building_wkt)
    target_ref = _ref_of_wkt(elements, target)
    geometries.append(
        cg.resolved_from(
            "TARGET_BUILDING", GeometryRole.TARGET_BUILDING,
            target_ref or "spatial_manifest:confirmed_building",
            cache_snapshot.snapshot_id, target,
            "empreinte confirmée du manifeste spatial",
            ["bâtiment confirmé par l'opérateur au manifeste spatial"],
            projection_service=projection_service,
        )
    )
    target_projected = cg.project(target, projection_service)

    # --- stationnement -------------------------------------------------------
    parking = _element_by_ref(elements, parking_ref) if parking_ref else None
    if parking is not None:
        shape = cg.shape_of(parking)
        if shape is not None and shape.geom_type in ("Polygon", "MultiPolygon"):
            geometries.append(
                cg.resolved_from(
                    "HOTEL_PARKING", GeometryRole.HOTEL_PARKING, parking_ref,
                    cache_snapshot.snapshot_id, shape,
                    "élément Overpass amenity=parking cité par le manifeste de site",
                    [f"tags OSM : {json.dumps(parking.get('tags', {}), ensure_ascii=False)}"],
                    projection_service=projection_service,
                )
            )
        else:
            geometries.append(
                cg.unresolved(
                    "HOTEL_PARKING", GeometryRole.HOTEL_PARKING,
                    f"{parking_ref} présent mais sans contour surfacique exploitable",
                    source_ref=parking_ref, snapshot_id=cache_snapshot.snapshot_id,
                )
            )
    else:
        geometries.append(
            cg.unresolved(
                "HOTEL_PARKING", GeometryRole.HOTEL_PARKING,
                f"référence {parking_ref!r} absente du cache d'éléments"
                if parking_ref
                else "aucun stationnement référencé au manifeste de site",
                source_ref=parking_ref, snapshot_id=cache_snapshot.snapshot_id,
            )
        )

    # --- voie d'accès --------------------------------------------------------
    if access_road_ref:
        access_geometry = _resolve_access_road(
            access_road_ref, access_element, access_error,
            access_snapshot.snapshot_id if access_snapshot else None,
            projection_service,
        )
        geometries.append(access_geometry)

        # La voie d'accès est écartée du réseau candidat pour ne pas être
        # résolue deux fois — mais sans corridor, elle disparaissait du
        # classement, et la voie la plus importante du site n'y figurait pas.
        if access_geometry.resolution_status is GeometryResolutionStatus.RESOLVED:
            tags = {}
            if access_element:
                tags = {k: str(v) for k, v in (access_element[0].get("tags") or {}).items()}
            status, access_why = cg.access_status_of(tags)
            target_projected_for_access = cg.project(target, projection_service)
            projected_access = shapely_wkt.loads(access_geometry.projected_wkt)
            corridors.append(
                RoadCorridor(
                    corridor_id="CORRIDOR_ACCESS_ROAD_MAIN",
                    feature_id=access_geometry.feature_id,
                    corridor_class=CorridorClass.ACCESS_MAIN,
                    access_status=status,
                    distance_to_building_m=round(
                        projected_access.distance(target_projected_for_access), 2
                    ),
                    osm_tags=tags,
                    rationale=(
                        "voie d'accès déclarée au manifeste de site ; "
                        f"accès : {access_why}"
                    ),
                    admissible_for_building=False,
                )
            )

    # --- réseau routier ------------------------------------------------------
    if roads_error:
        report.unresolved.append(
            {
                "feature_id": "ROAD_NETWORK",
                "reason": f"interrogation en échec : {roads_error}",
                "note": "une panne ne prouve aucune absence de route",
            }
        )
    else:
        parking_shape = next(
            (
                shapely_wkt.loads(g.projected_wkt)
                for g in geometries
                if g.feature_id == "HOTEL_PARKING" and g.projected_wkt
            ),
            None,
        )
        new_geometries, new_corridors = _resolve_roads(
            roads or [], target_projected, parking_shape, access_road_ref,
            roads_snapshot.snapshot_id, projection_service, adjacency_max_m,
        )
        geometries.extend(new_geometries)
        corridors.extend(new_corridors)

    # --- obstacles -----------------------------------------------------------
    geometries.extend(
        _resolve_obstacles(
            elements, target, target_ref, cache_snapshot.snapshot_id, report,
            projection_service,
        )
    )

    manifest.geometries = geometries
    manifest.corridors = corridors

    # --- rapport -------------------------------------------------------------
    for geometry in geometries:
        if geometry.resolution_status.value == "resolved":
            report.resolved[geometry.role.value] = report.resolved.get(geometry.role.value, 0) + 1
        else:
            report.unresolved.append(
                {
                    "feature_id": geometry.feature_id,
                    "role": geometry.role.value,
                    "reason": geometry.unresolved_reason,
                }
            )
    report.corridors = manifest.corridors_by_class()
    report.corridor_details = [json.loads(c.model_dump_json()) for c in corridors]
    report.crs_problems = cg.verify(manifest, projection_service)
    report.road_geometry_digest = cg.digest_of(
        [g for g in geometries if g.role in (GeometryRole.ACCESS_ROAD, GeometryRole.ROAD_CANDIDATE)]
    )
    report.obstacle_geometry_digest = cg.digest_of(
        [g for g in geometries if g.role is GeometryRole.OBSTACLE_BUILDING]
    )
    report.manifest_digest = hashlib.sha256(
        manifest.model_dump_json().encode("utf-8")
    ).hexdigest()[:16]

    return manifest, report


def _ref_of_wkt(elements: list[dict], target) -> str | None:  # noqa: ANN001
    """Retrouve la référence OSM d'une empreinte, par recouvrement de formes."""
    best, best_ratio = None, 0.0
    for element in elements:
        shape = cg.shape_of(element)
        if shape is None or shape.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        if not shape.intersects(target):
            continue
        ratio = shape.intersection(target).area / max(target.area, 1e-12)
        if ratio > best_ratio:
            best, best_ratio = element, ratio
    # Un recouvrement partiel ne suffit pas : deux bâtiments mitoyens se
    # touchent, et confondre la cible avec son voisin la ferait disparaître des
    # obstacles.
    return f"way/{best['id']}" if best is not None and best_ratio >= 0.8 else None


def _element_by_ref(elements: list[dict], ref: str | None) -> dict | None:
    if not ref:
        return None
    wanted = ref.split("/")[-1]
    return next((e for e in elements if str(e.get("id")) == wanted), None)


def _resolve_access_road(
    ref: str, element: list[dict] | None, error: str | None,
    snapshot_id: str | None, projection_service,
) -> ResolvedGeometry:
    """Résout la voie d'accès nommée par le manifeste de site.

    Elle ne figure pas au cache de collecte, qui ne contient que bâtiments et
    stationnements : l'y chercher aurait produit une absence inventée.
    """
    if error:
        return cg.unresolved(
            "ACCESS_ROAD_MAIN", GeometryRole.ACCESS_ROAD,
            f"interrogation en échec : {error} — l'objet reste inféré au "
            "manifeste de site, seule sa géométrie manque",
            source_ref=ref, snapshot_id=snapshot_id,
        )

    found = (element or [None])[0] if element else None
    if found is None:
        return cg.unresolved(
            "ACCESS_ROAD_MAIN", GeometryRole.ACCESS_ROAD,
            f"{ref} inconnu de la source — l'objet reste inféré, sa géométrie non",
            source_ref=ref, snapshot_id=snapshot_id,
        )

    shape = cg.shape_of(found)
    if shape is None:
        return cg.unresolved(
            "ACCESS_ROAD_MAIN", GeometryRole.ACCESS_ROAD,
            f"{ref} sans géométrie exploitable dans la réponse",
            source_ref=ref, snapshot_id=snapshot_id,
        )

    tags = found.get("tags") or {}
    if shape.geom_type not in ("LineString", "MultiLineString"):
        return cg.unresolved(
            "ACCESS_ROAD_MAIN", GeometryRole.ACCESS_ROAD,
            f"{ref} rendu comme {shape.geom_type}, incompatible avec une voie",
            source_ref=ref, snapshot_id=snapshot_id,
        )

    status, why = cg.access_status_of(tags)
    caveats = []
    if status is AccessStatus.PRIVATE:
        caveats.append(
            f"accès interdit ({why}) : la voie porte la géométrie, pas "
            "l'autorisation d'y capturer"
        )
    elif status is AccessStatus.RESTRICTED:
        caveats.append(
            f"accès conditionnel ({why}) : une capture autorisée par "
            "l'établissement y reste envisageable, ce n'est pas une interdiction"
        )
    return cg.resolved_from(
        "ACCESS_ROAD_MAIN", GeometryRole.ACCESS_ROAD, ref, snapshot_id, shape,
        "résolution explicite de la voie citée par le manifeste de site",
        [f"tags OSM : {json.dumps(tags, ensure_ascii=False)}"],
        projection_service=projection_service,
        caveats=caveats,
    )


def _resolve_roads(
    roads: list[dict], target_projected, parking_projected, access_ref: str | None,
    snapshot_id: str, projection_service,
    adjacency_max_m: float = cg.DEFAULT_ADJACENCY_M,
) -> tuple[list[ResolvedGeometry], list[RoadCorridor]]:
    geometries: list[ResolvedGeometry] = []
    corridors: list[RoadCorridor] = []

    for element in roads:
        ref = f"way/{element.get('id')}"
        if ref == access_ref:
            # Déjà résolue par son interrogation propre.
            continue
        shape = cg.shape_of(element)
        if shape is None or shape.geom_type not in ("LineString", "MultiLineString"):
            continue

        projected = cg.project(shape, projection_service)
        distance = projected.distance(target_projected)
        parking_distance = (
            projected.distance(parking_projected) if parking_projected is not None else None
        )
        corridor_class, why = cg.classify(
            element, distance, parking_distance, access_ref, adjacency_max_m
        )
        tags = {k: str(v) for k, v in (element.get("tags") or {}).items()}
        status, access_why = cg.access_status_of(tags)

        feature_id = f"ROAD_{element['id']}"
        geometries.append(
            cg.resolved_from(
                feature_id, GeometryRole.ROAD_CANDIDATE, ref, snapshot_id, shape,
                "voie du réseau routier interrogé dans le périmètre",
                [f"tags OSM : {json.dumps(tags, ensure_ascii=False)}"],
                projection_service=projection_service,
            )
        )
        corridors.append(
            RoadCorridor(
                corridor_id=f"CORRIDOR_{element['id']}",
                feature_id=feature_id,
                corridor_class=corridor_class,
                access_status=status,
                distance_to_building_m=round(distance, 2),
                distance_to_parking_m=(
                    round(parking_distance, 2) if parking_distance is not None else None
                ),
                osm_tags=tags,
                rationale=f"{why} ; accès : {access_why}",
                # Aucune admissibilité n'est accordée ici : elle viendra de la
                # visibilité multi-rayons, pas de la proximité.
                admissible_for_building=False,
            )
        )

    return geometries, corridors


def _resolve_obstacles(
    elements: list[dict], target, target_ref: str | None, snapshot_id: str,
    report: cg.ResolutionReport, projection_service,
) -> list[ResolvedGeometry]:
    """Bâtiments voisins susceptibles de masquer la cible.

    La cible en est retirée explicitement : l'oublier ferait rejeter toutes les
    vues pour occlusion par le bâtiment qu'on cherche à voir.
    """
    obstacles: list[ResolvedGeometry] = []

    for element in elements:
        tags = element.get("tags") or {}
        if "building" not in tags:
            continue
        ref = f"way/{element.get('id')}"
        if target_ref and ref == target_ref:
            report.target_exclusions.append(
                {
                    "source_ref": ref, "method": "exact_source_ref",
                    "overlap_ratio": None, "target_ref": target_ref,
                }
            )
            continue

        shape = cg.shape_of(element)
        if shape is None or shape.geom_type not in ("Polygon", "MultiPolygon"):
            report.invalid_geometries.append(f"{ref} : contour inexploitable")
            continue
        if not shape.is_valid:
            report.invalid_geometries.append(f"{ref} : polygone invalide")
            continue
        ratio = shape.intersection(target).area / max(target.area, 1e-12)
        if shape.equals(target):
            report.target_exclusions.append(
                {
                    "source_ref": ref, "method": "equals", "overlap_ratio": 1.0,
                    "target_ref": target_ref,
                }
            )
            continue
        if ratio >= 0.8:
            # Même empreinte que la cible sous une autre référence. Le repli
            # est nécessaire, mais une empreinte voisine retirée par erreur
            # serait autrement invisible.
            report.target_exclusions.append(
                {
                    "source_ref": ref, "method": "overlap_fallback",
                    "overlap_ratio": round(ratio, 4), "target_ref": target_ref,
                }
            )
            continue

        height, source = _height_of(tags)
        obstacles.append(
            cg.resolved_from(
                f"OBSTACLE_{element['id']}", GeometryRole.OBSTACLE_BUILDING, ref,
                snapshot_id, shape,
                "bâtiment voisin du cache d'éléments",
                [f"tags OSM : {json.dumps(tags, ensure_ascii=False)}"],
                projection_service=projection_service,
                height_known=height is not None,
                height_m=height,
                height_source=source,
                caveats=(
                    []
                    if height is not None
                    else ["hauteur inconnue : l'occultation restera un risque, non un fait"]
                ),
            )
        )

    return obstacles


def _height_of(tags: dict) -> tuple[float | None, str | None]:
    """Hauteur d'un bâtiment, si la source la donne. Jamais estimée."""
    raw = tags.get("height") or tags.get("building:height")
    if raw:
        try:
            return float(str(raw).split()[0]), "tag OSM height"
        except ValueError:
            return None, None
    levels = tags.get("building:levels")
    if levels:
        try:
            # Une hauteur par étage serait une invention : on garde le compte
            # d'étages comme source, sans le convertir en mètres.
            float(levels)
        except ValueError:
            return None, None
    return None, None
