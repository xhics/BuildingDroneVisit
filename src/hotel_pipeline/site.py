"""Instanciation du site depuis le gabarit (Lot 1B §4).

Construit un `SiteManifest` à partir de ce qui a été réellement établi : le
manifeste spatial, les éléments Overpass et les assets qualifiés.

Le principe est constant : **observer, mesurer ou marquer inconnu — jamais
inventer**. Chaque type du gabarit produit une instance, même quand rien ne
permet de la remplir ; elle porte alors son motif d'indétermination.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .logging import get_logger
from .schemas import ObjectState, Subject
from .schemas.critical_objects import REQUIRED_OBJECTS
from .schemas.site import SiteManifest, SiteObject, SiteRelation
from .visibility import haversine_m

log = get_logger("site")

#: Type d'objet porté par une empreinte de stationnement voisine à distinguer.
PARK_AND_RIDE = "PARK_AND_RIDE"
NEIGHBOURING = "NEIGHBOURING_ACCOMMODATION"

#: Types que seules des données non encore acquises peuvent établir. Les
#: instancier quand même rend l'attente visible.
PENDING_SOURCES: dict[str, str] = {
    "PROPERTY_PARCEL": "cadastre non acquis",
    "ROOFLINE_MAIN": "LiDAR non acquis (MNS − MNT)",
    "TERRAIN_MAIN": "MNT non acquis",
    "ENTRANCE_MAIN_CURRENT": "entrée non localisée sur l'empreinte",
    "DRIVEWAY_MAIN": "accès véhicule non tracé",
    "FACADE_PRIMARY": "façades non découpées sur l'empreinte",
    "FACADE_LEFT": "façades non découpées sur l'empreinte",
    "FACADE_RIGHT": "façades non découpées sur l'empreinte",
    "FACADE_REAR": "façades non découpées sur l'empreinte",
}


@dataclass
class SiteReport:
    created: int = 0
    confirmed: int = 0
    unresolved: int = 0
    excluded: int = 0
    relations: int = 0
    reasons: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "objects": self.created,
            "confirmed": self.confirmed,
            "unresolved": self.unresolved,
            "excluded_instances": self.excluded,
            "relations": self.relations,
            "unresolved_reasons": self.reasons,
        }


def object_id(hotel_id: str, kind: str) -> str:
    """Identifiant stable, indépendant des sources."""
    return f"{hotel_id}:{kind}"


def _feature(elements: list[dict[str, Any]], feature_id: str | None) -> dict | None:
    if not feature_id:
        return None
    return next(
        (e for e in elements if f"{e.get('type')}/{e.get('id')}" == feature_id), None
    )


def _geometry(element: dict | None) -> tuple[str | None, float | None, float | None]:
    from .resolve import _to_polygon

    if element is None:
        return None, None, None
    polygon = _to_polygon(element)
    if polygon is None:
        return None, None, None
    centroid = polygon.centroid
    return polygon.wkt, centroid.y, centroid.x


def _nearest_road(elements: list[dict[str, Any]], lat: float, lon: float) -> dict | None:
    """Voie carrossable la plus proche du bâtiment confirmé."""
    best, best_distance = None, float("inf")
    for element in elements:
        if "highway" not in (element.get("tags") or {}):
            continue
        for node in element.get("geometry") or []:
            distance = haversine_m(lat, lon, node["lat"], node["lon"])
            if distance < best_distance:
                best, best_distance = element, distance
    return best


def build(
    hotel_id: str,
    spatial,  # noqa: ANN001 — SpatialManifest
    elements: list[dict[str, Any]],
    roads: list[dict[str, Any]] | None = None,
    assets: list | None = None,  # noqa: ANN001
) -> tuple[SiteManifest, SiteReport]:
    """Instancie le site. Ne crée jamais une géométrie qui n'existe pas."""
    report = SiteReport()
    objects: list[SiteObject] = []

    def add(obj: SiteObject) -> SiteObject:
        objects.append(obj)
        report.created += 1
        if obj.state is ObjectState.CONFIRMED:
            report.confirmed += 1
        elif obj.state is ObjectState.UNRESOLVED:
            report.unresolved += 1
            if obj.unresolved_reason:
                report.reasons[obj.kind] = obj.unresolved_reason
        return obj

    building_uid = object_id(hotel_id, "BUILDING_MAIN")
    parking_uid = object_id(hotel_id, "PARKING_HOTEL")
    road_uid = object_id(hotel_id, "ACCESS_ROAD_MAIN")
    sign_uid = object_id(hotel_id, "PROPERTY_SIGN")
    park_ride_uid = object_id(hotel_id, PARK_AND_RIDE)

    # --- BUILDING_MAIN ---------------------------------------------------
    confirmed = spatial.candidate(spatial.confirmed_building_id or "")
    if confirmed is not None:
        add(
            SiteObject(
                object_id=building_uid,
                kind="BUILDING_MAIN",
                state=ObjectState.CONFIRMED,
                source_ref=confirmed.feature_id,
                geometry_wkt=confirmed.wkt,
                centroid_lat=confirmed.centroid_lat,
                centroid_lon=confirmed.centroid_lon,
                evidence=[
                    confirmed.feature_id,
                    *([spatial.geocode.provider] if spatial.geocode else []),
                ],
                confirmed_by=spatial.confirmed_by,
                confirmed_at=spatial.confirmed_at,
                confirmation_rationale=spatial.confirmation_rationale,
            )
        )
    else:
        add(
            SiteObject(
                object_id=building_uid,
                kind="BUILDING_MAIN",
                unresolved_reason="aucun bâtiment confirmé",
            )
        )

    # --- PARKING_HOTEL, relié au bâtiment --------------------------------
    parking_element = _feature(elements, spatial.parking_feature_id)
    wkt, lat, lon = _geometry(parking_element)
    if wkt:
        relations = [
            SiteRelation(
                predicate="adjacent_to",
                target_id=building_uid,
                evidence=[a.detail for a in spatial.assertions if "parking" in a.name],
            )
        ]
        if spatial.park_and_ride_feature_id:
            relations.append(SiteRelation(predicate="distinct_from", target_id=park_ride_uid))
        add(
            SiteObject(
                object_id=parking_uid,
                kind="PARKING_HOTEL",
                state=ObjectState.INFERRED,
                source_ref=spatial.parking_feature_id,
                geometry_wkt=wkt,
                centroid_lat=lat,
                centroid_lon=lon,
                relations=relations,
            )
        )
    else:
        add(
            SiteObject(
                object_id=parking_uid,
                kind="PARKING_HOTEL",
                unresolved_reason="aucun stationnement rattaché au bâtiment",
            )
        )

    # --- PARK_AND_RIDE : une instance réelle, pas un mot ------------------
    park_ride_element = _feature(elements, spatial.park_and_ride_feature_id)
    wkt, lat, lon = _geometry(park_ride_element)
    if wkt:
        add(
            SiteObject(
                object_id=park_ride_uid,
                kind=PARK_AND_RIDE,
                state=ObjectState.INFERRED,
                source_ref=spatial.park_and_ride_feature_id,
                geometry_wkt=wkt,
                centroid_lat=lat,
                centroid_lon=lon,
                relations=[
                    SiteRelation(predicate="distinct_from", target_id=building_uid),
                    SiteRelation(predicate="distinct_from", target_id=parking_uid),
                ],
            )
        )
        report.excluded += 1
    else:
        add(
            SiteObject(
                object_id=park_ride_uid,
                kind=PARK_AND_RIDE,
                unresolved_reason=(
                    "aucun parc-o-bus étiqueté dans le rayon — absence non vérifiée, "
                    "pas absence prouvée"
                ),
            )
        )
        report.excluded += 1

    # --- ACCESS_ROAD_MAIN -------------------------------------------------
    road_element = None
    if confirmed is not None:
        road_element = _nearest_road(
            roads or elements, confirmed.centroid_lat, confirmed.centroid_lon
        )
    if road_element is not None:
        tags = road_element.get("tags") or {}
        add(
            SiteObject(
                object_id=road_uid,
                kind="ACCESS_ROAD_MAIN",
                state=ObjectState.INFERRED,
                source_ref=f"{road_element['type']}/{road_element['id']}",
                evidence=[tags.get("name", "voie sans nom"), tags.get("highway", "")],
                relations=[SiteRelation(predicate="serves", target_id=building_uid)],
            )
        )
    else:
        add(
            SiteObject(
                object_id=road_uid,
                kind="ACCESS_ROAD_MAIN",
                unresolved_reason="aucune voie carrossable trouvée près du bâtiment",
            )
        )

    # --- PROPERTY_SIGN : établi par lecture d'enseigne --------------------
    sign_assets = [
        a
        for a in (assets or [])
        if Subject.SIGN in a.subjects and a.property_match_status.value == "match"
    ]
    if sign_assets:
        add(
            SiteObject(
                object_id=sign_uid,
                kind="PROPERTY_SIGN",
                state=ObjectState.INFERRED,
                evidence=[a.id for a in sign_assets[:5]],
                relations=[SiteRelation(predicate="belongs_to", target_id=building_uid)],
            )
        )
    else:
        add(
            SiteObject(
                object_id=sign_uid,
                kind="PROPERTY_SIGN",
                unresolved_reason="aucune enseigne lue et rattachée à l'établissement",
            )
        )

    # --- types en attente d'une source ------------------------------------
    created = {o.kind for o in objects}
    for kind in REQUIRED_OBJECTS:
        if kind in created:
            continue
        relations = (
            [SiteRelation(predicate="part_of", target_id=building_uid)]
            if kind.startswith("FACADE_") or kind in {"ROOFLINE_MAIN", "ENTRANCE_MAIN_CURRENT"}
            else []
        )
        add(
            SiteObject(
                object_id=object_id(hotel_id, kind),
                kind=kind,
                unresolved_reason=PENDING_SOURCES.get(kind, "non établi"),
                relations=relations,
            )
        )

    manifest = SiteManifest(
        hotel_id=hotel_id, objects=objects, built_at=datetime.now(timezone.utc)
    )
    report.relations = sum(len(o.relations) for o in objects)

    log.info(
        "site instancié : %d objet(s), %d confirmé(s), %d indéterminé(s), %d relation(s)",
        report.created,
        report.confirmed,
        report.unresolved,
        report.relations,
    )
    return manifest, report
