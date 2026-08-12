"""Résolution de propriété (plan directeur §3, §24 ; complément §4).

Le pipeline propose des candidats classés et vérifie des séparations
géométriques. **Il ne choisit pas.** Le §3 prévient que l'empreinte n'est pas
nommée « hôtel » et qu'un parc-o-bus voisin prête à confusion : le choix final
est humain, et il est persisté pour ne pas être redemandé à chaque exécution.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

from shapely.geometry import Polygon, shape
from shapely.ops import transform

from .logging import get_logger
from .schemas.spatial import (
    BuildingCandidate,
    GeocodeResult,
    GeometricAssertion,
    SpatialManifest,
)

log = get_logger("resolve")

#: Distance maximale entre deux géométries encore considérées comme contiguës.
ADJACENCY_TOLERANCE_M = 15.0

#: Indices textuels d'un parc-o-bus, à distinguer du stationnement de l'hôtel.
PARK_AND_RIDE_HINTS = ("incitatif", "park and ride", "park-and-ride", "stationnement incitatif")

HOTEL_TAG_HINTS = {"hotel", "motel"}


# --- géométrie -----------------------------------------------------------


def _to_polygon(element: dict[str, Any]) -> Polygon | None:
    """Construit un polygone WGS84 depuis un élément Overpass `out geom`."""
    geometry = element.get("geometry")
    if not geometry or len(geometry) < 4:
        return None
    coords = [(node["lon"], node["lat"]) for node in geometry]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    try:
        polygon = Polygon(coords)
    except (ValueError, TypeError):
        return None
    return polygon if polygon.is_valid and not polygon.is_empty else polygon.buffer(0) or None


def _local_metric(polygon: Polygon, origin_lat: float) -> Polygon:
    """Projette en mètres locaux — suffisant à l'échelle d'une parcelle.

    Une projection cartographique complète (pyproj) serait plus rigoureuse ;
    à 500 m de rayon, l'approximation équirectangulaire reste très en dessous
    des tolérances utilisées ici.
    """
    scale_x = 111_320.0 * math.cos(math.radians(origin_lat))
    return transform(lambda x, y, z=None: (x * scale_x, y * 110_540.0), polygon)


def _area_m2(polygon: Polygon, origin_lat: float) -> float:
    return _local_metric(polygon, origin_lat).area


def _distance_m(a: Polygon, b: Polygon, origin_lat: float) -> float:
    return _local_metric(a, origin_lat).distance(_local_metric(b, origin_lat))


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(a))


# --- classement ----------------------------------------------------------


def _score(candidate_tags: dict[str, str], distance_m: float, area_m2: float) -> tuple[float, list[str]]:
    """Score heuristique, borné à [0, 1], avec ses justifications.

    Ce score ordonne l'attention humaine. Il ne décide rien : le §12 du plan
    directeur réserve la décision aux règles et à l'humain.
    """
    score = 0.0
    reasons: list[str] = []

    blob = " ".join(f"{k}={v}" for k, v in candidate_tags.items()).lower()
    if any(hint in blob for hint in HOTEL_TAG_HINTS):
        score += 0.40
        reasons.append("étiquette hôtel/motel présente")
    if "name" in candidate_tags:
        score += 0.05
        reasons.append(f"nommé : {candidate_tags['name']}")

    if distance_m <= 30:
        score += 0.30
        reasons.append(f"à {distance_m:.0f} m du géocodage")
    elif distance_m <= 100:
        score += 0.18
        reasons.append(f"à {distance_m:.0f} m du géocodage")
    elif distance_m <= 250:
        score += 0.07
        reasons.append(f"à {distance_m:.0f} m du géocodage")

    # Un hôtel de 116 chambres occupe une emprise substantielle.
    if 1_500 <= area_m2 <= 12_000:
        score += 0.25
        reasons.append(f"emprise plausible ({area_m2:.0f} m²)")
    elif area_m2 > 12_000:
        reasons.append(f"emprise très grande ({area_m2:.0f} m²) — vérifier")
    else:
        reasons.append(f"emprise faible ({area_m2:.0f} m²)")

    return min(score, 1.0), reasons


def build_candidates(
    elements: Iterable[dict[str, Any]], geocode: GeocodeResult
) -> list[BuildingCandidate]:
    """Transforme les éléments Overpass en candidats classés."""
    candidates: list[BuildingCandidate] = []

    for element in elements:
        tags = element.get("tags") or {}
        if "building" not in tags:
            continue

        polygon = _to_polygon(element)
        if polygon is None:
            continue

        centroid = polygon.centroid
        area = _area_m2(polygon, geocode.lat)
        distance = _haversine_m(geocode.lat, geocode.lon, centroid.y, centroid.x)
        score, reasons = _score(tags, distance, area)

        candidates.append(
            BuildingCandidate(
                feature_id=f"{element['type']}/{element['id']}",
                source="overpass",
                tags={k: str(v) for k, v in tags.items()},
                centroid_lat=centroid.y,
                centroid_lon=centroid.x,
                area_m2=area,
                distance_to_geocode_m=distance,
                wkt=polygon.wkt,
                score=score,
                score_reasons=reasons,
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def parking_features(elements: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in elements if (e.get("tags") or {}).get("amenity") == "parking"]


def looks_like_park_and_ride(tags: dict[str, str]) -> bool:
    if tags.get("park_ride") not in (None, "no"):
        return True
    blob = " ".join(str(v) for v in tags.values()).lower()
    return any(hint in blob for hint in PARK_AND_RIDE_HINTS)


# --- assertions ----------------------------------------------------------


def check_separations(
    manifest: SpatialManifest, elements: list[dict[str, Any]]
) -> list[GeometricAssertion]:
    """Vérifie les séparations exigées par le §3 du plan directeur.

    Ces contrôles n'ont de sens qu'une fois `BUILDING_MAIN` confirmé : sans
    bâtiment de référence, il n'y a rien à séparer.
    """
    assertions: list[GeometricAssertion] = []

    confirmed = manifest.candidate(manifest.confirmed_building_id or "")
    if confirmed is None:
        return [
            GeometricAssertion(
                name="building_confirmed",
                passed=False,
                detail="aucun bâtiment confirmé — séparations non évaluables",
            )
        ]

    from shapely import wkt as shapely_wkt

    building = shapely_wkt.loads(confirmed.wkt)
    origin_lat = confirmed.centroid_lat

    parkings = parking_features(elements)
    hotel_parking = None
    park_and_ride = None

    for element in parkings:
        polygon = _to_polygon(element)
        if polygon is None:
            continue
        tags = {k: str(v) for k, v in (element.get("tags") or {}).items()}
        distance = _distance_m(building, polygon, origin_lat)
        feature_id = f"{element['type']}/{element['id']}"

        if looks_like_park_and_ride(tags):
            if park_and_ride is None or distance < park_and_ride[1]:
                park_and_ride = (feature_id, distance, polygon)
        elif distance <= ADJACENCY_TOLERANCE_M:
            if hotel_parking is None or distance < hotel_parking[1]:
                hotel_parking = (feature_id, distance, polygon)

    if hotel_parking:
        manifest.parking_feature_id = hotel_parking[0]
        assertions.append(
            GeometricAssertion(
                name="parking_adjacent_to_building",
                passed=True,
                detail=f"{hotel_parking[0]} à {hotel_parking[1]:.1f} m du bâtiment",
            )
        )
    else:
        assertions.append(
            GeometricAssertion(
                name="parking_adjacent_to_building",
                passed=False,
                detail=(
                    f"aucun stationnement contigu (tolérance {ADJACENCY_TOLERANCE_M:.0f} m) — "
                    f"{len(parkings)} stationnement(s) examiné(s)"
                ),
            )
        )

    if park_and_ride:
        manifest.park_and_ride_feature_id = park_and_ride[0]
        disjoint = not building.intersects(park_and_ride[2])
        assertions.append(
            GeometricAssertion(
                name="disjoint_from_park_and_ride",
                passed=disjoint,
                detail=(
                    f"{park_and_ride[0]} à {park_and_ride[1]:.1f} m, "
                    f"{'disjoint' if disjoint else 'INTERSECTE le bâtiment'}"
                ),
            )
        )
        if hotel_parking and park_and_ride[0] == hotel_parking[0]:
            assertions.append(
                GeometricAssertion(
                    name="parking_not_park_and_ride",
                    passed=False,
                    detail="le stationnement retenu est aussi identifié comme parc-o-bus",
                )
            )
    else:
        assertions.append(
            GeometricAssertion(
                name="disjoint_from_park_and_ride",
                passed=True,
                detail="aucun parc-o-bus détecté dans le rayon",
            )
        )

    return assertions


def resolve(hotel_id: str, address: str, radius_m: int = 500) -> SpatialManifest:
    """Construit le manifeste spatial. Effectue des appels réseau."""
    from .providers import features_around, geocode as geocode_address

    position = geocode_address(address)
    log.info("adresse résolue par %s : %.6f, %.6f", position.provider, position.lat, position.lon)

    elements = features_around(position.lat, position.lon, radius_m)
    candidates = build_candidates(elements, position)
    log.info("%d bâtiment(s) candidat(s) retenu(s)", len(candidates))

    return SpatialManifest(
        hotel_id=hotel_id,
        address=address,
        geocode=position,
        search_radius_m=radius_m,
        candidates=candidates,
    )
