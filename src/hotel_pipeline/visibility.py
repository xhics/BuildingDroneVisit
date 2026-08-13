"""Le cliché regarde-t-il le bâtiment ? (plan directeur §11, §14)

Une image « extérieure » n'est pas une image *de l'hôtel* : sur 300 m de
voirie, l'essentiel de l'imagerie de roulage cadre la chaussée, l'autoroute ou
les commerces voisins. La classification sémantique ne sait pas trancher cela.

La géométrie, si. Mapillary et Street View fournissent position et cap : on
peut donc calculer si l'empreinte confirmée tombe dans le champ de la caméra.
C'est un critère exact, sans modèle, et il applique le principe du plan
directeur — la géométrie confirme ce que la sémantique suggère.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shapely import wkt as shapely_wkt
from shapely.geometry import Point

from .logging import get_logger
from .schemas.policy import DEFAULT_POLICY, PipelinePolicy

log = get_logger("visibility")

#: Valeurs de repli, égales à celles de la politique par défaut. Toute fonction
#: publique accepte une `PipelinePolicy` : ces constantes ne servent qu'aux
#: appels internes et aux tests unitaires.
HALF_FOV_DEG = DEFAULT_POLICY.geometry.half_fov_deg
MAX_DISTANCE_M = DEFAULT_POLICY.geometry.max_distance_m


@dataclass
class Visibility:
    visible: bool
    distance_m: float
    offset_deg: float
    reason: str


def bearing_deg(from_lat: float, from_lon: float, to_lat: float, to_lon: float) -> float:
    """Azimut géographique, 0° au nord, sens horaire."""
    phi1, phi2 = math.radians(from_lat), math.radians(to_lat)
    dlambda = math.radians(to_lon - from_lon)

    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def angular_difference(a: float, b: float) -> float:
    """Écart angulaire signé le plus court, en valeur absolue."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(a))


def assess(
    camera_lat: float,
    camera_lon: float,
    heading_deg: float | None,
    building_wkt: str,
    half_fov_deg: float = HALF_FOV_DEG,
    max_distance_m: float = MAX_DISTANCE_M,
) -> Visibility:
    """Le bâtiment est-il dans le champ de la caméra ?

    Le cap est comparé à l'azimut du point de l'empreinte le plus proche, et
    non de son centroïde : de près, le centroïde peut tomber hors champ alors
    que la façade remplit l'image.
    """
    building = shapely_wkt.loads(building_wkt)
    camera = Point(camera_lon, camera_lat)

    from shapely.ops import nearest_points

    nearest = nearest_points(building, camera)[0]
    distance = haversine_m(camera_lat, camera_lon, nearest.y, nearest.x)

    if distance > max_distance_m:
        return Visibility(False, distance, 180.0, f"trop loin ({distance:.0f} m)")

    if heading_deg is None:
        # Sans cap, la proximité seule ne prouve rien : on ne tranche pas.
        return Visibility(False, distance, 180.0, "cap inconnu")

    target = bearing_deg(camera_lat, camera_lon, nearest.y, nearest.x)
    offset = angular_difference(heading_deg, target)

    if offset <= half_fov_deg:
        return Visibility(True, distance, offset, f"dans le champ ({offset:.0f}° d'écart)")
    return Visibility(False, distance, offset, f"hors champ ({offset:.0f}° d'écart)")


def is_occluded(
    camera_lat: float,
    camera_lon: float,
    target_lat: float,
    target_lon: float,
    obstacles: list,  # noqa: ANN001 — polygones Shapely
) -> str | None:
    """Un autre bâtiment coupe-t-il la ligne de visée ?

    Le champ de vision seul est insuffisant : sur 118 vues Street View visant
    l'empreinte, une majorité ne montrait que des pavillons interposés. La
    ligne caméra → cible est donc confrontée aux empreintes voisines.

    Retourne l'identifiant de l'obstacle rencontré, ou `None` si la vue est
    dégagée. Test purement 2D : il ignore la hauteur, et reste donc
    conservateur — il ne signale que ce qu'il peut prouver.
    """
    from shapely.geometry import LineString

    sight = LineString([(camera_lon, camera_lat), (target_lon, target_lat)])
    for identifier, polygon in obstacles:
        if polygon is None or polygon.is_empty:
            continue
        if sight.crosses(polygon) or sight.within(polygon):
            return identifier
    return None


def obstacles_from(elements: list[dict], exclude_id: str) -> list:
    """Empreintes de bâtiments susceptibles de masquer la cible."""
    from .resolve import _to_polygon

    obstacles = []
    for element in elements:
        identifier = f"{element.get('type')}/{element.get('id')}"
        if identifier == exclude_id:
            continue
        if "building" not in (element.get("tags") or {}):
            continue
        polygon = _to_polygon(element)
        if polygon is not None:
            obstacles.append((identifier, polygon))
    return obstacles


#: `annotate()` a été **supprimé**. Il jugeait la visibilité sur un seul rayon
#: vers le point le plus proche de l'empreinte : une tour devant ce point
#: condamnait la vue entière, un hangar masquant les trois quarts de la façade
#: passait inaperçu, et les 29 occultations qu'il avait déclarées se sont
#: toutes révélées non prouvées. La visibilité se produit désormais par
#: `visibility assess`, et s'applique par `visibility apply`.
REPLACED_BY = "hotel_pipeline.geo.visibility_engine"
