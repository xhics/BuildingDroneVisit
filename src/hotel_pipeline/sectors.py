"""Secteur observé du bâtiment (Lot 1B §4, §11).

Un azimut d'observation dit **d'où** l'on regarde. Il ne dit pas **quoi** l'on
regarde tant que l'orientation de la façade avant est inconnue : sans elle,
`front` et `rear` ne sont pas distinguables.

L'azimut avant est donc déduit de preuves déterministes — le stationnement de
l'hôtel, à défaut la voie d'accès — et la méthode employée est enregistrée
avec la valeur. Faute de preuve, le secteur reste `unknown` plutôt que d'être
supposé.
"""

from __future__ import annotations

from dataclasses import dataclass

from shapely import wkt as shapely_wkt

from .logging import get_logger
from .schemas import ViewSector
from .visibility import angular_difference, bearing_deg

log = get_logger("sectors")

#: Centre angulaire de chaque secteur, mesuré depuis la façade avant.
#: Les huit zones couvrent 360° sans trou : tout azimut d'observation tombe
#: dans l'une d'elles, et le secteur retenu est celui dont le centre est le
#: plus proche.
#:
#: `TRANSITION` n'y figure pas volontairement. Dans le Lot 1B, c'est une
#: notion sémantique — une vue reliant route, entrée et stationnement — et non
#: une plage angulaire. La géométrie ne peut donc pas la produire.
SECTOR_CENTRES: tuple[tuple[float, ViewSector], ...] = (
    (0.0, ViewSector.FRONT),
    (45.0, ViewSector.FRONT_RIGHT_CORNER),
    (90.0, ViewSector.RIGHT),
    (135.0, ViewSector.REAR_RIGHT_CORNER),
    (180.0, ViewSector.REAR),
    (225.0, ViewSector.REAR_LEFT_CORNER),
    (270.0, ViewSector.LEFT),
    (315.0, ViewSector.FRONT_LEFT_CORNER),
)


@dataclass
class FrontAzimuth:
    degrees: float
    method: str


def front_from_parking(building_wkt: str, parking_wkt: str) -> FrontAzimuth:
    """La façade avant regarde le stationnement de l'hôtel.

    C'est la preuve la plus solide disponible sans localiser l'entrée : un
    hôtel de bord d'autoroute ouvre sur son propre stationnement.
    """
    building = shapely_wkt.loads(building_wkt)
    parking = shapely_wkt.loads(parking_wkt)
    origin, target = building.centroid, parking.centroid
    return FrontAzimuth(
        bearing_deg(origin.y, origin.x, target.y, target.x), "centroïde du stationnement hôtel"
    )


def front_from_access(building_wkt: str, access_lat: float, access_lon: float) -> FrontAzimuth:
    """Repli : la façade avant regarde la voie d'accès la plus proche."""
    building = shapely_wkt.loads(building_wkt)
    origin = building.centroid
    return FrontAzimuth(
        bearing_deg(origin.y, origin.x, access_lat, access_lon), "voie d'accès la plus proche"
    )


def sector_for(observer_bearing_deg: float, front_azimuth_deg: float) -> ViewSector:
    """Traduit un azimut d'observation en secteur du bâtiment.

    `observer_bearing_deg` est l'azimut du bâtiment **vers** l'observateur :
    se tenir dans la direction de la façade avant, c'est la voir de face.
    """
    offset = (observer_bearing_deg - front_azimuth_deg + 360.0) % 360.0

    # Convention : à mesure que l'azimut croît depuis la face avant, on longe
    # le côté droit du bâtiment vu depuis l'extérieur.
    _, sector = min(
        SECTOR_CENTRES, key=lambda entry: angular_difference(offset, entry[0])
    )
    return sector


def resolve_front(spatial, elements: list[dict]) -> FrontAzimuth | None:  # noqa: ANN001
    """Détermine l'azimut avant à partir du manifeste spatial.

    Retourne `None` si aucune preuve ne le permet — auquel cas les secteurs
    resteront `unknown`, ce qui est préférable à une orientation inventée.
    """
    if spatial.front_azimuth_deg is not None:
        return FrontAzimuth(spatial.front_azimuth_deg, spatial.front_azimuth_method or "fourni")

    building = spatial.candidate(spatial.confirmed_building_id or "")
    if building is None:
        return None

    if spatial.parking_feature_id:
        from .resolve import _to_polygon

        for element in elements:
            if f"{element['type']}/{element['id']}" == spatial.parking_feature_id:
                polygon = _to_polygon(element)
                if polygon is not None:
                    return front_from_parking(building.wkt, polygon.wkt)

    log.warning("aucune preuve d'orientation de façade — secteurs laissés inconnus")
    return None
