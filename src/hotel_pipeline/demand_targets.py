"""Ce qu'un besoin vise réellement, et depuis où (collecte V2, durcissement).

`measure_all` calculait **une** géométrie — sur le bâtiment principal — puis la
recopiait à tous les besoins. Un besoin d'entrée était donc mesuré contre la
façade entière, et un besoin de façade arrière contre la même empreinte que la
façade avant : deux besoins opposés recevaient la même réponse.

Ce module résout la cible **par besoin**, à partir du manifeste géométrique et
du manifeste de site. Trois natures de cible, trois résolutions :

```text
site_object       l'empreinte de l'objet nommé
view_sector       la face du bâtiment, et le demi-plan d'où on la voit
context_corridor  la voie, dont on veut documenter la traversée
```

Une cible non résolue n'est jamais remplacée par le bâtiment principal : ce
serait exactement le défaut corrigé, avec un repli à la place d'une copie.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .logging import get_logger
from .schemas.acquisition import TargetKind
from .schemas.enums import ViewSector
from .schemas.geometry import GeometryRole

log = get_logger("demand-targets")


class TargetUnresolved(RuntimeError):
    """La cible d'un besoin n'a pas été trouvée, et rien ne l'a remplacée."""


#: Azimut d'observation attendu, par secteur, **relatif à la façade avant**.
#: Le secteur ne dit pas où est la caméra dans l'absolu : il dit de quel côté
#: du bâtiment elle regarde, et l'orientation de la façade vient du site.
SECTOR_BEARINGS: dict[str, float] = {
    ViewSector.FRONT.value: 0.0,
    ViewSector.FRONT_RIGHT_CORNER.value: 45.0,
    ViewSector.RIGHT.value: 90.0,
    ViewSector.REAR_RIGHT_CORNER.value: 135.0,
    ViewSector.REAR.value: 180.0,
    ViewSector.REAR_LEFT_CORNER.value: 225.0,
    ViewSector.LEFT.value: 270.0,
    ViewSector.FRONT_LEFT_CORNER.value: 315.0,
}

#: Valeur de repli, employée seulement si aucune politique n'est fournie. Le
#: seuil réel vient de `policy.geometry.sector_observer_half_width_deg` : le
#: tenir en double ici garantissait qu'un jour les deux divergent.
DEFAULT_SECTOR_HALF_WIDTH_DEG = 67.5


@dataclass(frozen=True)
class DemandTarget:
    """La cible d'un besoin, et ce qu'elle attend de l'observateur."""

    demand_id: str
    shape: object

    #: Azimut **depuis lequel** la cible doit être vue, en degrés absolus.
    #: `None` quand le besoin n'impose aucun côté — un corridor se documente
    #: d'où l'on veut.
    required_bearing_deg: float | None = None
    half_width_deg: float = DEFAULT_SECTOR_HALF_WIDTH_DEG
    description: str = ""

    def observer_is_admissible(self, observer_bearing_deg: float) -> bool:
        """La caméra regarde-t-elle depuis le côté demandé ?

        `observer_bearing_deg` est l'azimut de la caméra **vue depuis la
        cible** : c'est de là qu'on regarde, et c'est ce que le secteur nomme.
        """
        if self.required_bearing_deg is None:
            return True
        gap = abs(
            (observer_bearing_deg - self.required_bearing_deg + 180.0) % 360.0 - 180.0
        )
        return gap <= self.half_width_deg


def resolve(
    demand,  # noqa: ANN001 — CaptureDemand
    manifest,  # noqa: ANN001 — CaptureGeometryManifest
    front_azimuth_deg: float | None = None,
    site=None,  # noqa: ANN001 — SiteManifest
    half_width_deg: float = DEFAULT_SECTOR_HALF_WIDTH_DEG,
) -> DemandTarget:
    """Résout ce que ce besoin vise, sans jamais retomber sur le bâtiment.

    `front_azimuth_deg` oriente les secteurs : sans lui, « avant » et
    « arrière » ne se distinguent pas, et prétendre le contraire ferait passer
    n'importe quelle vue pour une vue de façade.
    """
    from shapely import wkt as shapely_wkt

    from .schemas.geometry import GeometryResolutionStatus

    resolved = {
        geometry.feature_id: geometry
        for geometry in manifest.geometries
        if geometry.resolution_status is GeometryResolutionStatus.RESOLVED
    }

    if demand.target_kind is TargetKind.CONTEXT_CORRIDOR:
        geometry = resolved.get(demand.target_ref)
        if geometry is None:
            raise TargetUnresolved(
                f"{demand.demand_id} : corridor {demand.target_ref!r} non résolu"
            )
        return DemandTarget(
            demand_id=demand.demand_id,
            shape=shapely_wkt.loads(geometry.projected_wkt),
            description=f"corridor {demand.target_ref}",
        )

    if demand.target_kind is TargetKind.SITE_OBJECT:
        geometry = (
            resolved.get(demand.target_ref)
            or _by_site_object(demand.target_ref, resolved, site)
            or _by_declared_role(demand.target_ref, resolved)
        )
        if geometry is None:
            raise TargetUnresolved(
                f"{demand.demand_id} : objet de site {demand.target_ref!r} sans "
                "géométrie résolue — il n'est pas remplacé par le bâtiment"
            )
        return DemandTarget(
            demand_id=demand.demand_id,
            shape=shapely_wkt.loads(geometry.projected_wkt),
            description=f"objet {demand.target_ref}",
        )

    if demand.target_kind is TargetKind.VIEW_SECTOR:
        building = next(
            (
                geometry for geometry in resolved.values()
                if geometry.role is GeometryRole.TARGET_BUILDING
            ),
            None,
        )
        if building is None:
            raise TargetUnresolved(
                f"{demand.demand_id} : aucune empreinte cible résolue"
            )
        if front_azimuth_deg is None:
            raise TargetUnresolved(
                f"{demand.demand_id} : orientation de façade inconnue — sans "
                "elle, « avant » et « arrière » ne se distinguent pas"
            )
        offset = SECTOR_BEARINGS.get(demand.target_ref)
        if offset is None:
            raise TargetUnresolved(
                f"{demand.demand_id} : secteur {demand.target_ref!r} sans "
                "azimut déclaré"
            )
        return DemandTarget(
            demand_id=demand.demand_id,
            shape=shapely_wkt.loads(building.projected_wkt),
            required_bearing_deg=(front_azimuth_deg + offset) % 360.0,
            half_width_deg=half_width_deg,
            description=f"secteur {demand.target_ref}",
        )

    if demand.target_kind is TargetKind.TRANSITION:
        # Une transition relie deux objets — route à entrée, entrée à
        # stationnement — et n'a pas d'empreinte propre. Tant qu'aucune
        # géométrie ne la porte, elle est **explicitement** non résolue : lui
        # prêter l'empreinte du bâtiment ferait juger la transition sur la
        # façade.
        geometry = resolved.get(demand.target_ref)
        if geometry is None:
            raise TargetUnresolved(
                f"{demand.demand_id} : transition {demand.target_ref!r} sans "
                "géométrie propre — elle relie deux objets et ne se mesure pas "
                "sur l'empreinte de l'un d'eux"
            )
        return DemandTarget(
            demand_id=demand.demand_id,
            shape=shapely_wkt.loads(geometry.projected_wkt),
            description=f"transition {demand.target_ref}",
        )

    raise TargetUnresolved(
        f"{demand.demand_id} : nature de cible {demand.target_kind.value!r} "
        "sans résolution déclarée"
    )


def _by_site_object(object_id: str, resolved: dict, site) -> object | None:  # noqa: ANN001
    """Géométrie citée par un objet du site, quand la référence est indirecte."""
    if site is None:
        return None
    for instance in getattr(site, "objects", []):
        if getattr(instance, "object_id", None) != object_id:
            continue
        for ref in getattr(instance, "geometry_refs", []) or []:
            if ref in resolved:
                return resolved[ref]
    return None


#: Correspondance **déclarée** entre un type d'objet de site et le rôle de la
#: géométrie qui le porte. Le stationnement de l'hôtel s'appelle
#: `PARKING_HOTEL` au manifeste de site et `HOTEL_PARKING` au rôle de
#: géométrie : deux vocabulaires pour une chose. Sans cette table, le besoin
#: cherchait un identifiant inexistant, ne trouvait rien, et se rabattait
#: implicitement sur la position du bâtiment — le repli que le contrat refuse.
#:
#: Une table, non une heuristique de nom : deviner par permutation de mots
#: marcherait ici et échouerait au premier type dont les deux noms diffèrent
#: vraiment.
OBJECT_KIND_ROLES: dict[str, GeometryRole] = {
    "PARKING_HOTEL": GeometryRole.HOTEL_PARKING,
    "ACCESS_ROAD_MAIN": GeometryRole.ACCESS_ROAD,
}


def _by_declared_role(object_kind: str, resolved: dict) -> object | None:
    """Géométrie portant le rôle déclaré pour ce type d'objet."""
    role = OBJECT_KIND_ROLES.get(object_kind)
    if role is None:
        return None
    return next(
        (geometry for geometry in resolved.values() if geometry.role is role),
        None,
    )


def observer_bearing(origin, target_shape) -> float:  # noqa: ANN001
    """Azimut de l'observateur **vu depuis la cible**.

    C'est ce que nomme un secteur : « façade avant » désigne le côté d'où l'on
    regarde, non la direction dans laquelle pointe l'objectif.
    """
    centre = target_shape.centroid
    return math.degrees(math.atan2(origin[0] - centre.x, origin[1] - centre.y)) % 360.0
