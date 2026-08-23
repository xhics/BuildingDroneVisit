"""Surfaces au sol d'un site : pelouse, asphalte, plantations.

Le LiDAR dit **où quelque chose s'élève**, jamais de quoi le sol est fait : ses
points de classe 2 ne distinguent pas une pelouse d'un stationnement. Or un plan
d'établissement montre surtout du sol, et un générateur laissé libre couvrira
d'asphalte une pelouse ou l'inverse.

OpenStreetMap porte cette information, mais le pipeline ne la demandait pas :
sur ce pilote, cinquante-six éléments collectés ne contenaient pas un seul tag
`landuse`, `natural` ou `leisure`. Ce module lit ce que la requête élargie
rapporte et le range par **nature de surface**, sans rien inventer là où la
carte se tait.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..logging import get_logger

log = get_logger("conditioning-ground")

#: Nature d'une surface, du point de vue du rendu. Le vocabulaire n'est pas
#: botanique : il dit ce qu'un générateur doit peindre.
SURFACE_KINDS: tuple[str, ...] = (
    "pelouse",
    "boise",
    "eau",
    "mineral",
    "inconnu",
)

#: Correspondance des tags OSM vers ces natures. Un tag absent ne devient pas
#: « pelouse par défaut » : il devient `inconnu`, et le rendu s'en abstient.
TAG_KINDS: dict[tuple[str, str], str] = {
    ("landuse", "grass"): "pelouse",
    ("landuse", "meadow"): "pelouse",
    ("landuse", "village_green"): "pelouse",
    ("landuse", "greenfield"): "pelouse",
    ("landuse", "farmland"): "pelouse",
    ("landuse", "forest"): "boise",
    ("natural", "wood"): "boise",
    ("natural", "scrub"): "boise",
    ("natural", "grassland"): "pelouse",
    ("natural", "tree_row"): "boise",
    ("natural", "water"): "eau",
    ("natural", "wetland"): "eau",
    ("leisure", "garden"): "pelouse",
    ("leisure", "park"): "pelouse",
    ("leisure", "pitch"): "pelouse",
    ("leisure", "golf_course"): "pelouse",
    ("leisure", "playground"): "mineral",
    ("surface", "grass"): "pelouse",
    ("surface", "dirt"): "mineral",
    ("surface", "gravel"): "mineral",
    ("surface", "paving_stones"): "mineral",
    ("surface", "concrete"): "mineral",
    ("surface", "asphalt"): "mineral",
    ("amenity", "parking"): "mineral",
}

#: Hauteur attribuée à un arbre isolé cartographié comme simple nœud, faute de
#: mesure. Elle est déclarée comme hypothèse, jamais présentée comme relevée.
LONE_TREE_HEIGHT_M = 8.0
LONE_TREE_RADIUS_M = 3.0


@dataclass
class GroundSurface:
    """Une emprise au sol, et la nature que ses tags établissent."""

    feature_id: str
    kind: str
    #: Contour en CRS projeté, fermé.
    ring: list[tuple[float, float]]
    tags: dict[str, str] = field(default_factory=dict)

    def area_m2(self) -> float:
        from shapely.geometry import Polygon

        if len(self.ring) < 4:
            return 0.0
        polygon = Polygon(self.ring)
        return float(polygon.area) if polygon.is_valid else 0.0

    def as_dict(self) -> dict:
        return {
            "feature_id": self.feature_id,
            "kind": self.kind,
            "area_m2": round(self.area_m2(), 1),
            "vertices": len(self.ring),
            "tags": self.tags,
        }


@dataclass
class LoneTree:
    """Un arbre cartographié comme nœud isolé."""

    feature_id: str
    position: tuple[float, float]
    height_m: float = LONE_TREE_HEIGHT_M
    radius_m: float = LONE_TREE_RADIUS_M
    height_assumed: bool = True

    def as_dict(self) -> dict:
        return {
            "feature_id": self.feature_id,
            "position": [round(c, 2) for c in self.position],
            "height_m": self.height_m,
            "radius_m": self.radius_m,
            "height_assumed": self.height_assumed,
        }


@dataclass
class GroundCover:
    """Les surfaces d'un site, rangées par nature."""

    hotel_id: str
    surfaces: list[GroundSurface] = field(default_factory=list)
    trees: list[LoneTree] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    def near(self, centre: tuple[float, float], radius_m: float) -> list[GroundSurface]:
        """Surfaces réellement susceptibles d'entrer dans le cadre.

        Mesuré sur ce pilote : la pelouse la plus proche cartographiée est à
        près de trois cents mètres, et n'apparaîtra sur aucun plan de
        l'établissement. Compter sa superficie dans un bilan de site donnerait
        l'illusion d'un environnement verdoyant que la caméra ne verra jamais.
        """
        kept = []
        for surface in self.surfaces:
            if not surface.ring:
                continue
            mid_x = sum(p[0] for p in surface.ring) / len(surface.ring)
            mid_y = sum(p[1] for p in surface.ring) / len(surface.ring)
            if not _far(mid_x, mid_y, centre, radius_m):
                kept.append(surface)
        return kept

    def by_kind(self) -> dict[str, float]:
        """Surface cumulée par nature, en mètres carrés."""
        totals: dict[str, float] = {}
        for surface in self.surfaces:
            totals[surface.kind] = totals.get(surface.kind, 0.0) + surface.area_m2()
        return {k: round(v, 1) for k, v in totals.items()}

    def as_dict(self) -> dict:
        return {
            "hotel_id": self.hotel_id,
            "surface_count": len(self.surfaces),
            "tree_count": len(self.trees),
            "area_by_kind_m2": self.by_kind(),
            "provenance": self.provenance,
            "surfaces": [s.as_dict() for s in self.surfaces],
            "trees": [t.as_dict() for t in self.trees],
            "caveats": [
                "une superficie cumulée ne dit pas ce que la caméra verra : "
                "sur ce site, les pelouses cartographiées sont à près de trois "
                "cents mètres, hors de tout plan de l'établissement",
                "la nature d'une surface vient des tags OpenStreetMap : là où "
                "la carte se tait, elle sort `inconnu` plutôt que d'être "
                "supposée en pelouse ou en asphalte",
                "la hauteur d'un arbre isolé est une hypothèse : un nœud OSM "
                "ne porte pas de mesure",
                "une emprise cartographiée n'atteste pas l'état actuel du "
                "terrain, seulement ce qu'un contributeur y a relevé",
            ],
        }


def kind_of(tags: dict) -> str:
    """Nature d'une surface d'après ses tags, `inconnu` à défaut."""
    for key, value in tags.items():
        found = TAG_KINDS.get((key, str(value)))
        if found is not None:
            return found
    return "inconnu"


def _project_ring(
    geometry: list[dict], transformer
) -> list[tuple[float, float]]:
    """Contour d'un élément Overpass, projeté dans le référentiel de travail."""
    ring: list[tuple[float, float]] = []
    for node in geometry:
        lon, lat = node.get("lon"), node.get("lat")
        if lon is None or lat is None:
            continue
        x, y = transformer.transform(lon, lat)
        ring.append((float(x), float(y)))
    if len(ring) >= 3 and ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def from_elements(
    elements: list[dict],
    projected_crs: str,
    hotel_id: str = "unknown",
    centre: tuple[float, float] | None = None,
    radius_m: float = 200.0,
) -> GroundCover:
    """Range des éléments Overpass en surfaces et arbres, projetés.

    Les emprises trop lointaines sont écartées : un parc à trois cents mètres
    ne se voit sur aucun plan d'établissement et alourdirait le rendu sans rien
    contraindre.
    """
    from pyproj import Transformer

    transformer = Transformer.from_crs(
        "EPSG:4326", projected_crs, always_xy=True
    )

    cover = GroundCover(hotel_id=hotel_id)
    skipped = 0

    for element in elements:
        tags = {k: str(v) for k, v in (element.get("tags") or {}).items()}
        kind = kind_of(tags)
        identifier = f"{element.get('type', 'way')}/{element.get('id', '?')}"

        if element.get("type") == "node":
            lon, lat = element.get("lon"), element.get("lat")
            if lon is None or lat is None:
                continue
            x, y = transformer.transform(lon, lat)
            if centre is not None and _far(x, y, centre, radius_m):
                skipped += 1
                continue
            cover.trees.append(LoneTree(feature_id=identifier, position=(float(x), float(y))))
            continue

        ring = _project_ring(element.get("geometry") or [], transformer)
        if len(ring) < 4:
            continue
        if centre is not None:
            mid_x = sum(p[0] for p in ring) / len(ring)
            mid_y = sum(p[1] for p in ring) / len(ring)
            if _far(mid_x, mid_y, centre, radius_m):
                skipped += 1
                continue

        cover.surfaces.append(
            GroundSurface(feature_id=identifier, kind=kind, ring=ring, tags=tags)
        )

    cover.provenance = {
        "projected_crs": projected_crs,
        "elements_received": len(elements),
        "skipped_out_of_range": skipped,
        "radius_m": radius_m,
    }
    log.info(
        "sol : %d surface(s), %d arbre(s) isolé(s), %d écarté(s) hors rayon",
        len(cover.surfaces),
        len(cover.trees),
        skipped,
    )
    return cover


def _far(
    x: float, y: float, centre: tuple[float, float], radius_m: float
) -> bool:
    return ((x - centre[0]) ** 2 + (y - centre[1]) ** 2) ** 0.5 > radius_m


def load(path: Path) -> GroundCover:
    """Relit une couverture de sol publiée."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cover = GroundCover(hotel_id=str(payload.get("hotel_id", "unknown")))
    cover.provenance = payload.get("provenance", {})
    for entry in payload.get("surfaces", []):
        cover.surfaces.append(
            GroundSurface(
                feature_id=entry["feature_id"],
                kind=entry["kind"],
                ring=[tuple(p) for p in entry.get("ring", [])],
                tags=entry.get("tags", {}),
            )
        )
    for entry in payload.get("trees", []):
        cover.trees.append(
            LoneTree(
                feature_id=entry["feature_id"],
                position=tuple(entry["position"]),
                height_m=float(entry.get("height_m", LONE_TREE_HEIGHT_M)),
                radius_m=float(entry.get("radius_m", LONE_TREE_RADIUS_M)),
            )
        )
    return cover
