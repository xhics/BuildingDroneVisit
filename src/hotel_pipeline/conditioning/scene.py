"""Scène volumique minimale, extrudée depuis le manifeste de capture.

La géométrie utile au conditionnement n'est pas un maillage : c'est un
squelette. Des prismes bien placés valent mieux qu'un maillage détaillé mal
aligné, parce que le générateur n'a besoin que de la parallaxe et des
occultations correctes.

Aucune hauteur n'est inventée en silence. `height_known: false` au manifeste
produit un prisme dont `height_assumed` est vrai et dont la provenance porte la
règle appliquée ; le masque de confiance déclasse ensuite ses pixels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from shapely import wkt as shapely_wkt
from shapely.geometry import Polygon

from ..logging import get_logger

log = get_logger("conditioning-scene")

#: Hauteur retenue faute de mesure, par rôle. Ce n'est pas une donnée du site :
#: c'est une consigne d'animation, tracée comme telle dans chaque prisme.
ASSUMED_HEIGHT_M: dict[str, float] = {
    "target_building": 12.0,
    "obstacle_building": 8.0,
}

TARGET_ROLES = frozenset({"target_building"})
OBSTACLE_ROLES = frozenset({"obstacle_building"})


@dataclass
class Prism:
    """Une emprise extrudée, et le statut de sa hauteur.

    Un bâtiment MultiPolygon produit plusieurs prismes : le principal porte
    le rôle et la cible éventuelle, les volumes secondaires (ailes,
    annexes) restent rattachés à leur bâtiment via ``parent_building_id``
    et vivent dans le même repère projeté — plus aucun n'est jeté.
    """

    feature_id: str
    role: str
    footprint: np.ndarray
    height_m: float
    height_assumed: bool
    height_source: str
    is_target: bool
    roof_vertices: np.ndarray | None = None
    roof_faces: np.ndarray | None = None
    facade_relief: object | None = None
    roof_planes: object | None = None
    geometry_simplification: dict | None = None
    #: Bâtiment parent pour un volume secondaire ; None sur un volume unique.
    parent_building_id: str | None = None
    #: Index de la partie dans le bâtiment parent (0 = principale).
    part_index: int = 0
    #: Maillage canonique partagé : le renderer, le textureur, la collision
    #: et l'export consomment exactement cette instance.
    canonical_mesh: object | None = None

    @property
    def roof_measured(self) -> bool:
        """Le toit vient-il d'une mesure, ou d'une fermeture géométrique ?"""
        return self.roof_faces is not None and len(self.roof_faces) > 0

    @property
    def confidence(self) -> float:
        """Ce que la géométrie de ce volume mérite comme crédit.

        Une hauteur supposée ne disqualifie pas l'emprise, qui elle est
        attestée : elle plafonne le crédit, sans l'annuler.
        """
        if self.height_assumed:
            return 0.45 if self.is_target else 0.30
        return 0.95 if self.is_target else 0.70

    @property
    def roof_confidence(self) -> float:
        """Crédit du toit, distinct de celui des murs.

        Aucune imagerie au sol n'atteste un toit : sans mesure, ces faces sont
        les moins fiables de la scène. Un nDSM aérien les atteste directement,
        et le crédit rejoint alors celui des murs.
        """
        return self.confidence if self.roof_measured else self.confidence * 0.35

    @property
    def roof_provenance_class(self) -> str:
        """Ce que la géométrie du toit prétend être.

        Sans toit mesuré, l'enveloppe plate est une borne de sécurité — elle
        dit UNKNOWN, jamais une forme architecturale inventée.
        """
        if self.roof_measured:
            return "LIDAR_MEASURED"
        return "UNKNOWN_MINIMAL_ENVELOPE"


@dataclass
class ConditioningScene:
    """Les volumes d'un site, dans un référentiel projeté unique."""

    hotel_id: str
    crs: str
    prisms: list[Prism] = field(default_factory=list)
    #: Centroïde de la cible, origine des trajectoires.
    centre: tuple[float, float] = (0.0, 0.0)
    provenance: dict = field(default_factory=dict)

    @property
    def target(self) -> Prism | None:
        for p in self.prisms:
            if p.is_target:
                return p
        return None

    @property
    def assumed_height_count(self) -> int:
        return sum(1 for p in self.prisms if p.height_assumed)

    def buildings(self) -> dict[str, list["Prism"]]:
        """Graphe de scène Site → Building → BuildingPart.

        Chaque bâtiment regroupe ses parties — principale et volumes
        secondaires d'un MultiPolygon — dans le même repère projeté. Un
        volume isolé forme un bâtiment à une seule partie.
        """
        grouped: dict[str, list[Prism]] = {}
        for prism in self.prisms:
            key = prism.parent_building_id or prism.feature_id
            grouped.setdefault(key, []).append(prism)
        return grouped

    def radius_m(self) -> float:
        """Rayon englobant de la cible, pour dimensionner l'orbite."""
        target = self.target
        if target is None:
            return 30.0
        cx, cy = self.centre
        d = np.hypot(target.footprint[:, 0] - cx, target.footprint[:, 1] - cy)
        return float(d.max())

    def summary(self) -> dict:
        buildings = self.buildings()
        secondary = sum(max(0, len(parts) - 1) for parts in buildings.values())
        return {
            "hotel_id": self.hotel_id,
            "crs": self.crs,
            "prism_count": len(self.prisms),
            "building_count": len(buildings),
            "secondary_parts": secondary,
            "target_resolved": self.target is not None,
            "assumed_height_count": self.assumed_height_count,
            "target_radius_m": round(self.radius_m(), 2),
            "centre": [round(c, 2) for c in self.centre],
            "provenance": self.provenance,
        }


def _polygons_of(entry: dict) -> tuple[list[Polygon], dict | None]:
    """Toutes les parties d'une emprise, principale d'abord.

    Un MultiPolygon ne perd plus ses volumes secondaires : chaque partie
    devient une BuildingPart rattachée au même bâtiment, dans le même
    repère. Les trous intérieurs restent ignorés, et dit comme tel.
    """
    raw = entry.get("projected_wkt")
    if not raw:
        return [], None
    geom = shapely_wkt.loads(raw)
    simplification: dict | None = None

    parts: list[Polygon] = []
    if geom.geom_type == "MultiPolygon":
        ordered = sorted(geom.geoms, key=lambda g: -g.area)
        for part in ordered:
            if isinstance(part, Polygon) and not part.is_empty:
                parts.append(part)
        simplification = {
            "rule": "MultiPolygon -> scene graph building parts",
            "parts_kept": len(parts),
            "holes_dropped": sum(len(g.interiors) for g in parts),
            "area_dropped_m2": 0.0,
        }
    elif isinstance(geom, Polygon) and not geom.is_empty:
        parts.append(geom)

    if not parts:
        return [], simplification

    kept: list[Polygon] = []
    for part in parts:
        if len(part.interiors) > 0 and not simplification:
            simplification = {
                "rule": "interior rings ignored",
                "parts_dropped": 0,
                "holes_dropped": len(part.interiors),
                "area_dropped_m2": round(
                    float(sum(g.area for g in part.interiors)), 2
                ),
            }
        exterior_only = Polygon(part.exterior)
        if not exterior_only.is_empty:
            kept.append(exterior_only)
    return kept, simplification


def _prisms_of(entry: dict) -> list[Prism]:
    """Une partie par volume : le bâtiment complet, pas son seul pan principal."""
    role = entry.get("role", "")
    if role not in TARGET_ROLES | OBSTACLE_ROLES:
        return []
    if entry.get("resolution_status") != "resolved":
        log.info(
            "volume écarté : %s est %s",
            entry.get("feature_id"),
            entry.get("resolution_status"),
        )
        return []
    polygons, simplification = _polygons_of(entry)
    if not polygons:
        return []

    known = bool(entry.get("height_known"))
    raw_height = entry.get("height_m")
    if known and raw_height:
        height, assumed = float(raw_height), False
        source = str(entry.get("height_source") or "manifeste")
    else:
        height, assumed = ASSUMED_HEIGHT_M.get(role, 8.0), True
        source = f"hypothèse d'animation, rôle {role!r} — aucune hauteur mesurée"

    feature_id = str(entry.get("feature_id"))
    prisms: list[Prism] = []
    for index, poly in enumerate(polygons):
        coords = np.asarray(poly.exterior.coords[:-1], dtype=np.float64)
        secondary = index > 0
        prisms.append(
            Prism(
                feature_id=feature_id if index == 0 else f"{feature_id}#part{index}",
                role=role,
                footprint=coords,
                height_m=height,
                height_assumed=assumed,
                height_source=source,
                # Seule la partie principale porte la cible : la trajectoire
                # ne doit pas orbiter autour d'une annexe.
                is_target=(role in TARGET_ROLES) and not secondary,
                geometry_simplification=simplification if index == 0 else None,
                parent_building_id=feature_id if secondary else None,
                part_index=index,
            )
        )
    log.info(
        "%s : %d partie(s) de bâtiment conservée(s)",
        feature_id,
        len(prisms),
    )
    return prisms


def load_scene(capture_geometry_path: Path) -> ConditioningScene:
    """Construit la scène volumique depuis `capture_geometry.json`."""
    payload = json.loads(Path(capture_geometry_path).read_text(encoding="utf-8"))
    entries = payload.get("geometries", [])

    prisms = [p for e in entries for p in _prisms_of(e)]
    if not prisms:
        raise ValueError(
            f"aucun volume exploitable dans {capture_geometry_path} — "
            "ni cible ni obstacle résolu"
        )

    crs = next(
        (e.get("projected_crs") for e in entries if e.get("projected_crs")),
        "unknown",
    )
    target = next((p for p in prisms if p.is_target), None)
    if target is None:
        raise ValueError(
            "aucun bâtiment cible résolu : une trajectoire ne peut pas orbiter "
            "autour d'un centre inexistant"
        )
    centre = (float(target.footprint[:, 0].mean()), float(target.footprint[:, 1].mean()))

    scene = ConditioningScene(
        hotel_id=str(payload.get("hotel_id", "unknown")),
        crs=str(crs),
        prisms=prisms,
        centre=centre,
        provenance={
            "capture_geometry": str(capture_geometry_path),
            "site_manifest_digest": payload.get("site_manifest_digest"),
            "policy_digest": payload.get("policy_digest"),
            "assumed_height_rule": ASSUMED_HEIGHT_M,
            "geometry_simplifications": [
                p.geometry_simplification for p in prisms if p.geometry_simplification
            ],
        },
    )
    log.info(
        "scène chargée : %d volumes, %d hauteurs supposées",
        len(prisms),
        scene.assumed_height_count,
    )
    return scene
