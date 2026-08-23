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
    """Une emprise extrudée, et le statut de sa hauteur."""

    feature_id: str
    role: str
    #: Sommets du contour extérieur, en CRS projeté, sens quelconque.
    footprint: np.ndarray
    height_m: float
    height_assumed: bool
    height_source: str
    is_target: bool
    #: Toit mesuré : (m, 3) sommets d'une surface triangulée, en CRS projeté.
    #: Vide tant qu'aucun nDSM ne couvre l'emprise — le toit est alors fermé
    #: par une approximation dont la carte de confiance tient compte.
    roof_vertices: np.ndarray | None = None
    roof_faces: np.ndarray | None = None
    #: Relief des murs relevé dans le nuage, quand il l'a été.
    facade_relief: object | None = None
    #: Décomposition de la toiture en pans, quand elle a été segmentée.
    roof_planes: object | None = None

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

    def radius_m(self) -> float:
        """Rayon englobant de la cible, pour dimensionner l'orbite."""
        target = self.target
        if target is None:
            return 30.0
        cx, cy = self.centre
        d = np.hypot(target.footprint[:, 0] - cx, target.footprint[:, 1] - cy)
        return float(d.max())

    def summary(self) -> dict:
        return {
            "hotel_id": self.hotel_id,
            "crs": self.crs,
            "prism_count": len(self.prisms),
            "target_resolved": self.target is not None,
            "assumed_height_count": self.assumed_height_count,
            "target_radius_m": round(self.radius_m(), 2),
            "centre": [round(c, 2) for c in self.centre],
            "provenance": self.provenance,
        }


def _polygon_of(entry: dict) -> Polygon | None:
    raw = entry.get("projected_wkt")
    if not raw:
        return None
    geom = shapely_wkt.loads(raw)
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    if geom.geom_type != "Polygon" or geom.is_empty:
        return None
    return geom


def _prism_of(entry: dict) -> Prism | None:
    role = entry.get("role", "")
    if role not in TARGET_ROLES | OBSTACLE_ROLES:
        return None
    if entry.get("resolution_status") != "resolved":
        log.info(
            "volume écarté : %s est %s",
            entry.get("feature_id"),
            entry.get("resolution_status"),
        )
        return None
    poly = _polygon_of(entry)
    if poly is None:
        return None

    known = bool(entry.get("height_known"))
    raw_height = entry.get("height_m")
    if known and raw_height:
        height, assumed = float(raw_height), False
        source = str(entry.get("height_source") or "manifeste")
    else:
        height, assumed = ASSUMED_HEIGHT_M.get(role, 8.0), True
        source = f"hypothèse d'animation, rôle {role!r} — aucune hauteur mesurée"

    coords = np.asarray(poly.exterior.coords[:-1], dtype=np.float64)
    return Prism(
        feature_id=str(entry.get("feature_id")),
        role=role,
        footprint=coords,
        height_m=height,
        height_assumed=assumed,
        height_source=source,
        is_target=role in TARGET_ROLES,
    )


def load_scene(capture_geometry_path: Path) -> ConditioningScene:
    """Construit la scène volumique depuis `capture_geometry.json`."""
    payload = json.loads(Path(capture_geometry_path).read_text(encoding="utf-8"))
    entries = payload.get("geometries", [])

    prisms = [p for p in (_prism_of(e) for e in entries) if p is not None]
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
        },
    )
    log.info(
        "scène chargée : %d volumes, %d hauteurs supposées",
        len(prisms),
        scene.assumed_height_count,
    )
    return scene
