"""Enveloppe géographique d'une scène, pour interroger les sources à sa mesure.

La découverte LiDAR interroge l'index avec l'empreinte du **bâtiment cible** :
72 × 77 m sur ce pilote. La scène rendue, elle, contient les voisins qui
occultent la vue, et s'étend sur plus d'un kilomètre — d'où vingt volumes sur
vingt-sept hors de la seule tuile téléchargée, et autant de hauteurs restées
supposées faute d'avoir demandé les bonnes tuiles.

Le module ne duplique aucune logique de découverte : il produit le WKT que
`geo.lidar.discover` attend déjà, à l'échelle de ce qui sera rendu.
"""

from __future__ import annotations

import json
from pathlib import Path

from shapely import wkt as shapely_wkt
from shapely.ops import unary_union

from ..logging import get_logger

log = get_logger("scene-extent")

#: Rôles dont la géométrie entre dans une scène rendue.
RENDERED_ROLES = frozenset({"target_building", "obstacle_building"})


def scene_envelope_wkt(
    capture_geometry_path: Path, geographic: bool = True
) -> str | None:
    """Enveloppe convexe des volumes rendus, en WKT.

    `geographic` rend l'enveloppe en WGS84, seule projection que l'index WFS
    accepte. Les géométries non résolues sont écartées : demander des tuiles
    pour une emprise démentie reviendrait à télécharger sur la foi d'une erreur.
    """
    payload = json.loads(Path(capture_geometry_path).read_text(encoding="utf-8"))
    key = "wgs84_wkt" if geographic else "projected_wkt"

    shapes = []
    for entry in payload.get("geometries", []):
        if entry.get("role") not in RENDERED_ROLES:
            continue
        if entry.get("resolution_status") != "resolved":
            continue
        raw = entry.get(key)
        if not raw:
            continue
        geometry = shapely_wkt.loads(raw)
        if geometry.is_valid and not geometry.is_empty:
            shapes.append(geometry)

    if not shapes:
        log.info("aucune géométrie rendue dans %s", capture_geometry_path)
        return None

    envelope = unary_union(shapes).convex_hull
    log.info(
        "enveloppe de scène : %d volume(s), emprise %s",
        len(shapes),
        [round(v, 5) for v in envelope.bounds],
    )
    return envelope.wkt


def uncovered_volumes(scene, tile_bounds: tuple[float, float, float, float]) -> list[str]:
    """Volumes de la scène qu'une emprise de tuile ne contient pas.

    Sert à dire *pourquoi* une hauteur reste supposée : hors couverture est un
    motif, « pas mesuré » n'en est pas un.
    """
    minx, miny, maxx, maxy = tile_bounds
    outside = []
    for prism in scene.prisms:
        x = prism.footprint[:, 0]
        y = prism.footprint[:, 1]
        if x.min() < minx or x.max() > maxx or y.min() < miny or y.max() > maxy:
            outside.append(prism.feature_id)
    return outside
