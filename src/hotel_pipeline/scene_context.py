"""Contexte spatial d'un site, chargé une fois (Lot 2).

Chaque analyse de scène — couverture, recadrage, confiance — a besoin des mêmes
quatre choses : l'empreinte cible, les obstacles, les façades, et de quoi
projeter des coordonnées. Les recharger à chaque commande dupliquait la
jointure, et deux pièges s'y logeaient :

- `BUILDING_MAIN.geometry_wkt` du manifeste de site est en **degrés**, alors
  que les façades sont en **mètres projetés**. Mélanger les deux donnait des
  couvertures inversées — l'arrière « entièrement vu », l'avant « jamais » ;
- le CRS de travail est **déclaré par le territoire**, jamais supposé. Une
  constante `EPSG:2950` marchait au pilote et refusait tout autre site.

Ce module fait la jointure une fois, dans le bon référentiel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from .logging import get_logger

log = get_logger("scene-context")

#: Pas d'échantillonnage des murs, en mètres.
SAMPLE_STEP_M = 2.0


@dataclass
class SceneContext:
    """Ce qu'il faut savoir du site pour raisonner sur ses vues."""

    hotel_id: str
    working_crs: str
    footprint: object = None
    obstacles: list = field(default_factory=list)
    #: `{facade_id: [FacadeSample, ...]}`, en CRS projeté.
    facades: dict = field(default_factory=dict)
    #: Positions de caméra projetées : `[(asset_id, provider_id, (x, y))]`.
    viewpoints: list = field(default_factory=list)
    front_azimuth_deg: float | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.footprint is not None and bool(self.facades)


def _project(transformer, lon: float, lat: float) -> tuple[float, float]:
    return transformer.transform(lon, lat)


def _dense_samples(geometry, footprint, step: float = SAMPLE_STEP_M) -> list:
    """Échantillonne un mur, normale extérieure comprise.

    `sample_facade` fixe le nombre de points par segment ; ici on veut un pas
    **métrique**, pour qu'un mur de 88 m et un mur de 8 m ne portent pas le
    même nombre d'échantillons.
    """
    from shapely.geometry import Point

    from .geo.facade_coverage import FacadeSample

    parts = (
        list(geometry.geoms)
        if geometry.geom_type == "MultiLineString"
        else [geometry]
    )
    samples: list = []
    for part in parts:
        coords = list(part.coords)
        for (x0, y0), (x1, y1) in zip(coords, coords[1:]):
            length = math.hypot(x1 - x0, y1 - y0)
            if length <= 0:
                continue
            nx, ny = (y1 - y0) / length, -(x1 - x0) / length
            probe = Point(
                (x0 + x1) / 2 + nx * 0.5, (y0 + y1) / 2 + ny * 0.5
            )
            # La normale extérieure est celle qui **sort** de l'empreinte. Sur
            # un plan concave elle peut pointer vers le centroïde : le test de
            # containment est le bon, non un test d'orientation.
            if footprint.contains(probe):
                nx, ny = -nx, -ny
            for index in range(max(1, int(length // step)) + 1):
                ratio = min(1.0, index * step / length)
                samples.append(
                    FacadeSample(
                        x0 + (x1 - x0) * ratio, y0 + (y1 - y0) * ratio, (nx, ny)
                    )
                )
    return samples


def load(hotel_id: str, *, context=None) -> SceneContext:  # noqa: ANN001
    """Charge le contexte spatial d'un site, ou dit ce qui manque."""
    import pyproj
    from shapely import wkt as shapely_wkt

    from .capabilities import Capability
    from .geo.geometry_loader import load_capture_geometry
    from .workspace import Workspace

    if context is None:
        from .cli import _context

        context = _context(hotel_id, Capability.INSPECTION)

    workspace = Workspace(hotel_id)
    problems: list[str] = []

    geometry_path = workspace.path("06_geo", "capture_geometry.json")
    if not geometry_path.is_file():
        return SceneContext(
            hotel_id=hotel_id, working_crs="",
            problems=["géométrie de capture absente : lancez `geo derive`"],
        )

    manifest, _ = load_capture_geometry(geometry_path, context.spatial_reference)
    working_crs = manifest.working_crs

    targets = [
        g for g in manifest.geometries
        if g.role.value == "target_building" and g.projected_wkt
    ]
    if not targets:
        return SceneContext(
            hotel_id=hotel_id, working_crs=working_crs,
            problems=["aucune empreinte cible projetée"],
        )
    footprint = shapely_wkt.loads(targets[0].projected_wkt)

    obstacles = [
        shapely_wkt.loads(g.projected_wkt)
        for g in manifest.geometries
        if g.role.value == "obstacle_building" and g.projected_wkt
    ]
    if len(obstacles) < 5:
        # Constaté au pilote : 27 obstacles pour tout un quartier. Les
        # pavillons n'y figurent pas, et la ligne de vue 2D les traverse.
        problems.append(
            f"modèle d'obstacles pauvre ({len(obstacles)}) : la visibilité "
            "géométrique surestimera les vues dégagées"
        )

    site = workspace.read_site()
    facades: dict = {}
    if site is not None:
        for obj in site.objects:
            if not obj.kind.startswith("FACADE_"):
                continue
            if not getattr(obj, "geometry_wkt", None):
                continue
            samples = _dense_samples(
                shapely_wkt.loads(obj.geometry_wkt), footprint
            )
            if samples:
                facades[obj.kind] = samples
    if not facades:
        problems.append("aucune façade résolue : la couverture par mur est indisponible")

    spatial = workspace.read_spatial()
    front = getattr(spatial, "front_azimuth_deg", None) if spatial else None
    if front is None:
        problems.append("orientation de façade inconnue : les secteurs restent indéterminés")

    transformer = pyproj.Transformer.from_crs("EPSG:4326", working_crs, always_xy=True)
    assets = workspace.read_assets()
    viewpoints: list = []
    if assets is not None:
        for asset in assets.assets:
            if asset.camera_lat is None or asset.camera_lon is None:
                continue
            provider = asset.id.split("-", 1)[-1] if "-" in asset.id else asset.id
            viewpoints.append(
                (asset.id, provider, _project(transformer, asset.camera_lon, asset.camera_lat))
            )

    log.info(
        "contexte %s : %d façade(s), %d obstacle(s), %d point(s) de vue",
        hotel_id, len(facades), len(obstacles), len(viewpoints),
    )
    return SceneContext(
        hotel_id=hotel_id,
        working_crs=working_crs,
        footprint=footprint,
        obstacles=obstacles,
        facades=facades,
        viewpoints=viewpoints,
        front_azimuth_deg=front,
        problems=problems,
    )


def viewpoints_from_candidates(
    scene: SceneContext, candidates_path: Path
) -> list:
    """Points de vue tirés d'un manifeste de candidats, non du corpus acquis.

    La découverte rend des positions bien avant qu'aucune image ne soit
    téléchargée : raisonner sur le seul corpus acquis ignorerait tout ce que la
    recherche vient de trouver.
    """
    import json

    import pyproj

    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", scene.working_crs, always_xy=True
    )
    payload = json.loads(Path(candidates_path).read_text("utf-8"))
    seen: set[str] = set()
    out: list = []
    for candidate in payload.get("candidates") or []:
        if candidate.get("camera_lat") is None:
            continue
        spec = candidate.get("request_spec") or {}
        provider = spec.get("pano_id") or candidate.get("provider_id")
        if not provider or provider in seen:
            continue
        seen.add(provider)
        out.append(
            (
                candidate["candidate_id"],
                provider,
                _project(
                    transformer, candidate["camera_lon"], candidate["camera_lat"]
                ),
            )
        )
    return out


__all__ = ["SAMPLE_STEP_M", "SceneContext", "load", "viewpoints_from_candidates"]
