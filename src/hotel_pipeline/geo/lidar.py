"""Découverte des tuiles LiDAR du Québec, sans télécharger de LAZ (Lot 1B §9).

L'index complet pèse près de 159 Mo en GPKG. Le service WFS permet la même
interrogation pour quelques kilo-octets : on ne télécharge donc que des
métadonnées, et l'accord de téléchargement se demande avec le volume exact.

Deux règles gouvernent ce module.

**Une boîte englobante ne prouve pas une couverture.** Le WFS filtre par
`bbox` ; la géométrie retournée doit ensuite être intersectée avec l'empreinte
réelle du bâtiment. Une tuile voisine peut recouvrir la boîte sans toucher le
bâtiment.

**Un échec TLS n'est pas une absence de donnée.** Il produit
`discovery_error`, jamais `not_covered`, et la validation des certificats
n'est jamais désactivée : un pipeline qui contourne silencieusement TLS
transforme une panne d'infrastructure en conclusion métier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import requests

from ..logging import get_logger
from ..providers.cache import cached_call, ensure_online
from .catalog import CoverageState

log = get_logger("lidar")

WFS_URL = "https://servicesvecto3.mern.gouv.qc.ca/geoserver/Index_Telechargement_Lidar_Pub/wfs"
LAYER = "Index_Telechargement_Lidar_Pub:IndexTelechargementLidarPlusRecent"
TIMEOUT = 60

#: Marge autour de l'empreinte, en degrés — environ 250 m sous cette latitude.
#: Elle sert à ne pas rater une tuile dont la limite frôle le bâtiment.
DEFAULT_MARGIN_DEG = 0.0025


@dataclass
class TileCandidate:
    """Une tuile de l'index, telle que le WFS la décrit."""

    tile_id: str
    url: str
    project: str | None = None
    acquired_on: date | None = None
    point_density_per_m2: float | None = None
    classification: str | None = None
    file_format: str | None = None
    crs_horizontal: str | None = None
    crs_vertical: str | None = None
    licence: str | None = None
    announced_size: str | None = None
    geometry_wkt: str | None = None

    #: Taille exacte, obtenue par une requête HEAD. Le volume annoncé dans les
    #: métadonnées est arrondi ; le consentement se demande sur l'exact.
    exact_size_bytes: int | None = None

    def intersects(self, footprint_wkt: str) -> bool:
        """La tuile recouvre-t-elle réellement l'empreinte ?"""
        if not self.geometry_wkt:
            return False
        from shapely import wkt as shapely_wkt

        return shapely_wkt.loads(self.geometry_wkt).intersects(
            shapely_wkt.loads(footprint_wkt)
        )

    def as_dict(self) -> dict:
        return {
            "tile_id": self.tile_id,
            "project": self.project,
            "acquired_on": self.acquired_on.isoformat() if self.acquired_on else None,
            "point_density_per_m2": self.point_density_per_m2,
            "classification": self.classification,
            "format": self.file_format,
            "crs_horizontal": self.crs_horizontal,
            "crs_vertical": self.crs_vertical,
            "licence": self.licence,
            "announced_size": self.announced_size,
            "exact_size_bytes": self.exact_size_bytes,
            "url": self.url,
        }


@dataclass
class DiscoveryResult:
    state: CoverageState = CoverageState.UNKNOWN
    tiles: list[TileCandidate] = field(default_factory=list)
    considered: int = 0
    error: str | None = None

    @property
    def total_bytes(self) -> int:
        return sum(t.exact_size_bytes or 0 for t in self.tiles)

    def as_dict(self) -> dict:
        return {
            "coverage": self.state.value,
            "tiles_considered": self.considered,
            "tiles_intersecting": len(self.tiles),
            "total_bytes": self.total_bytes,
            "error": self.error,
            "tiles": [t.as_dict() for t in self.tiles],
        }


def bbox_around(footprint_wkt: str, margin_deg: float = DEFAULT_MARGIN_DEG) -> str:
    """Emprise élargie de l'empreinte, au format attendu par ce WFS.

    Ordre des axes : **longitude, latitude**. WFS 2.0 avec EPSG:4326 impose en
    principe latitude d'abord, mais ce GeoServer attend l'ordre inverse. Émettre
    la mauvaise convention ne provoque aucune erreur : le service répond zéro
    entité, ce qui se lit comme une absence de couverture. C'est le pire mode
    de défaillance possible, et il s'est produit au premier appel réel.
    """
    from shapely import wkt as shapely_wkt

    minx, miny, maxx, maxy = shapely_wkt.loads(footprint_wkt).bounds
    return (
        f"{minx - margin_deg},{miny - margin_deg},"
        f"{maxx + margin_deg},{maxy + margin_deg},EPSG:4326"
    )


def _query_wfs(bbox: str, url: str = WFS_URL) -> dict:
    """Interroge le WFS. **Ne désactive jamais la validation TLS.**"""
    ensure_online("index LiDAR")
    response = requests.get(
        url,
        params={
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeNames": LAYER,
            "outputFormat": "application/json",
            "srsName": "EPSG:4326",
            "bbox": bbox,
        },
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def _field(properties: dict, *names: str):
    """Premier champ non vide parmi plusieurs orthographes possibles."""
    for name in names:
        for key, value in properties.items():
            if key.lower() == name.lower() and value not in (None, ""):
                return value
    return None


def _parse_date(value) -> date | None:  # noqa: ANN001
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def to_tile(feature: dict) -> TileCandidate | None:
    """Convertit une entité WFS en tuile candidate.

    Les noms d'attributs sont ceux du service québécois, relevés sur une
    réponse réelle. Les alias supplémentaires couvrent d'autres services sans
    dépendre d'eux.
    """
    properties = feature.get("properties") or {}
    url = _field(
        properties, "TELECHARGEMENT_TUILE", "url", "lien", "chemin", "download_url"
    )
    if not url:
        return None

    geometry_wkt = None
    geometry = feature.get("geometry")
    if geometry:
        from shapely.geometry import shape

        geometry_wkt = shape(geometry).wkt

    return TileCandidate(
        tile_id=str(
            _field(properties, "NOM_TUILE", "tuile", "tile", "nom", "name")
            or url.rsplit("/", 1)[-1]
        ),
        url=str(url),
        project=_field(properties, "NOM_PROJET", "PROJET", "projet", "project"),
        acquired_on=_parse_date(
            _field(properties, "DATE_ACQUISITION", "DATE_FIN", "date_acquisition", "date")
        ),
        point_density_per_m2=_as_float(_field(properties, "DENSITE", "densite", "density")),
        classification=_stringify(
            _field(properties, "CLASSIFICATION", "CLASSES", "classification")
        ),
        file_format=_field(properties, "FORMAT_FICHIER", "FORMAT", "format"),
        crs_horizontal=_as_epsg(
            _field(properties, "CODE_EPSG", "EPSG", "crs", "srs", "projection")
        ),
        crs_vertical=_field(
            properties, "SYSREF_ALTIMETRIQUE", "referentiel_vertical", "vertical"
        ),
        licence=_field(properties, "LICENCE", "licence", "license"),
        announced_size=_stringify(
            _field(properties, "TAILLE_FICHIER", "taille", "size", "volume")
        ),
        geometry_wkt=geometry_wkt,
    )


def _as_float(value):  # noqa: ANN001
    """Premier nombre d'une valeur, même accompagnée d'une unité.

    Le service rend « 15 pts/m2 » : une conversion directe échouait et perdait
    silencieusement la densité.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    import re

    match = re.search(r"[-+]?\d+(?:[.,]\d+)?", str(value))
    return float(match.group().replace(",", ".")) if match else None


def _as_epsg(value):  # noqa: ANN001
    """Normalise un code de projection en identifiant complet.

    Le service rend `CODE_EPSG=2950` ; un nombre nu n'est pas un référentiel.
    """
    if value is None:
        return None
    text = str(value).strip()
    return f"EPSG:{text}" if text.isdigit() else text


def _stringify(value):  # noqa: ANN001
    return None if value is None else str(value)


def exact_size(url: str) -> int | None:
    """Taille exacte annoncée par le serveur, sans télécharger le fichier.

    Le consentement se demande sur ce nombre : le volume des métadonnées est
    arrondi, et un arrondi n'engage personne.
    """
    ensure_online("taille de tuile")
    response = requests.head(url, timeout=TIMEOUT, allow_redirects=True)
    response.raise_for_status()
    length = response.headers.get("Content-Length")
    return int(length) if length and length.isdigit() else None


def discover(
    footprint_wkt: str,
    margin_deg: float = DEFAULT_MARGIN_DEG,
    url: str = WFS_URL,
    measure_sizes: bool = True,
) -> DiscoveryResult:
    """Tuiles recouvrant réellement l'empreinte, sans télécharger de LAZ."""
    bbox = bbox_around(footprint_wkt, margin_deg)

    try:
        payload = cached_call(f"lidar-wfs::{bbox}", lambda: _query_wfs(bbox, url))
    except requests.exceptions.SSLError as exc:
        # Une chaîne TLS non reconnue est une panne d'infrastructure. La lire
        # comme une absence de donnée ferait conclure « pas de LiDAR ici ».
        log.error("validation TLS impossible sur l'index LiDAR : %s", exc)
        return DiscoveryResult(
            state=CoverageState.DISCOVERY_ERROR,
            error=f"validation TLS impossible : {exc}",
        )
    except (requests.RequestException, ValueError) as exc:
        log.error("index LiDAR injoignable : %s", exc)
        return DiscoveryResult(
            state=CoverageState.DISCOVERY_ERROR, error=f"index injoignable : {exc}"
        )

    features = payload.get("features") or []
    result = DiscoveryResult(considered=len(features))

    for feature in features:
        tile = to_tile(feature)
        # Le filtre WFS porte sur une boîte : l'intersection réelle décide.
        if tile is not None and tile.intersects(footprint_wkt):
            result.tiles.append(tile)

    if not result.tiles:
        result.state = CoverageState.NOT_COVERED
        log.info(
            "aucune tuile n'intersecte l'empreinte (%d examinée(s))", result.considered
        )
        return result

    if measure_sizes:
        for tile in result.tiles:
            try:
                tile.exact_size_bytes = exact_size(tile.url)
            except (requests.RequestException, ValueError) as exc:
                log.warning("taille indisponible pour %s : %s", tile.tile_id, exc)

    result.state = CoverageState.COVERED
    log.info(
        "couverture confirmée : %d tuile(s), %d octet(s) à télécharger",
        len(result.tiles),
        result.total_bytes,
    )
    return result
