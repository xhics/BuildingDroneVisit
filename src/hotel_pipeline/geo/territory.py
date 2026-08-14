"""Résolution du territoire et du référentiel de travail (portabilité, commit 2).

`territories_for()` partait de `{"QC"}` inconditionnellement : tout point de la
Terre appartenait au Québec, et Lyon se voyait proposer `lidar-quebec`. Le
routage réussissait, l'acquisition échouait plus tard, et la cause n'était plus
lisible.

Deux principes remplacent la boîte englobante implicite :

- une juridiction s'établit ou ne s'établit pas ; il n'y a pas de valeur de
  repli, et `unknown` est une réponse ;
- le référentiel de travail se **choisit** en fonction de la position, et son
  emprise est vérifiée avant tout calcul, jamais après.

Les emprises restent approchées et déclarées comme telles : elles seront
remplacées par une intersection avec des limites officielles. Ce qui change est
qu'elles ne s'appliquent plus par défaut.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..logging import get_logger
from ..schemas.spatial_reference import (
    HeightType,
    SpatialReferenceContext,
    TerritoryState,
    VerticalReference,
)

log = get_logger("territory")


@dataclass(frozen=True)
class JurisdictionBox:
    """Une juridiction et son emprise approchée, avec la source de l'emprise."""

    code: str
    west: float
    south: float
    east: float
    north: float
    evidence: str
    parent: str | None = None

    def contains(self, lat: float, lon: float) -> bool:
        return self.west <= lon <= self.east and self.south <= lat <= self.north


#: Juridictions connues. Ajouter un territoire consiste à décrire son emprise,
#: jamais à modifier le code qui la consomme — et n'en pas décrire une signifie
#: `unknown`, non « le Québec ».
JURISDICTIONS: tuple[JurisdictionBox, ...] = (
    JurisdictionBox(
        code="CA", west=-141.0, south=41.6, east=-52.6, north=83.2,
        evidence="emprise approchée du Canada, à remplacer par les limites officielles",
    ),
    JurisdictionBox(
        code="QC", parent="CA", west=-79.8, south=44.9, east=-56.9, north=62.6,
        evidence="emprise approchée du Québec",
    ),
    JurisdictionBox(
        code="QC-CMM", parent="QC", west=-74.35, south=45.30, east=-73.20, north=45.90,
        evidence="emprise approchée de la Communauté métropolitaine de Montréal",
    ),
    JurisdictionBox(
        code="QC-MONTEREGIE", parent="QC", west=-74.10, south=44.98, east=-72.40, north=45.85,
        evidence="emprise approchée de la Montérégie",
    ),
    JurisdictionBox(
        code="FR", west=-5.2, south=41.3, east=9.6, north=51.1,
        evidence="emprise approchée de la France métropolitaine",
    ),
)


@dataclass(frozen=True)
class WorkingCrsOption:
    """Un référentiel projeté candidat, et où il vaut."""

    crs: str
    jurisdiction: str
    west: float
    south: float
    east: float
    north: float
    unit: str
    axes: str
    rationale: str

    def contains(self, lat: float, lon: float) -> bool:
        return self.west <= lon <= self.east and self.south <= lat <= self.north


#: Référentiels de travail connus. `EPSG:2950` y figure comme un candidat parmi
#: d'autres, borné à son fuseau — et non plus comme la constante du module.
WORKING_CRS: tuple[WorkingCrsOption, ...] = (
    WorkingCrsOption(
        crs="EPSG:2950", jurisdiction="QC",
        west=-75.0, south=44.98, east=-72.0, north=62.53,
        unit="m", axes="easting,northing",
        rationale="NAD83(CSRS) / MTM fuseau 8 — couvre Montréal et la Montérégie",
    ),
    WorkingCrsOption(
        crs="EPSG:2949", jurisdiction="QC",
        west=-78.0, south=44.98, east=-75.0, north=62.53,
        unit="m", axes="easting,northing",
        rationale="NAD83(CSRS) / MTM fuseau 7 — Québec à l'ouest du 75e",
    ),
    WorkingCrsOption(
        crs="EPSG:2154", jurisdiction="FR",
        west=-9.62, south=41.18, east=10.3, north=51.56,
        unit="m", axes="easting,northing",
        rationale="RGF93 / Lambert-93 — référentiel légal français",
    ),
)


def jurisdictions_for(lat: float, lon: float) -> list[str]:
    """Juridictions contenant ce point, du plus large au plus fin.

    Rend une liste **vide** quand rien ne correspond. C'est le changement
    central : l'absence de correspondance ne vaut plus `QC`.
    """
    found = [box for box in JURISDICTIONS if box.contains(lat, lon)]
    order = {box.code: index for index, box in enumerate(JURISDICTIONS)}
    return [box.code for box in sorted(found, key=lambda b: order[b.code])]


def evidence_for(codes: list[str]) -> list[str]:
    by_code = {box.code: box for box in JURISDICTIONS}
    return [f"{code} : {by_code[code].evidence}" for code in codes if code in by_code]


def working_crs_for(lat: float, lon: float, codes: list[str]) -> WorkingCrsOption | None:
    """Référentiel projeté couvrant ce point, dans une juridiction établie.

    Deux conditions, et non une : la juridiction doit être établie **et**
    l'emprise contenir le point. Se fier à la seule emprise laisserait un
    référentiel s'appliquer sur un territoire non résolu, ce qui est le défaut
    d'origine sous un autre nom.
    """
    for option in WORKING_CRS:
        if option.jurisdiction in codes and option.contains(lat, lon):
            return option
    return None


def vertical_from_acquisition(report: dict | None) -> VerticalReference:
    """Référentiel vertical déclaré par les données acquises.

    Il ne se déduit ni du pays ni du référentiel horizontal : il vient de la
    source, ou reste inconnu. Le type de hauteur n'est pas davantage supposé —
    un nom de référentiel ne dit pas à lui seul s'il porte des altitudes
    orthométriques, même quand c'est le cas le plus courant.
    """
    if not report:
        return VerticalReference()

    # Le référentiel vertical décrit la **source**, non le téléchargement :
    # `acquisitions` porte URL, taille et empreinte, `sources` porte les
    # métadonnées de la tuile. Les autres clés sont acceptées pour ne pas
    # dépendre d'un seul producteur.
    entries = (
        report.get("sources") or report.get("tiles") or [report]
    )
    declared = {
        entry.get("crs_vertical")
        for entry in entries
        if isinstance(entry, dict) and entry.get("crs_vertical")
    }

    if len(declared) != 1:
        # Plusieurs référentiels, ou aucun : dans les deux cas, rien n'est
        # établi. Choisir le plus fréquent reviendrait à trancher au jugé.
        return VerticalReference(
            provenance=(
                f"{len(declared)} référentiels verticaux déclarés par les "
                "tuiles acquises — aucun ne fait autorité"
                if declared else None
            )
        )

    name = declared.pop()
    return VerticalReference(
        crs=name,
        height_type=HEIGHT_TYPES.get(name, HeightType.UNKNOWN),
        provenance=f"déclaré par la source acquise : {name}",
    )


#: Types de hauteur des référentiels verticaux connus. Ce qui n'y figure pas
#: reste `UNKNOWN` : le nom d'un référentiel ne dit pas ce qu'il mesure, et le
#: supposer ferait passer un écart de plusieurs dizaines de mètres pour zéro.
HEIGHT_TYPES: dict[str, HeightType] = {
    "CGVD 1928": HeightType.ORTHOMETRIC,
    "CGVD28": HeightType.ORTHOMETRIC,
    "CGVD2013": HeightType.ORTHOMETRIC,
    "NGF-IGN69": HeightType.ORTHOMETRIC,
}


def resolve(
    hotel_id: str, lat: float, lon: float,
    dependency_digests: dict[str, str] | None = None,
    vertical: VerticalReference | None = None,
) -> SpatialReferenceContext:
    """Contexte spatial d'un site. Ne devine rien, et le dit quand il ignore."""
    codes = jurisdictions_for(lat, lon)

    if not codes:
        log.info("territoire non résolu en (%.5f, %.5f) : aucune juridiction connue", lat, lon)
        return SpatialReferenceContext(
            hotel_id=hotel_id, reference_lat=lat, reference_lon=lon,
            territory_state=TerritoryState.UNKNOWN,
            dependency_digests=dict(dependency_digests or {}),
        )

    option = working_crs_for(lat, lon, codes)
    if option is None:
        log.info(
            "territoire %s connu mais sans référentiel de travail déclaré", codes
        )
        return SpatialReferenceContext(
            hotel_id=hotel_id, reference_lat=lat, reference_lon=lon,
            territory_state=TerritoryState.UNSUPPORTED,
            jurisdictions=codes, territory_evidence=evidence_for(codes),
            dependency_digests=dict(dependency_digests or {}),
        )

    log.info("territoire %s, référentiel de travail %s", codes, option.crs)
    return SpatialReferenceContext(
        hotel_id=hotel_id, reference_lat=lat, reference_lon=lon,
        territory_state=TerritoryState.RESOLVED,
        jurisdictions=codes, territory_evidence=evidence_for(codes),
        working_crs=option.crs, working_unit=option.unit, working_axes=option.axes,
        working_area_of_use=[option.west, option.south, option.east, option.north],
        selection_method=(
            f"juridiction {option.jurisdiction} et emprise du référentiel — "
            f"{option.rationale}"
        ),
        # Le référentiel vertical ne se déduit pas de l'horizontal : il vient
        # des données acquises, et reste inconnu tant qu'aucune ne le déclare.
        vertical=vertical or VerticalReference(),
        dependency_digests=dict(dependency_digests or {}),
    )
