"""Catalogue des sources géospatiales, avec routage territorial.

Une source ouverte n'est pas une source **disponible ici**. L'orthophoto
GéoMont 2023 à 20 cm est excellente, et exclut explicitement le territoire de
la CMM — dont Boucherville fait partie via l'agglomération de Longueuil. La
retenir par défaut aurait produit un téléchargement inutile, puis une absence
inexpliquée.

Chaque source déclare donc son emprise, sa résolution et ce qu'elle permet
réellement d'établir. Le routage se fait sur le territoire, pas sur l'espoir.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..logging import get_logger

log = get_logger("geo-catalog")


@dataclass(frozen=True)
class GeoSource:
    """Une source géospatiale et ce qu'elle autorise à conclure."""

    source_id: str
    dataset: str
    url: str

    #: Territoires couverts et exclus, en identifiants administratifs libres.
    covers: tuple[str, ...] = ()
    excludes: tuple[str, ...] = ()

    resolution_m: float | None = None

    #: Objets du gabarit que cette source permet d'établir — et **seulement**
    #: ceux-là. Une orthophoto ne fonde pas une limite cadastrale.
    establishes: tuple[str, ...] = ()

    #: Ce qu'elle ne peut pas établir, malgré l'apparence. Le champ existe pour
    #: que la limite soit lisible dans le catalogue, pas seulement connue.
    cannot_establish: tuple[str, ...] = ()

    licence: str | None = None
    notes: str | None = None

    def serves(self, territories: set[str]) -> bool:
        if territories & set(self.excludes):
            return False
        return not self.covers or bool(territories & set(self.covers))


#: Catalogue. Ajouter une source consiste à décrire son emprise et sa portée,
#: jamais à modifier le code qui la consomme.
SOURCES: tuple[GeoSource, ...] = (
    GeoSource(
        source_id="lidar-quebec",
        dataset="Données LiDAR du Québec",
        url="https://www.donneesquebec.ca/recherche/dataset/donnees-lidar-du-quebec",
        covers=("QC",),
        establishes=("TERRAIN_MAIN", "ROOFLINE_MAIN"),
        cannot_establish=("PROPERTY_PARCEL",),
        licence="Licence ouverte du gouvernement du Québec",
        notes=(
            "Nuage LAZ classifié. MNT et MNS s'en dérivent ; la hauteur des "
            "façades aussi, mais jamais leur apparence."
        ),
    ),
    GeoSource(
        source_id="geomont-ortho-2023",
        dataset="GéoMont — orthophotographies 2023, Montérégie",
        url="https://www.donneesquebec.ca/recherche/dataset/geomont-orthophotographies-2023-region-de-la-monteregie",
        covers=("QC-MONTEREGIE",),
        # Exclusion déterminante pour ce pilote : Boucherville relève de la CMM.
        excludes=("QC-CMM",),
        resolution_m=0.20,
        establishes=(),
        cannot_establish=("PROPERTY_PARCEL",),
        licence="Licence ouverte",
        notes="20 cm, mais le territoire de la CMM est hors emprise.",
    ),
    GeoSource(
        source_id="cmm-ortho",
        dataset="Orthophotos ouvertes de la CMM",
        url="https://observatoire.cmm.qc.ca/produits/donnees-georeferencees/",
        covers=("QC-CMM",),
        resolution_m=5.0,
        establishes=(),
        cannot_establish=("PROPERTY_PARCEL", "ROOFLINE_MAIN"),
        licence="Licence ouverte",
        notes=(
            "Mosaïques à 5 m : utiles comme verrou de contexte, insuffisantes "
            "pour découper un toit ou une parcelle."
        ),
    ),
    GeoSource(
        source_id="cadastre-quebec",
        dataset="Cadastre du Québec (Infolot)",
        url="https://www.quebec.ca/habitation-territoire/information-fonciere/cadastre/consulter-cadastre",
        covers=("QC",),
        establishes=("PROPERTY_PARCEL",),
        licence="consultation ; acquisition à formaliser",
        notes=(
            "Seule source juridique de la limite de propriété. Le Référentiel "
            "québécois des adresses peut donner un numéro de lot, jamais la "
            "géométrie officielle."
        ),
    ),
)


@dataclass
class Routing:
    territories: set[str] = field(default_factory=set)
    available: list[GeoSource] = field(default_factory=list)
    rejected: dict[str, str] = field(default_factory=dict)

    def for_object(self, kind: str) -> list[GeoSource]:
        return [s for s in self.available if kind in s.establishes]

    def as_dict(self) -> dict:
        return {
            "territories": sorted(self.territories),
            "available": [s.source_id for s in self.available],
            "rejected": self.rejected,
        }


def territories_for(lat: float, lon: float) -> set[str]:
    """Appartenances territoriales d'un point.

    Implémentation volontairement minimale et explicite : une boîte englobante
    approchée de la CMM, suffisante pour router ce pilote. Elle sera remplacée
    par une intersection avec les limites administratives officielles dès leur
    acquisition — et le catalogue n'aura pas à changer.
    """
    territories = {"QC"}

    # Communauté métropolitaine de Montréal, emprise approchée.
    if 45.30 <= lat <= 45.90 and -74.35 <= lon <= -73.20:
        territories.add("QC-CMM")

    # Montérégie, emprise approchée. Les deux se recouvrent : l'appartenance
    # à la CMM prime pour l'exclusion GéoMont.
    if 44.98 <= lat <= 45.85 and -74.10 <= lon <= -72.40:
        territories.add("QC-MONTEREGIE")

    return territories


def route(lat: float, lon: float) -> Routing:
    """Sources réellement utilisables à cette position."""
    territories = territories_for(lat, lon)
    routing = Routing(territories=territories)

    for source in SOURCES:
        if source.serves(territories):
            routing.available.append(source)
        else:
            blocked = territories & set(source.excludes)
            routing.rejected[source.source_id] = (
                f"territoire exclu : {sorted(blocked)}"
                if blocked
                else f"hors emprise ({sorted(source.covers)})"
            )

    log.info(
        "routage géospatial : %s → %d source(s) disponible(s), %d écartée(s)",
        sorted(territories),
        len(routing.available),
        len(routing.rejected),
    )
    return routing
