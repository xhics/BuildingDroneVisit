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

from enum import StrEnum

from ..logging import get_logger
from ..schemas.critical_objects import EXCLUDED_KINDS, REQUIRED_OBJECTS

log = get_logger("geo-catalog")

#: Types connus du gabarit. Valider `establishes` contre eux empêche qu'une
#: faute de frappe crée une capacité fantôme — une source déclarant établir
#: `ROOFLINE_MAIN2` n'établirait rien, sans que rien ne le signale.
KNOWN_KINDS: frozenset[str] = frozenset(REQUIRED_OBJECTS) | frozenset(EXCLUDED_KINDS)


class CoverageState(StrEnum):
    """Couverture réelle d'une source à un endroit donné.

    Distincte de l'admissibilité territoriale : le LiDAR québécois est
    pertinent partout au Québec, sans y être acquis partout. La documentation
    officielle renvoie d'ailleurs à une carte de couverture.
    """

    UNKNOWN = "unknown"
    COVERED = "covered"
    NOT_COVERED = "not_covered"
    DISCOVERY_ERROR = "discovery_error"
    MANUAL_ACQUISITION_REQUIRED = "manual_acquisition_required"


class CoverageBasis(StrEnum):
    """Comment la couverture peut être établie sans confondre les preuves."""

    INDEX_INTERSECTION = "index_intersection"
    PUBLISHER_DECLARED_TERRITORY = "publisher_declared_territory"
    MANUAL_SERVICE = "manual_service"


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

    #: L'acquisition est-elle automatisée ? Le cadastre est territorialement
    #: pertinent, mais sa géométrie officielle se consulte, elle ne se
    #: télécharge pas encore ici.
    acquisition_automated: bool = True

    #: Méthode admise pour décider de la couverture. Une déclaration de
    #: territoire convient à un verrou de contexte ; elle ne remplace jamais
    #: l'intersection d'une tuile lorsqu'une géométrie doit être dérivée.
    coverage_basis: CoverageBasis = CoverageBasis.INDEX_INTERSECTION
    index_url: str | None = None
    vintage: str | None = None

    def __post_init__(self) -> None:
        unknown = (set(self.establishes) | set(self.cannot_establish)) - KNOWN_KINDS
        if unknown:
            raise ValueError(
                f"source {self.source_id!r} : types inconnus du gabarit {sorted(unknown)}"
            )

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
        coverage_basis=CoverageBasis.INDEX_INTERSECTION,
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
        coverage_basis=CoverageBasis.INDEX_INTERSECTION,
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
        coverage_basis=CoverageBasis.PUBLISHER_DECLARED_TERRITORY,
        index_url=(
            "https://observatoire.cmm.qc.ca/produits/"
            "donnees-georeferencees/"
        ),
        vintage="2023-08",
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
        acquisition_automated=False,
        coverage_basis=CoverageBasis.MANUAL_SERVICE,
        index_url="https://appli.foncier.gouv.qc.ca/Infolot/",
        notes=(
            "Source officielle de la représentation cadastrale. Elle permet "
            "d'instancier PROPERTY_PARCEL une fois l'extrait acquis et vérifié, "
            "mais ne vaut ni arpentage ni titre de propriété."
        ),
    ),
)


@dataclass
class Routing:
    """Admissibilité territoriale — **pas** une couverture confirmée.

    Une source retenue ici est pertinente pour ce territoire. Savoir si elle
    couvre réellement l'empreinte demande d'interroger son index, ce qui est
    l'objet de la découverte.
    """

    territories: set[str] = field(default_factory=set)
    territorial_candidates: list[GeoSource] = field(default_factory=list)
    rejected: dict[str, str] = field(default_factory=dict)

    #: État de couverture par source, renseigné par la découverte.
    coverage: dict[str, CoverageState] = field(default_factory=dict)

    def for_object(self, kind: str) -> list[GeoSource]:
        """Candidats territoriaux susceptibles d'établir ce type."""
        return [s for s in self.territorial_candidates if kind in s.establishes]

    def confirmed_for(self, kind: str) -> list[GeoSource]:
        """Sources dont la couverture est **confirmée** pour ce type."""
        return [
            s
            for s in self.for_object(kind)
            if self.coverage.get(s.source_id) is CoverageState.COVERED
        ]

    def state_of(self, source_id: str) -> CoverageState:
        return self.coverage.get(source_id, CoverageState.UNKNOWN)

    def as_dict(self) -> dict:
        return {
            "territories": sorted(self.territories),
            "territorial_candidates": [s.source_id for s in self.territorial_candidates],
            "rejected": self.rejected,
            "coverage": {k: v.value for k, v in self.coverage.items()},
        }


def territories_for(lat: float, lon: float) -> set[str]:
    """Appartenances territoriales d'un point.

    Déléguée à l'adaptateur territorial. Cette fonction partait de `{"QC"}`
    inconditionnellement : tout point de la Terre appartenait au Québec, et
    Lyon se voyait proposer le LiDAR québécois. Un point hors des juridictions
    déclarées rend désormais un ensemble **vide**, ce qui n'est pas la même
    chose qu'un territoire sans source.
    """
    from .territory import jurisdictions_for

    return set(jurisdictions_for(lat, lon))


def route(lat: float, lon: float) -> Routing:
    """Sources réellement utilisables à cette position.

    Un territoire inconnu ne propose **rien**. C'est un état distinct de
    « territoire connu sans source » : le premier dit qu'on ne sait pas où on
    est, le second qu'il n'y a rien à télécharger ici.
    """
    territories = territories_for(lat, lon)
    routing = Routing(territories=territories)

    if not territories:
        log.info(
            "routage géospatial : territoire non résolu en (%.5f, %.5f) — "
            "aucune source proposée",
            lat, lon,
        )
        for source in SOURCES:
            routing.rejected[source.source_id] = (
                "territoire non résolu : aucune juridiction déclarée ne "
                "contient ce point"
            )
        return routing

    for source in SOURCES:
        if source.serves(territories):
            routing.territorial_candidates.append(source)
            routing.coverage[source.source_id] = CoverageState.UNKNOWN
        else:
            blocked = territories & set(source.excludes)
            routing.rejected[source.source_id] = (
                f"territoire exclu : {sorted(blocked)}"
                if blocked
                else f"hors emprise ({sorted(source.covers)})"
            )

    log.info(
        "routage géospatial : %s → %d candidat(s) territorial(aux), %d écarté(s) ; "
        "couverture réelle non encore vérifiée",
        sorted(territories),
        len(routing.territorial_candidates),
        len(routing.rejected),
    )
    return routing
