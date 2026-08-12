"""Rapports de couverture (Lot 1B §7, §11).

Ces rapports doivent être **reproductibles** : produits par le pipeline, pas
assemblés à la main dans un script ponctuel. Un chiffre que l'on ne peut pas
régénérer n'est pas une mesure, c'est un souvenir.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .logging import get_logger
from .schemas import Asset, PropertyMatchStatus

log = get_logger("coverage")


@dataclass
class StreetViewCoverage:
    """Ventilation exhaustive et disjointe des positions Street View.

    Les quatre catégories partitionnent l'ensemble : leur somme égale le nombre
    de positions. `undetermined` est un **sous-ensemble** de `context_only` —
    les positions dont le contenu n'a pas pu être tranché — et n'entre donc pas
    dans le total, sous peine de compter deux fois.
    """

    positions: int = 0
    visible: int = 0
    occluded: int = 0
    wrong_building: int = 0
    context_only: int = 0
    undetermined: int = 0
    by_sector: dict[str, int] = field(default_factory=dict)

    @property
    def partition_total(self) -> int:
        return self.visible + self.occluded + self.wrong_building + self.context_only

    def as_dict(self) -> dict:
        return {
            "positions": self.positions,
            "partition": {
                "visible": self.visible,
                "occluded": self.occluded,
                "wrong_building": self.wrong_building,
                "context_only": self.context_only,
            },
            "undetermined_subset_of_context_only": self.undetermined,
            "visible_by_sector": self.by_sector,
        }


def street_view_coverage(assets: list[Asset], source: str = "street_view") -> StreetViewCoverage:
    """Ventile les positions d'une source d'imagerie de rue.

    L'ordre des tests garantit la disjonction : une position ne peut appartenir
    qu'à une seule catégorie.
    """
    subset = [a for a in assets if a.source == source]
    coverage = StreetViewCoverage(positions=len(subset))

    for asset in subset:
        if asset.property_match_status is PropertyMatchStatus.MISMATCH:
            coverage.wrong_building += 1
        elif asset.occluded_by:
            coverage.occluded += 1
        elif asset.target_building_visible is True:
            coverage.visible += 1
            sector = asset.view_sector.value
            coverage.by_sector[sector] = coverage.by_sector.get(sector, 0) + 1
        else:
            coverage.context_only += 1
            if asset.target_building_visible is None:
                coverage.undetermined += 1

    if coverage.partition_total != coverage.positions:
        raise AssertionError(
            f"ventilation incohérente : {coverage.partition_total} != {coverage.positions}"
        )

    log.info(
        "couverture %s : %d position(s), %d visible(s), %d occultée(s), %d contexte",
        source,
        coverage.positions,
        coverage.visible,
        coverage.occluded,
        coverage.context_only,
    )
    return coverage
