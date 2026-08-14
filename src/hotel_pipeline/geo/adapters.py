"""Adaptateurs de découverte, par source (portabilité, commit 2b).

`geo discover` calculait une route territoriale, puis appelait le WFS LiDAR du
Québec quelle qu'en fût la réponse : la variable `source` n'était pas utilisée.
Un site lyonnais dont le routage ne proposait rien interrogeait donc quand même
un service québécois, et son silence se lisait comme une absence de couverture.

Une source n'est découvrable que si un adaptateur la sait interroger. Sans
adaptateur, l'état est `unsupported` et **aucune requête n'est émise** — ce qui
n'est pas la même chose que « non couvert », qui, lui, suppose d'avoir demandé.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..logging import get_logger

log = get_logger("geo-adapters")


@dataclass(frozen=True)
class DiscoveryAdapter:
    """Ce qui sait interroger une source, et ce qu'il faut lui donner."""

    source_id: str
    service: str
    discover: Callable

    def __call__(self, footprint_wkt: str, measure_sizes: bool = True):  # noqa: ANN201
        log.info("découverte via %s (%s)", self.source_id, self.service)
        return self.discover(footprint_wkt, measure_sizes=measure_sizes)


def _lidar_quebec_adapter() -> DiscoveryAdapter:
    from .lidar import discover

    return DiscoveryAdapter(
        source_id="lidar-quebec",
        service="GeoServer WFS du MERN — convention d'axes lon,lat",
        discover=discover,
    )


#: Adaptateurs disponibles, par identifiant de source du catalogue. Ajouter un
#: territoire consiste à écrire un adaptateur, jamais à élargir un existant.
ADAPTERS: dict[str, Callable[[], DiscoveryAdapter]] = {
    "lidar-quebec": _lidar_quebec_adapter,
}


def adapter_for(source_id: str) -> DiscoveryAdapter | None:
    """Adaptateur d'une source, ou `None` si personne ne sait l'interroger."""
    factory = ADAPTERS.get(source_id)
    return factory() if factory else None


def elevation_adapter(routing) -> tuple[DiscoveryAdapter | None, list[str]]:  # noqa: ANN001
    """Premier adaptateur capable de servir une source portant l'altimétrie.

    Rend aussi les motifs de refus, pour que « rien à interroger » soit une
    réponse argumentée plutôt qu'un silence.
    """
    reasons: list[str] = []
    if not routing.territorial_candidates:
        reasons.append(
            "aucune source territorialement admissible : le territoire n'est "
            "pas couvert par le catalogue, ou n'a pas été résolu"
        )
        return None, reasons

    for source in routing.territorial_candidates:
        adapter = adapter_for(source.source_id)
        if adapter is None:
            reasons.append(
                f"{source.source_id} : territorialement admissible, mais aucun "
                "adaptateur ne sait l'interroger"
            )
            continue
        return adapter, reasons

    return None, reasons
