"""Comparer des sites plutôt que régler des seuils sur un seul.

Tous les seuils du dépôt sont aujourd'hui calibrés sur un pilote : un motel de
douze mètres, en périphérie, photographié depuis la rue. Rien ne dit qu'ils
tiennent sur une tour urbaine, un bâtiment mitoyen ou un site sous la neige —
et un seuil qui ne tient pas se révèle en production, quand il est trop tard
pour le mesurer.

Ce module ne fabrique pas de sites : il **recueille** ce que chaque site
mesuré donne, sous une forme comparable, et dit ce que la comparaison
autorise. Avec un seul site, elle n'autorise rien — et c'est ce qu'il rapporte,
plutôt qu'une moyenne rassurante calculée sur un point.

**Ce qu'un site apporte, et ce qu'il n'apporte pas.** Chaque entrée porte les
mesures que le pipeline produit déjà — vues confirmées, cellules triangulables,
connexité, couverture — et les caractéristiques qui font varier ces mesures :
hauteur, mitoyenneté, densité d'imagerie. Sans ces caractéristiques, deux
chiffres ne se comparent pas : 14 % de cellules triangulables ne veut pas dire
la même chose sur un bâtiment isolé et sur un bâtiment mitoyen dont deux murs
sont inaccessibles.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from .logging import get_logger

log = get_logger("benchmark")

#: Nombre de sites en deçà duquel aucune dispersion n'est calculable. Un seuil
#: réglé sur moins que cela reste une hypothèse, non une calibration.
MIN_SITES_FOR_SPREAD = 3

#: Nombre de sites à partir duquel une calibration commence à valoir. En
#: dessous, la comparaison décrit des cas, elle ne mesure pas une tendance.
MIN_SITES_FOR_CALIBRATION = 8


@dataclass
class SiteRecord:
    """Ce qu'un site a donné, et ce qui le caractérise."""

    hotel_id: str
    #: Ce qui fait varier les mesures, et sans quoi elles ne se comparent pas.
    height_m: float | None = None
    attached: bool | None = None
    urban: bool | None = None
    season: str | None = None

    #: Mesures produites par le pipeline.
    assets_total: int = 0
    views_confirming: int = 0
    cells_total: int = 0
    cells_triangulable: int = 0
    registered_images: int = 0
    largest_component: int = 0
    facade_coverage: float | None = None

    @property
    def confirming_fraction(self) -> float:
        return self.views_confirming / max(self.assets_total, 1)

    @property
    def triangulable_fraction(self) -> float:
        return self.cells_triangulable / max(self.cells_total, 1)

    @property
    def connected_fraction(self) -> float:
        return self.largest_component / max(self.registered_images, 1)

    def as_dict(self) -> dict:
        return {
            "hotel_id": self.hotel_id,
            "height_m": self.height_m,
            "attached": self.attached,
            "urban": self.urban,
            "season": self.season,
            "assets_total": self.assets_total,
            "views_confirming": self.views_confirming,
            "confirming_fraction": round(self.confirming_fraction, 3),
            "cells_total": self.cells_total,
            "cells_triangulable": self.cells_triangulable,
            "triangulable_fraction": round(self.triangulable_fraction, 3),
            "connected_fraction": round(self.connected_fraction, 3),
            "facade_coverage": self.facade_coverage,
        }


@dataclass
class Benchmark:
    """Les sites mesurés, et ce que leur comparaison permet de dire."""

    sites: list[SiteRecord] = field(default_factory=list)

    def spread(self, metric: str) -> dict:
        """Dispersion d'une mesure entre sites, quand elle est calculable."""
        values = [
            getattr(site, metric)
            for site in self.sites
            if getattr(site, metric, None) is not None
        ]
        values = [float(v) for v in values if isinstance(v, (int, float))]
        if len(values) < MIN_SITES_FOR_SPREAD:
            return {
                "metric": metric,
                "sites": len(values),
                "computable": False,
                "reason": (
                    f"{len(values)} site(s) mesuré(s) : une dispersion demande "
                    f"au moins {MIN_SITES_FOR_SPREAD}"
                ),
            }
        return {
            "metric": metric,
            "sites": len(values),
            "computable": True,
            "min": round(min(values), 3),
            "max": round(max(values), 3),
            "median": round(statistics.median(values), 3),
            "stdev": round(statistics.stdev(values), 3),
        }

    def calibration_status(self) -> str:
        """Ce que le corpus de sites autorise aujourd'hui."""
        count = len(self.sites)
        if count <= 1:
            return (
                "un seul site : aucun seuil n'est calibré, ils sont tous des "
                "hypothèses tirées de ce cas"
            )
        if count < MIN_SITES_FOR_SPREAD:
            return (
                f"{count} sites : les mesures se comparent, leur dispersion non"
            )
        if count < MIN_SITES_FOR_CALIBRATION:
            return (
                f"{count} sites : la dispersion se mesure, mais la variété reste "
                f"trop faible pour calibrer un seuil ({MIN_SITES_FOR_CALIBRATION} "
                "attendus)"
            )
        return f"{count} sites : la calibration d'un seuil devient défendable"

    def outliers(self, metric: str) -> list[str]:
        """Sites s'écartant de plus d'un écart-type de la médiane.

        Ce ne sont pas des erreurs : ce sont les cas qui révèlent qu'un seuil
        réglé ailleurs ne tient pas ici.
        """
        found = self.spread(metric)
        if not found.get("computable") or found["stdev"] == 0:
            return []
        return [
            site.hotel_id
            for site in self.sites
            if getattr(site, metric, None) is not None
            and abs(float(getattr(site, metric)) - found["median"]) > found["stdev"]
        ]

    def as_dict(self) -> dict:
        metrics = (
            "confirming_fraction",
            "triangulable_fraction",
            "connected_fraction",
        )
        return {
            "site_count": len(self.sites),
            "calibration_status": self.calibration_status(),
            "sites": [s.as_dict() for s in self.sites],
            "spreads": {m: self.spread(m) for m in metrics},
            "outliers": {m: self.outliers(m) for m in metrics},
            "caveats": [
                "deux chiffres ne se comparent pas sans leurs caractéristiques : "
                "un bâtiment mitoyen a deux murs inaccessibles par construction",
                "un seuil réglé sur un site reste une hypothèse tant qu'un "
                "second ne l'a pas éprouvé",
            ],
        }


def collect(workspace) -> SiteRecord:  # noqa: ANN001
    """Recueille les mesures d'un site depuis ce que le pipeline a déjà écrit.

    Aucune mesure n'est recalculée : le module lit les rapports produits par
    les commandes, et laisse vide ce qui n'a pas été mesuré. Un champ absent
    n'est pas un zéro — c'est une mesure qui n'a pas eu lieu.
    """
    import json

    record = SiteRecord(hotel_id=str(getattr(workspace, "hotel_id", "inconnu")))

    manifest = workspace.assets_path
    if manifest.is_file():
        try:
            record.assets_total = len(
                json.loads(manifest.read_text(encoding="utf-8")).get("assets", [])
            )
        except (OSError, ValueError):
            pass

    in_frame = workspace.path("06_geo", "in_frame.json")
    if in_frame.is_file():
        try:
            record.views_confirming = len(
                json.loads(in_frame.read_text(encoding="utf-8")).get("visible", [])
            )
        except (OSError, ValueError):
            pass

    observation = workspace.path("06_geo", "observation_map.json")
    if observation.is_file():
        try:
            payload = json.loads(observation.read_text(encoding="utf-8"))
            record.cells_total = int(payload.get("cell_count", 0))
            record.cells_triangulable = int(payload.get("triangulable_count", 0))
        except (OSError, ValueError):
            pass

    log.info(
        "%s : %d asset(s), %d vue(s) confirmante(s), %d/%d cellule(s)",
        record.hotel_id,
        record.assets_total,
        record.views_confirming,
        record.cells_triangulable,
        record.cells_total,
    )
    return record


__all__ = [
    "MIN_SITES_FOR_CALIBRATION",
    "MIN_SITES_FOR_SPREAD",
    "Benchmark",
    "SiteRecord",
    "collect",
]
