"""Projection probabiliste d'un gain de fidélité, et sa calibration.

Une projection déterministe annonce un chiffre unique — « cette source vaut
+38,9 points » — sans dire ce qui pourrait le faire varier. Or deux inconnues
pèsent avant tout téléchargement :

- **la couverture** : la tuile contient-elle vraiment les volumes visés ? La
  découverte donne une borne géométrique, pas une certitude ;
- **le rendement** : un volume couvert est-il effectivement mesurable ? Cela
  dépend de la densité de points et de la classification, qui varient d'une
  tuile à l'autre.

Le module tire des scénarios sur ces deux inconnues et rend une **distribution**
de gains, non un point. Il enregistre ensuite le réalisé, pour que les
hypothèses de départ se corrigent sur des faits plutôt que sur un jugement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..logging import get_logger

log = get_logger("conditioning-projection")

#: Nombre de scénarios tirés. Mille suffit à stabiliser les quantiles au
#: dixième de point, et se calcule instantanément.
SCENARIOS = 1000

#: Rendement d'une source, en part des volumes couverts qui deviennent
#: mesurables. Ce sont les valeurs de départ, remplacées dès qu'un réalisé est
#: enregistré : elles disent une attente, pas une loi.
DEFAULT_YIELD: dict[str, tuple[float, float]] = {
    # (rendement moyen, écart-type)
    "lidar_cloud": (0.90, 0.10),
    "lidar_ndsm": (0.95, 0.05),
    "image_inferred": (0.55, 0.20),
}

#: Incertitude sur la couverture annoncée par une découverte. Une emprise
#: calculée sur des boîtes englobantes déborde parfois de la tuile réelle.
COVERAGE_SIGMA = 0.08

#: Fichier où s'accumulent les projections et leur réalisé.
LEDGER_FILE = "09_confidence/projection_ledger.json"


@dataclass
class ProbabilisticGain:
    """Distribution du gain d'une source, en points de fidélité."""

    levier: str
    samples: np.ndarray
    coverage: float
    yield_mean: float
    yield_sigma: float

    def quantile(self, q: float) -> float:
        return float(np.percentile(self.samples, q * 100))

    @property
    def median(self) -> float:
        return self.quantile(0.5)

    def as_dict(self) -> dict:
        return {
            "levier": self.levier,
            "median_points": round(self.median, 2),
            "p10_points": round(self.quantile(0.10), 2),
            "p90_points": round(self.quantile(0.90), 2),
            "coverage": round(self.coverage, 3),
            "yield_mean": round(self.yield_mean, 3),
            "yield_sigma": round(self.yield_sigma, 3),
            "scenarios": int(self.samples.size),
        }


def simulate(
    levier: str,
    surface_concernee: float,
    surface_totale: float,
    fidelite_actuelle: float,
    fidelite_visee: float,
    coverage: float,
    source: str = "lidar_cloud",
    yields: dict[str, tuple[float, float]] | None = None,
    scenarios: int = SCENARIOS,
    seed: int = 0,
) -> ProbabilisticGain:
    """Tire des scénarios de gain sur la couverture et le rendement.

    Le gain d'un scénario vaut la part de surface effectivement portée par la
    nouvelle source, multipliée par le progrès de fidélité qu'elle apporte.
    Deux facteurs aléatoires s'y composent : la fraction des volumes que la
    tuile contient réellement, et celle qui en devient mesurable.
    """
    if surface_totale <= 0 or surface_concernee <= 0:
        return ProbabilisticGain(levier, np.zeros(1), coverage, 0.0, 0.0)

    mean, sigma = (yields or DEFAULT_YIELD).get(source, (0.85, 0.12))
    rng = np.random.default_rng(seed)

    # La couverture annoncée est une borne : elle se réalise à quelques points
    # près, jamais au-delà de un.
    covered = np.clip(rng.normal(coverage, COVERAGE_SIGMA, scenarios), 0.0, 1.0)
    yielded = np.clip(rng.normal(mean, sigma, scenarios), 0.0, 1.0)

    part = (surface_concernee / surface_totale) * covered * yielded
    gains = part * (fidelite_visee - fidelite_actuelle) * 100.0

    result = ProbabilisticGain(levier, gains, coverage, mean, sigma)
    log.info(
        "%s : gain médian %.1f pts (p10 %.1f, p90 %.1f)",
        levier,
        result.median,
        result.quantile(0.10),
        result.quantile(0.90),
    )
    return result


@dataclass
class LedgerEntry:
    """Une projection et, quand il est connu, ce qu'elle a réellement donné."""

    hotel_id: str
    levier: str
    source: str
    predicted_median: float
    predicted_p10: float
    predicted_p90: float
    coverage: float
    score_before: float
    recorded_at: str
    realised_points: float | None = None
    score_after: float | None = None

    @property
    def error(self) -> float | None:
        """Écart entre le réalisé et la médiane projetée, en points."""
        if self.realised_points is None:
            return None
        return self.realised_points - self.predicted_median

    @property
    def within_interval(self) -> bool | None:
        """Le réalisé tombe-t-il dans l'intervalle annoncé ?"""
        if self.realised_points is None:
            return None
        return self.predicted_p10 <= self.realised_points <= self.predicted_p90

    def as_dict(self) -> dict:
        return {
            "hotel_id": self.hotel_id,
            "levier": self.levier,
            "source": self.source,
            "predicted_median": round(self.predicted_median, 2),
            "predicted_p10": round(self.predicted_p10, 2),
            "predicted_p90": round(self.predicted_p90, 2),
            "coverage": round(self.coverage, 3),
            "score_before": round(self.score_before, 4),
            "recorded_at": self.recorded_at,
            "realised_points": (
                None if self.realised_points is None else round(self.realised_points, 2)
            ),
            "score_after": (
                None if self.score_after is None else round(self.score_after, 4)
            ),
            "error": None if self.error is None else round(self.error, 2),
            "within_interval": self.within_interval,
        }


@dataclass
class Ledger:
    """Historique des projections, et calibration qu'il permet."""

    entries: list[LedgerEntry] = field(default_factory=list)

    def closed(self) -> list[LedgerEntry]:
        return [e for e in self.entries if e.realised_points is not None]

    def calibration(self) -> dict[str, tuple[float, float]]:
        """Rendements corrigés par source, d'après les réalisés enregistrés.

        Le rendement observé se déduit du gain : un réalisé supérieur à la
        prévision signifie que la source a mieux porté qu'attendu. La
        correction reste **prudente** — elle se mélange à l'attente de départ
        plutôt que de la remplacer — parce qu'un seul site ne fait pas une loi.
        """
        by_source: dict[str, list[float]] = {}
        for entry in self.closed():
            if entry.predicted_median <= 0:
                continue
            ratio = entry.realised_points / entry.predicted_median
            by_source.setdefault(entry.source, []).append(ratio)

        calibrated: dict[str, tuple[float, float]] = {}
        for source, ratios in by_source.items():
            base_mean, base_sigma = DEFAULT_YIELD.get(source, (0.85, 0.12))
            observed = float(np.median(ratios))
            # Poids croissant avec le nombre d'observations : une seule mesure
            # infléchit à peine, dix comptent nettement.
            weight = min(len(ratios) / 10.0, 0.8)
            mean = np.clip(base_mean * (1 - weight) + base_mean * observed * weight,
                           0.05, 1.0)
            # L'écart-type suit la dispersion constatée, jamais sous un plancher.
            sigma = max(
                float(np.std(ratios)) * base_mean if len(ratios) > 1 else base_sigma,
                0.03,
            )
            calibrated[source] = (float(mean), float(sigma))
        return calibrated

    def coverage_bias(self) -> float:
        """Correction à appliquer à la couverture annoncée par la découverte.

        Le rendement plafonne à un : quand un réalisé le dépasse largement,
        ce n'est pas la source qui a mieux porté, c'est la **couverture** qui
        était sous-estimée. Mesuré sur ce pilote : une couverture annoncée à
        78 % pour un réalisé qui supposait la totalité, parce que l'estimation
        raisonnait sur la distance au centre sans savoir quelles tuiles
        seraient effectivement acquises.

        Le biais est le rapport médian réalisé/projeté, borné : il corrige une
        estimation systématiquement prudente sans jamais promettre plus que
        ce qu'une emprise peut contenir.
        """
        closed = [e for e in self.closed() if e.predicted_median > 0]
        if not closed:
            return 1.0
        ratios = [e.realised_points / e.predicted_median for e in closed]
        weight = min(len(ratios) / 10.0, 0.8)
        observed = float(np.median(ratios))
        return float(np.clip(1.0 * (1 - weight) + observed * weight, 0.5, 2.0))

    def summary(self) -> dict:
        closed = self.closed()
        if not closed:
            return {"entries": len(self.entries), "closed": 0}
        errors = [e.error for e in closed]
        inside = sum(1 for e in closed if e.within_interval)
        return {
            "entries": len(self.entries),
            "closed": len(closed),
            "mean_error_points": round(float(np.mean(errors)), 2),
            "median_error_points": round(float(np.median(errors)), 2),
            "within_interval": f"{inside}/{len(closed)}",
            "calibration": {
                k: [round(v[0], 3), round(v[1], 3)]
                for k, v in self.calibration().items()
            },
            "coverage_bias": round(self.coverage_bias(), 3),
        }

    def as_dict(self) -> dict:
        return {
            "summary": self.summary(),
            "entries": [e.as_dict() for e in self.entries],
            "caveats": [
                "un réalisé mesure ce qu'une source a donné sur CE site : la "
                "calibration ne devient significative qu'après plusieurs",
                "l'intervalle décrit l'incertitude modélisée — couverture et "
                "rendement — non toutes les façons dont une acquisition peut "
                "décevoir",
            ],
        }


def load_ledger(workspace) -> Ledger:  # noqa: ANN001
    """Relit le registre des projections d'un site."""
    path = workspace.path(*LEDGER_FILE.split("/"))
    if not path.is_file():
        return Ledger()
    payload = json.loads(path.read_text(encoding="utf-8"))
    ledger = Ledger()
    for raw in payload.get("entries", []):
        ledger.entries.append(
            LedgerEntry(
                hotel_id=raw["hotel_id"],
                levier=raw["levier"],
                source=raw["source"],
                predicted_median=float(raw["predicted_median"]),
                predicted_p10=float(raw["predicted_p10"]),
                predicted_p90=float(raw["predicted_p90"]),
                coverage=float(raw["coverage"]),
                score_before=float(raw["score_before"]),
                recorded_at=raw["recorded_at"],
                realised_points=raw.get("realised_points"),
                score_after=raw.get("score_after"),
            )
        )
    return ledger


def save_ledger(workspace, ledger: Ledger) -> Path:  # noqa: ANN001
    """Publie le registre."""
    return workspace.write_json(LEDGER_FILE, ledger.as_dict())


def record(
    workspace,  # noqa: ANN001
    hotel_id: str,
    gain: ProbabilisticGain,
    source: str,
    score_before: float,
) -> Ledger:
    """Enregistre une projection, en attente de son réalisé."""
    ledger = load_ledger(workspace)
    ledger.entries.append(
        LedgerEntry(
            hotel_id=hotel_id,
            levier=gain.levier,
            source=source,
            predicted_median=gain.median,
            predicted_p10=gain.quantile(0.10),
            predicted_p90=gain.quantile(0.90),
            coverage=gain.coverage,
            score_before=score_before,
            recorded_at=datetime.now(timezone.utc).isoformat(),
        )
    )
    save_ledger(workspace, ledger)
    return ledger


def close(
    workspace,  # noqa: ANN001
    levier: str,
    score_after: float,
) -> LedgerEntry | None:
    """Referme la projection ouverte d'un levier, avec le score constaté."""
    ledger = load_ledger(workspace)
    pending = [
        e for e in ledger.entries
        if e.levier == levier and e.realised_points is None
    ]
    if not pending:
        log.info("aucune projection ouverte pour %r", levier)
        return None

    entry = pending[-1]
    entry.score_after = score_after
    entry.realised_points = (score_after - entry.score_before) * 100.0
    save_ledger(workspace, ledger)
    log.info(
        "%s : projeté %.1f, réalisé %.1f (écart %+.1f)",
        levier,
        entry.predicted_median,
        entry.realised_points,
        entry.error,
    )
    return entry
