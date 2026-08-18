"""Consensus et sélection de reconstruction (Lot 2 — P3).

Ce module compare plusieurs `ReconstructionRun`, aligne leurs poses en
Sim(3), et sélectionne la meilleure reconstruction selon des critères
quantitatifs. Il produit également un `ReconstructionConsensusReport`
et des entrées `CameraConsensusEntry` par image.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pydantic import BaseModel, ConfigDict, Field

from .schemas.reconstruction import (
    CameraConsensusEntry,
    ReconstructionConsensusReport,
    ReconstructionRun,
)
from .workspace import Workspace


class ConsensusBuilder:
    """Construit un `ReconstructionConsensusReport` depuis plusieurs runs."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def build(self, run_ids: list[str]) -> ReconstructionConsensusReport:
        runs = [self._load_run(rid) for rid in run_ids]
        runs = [r for r in runs if r is not None and r.status == "completed"]
        if len(runs) < 2:
            raise ValueError("au moins deux runs complétés sont nécessaires pour un consensus")

        pairwise = self._pairwise_alignment_errors(runs)
        camera_consensus = self._camera_consensus(runs)

        consensus_id = (
            f"consensus-{runs[0].reconstruction_input_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )

        selected = self._select_best_run(runs, pairwise, camera_consensus)

        return ReconstructionConsensusReport(
            consensus_id=consensus_id,
            reconstruction_input_id=runs[0].reconstruction_input_id,
            run_ids=[r.run_id for r in runs],
            pairwise_alignment_errors=pairwise,
            camera_consensus=camera_consensus,
            selected_run_id=selected.run_id if selected else None,
            selection_rationale=self._selection_rationale(selected, runs, pairwise) if selected else None,
        )

    def _load_run(self, run_id: str) -> ReconstructionRun | None:
        path = self.workspace.path("07_reconstruction", "runs", f"{run_id}.json")
        if not path.is_file():
            return None
        try:
            return ReconstructionRun.model_validate_json(path.read_text("utf-8"))
        except Exception:
            return None

    @staticmethod
    def _pairwise_alignment_errors(runs: list[ReconstructionRun]) -> dict[str, float]:
        """Estime l'erreur d'alignement Sim(3) entre chaque paire de runs.

        Pour le MVP, on utilise les métriques publiées par chaque run.
        """
        errors: dict[str, float] = {}
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                key = f"{runs[i].run_id}__{runs[j].run_id}"
                mi = runs[i].metrics.get("alignment_rmse_m", 1.0)
                mj = runs[j].metrics.get("alignment_rmse_m", 1.0)
                errors[key] = round(abs(mi - mj), 4) if isinstance(mi, (int, float)) and isinstance(mj, (int, float)) else 1.0
        return errors

    def _camera_consensus(self, runs: list[ReconstructionRun]) -> list[CameraConsensusEntry]:
        """Construit le consensus par image.

        Pour le MVP, on agrège les métriques de chaque run sans alignement
        pose-par-pose. Les entrées sont donc des proxies.
        """
        entries: list[CameraConsensusEntry] = []
        all_asset_ids: set[str] = set()
        for r in runs:
            if isinstance(r.metrics, dict):
                for aid in r.metrics.get("registered_assets", []):
                    all_asset_ids.add(aid)

        for asset_id in sorted(all_asset_ids):
            backends = []
            spreads_t = []
            spreads_r = []
            spreads_f = []
            aberrants = []

            for r in runs:
                if isinstance(r.metrics, dict):
                    asset_metrics = r.metrics.get("per_asset", {}).get(asset_id)
                    if asset_metrics:
                        backends.append(r.backend)
                        if "translation_spread_m" in asset_metrics:
                            spreads_t.append(asset_metrics["translation_spread_m"])
                        if "rotation_spread_deg" in asset_metrics:
                            spreads_r.append(asset_metrics["rotation_spread_deg"])
                        if "focal_spread_px" in asset_metrics:
                            spreads_f.append(asset_metrics["focal_spread_px"])

            confidence = "none"
            if len(backends) >= 3:
                confidence = "high"
            elif len(backends) == 2:
                confidence = "medium"

            entries.append(CameraConsensusEntry(
                asset_id=asset_id,
                backends=backends,
                translation_spread_m=round(float(np.mean(spreads_t)), 3) if spreads_t else 0.0,
                rotation_spread_deg=round(float(np.mean(spreads_r)), 3) if spreads_r else 0.0,
                focal_spread_px=round(float(np.mean(spreads_f)), 3) if spreads_f else 0.0,
                confidence=confidence,
                aberrants=aberrants,
            ))

        return entries

    @staticmethod
    def _select_best_run(
        runs: list[ReconstructionRun],
        pairwise: dict[str, float],
        consensus: list[CameraConsensusEntry],
    ) -> ReconstructionRun | None:
        """Sélectionne le meilleur run selon les métriques."""
        scored = []
        for r in runs:
            m = r.metrics if isinstance(r.metrics, dict) else {}
            registered = m.get("registered_ratio", 0.0)
            error = m.get("alignment_rmse_m", 1.0)
            score = registered - error
            scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored else None

    @staticmethod
    def _selection_rationale(
        selected: ReconstructionRun,
        runs: list[ReconstructionRun],
        pairwise: dict[str, float],
    ) -> str:
        m = selected.metrics if isinstance(selected.metrics, dict) else {}
        return (
            f"run {selected.run_id} sélectionné : "
            f"registered_ratio={m.get('registered_ratio', 0):.2f}, "
            f"alignment_rmse={m.get('alignment_rmse_m', 0):.3f}m"
        )


def publish_consensus(
    report: ReconstructionConsensusReport,
    workspace: Workspace,
) -> Path:
    """Publie le rapport de consensus sous `07_reconstruction/consensus/`."""
    path = workspace.path("07_reconstruction", "consensus", f"{report.consensus_id}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n")
    return path


__all__ = [
    "ConsensusBuilder",
    "publish_consensus",
]
