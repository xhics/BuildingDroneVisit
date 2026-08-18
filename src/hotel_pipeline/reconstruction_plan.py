"""Plan de reconstruction (Lot 2 — P2).

Ce module sélectionne les backends de reconstruction en fonction du
`ViewGraphReport`. Il est indépendant du Router Lot 1B : celui-ci
répond à « quelles preuves/matières ? », tandis que `ReconstructionPlan`
répond à « quel solveur ? ».
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .schemas.reconstruction import ReconstructionPlan, ReconstructionInputManifest, ViewGraphManifest
from .workspace import Workspace


class ReconstructionPlanner:
    """Sélectionne les backends de reconstruction selon le ViewGraphReport."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def plan(
        self,
        input_manifest: ReconstructionInputManifest,
        view_graph: ViewGraphManifest,
    ) -> ReconstructionPlan:
        """Construit un `ReconstructionPlan` depuis le graphe de vue.

        Règges :
        - COLMAP incremental toujours inclus comme baseline
        - COLMAP global ajouté si graphe dense (valid_pairs > 50)
        - GLUEMAP ajouté si risque répétitif medium/high
        - MP-SfM ajouté si overlap faible (< 20% registered_candidate_ratio)
        """
        backends = ["colmap_incremental"]
        rationale = "baseline systématique"

        if view_graph.report.valid_pairs > 50:
            backends.append("colmap_global")
            rationale += " + COLMAP global (graphe dense)"

        if view_graph.report.repetitive_risk in ("medium", "high"):
            backends.append("gluemap")
            rationale += " + GLUEMAP (structure répétitive)"

        if view_graph.report.registered_candidate_ratio < 0.2:
            backends.append("mpsfm")
            rationale += " + MP-SfM (faible overlap)"

        fallback = [b for b in backends if b != "colmap_incremental"]

        plan_id = (
            f"plan-{input_manifest.reconstruction_input_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )

        return ReconstructionPlan(
            plan_id=plan_id,
            reconstruction_input_id=input_manifest.reconstruction_input_id,
            view_graph_id=view_graph.view_graph_id,
            selected_backends=backends,
            fallback_chain=fallback,
            rationale=rationale,
        )


def publish_plan(plan: ReconstructionPlan, workspace: Workspace) -> Path:
    """Publie le ReconstructionPlan sous `07_reconstruction/plans/`."""
    output_dir = workspace.path("07_reconstruction", "plans")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{plan.plan_id}.json"
    output_path.write_text(json.dumps(plan.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n")
    return output_path


def load_plan(plan_id: str, workspace: Workspace) -> ReconstructionPlan | None:
    """Charge un `ReconstructionPlan` publié."""
    path = workspace.path("07_reconstruction", "plans", f"{plan_id}.json")
    if not path.is_file():
        return None
    return ReconstructionPlan.model_validate_json(path.read_text("utf-8"))


__all__ = [
    "ReconstructionPlanner",
    "publish_plan",
    "load_plan",
]
