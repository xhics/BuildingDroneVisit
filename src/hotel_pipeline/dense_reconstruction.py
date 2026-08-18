"""Reconstruction dense (Lot 2 — P5).

Ce module place les briques de densification (Brush, 3DGS) **après**
sélection du sparse/poses et alignement géospatial. Il ne s'exécute
qu'une fois la consensus et l'alignement validés.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .schemas.reconstruction import ReconstructionBackend
from .workspace import Workspace


class DenseReconstructionResult(BaseModel):
    """Résultat d'une reconstruction dense."""

    model_config = ConfigDict(extra="forbid")

    result_id: str
    reconstruction_run_id: str
    backend: str
    status: str
    output_path: str | None = None
    metrics: dict = Field(default_factory=dict)
    error: str | None = None


def run_dense_reconstruction(
    workspace: Workspace,
    reconstruction_run_id: str,
    *,
    backend: ReconstructionBackend = ReconstructionBackend.BRUSH,
) -> DenseReconstructionResult:
    """Exécute la reconstruction dense sur le run sélectionné.

    Pour le MVP, tous les backends sont des placeholders qui journalisent
    l'intention. L'intégration réelle de Brush / gsplat viendra en P5
    quand le solver sélectionné sera validé.
    """
    result_id = (
        f"dense-{backend.value}-{reconstruction_run_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    return DenseReconstructionResult(
        result_id=result_id,
        reconstruction_run_id=reconstruction_run_id,
        backend=backend.value,
        status="pending",
        error=f"backend {backend.value} non implémenté dans cette phase",
    )


def publish_dense_result(result: DenseReconstructionResult, workspace: Workspace) -> Path:
    """Publie le résultat dense sous `07_reconstruction/dense/`."""
    path = workspace.path("07_reconstruction", "dense", f"{result.result_id}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n")
    return path


__all__ = [
    "DenseReconstructionResult",
    "run_dense_reconstruction",
    "publish_dense_result",
]
