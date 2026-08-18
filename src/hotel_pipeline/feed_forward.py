"""Vérificateurs feed-forward pour le Lot 2 — P3.

Ce module fournit des vérificateurs indépendants (MapAnything, VGGT)
qui produisent des poses et de la géométrie sans dépendre du solveur
SfM principal. Leurs sorties sont normalisées vers les mêmes schémas
que les backends classiques.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .schemas.reconstruction import ReconstructionBackend, ReconstructionInputManifest, ReconstructionRun
from .workspace import Workspace
from .reconstruction_run import ReconstructionRunner, publish_run


class FeedForwardRun(BaseModel):
    """Exécution d'un vérificateur feed-forward."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    backend: str
    reconstruction_input_id: str
    status: str
    output_path: str | None = None
    metrics: dict = Field(default_factory=dict)
    error: str | None = None


def run_mapanything(
    workspace: Workspace,
    input_manifest: ReconstructionInputManifest,
) -> FeedForwardRun:
    """Exécute MapAnything en mode vérification."""
    runner = ReconstructionRunner(workspace)
    run_id = _new_run_id("mapanything", input_manifest.reconstruction_input_id)
    run = runner.run_map_anything(input_manifest, run_id)
    publish_run(run, workspace)
    return _to_feed_forward_run(run)


def run_vggt(
    workspace: Workspace,
    input_manifest: ReconstructionInputManifest,
) -> FeedForwardRun:
    """Exécute VGGT en mode vérification."""
    runner = ReconstructionRunner(workspace)
    run_id = _new_run_id("vggt", input_manifest.reconstruction_input_id)
    run = runner.run_vggt(input_manifest, run_id)
    publish_run(run, workspace)
    return _to_feed_forward_run(run)


def publish_feed_forward(run: FeedForwardRun, workspace: Workspace) -> Path:
    """Publie un `FeedForwardRun` sous `07_reconstruction/feed_forward/`."""
    output_dir = workspace.path("07_reconstruction", "feed_forward")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run.run_id}.json"
    output_path.write_text(json.dumps(run.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n")
    return output_path


def _new_run_id(backend: str, input_id: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"run-{backend}-{input_id}-{stamp}"


def _to_feed_forward_run(run: ReconstructionRun) -> FeedForwardRun:
    """Convertit un ReconstructionRun en FeedForwardRun."""
    return FeedForwardRun(
        run_id=run.run_id,
        backend=run.backend,
        reconstruction_input_id=run.reconstruction_input_id,
        status=run.status,
        output_path=run.output_path,
        metrics=run.metrics,
        error=run.error,
    )


__all__ = [
    "FeedForwardRun",
    "run_mapanything",
    "run_vggt",
    "publish_feed_forward",
]
