"""Reconstruction dense (Lot 2 — P5).

Ce module place les briques de densification (Brush, 3DGS) **après**
sélection du sparse/poses et alignement géospatial. Il ne s'exécute
qu'une fois la consensus et l'alignement validés.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .schemas.reconstruction import ReconstructionBackend, ReconstructionRun
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
    """Exécute la reconstruction dense sur le run sélectionné."""
    result_id = (
        f"dense-{backend.value}-{reconstruction_run_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )

    if backend is ReconstructionBackend.BRUSH:
        return _run_brush(workspace, reconstruction_run_id, result_id)
    if backend is ReconstructionBackend.GSPLAT:
        return _run_gsplat(workspace, reconstruction_run_id, result_id)

    return DenseReconstructionResult(
        result_id=result_id,
        reconstruction_run_id=reconstruction_run_id,
        backend=backend.value,
        status="failed",
        error=f"backend {backend.value} non supporté pour la reconstruction dense",
    )


def _load_sparse_run(reconstruction_run_id: str, workspace: Workspace) -> tuple[Path, dict] | None:
    """Charge le run sparse sélectionné et retourne (run_dir, data)."""
    run_json_path = workspace.path("07_reconstruction", "runs", f"{reconstruction_run_id}.json")
    if not run_json_path.is_file():
        return None
    try:
        data = json.loads(run_json_path.read_text("utf-8"))
        return Path(data.get("output_path", "")), data
    except Exception:
        return None


def _run_brush(
    workspace: Workspace,
    reconstruction_run_id: str,
    result_id: str,
) -> DenseReconstructionResult:
    """Exécute Brush sur la reconstruction sparse."""
    started = datetime.now(timezone.utc).isoformat()
    try:
        sparse_info = _load_sparse_run(reconstruction_run_id, workspace)
        if sparse_info is None:
            return DenseReconstructionResult(
                result_id=result_id,
                reconstruction_run_id=reconstruction_run_id,
                backend="brush",
                status="failed",
                started_at=started,
                finished_at=datetime.now(timezone.utc).isoformat(),
                error="run sparse introuvable",
            )

        sparse_dir, _ = sparse_info
        output_dir = workspace.path("07_reconstruction", "dense", "brush", result_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "brush",
            "--sparse_model", str(sparse_dir),
            "--output_path", str(output_dir),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)

        output_path = str(output_dir) if output_dir.exists() and any(output_dir.iterdir()) else None
        error = None if proc.returncode == 0 and output_path else (proc.stderr.strip() or proc.stdout.strip() or "brush a échoué")

        return DenseReconstructionResult(
            result_id=result_id,
            reconstruction_run_id=reconstruction_run_id,
            backend="brush",
            status="completed" if output_path else "failed",
            started_at=started,
            finished_at=datetime.now(timezone.utc).isoformat(),
            output_path=output_path,
            error=error,
        )
    except FileNotFoundError as exc:
        if "brush" in str(exc):
            return DenseReconstructionResult(
                result_id=result_id,
                reconstruction_run_id=reconstruction_run_id,
                backend="brush",
                status="failed",
                started_at=started,
                finished_at=datetime.now(timezone.utc).isoformat(),
                error="binaire Brush introuvable",
            )
        raise
    except Exception as exc:
        return DenseReconstructionResult(
            result_id=result_id,
            reconstruction_run_id=reconstruction_run_id,
            backend="brush",
            status="failed",
            started_at=started,
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
        )


def _run_gsplat(
    workspace: Workspace,
    reconstruction_run_id: str,
    result_id: str,
) -> DenseReconstructionResult:
    """Exécute gsplat sur la reconstruction sparse."""
    started = datetime.now(timezone.utc).isoformat()
    try:
        sparse_info = _load_sparse_run(reconstruction_run_id, workspace)
        if sparse_info is None:
            return DenseReconstructionResult(
                result_id=result_id,
                reconstruction_run_id=reconstruction_run_id,
                backend="gsplat",
                status="failed",
                started_at=started,
                finished_at=datetime.now(timezone.utc).isoformat(),
                error="run sparse introuvable",
            )

        sparse_dir, _ = sparse_info
        output_dir = workspace.path("07_reconstruction", "dense", "gsplat", result_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "gsplat",
            "--sparse_model", str(sparse_dir),
            "--output_path", str(output_dir),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)

        output_path = str(output_dir) if output_dir.exists() and any(output_dir.iterdir()) else None
        error = None if proc.returncode == 0 and output_path else (proc.stderr.strip() or proc.stdout.strip() or "gsplat a échoué")

        return DenseReconstructionResult(
            result_id=result_id,
            reconstruction_run_id=reconstruction_run_id,
            backend="gsplat",
            status="completed" if output_path else "failed",
            started_at=started,
            finished_at=datetime.now(timezone.utc).isoformat(),
            output_path=output_path,
            error=error,
        )
    except FileNotFoundError as exc:
        if "gsplat" in str(exc):
            return DenseReconstructionResult(
                result_id=result_id,
                reconstruction_run_id=reconstruction_run_id,
                backend="gsplat",
                status="failed",
                started_at=started,
                finished_at=datetime.now(timezone.utc).isoformat(),
                error="binaire gsplat introuvable",
            )
        raise
    except Exception as exc:
        return DenseReconstructionResult(
            result_id=result_id,
            reconstruction_run_id=reconstruction_run_id,
            backend="gsplat",
            status="failed",
            started_at=started,
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
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
