"""Validations de reconstruction (Lot 2 — P5).

Ce module fournit trois validations obligatoires avant `ENVIRONMENT_3D_READY` :
- Held-out novel view : réserve 20% des images, rendu depuis poses cachées
- Stability : reconstruit avec 100%/90%/80% des images, vérifie stabilité Sim(3)
- Cross-solver consensus : vérifie que COLMAP/GLUEMAP/feed-forward ne divergent pas
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .schemas.reconstruction import ReconstructionRun
from .workspace import Workspace


class HeldOutValidation(BaseModel):
    """Validation par vue cachée."""

    model_config = ConfigDict(extra="forbid")

    validation_id: str
    reconstruction_run_id: str
    status: str
    metrics: dict = Field(default_factory=dict)
    error: str | None = None


class StabilityValidation(BaseModel):
    """Validation de stabilité par sous-échantillonage."""

    model_config = ConfigDict(extra="forbid")

    validation_id: str
    reconstruction_run_id: str
    status: str
    subsets: list[dict] = Field(default_factory=list)
    error: str | None = None


class CrossSolverValidation(BaseModel):
    """Validation de consensus cross-solver."""

    model_config = ConfigDict(extra="forbid")

    validation_id: str
    reconstruction_run_id: str
    status: str
    consensus_score: float = Field(default=0.0, ge=0.0, le=1.0)
    divergent_solvers: list[str] = Field(default_factory=list)
    error: str | None = None


class ReconstructionValidationReport(BaseModel):
    """Rapport global de validation."""

    model_config = ConfigDict(extra="forbid")

    validation_id: str
    reconstruction_run_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    held_out: HeldOutValidation | None = None
    stability: StabilityValidation | None = None
    cross_solver: CrossSolverValidation | None = None
    overall_status: str = "pending"


def validate_held_out(
    workspace: Workspace,
    reconstruction_run_id: str,
) -> HeldOutValidation:
    """Valide sur des vues cachées.

    Pour le MVP, c'est un placeholder.
    """
    validation_id = (
        f"heldout-{reconstruction_run_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    return HeldOutValidation(
        validation_id=validation_id,
        reconstruction_run_id=reconstruction_run_id,
        status="pending",
        error="validation held-out non implémentée dans cette phase",
    )


def validate_stability(
    workspace: Workspace,
    reconstruction_run_id: str,
) -> StabilityValidation:
    """Valide la stabilité par sous-échantillonage.

    Pour le MVP, c'est un placeholder.
    """
    validation_id = (
        f"stability-{reconstruction_run_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    return StabilityValidation(
        validation_id=validation_id,
        reconstruction_run_id=reconstruction_run_id,
        status="pending",
        error="validation stability non implémentée dans cette phase",
    )


def validate_cross_solver(
    workspace: Workspace,
    reconstruction_run_id: str,
) -> CrossSolverValidation:
    """Valide le consensus cross-solver.

    Pour le MVP, c'est un placeholder.
    """
    validation_id = (
        f"crosssolver-{reconstruction_run_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    return CrossSolverValidation(
        validation_id=validation_id,
        reconstruction_run_id=reconstruction_run_id,
        status="pending",
        consensus_score=0.0,
        error="validation cross-solver non implémentée dans cette phase",
    )


def build_validation_report(
    workspace: Workspace,
    reconstruction_run_id: str,
) -> ReconstructionValidationReport:
    """Construit le rapport de validation complet."""
    held_out = validate_held_out(workspace, reconstruction_run_id)
    stability = validate_stability(workspace, reconstruction_run_id)
    cross_solver = validate_cross_solver(workspace, reconstruction_run_id)

    statuses = [v.status for v in (held_out, stability, cross_solver) if v]
    overall = "passed" if all(s == "passed" for s in statuses) else "pending"

    validation_id = (
        f"validation-{reconstruction_run_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )

    return ReconstructionValidationReport(
        validation_id=validation_id,
        reconstruction_run_id=reconstruction_run_id,
        held_out=held_out,
        stability=stability,
        cross_solver=cross_solver,
        overall_status=overall,
    )


def publish_validation_report(
    report: ReconstructionValidationReport,
    workspace: Workspace,
) -> Path:
    """Publie le rapport de validation sous `07_reconstruction/validation/`."""
    output_dir = workspace.path("07_reconstruction", "validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{report.validation_id}.json"
    output_path.write_text(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n")
    return output_path


__all__ = [
    "HeldOutValidation",
    "StabilityValidation",
    "CrossSolverValidation",
    "ReconstructionValidationReport",
    "validate_held_out",
    "validate_stability",
    "validate_cross_solver",
    "build_validation_report",
    "publish_validation_report",
]
