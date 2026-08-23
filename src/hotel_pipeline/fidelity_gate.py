"""Porte finale de fidélité (Lot 2 — Porte D).

État final d'une cible. Les états préservent l'inconnu : `INSUFFICIENT_EVIDENCE`
n'est pas un échec géométrique démontré, c'est une impossibilité de conclure.
Pour un MUST_SHOW, cela bloque quand même le pipeline, mais avec une raison
explicite. `NOT_APPLICABLE` = cible hors périmètre courant.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from .schemas.reconstruction import (
    Criticality,
    FidelityGate,
    GateResult,
    GeoAlignmentGate,
    NovelViewValidationGate,
    ReconstructionTarget,
    SparseConsensusGate,
    StabilityResult,
)
from .workspace import Workspace


def _sparse_gate_passes(gate: SparseConsensusGate | None) -> bool:
    if gate is None:
        return False
    return (
        gate.validated_registration_rate >= 0.60
        and gate.validated_main_component_ratio >= 0.70
        and gate.external_pose_consistency
        and gate.largest_component_size >= 3
        and gate.median_reprojection_px < 5.0
    )


def _geo_gate_passes(gate: GeoAlignmentGate | None) -> bool:
    if gate is None:
        return False
    return gate.alignment_rmse_m < 2.0 and gate.footprint_error_m < 2.0


def _novel_gate_passes(gate: NovelViewValidationGate | None) -> bool:
    if gate is None:
        return False
    # Des métriques non mesurées ne valent pas un passage : avec les seuils
    # permissifs par défaut, les valeurs par défaut du schéma passeraient
    # toutes les comparaisons ci-dessous sans qu'aucun rendu n'ait eu lieu.
    if not gate.metrics_measured:
        return False
    c = gate.pass_criteria
    return (
        gate.feature_inliers >= c.feature_inliers_min
        and gate.ssim >= c.ssim_min
        and gate.reprojection_px <= c.reprojection_px_max
        and gate.structural_similarity >= c.structural_similarity_min
    )


def evaluate_fidelity(
    target: ReconstructionTarget,
    *,
    sparse_gate: SparseConsensusGate | None = None,
    geo_gate: GeoAlignmentGate | None = None,
    novel_view_gate: NovelViewValidationGate | None = None,
    stability_gate: StabilityResult | None = None,
    unsupported_geometry_gate: bool = False,
) -> FidelityGate:
    """Évalue la porte de fidélité pour une cible.

    Règles :
    - MUST_SHOW : toutes les portes doivent PASSER ; INSUFFICIENT_EVIDENCE
      bloque aussi (raison enregistrée).
    - SHOULD_SHOW : dense + novel_view doivent PASSER.
    - OPTIONAL : best-effort.
    - unsupported_geometry (géométrie non mesurée sur cible MUST_SHOW) → échec.
    """
    # `missing` ne contient que des None : `any(missing)` serait toujours faux
    # et ferait tomber une porte absente en FAIL, c'est-à-dire un échec
    # géométrique démontré. On teste donc la liste elle-même.
    missing = [g for g in (sparse_gate, geo_gate, novel_view_gate) if g is None]
    has_insufficient = (
        stability_gate is StabilityResult.INSUFFICIENT_EVIDENCE
        or (novel_view_gate is not None and not novel_view_gate.metrics_measured)
        # Une cible dont la géométrie n'est qu'**inférée** n'est pas une cible
        # en échec : c'est une cible dont on ignore si la reconstruction porte
        # sur la bonne surface. La distinguer d'un échec démontré est tout
        # l'objet de `INSUFFICIENT_EVIDENCE`.
        #
        # Auparavant, une géométrie inférée rétrogradait la cible en
        # CONTEXT_ONLY et la porte répondait NOT_APPLICABLE : l'exigence
        # disparaissait au lieu de se dire.
        or not target.geometry_confirmed
    )

    if target.criticality is Criticality.MUST_SHOW:
        if unsupported_geometry_gate:
            overall = GateResult.FAIL
        elif missing or has_insufficient:
            overall = GateResult.INSUFFICIENT_EVIDENCE
        elif (
            _sparse_gate_passes(sparse_gate)
            and _geo_gate_passes(geo_gate)
            and _novel_gate_passes(novel_view_gate)
        ):
            overall = GateResult.PASS
        else:
            overall = GateResult.FAIL
    elif target.criticality is Criticality.SHOULD_SHOW:
        # SHOULD_SHOW ne dépend que de geo + novel_view ; l'absence de la porte
        # creuse ne rend pas le verdict non concluant.
        needed_missing = [g for g in (geo_gate, novel_view_gate) if g is None]
        if needed_missing or has_insufficient:
            overall = GateResult.INSUFFICIENT_EVIDENCE
        elif _novel_gate_passes(novel_view_gate) and _geo_gate_passes(geo_gate):
            overall = GateResult.PASS
        else:
            overall = GateResult.FAIL
    elif target.criticality is Criticality.OPTIONAL:
        overall = GateResult.PASS if _novel_gate_passes(novel_view_gate) else GateResult.INSUFFICIENT_EVIDENCE
    elif target.criticality is Criticality.FORBIDDEN:
        overall = GateResult.NOT_APPLICABLE
    else:  # CONTEXT_ONLY
        overall = GateResult.NOT_APPLICABLE

    return FidelityGate(
        target_id=target.target_id,
        criticality=target.criticality,
        sparse_gate=sparse_gate,
        geo_gate=geo_gate,
        novel_view_gate=novel_view_gate,
        stability_gate=stability_gate,
        unsupported_geometry_gate=unsupported_geometry_gate,
        overall=overall,
    )


def evaluate_targets(
    targets: list[ReconstructionTarget],
    gates: dict[str, dict[str, Any]],
) -> list[FidelityGate]:
    """Évalue plusieurs cibles depuis un dictionnaire de portes.

    `gates[target_id]` contient les clés optionnelles sparse_gate, geo_gate,
    novel_view_gate, stability_gate, unsupported_geometry_gate.
    """
    results = []
    for target in targets:
        g = gates.get(target.target_id, {})
        results.append(
            evaluate_fidelity(
                target,
                sparse_gate=g.get("sparse_gate"),
                geo_gate=g.get("geo_gate"),
                novel_view_gate=g.get("novel_view_gate"),
                stability_gate=g.get("stability_gate"),
                unsupported_geometry_gate=g.get("unsupported_geometry_gate", False),
            )
        )
    return results


def publish_fidelity_gates(gates: list[FidelityGate], workspace: Workspace) -> Path:
    """Publie les portes sous `07_reconstruction/fidelity/`."""
    output_dir = workspace.path("07_reconstruction", "fidelity")
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"fidelity_{ts}.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gates": [g.model_dump(mode="json") for g in gates],
        "blocked": [g.target_id for g in gates if g.overall in (GateResult.FAIL, GateResult.INSUFFICIENT_EVIDENCE)],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path


__all__ = [
    "evaluate_fidelity",
    "evaluate_targets",
    "publish_fidelity_gates",
]
