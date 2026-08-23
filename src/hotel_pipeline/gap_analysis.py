"""Analyse des lacunes de reconstruction (Lot 2 — P4.5).

Consomme FidelityGate + ViewGraph + SurfaceConfidence pour produire des
`ReconstructionGap` **structurées**. C'est cet objet — pas un vague
« besoin de plus de photos » — qui génère le prochain `CaptureDemand`.
Chaque `gap_type` entraîne une observation ciblée (secteur, baseline, angle
de vue préférés).

Deux règles tiennent ce module :

1. La géométrie demandée est **dérivée**, pas constante. Une baseline de 10 m
   répétée pour toute lacune ne demande rien de particulier ; c'est du
   remplissage. La baseline vient de l'étendue du nuage caméra observé et du
   type de lacune (une faible parallaxe exige un écart plus large qu'une
   simple vue de renfort).
2. Les assets affectés sont **localisés**. Lister tous les nœuds du graphe
   pour chaque lacune n'oriente aucune collecte : on ne retient que les nœuds
   réellement en cause (non enregistrés, isolés, ou hors de la plus grande
   composante).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .schemas.reconstruction import (
    FidelityGate,
    GapType,
    ReconstructionGap,
    ViewGraphManifest,
)
from .workspace import Workspace

#: Azimut d'observation par secteur, relatif à la façade avant. Aligné sur
#: `demand_targets.SECTOR_BEARINGS` : deux tables divergeraient tôt ou tard.
from .demand_targets import SECTOR_BEARINGS

#: Secteur d'observation par cible. La table reste explicite (une cible n'a
#: pas toujours de secteur naturel), mais l'azimut, lui, est dérivé.
_TARGET_SECTORS: dict[str, str] = {
    "FACADE_PRIMARY": "front",
    "FACADE_LEFT": "left",
    "FACADE_RIGHT": "right",
    "FACADE_REAR": "rear",
    "ENTRANCE": "front",
}

#: Multiplicateur de baseline par type de lacune, appliqué à l'étendue
#: observée du nuage caméra.
_BASELINE_FACTOR: dict[GapType, float] = {
    GapType.LOW_PARALLAX: 0.60,       # écarter franchement les points de vue
    GapType.DISCONNECTED_GRAPH: 0.35,  # rapprocher pour reconnecter le graphe
    GapType.LOW_SUPPORT: 0.45,
    GapType.POSE_UNCERTAINTY: 0.30,
    GapType.APPEARANCE_GAP: 0.25,
    GapType.GEO_MISMATCH: 0.50,
}

#: Baseline de repli (m) quand aucune étendue caméra n'est observable.
_FALLBACK_BASELINE_M = 8.0


def _assign_gap_type(gate: FidelityGate, view_graph: ViewGraphManifest | None) -> GapType:
    """Déduit le type de lacune depuis la porte et le graphe de vue."""
    if view_graph is not None:
        valid = [p for p in view_graph.pairs if p.status == "valid"]
        if len(valid) == 0:
            return GapType.DISCONNECTED_GRAPH
        if view_graph.report.largest_component < 3:
            return GapType.DISCONNECTED_GRAPH
    if gate.sparse_gate is not None and gate.sparse_gate.largest_component_size < 3:
        return GapType.DISCONNECTED_GRAPH
    if gate.geo_gate is not None and (
        gate.geo_gate.alignment_rmse_m >= 2.0 or gate.geo_gate.footprint_error_m >= 2.0
    ):
        return GapType.GEO_MISMATCH
    if gate.novel_view_gate is None:
        return GapType.LOW_SUPPORT
    if not gate.novel_view_gate.metrics_measured:
        # Rien n'a été mesuré : le manque est un manque de support, pas une
        # divergence d'apparence démontrée.
        return GapType.LOW_SUPPORT
    if gate.overall.value == "fail":
        return GapType.APPEARANCE_GAP
    return GapType.LOW_PARALLAX


def _sector_for_target(target_id: str) -> str:
    """Secteur préféré pour capturer une cible, ou `unknown` si non établi."""
    return _TARGET_SECTORS.get(target_id, "unknown")


def _view_angle_for_sector(sector: str, front_azimuth_deg: float | None) -> float:
    """Azimut d'observation absolu pour un secteur.

    Sans orientation de façade connue, l'angle absolu n'est pas calculable :
    on retourne 0.0 et le secteur reste la seule consigne exploitable.
    """
    offset = SECTOR_BEARINGS.get(sector)
    if offset is None or front_azimuth_deg is None:
        return 0.0
    return (front_azimuth_deg + offset) % 360.0


def _camera_extent_m(view_graph: ViewGraphManifest | None) -> float | None:
    """Étendue observée du dispositif de prise de vue, si mesurable.

    On approxime l'étendue par le nombre de nœuds enregistrés : sans poses
    métriques dans le manifeste, une distance réelle n'est pas disponible.
    Retourne None quand rien n'est observable — la baseline retombe alors sur
    une valeur de repli déclarée, pas sur une constante déguisée en mesure.
    """
    if view_graph is None or not view_graph.nodes:
        return None
    registered = [n for n in view_graph.nodes if n.pose_status == "registered"]
    if len(registered) < 2:
        return None
    # Heuristique déclarée : une façade d'hôtel couverte par N vues
    # enregistrées s'étend sur l'ordre de 4 m par vue, plafonnée à 40 m.
    return min(4.0 * len(registered), 40.0)


def _affected_assets(
    gate: FidelityGate,
    gap_type: GapType,
    view_graph: ViewGraphManifest | None,
) -> list[str]:
    """Nœuds réellement en cause dans la lacune.

    Lister tout le graphe n'orienterait aucune collecte.
    """
    if view_graph is None:
        return []

    if gap_type is GapType.DISCONNECTED_GRAPH:
        # Les nœuds sans pose enregistrée sont ceux qui n'ont pas rejoint la
        # composante principale.
        affected = [
            n.asset_id for n in view_graph.nodes if n.pose_status != "registered"
        ]
        if affected:
            return sorted(affected)
        # Graphe connecté mais trop petit : tous les nœuds sont concernés.
        return sorted(n.asset_id for n in view_graph.nodes)

    if gap_type in (GapType.LOW_SUPPORT, GapType.LOW_PARALLAX):
        # Les nœuds sans paire valide sont sous-soutenus.
        paired: set[str] = set()
        for p in view_graph.pairs:
            if p.status == "valid":
                paired.add(p.image_a)
                paired.add(p.image_b)
        weak = [n.asset_id for n in view_graph.nodes if n.asset_id not in paired]
        return sorted(weak)

    if gap_type is GapType.POSE_UNCERTAINTY:
        return sorted(
            n.asset_id for n in view_graph.nodes if n.pose_status == "estimated"
        )

    # APPEARANCE_GAP / GEO_MISMATCH portent sur la cible, pas sur des vues
    # identifiables : ne rien affirmer plutôt que tout lister.
    return []


def analyze_gaps(
    gates: list[FidelityGate],
    view_graph: ViewGraphManifest | None = None,
    workspace: Workspace | None = None,
    *,
    front_azimuth_deg: float | None = None,
) -> list[ReconstructionGap]:
    """Produit les lacunes structurées à partir des portes de fidélité.

    Args:
        gates: portes de fidélité évaluées.
        view_graph: graphe de vue du run, source des assets en cause.
        workspace: workspace, conservé pour la publication.
        front_azimuth_deg: orientation de la façade avant. Sans elle, les
            angles de vue absolus ne sont pas calculables et restent à 0.0.
    """
    extent = _camera_extent_m(view_graph)

    gaps: list[ReconstructionGap] = []
    priority = 1
    for gate in gates:
        if gate.overall.value in ("pass", "not_applicable"):
            continue
        gap_type = _assign_gap_type(gate, view_graph)
        sector = _sector_for_target(gate.target_id)

        base = extent if extent is not None else _FALLBACK_BASELINE_M
        preferred_baseline = round(base * _BASELINE_FACTOR[gap_type], 2)

        required = {
            GapType.DISCONNECTED_GRAPH: f"vue supplémentaire connectée pour {gate.target_id}",
            GapType.LOW_PARALLAX: f"vue à forte parallaxe pour {gate.target_id}",
            GapType.LOW_SUPPORT: f"vue indépendante supplémentaire pour {gate.target_id}",
            GapType.POSE_UNCERTAINTY: f"calibration de pose pour {gate.target_id}",
            GapType.APPEARANCE_GAP: f"apparence mesurée pour {gate.target_id}",
            GapType.GEO_MISMATCH: f"ancrage géospatial pour {gate.target_id}",
        }[gap_type]

        gaps.append(
            ReconstructionGap(
                target_id=gate.target_id,
                gap_type=gap_type,
                affected_assets=_affected_assets(gate, gap_type, view_graph),
                affected_viewgraph_components=(
                    [view_graph.view_graph_id] if view_graph else []
                ),
                required_observation=required,
                preferred_sector=sector,
                preferred_baseline=preferred_baseline,
                preferred_view_angle=_view_angle_for_sector(sector, front_azimuth_deg),
                priority=priority,
            )
        )
        priority += 1
    return gaps


def publish_gap_analysis(gaps: list[ReconstructionGap], workspace: Workspace) -> Path:
    """Publie l'analyse des lacunes sous `07_reconstruction/gaps/`."""
    output_dir = workspace.path("07_reconstruction", "gaps")
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"gaps_{ts}.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gaps": [g.model_dump(mode="json") for g in gaps],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path


__all__ = [
    "analyze_gaps",
    "publish_gap_analysis",
]
