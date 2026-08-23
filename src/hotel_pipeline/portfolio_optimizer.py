"""Optimiseur de portefeuille d'acquisition (Lot 2 — P4.4).

Deux portes, dans l'ordre :
1. **Pre-SfM Collection Gate** — ne bloque PAS le premier ViewGraph. Seul
   `STRUCTURALLY_IMPOSSIBLE` bloque. Pour le corpus existant, on tente quand
   même : le ViewGraph est la mesure qu'on cherche. `WEAK_BUT_ATTEMPT` construit
   quand même le graphe.
2. **Post-ViewGraph Gate** — verdict réel de reconstructibilité (connectivité
   du graphe de paires).

Le résultat de la porte 1 est `READY_TO_ATTEMPT | WEAK_BUT_ATTEMPT |
STRUCTURALLY_IMPOSSIBLE`.
"""

from __future__ import annotations

from enum import StrEnum

from .schemas.reconstruction import ViewGraphManifest
from .workspace import Workspace


class PreSfMVerdict(StrEnum):
    READY_TO_ATTEMPT = "ready_to_attempt"
    WEAK_BUT_ATTEMPT = "weak_but_attempt"
    STRUCTURALLY_IMPOSSIBLE = "structurally_impossible"


class PostViewGraphVerdict(StrEnum):
    RECONSTRUCTION_VIABLE = "reconstruction_viable"
    LOW_CONNECTIVITY = "low_connectivity"
    NOT_VIABLE = "not_viable"


def pre_sfm_collection_gate(
    candidate_observations: int,
    angular_diversity: float,
    resolution_ok: bool = True,
    currentness_ok: bool = True,
) -> PreSfMVerdict:
    """Porte 1 — Pre-SfM Collection.

    Seul `STRUCTURALLY_IMPOSSIBLE` bloque (aucune observation candidate,
    ou diversité angulaire nulle avec peu d'images). Sinon on tente.
    """
    if candidate_observations == 0:
        return PreSfMVerdict.STRUCTURALLY_IMPOSSIBLE
    if candidate_observations >= 3 and angular_diversity > 0.0 and resolution_ok and currentness_ok:
        return PreSfMVerdict.READY_TO_ATTEMPT
    return PreSfMVerdict.WEAK_BUT_ATTEMPT


def post_view_graph_gate(view_graph: ViewGraphManifest) -> PostViewGraphVerdict:
    """Porte 2 — Post-ViewGraph (verdict réel de reconstructibilité)."""
    report = view_graph.report
    valid = report.valid_pairs
    largest = report.largest_component
    if valid < 1 or largest < 2:
        return PostViewGraphVerdict.NOT_VIABLE
    if largest < 3 or report.registered_candidate_ratio < 0.3:
        return PostViewGraphVerdict.LOW_CONNECTIVITY
    return PostViewGraphVerdict.RECONSTRUCTION_VIABLE


class AcquisitionPortfolioOptimizer:
    """Guide la collecte depuis les lacunes démontrées du ViewGraph/reconstruction.

    Le workspace est conservé pour les publications futures ; les deux portes
    elles-mêmes sont des fonctions pures, utilisables sans instance.
    """

    def __init__(self, workspace: Workspace | None = None):
        self.workspace = workspace

    def evaluate_pre_sfm(
        self,
        candidate_observations: int,
        angular_diversity: float,
        resolution_ok: bool = True,
        currentness_ok: bool = True,
    ) -> PreSfMVerdict:
        return pre_sfm_collection_gate(
            candidate_observations, angular_diversity, resolution_ok, currentness_ok
        )

    def evaluate_post_view_graph(self, view_graph: ViewGraphManifest) -> PostViewGraphVerdict:
        return post_view_graph_gate(view_graph)


__all__ = [
    "PreSfMVerdict",
    "PostViewGraphVerdict",
    "AcquisitionPortfolioOptimizer",
    "pre_sfm_collection_gate",
    "post_view_graph_gate",
]
