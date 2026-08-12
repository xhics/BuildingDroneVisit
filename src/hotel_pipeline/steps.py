"""Étapes du pipeline (plan directeur §18).

Au Lot 0, les étapes de traitement ne sont pas implémentées : elles lèvent
`StepNotImplemented`. Le squelette s'exécute néanmoins de bout en bout et
s'arrête proprement sur la première étape manquante, ce qui valide
l'orchestration avant d'y brancher la vision.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Ordre des étapes de la Phase 1 (plan directeur §18).
STEP_ORDER: tuple[str, ...] = (
    "collect",
    "preflight",
    "reconstruct",
    "align",
    "validate",
)


class StepNotImplemented(RuntimeError):
    """Étape prévue par le plan directeur, pas encore construite."""

    def __init__(self, step: str, lot: str) -> None:
        super().__init__(
            f"l'étape {step!r} n'est pas implémentée au Lot 0 — prévue au {lot}."
        )
        self.step = step
        self.lot = lot


class StepBlocked(RuntimeError):
    """Étape suspendue en attente d'une décision humaine."""

    def __init__(self, step: str, awaiting: str, expected_form: str) -> None:
        super().__init__(f"étape {step!r} bloquée : {awaiting}")
        self.step = step
        self.awaiting = awaiting
        self.expected_form = expected_form


@dataclass(frozen=True)
class Step:
    name: str
    lot: str
    summary: str


STEPS: dict[str, Step] = {
    "collect": Step(
        "collect",
        "Lot 1 puis Lot 4",
        "Résolution de propriété, collecte, droits et manifeste d'assets.",
    ),
    "preflight": Step(
        "preflight",
        "Lot 2 puis Lot 4",
        "Cascade G0 à G5, du comptage au SfM sparse réel.",
    ),
    "reconstruct": Step(
        "reconstruct",
        "Lot 2 puis Lot 5",
        "Route de reconstruction et production du modèle 3D.",
    ),
    "align": Step(
        "align",
        "Lot 6",
        "Géoréférencement, alignement et environnement composite.",
    ),
    "validate": Step(
        "validate",
        "Lot 7",
        "Carte de confiance, comparaison aux références, rapport.",
    ),
}


def run_step(name: str, workspace) -> None:  # noqa: ANN001 - Workspace, import circulaire
    """Exécute une étape. Toutes lèvent `StepNotImplemented` au Lot 0."""
    step = STEPS[name]
    raise StepNotImplemented(step.name, step.lot)
