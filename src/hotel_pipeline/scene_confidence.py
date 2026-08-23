"""Confiance jointe par surface (Lot 2 — composition des axes).

Chaque porte du pipeline mesurait une grandeur continue, la comparait à un
seuil, et ne transmettait que le verdict. Trois passages tout juste au-dessus
du seuil devenaient donc indiscernables de trois passages francs : le `ET`
booléen ne sait pas composer des marges.

Ce module compose les axes **avant** de trancher, sur trois questions
distinctes qu'il ne faut pas confondre :

- **couverture** — a-t-on assez observé la surface pour la reconstruire ?
- **apparence** — les images qui la montrent sont-elles exploitables ?
- **fidélité** — la reconstruction prédit-elle une vue qu'elle n'a pas vue ?

Le produit est une **moyenne géométrique** : une composante nulle annule le
tout. Une surface parfaitement couverte dont aucune image n'est exploitable ne
donne pas une belle vidéo, et une moyenne arithmétique le laisserait croire.

Ce que ce module ne fait pas : inventer une valeur pour un axe non mesuré. Un
axe absent est déclaré, la confiance jointe reste `None`, et le rapport nomme
ce qui manque. C'est l'écart entre « faible » et « inconnu », que tout le reste
du dispositif s'attache à préserver.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Au-dessus, la surface porte une vidéo sans réserve.
STRONG = 0.70

#: En dessous, la surface ne porte rien : il faut acquérir.
WEAK = 0.40


@dataclass
class SceneConfidence:
    """Confiance jointe d'une surface, et ce qui la borne."""

    surface_id: str
    coverage: float | None = None
    appearance: float | None = None
    fidelity: float | None = None
    joint: float | None = None
    limiting_factor: str | None = None
    verdict: str = "unmeasured"
    missing_axes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        def rounded(value: float | None) -> float | None:
            return round(value, 3) if value is not None else None

        return {
            "surface_id": self.surface_id,
            "coverage": rounded(self.coverage),
            "appearance": rounded(self.appearance),
            "fidelity": rounded(self.fidelity),
            "joint": rounded(self.joint),
            "limiting_factor": self.limiting_factor,
            "verdict": self.verdict,
            "missing_axes": list(self.missing_axes),
        }


def compose(
    surface_id: str,
    *,
    coverage: float | None = None,
    appearance: float | None = None,
    fidelity: float | None = None,
) -> SceneConfidence:
    """Compose les axes mesurés en une confiance jointe.

    Chaque axe est une part dans [0, 1]. Un axe `None` n'est **pas** traité
    comme zéro : il est déclaré manquant, et la confiance jointe reste
    indéterminée. Le confondre avec zéro affirmerait un échec là où l'on n'a
    rien mesuré.
    """
    axes = {
        "couverture": coverage,
        "apparence": appearance,
        "fidélité": fidelity,
    }
    missing = sorted(name for name, value in axes.items() if value is None)
    measured = {
        name: float(min(1.0, max(0.0, value)))
        for name, value in axes.items()
        if value is not None
    }

    if missing:
        return SceneConfidence(
            surface_id=surface_id,
            coverage=coverage,
            appearance=appearance,
            fidelity=fidelity,
            joint=None,
            limiting_factor=(
                min(measured, key=lambda k: measured[k]) if measured else None
            ),
            verdict="unmeasured",
            missing_axes=missing,
        )

    product = 1.0
    for value in measured.values():
        product *= value
    joint = float(product ** (1 / len(measured)))

    if joint >= STRONG:
        verdict = "carries_video"
    elif joint >= WEAK:
        verdict = "marginal"
    else:
        verdict = "insufficient"

    return SceneConfidence(
        surface_id=surface_id,
        coverage=coverage,
        appearance=appearance,
        fidelity=fidelity,
        joint=joint,
        limiting_factor=min(measured, key=lambda k: measured[k]),
        verdict=verdict,
        missing_axes=[],
    )


def deliverable_confidence(
    surfaces: list[SceneConfidence],
    required: set[str] | None = None,
) -> tuple[float | None, str]:
    """Confiance du livrable, à partir des surfaces qu'il doit montrer.

    Le livrable ne vaut pas la moyenne de ses surfaces : il vaut sa surface la
    plus faible parmi celles qu'il doit montrer. Une vidéo dont la façade
    principale est bonne et l'entrée absente n'est pas « à moitié bonne » —
    elle est bloquée par l'entrée.

    `required` restreint aux surfaces exigées ; sans elle, toutes comptent.
    """
    considered = [
        s for s in surfaces
        if required is None or s.surface_id in required
    ]
    if not considered:
        return None, "aucune surface exigée"

    unmeasured = [s.surface_id for s in considered if s.joint is None]
    if unmeasured:
        return None, (
            "axes non mesurés sur : " + ", ".join(sorted(unmeasured))
        )

    weakest = min(considered, key=lambda s: s.joint or 0.0)
    return weakest.joint, (
        f"borné par {weakest.surface_id} "
        f"({weakest.limiting_factor or 'inconnu'})"
    )


__all__ = [
    "STRONG",
    "WEAK",
    "SceneConfidence",
    "compose",
    "deliverable_confidence",
]
