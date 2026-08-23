"""Confiance de couverture par surface (Lot 2 — complément à la porte C).

La porte C détecte l'**hallucination** : une géométrie inventée pour satisfaire
les vues d'entraînement. C'est le bon risque à surveiller quand un modèle
génératif complète une scène.

Ce n'est pas le risque dominant d'une démo bâtie sur des photographies réelles.
Là, la question est plus simple et plus décisive : **a-t-on assez observé cette
surface pour la reconstruire ?** Une façade vue par quinze caméras réparties
sur 60° se reconstruit ; la même vue deux fois depuis le même trottoir, non —
et aucun solveur n'y changera rien.

Cette confiance-là se calcule sur des preuves déjà disponibles, sans
reconstruction ni rendu : positions de caméra, caps mesurés, empreinte du
bâtiment. Elle répond avant la reconstruction, et dit quoi aller chercher.

Trois composantes, toutes mesurées :

- **support** — nombre de vues indépendantes qui cadrent la surface ;
- **parallaxe** — étendue angulaire réellement occupée autour d'elle ;
- **proximité** — distance de la plus proche vue exploitable.

Aucune n'est suffisante seule : quinze vues confondues n'ont pas de parallaxe,
et deux vues très écartées n'ont pas de support.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Vues indépendantes au-delà desquelles le support cesse d'être limitant.
SUPPORT_SATURATION = 8

#: Étendue angulaire (degrés) au-delà de laquelle la parallaxe est suffisante.
#: Une façade se reconstruit bien à partir d'un arc de 60° ; au-delà, le gain
#: décroît et le recouvrement se dégrade.
PARALLAX_TARGET_DEG = 60.0

#: Distance (m) en deçà de laquelle la proximité cesse d'être limitante.
PROXIMITY_TARGET_M = 80.0

#: Distance (m) au-delà de laquelle une vue n'apporte plus de détail de façade.
PROXIMITY_LIMIT_M = 250.0


@dataclass
class SurfaceEvidence:
    """Ce qu'on a réellement observé d'une surface."""

    surface_id: str
    #: Azimuts d'observation, en degrés, depuis la surface vers chaque caméra.
    bearings_deg: list[float] = field(default_factory=list)
    #: Distances caméra → surface, en mètres, dans le même ordre.
    distances_m: list[float] = field(default_factory=list)

    def usable(self) -> list[tuple[float, float]]:
        """Couples (azimut, distance) exploitables pour de la façade."""
        return [
            (b, d)
            for b, d in zip(self.bearings_deg, self.distances_m)
            if d <= PROXIMITY_LIMIT_M
        ]


@dataclass
class SurfaceConfidence:
    """Confiance de couverture d'une surface, et ce qui la borne."""

    surface_id: str
    support: float
    parallax: float
    proximity: float
    confidence: float
    n_usable: int
    arc_deg: float
    nearest_m: float | None
    limiting_factor: str
    verdict: str

    def as_dict(self) -> dict:
        return {
            "surface_id": self.surface_id,
            "support": round(self.support, 3),
            "parallax": round(self.parallax, 3),
            "proximity": round(self.proximity, 3),
            "confidence": round(self.confidence, 3),
            "n_usable": self.n_usable,
            "arc_deg": round(self.arc_deg, 1),
            "nearest_m": round(self.nearest_m, 1) if self.nearest_m is not None else None,
            "limiting_factor": self.limiting_factor,
            "verdict": self.verdict,
        }


def occupied_arc(bearings_deg: list[float]) -> float:
    """Étendue angulaire réellement occupée, en degrés.

    Le plus grand **trou** est cherché, et l'arc occupé est son complément :
    c'est ce qui gère le passage 359° → 0° sans cas particulier. Prendre
    `max - min` compterait 344° pour des vues toutes groupées autour du nord.
    """
    if len(bearings_deg) < 2:
        return 0.0
    # Les azimuts **distincts** : quinze vues au même cap ne décrivent qu'une
    # direction. En les gardant tous, tous les écarts valaient 0, le plus
    # grand trou aussi, et l'arc rendu était 360° — exactement l'inverse de
    # la vérité, pour le cas le plus dégénéré qui soit.
    ordered = sorted({round(b % 360.0, 6) for b in bearings_deg})
    if len(ordered) < 2:
        return 0.0
    gaps = [
        (ordered[(i + 1) % len(ordered)] - ordered[i]) % 360.0
        for i in range(len(ordered))
    ]
    widest = max(gaps)
    return max(0.0, 360.0 - widest)


def assess(evidence: SurfaceEvidence) -> SurfaceConfidence:
    """Évalue la confiance de couverture d'une surface."""
    usable = evidence.usable()
    n = len(usable)

    if n == 0:
        return SurfaceConfidence(
            surface_id=evidence.surface_id,
            support=0.0, parallax=0.0, proximity=0.0, confidence=0.0,
            n_usable=0, arc_deg=0.0, nearest_m=None,
            limiting_factor="aucune vue exploitable",
            verdict="unreachable",
        )

    bearings = [b for b, _ in usable]
    distances = [d for _, d in usable]
    nearest = min(distances)
    arc = occupied_arc(bearings)

    support = min(1.0, n / SUPPORT_SATURATION)
    parallax = min(1.0, arc / PARALLAX_TARGET_DEG)
    if nearest <= PROXIMITY_TARGET_M:
        proximity = 1.0
    else:
        span = PROXIMITY_LIMIT_M - PROXIMITY_TARGET_M
        proximity = max(0.0, 1.0 - (nearest - PROXIMITY_TARGET_M) / span)

    # Moyenne **géométrique** : une composante nulle annule la confiance.
    # Une moyenne arithmétique laisserait quinze vues confondues compenser une
    # parallaxe nulle, alors qu'aucune reconstruction n'en sortirait.
    confidence = float((support * parallax * proximity) ** (1 / 3))

    components = {
        "support": support,
        "parallaxe": parallax,
        "proximité": proximity,
    }
    limiting = min(components, key=lambda k: components[k])

    if confidence >= 0.7:
        verdict = "reconstructible"
    elif confidence >= 0.4:
        verdict = "marginal"
    elif confidence > 0.0:
        verdict = "insufficient"
    else:
        verdict = "unreachable"

    return SurfaceConfidence(
        surface_id=evidence.surface_id,
        support=support,
        parallax=parallax,
        proximity=proximity,
        confidence=confidence,
        n_usable=n,
        arc_deg=arc,
        nearest_m=nearest,
        limiting_factor=limiting,
        verdict=verdict,
    )


def assess_all(evidences: list[SurfaceEvidence]) -> list[SurfaceConfidence]:
    return [assess(e) for e in evidences]


__all__ = [
    "PARALLAX_TARGET_DEG",
    "PROXIMITY_LIMIT_M",
    "PROXIMITY_TARGET_M",
    "SUPPORT_SATURATION",
    "SurfaceConfidence",
    "SurfaceEvidence",
    "assess",
    "assess_all",
    "occupied_arc",
]
