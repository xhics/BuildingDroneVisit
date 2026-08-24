"""Élagage d'un corpus par redondance visuelle.

Une reconstruction ne gagne rien à recevoir quinze fois la même façade sous le
même angle. Elle y perd même deux fois : le temps de mise en correspondance
croît avec le carré du nombre d'images, et un groupe de vues quasi identiques
pèse dans le graphe autant qu'un point de vue unique — au risque d'ancrer le
modèle sur ce qui est sur-représenté plutôt que sur ce qui est bien vu.

Le tri se fait dans l'espace du modèle déjà chargé pour l'identité : deux
images dont les vecteurs sont presque colinéaires montrent la même chose. Rien
n'est encodé pour l'occasion, les vecteurs viennent de l'index existant.

**Ce que l'élagage ne fait pas.** Il ne juge pas la qualité d'une image — cela
relève d'`appearance_quality`, dont il consomme la note quand elle existe. Il
ne décide pas non plus qu'une image est inutile : il désigne, dans un groupe de
vues équivalentes, celle qui représente le groupe. Les autres sont écartées
avec la raison et le représentant, jamais supprimées en silence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..logging import get_logger

log = get_logger("identity-prune")

#: Similarité cosinus au-delà de laquelle deux vues sont tenues pour
#: redondantes. Mesuré sur le corpus pilote : deux recadrages d'un même
#: panorama dépassent 0,97, deux vues distinctes d'une même façade restent
#: sous 0,93. Le seuil est délibérément haut — écarter une vue utile coûte
#: bien plus qu'en garder une de trop.
REDUNDANCY_THRESHOLD = 0.95

#: Nombre minimal de vues conservées, quoi qu'en dise la redondance. Un corpus
#: réduit à deux images ne reconstruit rien, même si ses vues se ressemblent.
MIN_KEPT = 8


@dataclass
class PrunedView:
    """Une vue et son sort, avec la raison."""

    asset_id: str
    kept: bool
    #: Vue retenue qui représente ce groupe. `None` pour un représentant.
    represented_by: str | None = None
    similarity: float | None = None
    quality: float | None = None
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "kept": self.kept,
            "represented_by": self.represented_by,
            "similarity": round(self.similarity, 4) if self.similarity is not None else None,
            "quality": round(self.quality, 4) if self.quality is not None else None,
            "reason": self.reason,
        }


@dataclass
class PruneReport:
    """Ce que l'élagage a retenu, et ce qu'il a regroupé."""

    views: list[PrunedView] = field(default_factory=list)
    threshold: float = REDUNDANCY_THRESHOLD
    provenance: dict = field(default_factory=dict)

    @property
    def kept(self) -> list[str]:
        return [v.asset_id for v in self.views if v.kept]

    @property
    def dropped(self) -> list[str]:
        return [v.asset_id for v in self.views if not v.kept]

    def as_dict(self) -> dict:
        return {
            "kept_count": len(self.kept),
            "dropped_count": len(self.dropped),
            "total": len(self.views),
            "threshold": self.threshold,
            "kept": self.kept,
            "views": [v.as_dict() for v in self.views],
            "provenance": self.provenance,
            "caveats": [
                "une vue écartée est redondante, non mauvaise : elle est "
                "représentée par une autre du même groupe",
                "la redondance est mesurée dans l'espace du modèle, elle ne "
                "prouve pas que deux vues montrent le même bâtiment",
            ],
        }


def _group_of(
    vectors: np.ndarray, order: list[int], threshold: float
) -> tuple[list[int], dict[int, tuple[int, float]]]:
    """Regroupe les vues par voisinage, en parcourant du meilleur au moins bon.

    Le premier arrivé d'un groupe le représente : comme l'ordre suit la note de
    qualité, le représentant est la meilleure vue de son groupe et non la
    première rencontrée dans le corpus.
    """
    representatives: list[int] = []
    absorbed: dict[int, tuple[int, float]] = {}

    for index in order:
        if not representatives:
            representatives.append(index)
            continue

        # Similarité à tous les représentants déjà retenus, d'un coup.
        scores = vectors[representatives] @ vectors[index]
        best = int(np.argmax(scores))
        if float(scores[best]) >= threshold:
            absorbed[index] = (representatives[best], float(scores[best]))
        else:
            representatives.append(index)

    return representatives, absorbed


def prune(
    views: list[tuple[str, Path]],
    index,  # noqa: ANN001 - EmbeddingIndex, importé paresseusement par l'appelant
    quality: dict[str, float] | None = None,
    threshold: float = REDUNDANCY_THRESHOLD,
    min_kept: int = MIN_KEPT,
) -> PruneReport:
    """Retient une vue par groupe de vues équivalentes.

    `quality` porte la note d'apparence quand elle a été mesurée ; elle décide
    seulement **qui représente** un groupe, jamais si le groupe est gardé. Une
    vue sans note est traitée comme moyenne : l'absence de mesure ne la
    disqualifie pas.
    """
    report = PruneReport(threshold=threshold)
    if not views:
        return report

    quality = quality or {}
    vectors = []
    usable: list[tuple[str, Path]] = []
    for asset_id, path in views:
        try:
            vectors.append(index.vector_of(Path(path)))
            usable.append((asset_id, path))
        except (OSError, ValueError) as exc:
            # Une image illisible n'est pas élaguée : elle est signalée telle
            # quelle, et le reste du pipeline décidera de son sort.
            report.views.append(
                PrunedView(
                    asset_id=asset_id,
                    kept=True,
                    reason=f"non encodable ({exc}) — conservée sans jugement",
                )
            )

    if not usable:
        return report

    matrix = np.stack(vectors)
    # Les vecteurs du modèle sont déjà unitaires ; la renormalisation protège
    # d'un index écrit par une version antérieure qui ne l'aurait pas fait.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.where(norms < 1e-12, 1.0, norms)

    notes = np.array([quality.get(a, 0.5) for a, _p in usable])
    # Du mieux noté au moins bien : le représentant d'un groupe est sa
    # meilleure vue.
    order = list(np.argsort(-notes))

    if len(usable) <= min_kept:
        for slot, (asset_id, _path) in enumerate(usable):
            report.views.append(
                PrunedView(
                    asset_id=asset_id,
                    kept=True,
                    quality=float(notes[slot]),
                    reason=(
                        f"corpus de {len(usable)} vue(s), au plus bas de ce que "
                        f"la reconstruction demande ({min_kept}) — rien n'est écarté"
                    ),
                )
            )
        log.info("élagage sans effet : %d vue(s) seulement", len(usable))
        return report

    representatives, absorbed = _group_of(matrix, order, threshold)

    # Un élagage trop mordant est rattrapé : les vues absorbées les moins
    # ressemblantes à leur représentant reviennent, car ce sont elles qui
    # apportent le plus au graphe.
    if len(representatives) < min_kept and absorbed:
        recovered = sorted(absorbed.items(), key=lambda kv: kv[1][1])
        for slot, _pair in recovered[: min_kept - len(representatives)]:
            representatives.append(slot)
            absorbed.pop(slot, None)

    kept = set(representatives)
    for slot, (asset_id, _path) in enumerate(usable):
        if slot in kept:
            report.views.append(
                PrunedView(
                    asset_id=asset_id,
                    kept=True,
                    quality=float(notes[slot]),
                    reason="représente son groupe de vues équivalentes",
                )
            )
            continue
        anchor, score = absorbed[slot]
        report.views.append(
            PrunedView(
                asset_id=asset_id,
                kept=False,
                represented_by=usable[anchor][0],
                similarity=score,
                quality=float(notes[slot]),
                reason=(
                    f"redondante à {score:.3f} avec {usable[anchor][0]} "
                    f"(seuil {threshold})"
                ),
            )
        )

    report.provenance = {
        "threshold": threshold,
        "min_kept": min_kept,
        "quality_supplied": len(quality),
        "model": getattr(getattr(index, "embedder", None), "model_name", "unknown"),
    }
    log.info(
        "élagage : %d vue(s) retenue(s) sur %d, %d groupe(s) redondant(s)",
        len(report.kept),
        len(report.views),
        len(report.dropped),
    )
    return report


__all__ = [
    "MIN_KEPT",
    "REDUNDANCY_THRESHOLD",
    "PruneReport",
    "PrunedView",
    "prune",
]
