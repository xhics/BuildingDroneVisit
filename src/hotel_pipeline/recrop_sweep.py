"""Balayage des recadrages : proposer par géométrie, **vérifier** sur pixels.

Le sweep applique la leçon du pilote : la géométrie 2D propose, elle ne décide
pas. Sur la façade arrière, ses six meilleurs candidats montraient tous des
maisons — le modèle d'obstacles d'OSM porte 27 bâtiments pour tout un quartier,
et les pavillons de la rue arrière n'y figurent pas. La ligne de vue traversait
donc des maisons de plain-pied comme si elles n'existaient pas.

Le recadrage réellement exploitable a été trouvé ailleurs : un Panosphere du
stationnement voisin, à 90 m, cap 102° — noté 0,994 par la lecture pixel.

D'où la discipline de ce module :

1. **proposer** un cap par (panorama, façade), depuis la visibilité de position ;
2. **élaguer** avant tout appel réseau — un recadrage coûte une requête
   facturée, et 124 panoramas × 4 façades en coûteraient 496 ;
3. **acquérir** les candidats retenus ;
4. **vérifier** sur les pixels, et ne conserver que ce que l'image confirme.

Rien n'entre au corpus sur la foi de la géométrie seule.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .logging import get_logger

log = get_logger("recrop-sweep")

#: Au-delà, deux caps proposés depuis le même panorama montrent la même chose.
#: Les demander tous deux paierait deux fois la même image.
HEADING_DEDUP_DEG = 25.0

#: Score de prominence au-delà duquel le recadrage entre au corpus.
#: Aligné sur `subject_prominence.PROMINENT_THRESHOLD`, calibré à 100 % de
#: précision sur les décisions humaines du pilote : un faux positif ferait
#: entrer une concession voisine dans les références d'une vidéo commerciale.
ACCEPT_THRESHOLD = 0.60

#: Bande « partielle » : le sujet est là sans dominer. Ces vues ne sont pas
#: rejetées — leur union couvre ce qu'aucune ne montre seule.
PARTIAL_THRESHOLD = 0.15


@dataclass
class SweepCandidate:
    """Un recadrage proposé, avant toute dépense."""

    panorama_id: str
    facade_id: str
    heading_deg: float
    distance_m: float
    covers: int

    def key(self) -> tuple[str, str, int]:
        """Identité d'un recadrage, au regroupement de caps près.

        La façade fait **partie** de la clé. Sans elle, deux propositions du
        même panorama à des caps voisins fusionnaient bien qu'elles servent
        des murs différents — un cliché d'angle montre légitimement les deux.
        Mesuré sur le pilote : la meilleure vue arrière (0,994, cap 102°) a
        été perdue parce qu'une proposition à 99° pour la façade avant l'avait
        absorbée, et l'arrière n'a jamais été demandé depuis ce panorama.
        """
        return (
            self.panorama_id,
            self.facade_id,
            int(self.heading_deg // HEADING_DEDUP_DEG),
        )


@dataclass
class SweepResult:
    """Ce qu'un recadrage a réellement montré."""

    candidate: SweepCandidate
    path: Path | None = None
    score: float | None = None
    verdict: str = "unfetched"
    reason: str | None = None

    @property
    def accepted(self) -> bool:
        return self.score is not None and self.score >= ACCEPT_THRESHOLD

    @property
    def partial(self) -> bool:
        return (
            self.score is not None
            and PARTIAL_THRESHOLD <= self.score < ACCEPT_THRESHOLD
        )

    def as_dict(self) -> dict:
        return {
            "panorama_id": self.candidate.panorama_id,
            "facade_id": self.candidate.facade_id,
            "heading_deg": round(self.candidate.heading_deg, 1),
            "distance_m": round(self.candidate.distance_m, 1),
            "score": round(self.score, 4) if self.score is not None else None,
            "verdict": self.verdict,
            "accepted": self.accepted,
            "partial": self.partial,
            "reason": self.reason,
            "path": str(self.path) if self.path else None,
        }


def prune(candidates: list[SweepCandidate]) -> list[SweepCandidate]:
    """Écarte les recadrages redondants **avant** de payer pour eux.

    Deux caps proches sur le même panorama rendent la même image. On garde
    celui qui couvre le plus de mur, et à couverture égale le plus proche.
    """
    best: dict[tuple[str, int], SweepCandidate] = {}
    for candidate in candidates:
        key = candidate.key()
        current = best.get(key)
        if current is None or (candidate.covers, -candidate.distance_m) > (
            current.covers, -current.distance_m
        ):
            best[key] = candidate
    kept = sorted(
        best.values(), key=lambda c: (c.facade_id, c.distance_m)
    )
    log.info(
        "élagage : %d candidat(s) → %d recadrage(s) distinct(s)",
        len(candidates), len(kept),
    )
    return kept


def verify(
    results: list[SweepResult],
    reader=None,  # noqa: ANN001
) -> list[SweepResult]:
    """Lit chaque recadrage acquis et n'accepte que ce que l'image confirme.

    La géométrie a déjà eu son mot ; ici seul le contenu compte. Un candidat
    non acquis reste `unfetched` — l'absence de mesure n'est pas un rejet.
    """
    from .subject_prominence import ProminenceReader

    if reader is None:
        reader = ProminenceReader()

    fetched = [r for r in results if r.path is not None and r.path.is_file()]
    readings = reader.read_many([(str(r.path), r.path) for r in fetched])

    for result, reading in zip(fetched, readings):
        if not reading.measured:
            result.verdict = "unmeasured"
            result.reason = reading.reason
            continue
        result.score = reading.score
        result.verdict = reading.verdict

    accepted = sum(1 for r in results if r.accepted)
    partial = sum(1 for r in results if r.partial)
    log.info(
        "vérification : %d retenu(s), %d partiel(s), %d écarté(s)",
        accepted, partial, len(fetched) - accepted - partial,
    )
    return results


def summarise(results: list[SweepResult]) -> dict:
    """Bilan par façade : ce qui est confirmé, ce qui reste partiel."""
    per_facade: dict[str, dict] = {}
    for result in results:
        facade = result.candidate.facade_id
        bucket = per_facade.setdefault(
            facade, {"accepted": 0, "partial": 0, "rejected": 0, "best": None}
        )
        if result.accepted:
            bucket["accepted"] += 1
        elif result.partial:
            bucket["partial"] += 1
        elif result.score is not None:
            bucket["rejected"] += 1
        if result.score is not None and (
            bucket["best"] is None or result.score > bucket["best"]["score"]
        ):
            bucket["best"] = {
                "score": round(result.score, 4),
                "panorama_id": result.candidate.panorama_id,
                "heading_deg": round(result.candidate.heading_deg, 1),
                "distance_m": round(result.candidate.distance_m, 1),
            }
    return per_facade


__all__ = [
    "ACCEPT_THRESHOLD",
    "HEADING_DEDUP_DEG",
    "PARTIAL_THRESHOLD",
    "SweepCandidate",
    "SweepResult",
    "prune",
    "summarise",
    "verify",
]
