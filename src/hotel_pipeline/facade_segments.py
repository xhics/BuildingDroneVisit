"""Couverture cumulée par **segment** de façade (Lot 2).

`subject_prominence` note chaque image isolément, puis on classe. Cela répond à
« quelle est la meilleure vue ? » — mais ce n'est pas la question qui décide
d'une vidéo. Une vue partielle n'est pas une vue pauvre : une façade masquée
par un arbre reste pleine d'information, et un autre cliché pris vingt mètres
plus loin montre précisément ce que celui-ci cache.

Ce qui compte est donc l'**union** : la réunion des vues couvre-t-elle toute la
façade, et avec quelle profondeur de preuve ? Mesuré sur le pilote, les six
vues où le bâtiment remplit le cadre occupent un arc de 43,6°. Trois vues
« incidentes » — trop occultées pour être retenues seules — portent cet arc à
76,4°, parce qu'elles voient les extrémités que les vues rapprochées coupent.

D'où ce module. Il découpe la façade en segments, compte pour chacun combien de
vues **indépendantes** le montrent, et rend :

- les segments couverts par plusieurs vues — reconstructibles et texturables ;
- ceux couverts par une seule — géométrie fragile, apparence non corroborée ;
- ceux que rien ne montre — et c'est **cela** qu'on demande à l'hôtel.

`facade_coverage.visible_points` fait déjà le travail de visibilité, occlusion
et cadrage comprises. Ici on ne fait que l'accumuler par segment, sans refaire
sa géométrie.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Nombre de segments par façade. Dix découpe un mur de 60 m en tronçons de
#: 6 m — l'ordre de grandeur d'une travée, assez fin pour qu'un trou soit
#: nommable, assez large pour qu'un segment ne dépende pas d'un pixel.
DEFAULT_SEGMENTS = 10

#: Vues indépendantes au-delà desquelles un segment cesse d'être fragile.
#: Deux vues suffisent à trianguler ; une troisième corrobore.
WELL_COVERED = 3


@dataclass
class SegmentCoverage:
    """Ce qu'on a vu d'un tronçon de façade."""

    index: int
    #: Identifiants des vues qui montrent ce segment, sans doublon.
    views: list[str] = field(default_factory=list)
    #: Distance de la vue la plus proche qui le montre.
    nearest_m: float | None = None

    @property
    def depth(self) -> int:
        return len(self.views)

    @property
    def state(self) -> str:
        if self.depth == 0:
            return "unseen"
        if self.depth < WELL_COVERED:
            return "thin"
        return "covered"

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "depth": self.depth,
            "state": self.state,
            "nearest_m": round(self.nearest_m, 1) if self.nearest_m is not None else None,
            "views": list(self.views),
        }


@dataclass
class FacadeSegmentReport:
    """Couverture d'une façade, segment par segment."""

    facade_id: str
    segments: list[SegmentCoverage]

    @property
    def union_fraction(self) -> float:
        """Part de la façade vue par **au moins une** vue.

        C'est la mesure qui rend justice aux vues partielles : chacune peut
        n'en montrer qu'un tiers, leur réunion peut montrer le tout.
        """
        if not self.segments:
            return 0.0
        seen = sum(1 for s in self.segments if s.depth > 0)
        return seen / len(self.segments)

    @property
    def corroborated_fraction(self) -> float:
        """Part couverte par assez de vues pour être reconstructible."""
        if not self.segments:
            return 0.0
        strong = sum(1 for s in self.segments if s.state == "covered")
        return strong / len(self.segments)

    @property
    def best_single_fraction(self) -> float:
        """Part que la **meilleure vue seule** montre.

        Conservée à côté de l'union : reconstituer une façade par recouvrement
        suppose un recalage, alors qu'une vue unique le montre sans hypothèse.
        L'écart entre les deux dit ce que la combinaison apporte — et ce
        qu'elle suppose.
        """
        if not self.segments:
            return 0.0
        tally: dict[str, int] = {}
        for segment in self.segments:
            for view in segment.views:
                tally[view] = tally.get(view, 0) + 1
        return (max(tally.values()) / len(self.segments)) if tally else 0.0

    def unseen(self) -> list[int]:
        return [s.index for s in self.segments if s.state == "unseen"]

    def thin(self) -> list[int]:
        return [s.index for s in self.segments if s.state == "thin"]

    def verdict(self) -> str:
        if self.union_fraction == 0.0:
            return "unseen"
        if self.unseen():
            return "partial"
        if self.corroborated_fraction >= 0.8:
            return "corroborated"
        return "thin"

    def as_dict(self) -> dict:
        return {
            "facade_id": self.facade_id,
            "verdict": self.verdict(),
            "union_fraction": round(self.union_fraction, 3),
            "corroborated_fraction": round(self.corroborated_fraction, 3),
            "best_single_fraction": round(self.best_single_fraction, 3),
            "unseen_segments": self.unseen(),
            "thin_segments": self.thin(),
            "segments": [s.as_dict() for s in self.segments],
        }


def accumulate(
    facade_id: str,
    samples: list,
    subjects,  # noqa: ANN001 — (view_id, origin, heading_deg, fov_deg, distance_m)
    footprint,  # noqa: ANN001
    obstacles: list,
    *,
    segments: int = DEFAULT_SEGMENTS,
    max_distance_m: float = 150.0,
) -> FacadeSegmentReport:
    """Accumule la couverture par segment, sur les vues fournies.

    Chaque vue passe par `facade_coverage.visible_points`, qui tient déjà
    compte du cadrage, de l'occlusion par le bâti voisin et de la distance.
    On ne réinterprète pas son verdict : on note quels **points** de mur elle
    montre, puis on agrège par segment.
    """
    from .geo.facade_coverage import visible_points

    if not samples or segments <= 0:
        return FacadeSegmentReport(facade_id=facade_id, segments=[])

    buckets = [SegmentCoverage(index=i) for i in range(segments)]
    per_sample = max(1, len(samples) // segments)

    for subject in subjects:
        view_id = subject[0]
        origin = subject[1]
        heading = subject[2] if len(subject) > 2 else None
        fov = subject[3] if len(subject) > 3 else None
        distance = subject[4] if len(subject) > 4 else None

        seen, _ = visible_points(
            samples, origin, footprint, obstacles, heading, fov, max_distance_m
        )
        if not seen:
            continue

        touched: set[int] = set()
        for sample_index in seen:
            bucket = min(segments - 1, sample_index // per_sample)
            touched.add(bucket)

        for bucket in touched:
            target = buckets[bucket]
            if view_id not in target.views:
                target.views.append(view_id)
            if distance is not None:
                if target.nearest_m is None or distance < target.nearest_m:
                    target.nearest_m = float(distance)

    return FacadeSegmentReport(facade_id=facade_id, segments=buckets)


def capture_request(report: FacadeSegmentReport) -> str | None:
    """Ce qu'il reste à photographier, dit en clair.

    Rendre `None` quand rien ne manque : une demande de capture vide vaudrait
    mieux ne pas être émise que d'être émise sans objet.
    """
    unseen = report.unseen()
    thin = report.thin()
    if not unseen and not thin:
        return None

    parts = []
    if unseen:
        parts.append(
            f"{len(unseen)} tronçon(s) jamais vu(s) ({', '.join(map(str, unseen))})"
        )
    if thin:
        parts.append(
            f"{len(thin)} tronçon(s) vus par moins de {WELL_COVERED} vues "
            f"({', '.join(map(str, thin))})"
        )
    return f"{report.facade_id} : " + " ; ".join(parts)


__all__ = [
    "DEFAULT_SEGMENTS",
    "WELL_COVERED",
    "FacadeSegmentReport",
    "SegmentCoverage",
    "accumulate",
    "capture_request",
]
