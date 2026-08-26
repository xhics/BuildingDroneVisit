"""Dimensionne la production selon la taille du domaine, et fixe l'heure.

Un pavillon et un complexe hôtelier ne demandent ni le même nombre de plans,
ni la même durée : couvrir un domaine plus vaste exige davantage de points
d'ancrage, sous peine de survoler en montrant peu, et davantage de temps,
sous peine de tout enchaîner trop vite. Ces règles sont ici explicites et
vérifiables plutôt que dispersées en constantes dans le code de rendu.

L'heure de tournage est un choix cinématographique à part entière : elle
pilote à la fois l'éclairage de la scène 3D et le vocabulaire des prompts de
génération. Elle doit être décidée, pas subie.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Diagonale de référence d'un établissement isolé (voir `video.BASELINE_DIAGONAL_M`).
BASELINE_DIAGONAL_M = 350.0

#: Bornes du nombre d'images de référence. En dessous de 6 le parcours n'a
#: plus assez d'ancrages pour rester lisible ; au-delà de 30 on dépasse ce
#: qu'un seul appel de génération accepte en références.
MIN_REFERENCES = 6
MAX_REFERENCES = 30

#: Bornes de durée. Sous 20 s une visite n'a pas le temps de raconter ;
#: au-delà de 2 min elle lasse, quel que soit le domaine.
MIN_DURATION_S = 20.0
MAX_DURATION_S = 120.0


@dataclass(frozen=True)
class TimeOfDay:
    key: str
    label_fr: str
    #: Élévation solaire approximative, pour l'éclairage de la scène 3D.
    sun_elevation_deg: float
    #: Azimut solaire, qui décide du côté éclairé et de la direction des ombres.
    sun_azimuth_deg: float
    #: Vocabulaire injecté dans les prompts de génération.
    look_fr: str


#: Presets choisis pour ce qu'ils apportent à l'image, pas pour couvrir
#: mécaniquement la journée : l'heure bleue et l'heure dorée sont les deux
#: moments où l'architecture se détache le mieux.
TIMES_OF_DAY: dict[str, TimeOfDay] = {
    "aube": TimeOfDay(
        "aube", "Aube", 4.0, 80.0,
        "lumière rasante et froide de l'aube, brume basse, ombres très longues, "
        "ciel pâle qui rosit à l'horizon",
    ),
    "matin": TimeOfDay(
        "matin", "Matin", 30.0, 110.0,
        "lumière claire du matin, ombres nettes et allongées, ciel franc, "
        "atmosphère calme et propre",
    ),
    "midi": TimeOfDay(
        "midi", "Midi", 65.0, 180.0,
        "lumière zénithale dure, ombres courtes et contrastées, couleurs saturées, "
        "ciel profond",
    ),
    "doree": TimeOfDay(
        "doree", "Heure dorée", 8.0, 265.0,
        "lumière dorée et chaude de fin de journée, contre-jours, ombres très "
        "longues, reflets ambrés sur les façades vitrées",
    ),
    "bleue": TimeOfDay(
        "bleue", "Heure bleue", -4.0, 285.0,
        "crépuscule bleu profond, éclairages intérieurs et extérieurs allumés qui "
        "ressortent, ciel dégradé, ambiance feutrée",
    ),
    "nuit": TimeOfDay(
        "nuit", "Nuit", -25.0, 300.0,
        "nuit, façades éclairées, lumières chaudes aux fenêtres, éclairage "
        "paysager, ciel noir, ambiance intime",
    ),
}

DEFAULT_TIME_OF_DAY = "doree"


@dataclass
class ProductionPlan:
    diagonal_m: float
    reference_count: int
    duration_s: float
    exterior_share: float
    time_of_day: TimeOfDay

    @property
    def exterior_duration_s(self) -> float:
        return round(self.duration_s * self.exterior_share, 1)

    @property
    def interior_duration_s(self) -> float:
        return round(self.duration_s * (1.0 - self.exterior_share), 1)

    def describe_fr(self) -> str:
        return (
            f"Domaine ~{self.diagonal_m:.0f} m -> {self.reference_count} références, "
            f"{self.duration_s:.0f} s ({self.exterior_duration_s:.0f} s extérieur / "
            f"{self.interior_duration_s:.0f} s intérieur), {self.time_of_day.label_fr.lower()}"
        )


def plan_production(
    diagonal_m: float | None,
    *,
    time_of_day: str = DEFAULT_TIME_OF_DAY,
    available_interior_photos: int = 0,
) -> ProductionPlan:
    """Dimensionne la production à partir de l'étendue mesurée du domaine.

    ``diagonal_m`` vient de ``places.fetch_viewport_extent_m``. ``None``
    signifie « non mesuré » : on retombe sur le gabarit d'un établissement
    isolé plutôt que de deviner grand.

    ``available_interior_photos`` borne la part intérieure : promettre dix
    étapes d'intérieur quand trois photos existent produirait des plans sans
    référence, donc inventés — exactement ce qu'on cherche à éviter.
    """
    diagonal = BASELINE_DIAGONAL_M if diagonal_m is None else max(50.0, diagonal_m)
    ratio = diagonal / BASELINE_DIAGONAL_M

    # Croissance en racine : la surface croît comme le carré de la diagonale,
    # mais le nombre de points de vue *intéressants* croît bien plus lentement
    # — un domaine 4 fois plus large ne demande pas 4 fois plus de plans.
    references = round(MIN_REFERENCES + 6.0 * (ratio**0.5 - 1.0) * 2.0 + 4.0)
    references = max(MIN_REFERENCES, min(MAX_REFERENCES, references))

    duration = MIN_DURATION_S + 22.0 * (ratio**0.5 - 1.0) * 2.0 + 8.0
    duration = max(MIN_DURATION_S, min(MAX_DURATION_S, duration))

    # Sans photos d'intérieur, la visite reste un survol : lui réserver du
    # temps intérieur créerait des plans sans matière.
    if available_interior_photos <= 0:
        exterior_share = 1.0
    else:
        interior_capacity = min(1.0, available_interior_photos / max(1, references))
        exterior_share = max(0.35, 1.0 - 0.65 * interior_capacity)

    return ProductionPlan(
        diagonal_m=diagonal,
        reference_count=references,
        duration_s=round(duration, 1),
        exterior_share=exterior_share,
        time_of_day=TIMES_OF_DAY.get(time_of_day, TIMES_OF_DAY[DEFAULT_TIME_OF_DAY]),
    )


__all__ = [
    "DEFAULT_TIME_OF_DAY",
    "TIMES_OF_DAY",
    "ProductionPlan",
    "TimeOfDay",
    "plan_production",
]
