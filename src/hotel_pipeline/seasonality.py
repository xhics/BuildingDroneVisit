"""Saison d'une prise de vue, et ce qu'elle autorise à dire (Lot 2).

Le pipeline traitait tout le corpus comme intemporel. `season` existait sur
`ImageAsset` et valait `None` sur les 349 actifs du pilote, alors même que la
donnée était disponible **et déjà payée** : l'endpoint metadata de Street View
rend un champ `date` au format `AAAA-MM`, mis en cache lors de la découverte.
721 panoramas du pilote sont ainsi datés sans un seul appel supplémentaire.

Cela compte dès qu'on sort du bâtiment. Un mur de brique est le même en
janvier et en juillet ; une plate-bande, un gazon, un massif ne le sont pas.
Composer une vidéo à partir de références de saisons différentes juxtapose des
états qui n'ont jamais coexisté — un jardin fleuri contre des arbres nus.

Deux niveaux, et l'ordre importe
--------------------------------
1. **Le mois propose.** Il est gratuit, disponible partout, et suffit à
   classer grossièrement. Mais il ne décide pas : au Québec, le 15 avril peut
   être enneigé comme printanier. Mesuré sur le pilote — une vue d'avril 2025
   montre un arbre en fleurs et un gazon vert, sans trace de neige.
2. **Les pixels confirment.** La verdure au sol et l'indice de neige mesurent
   l'état réel. C'est la même discipline que `subject_prominence` : la
   géométrie ou le calendrier proposent, l'image tranche.

Ce que l'indice de neige vaut, et ne vaut pas
---------------------------------------------
Il est **délibérément faible**, et le module le déclare plutôt que de le
maquiller. La neige au sol n'est pas séparable d'une chaussée claire et sèche
par la seule couleur : les deux sont neutres, lumineuses et lisses.

Trois filtres successifs, mesurés sur le pilote (moyenne mensuelle de la part
de sol candidate) :

```text
filtre                              avril   mai   juillet
clair et peu saturé                  30 %   37 %    4 %     ← inutilisable
+ neutralité chromatique (RVB)       21 %    5 %    2 %
+ texture lisse (écart-type < 6)     10 %    1 %    0 %
```

La progression est réelle — mai passe de 37 % à 0,7 % — mais 16 % subsistent
sur une image d'avril **sans aucune neige**, où le candidat est la chaussée
d'un cul-de-sac résidentiel. D'où `snow_index` rendu comme indice, jamais
comme fraction de neige, et un verdict `snow_possible` qui n'affirme rien
seul : c'est le croisement avec le mois qui lui donne du poids.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .logging import get_logger

log = get_logger("seasonality")

#: Saisons, au sens de ce que le sol et la végétation montrent — non au sens
#: astronomique. Bornes fixées pour un climat continental humide (Québec).
#: Un site méditerranéen ou tropical demanderait d'autres bornes : elles sont
#: donc paramétrables, jamais gravées dans les décisions.
NORTHERN_SEASONS: dict[int, str] = {
    1: "winter", 2: "winter", 3: "winter",
    4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer", 9: "summer",
    10: "autumn", 11: "autumn", 12: "winter",
}

#: Mois où la végétation porte son feuillage : les seuls où l'apparence d'un
#: jardin est représentative de sa saison de croissance.
FOLIAGE_MONTHS = frozenset({5, 6, 7, 8, 9})

#: Part de sol verte au-delà de laquelle la végétation est active.
GREEN_ACTIVE = 0.08

#: Indice de neige au-delà duquel un enneigement est *possible*. Volontairement
#: haut : sous ce seuil, une chaussée claire suffit à faire du bruit.
SNOW_SUSPECT = 0.20

#: Bornes pixel de l'indice de neige. `sd` est l'écart-type local sur 9×9.
SNOW_MIN_VALUE = 195
SNOW_MAX_SATURATION = 28
SNOW_MAX_CHANNEL_SPREAD = 18
SNOW_MAX_TEXTURE = 6.0

#: Teinte de la végétation vivante, en HSV OpenCV.
GREEN_HUE_MIN, GREEN_HUE_MAX = 35, 85
GREEN_MIN_SATURATION, GREEN_MIN_VALUE = 60, 50

#: Sous cette part de l'image, la bande de sol est trop maigre pour mesurer.
MIN_GROUND_PIXELS = 5000


@dataclass
class SeasonReading:
    """Ce qu'une image dit de sa saison."""

    #: Saison déduite du mois de capture, quand il est connu.
    declared_season: str | None = None
    capture_month: int | None = None
    #: Part de sol verte, et indice de neige — `None` si non mesurés.
    green_fraction: float | None = None
    snow_index: float | None = None
    ground_pixels: int = 0
    status: str = "unknown"
    reason: str | None = None
    conflicts: list[str] = field(default_factory=list)

    @property
    def measured(self) -> bool:
        return self.green_fraction is not None

    @property
    def foliage_expected(self) -> bool | None:
        """Le feuillage devrait-il être présent à cette date ?"""
        if self.capture_month is None:
            return None
        return self.capture_month in FOLIAGE_MONTHS

    @property
    def vegetation_active(self) -> bool | None:
        """La végétation est-elle verte **sur l'image** ? `None` si non lu."""
        if self.green_fraction is None:
            return None
        return self.green_fraction >= GREEN_ACTIVE

    @property
    def snow_possible(self) -> bool | None:
        """Indice seulement : une chaussée claire suffit à le lever."""
        if self.snow_index is None:
            return None
        return self.snow_index >= SNOW_SUSPECT

    def as_dict(self) -> dict:
        return {
            "declared_season": self.declared_season,
            "capture_month": self.capture_month,
            "status": self.status,
            "reason": self.reason,
            "green_fraction": (
                round(self.green_fraction, 4) if self.green_fraction is not None else None
            ),
            "snow_index": (
                round(self.snow_index, 4) if self.snow_index is not None else None
            ),
            "vegetation_active": self.vegetation_active,
            "snow_possible": self.snow_possible,
            "foliage_expected": self.foliage_expected,
            "ground_pixels": self.ground_pixels,
            "conflicts": list(self.conflicts),
        }


def season_of_month(month: int | None, table: dict[int, str] | None = None) -> str | None:
    """Saison d'un mois, ou `None` si le mois est inconnu.

    Un mois absent rend `None`, jamais une saison par défaut : supposer l'été
    ferait entrer un jardin fleuri dans les références d'une scène hivernale.
    """
    if month is None:
        return None
    return (table or NORTHERN_SEASONS).get(int(month))


def parse_capture_date(date: str | None) -> tuple[int | None, int | None]:
    """`AAAA-MM` → `(année, mois)`. Tolère `AAAA` seul et les valeurs vides."""
    if not date:
        return None, None
    parts = str(date).strip().split("-")
    try:
        year = int(parts[0])
    except (ValueError, IndexError):
        return None, None
    month = None
    if len(parts) > 1:
        try:
            candidate = int(parts[1])
            if 1 <= candidate <= 12:
                month = candidate
        except ValueError:
            month = None
    return year, month


def ground_mask(image, horizon=None):  # noqa: ANN001
    """Pixels appartenant au sol : sous la ligne de toit, moitié basse.

    Le ciel et les façades fausseraient les deux mesures — un mur clair
    ressemble à de la neige, un arbre au-dessus du toit à du gazon. La
    frontière de ciel de `silhouette_fit` sert de plafond quand elle est
    disponible ; à défaut on retombe sur une borne fixe.
    """
    import math

    import numpy as np

    height, width = image.shape[:2]
    floor = int(height * 0.45)
    mask = np.zeros((height, width), dtype=bool)
    for column in range(width):
        top = floor
        if horizon is not None and not math.isnan(float(horizon[column])):
            top = max(floor, int(horizon[column]) + 4)
        mask[top:, column] = True
    return mask


def read(image, capture_date: str | None = None, horizon=None) -> SeasonReading:  # noqa: ANN001
    """Lit la saison d'une image : ce que le mois annonce, ce que le sol montre."""
    import cv2
    import numpy as np

    _year, month = parse_capture_date(capture_date)
    reading = SeasonReading(
        declared_season=season_of_month(month), capture_month=month,
    )

    mask = ground_mask(image, horizon)
    count = int(mask.sum())
    reading.ground_pixels = count
    if count < MIN_GROUND_PIXELS:
        reading.status = "no_ground"
        reading.reason = (
            f"{count} pixel(s) de sol : cadrage trop serré pour lire la saison"
        )
        return reading

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(int)
    saturation = hsv[:, :, 1].astype(int)
    value = hsv[:, :, 2].astype(int)
    blue = image[:, :, 0].astype(int)
    green = image[:, :, 1].astype(int)
    red = image[:, :, 2].astype(int)

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.blur(grey, (9, 9))
    deviation = np.sqrt(np.maximum(cv2.blur(grey * grey, (9, 9)) - mean * mean, 0.0))

    verdant = (
        (hue > GREEN_HUE_MIN) & (hue < GREEN_HUE_MAX)
        & (saturation > GREEN_MIN_SATURATION) & (value > GREEN_MIN_VALUE)
    )
    neutral = (
        (np.abs(red - green) < SNOW_MAX_CHANNEL_SPREAD)
        & (np.abs(green - blue) < SNOW_MAX_CHANNEL_SPREAD)
        & (np.abs(red - blue) < SNOW_MAX_CHANNEL_SPREAD)
    )
    snowy = (
        neutral & (value > SNOW_MIN_VALUE)
        & (saturation < SNOW_MAX_SATURATION)
        & (deviation < SNOW_MAX_TEXTURE)
    )

    reading.green_fraction = float((verdant & mask).sum() / count)
    reading.snow_index = float((snowy & mask).sum() / count)
    reading.status = "measured"

    # Le désaccord entre calendrier et pixels n'est pas une erreur à taire :
    # c'est l'information. Une vue d'avril sans neige et déjà verte est un
    # printemps précoce, et la référence reste utilisable pour une scène
    # printanière — mais pas pour une scène hivernale.
    if reading.foliage_expected is False and reading.vegetation_active:
        reading.conflicts.append(
            f"mois {month} hors saison de feuillage, pourtant "
            f"{reading.green_fraction:.0%} de sol vert : saison précoce ou tardive"
        )
    if reading.foliage_expected and reading.snow_possible:
        reading.conflicts.append(
            f"mois {month} en pleine végétation, pourtant indice de neige "
            f"{reading.snow_index:.0%} : surface claire probablement confondue"
        )
    return reading


def summarise(readings: list[SeasonReading]) -> dict:
    """Répartition saisonnière d'un corpus, et ce qui manque.

    `missing_seasons` est le plus utile : il dit quelles scènes ne peuvent
    **pas** être référencées par du réel. Sur le pilote, l'hiver est absent des
    721 panoramas datés — une vidéo enneigée y serait entièrement inventée.
    """
    seasons: dict[str, int] = {}
    undated = 0
    for item in readings:
        if item.declared_season is None:
            undated += 1
            continue
        seasons[item.declared_season] = seasons.get(item.declared_season, 0) + 1

    known = set(NORTHERN_SEASONS.values())
    measured = [r for r in readings if r.measured]
    return {
        "total": len(readings),
        "undated": undated,
        "by_season": dict(sorted(seasons.items())),
        "missing_seasons": sorted(known - set(seasons)),
        "measured": len(measured),
        "with_conflicts": sum(1 for r in readings if r.conflicts),
    }


__all__ = [
    "FOLIAGE_MONTHS",
    "GREEN_ACTIVE",
    "MIN_GROUND_PIXELS",
    "NORTHERN_SEASONS",
    "SNOW_SUSPECT",
    "SeasonReading",
    "ground_mask",
    "parse_capture_date",
    "read",
    "season_of_month",
    "summarise",
]
