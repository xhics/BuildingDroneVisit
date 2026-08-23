"""Ce qu'on a le droit d'affirmer d'un objet, selon sa permanence (Lot 2).

Tout le pipeline suppose un sujet **rigide et permanent**. C'est vrai d'un
bâtiment ; ce l'est de moins en moins à mesure qu'on s'éloigne de ses murs. Un
massif change de forme entre juin et octobre, disparaît sous la neige en
janvier, et n'existait pas il y a trois ans. Lui appliquer la même chaîne —
empreinte, hauteur, maillage — produirait une géométrie que rien n'atteste.

Ce module pose la règle inverse : **chaque objet reçoit une classe de
permanence, et cette classe borne ce qu'on produit pour lui.** Ce n'est pas
une taxinomie botanique, c'est un contrat sur la sortie.

```text
classe               exemples                    production autorisée
permanent            bâtiment, muret, asphalte   géométrie 3D mesurée
seasonal_structure   arbre mature, haie          volume approché + état saisonnier
seasonal_surface     plate-bande, gazon          surface 2D + palette saisonnière
ephemeral            fleurs, mobilier, neige     apparence seule, aucune géométrie
```

La règle qui rend le contrat utile
----------------------------------
**Une classe ne se promeut jamais par manque de données.** Un objet non
identifié reste `ephemeral`, donc ne produit aucune géométrie. C'est la même
discipline que `unknown_provenance` dans `panorama_provenance`, qui ne devient
jamais `attested` par défaut : l'ignorance penche du côté qui n'invente rien.

Pourquoi pas d'identification d'espèces
---------------------------------------
À 64–176 m, un massif occupe quelques dizaines de pixels. Aucun classifieur ne
distinguera là-dedans un hosta d'une hémérocalle, et prétendre le contraire
mettrait une étiquette botanique inventée sur une image illisible. Ce qui est
mesurable honnêtement — et suffisant pour composer une scène — est le **port**
(arbre, arbuste, couvre-sol), la **position**, et l'**état saisonnier**.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .logging import get_logger

log = get_logger("permanence")


class Permanence(StrEnum):
    """Degré de stabilité d'un objet dans le temps."""

    PERMANENT = "permanent"
    SEASONAL_STRUCTURE = "seasonal_structure"
    SEASONAL_SURFACE = "seasonal_surface"
    EPHEMERAL = "ephemeral"


class Production(StrEnum):
    """Ce qu'on est autorisé à produire pour un objet."""

    MESH_3D = "mesh_3d"
    APPROXIMATE_VOLUME = "approximate_volume"
    SURFACE_2D = "surface_2d"
    APPEARANCE_ONLY = "appearance_only"


#: Contrat : ce que chaque classe autorise. Une classe absente de cette table
#: ne produit rien — l'oubli penche du côté prudent.
ALLOWED: dict[Permanence, Production] = {
    Permanence.PERMANENT: Production.MESH_3D,
    Permanence.SEASONAL_STRUCTURE: Production.APPROXIMATE_VOLUME,
    Permanence.SEASONAL_SURFACE: Production.SURFACE_2D,
    Permanence.EPHEMERAL: Production.APPEARANCE_ONLY,
}

#: Classes dont l'apparence dépend de la saison : une référence d'une saison
#: ne vaut pas pour une autre.
SEASON_DEPENDENT = frozenset({
    Permanence.SEASONAL_STRUCTURE,
    Permanence.SEASONAL_SURFACE,
    Permanence.EPHEMERAL,
})

#: Correspondance entre types d'objets connus et permanence. Volontairement
#: courte : ce qui n'y figure pas reste `ephemeral`.
KNOWN_KINDS: dict[str, Permanence] = {
    "building": Permanence.PERMANENT,
    "wall": Permanence.PERMANENT,
    "parking": Permanence.PERMANENT,
    "road": Permanence.PERMANENT,
    "sidewalk": Permanence.PERMANENT,
    "sign": Permanence.PERMANENT,
    "tree": Permanence.SEASONAL_STRUCTURE,
    "hedge": Permanence.SEASONAL_STRUCTURE,
    "shrub": Permanence.SEASONAL_STRUCTURE,
    "lawn": Permanence.SEASONAL_SURFACE,
    "flowerbed": Permanence.SEASONAL_SURFACE,
    "planting": Permanence.SEASONAL_SURFACE,
    "flowers": Permanence.EPHEMERAL,
    "furniture": Permanence.EPHEMERAL,
    "snow": Permanence.EPHEMERAL,
    "vehicle": Permanence.EPHEMERAL,
}


@dataclass
class SceneObject:
    """Un objet de la scène, et ce que ses observations autorisent."""

    object_id: str
    kind: str | None = None
    permanence: Permanence = Permanence.EPHEMERAL
    #: Dates distinctes (`AAAA-MM`) où l'objet a été observé.
    observed_dates: set[str] = field(default_factory=set)
    #: Saisons distinctes où il a été observé.
    observed_seasons: set[str] = field(default_factory=set)
    #: Variation d'apparence entre dates, quand elle a été mesurée.
    temporal_variance: float | None = None
    evidence: list[str] = field(default_factory=list)

    @property
    def production(self) -> Production:
        """Ce qu'on est autorisé à produire. Jamais plus que la classe permet."""
        return ALLOWED.get(self.permanence, Production.APPEARANCE_ONLY)

    @property
    def season_dependent(self) -> bool:
        return self.permanence in SEASON_DEPENDENT

    @property
    def seasons_missing(self) -> bool:
        """Une classe saisonnière observée en une seule saison est incomplète."""
        return self.season_dependent and len(self.observed_seasons) < 2

    def as_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "kind": self.kind,
            "permanence": str(self.permanence),
            "production": str(self.production),
            "season_dependent": self.season_dependent,
            "seasons_missing": self.seasons_missing,
            "observed_dates": sorted(self.observed_dates),
            "observed_seasons": sorted(self.observed_seasons),
            "temporal_variance": (
                round(self.temporal_variance, 4)
                if self.temporal_variance is not None else None
            ),
            "evidence": list(self.evidence),
        }


def classify_kind(kind: str | None) -> Permanence:
    """Permanence d'un type d'objet. L'inconnu reste éphémère.

    Ne jamais rendre `PERMANENT` par défaut : une classe promue sans preuve
    ferait produire un maillage 3D pour un massif de fleurs.
    """
    if not kind:
        return Permanence.EPHEMERAL
    return KNOWN_KINDS.get(str(kind).strip().lower(), Permanence.EPHEMERAL)


#: Variance d'apparence sous laquelle un objet est tenu pour permanent, et
#: au-dessus de laquelle il est tenu pour saisonnier. Entre les deux, la
#: mesure ne tranche pas.
STABLE_BELOW = 0.10
SEASONAL_ABOVE = 0.25

#: Dates distinctes nécessaires pour que la variance signifie quelque chose.
MIN_DATES_FOR_VARIANCE = 3


def infer_from_variance(
    variance: float | None, distinct_dates: int
) -> tuple[Permanence | None, str]:
    """Permanence déduite de la variation d'apparence entre dates.

    Le principe : **ce qui apparaît identique à toutes les dates est
    permanent ; ce qui varie est saisonnier.** Une zone verte en juin, brune en
    avril et blanche en janvier est du gazon ; une zone identique partout est
    de l'asphalte ou du bâti. La variance temporelle *est* le signal, sans
    qu'aucun classifieur botanique n'intervienne.

    Rend `None` quand la mesure ne tranche pas — trop peu de dates, ou variance
    dans la zone grise. Un `None` n'est pas un défaut : c'est le refus de
    conclure, et l'appelant garde alors la classe la plus prudente.
    """
    if variance is None:
        return None, "variance non mesurée"
    if distinct_dates < MIN_DATES_FOR_VARIANCE:
        return None, (
            f"{distinct_dates} date(s) distincte(s) : il en faut "
            f"{MIN_DATES_FOR_VARIANCE} pour que la stabilité signifie quelque chose"
        )
    if variance < STABLE_BELOW:
        return Permanence.PERMANENT, (
            f"apparence stable sur {distinct_dates} dates (variance "
            f"{variance:.2f} < {STABLE_BELOW})"
        )
    if variance > SEASONAL_ABOVE:
        return Permanence.SEASONAL_SURFACE, (
            f"apparence variable sur {distinct_dates} dates (variance "
            f"{variance:.2f} > {SEASONAL_ABOVE})"
        )
    return None, (
        f"variance {variance:.2f} entre {STABLE_BELOW} et {SEASONAL_ABOVE} : "
        "ni stable ni franchement saisonnier"
    )


def resolve(
    object_id: str,
    *,
    kind: str | None = None,
    variance: float | None = None,
    dates: set[str] | None = None,
    seasons: set[str] | None = None,
) -> SceneObject:
    """Compose la permanence d'un objet à partir de tout ce qu'on en sait.

    Le type déclaré prime quand il est connu — il vient d'une source, non d'une
    inférence. La variance ne sert qu'à trancher ce que le type laisse ouvert,
    et ne peut que **rétrograder** un objet vers moins de permanence : mesurer
    une apparence stable sur trois photos ne suffit pas à promouvoir un massif
    au rang de bâtiment.
    """
    dates = dates or set()
    seasons = seasons or set()
    scene_object = SceneObject(
        object_id=object_id, kind=kind, observed_dates=set(dates),
        observed_seasons=set(seasons), temporal_variance=variance,
    )

    declared = classify_kind(kind)
    if kind and declared is not Permanence.EPHEMERAL:
        scene_object.permanence = declared
        scene_object.evidence.append(f"type déclaré « {kind} » → {declared}")
    else:
        inferred, why = infer_from_variance(variance, len(dates))
        scene_object.evidence.append(why)
        if inferred is not None:
            scene_object.permanence = inferred
        else:
            scene_object.permanence = Permanence.EPHEMERAL
            scene_object.evidence.append(
                "faute de preuve, classé éphémère : aucune géométrie produite"
            )

    # Garde-fou : une apparence franchement variable interdit `permanent`,
    # même si le type le prétendait. Un « mur » couvert de vigne vierge n'est
    # pas rendu correctement par un maillage sans saison.
    if (
        scene_object.permanence is Permanence.PERMANENT
        and variance is not None
        and variance > SEASONAL_ABOVE
        and len(dates) >= MIN_DATES_FOR_VARIANCE
    ):
        scene_object.permanence = Permanence.SEASONAL_STRUCTURE
        scene_object.evidence.append(
            f"rétrogradé : variance {variance:.2f} incompatible avec un objet permanent"
        )
    return scene_object


def summarise(objects: list[SceneObject]) -> dict:
    """Bilan : ce qui est modélisable, ce qui ne l'est pas, ce qui manque."""
    by_permanence: dict[str, int] = {}
    by_production: dict[str, int] = {}
    for item in objects:
        by_permanence[str(item.permanence)] = by_permanence.get(str(item.permanence), 0) + 1
        by_production[str(item.production)] = by_production.get(str(item.production), 0) + 1
    incomplete = [o.object_id for o in objects if o.seasons_missing]
    return {
        "total": len(objects),
        "by_permanence": dict(sorted(by_permanence.items())),
        "by_production": dict(sorted(by_production.items())),
        "geometry_eligible": sum(
            1 for o in objects if o.production is Production.MESH_3D
        ),
        "single_season_objects": incomplete,
    }


__all__ = [
    "ALLOWED",
    "KNOWN_KINDS",
    "MIN_DATES_FOR_VARIANCE",
    "Permanence",
    "Production",
    "SEASONAL_ABOVE",
    "SEASON_DEPENDENT",
    "STABLE_BELOW",
    "SceneObject",
    "classify_kind",
    "infer_from_variance",
    "resolve",
    "summarise",
]
