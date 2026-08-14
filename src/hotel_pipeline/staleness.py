"""Péremption sélective : ce qui périme quoi (portabilité).

L'empreinte du profil a changé en gagnant pays, fuseau et langues. Faire de ce
changement une péremption générale rendrait périmés le MNT, la toiture et le
nuage LiDAR — qui ne dépendent d'aucun nom d'établissement. Un rapport
géospatial cite le profil pour dire **de quel site il parle**, non parce que le
profil a décidé de ses valeurs.

D'où la séparation demandée :

```text
provenance générale   le contexte enregistré, pour lire le rapport
dependency_digests    les entrées ayant réellement influencé le calcul
```

Un digest présent dans la provenance n'est pas automatiquement une dépendance.
Ce module dit lesquelles le sont, par nature de calcul.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .logging import get_logger

log = get_logger("staleness")


class Facet(StrEnum):
    """Ce qui peut changer dans un profil, et ce que cela engage.

    Découpé par **conséquence**, non par champ : deux champs qui périment les
    mêmes calculs appartiennent à la même facette, et un champ qui ne périme
    rien n'en a aucune.
    """

    #: Noms, alias, concurrents, langues d'OCR.
    IDENTITY = "identity"

    #: Fuseau horaire.
    TIMEZONE = "timezone"

    #: Position et adresse — ce dont un calcul part réellement.
    LOCATION = "location"

    #: Pays et subdivision. **Déclaratifs** : le territoire se résout depuis la
    #: position, jamais depuis ces champs. Les corriger ne périme donc rien,
    #: et aucun consommateur ne les cite — ce qui est le résultat voulu, non un
    #: oubli.
    TERRITORY_DECLARATION = "territory_declaration"

    #: Chambres, étages, emprise déclarée.
    SIZE = "size"

    #: Travaux datés.
    RENOVATION = "renovation"


#: Champs du profil, par facette. Ce qui n'y figure pas ne périme rien : une
#: URL de site ou une requête Places n'engage aucun calcul déjà produit.
FACET_FIELDS: dict[Facet, tuple[str, ...]] = {
    Facet.IDENTITY: ("official_name", "aliases", "competitor_names", "ocr_languages"),
    Facet.TIMEZONE: ("timezone",),
    Facet.LOCATION: ("lat", "lon", "address"),
    Facet.TERRITORY_DECLARATION: ("country_code", "subdivision_code"),
    Facet.SIZE: ("room_count", "expected_levels", "footprint_min_m2", "footprint_max_m2"),
    Facet.RENOVATION: ("renovation_events",),
}


@dataclass(frozen=True)
class Consumer:
    """Un type de production, et les facettes dont il dépend réellement."""

    name: str
    depends_on: frozenset[Facet]
    rationale: str

    def invalidated_by(self, changed: set[Facet]) -> bool:
        return bool(self.depends_on & changed)


#: Qui dépend de quoi. La règle qui gouverne ce tableau : une production dépend
#: d'une facette si sa **valeur** en découle — pas si elle la mentionne.
CONSUMERS: tuple[Consumer, ...] = (
    Consumer(
        "identity_classification", frozenset({Facet.IDENTITY}),
        "lire une enseigne suppose de savoir quels termes confirment et "
        "lesquels disqualifient",
    ),
    Consumer(
        "asset_review", frozenset({Facet.IDENTITY}),
        "une décision de visibilité porte sur « le » bâtiment : changer les "
        "noms change ce qui devait être reconnu",
    ),
    Consumer(
        "temporal_assessment", frozenset({Facet.TIMEZONE, Facet.RENOVATION}),
        "une année de capture se compare à des dates civiles locales",
    ),
    Consumer(
        "building_candidates", frozenset({Facet.LOCATION, Facet.SIZE}),
        "le classement des empreintes dérive de la position et de l'emprise "
        "plausible",
    ),
    Consumer(
        "spatial_reference", frozenset({Facet.LOCATION}),
        "territoire et référentiel de travail se résolvent depuis la position",
    ),
    # Les trois suivantes ne dépendent d'aucune facette du profil : elles
    # dépendent du contexte spatial et des données acquises, qui ont leurs
    # propres empreintes. Les périmer sur un renommage serait faux.
    Consumer(
        "elevation_derivation", frozenset(),
        "MNT, MNS et nDSM dérivent d'une tuile LiDAR et d'un référentiel, non "
        "d'un nom d'établissement",
    ),
    Consumer(
        "lidar_acquisition", frozenset(),
        "un fichier acquis est identifié par son empreinte, que le profil "
        "n'influence pas",
    ),
    Consumer(
        "geospatial_qualification", frozenset(),
        "les seuils portent sur une couverture et une erreur d'interpolation ; "
        "aucun ne lit le profil",
    ),
)


@dataclass
class StalenessReport:
    """Ce qui périme, ce qui survit, et pourquoi."""

    changed_facets: list[str] = field(default_factory=list)
    invalidated: list[str] = field(default_factory=list)
    preserved: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "changed_facets": self.changed_facets,
            "invalidated": self.invalidated,
            "preserved": self.preserved,
            "note": (
                "un changement de profil ne périme que ce dont la valeur en "
                "découle. Citer le profil dans la provenance ne suffit pas à "
                "créer une dépendance."
            ),
        }


def _normalise(value):  # noqa: ANN001, ANN201
    """Deux écritures d'une même valeur ne sont pas deux valeurs.

    Un champ facultatif absent et le même champ sérialisé à `null` disent la
    même chose. Les distinguer faisait périmer la datation du pilote au seul
    motif que le modèle écrivait désormais `started_on: null` explicitement.
    """
    if isinstance(value, dict):
        return {
            key: _normalise(item)
            for key, item in sorted(value.items())
            if item is not None
        }
    if isinstance(value, list):
        return [_normalise(item) for item in value]
    return value


def changed_facets(before: dict, after: dict) -> set[Facet]:
    """Facettes réellement modifiées entre deux profils sérialisés."""
    changed: set[Facet] = set()
    for facet, names in FACET_FIELDS.items():
        if any(
            _normalise(before.get(name)) != _normalise(after.get(name))
            for name in names
        ):
            changed.add(facet)
    return changed


def assess(before: dict, after: dict) -> StalenessReport:
    """Ce qu'un changement de profil périme, et ce qu'il laisse intact."""
    changed = changed_facets(before, after)
    report = StalenessReport(changed_facets=sorted(f.value for f in changed))

    for consumer in CONSUMERS:
        if consumer.invalidated_by(changed):
            report.invalidated.append(consumer.name)
        else:
            report.preserved.append(
                {"production": consumer.name, "because": consumer.rationale}
            )

    log.info(
        "péremption : %d facette(s) changée(s), %d production(s) périmée(s), "
        "%d préservée(s)",
        len(changed), len(report.invalidated), len(report.preserved),
    )
    return report
