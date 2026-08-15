"""Résolution d'un objet de site par son **type** (collecte V2, correctif).

`demands build` indexait le manifeste de site par `object_id`, en cherchant
ensuite par type. Les identifiants réels étant préfixés du site —
`welcominns-boucherville:PARKING_HOTEL` — la jointure ne pouvait jamais
aboutir : trois objets bel et bien présents, dont un géoréférencé, étaient
déclarés absents.

Le défaut n'était pas seulement une clé fausse. Il produisait une **affirmation
fausse** — « la cible n'existe pas » — que le reste du pipeline prenait pour un
constat, et il supprimait deux besoins obligatoires du manifeste. Un système
sûr, et bloqué.

D'où ce point de passage unique. Il compare le type au type, conserve
l'identifiant d'instance complet, et distingue quatre situations que la
question « existe-t-il ? » confondait :

```text
absent                  aucun objet de ce type sur le site
présent, non résolu     l'objet est connu, sa géométrie n'est pas établie
présent, sans géométrie  inféré, mais rien à viser
présent, ciblable       une géométrie exploitable existe
```

Aucun découpage textuel de l'identifiant : dépendre du format `site:TYPE`
remplacerait une jointure fausse par une jointure fragile.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .logging import get_logger
from .schemas.enums import ObjectState

log = get_logger("site-resolution")


class Resolution(StrEnum):
    """Ce qu'on peut dire d'un objet de site, pour un besoin donné."""

    #: Aucun objet de ce type au manifeste.
    ABSENT = "absent"

    #: Présent, mais l'opérateur ne l'a pas encore résolu.
    UNRESOLVED = "unresolved"

    #: Présent et inféré, sans géométrie : on sait qu'il existe, pas où.
    NO_GEOMETRY = "no_geometry"

    #: Présent avec une géométrie exploitable : on sait le viser.
    TARGETABLE = "targetable"


#: Types dont une seule instance a un sens sur un site. Deux façades avant, ou
#: deux entrées principales, signalent une erreur de construction du manifeste
#: — choisir la première la masquerait.
SINGLETON_KINDS: frozenset[str] = frozenset(
    {
        "BUILDING_MAIN", "PROPERTY_PARCEL", "ENTRANCE_MAIN_CURRENT",
        "ROOFLINE_MAIN", "TERRAIN_MAIN", "PROPERTY_SIGN", "ACCESS_ROAD_MAIN",
        "DRIVEWAY_MAIN", "PARKING_HOTEL",
        "FACADE_PRIMARY", "FACADE_LEFT", "FACADE_RIGHT", "FACADE_REAR",
    }
)


class AmbiguousSiteObject(RuntimeError):
    """Plusieurs instances d'un type qui n'en admet qu'une."""


@dataclass(frozen=True)
class SiteObjectResolution:
    """Ce qu'on sait d'un type d'objet sur ce site."""

    kind: str
    resolution: Resolution

    #: Identifiant **d'instance**, complet. C'est lui qui référence l'objet
    #: ailleurs ; le type seul ne suffit pas à le désigner.
    object_id: str | None = None
    state: ObjectState | None = None
    has_geometry: bool = False

    @property
    def exists(self) -> bool:
        """L'objet est-il **établi** sur ce site ?

        `unresolved` ne l'établit pas : le manifeste de site instancie tous les
        types du gabarit, et un objet non résolu n'y est qu'un emplacement
        réservé. Le compter comme existant ferait consacrer des requêtes à une
        allée dont rien ne dit qu'elle existe.
        """
        return self.resolution in (Resolution.NO_GEOMETRY, Resolution.TARGETABLE)

    @property
    def is_instantiated(self) -> bool:
        """Un objet de ce type figure au manifeste, quel que soit son état."""
        return self.resolution is not Resolution.ABSENT

    @property
    def is_targetable(self) -> bool:
        return self.resolution is Resolution.TARGETABLE

    def why(self) -> str:
        """Ce qu'il faut dire à un lecteur de rapport."""
        if self.resolution is Resolution.ABSENT:
            return "aucun objet de ce type au manifeste de site"
        if self.resolution is Resolution.UNRESOLVED:
            return (
                f"{self.object_id} présent mais non résolu : le besoin est réel, "
                "sa cible précise ne l'est pas encore"
            )
        if self.resolution is Resolution.NO_GEOMETRY:
            return (
                f"{self.object_id} inféré sans géométrie : on sait qu'il existe, "
                "pas où le viser"
            )
        return f"{self.object_id} ciblable"


def resolve_site_object(site, kind: str) -> SiteObjectResolution:  # noqa: ANN001
    """Résout un **type** d'objet sur ce site.

    Compare `kind` à `kind`, jamais à l'identifiant d'instance : c'est
    précisément la confusion qui déclarait absents trois objets présents.
    """
    matches = [
        obj for obj in getattr(site, "objects", []) or []
        if getattr(obj, "kind", None) == kind
    ]

    if not matches:
        return SiteObjectResolution(kind=kind, resolution=Resolution.ABSENT)

    if len(matches) > 1 and kind in SINGLETON_KINDS:
        # Prendre la première masquerait une erreur de construction : deux
        # entrées principales ne sont pas un choix, c'est une contradiction.
        raise AmbiguousSiteObject(
            f"{kind} : {len(matches)} instances — "
            f"{sorted(obj.object_id for obj in matches)}. Ce type n'en admet "
            "qu'une ; en choisir une masquerait l'erreur."
        )

    instance = matches[0]
    has_geometry = bool(getattr(instance, "geometry_wkt", None))

    if instance.state is ObjectState.UNRESOLVED:
        resolution = Resolution.UNRESOLVED
    elif has_geometry:
        resolution = Resolution.TARGETABLE
    else:
        resolution = Resolution.NO_GEOMETRY

    return SiteObjectResolution(
        kind=kind, resolution=resolution, object_id=instance.object_id,
        state=instance.state, has_geometry=has_geometry,
    )


def resolve_all(site, kinds: list[str]) -> dict[str, SiteObjectResolution]:  # noqa: ANN001
    """Résout plusieurs types d'un coup, en propageant les ambiguïtés."""
    return {kind: resolve_site_object(site, kind) for kind in kinds}
