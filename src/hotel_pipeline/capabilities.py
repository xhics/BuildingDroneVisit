"""Prérequis des commandes, déclarés une fois (portabilité, commit 1).

Dix-neuf commandes appelaient le même chargeur permissif : profil absent,
avertissement jaune, exécution quand même. Le verrou anti-confusion — le
risque nº 1 du plan directeur — se désarmait donc sur une ligne d'avertissement,
et une enseigne concurrente lue sans profil rendait `uncertain` au lieu de
`mismatch`.

Recopier un garde dans chaque commande aurait produit dix-neuf occasions d'en
oublier un ; c'est exactement ainsi que `blocking()` avait divergé de
`role_for`. Les prérequis se déclarent donc **par capacité**, une fois, et le
contexte les valide avant toute mutation.

Deux règles tiennent l'ensemble :

- `load_lenient` n'est plus la voie d'accès par défaut : seules les capacités
  `BOOTSTRAP` et `INSPECTION` y ont droit ;
- « lecture partielle autorisée » n'est pas « silencieuse » — une inspection
  sans profil doit dire ce qu'elle ne peut pas établir, sinon elle redevient
  le faux succès qu'on corrige.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .logging import get_logger

log = get_logger("capabilities")


class Capability(StrEnum):
    """Ce qu'une commande a besoin d'avoir sous la main."""

    #: Crée ce que les autres exigent. Ne peut donc rien exiger.
    BOOTSTRAP = "bootstrap"

    #: Lecture partielle autorisée, et signalée comme partielle.
    INSPECTION = "inspection"

    #: Décider si une image montre **le** bâtiment demande de savoir lequel.
    IDENTITY_CLASSIFICATION = "identity_classification"

    #: Interroger une source autour d'un point demande le point.
    TARGETED_COLLECTION = "targeted_collection"

    #: Territoire et référentiels résolus.
    GEOSPATIAL = "geospatial"

    #: Seuils matérialisés sur le disque, non des défauts implicites.
    QUALIFICATION = "qualification"


class Requirement(StrEnum):
    """Un élément qu'une capacité exige du contexte."""

    PROFILE = "profile"
    POSITION = "position"
    POLICY = "policy"
    MATERIALISED_POLICY = "materialised_policy"
    SPATIAL_CONTEXT = "spatial_context"


#: La matrice. Elle est la seule autorité : une commande ne redéclare rien.
REQUIREMENTS: dict[Capability, tuple[Requirement, ...]] = {
    Capability.BOOTSTRAP: (),
    Capability.INSPECTION: (),
    Capability.IDENTITY_CLASSIFICATION: (Requirement.PROFILE,),
    Capability.TARGETED_COLLECTION: (
        Requirement.PROFILE, Requirement.POSITION, Requirement.POLICY,
    ),
    Capability.GEOSPATIAL: (Requirement.SPATIAL_CONTEXT,),
    Capability.QUALIFICATION: (Requirement.MATERIALISED_POLICY,),
}

#: Capacités autorisées à travailler sans profil. Ailleurs, son absence arrête.
LENIENT: frozenset[Capability] = frozenset(
    {Capability.BOOTSTRAP, Capability.INSPECTION}
)

#: Ce que chaque élément manquant coûte, et comment l'obtenir. Une erreur qui
#: nomme la commande à lancer épargne une lecture du code.
REMEDY: dict[Requirement, tuple[str, str]] = {
    Requirement.PROFILE: (
        "l'établissement visé n'est pas décrit : identité, concurrents, pays, "
        "fuseau et langues sont inconnus",
        "hotel-pipeline init <hotel_id> --address … puis renseignez "
        "profiles/<property_id>.json",
    ),
    Requirement.POSITION: (
        "aucune position connue : une collecte ciblée n'a pas de centre",
        "renseignez lat/lon dans le profil, ou lancez la résolution d'adresse",
    ),
    Requirement.POLICY: (
        "aucune politique chargée",
        "hotel-pipeline init <hotel_id>",
    ),
    Requirement.MATERIALISED_POLICY: (
        "la politique n'est pas matérialisée : les seuils viendraient de "
        "valeurs implicites, qu'aucun rapport ne pourrait citer",
        "hotel-pipeline init <hotel_id> écrit 00_manifest/pipeline_policy.json",
    ),
    Requirement.SPATIAL_CONTEXT: (
        "territoire et référentiels non résolus",
        "hotel-pipeline geo sources <hotel_id>",
    ),
}


class CapabilityUnavailable(RuntimeError):
    """Une capacité manque. Typée, et porteuse de ce qui la rétablirait."""

    def __init__(self, capability: Capability, missing: list[Requirement]) -> None:
        self.capability = capability
        self.missing = list(missing)
        details = "\n".join(
            f"    · {item.value} — {REMEDY[item][0]}\n      → {REMEDY[item][1]}"
            for item in self.missing
        )
        super().__init__(
            f"capacité « {capability.value} » indisponible :\n{details}"
        )


@dataclass
class CapabilityCheck:
    """Résultat d'une vérification, y compris ce qu'elle laisse indéterminé."""

    capability: Capability
    satisfied: bool = True
    missing: list[Requirement] = field(default_factory=list)
    partial: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "capability": self.capability.value,
            "satisfied": self.satisfied,
            "missing": [item.value for item in self.missing],
            "partial": self.partial,
        }

    def raise_if_unsatisfied(self) -> None:
        if not self.satisfied:
            raise CapabilityUnavailable(self.capability, self.missing)


def available(context, capability: Capability) -> CapabilityCheck:  # noqa: ANN001
    """Le contexte satisfait-il cette capacité ?

    Ne mute rien et ne journalise aucun avertissement rassurant : le résultat
    est rendu à l'appelant, qui décide d'arrêter ou de signaler.
    """
    check = CapabilityCheck(capability=capability)

    for requirement in REQUIREMENTS[capability]:
        if not _satisfied(context, requirement):
            check.missing.append(requirement)

    check.satisfied = not check.missing

    # Une capacité permissive reste permissive, mais dit ce qui lui manque.
    if capability in LENIENT:
        check.partial = [
            REMEDY[item][0]
            for item in (Requirement.PROFILE, Requirement.SPATIAL_CONTEXT)
            if not _satisfied(context, item)
        ]
        check.satisfied = True
        check.missing = []

    return check


def _satisfied(context, requirement: Requirement) -> bool:  # noqa: ANN001
    profile = getattr(context, "profile", None)

    if requirement is Requirement.PROFILE:
        return profile is not None
    if requirement is Requirement.POSITION:
        return (
            profile is not None
            and profile.lat is not None
            and profile.lon is not None
        )
    if requirement is Requirement.POLICY:
        return getattr(context, "policy", None) is not None
    if requirement is Requirement.MATERIALISED_POLICY:
        # Une politique dont les seuils viennent de défauts implicites ne peut
        # pas être citée : le rapport nommerait des valeurs qu'aucun fichier ne
        # porte. `policy_defaults_applied` les recense déjà.
        return getattr(context, "policy", None) is not None and not getattr(
            context, "policy_defaults_applied", ()
        )
    if requirement is Requirement.SPATIAL_CONTEXT:
        return getattr(context, "spatial_reference", None) is not None
    return False  # pragma: no cover — l'énumération est close


def require(context, capability: Capability) -> CapabilityCheck:  # noqa: ANN001
    """Vérifie, et arrête si la capacité manque."""
    check = available(context, capability)
    check.raise_if_unsatisfied()
    if check.partial:
        log.info(
            "capacité %s : lecture partielle — %s",
            capability.value, " ; ".join(check.partial),
        )
    return check
