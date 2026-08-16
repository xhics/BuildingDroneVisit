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

La règle exacte tient en une phrase :

> une capacité ne peut manquer aucun de ses prérequis déclarés ; l'absence de
> profil n'est une erreur que si `PROFILE` figure dans ses prérequis.

Ce n'est pas la même chose que « seules `BOOTSTRAP` et `INSPECTION` tournent
sans profil », qui serait faux : la qualification géospatiale n'exige aucune
identité d'établissement — des seuils de terrain ne dépendent pas du nom de
l'hôtel — et tourne donc légitimement sans profil tout en ayant, elle, des
prérequis stricts.

`PARTIAL_CONTEXT_ALLOWED` décrit autre chose encore : la possibilité de rendre
un résultat **partiel**, en disant ce qui manque. Ce n'est ni une dispense de
prérequis, ni une autorisation générale de tourner sans profil. Et « partiel »
n'est pas « silencieux » — une inspection qui tait ce qu'elle ne peut pas
établir redevient le faux succès qu'on corrige.
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

    #: Qualifier un terrain ou une toiture. Nommée « géospatiale » parce
    #: qu'une qualification **photographique** viendra, et qu'elle n'aura pas
    #: besoin du contexte spatial : les deux ne partagent que le mot.
    GEOSPATIAL_QUALIFICATION = "geospatial_qualification"


class Requirement(StrEnum):
    """Un élément qu'une capacité exige du contexte."""

    PROFILE = "profile"
    POSITION = "position"
    POLICY = "policy"
    MATERIALISED_POLICY = "materialised_policy"
    SPATIAL_CONTEXT = "spatial_context"
    SITE_MANIFEST = "site_manifest"

    #: Les artefacts attendus **et** leurs empreintes. Qualifier une dérivation
    #: dont on ne peut pas nommer les entrées produirait un verdict qu'aucun
    #: rapport ne pourrait rattacher à ce qui l'a fondé.
    EXPECTED_ARTIFACTS = "expected_artifacts"

    #: Provenance verticale suffisante pour les critères qui en dépendent.
    #: Sans elle, une hauteur qualifiée reposerait sur un référentiel supposé.
    VERTICAL_PROVENANCE = "vertical_provenance"


#: La matrice. Elle est la seule autorité : une commande ne redéclare rien.
REQUIREMENTS: dict[Capability, tuple[Requirement, ...]] = {
    Capability.BOOTSTRAP: (),
    Capability.INSPECTION: (),
    Capability.IDENTITY_CLASSIFICATION: (Requirement.PROFILE,),
    # La collecte ciblée **décide** sur des seuils : secteur, portée, quotas,
    # résolutions. Une politique implicite les ferait venir du code, et aucun
    # rapport ne pourrait citer ce sur quoi il a jugé. `POLICY` seul exigeait
    # qu'une politique existe, non que ses facettes décisionnelles soient
    # inscrites.
    Capability.TARGETED_COLLECTION: (
        Requirement.PROFILE,
        Requirement.POSITION,
        Requirement.MATERIALISED_POLICY,
    ),
    Capability.GEOSPATIAL: (Requirement.SPATIAL_CONTEXT,),
    Capability.GEOSPATIAL_QUALIFICATION: (
        Requirement.MATERIALISED_POLICY,
        Requirement.SITE_MANIFEST,
        Requirement.EXPECTED_ARTIFACTS,
        Requirement.SPATIAL_CONTEXT,
        Requirement.VERTICAL_PROVENANCE,
    ),
}

#: Capacités pouvant rendre un résultat **partiel** en disant ce qui manque.
#: Ce n'est ni une dispense de prérequis — elles n'en ont aucun — ni la liste
#: des capacités tournant sans profil : `GEOSPATIAL_QUALIFICATION` n'exige
#: aucune identité et n'y figure pourtant pas, parce qu'elle ne rend rien de
#: partiel : ses prérequis sont satisfaits ou elle s'arrête.
PARTIAL_CONTEXT_ALLOWED: frozenset[Capability] = frozenset(
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
    Requirement.SITE_MANIFEST: (
        "aucun manifeste de site : les instances à qualifier ne sont pas nommées",
        "hotel-pipeline site build <hotel_id>",
    ),
    Requirement.EXPECTED_ARTIFACTS: (
        "artefacts attendus absents ou sans empreinte : le verdict ne pourrait "
        "pas être rattaché à ce qui l'a fondé",
        "hotel-pipeline geo derive <hotel_id>",
    ),
    Requirement.VERTICAL_PROVENANCE: (
        "provenance verticale insuffisante : une hauteur qualifiée reposerait "
        "sur un référentiel supposé",
        "hotel-pipeline geo acquire <hotel_id> puis geo derive",
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
    if capability in PARTIAL_CONTEXT_ALLOWED:
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
    if requirement is Requirement.SITE_MANIFEST:
        return getattr(context, "site_manifest", None) is not None
    if requirement is Requirement.EXPECTED_ARTIFACTS:
        return bool(getattr(context, "artifact_digests", None))
    if requirement is Requirement.VERTICAL_PROVENANCE:
        # Un référentiel vertical **connu**, ou une transformation déclarée.
        # « Inconnu » n'interdit pas de mesurer ; il interdit de qualifier une
        # hauteur, car le seuil porterait sur une origine supposée.
        reference = getattr(context, "spatial_reference", None)
        return reference is not None and reference.vertical_is_usable
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
