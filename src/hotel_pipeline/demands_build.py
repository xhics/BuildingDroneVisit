"""Instanciation des besoins depuis les obligations (collecte V2).

Le chaînon entre le gabarit et la collecte. Sans lui, `capture_demands.json`
s'écrivait à la main : rien ne garantissait qu'il couvre les obligations, et un
oubli de façade arrière ne se distinguait pas d'un choix.

Ce module ne décide rien de nouveau. Il traduit — obligation par obligation —
ce que le gabarit exige en besoins que le reste du pipeline sait juger. Trois
refus le tiennent :

```text
cible inconnue        une demande qui ne vise rien ne se satisfait jamais
objet non résolu      il produit un besoin non résolu, jamais une dispense
seuil codé ici        les valeurs viennent de la politique, pas du générateur
```

Le deuxième est le plus important. Convertir automatiquement un objet
`unresolved` en dispense ferait disparaître un manque en le déclarant sans
objet : c'est exactement l'oubli silencieux que les obligations empêchent, et
il reviendrait par la porte du générateur.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .logging import get_logger
from .schemas.acquisition import (
    CaptureDemand,
    CaptureDemandManifest,
    CaptureIntent,
    TargetKind,
)

log = get_logger("demands-build")

#: Préfixe des besoins générés. Déterministe : deux exécutions sur le même
#: gabarit produisent les mêmes identifiants, et un besoin conserve le sien
#: d'une reconstruction à l'autre.
GENERATED_PREFIX = "obligation:"


class DemandsRefused(RuntimeError):
    """Rien n'a été construit, et rien n'a été écrit."""


@dataclass
class BuildReport:
    """Ce qui a été généré, préservé, ou laissé sans besoin."""

    generated_from_obligation: list[str] = field(default_factory=list)
    operator_defined: list[str] = field(default_factory=list)
    waived: dict[str, str] = field(default_factory=dict)
    not_applicable: dict[str, str] = field(default_factory=dict)
    unresolved_target: dict[str, str] = field(default_factory=dict)

    #: Identifiant d'instance retenu, par type. Le type seul ne désigne rien :
    #: c'est l'instance qui est référencée ailleurs.
    resolved_instances: dict[str, str] = field(default_factory=dict)

    #: Où chercher tant qu'une cible n'est pas résolue. Une vue obtenue par ce
    #: détour demande vérification, elle n'établit pas la cible.
    search_proxies: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "generated_from_obligation": sorted(self.generated_from_obligation),
            "operator_defined": sorted(self.operator_defined),
            "waived": self.waived,
            "not_applicable": self.not_applicable,
            "unresolved_target": self.unresolved_target,
            "resolved_instances": self.resolved_instances,
            "search_proxies": self.search_proxies,
            "bytes_downloaded": 0,
            "note": (
                "un objet non résolu produit un besoin non résolu, jamais une "
                "dispense : le déclarer sans objet ferait disparaître un manque"
            ),
        }


def demand_id_for(object_id: str) -> str:
    return f"{GENERATED_PREFIX}{object_id}"


def is_generated(demand_id: str) -> bool:
    return demand_id.startswith(GENERATED_PREFIX)


def _thresholds(intent: CaptureIntent, coverage) -> dict:  # noqa: ANN001
    """Ce qu'exige une intention, **depuis la politique**.

    Les coder ici les aurait mis deux fois — une dans le générateur, une dans
    le manifeste produit — et le second aurait fini par mentir sur le premier.
    """
    if intent is CaptureIntent.BUILDING_CAPTURE:
        return {
            "viewpoints_required": coverage.building_viewpoints_required,
            "continuity_required": coverage.building_continuity_required,
            "min_projected_width_fraction": coverage.building_min_projected_width,
            "min_visible_fraction": coverage.building_min_visible_fraction,
        }
    return {
        "viewpoints_required": coverage.context_viewpoints_required,
        "continuity_required": coverage.context_continuity_required,
        "min_projected_width_fraction": coverage.context_min_projected_width,
        "min_visible_fraction": coverage.context_min_visible_fraction,
    }


def build(
    hotel_id: str,
    site,  # noqa: ANN001 — SiteManifest
    coverage,  # noqa: ANN001 — CoveragePolicy
    existing: CaptureDemandManifest | None = None,
    waivers: list | None = None,
    digests: dict[str, str | None] | None = None,
) -> tuple[CaptureDemandManifest, BuildReport]:
    """Traduit les obligations en besoins, sans jamais rien télécharger.

    Les demandes écrites par l'opérateur sont **préservées** : le générateur ne
    possède pas le manifeste, il y ajoute ce que le gabarit exige. Écraser
    aurait fait disparaître à chaque exécution ce qu'une personne avait ajouté.
    """
    from .coverage_obligations import OBLIGATIONS, Applicability, ObligationStatus
    from .site_resolution import Resolution, resolve_site_object

    if site is None:
        raise DemandsRefused(
            "aucun manifeste de site : les objets à couvrir ne sont pas nommés. "
            "Lancez « site build » d'abord."
        )

    dispensed = {waiver.object_id: waiver for waiver in (waivers or [])}

    report = BuildReport()
    demands: dict[str, CaptureDemand] = {}

    # Les demandes de l'opérateur d'abord : elles gardent leur place, et une
    # obligation qui viserait la même cible n'en crée pas une seconde.
    for demand in getattr(existing, "demands", []) or []:
        if is_generated(demand.demand_id):
            continue
        demands[demand.demand_id] = demand
        report.operator_defined.append(demand.demand_id)

    for obligation in OBLIGATIONS:
        object_id = obligation.object_id
        demand_id = demand_id_for(object_id)

        waiver = dispensed.get(object_id)
        if waiver is not None:
            bucket = (
                report.waived
                if waiver.status is ObligationStatus.WAIVED
                else report.not_applicable
            )
            bucket[object_id] = waiver.rationale
            continue

        # Résolution **par type**, jamais par identifiant d'instance : les
        # identifiants sont préfixés du site, et comparer les deux déclarait
        # absents des objets bel et bien présents.
        resolved = resolve_site_object(site, object_id)

        if obligation.applicability is Applicability.OPERATOR_ONLY:
            continue
        if (
            obligation.applicability is Applicability.WHEN_OBJECT_EXISTS
            and not resolved.exists
        ):
            # Son absence est établie : rien n'est dû, et le dire évite qu'on
            # la croie oubliée.
            report.not_applicable[object_id] = resolved.why()
            continue

        # Le besoin existe **même** si sa cible n'est pas encore résolue. Le
        # supprimer confondrait « la cible n'est pas résolue » avec « le besoin
        # n'existe pas » : la découverte ne chercherait jamais cette cible, et
        # le plan serait bloqué sans rien pour le débloquer.
        targetable = resolved.is_targetable
        if not targetable and obligation.target_kind is TargetKind.SITE_OBJECT:
            report.unresolved_target[object_id] = resolved.why()

        demands[demand_id] = CaptureDemand(
            demand_id=demand_id,
            intent=obligation.intent,
            target_kind=obligation.target_kind,
            target_ref=obligation.expected_target_ref,
            rationale=obligation.rationale,
            **_thresholds(obligation.intent, coverage),
        )
        report.generated_from_obligation.append(demand_id)
        if resolved.object_id:
            report.resolved_instances[object_id] = resolved.object_id
        if not targetable and obligation.search_proxy_ref:
            # Où chercher tant que la cible précise manque. Une vue obtenue par
            # ce détour demandera vérification : elle ne vaut pas preuve.
            report.search_proxies[demand_id] = obligation.search_proxy_ref

    manifest = CaptureDemandManifest(
        hotel_id=hotel_id,
        demands=[demands[key] for key in sorted(demands)],
        **{k: v for k, v in (digests or {}).items() if v is not None},
    )
    log.info(
        "besoins : %d généré(s), %d conservé(s), %d non ciblable(s), "
        "%d dispensé(s) — 0 octet téléchargé",
        len(report.generated_from_obligation), len(report.operator_defined),
        len(report.unresolved_target), len(report.waived) + len(report.not_applicable),
    )
    return manifest, report


def validate_targets(  # noqa: ANN001
    manifest: CaptureDemandManifest, site, geometry=None,
) -> list[str]:
    """Refuse les cibles inconnues ou ambiguës, avant toute écriture.

    Une demande qui ne vise rien ne se satisfait jamais : elle resterait
    ouverte indéfiniment, et le Router la compterait comme un manque réel.

    Le registre de corridors vient de la géométrie de capture. **Absent** et
    **vide** ne disent pas la même chose : le premier signifie qu'on ne peut
    pas valider, le second que toute référence de corridor est fausse. Passer
    `None` faute de géométrie serait donc plus honnête qu'un ensemble vide,
    mais la validation refuserait tout — on distingue donc explicitement.
    """
    from .schemas.acquisition import validate_targets as check

    # Les besoins visent un **type** ; le manifeste porte des instances
    # préfixées du site. Comparer les deux déclarait inconnus des objets
    # présents — le même défaut de jointure, une couche plus bas.
    object_ids = {
        kind
        for obj in getattr(site, "objects", []) or []
        if (kind := getattr(obj, "kind", None))
    }
    corridor_ids = None
    if geometry is not None:
        corridor_ids = {
            corridor.corridor_id for corridor in getattr(geometry, "corridors", [])
        } | {
            resolved.feature_id for resolved in getattr(geometry, "geometries", [])
        } | {
            # Une transition dont l'objet existe au site est une cible
            # légitime, même sans géométrie propre : c'est `demand_targets`
            # qui refusera de la mesurer, en le disant.
            kind
            for obj in getattr(site, "objects", []) or []
            if (kind := getattr(obj, "kind", None))
        }
    return check(manifest, object_ids, corridor_ids)
