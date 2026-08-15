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
from .schemas.acquisition import CaptureDemand, CaptureDemandManifest, CaptureIntent
from .schemas.enums import ObjectState

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

    def as_dict(self) -> dict:
        return {
            "generated_from_obligation": sorted(self.generated_from_obligation),
            "operator_defined": sorted(self.operator_defined),
            "waived": self.waived,
            "not_applicable": self.not_applicable,
            "unresolved_target": self.unresolved_target,
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
    from .coverage_obligations import OBLIGATIONS, ObligationStatus

    if site is None:
        raise DemandsRefused(
            "aucun manifeste de site : les objets à couvrir ne sont pas nommés. "
            "Lancez « site build » d'abord."
        )

    dispensed = {waiver.object_id: waiver for waiver in (waivers or [])}
    states = {obj.object_id: obj.state for obj in getattr(site, "objects", [])}

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

        state = states.get(object_id)
        if state is ObjectState.UNRESOLVED:
            # Le besoin existe et n'est pas ciblable : ni satisfait, ni
            # dispensé. Le convertir en dispense ferait disparaître le manque.
            report.unresolved_target[object_id] = (
                "objet non résolu au manifeste de site : le besoin est réel, "
                "mais rien ne permet encore de le viser"
            )
            continue
        if state is None and obligation.target_kind.value == "site_object":
            report.unresolved_target[object_id] = (
                "objet absent du manifeste de site : la cible n'existe pas"
            )
            continue

        if not obligation.mandatory and demand_id not in demands:
            # Facultative et non demandée : rien n'est dû.
            continue

        demands[demand_id] = CaptureDemand(
            demand_id=demand_id,
            intent=obligation.intent,
            target_kind=obligation.target_kind,
            target_ref=obligation.expected_target_ref,
            rationale=obligation.rationale,
            **_thresholds(obligation.intent, coverage),
        )
        report.generated_from_obligation.append(demand_id)

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

    object_ids = {obj.object_id for obj in getattr(site, "objects", [])}
    corridor_ids = None
    if geometry is not None:
        corridor_ids = {
            corridor.corridor_id for corridor in getattr(geometry, "corridors", [])
        } | {
            resolved.feature_id for resolved in getattr(geometry, "geometries", [])
        }
    return check(manifest, object_ids, corridor_ids)
