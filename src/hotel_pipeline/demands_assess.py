"""Évaluation des besoins sur un corpus existant (collecte V2).

L'étape manquante entre les besoins et la recherche adaptative. Sans elle,
Mapillary ne sait pas quels secteurs sont déficitaires — et commencer par sa
stratégie obligerait à recréer les objectifs de couverture dans le collecteur,
donc deux sources d'autorité.

Ce module ne collecte rien. Il regarde ce qu'on possède déjà et répond, besoin
par besoin : combien de points de vue indépendants le servent, lesquels, et ce
qui manque encore.

Une vue ne compte que si elle est **toutes** ces choses à la fois :

```text
active après déduplication   un doublon n'est pas une observation de plus
attribuée à la cible         la bonne cible, pas le bâtiment par défaut
porteuse de géométrie        `photo_geometry`, non un verrou de contexte
apte                         `primary` ou `auxiliary`
compatible du secteur        vue du bon côté
sur un point de vue distinct neuf fichiers à six positions font six vues
```

Deux précautions que le vocabulaire seul ne donne pas : une cible non résolue
n'est pas un besoin inatteignable — l'un dit qu'on ne sait pas viser, l'autre
qu'aucune vue n'existera jamais — et une continuité non mesurable reste `None`,
jamais zéro ni « insuffisante ».
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from .logging import get_logger
from .schemas.acquisition import (
    DemandAssessment,
    DemandAssessmentManifest,
    DemandStatus,
)
from .schemas.enums import ClusterRole, GeometrySuitability, ReconstructionRole

log = get_logger("demands-assess")


@dataclass
class AssessReport:
    """Ce que le corpus sert, et ce qu'il laisse ouvert."""

    corpus_digest: str = ""
    assets_considered: int = 0
    by_status: dict[str, list[str]] = field(default_factory=dict)
    viewpoints_by_demand: dict[str, list[str]] = field(default_factory=dict)
    unresolved_targets: dict[str, str] = field(default_factory=dict)

    #: Assets qui servent un besoin **parce qu'un aperçu l'a établi**, non
    #: parce que leur secteur le laissait supposer. La distinction compte : le
    #: premier est une mesure, le second une inférence.
    established_by_preview: dict[str, list[str]] = field(default_factory=dict)

    @property
    def open_demands(self) -> list[str]:
        return sorted(
            self.by_status.get(DemandStatus.OPEN.value, [])
            + self.by_status.get(DemandStatus.PARTIALLY_MET.value, [])
        )

    def as_dict(self) -> dict:
        return {
            "corpus_digest": self.corpus_digest,
            "assets_considered": self.assets_considered,
            "by_status": {k: sorted(v) for k, v in sorted(self.by_status.items())},
            "viewpoints_by_demand": self.viewpoints_by_demand,
            "unresolved_targets": self.unresolved_targets,
            "established_by_preview": self.established_by_preview,
            "open_demands": self.open_demands,
            "bytes_downloaded": 0,
            "note": (
                "une cible non résolue n'est pas un besoin inatteignable : "
                "l'un dit qu'on ne sait pas viser, l'autre qu'aucune vue "
                "n'existera jamais"
            ),
        }


def counts_towards(asset) -> tuple[bool, str]:  # noqa: ANN001
    """Cet asset est-il une observation utilisable ? Sinon, pourquoi ?

    Six conditions, et non une : un fichier peut être présent, net et bien
    situé sans rien apporter à un besoin — parce qu'il double une vue, parce
    qu'il ne porte pas de géométrie, ou parce qu'il montre autre chose.
    """
    if asset.cluster_role not in (ClusterRole.CANONICAL, ClusterRole.OVERLAP):
        return False, "inactif après déduplication"
    if asset.reconstruction_role is not ReconstructionRole.PHOTO_GEOMETRY:
        return False, f"rôle {asset.reconstruction_role.value}, non porteur de géométrie"
    if asset.geometry_suitability not in (
        GeometrySuitability.PRIMARY, GeometrySuitability.AUXILIARY
    ):
        return False, f"aptitude {asset.geometry_suitability.value}"
    if asset.target_building_visible is not True:
        return False, "cible non établie sur cette vue"
    return True, ""


def assess(
    hotel_id: str,
    demands: list,
    assets: list,
    corpus_digest: str,
    viewpoints: dict[str, str] | None = None,
    sector_of: dict[str, str] | None = None,
    unresolved_targets: dict[str, str] | None = None,
    demand_digest: str | None = None,
    previews=None,  # noqa: ANN001 — PreviewAssessmentLog
) -> tuple[DemandAssessmentManifest, AssessReport]:
    """Confronte chaque besoin au corpus, sans rien collecter.

    `viewpoints` associe un identifiant d'asset à son point de vue : c'est lui
    qui fait qu'un besoin de deux observations n'est pas servi par deux
    cadrages d'une même position.
    """
    grouping = viewpoints or {}
    sectors = sector_of or {}
    unresolved = unresolved_targets or {}

    report = AssessReport(corpus_digest=corpus_digest, assets_considered=len(assets))
    assessments: list[DemandAssessment] = []

    usable = []
    for asset in assets:
        ok, _ = counts_towards(asset)
        if ok:
            usable.append(asset)

    for demand in demands:
        if demand.demand_id in unresolved:
            # Non ciblable : le besoin est réel, on ne sait pas le viser. Ce
            # n'est ni « ouvert » — on ne peut pas chercher — ni
            # « inatteignable », qui affirmerait qu'aucune vue n'existera.
            report.unresolved_targets[demand.demand_id] = unresolved[demand.demand_id]
            assessments.append(
                DemandAssessment(
                    demand_id=demand.demand_id, corpus_digest=corpus_digest,
                    status=DemandStatus.OPEN, viewpoints_found=0,
                    rationale=f"cible non résolue : {unresolved[demand.demand_id]}",
                )
            )
            report.by_status.setdefault(DemandStatus.OPEN.value, []).append(
                demand.demand_id
            )
            continue

        # Un aperçu **établit** ce que les champs plats de l'asset ne disent
        # pas : la mesure vit dans le constat, pas dans l'inventaire. Sans ce
        # raccord, une preview téléchargée puis examinée ne changeait rien à
        # l'évaluation du besoin qui l'avait motivée.
        established = (
            previews.established_for(demand.demand_id) if previews else set()
        )
        serving = [
            asset for asset in usable
            if asset.id in established or _serves(asset, demand, sectors)
        ]
        if established:
            report.established_by_preview[demand.demand_id] = sorted(
                {asset.id for asset in serving} & established
            )
        found_viewpoints = sorted({
            grouping.get(asset.id, f"asset:{asset.id}") for asset in serving
        })

        assessment = DemandAssessment(
            demand_id=demand.demand_id,
            corpus_digest=corpus_digest,
            viewpoints_found=len(found_viewpoints),
            # Aucune continuité n'est mesurée sur un corpus existant : le
            # recouvrement se mesure sur les images, et rien ici ne les ouvre.
            # `None` dit « non mesurée » ; zéro dirait « mesurée, et nulle ».
            continuity_achieved=None,
            continuity_level="planned" if demand.continuity_required > 0 else None,
            best_projected_width_fraction=_best(serving, "target_in_frame_fraction"),
            status=DemandStatus.OPEN,
        )
        assessment = assessment.model_copy(
            update={"status": _status_of(assessment, demand),
                    "rationale": _rationale(assessment, demand, found_viewpoints)}
        )

        assessments.append(assessment)
        report.viewpoints_by_demand[demand.demand_id] = found_viewpoints
        report.by_status.setdefault(assessment.status.value, []).append(
            demand.demand_id
        )

    manifest = DemandAssessmentManifest(
        hotel_id=hotel_id,
        corpus_digest=corpus_digest,
        # Obligatoire au schéma, et à raison : un rapport d'évaluation qui ne
        # dit pas **quels besoins** il a jugés ne se rattache à rien.
        demand_digest=demand_digest or "non-déclaré",
        assessments=assessments,
    )

    log.info(
        "besoins évalués sur %d asset(s) : %s — 0 octet téléchargé",
        len(assets),
        ", ".join(f"{k}={len(v)}" for k, v in sorted(report.by_status.items())),
    )
    return manifest, report


def _serves(asset, demand, sectors: dict[str, str]) -> bool:  # noqa: ANN001
    """Cette vue sert-elle **ce** besoin ?

    Le secteur se lit sur l'asset : une vue du coin avant-droit ne fait pas
    croire que l'arrière possède une ancre.
    """
    from .schemas.acquisition import TargetKind

    if demand.target_kind is TargetKind.VIEW_SECTOR:
        observed = sectors.get(asset.id) or asset.view_sector.value
        return observed == demand.target_ref
    # Pour un objet ou un corridor, l'attribution vient de l'évaluation
    # géométrique, qui n'existe pas sur le corpus historique : on ne l'invente
    # pas, et le besoin reste ouvert.
    return False


def _best(assets: list, attribute: str) -> float | None:  # noqa: ANN001
    values = [
        getattr(asset, attribute) for asset in assets
        if getattr(asset, attribute, None) is not None
    ]
    return max(values) if values else None


def _status_of(assessment: DemandAssessment, demand) -> DemandStatus:  # noqa: ANN001
    """`met` exige tout ; le reste se distingue par ce qui manque."""
    if assessment.meets(demand):
        return DemandStatus.MET
    if assessment.viewpoints_found > 0:
        return DemandStatus.PARTIALLY_MET
    return DemandStatus.OPEN


def _rationale(assessment: DemandAssessment, demand, viewpoints: list[str]) -> str:  # noqa: ANN001
    missing = []
    if assessment.viewpoints_found < demand.viewpoints_required:
        missing.append(
            f"{assessment.viewpoints_found}/{demand.viewpoints_required} "
            "point(s) de vue"
        )
    if demand.continuity_required > 0 and assessment.continuity_achieved is None:
        missing.append("continuité non mesurée sur ce corpus")
    if not missing:
        return f"servi par {len(viewpoints)} point(s) de vue indépendant(s)"
    return "manque : " + " ; ".join(missing)
