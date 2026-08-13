"""Projection d'une exécution de visibilité vers les assets (Lot 1B V2).

Tout est vérifié avant la moindre mutation, et le manifeste entier est
construit en mémoire : une application à moitié faite serait pire qu'un échec,
puisqu'elle paraîtrait complète.

Ce que la projection écrit décrit la **géométrie**, jamais le contenu. Aucun
de ces champs ne dit que la caméra vise le bâtiment ni qu'il entre dans
l'image : `sees_building`, `target_building_visible`, les historiques humains,
l'aptitude géométrique et les scores du modèle sont laissés intacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..logging import get_logger
from ..schemas.visibility import LineOfSightStatus, RayPartition, VisibilityRun

log = get_logger("visibility-apply")


class ApplicationRefused(RuntimeError):
    """Rien n'a été modifié."""


@dataclass
class ApplicationReport:
    run_id: str = ""
    run_digest: str = ""
    status: str = "applied"
    assets_updated: int = 0
    manifest_digest_before: str = ""
    manifest_digest_after: str = ""
    base_digest: str = ""
    roles_before: dict[str, int] = field(default_factory=dict)
    roles_after: dict[str, int] = field(default_factory=dict)
    former_occlusions: list[dict] = field(default_factory=list)
    fields_written: list[str] = field(default_factory=list)
    occluded_by_kept: list[str] = field(default_factory=list)

    @property
    def note(self) -> str | None:
        """Ce qu'un reçu reconstruit ne peut plus dire.

        Rejoué après coup, il lit un manifeste déjà projeté : l'état antérieur
        n'y est plus observable, et zéro occultation retirée ne signifie pas
        qu'il n'y en avait aucune.
        """
        if self.status != "already_applied":
            return None
        return (
            "reçu reconstruit après application : l'état antérieur n'était plus "
            "observable, les comptes d'occultations retirées ne sont donc pas "
            "ceux de l'application d'origine"
        )

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "note": self.note,
            "run_id": self.run_id,
            "run_digest": self.run_digest,
            "assets_updated": self.assets_updated,
            "manifest_digest": {
                "before": self.manifest_digest_before,
                "after": self.manifest_digest_after,
                "base": self.base_digest,
            },
            "roles": {"before": self.roles_before, "after": self.roles_after},
            # Le cœur de l'affaire : ce que deviennent les 29 affirmations
            # d'occultation que l'ancien annotateur avait posées.
            "former_occlusions": self.former_occlusions,
            "occluded_by_kept": self.occluded_by_kept,
            "fields_written": self.fields_written,
        }


PROJECTED = [
    "visibility_run_id",
    "visibility_run_digest",
    "visibility_assessment_id",
    "line_of_sight_status",
    "occlusion_risk_by",
    "occlusion_blocked_by",
    "occluded_by",
]


def sole_blocker(assessment) -> str | None:  # noqa: ANN001
    """Responsable unique d'un blocage **intégral**, s'il y en a un.

    Trois conditions : toute la silhouette bloquée, et un seul obstacle
    responsable sur chacune de ses cellules. Un blocage partiel, ou partagé,
    reste dans l'évaluation détaillée — le champ singulier ne saurait pas le
    dire.
    """
    if assessment.proven_blocked_fraction < 1.0:
        return None
    blockers: set[str] = set()
    for ray in assessment.rays:
        if ray.partition is not RayPartition.BLOCKED_2_5D:
            return None
        blockers.update(ray.blocking)
    return next(iter(blockers)) if len(blockers) == 1 else None


def projection_for(assessment) -> dict:  # noqa: ANN001
    """Champs qu'une évaluation pose sur son asset."""
    blocked = sorted({name for ray in assessment.rays for name in ray.blocking})
    return {
        "visibility_assessment_id": assessment.assessment_id,
        "line_of_sight_status": assessment.status.value,
        "occlusion_risk_by": list(assessment.obstacles_at_risk),
        "occlusion_blocked_by": blocked,
        # Compatibilité : le champ singulier ne se renseigne que sur un blocage
        # intégral à responsable unique. Les 29 anciennes valeurs, posées sur
        # une intersection en plan sans aucune donnée verticale, retombent donc
        # à `None`.
        "occluded_by": sole_blocker(assessment),
    }


def verify(
    run: VisibilityRun,
    manifest,  # noqa: ANN001 — AssetManifest
    hotel_id: str,
    current_digests: dict[str, str],
) -> list[str]:
    """Tous les contrôles préalables, sans en écrire un seul résultat."""
    from .visibility_run import base_manifest_digest

    problems: list[str] = []

    if run.hotel_id != hotel_id:
        problems.append(
            f"exécution de {run.hotel_id!r} appliquée à {hotel_id!r}"
        )

    expected = {
        "policy_digest": current_digests.get("policy"),
        "capture_geometry_digest": current_digests.get("capture_geometry"),
        "site_manifest_digest": current_digests.get("site_manifest"),
        "asset_files_digest": current_digests.get("asset_files"),
        "obstacles_digest": current_digests.get("obstacles"),
        "road_geometry_digest": current_digests.get("roads"),
    }
    for name, current in expected.items():
        recorded = getattr(run, name, None)
        if current is None:
            problems.append(f"{name} : état courant inconnu")
        elif recorded != current:
            problems.append(
                f"{name} : exécution {recorded[:12]}… ≠ courant {current[:12]}…"
            )

    base = base_manifest_digest(manifest)
    if run.asset_manifest_digest != base:
        problems.append(
            f"le manifeste a changé depuis la mesure ({run.asset_manifest_digest[:12]}… "
            f"≠ {base[:12]}…) — une revue, un cap ou une position ont bougé"
        )

    # Exactement une évaluation par asset situé, et le même ensemble en
    # cadrages : une mesure manquante laisserait un asset sans verdict, une
    # mesure surnuméraire décrirait un asset qui n'existe plus.
    located = {a.id for a in manifest.assets if a.camera_lat is not None and a.camera_lon is not None}
    assessed = {a.subject_ref for a in run.assessments}
    framed = {f.subject_ref for f in run.framings}

    if assessed != located:
        missing = sorted(located - assessed)[:5]
        extra = sorted(assessed - located)[:5]
        problems.append(
            f"évaluations et assets situés diffèrent — manquants {missing}, "
            f"surnuméraires {extra}"
        )
    if framed != assessed:
        problems.append("cadrages et évaluations ne portent pas sur le même ensemble")

    if not run.elevation_sources:
        decided = any(
            ray.vertical_status.value == "fully_known"
            for assessment in run.assessments
            for ray in assessment.rays
        )
        if decided:
            problems.append("verdicts verticaux sans source d'élévation citée")

    return problems


def already_applied(manifest, run: VisibilityRun) -> tuple[bool, list[str]]:  # noqa: ANN001
    """Ce run est-il déjà posé, à l'identique ?

    Rejouer une commande interrompue doit reconstruire le reçu manquant sans
    remuter quoi que ce soit ; mais un champ modifié à la main depuis n'est pas
    une application idempotente, et se déclare tel.
    """
    by_id = {a.id: a for a in manifest.assets}
    touched = [a for a in manifest.assets if a.visibility_run_id]
    if not touched:
        return False, []

    foreign = sorted({a.visibility_run_id for a in touched} - {run.run_id})
    if foreign:
        return False, [f"exécution(s) déjà appliquée(s) : {foreign}"]

    divergent: list[str] = []
    for assessment in run.assessments:
        asset = by_id.get(assessment.subject_ref)
        if asset is None:
            continue
        expected = projection_for(assessment)
        for name, value in expected.items():
            if getattr(asset, name) != value:
                divergent.append(f"{asset.id}.{name}")
    return not divergent, divergent


def project(manifest, run: VisibilityRun, run_digest: str, policy) -> ApplicationReport:  # noqa: ANN001
    """Construit le manifeste projeté **en mémoire**, rôles vérifiés.

    Aucun rôle ne doit changer : la visibilité géométrique ne promeut rien, et
    un rôle qui bougerait signalerait qu'un champ interdit a été touché.
    """
    from ..roles import role_for
    from ..schemas import Asset, AssetManifest
    from .visibility_run import base_manifest_digest, digest

    report = ApplicationReport(run_id=run.run_id, run_digest=run_digest)
    report.manifest_digest_before = digest(manifest.model_dump(mode="json"))
    report.base_digest = base_manifest_digest(manifest)
    report.fields_written = list(PROJECTED)

    by_id = {a.subject_ref: a for a in run.assessments}
    updated: list[Asset] = []
    roles_before: dict[str, int] = {}
    roles_after: dict[str, int] = {}

    for asset in manifest.assets:
        role_before, _ = role_for(asset, policy)
        roles_before[role_before.value] = roles_before.get(role_before.value, 0) + 1

        assessment = by_id.get(asset.id)
        if assessment is None:
            updated.append(asset)
            roles_after[role_before.value] = roles_after.get(role_before.value, 0) + 1
            continue

        projection = projection_for(assessment)
        if asset.occluded_by and not projection["occluded_by"]:
            report.former_occlusions.append(
                {
                    "asset_id": asset.id,
                    "was_occluded_by": asset.occluded_by,
                    "now": assessment.status.value,
                    "proven_blocked_fraction": assessment.proven_blocked_fraction,
                    "risk_fraction": assessment.risk_unknown_height_fraction,
                    "obstacles_at_risk": assessment.obstacles_at_risk,
                }
            )
        if projection["occluded_by"]:
            report.occluded_by_kept.append(asset.id)

        candidate = Asset.model_validate(
            asset.model_copy(
                update={
                    **projection,
                    "visibility_run_id": run.run_id,
                    "visibility_run_digest": run_digest,
                }
            ).model_dump()
        )
        role_after, _ = role_for(candidate, policy)
        if role_after is not role_before:
            raise ApplicationRefused(
                f"{asset.id} : le rôle passerait de {role_before.value!r} à "
                f"{role_after.value!r} — la visibilité géométrique ne promeut rien"
            )
        roles_after[role_after.value] = roles_after.get(role_after.value, 0) + 1
        updated.append(candidate)
        report.assets_updated += 1

    projected = AssetManifest.model_validate(
        manifest.model_copy(update={"assets": updated}).model_dump()
    )
    report.roles_before = dict(sorted(roles_before.items()))
    report.roles_after = dict(sorted(roles_after.items()))
    report.manifest_digest_after = digest(projected.model_dump(mode="json"))

    if base_manifest_digest(projected) != report.base_digest:
        raise ApplicationRefused(
            "la projection a modifié un champ hors de son périmètre"
        )
    return report, projected


def receipt_name(run_id: str, run_digest: str) -> str:
    """Nom déterministe : rejouer une commande retrouve son reçu."""
    return f"visibility_application_{run_id}_{run_digest}.json"
