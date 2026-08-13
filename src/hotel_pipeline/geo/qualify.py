"""Qualification des objets dérivés (Lot 1B §9).

Un GeoTIFF produit n'est pas une géométrie qualifiée. Cette étape confronte les
métriques d'une dérivation à des seuils versionnés, et fait passer un objet en
`inferred` — jamais en `confirmed`, puisqu'il procède d'une inférence.

Les deux objets ne se jugent pas de la même façon, et c'est le point central.

`TERRAIN_MAIN` est **interpolé** : pas une cellule de sol n'est mesurée sous
l'emprise. Ses seuils portent sur la fiabilité de l'interpolation, éprouvée là
où la vérité est connue.

`ROOFLINE_MAIN` est **observé** : 25 points par mètre carré. Ses seuils portent
sur la couverture et la densité de l'observation.

Les confondre reviendrait à traiter une mesure et une déduction comme la même
espèce de preuve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..logging import get_logger
from ..schemas import ObjectState

log = get_logger("qualify")

#: Rôles d'artefacts qui portent la substance de chaque objet. Un objet cite ce
#: dont il est fait, pas tout ce que la dérivation a produit : les masques de
#: diagnostic ne fondent aucune géométrie.
OBJECT_ROLES: dict[str, tuple[str, ...]] = {
    "TERRAIN_MAIN": ("dtm",),
    "ROOFLINE_MAIN": ("dsm_roof", "ndsm"),
}


def select_artifacts(site) -> dict[str, list[str]]:  # noqa: ANN001
    """Artefacts **actifs** portant chaque objet, par rôle.

    Seuls les artefacts actifs sont éligibles : un objet ne peut pas être
    qualifié sur une production déjà remplacée.
    """
    return {
        kind: sorted(
            a.artifact_id
            for a in site.artifacts
            if a.is_active and a.role in roles
        )
        for kind, roles in OBJECT_ROLES.items()
    }


def _run_suffix(artifact_id: str) -> str | None:
    """Identifiant d'exécution d'un artefact — `dtm@20260813T124251Z`."""
    return artifact_id.rsplit("@", 1)[1] if "@" in artifact_id else None


def check_series(site, derivation: dict, run_id: str | None = None) -> list[str]:  # noqa: ANN001
    """Vérifie que la série active est bien celle que le rapport décrit.

    Choisir le dernier rapport et sélectionner séparément les artefacts actifs
    laisse les deux dériver l'un de l'autre : une dérivation interrompue après
    l'écriture du rapport, une supersession partielle, une reprise manuelle, et
    la qualification jugerait des chiffres qui ne décrivent pas les fichiers
    cités. Rien ici ne le signalerait — les deux moitiés seraient cohérentes
    séparément.
    """
    problems: list[str] = []
    by_id = {a.artifact_id: a for a in site.artifacts}
    declared = {
        a["artifact_id"]: a for a in (derivation.get("artifacts") or []) if "artifact_id" in a
    }
    if not declared:
        problems.append(
            "le rapport jugé ne déclare aucun artefact — le lien avec la série "
            "active ne peut pas être établi"
        )

    for kind, roles in OBJECT_ROLES.items():
        for role in roles:
            actives = [a for a in site.artifacts if a.is_active and a.role == role]
            if len(actives) != 1:
                problems.append(
                    f"{kind} : {len(actives)} artefact(s) actif(s) de rôle {role!r} — "
                    "il en faut exactement un pour que la citation soit sans ambiguïté"
                )

    selected = sorted({a for ids in select_artifacts(site).values() for a in ids})
    suffixes = {_run_suffix(a) for a in selected}
    if len(suffixes) > 1:
        problems.append(
            "les artefacts actifs proviennent de plusieurs exécutions : "
            f"{sorted(s or '—' for s in suffixes)}"
        )
    if run_id and suffixes and suffixes != {run_id}:
        problems.append(
            f"le rapport jugé porte l'exécution {run_id}, les artefacts actifs "
            f"{sorted(s or '—' for s in suffixes)}"
        )

    for artifact_id in selected:
        artifact = by_id[artifact_id]
        recorded = declared.get(artifact_id)
        if recorded is None:
            problems.append(
                f"{artifact_id} ne figure pas dans le rapport jugé — la "
                "qualification porterait sur des mesures d'une autre production"
            )
            continue
        if recorded.get("sha256") != artifact.sha256:
            problems.append(
                f"{artifact_id} : empreintes divergentes entre le manifeste et le "
                f"rapport ({artifact.sha256[:12]}… ≠ {str(recorded.get('sha256'))[:12]}…)"
            )
        if Path(recorded.get("path", "")).name != Path(artifact.path).name:
            problems.append(
                f"{artifact_id} : chemins divergents entre le manifeste et le rapport"
            )

    return problems


@dataclass
class Criterion:
    """Un seuil, sa mesure et son verdict."""

    name: str
    threshold: str
    measured: str
    passed: bool

    def as_dict(self) -> dict:
        return {
            "criterion": self.name,
            "threshold": self.threshold,
            "measured": self.measured,
            "passed": self.passed,
        }


@dataclass
class ObjectVerdict:
    kind: str
    #: Ce que la qualification affirme, en une phrase. Sans elle, `inferred` ne
    #: dit pas de quoi l'objet est fait.
    rationale: str = ""
    criteria: list[Criterion] = field(default_factory=list)
    reservations: list[str] = field(default_factory=list)
    confidence: str = "unknown"

    @property
    def passed(self) -> bool:
        return bool(self.criteria) and all(c.passed for c in self.criteria)

    @property
    def failures(self) -> list[str]:
        return [c.name for c in self.criteria if not c.passed]

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "passed": self.passed,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "criteria": [c.as_dict() for c in self.criteria],
            "reservations": self.reservations,
            "failures": self.failures,
        }


@dataclass
class QualificationReport:
    evaluated_at: str = ""
    #: Empreinte du rapport de **dérivation** évalué. Une qualification qui ne
    #: dit pas ce qu'elle a jugé n'est pas rejouable.
    qualified_derivation_digest: str = ""
    #: Identifiant de l'exécution jugée, celui que portent les artefacts.
    run_id: str = ""
    selected_artifacts: list[str] = field(default_factory=list)
    policy_version: str = ""
    policy_digest: str = ""
    qualification_status: str = ""
    intended_use: str = ""
    calibration_id: str = ""
    calibrated_on_sites: int = 0
    verdicts: dict[str, ObjectVerdict] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Nom de publication : une décision par exécution **et** par politique.

        Écraser un rapport de qualification effacerait la décision précédente
        alors que la dérivation et les artefacts qui la fondaient, eux, sont
        conservés. On ne saurait plus ce qu'une politique antérieure avait
        conclu des mêmes mesures.
        """
        return f"qualification_report_{self.run_id}_{self.policy_digest}.json"

    def as_dict(self) -> dict:
        return {
            "evaluated_at": self.evaluated_at,
            "run_id": self.run_id,
            "qualified_derivation_digest": self.qualified_derivation_digest,
            "selected_artifacts": self.selected_artifacts,
            "policy": {
                "version": self.policy_version,
                "digest": self.policy_digest,
                "status": self.qualification_status,
                "intended_use": self.intended_use,
                "calibration_id": self.calibration_id,
                "calibrated_on_sites": self.calibrated_on_sites,
            },
            "verdicts": {k: v.as_dict() for k, v in self.verdicts.items()},
        }


def _criterion(name: str, threshold, measured, passed: bool) -> Criterion:  # noqa: ANN001
    return Criterion(name, str(threshold), str(measured), bool(passed))


# --- terrain ---------------------------------------------------------------


def evaluate_terrain(metrics: dict, limits) -> ObjectVerdict:  # noqa: ANN001
    """Confronte les métriques d'interpolation aux seuils du terrain."""
    verdict = ObjectVerdict(kind="TERRAIN_MAIN")
    pseudo = metrics["pseudo_footprint_validation"]
    trials = pseudo["trials"]

    # Le pire essai décide : une moyenne dissimulerait un essai médiocre
    # derrière deux bons.
    worst_rmse = max((t["rmse_m"] for t in trials if t["rmse_m"] is not None), default=None)
    worst_p95 = max((t["p95_m"] for t in trials if t["p95_m"] is not None), default=None)
    worst_bias = max(
        (abs(t["bias_m"]) for t in trials if t["bias_m"] is not None), default=None
    )

    verdict.criteria = [
        _criterion(
            "dtm_defined",
            f"≥ {limits.min_dtm_defined:.0%}",
            f"{metrics['coverage']['dtm_defined']:.1%}",
            metrics["coverage"]["dtm_defined"] >= limits.min_dtm_defined,
        ),
        _criterion(
            "search_area_within_tile",
            "obligatoire" if limits.require_search_area_within_tile else "facultatif",
            pseudo["search_area_within_tile"],
            pseudo["search_area_within_tile"] is True
            or not limits.require_search_area_within_tile,
        ),
        _criterion(
            "accepted_trials",
            f"≥ {limits.min_accepted_trials}",
            len(trials),
            len(trials) >= limits.min_accepted_trials,
        ),
        _criterion(
            "worst_trial_rmse_m",
            f"≤ {limits.max_worst_trial_rmse_m}",
            worst_rmse,
            worst_rmse is not None and worst_rmse <= limits.max_worst_trial_rmse_m,
        ),
        _criterion(
            "worst_trial_p95_m",
            f"≤ {limits.max_worst_trial_p95_m}",
            worst_p95,
            worst_p95 is not None and worst_p95 <= limits.max_worst_trial_p95_m,
        ),
        _criterion(
            "worst_abs_bias_m",
            f"≤ {limits.max_abs_bias_m}",
            worst_bias,
            worst_bias is not None and worst_bias <= limits.max_abs_bias_m,
        ),
        _criterion(
            "max_support_distance_m",
            f"≤ {limits.max_support_distance_m}",
            metrics["support_distance_in_footprint"]["max_m"],
            metrics["support_distance_in_footprint"]["max_m"]
            <= limits.max_support_distance_m,
        ),
        _criterion(
            "rejected_extrapolation",
            f"≤ {limits.max_rejected_extrapolation:.0%}",
            f"{metrics['extrapolation_rejected']['fraction_of_footprint']:.1%}",
            metrics["extrapolation_rejected"]["fraction_of_footprint"]
            <= limits.max_rejected_extrapolation,
        ),
        _criterion(
            "tin_idw_mae_m",
            f"≤ {limits.max_tin_idw_mae_m}",
            metrics["tin_vs_idw"]["mae_m"],
            metrics["tin_vs_idw"]["mae_m"] <= limits.max_tin_idw_mae_m,
        ),
    ]

    verdict.reservations = [
        f"{len(trials)} essais seulement — un compte suffisant à franchir le seuil, "
        "pas à caractériser une distribution",
        "aucune cellule de terrain mesurée sous l'emprise : la surface est "
        "entièrement inférée",
        "seuils éprouvés sur un seul site",
        "la validation par blocs reste diagnostique et n'a pas décidé du passage : "
        "elle est structurellement optimiste",
    ]
    if worst_rmse is not None and trials:
        best = min(t["rmse_m"] for t in trials if t["rmse_m"] is not None)
        if worst_rmse > 2 * best:
            verdict.reservations.append(
                f"dispersion des essais d'un facteur {worst_rmse / best:.1f} "
                f"({best:.3f} à {worst_rmse:.3f} m)"
            )
    verdict.confidence = "medium" if verdict.passed else "insufficient"
    verdict.rationale = (
        "surface de terrain interpolée depuis les appuis de sol du pourtour "
        f"(TIN), éprouvée sur {len(trials)} pseudo-empreintes : pire essai "
        f"RMSE {worst_rmse} m, p95 {worst_p95} m. Proxy visuel, non donnée "
        "d'arpentage."
    )
    return verdict


# --- toiture ---------------------------------------------------------------


#: Deux connexités, et le choix n'est pas cosmétique. Chacune est prise dans le
#: sens **défavorable** à l'objet jugé : la surface observée se compte en
#: 4-connexité (deux cellules en diagonale ne forment pas une surface continue),
#: les lacunes en 8-connexité (deux trous en diagonale forment bien un seul
#: trou pour une caméra). L'inverse flatterait les deux mesures à la fois.
_ORTHOGONAL = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
_DIAGONAL = np.ones((3, 3), dtype=bool)


def _components(mask: np.ndarray, structure: np.ndarray) -> tuple[int, int]:
    """(nombre de composantes, cellules de la plus grande)."""
    from scipy import ndimage

    labels, count = ndimage.label(mask, structure=structure)
    if count == 0:
        return 0, 0
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    return int(count), int(sizes.max())


def roof_gaps(observed_mask: np.ndarray, footprint_mask: np.ndarray, cell_m: float) -> dict:
    """Analyse l'observation de la toiture en composantes connexes.

    Une fraction observée globale ne dit pas ce qu'on peut filmer : 96,9 % de
    cellules vues peuvent former une seule surface continue, ou deux surfaces
    séparées par une lacune traversante. C'est la plus grande composante
    **observée** qui porte la silhouette, et la plus grande lacune contiguë qui
    borne ce qu'un plan rapproché peut montrer.
    """
    observed = np.asarray(observed_mask, dtype=bool) & np.asarray(footprint_mask, dtype=bool)
    missing = np.asarray(footprint_mask, dtype=bool) & ~observed
    domain = max(int(np.asarray(footprint_mask).sum()), 1)

    _, main = _components(observed, _ORTHOGONAL)
    gap_count, largest_gap = _components(missing, _DIAGONAL)
    return {
        "main_observed_cells": main,
        "main_observed_fraction": round(main / domain, 4),
        "main_observed_connectivity": 4,
        "missing_cells": int(missing.sum()),
        "missing_components": gap_count,
        "missing_connectivity": 8,
        "largest_gap_cells": largest_gap,
        # Aire **de grille** : somme des cellules entières, sans découpe par le
        # polygone d'emprise. Une cellule de bordure n'appartient au bâtiment
        # que pour partie ; l'appeler « aire » tout court surestimerait la
        # lacune. La surestimation est du bon côté, elle doit rester nommée.
        "largest_gap_grid_m2": round(largest_gap * cell_m**2, 1),
        "largest_gap_fraction": round(largest_gap / domain, 4),
    }


def evaluate_roofline(
    metrics: dict, limits, terrain_passed: bool, gaps: dict | None = None  # noqa: ANN001
) -> ObjectVerdict:
    """Confronte les métriques d'observation aux seuils de la toiture."""
    verdict = ObjectVerdict(kind="ROOFLINE_MAIN")
    observed = metrics["coverage"]["roof_observed"]
    gaps = gaps if gaps is not None else metrics.get("roof_gaps") or {}
    main_component = gaps.get("main_observed_fraction")
    heights = metrics["height_statistics"]
    negative_fraction = heights["negative_cells"] / max(heights["count"], 1)

    verdict.criteria = [
        _criterion(
            "roof_observed",
            f"≥ {limits.min_roof_observed:.0%}",
            f"{observed:.1%}",
            observed >= limits.min_roof_observed,
        ),
        _criterion(
            "main_observed_component",
            f"≥ {limits.min_main_component:.0%}",
            # Une mesure absente ne vaut pas une mesure réussie : sans analyse
            # de composantes, le critère échoue.
            f"{main_component:.1%}" if main_component is not None else "non mesuré",
            main_component is not None and main_component >= limits.min_main_component,
        ),
        _criterion(
            "class6_density_per_m2",
            f"≥ {limits.min_point_density_per_m2}",
            metrics.get("roof_density_per_m2"),
            (metrics.get("roof_density_per_m2") or 0) >= limits.min_point_density_per_m2,
        ),
        _criterion(
            "ndsm_valid",
            f"≥ {limits.min_ndsm_valid:.0%}",
            f"{metrics['coverage']['ndsm_valid']:.1%}",
            metrics["coverage"]["ndsm_valid"] >= limits.min_ndsm_valid,
        ),
        _criterion(
            "negative_height_fraction",
            f"≤ {limits.max_negative_height_fraction:.1%}",
            f"{negative_fraction:.2%}",
            negative_fraction <= limits.max_negative_height_fraction,
        ),
        _criterion(
            "qualified_terrain",
            "obligatoire" if limits.require_qualified_terrain else "facultatif",
            terrain_passed,
            terrain_passed or not limits.require_qualified_terrain,
        ),
    ]

    verdict.reservations = [
        "« surface et silhouette principales », non « tous les équipements de "
        "toiture »",
    ]
    if gaps.get("missing_components"):
        verdict.reservations.append(
            f"{gaps['missing_cells']} cellule(s) non observée(s) en "
            f"{gaps['missing_components']} composante(s) ; la plus grande fait "
            f"{gaps.get('largest_gap_grid_m2', gaps.get('largest_gap_m2'))} m² "
            f"d'aire de grille ({gaps['largest_gap_fraction']:.1%} de l'emprise, "
            "cellules entières non découpées par le polygone) — zones de faible "
            "confiance, interdites aux plans rapprochés"
        )
    if metrics["coverage"].get("class1_candidates"):
        verdict.reservations.append(
            f"{metrics['coverage']['class1_candidates']:.1%} de candidats classe 1 "
            "non fusionnés — avertissement de détail, non motif de rejet"
        )
    verdict.confidence = "medium" if verdict.passed else "insufficient"
    verdict.rationale = (
        "surface principale de toiture observée par LiDAR aérien "
        f"({metrics.get('roof_density_per_m2')} pt/m² de classe 6), "
        f"{observed:.1%} de l'emprise vue"
    )
    if main_component is not None:
        verdict.rationale += (
            f", dont {main_component:.1%} en une seule composante continue"
        )
    return verdict


# --- péremption ------------------------------------------------------------


def mark_stale(site) -> list[str]:  # noqa: ANN001
    """Repasse en `stale` les objets citant un artefact non actif.

    La décision antérieure et son motif sont conservés : la qualification n'est
    pas devenue fausse, elle a perdu son support. Elle est retrouvable telle
    quelle si la nouvelle dérivation la reconduit.
    """
    by_artifact = {a.artifact_id: a for a in site.artifacts}
    marked: list[str] = []

    for index, obj in enumerate(site.objects):
        if not obj.artifact_ids or obj.state is ObjectState.STALE:
            continue
        lost = [
            a for a in obj.artifact_ids if a in by_artifact and not by_artifact[a].is_active
        ]
        if not lost:
            continue

        site.objects[index] = obj.model_copy(
            update={
                "state": ObjectState.STALE,
                "previous_state": obj.state,
                "qualification_rationale": obj.qualification_rationale,
                "unresolved_reason": (
                    f"artefact(s) cité(s) remplacé(s) depuis : {sorted(lost)} ; "
                    f"décision « {obj.state.value} » et son motif conservés en "
                    "l'état, à reconduire sur la dérivation courante"
                ),
            }
        )
        marked.append(obj.object_id)

    if marked:
        log.info("%d objet(s) repassé(s) en 'stale' : %s", len(marked), sorted(marked))
    return marked


def apply(  # noqa: A001
    site, report: QualificationReport, mapping: dict[str, list[str]],  # noqa: ANN001
    report_digest: str = "",
) -> list[str]:
    """Inscrit les verdicts au manifeste. N'écrit **jamais** `confirmed`.

    Un objet dérivé d'une interpolation ou d'une observation aérienne est
    inféré : `confirmed` supposerait une vérification indépendante de la
    dérivation elle-même, qui n'a pas eu lieu.
    """
    qualified: list[str] = []

    for index, obj in enumerate(site.objects):
        verdict = report.verdicts.get(obj.kind)
        if verdict is None:
            continue

        artifacts = mapping.get(obj.kind, [])
        # Une géométrie sans artefact actif n'a pas de support : le verdict
        # porterait sur des mesures que le manifeste ne peut plus montrer.
        if verdict.passed and artifacts:
            update = {
                "state": ObjectState.INFERRED,
                "previous_state": None,
                "artifact_ids": artifacts,
                "qualification_report": report.name,
                "qualification_report_digest": report_digest or None,
                "qualified_derivation_digest": report.qualified_derivation_digest,
                "qualification_rationale": verdict.rationale,
                "qualification_confidence": verdict.confidence,
                "qualification_reservations": verdict.reservations,
                "unresolved_reason": None,
            }
            qualified.append(obj.object_id)
        else:
            update = {
                "state": ObjectState.UNRESOLVED,
                "previous_state": None,
                # Un objet non qualifié ne garde pas les artefacts d'une
                # dérivation qui n'a pas passé ses propres seuils.
                "artifact_ids": [],
                "geometry_wkt": None,
                "qualification_report": report.name,
                "qualification_report_digest": report_digest or None,
                "qualified_derivation_digest": report.qualified_derivation_digest,
                "qualification_rationale": None,
                "qualification_confidence": verdict.confidence,
                "qualification_reservations": verdict.reservations,
                "unresolved_reason": (
                    "seuil(s) de qualification non franchi(s) : "
                    + ", ".join(verdict.failures)
                    if verdict.failures
                    else "aucun artefact actif ne porte cet objet"
                ),
            }
        site.objects[index] = obj.model_copy(update=update)

    return qualified


def report(  # noqa: A001
    metrics: dict, policy, digest: str, artifacts: list[str],  # noqa: ANN001
    run_id: str = "",
) -> QualificationReport:
    """Évalue les deux objets et assemble le rapport."""
    from ..provenance import policy_digest

    terrain = evaluate_terrain(metrics, policy.qualification.terrain)
    roofline = evaluate_roofline(metrics, policy.qualification.roofline, terrain.passed)
    return QualificationReport(
        evaluated_at=datetime.now(timezone.utc).isoformat(),
        qualified_derivation_digest=digest,
        run_id=run_id,
        selected_artifacts=artifacts,
        policy_version=policy.version,
        policy_digest=policy_digest(policy),
        qualification_status=policy.qualification.status,
        intended_use=policy.qualification.intended_use,
        calibration_id=policy.qualification.calibration_id,
        calibrated_on_sites=policy.qualification.calibrated_on_sites,
        verdicts={"TERRAIN_MAIN": terrain, "ROOFLINE_MAIN": roofline},
    )
