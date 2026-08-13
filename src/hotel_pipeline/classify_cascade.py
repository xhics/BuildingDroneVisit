"""Cascade de catégorisation (Lot 1B §6, §13 étape 3).

Le classifieur ne décide plus seul. Il intervient en dernier, après ce qui est
certain, et il a le droit de ne pas conclure.

1. métadonnées déterministes de la source
2. position, cap et visibilité géométrique
3. OCR et nom de propriété
4. classifieur multi-étiquette, avec seuils
5. revue humaine pour les cas décisifs et ambigus

Deux règles portent l'essentiel de la valeur : **appartenance et catégorie
sont deux décisions indépendantes**, et **une confiance insuffisante produit
`unknown`, jamais un choix par défaut**.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .logging import get_logger
from .schemas import (
    Asset,
    CaptureType,
    PropertyMatchStatus,
    ReviewDecision,
    ReviewStatus,
    Subject,
    TemporalStatus,
    ViewSector,
)
from .schemas.policy import DEFAULT_POLICY, PipelinePolicy
from .sectors import sector_for

log = get_logger("cascade")

#: Sujets dont une erreur est coûteuse : ils décident de la couverture par
#: secteur, donc de ce qu'une caméra future aura le droit de montrer.
DECISIVE_SUBJECTS = frozenset({Subject.BUILDING, Subject.ENTRANCE, Subject.ROOF})

#: Correspondance stricte entre la décision humaine et le statut de revue.
#: Les deux champs ne peuvent pas diverger : `confirmed` ne peut pas coexister
#: avec `needs_review`, ni `rejected` avec `automatic_accepted`.
_HUMAN_STATUS: dict[ReviewDecision, ReviewStatus] = {
    ReviewDecision.CONFIRMED: ReviewStatus.HUMAN_ACCEPTED,
    ReviewDecision.REJECTED: ReviewStatus.REJECTED,
    # Examiné sans conclure : c'est une information, pas une absence de revue.
    # L'acceptation automatique ne peut pas la recouvrir.
    ReviewDecision.UNRESOLVED: ReviewStatus.NEEDS_REVIEW,
}


@dataclass
class CascadeReport:
    total: int = 0
    by_stage: dict[str, int] = field(default_factory=dict)
    subjects_assigned: dict[str, int] = field(default_factory=dict)
    sectors_assigned: dict[str, int] = field(default_factory=dict)
    needs_review: int = 0
    occlusion_conflicts: int = 0
    unknown_sector: int = 0

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "decided_by_stage": self.by_stage,
            "subjects": self.subjects_assigned,
            "sectors": self.sectors_assigned,
            "needs_review": self.needs_review,
            "occlusion_conflicts": self.occlusion_conflicts,
            "unknown_sector": self.unknown_sector,
        }


def _stage_source(asset: Asset) -> tuple[list[Subject], str | None]:
    """Étape 1 — ce que la source établit sans ambiguïté.

    L'imagerie de roulage montre nécessairement la voie depuis laquelle elle
    est prise. C'est peu, mais c'est certain, et cela n'exige aucun modèle.
    """
    if asset.capture_type is CaptureType.STREET_IMAGERY:
        return [Subject.ROAD], "source:street_imagery"
    return [], None


def _stage_geometry(asset: Asset, front_azimuth: float | None) -> tuple[list[Subject], ViewSector, str | None]:
    """Étape 2 — ce que la géométrie établit.

    `sees_building` reste déterminé par la position et le cap lorsqu'ils
    existent : aucune probabilité ne doit contredire une mesure.
    """
    subjects: list[Subject] = []
    sector = ViewSector.UNKNOWN
    method = None

    # La géométrie n'établit le contenu que si le cap est observé. Avec un cap
    # que nous avons nous-mêmes dirigé vers l'empreinte — Street View —, elle
    # n'établit que la direction de visée : le contenu revient au modèle.
    if asset.sees_building and asset.heading_is_measured:
        subjects.append(Subject.BUILDING)
        method = "geometry:fov"
    elif asset.sees_building:
        method = "geometry:aim_only"

    if asset.bearing_from_building_deg is not None and front_azimuth is not None:
        sector = sector_for(asset.bearing_from_building_deg, front_azimuth)
        method = "geometry:sector" if method is None else f"{method}+sector"

    return subjects, sector, method


def _stage_ocr(asset: Asset) -> tuple[list[Subject], str | None]:
    """Étape 3 — ce que le texte lu établit.

    L'appartenance est traitée ailleurs : lire l'enseigne prouve qu'une
    enseigne est visible, pas que le cliché est celui du bon établissement.
    Ces deux conclusions ne doivent pas être confondues.
    """
    if asset.sign_text and asset.sign_text.strip():
        return [Subject.SIGN], "ocr:text_present"
    return [], None


def _stage_model(
    result, asset: Asset, policy: PipelinePolicy  # noqa: ANN001
) -> tuple[list[Subject], list[Subject], float, str]:
    """Étape 4 — ce que le modèle propose, avec ses seuils.

    La confiance porte sur les sujets décisifs, et non sur l'ensemble : une
    classe hors sujet nettement rejetée gonflait l'agrégat à 0,999 sur des
    décisions médiocres.
    """
    accepted = [Subject(s) for s in result.accepted(policy.model.subject_accept)]
    uncertain = [
        Subject(s)
        for s in result.uncertain(policy.model.subject_reject, policy.model.subject_accept)
    ]
    decisive = [s.value for s in DECISIVE_SUBJECTS if s.value in result.scores]
    return accepted, uncertain, result.confidence(decisive), "openclip:multilabel"


def _target_visibility(
    asset: Asset,
    model_contains_building: bool | None,
    target_in_fov: bool,
) -> tuple[bool | None, str | None]:
    """Le bâtiment **cible** est-il visible, et sur quelle preuve ?

    Deux axes strictement distincts, qu'il ne faut jamais fusionner :

    ```text
    enseigne reconnue   → propriété confirmée   (identité)
    bâtiment détecté    → contenu confirmé      (visibilité)
    les deux ensemble   → cible probablement visible
    revue humaine       → cible confirmée
    ```

    `model_contains_building` vient **exclusivement** du modèle. Le calculer
    depuis la liste `subjects` était fautif : cette liste fusionne modèle,
    géométrie et OCR, si bien que la géométrie forçait le drapeau à vrai —
    11 vues sur 13 déclarées « bâtiment confirmé » avec des scores tombant à
    0,0006.
    """
    # Une décision humaine prime sur toute déduction et n'est jamais écrasée.
    if asset.target_visibility_decision is ReviewDecision.CONFIRMED:
        return True, f"revue humaine : {asset.review_rationale or 'confirmé'}"
    if asset.target_visibility_decision is ReviewDecision.REJECTED:
        return False, f"revue humaine : {asset.review_rationale or 'rejeté'}"

    if asset.property_match_status is PropertyMatchStatus.MISMATCH:
        return False, "enseigne d'un autre établissement"

    if asset.occluded_by:
        return False, f"masqué par {asset.occluded_by}"

    if model_contains_building is False:
        # Le modèle affirme qu'aucun bâtiment n'est visible : viser la bonne
        # direction n'y change rien.
        return False, "aucun bâtiment détecté par le modèle"

    if model_contains_building is None:
        return None, "contenu non évalué"

    identity = asset.property_match_status is PropertyMatchStatus.MATCH

    if target_in_fov:
        return True, "cap observé cadrant l'empreinte, bâtiment confirmé par le modèle"

    if identity:
        return True, "enseigne de l'établissement et bâtiment confirmé par le modèle"

    # Un bâtiment est là, mais rien ne dit que c'est le nôtre.
    return None, "bâtiment présent, identité non établie"


def classify(
    assets: list[Asset],
    classifier=None,  # noqa: ANN001
    front_azimuth: float | None = None,
    policy: PipelinePolicy = DEFAULT_POLICY,
) -> CascadeReport:
    """Applique la cascade à chaque asset, en place."""
    report = CascadeReport(total=len(assets))

    for index, asset in enumerate(assets):
        subjects: list[Subject] = []
        methods: list[str] = []
        confidence: float | None = None
        uncertain: list[Subject] = []

        found, method = _stage_source(asset)
        subjects.extend(found)
        if method:
            methods.append(method)
            report.by_stage["source"] = report.by_stage.get("source", 0) + 1

        found, sector, method = _stage_geometry(asset, front_azimuth)
        subjects.extend(found)
        if method:
            methods.append(method)
            report.by_stage["geometry"] = report.by_stage.get("geometry", 0) + 1

        found, method = _stage_ocr(asset)
        subjects.extend(found)
        if method:
            methods.append(method)
            report.by_stage["ocr"] = report.by_stage.get("ocr", 0) + 1

        scores: dict[str, float] = {}
        if classifier is not None and asset.local_path and Path(asset.local_path).is_file():
            try:
                result = classifier.multi_label(Path(asset.local_path))
            except (OSError, ValueError, RuntimeError) as exc:
                log.warning("classification impossible pour %s : %s", asset.id, exc)
            else:
                accepted, uncertain, confidence, method = _stage_model(result, asset, policy)
                # Le modèle complète une géométrie mesurée ; sur un cap choisi,
                # il est la seule preuve du contenu.
                subjects.extend(s for s in accepted if s not in subjects)
                methods.append(method)
                scores = {k: round(v, 4) for k, v in result.scores.items()}
                report.by_stage["model"] = report.by_stage.get("model", 0) + 1

        # --- identité de la cible, distincte de la présence d'un bâtiment ---
        # `contains_building` est la réponse du **modèle seul**. Le déduire de
        # `subjects` mêlerait géométrie et OCR à ce qui doit rester une mesure
        # de contenu.
        model_contains: bool | None = None
        if scores:
            model_contains = (
                scores.get(Subject.BUILDING.value, 0.0) >= policy.model.subject_accept
            )

        target_in_fov = bool(asset.sees_building and asset.heading_is_measured)
        target, evidence = _target_visibility(asset, model_contains, target_in_fov)
        contains = model_contains

        # Étape 5 — ce qui doit passer devant un humain.
        review = ReviewStatus.NEEDS_REVIEW
        decisive_uncertain = [s for s in uncertain if s in DECISIVE_SUBJECTS]
        blocked_by_occlusion = bool(asset.occluded_by) and contains
        if (
            not decisive_uncertain
            and not blocked_by_occlusion
            and (confidence is None or confidence >= policy.model.review_confidence_floor)
            and (subjects or asset.sees_building is not None)
        ):
            review = ReviewStatus.AUTOMATIC_ACCEPTED

        # Une décision humaine emporte **les deux** champs. Ne préserver que
        # `target_visibility_decision` laissait la cascade recalculer le
        # statut : le verdict de visibilité survivait, tandis que
        # `human_accepted` ou `rejected` retombait en `needs_review` ou en
        # `automatic_accepted`. Une revue aurait été à refaire sans que rien
        # ne le dise, et une acceptation automatique aurait pu se substituer à
        # un rejet humain.
        #
        # `target_visibility_decision` vaut `unresolved` par défaut sur tout
        # asset : s'y fier seul ferait passer chaque image jamais examinée pour
        # une revue sans conclusion. Seul l'historique dit qu'une personne a
        # réellement regardé.
        if asset.has_been_reviewed:
            review = _HUMAN_STATUS[asset.target_visibility_decision]

        deduped = sorted(set(subjects), key=lambda s: s.value)
        assets[index] = asset.model_copy(
            update={
                "subjects": deduped,
                "view_sector": sector,
                "classification_confidence": confidence,
                "classification_method": "+".join(methods) if methods else None,
                "review_status": review,
                "contains_building": contains,
                "target_building_visible": target,
                "target_evidence": evidence,
                "subject_scores": scores,
            }
        )
        if blocked_by_occlusion:
            report.occlusion_conflicts += 1

        for subject in deduped:
            report.subjects_assigned[subject.value] = (
                report.subjects_assigned.get(subject.value, 0) + 1
            )
        report.sectors_assigned[sector.value] = report.sectors_assigned.get(sector.value, 0) + 1
        if review is ReviewStatus.NEEDS_REVIEW:
            report.needs_review += 1
        if sector is ViewSector.UNKNOWN:
            report.unknown_sector += 1

    log.info(
        "cascade : %d asset(s), %d en revue, %d sans secteur",
        report.total,
        report.needs_review,
        report.unknown_sector,
    )
    return report


def property_status(asset: Asset, expected: list[str], excluded: list[str]) -> PropertyMatchStatus:
    """Appartenance — décision **indépendante** de la catégorie (§6).

    Une image peut montrer une magnifique façade d'hôtel sans être celle du
    bon hôtel. La géométrie et l'OCR y répondent ; le classifieur, jamais.

    La géométrie ne vaut ici que si le cap est **observé** : viser soi-même
    l'empreinte depuis un panorama sphérique ne prouve pas l'appartenance, pas
    davantage qu'il ne prouvait le contenu.
    """
    from .triage import evaluate

    if asset.sees_building and asset.heading_is_measured:
        return PropertyMatchStatus.MATCH

    if asset.sign_text:
        return evaluate(asset.sign_text, expected, excluded).status

    return PropertyMatchStatus.UNCERTAIN


def entrance_version_guard(asset: Asset) -> TemporalStatus:
    """La version de l'entrée ne s'infère jamais sans preuve datée (§6).

    Une année de capture ne suffit pas : une photo promotionnelle publiée en
    2025 peut montrer l'entrée d'avant travaux.
    """
    if asset.temporal_status is not TemporalStatus.UNKNOWN:
        return asset.temporal_status
    return TemporalStatus.UNKNOWN
