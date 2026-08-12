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
    ReviewStatus,
    Subject,
    TemporalStatus,
    ViewSector,
)
from .sectors import sector_for

log = get_logger("cascade")

#: Sujets dont une erreur est coûteuse : ils décident de la couverture par
#: secteur, donc de ce qu'une caméra future aura le droit de montrer.
DECISIVE_SUBJECTS = frozenset({Subject.BUILDING, Subject.ENTRANCE, Subject.ROOF})


@dataclass
class CascadeReport:
    total: int = 0
    by_stage: dict[str, int] = field(default_factory=dict)
    subjects_assigned: dict[str, int] = field(default_factory=dict)
    sectors_assigned: dict[str, int] = field(default_factory=dict)
    needs_review: int = 0
    unknown_sector: int = 0

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "decided_by_stage": self.by_stage,
            "subjects": self.subjects_assigned,
            "sectors": self.sectors_assigned,
            "needs_review": self.needs_review,
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

    if asset.sees_building:
        subjects.append(Subject.BUILDING)
        method = "geometry:fov"

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


def _stage_model(result, asset: Asset) -> tuple[list[Subject], list[Subject], float, str]:  # noqa: ANN001
    """Étape 4 — ce que le modèle propose, avec ses seuils."""
    accepted = [Subject(s) for s in result.accepted()]
    uncertain = [Subject(s) for s in result.uncertain()]
    return accepted, uncertain, result.confidence(), "openclip:multilabel"


def classify(
    assets: list[Asset],
    classifier=None,  # noqa: ANN001
    front_azimuth: float | None = None,
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

        if classifier is not None and asset.local_path and Path(asset.local_path).is_file():
            try:
                result = classifier.multi_label(Path(asset.local_path))
            except (OSError, ValueError, RuntimeError) as exc:
                log.warning("classification impossible pour %s : %s", asset.id, exc)
            else:
                accepted, uncertain, confidence, method = _stage_model(result, asset)
                # Le modèle complète, il ne contredit pas la géométrie.
                subjects.extend(s for s in accepted if s not in subjects)
                methods.append(method)
                report.by_stage["model"] = report.by_stage.get("model", 0) + 1

        # Étape 5 — ce qui doit passer devant un humain.
        review = ReviewStatus.NEEDS_REVIEW
        decisive_uncertain = [s for s in uncertain if s in DECISIVE_SUBJECTS]
        if not decisive_uncertain and (confidence is None or confidence >= 0.6):
            if subjects or asset.sees_building is not None:
                review = ReviewStatus.AUTOMATIC_ACCEPTED

        deduped = sorted(set(subjects), key=lambda s: s.value)
        assets[index] = asset.model_copy(
            update={
                "subjects": deduped,
                "view_sector": sector,
                "classification_confidence": confidence,
                "classification_method": "+".join(methods) if methods else None,
                "review_status": review,
            }
        )

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
    """
    from .triage import evaluate

    if asset.sees_building:
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
