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
    # L'acceptation automatique ne peut pas la recouvrir, et la file d'attente
    # ne doit pas la réclamer indéfiniment.
    ReviewDecision.UNRESOLVED: ReviewStatus.HUMAN_UNRESOLVED,
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


def _framing(
    asset: Asset,
    building_wkt: str | None,
    policy: PipelinePolicy,
) -> tuple[bool | None, str | None]:
    """L'empreinte cible entre-t-elle dans le champ, d'après la **mesure** ?

    Position, cap et empreinte sont mesurés ; l'avis d'un classifieur sur une
    vignette compressée est inféré. Quand les trois existent, le cadrage
    tranche donc avant le modèle.

    `visibility.assess` compare le cap au point de l'empreinte **le plus
    proche**, non au centroïde : de près, une façade peut remplir l'image
    alors que le centroïde est sorti du champ.

    Retourne `(True | False, motif)`, ou `(None, None)` quand une des mesures
    manque — auquel cas on ne conclut pas.
    """
    if building_wkt is None:
        return None, None
    if asset.camera_lat is None or asset.camera_lon is None:
        return None, None
    if asset.heading_deg is None or not asset.heading_is_measured:
        # Un cap que nous avons nous-mêmes dirigé vers l'empreinte ne prouve
        # rien sur le contenu : il ne fait que répéter notre intention.
        return None, None

    from .visibility import assess

    try:
        verdict = assess(
            asset.camera_lat,
            asset.camera_lon,
            asset.heading_deg,
            building_wkt,
            half_fov_deg=policy.geometry.half_fov_deg,
        )
    except Exception:  # géométrie illisible : on ne conclut pas
        return None, None

    if verdict.visible:
        return True, f"empreinte cadrée par un cap mesuré ({verdict.reason})"
    return False, f"empreinte hors du champ d'un cap mesuré ({verdict.reason})"


def _framing_strength(
    asset: Asset,
    building_wkt: str | None,
    policy: PipelinePolicy,
) -> float | None:
    """Force continue du cadrage, conservée à côté du verdict booléen."""
    if building_wkt is None or asset.camera_lat is None or asset.camera_lon is None:
        return None
    if asset.heading_deg is None or not asset.heading_is_measured:
        return None

    from .visibility import assess

    try:
        return assess(
            asset.camera_lat,
            asset.camera_lon,
            asset.heading_deg,
            building_wkt,
            half_fov_deg=policy.geometry.half_fov_deg,
        ).framing_strength
    except Exception:
        return None


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
    framed: bool | None = None,
    framing_reason: str | None = None,
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

    #: Après un `unresolved` humain, le système peut constater qu'il ne voit
    #: rien ; il ne peut pas établir la cible. La déduction est donc plafonnée.
    reviewed_undecided = (
        asset.has_been_reviewed
        and asset.target_visibility_decision is ReviewDecision.UNRESOLVED
    )

    if asset.property_match_status is PropertyMatchStatus.MISMATCH:
        return False, "enseigne d'un autre établissement"

    if asset.occluded_by:
        return False, f"masqué par {asset.occluded_by}"

    # --- Le cadrage mesuré passe avant le modèle -------------------------
    #
    # Mesuré sur le corpus pilote, contre 26 décisions humaines :
    #
    #   classifieur seul            rappel  26 %   précision 38 %
    #   cadrage géométrique mesuré  rappel 100 %   précision 71 %
    #
    # Un classifieur à 26 % de rappel qui dit « non » énonce son état de
    # repos, pas une observation. Une empreinte hors du champ d'un cap
    # mesuré, elle, est une mesure : c'est elle qui a le droit d'exclure.
    if framed is False:
        return False, framing_reason or "empreinte hors du champ d'un cap mesuré"

    if framed is True:
        # Une personne qui a regardé sans conclure n'est pas contredite par
        # une déduction : après un `unresolved` humain, le cadrage peut
        # constater, jamais établir.
        if reviewed_undecided:
            return None, (
                "revue humaine non conclusive : le cadrage mesuré ne peut pas "
                "établir ce qu'une lecture directe a refusé d'établir"
            )
        # Le cadrage établit que la cible est dans l'image. L'identité reste
        # une question distincte, traitée plus bas.
        return True, framing_reason or "empreinte cadrée par un cap mesuré"

    if model_contains_building is False:
        # Sans cadrage mesuré, l'avis du modèle ne suffit pas à **prouver**
        # l'absence : on ne conclut pas, et la revue tranchera. Le traiter
        # comme une preuve retirait 271 assets jamais regardés par personne.
        return None, "aucun bâtiment détecté par le modèle : à établir"

    if model_contains_building is None:
        return None, "contenu non évalué"

    identity = asset.property_match_status is PropertyMatchStatus.MATCH

    if target_in_fov:
        if reviewed_undecided:
            return None, (
                "revue humaine non conclusive : la géométrie et le modèle ne "
                "peuvent pas établir ce qu'une lecture directe a refusé d'établir"
            )
        return True, "cap observé cadrant l'empreinte, bâtiment confirmé par le modèle"

    if identity:
        if reviewed_undecided:
            return None, (
                "revue humaine non conclusive : l'enseigne ne suffit pas à "
                "rétablir ce qu'une lecture directe a refusé d'établir"
            )
        return True, "enseigne de l'établissement et bâtiment confirmé par le modèle"

    # Un bâtiment est là, mais rien ne dit que c'est le nôtre.
    return None, "bâtiment présent, identité non établie"


def classify(
    assets: list[Asset],
    classifier=None,  # noqa: ANN001
    front_azimuth: float | None = None,
    policy: PipelinePolicy = DEFAULT_POLICY,
    building_wkt: str | None = None,
) -> CascadeReport:
    """Applique la cascade à chaque asset, en place.

    `building_wkt` est l'empreinte confirmée du bâtiment cible. Fournie, elle
    active le test de cadrage mesuré, qui prime sur le classifieur. Absente,
    la cascade retombe sur le comportement antérieur.
    """
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

        # Les scores déjà mesurés servent de repli : rejouer la cascade après
        # une correction de cadrage ne doit pas exiger de relancer OpenCLIP,
        # ni faire disparaître ce que le modèle avait vu.
        scores: dict[str, float] = dict(asset.subject_scores)
        if scores:
            methods.append("openclip:scores_conservés")
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

        # `sees_building` vient de l'ancien annotateur mono-rayon : il est
        # conservé comme trace historique, non comme preuve. Seul un cadrage
        # réellement calculé dit que la cible entre dans l'image — et le corpus
        # n'en compte aujourd'hui aucun.
        frame_fraction = asset.target_in_frame_fraction
        target_in_fov = bool(
            frame_fraction and frame_fraction > 0 and asset.heading_is_measured
        )
        framed, framing_reason = _framing(asset, building_wkt, policy)
        framing_strength = _framing_strength(asset, building_wkt, policy)
        if framing_reason:
            methods.append("geometry:framing")
            report.by_stage["framing"] = report.by_stage.get("framing", 0) + 1
        target, evidence = _target_visibility(
            asset, model_contains, target_in_fov, framed, framing_reason
        )
        contains = model_contains

        # Étape 5 — ce qui doit passer devant un humain.
        review = ReviewStatus.NEEDS_REVIEW
        decisive_uncertain = [s for s in uncertain if s in DECISIVE_SUBJECTS]
        blocked_by_occlusion = bool(asset.occluded_by) and contains
        # Une cible non établie est précisément ce qu'une revue tranche. Sans
        # cette condition, un asset dont la visibilité reste inconnue était
        # « accepté automatiquement » et n'entrait dans aucune file : 113 vues
        # du corpus pilote n'étaient ni exploitées ni jamais soumises à
        # personne. Accepter automatiquement ne peut pas signifier « clore une
        # question sans y répondre ».
        # Distinguer « rien n'a été évalué » de « évalué, non concluant ».
        # Sans contenu mesuré ni modèle, il n'y a pas de question ouverte à
        # soumettre : c'est l'état antérieur du corpus, pas une indécision.
        # En revanche, un modèle qui a répondu sans établir la cible laisse
        # une question ouverte — et c'est une revue qui la tranche.
        target_undecided = (
            target is None
            and not asset.has_been_reviewed
            and (model_contains is not None or framed is not None)
        )
        if (
            not decisive_uncertain
            and not blocked_by_occlusion
            and not target_undecided
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
                "framing_strength": framing_strength,
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
