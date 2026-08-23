"""Aptitude d'une vue à servir de **référence d'apparence** (Lot 2).

`surface_coverage_confidence` répond « peut-on reconstruire cette surface ? ».
C'est une question de géométrie : assez de vues, assez écartées, assez proches.

Une vidéo promotionnelle en pose une autre : **cette image est-elle belle, et
montre-t-elle assez le bâtiment pour servir de référence ?** Une façade
parfaitement reconstructible peut n'être documentée que par des clichés de nuit,
floués, sous la neige, avec l'hôtel au fond derrière un concessionnaire. La
structure serait juste et l'apparence inutilisable.

Les deux questions sont indépendantes et se mesurent séparément. Ce module ne
juge que la seconde, et seulement sur ce qui est **mesurable sur le fichier** :

- **netteté** — variance du laplacien, déjà calculée par `triage.quality` ;
- **exposition** — ni bouchée ni brûlée ;
- **prominence** — part du cadre que le bâtiment occupe, dérivée de la
  distance et du champ de vision ; une vue à 200 m ne montre pas une façade ;
- **saison** — la neige et les arbres nus changent l'apparence sans changer la
  géométrie ; c'est une information, jamais un rejet.

Ce qui n'est **pas** mesuré, et qu'aucun score ne doit laisser croire acquis :
la composition, la qualité de la lumière, la propreté du sujet. Ces jugements
demandent un regard, et le module dit lesquels restent à porter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Variance du laplacien au-delà de laquelle la netteté cesse d'être limitante.
#: Aligné sur `triage.quality.BLUR_VARIANCE_THRESHOLD`, dont c'est le seuil de
#: rejet : on sature à dix fois celui-ci.
SHARPNESS_SATURATION = 600.0

#: Luminance moyenne idéale, et demi-largeur de la plage acceptable.
BRIGHTNESS_IDEAL = 128.0
BRIGHTNESS_TOLERANCE = 70.0

#: Part de la largeur du cadre au-delà de laquelle le sujet est assez présent
#: pour servir de référence d'apparence.
#:
#: Une première valeur de 0,12 saturait dès 400 m : un bâtiment occupant un
#: huitième du cadre y obtenait la note pleine, alors qu'aucune texture de
#: façade n'y est lisible. À 60 m de large et 70° de champ, la moitié du cadre
#: correspond à ~95 m — la distance des vues de rue les plus proches, et
#: l'ordre de grandeur où une façade devient exploitable.
PROMINENCE_TARGET = 0.50

#: Champ de vision horizontal supposé quand la source n'en déclare pas.
#: Ordre de grandeur d'une caméra de roulage ; utilisé faute de mieux, et le
#: rapport le dit.
DEFAULT_FOV_DEG = 70.0


@dataclass
class AppearanceEvidence:
    """Ce qu'on peut mesurer d'une vue, avant tout jugement."""

    asset_id: str
    sharpness: float | None = None
    brightness: float | None = None
    #: Distance caméra → bâtiment, en mètres.
    distance_m: float | None = None
    #: Plus grande dimension apparente du bâtiment, en mètres.
    subject_size_m: float = 60.0
    fov_deg: float | None = None
    #: Cap de la caméra et azimut vers le bâtiment, en degrés. Sans eux, la
    #: prominence ne sait pas si l'objectif regarde seulement la cible.
    heading_deg: float | None = None
    bearing_to_subject_deg: float | None = None
    heading_is_measured: bool = False
    #: Prominence **lue sur les pixels** par `subject_prominence`. Quand elle
    #: existe, elle prime : la géométrie ne sait pas ce qui se met devant.
    measured_prominence: float | None = None
    captured_year: int | None = None
    #: Mois de prise de vue, quand la source le publie.
    captured_month: int | None = None


@dataclass
class AppearanceQuality:
    """Aptitude d'une vue à servir de référence d'apparence."""

    asset_id: str
    sharpness: float
    exposure: float
    prominence: float
    score: float
    limiting_factor: str
    verdict: str
    season: str
    unmeasured: list[str]

    def as_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "sharpness": round(self.sharpness, 3),
            "exposure": round(self.exposure, 3),
            "prominence": round(self.prominence, 3),
            "score": round(self.score, 3),
            "limiting_factor": self.limiting_factor,
            "verdict": self.verdict,
            "season": self.season,
            "unmeasured": list(self.unmeasured),
        }


def _season(month: int | None) -> str:
    """Saison nordique, ou `unknown`.

    La neige et les arbres nus changent l'apparence d'une façade sans rien
    changer à sa géométrie. On l'enregistre pour que la sélection puisse
    préférer une saison, jamais pour écarter une vue.
    """
    if month is None:
        return "unknown"
    if month in (12, 1, 2, 3):
        return "winter"
    if month in (4, 5):
        return "spring"
    if month in (6, 7, 8, 9):
        return "summer"
    return "autumn"


def prominence(
    distance_m: float | None,
    subject_size_m: float,
    fov_deg: float | None,
) -> float | None:
    """Part de la largeur du cadre occupée par le sujet.

    Géométrie élémentaire : à distance `d`, un sujet large de `s` sous-tend
    `2·atan(s / 2d)`, à comparer au champ de vision. Sans distance, la mesure
    n'est pas possible — et l'absence se dit, elle ne vaut pas zéro.
    """
    import math

    if distance_m is None or distance_m <= 0 or subject_size_m <= 0:
        return None
    field_of_view = fov_deg or DEFAULT_FOV_DEG
    subtended = 2.0 * math.degrees(math.atan(subject_size_m / (2.0 * distance_m)))
    return float(min(1.0, subtended / field_of_view))


def _framing_factor(evidence: "AppearanceEvidence") -> float | None:
    """Part du cadrage réellement dirigée vers le sujet, ou None.

    Retourne 1.0 quand la cible est proche de l'axe optique, décroît jusqu'à
    0 au bord du champ, et 0 au-delà. `None` quand le cap n'est pas mesuré :
    l'absence d'information ne vaut pas un cadrage réussi.
    """
    if not evidence.heading_is_measured:
        return None
    if evidence.heading_deg is None or evidence.bearing_to_subject_deg is None:
        return None

    offset = abs(
        (evidence.heading_deg - evidence.bearing_to_subject_deg + 180.0) % 360.0
        - 180.0
    )
    half_fov = (evidence.fov_deg or DEFAULT_FOV_DEG) / 2.0
    if half_fov <= 0:
        return None
    if offset >= half_fov:
        return 0.0
    return float(1.0 - offset / half_fov)


def assess(evidence: AppearanceEvidence) -> AppearanceQuality:
    """Évalue l'aptitude d'une vue à servir de référence d'apparence."""
    unmeasured: list[str] = [
        "composition",
        "qualité de la lumière",
        "propreté du sujet",
    ]

    if evidence.sharpness is None:
        sharpness = 0.0
        unmeasured.append("netteté")
    else:
        sharpness = min(1.0, max(0.0, evidence.sharpness / SHARPNESS_SATURATION))

    if evidence.brightness is None:
        exposure = 0.0
        unmeasured.append("exposition")
    else:
        deviation = abs(evidence.brightness - BRIGHTNESS_IDEAL)
        exposure = max(0.0, 1.0 - deviation / BRIGHTNESS_TOLERANCE)

    # La prominence est une **prédiction géométrique**, non une observation :
    # elle déduit du couple (distance, champ de vision) la place que le sujet
    # *devrait* occuper. Elle ignore ce qui se met devant — clôture, lampadaire,
    # stationnement incitatif — et, sans cap, elle ignore même si l'objectif
    # regarde la cible.
    #
    # Mesuré sur le pilote : une vue à 63 m notée 0,99 « reference_grade »
    # montrait l'hôtel comme un petit bloc lointain derrière un parc-o-bus.
    # Le score confondait « proche » et « bien cadré ».
    if evidence.measured_prominence is not None:
        # Preuve pixel : elle remplace la prédiction, elle ne la pondère pas.
        subject = float(min(1.0, max(0.0, evidence.measured_prominence)))
        raw_prominence = subject
    else:
        raw_prominence = prominence(
            evidence.distance_m, evidence.subject_size_m, evidence.fov_deg
        )
    if evidence.measured_prominence is not None:
        pass
    elif raw_prominence is None:
        subject = 0.0
        unmeasured.append("prominence")
    else:
        subject = min(1.0, raw_prominence / PROMINENCE_TARGET)
        # Le cadrage borne la prominence : hors champ, la taille apparente
        # théorique ne décrit plus rien.
        framing = _framing_factor(evidence)
        if framing is None:
            unmeasured.append("cadrage (cap non mesuré)")
            # Sans cap, la prominence reste une hypothèse : on la plafonne
            # pour qu'elle ne puisse pas seule porter un verdict de référence.
            subject = min(subject, 0.5)
        else:
            subject *= framing
    if evidence.measured_prominence is None:
        unmeasured.append("occlusion de premier plan")

    # Moyenne géométrique : une composante nulle annule le score. Une image
    # parfaitement nette d'un bâtiment à 400 m ne sert pas de référence
    # d'apparence, et une moyenne arithmétique le laisserait croire.
    score = float((sharpness * exposure * subject) ** (1 / 3))

    components = {
        "netteté": sharpness,
        "exposition": exposure,
        "prominence": subject,
    }
    limiting = min(components, key=lambda key: components[key])

    if score >= 0.7:
        verdict = "reference_grade"
    elif score >= 0.4:
        verdict = "usable"
    elif score > 0.0:
        verdict = "weak"
    else:
        verdict = "unusable"

    return AppearanceQuality(
        asset_id=evidence.asset_id,
        sharpness=sharpness,
        exposure=exposure,
        prominence=subject,
        score=score,
        limiting_factor=limiting,
        verdict=verdict,
        season=_season(evidence.captured_month),
        unmeasured=unmeasured,
    )


def assess_all(evidences: list[AppearanceEvidence]) -> list[AppearanceQuality]:
    return [assess(e) for e in evidences]


def evidence_from_file(
    asset_id: str,
    image_path: Path,
    *,
    distance_m: float | None = None,
    subject_size_m: float = 60.0,
    fov_deg: float | None = None,
    captured_month: int | None = None,
    heading_deg: float | None = None,
    bearing_to_subject_deg: float | None = None,
    heading_is_measured: bool = False,
) -> AppearanceEvidence:
    """Mesure netteté et exposition sur le fichier, sans les inventer."""
    from .triage.quality import basic_scores

    try:
        scores = basic_scores(image_path)
    except Exception:
        return AppearanceEvidence(
            asset_id=asset_id,
            distance_m=distance_m,
            subject_size_m=subject_size_m,
            fov_deg=fov_deg,
            captured_month=captured_month,
            heading_deg=heading_deg,
            bearing_to_subject_deg=bearing_to_subject_deg,
            heading_is_measured=heading_is_measured,
        )

    return AppearanceEvidence(
        asset_id=asset_id,
        sharpness=scores.get("sharpness"),
        brightness=scores.get("brightness"),
        distance_m=distance_m,
        subject_size_m=subject_size_m,
        fov_deg=fov_deg,
        captured_month=captured_month,
        heading_deg=heading_deg,
        bearing_to_subject_deg=bearing_to_subject_deg,
        heading_is_measured=heading_is_measured,
    )


__all__ = [
    "AppearanceEvidence",
    "AppearanceQuality",
    "assess",
    "assess_all",
    "evidence_from_file",
    "prominence",
]
