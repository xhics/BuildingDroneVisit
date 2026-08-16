"""Contrat d'acquisition photographique ciblée (Lot 1B V2).

La chaîne précédente découvrait, téléchargeait tout, puis qualifiait. Elle
traitait donc une image trouvée comme une image acquise, et une image acquise
comme une preuve. Quatre vérités s'y confondaient :

```text
ce qu'on cherche              → CaptureDemand        (objectif, stable)
où l'on en est                → DemandAssessment     (mesuré sur un corpus)
ce qu'on a trouvé             → CaptureCandidate     (métadonnées seules)
ce que ça vaut pour un besoin → CandidateEvaluation  (un couple à la fois)
ce qu'on a décidé de prendre  → AcquisitionPlan
ce qu'on a réellement pris    → Asset + AcquisitionProvenance
```

Trois séparations méritent d'être justifiées.

Le besoin est **immuable** ; « satisfait » ou « inatteignable » dépend du
corpus du jour. Les tenir ensemble ferait d'un objectif une variable.

Un candidat vaut différemment selon le besoin : la même vue peut cadrer la
façade avant, manquer l'entrée et documenter la voie d'accès. Une seule
géométrie et une seule intention par candidat ne peuvent pas le dire.

Aucun manifeste ne conserve d'URL. Celles des CDN expirent, les signées
portent une clé d'API : l'adresse se reconstruit en mémoire au moment du
téléchargement, à partir d'une spécification sans secret.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .enums import ViewSector

#: Champs d'empreinte qu'un plan exécutable doit **tous** porter. Une liste
#: fermée, vérifiée par le plan lui-même : laisser l'appelant choisir ce qu'il
#: transmet permettait de déclarer courant un plan sans lien avec le site.
#: `policy_digest` y figure pour la **provenance** : un plan doit dire avec
#: quels réglages il a été produit. Il n'y sert pas de dépendance — c'est
#: `policy_dependency_digests` qui décide de sa péremption, sans quoi un seuil
#: de terrain périmerait un plan photographique.
REQUIRED_PLAN_DIGESTS: tuple[str, ...] = (
    "candidate_manifest_digest",
    "demand_digest",
    "policy_digest",
    "site_manifest_digest",
    "spatial_manifest_digest",
    "corpus_digest",
    "road_geometry_digest",
    "obstacle_geometry_digest",
)


class CaptureIntent(StrEnum):
    """Ce qu'une prise de vue est censée servir.

    Les deux intentions n'ont pas les mêmes exigences : une vue de bâtiment
    demande une ligne de vue utile et une taille projetée suffisante, une vue
    de contexte documente un accès, une orientation ou une transition. Les
    confondre a produit 302 verrous de contexte par défaut plutôt que par
    choix.
    """

    BUILDING_CAPTURE = "building_capture"
    CONTEXT_CAPTURE = "context_capture"


class TargetKind(StrEnum):
    """Nature d'une cible de besoin.

    Une chaîne libre recréerait les décalages de vocabulaire déjà rencontrés —
    `ROOFLINE_MAIN2` n'établissait rien sans que rien ne le signale.
    """

    SITE_OBJECT = "site_object"
    VIEW_SECTOR = "view_sector"
    CONTEXT_CORRIDOR = "context_corridor"
    TRANSITION = "transition"


class DemandStatus(StrEnum):
    OPEN = "open"
    PARTIALLY_MET = "partially_met"
    MET = "met"
    #: Aucune acquisition possible ne peut le satisfaire — voie privée,
    #: façade sans accès public. Un besoin clos, non un besoin ouvert.
    UNREACHABLE = "unreachable"


class Eligibility(StrEnum):
    ELIGIBLE = "eligible"
    REJECTED = "rejected"
    #: Admissible sur la géométrie, mais son contenu demande une vérification
    #: par miniature avant d'engager la pleine résolution.
    PREVIEW_REQUIRED = "preview_required"


class VolumeStatus(StrEnum):
    EXACT = "exact"
    PARTIAL = "partial"
    #: Aucune taille connue et aucune estimation produite. « Estimé » aurait
    #: laissé croire qu'un calcul a eu lieu.
    UNKNOWN = "unknown"


class PlanStatus(StrEnum):
    #: Un brouillon existe pour être discuté ; il ne s'acquiert jamais.
    DRAFT = "draft"
    EXECUTABLE = "executable"


# --- besoins ----------------------------------------------------------------


class CaptureDemand(BaseModel):
    """Un objectif de couverture, **immuable**.

    Chercher d'abord et constater ensuite reviendrait à laisser le corpus
    définir l'objectif. Le besoin est donc énoncé en premier, et c'est lui qui
    juge la collecte — jamais l'inverse.
    """

    model_config = ConfigDict(extra="forbid")

    demand_id: str
    intent: CaptureIntent

    target_kind: TargetKind
    #: Identifiant d'objet du site, secteur du vocabulaire officiel, ou
    #: corridor. Vérifié par `validate_targets`, jamais librement interprété.
    target_ref: str

    #: Nombre de **points de vue indépendants** attendus, jamais de fichiers :
    #: deux photographies d'un même point ne font pas deux observations.
    viewpoints_required: int = Field(default=1, ge=1)

    #: Recouvrement attendu entre vues voisines, pour qu'un SfM puisse les
    #: relier. Zéro signifie « vues indépendantes acceptées ».
    continuity_required: float = Field(default=0.0, ge=0.0, le=1.0)

    #: Taille projetée minimale de la cible, en fraction de la largeur du
    #: cadre. Le critère utile n'est pas la distance : un téléobjectif à 117 m
    #: vaut mieux qu'un grand-angle à 40.
    min_projected_width_fraction: float = Field(default=0.0, ge=0.0, le=1.0)

    #: Fraction de façade utile non masquée, en deçà de laquelle la vue ne
    #: répond pas au besoin.
    min_visible_fraction: float = Field(default=0.0, ge=0.0, le=1.0)

    #: Zones interdites aux plans rapprochés, **désignées** : les 18 lacunes de
    #: toiture, une géométrie versionnée. Un booléen ne disait pas laquelle.
    forbidden_zone_refs: list[str] = Field(default_factory=list)

    rationale: str | None = None


class DemandAssessment(BaseModel):
    """Où en est un besoin, **contre un corpus précis**.

    Le même objectif est satisfait ou non selon ce qu'on possède : l'état est
    donc daté et rattaché à une empreinte de corpus, sans quoi « couvert »
    serait un souvenir.
    """

    model_config = ConfigDict(extra="forbid")

    demand_id: str
    corpus_digest: str = Field(min_length=1)
    assessed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: DemandStatus = DemandStatus.OPEN

    viewpoints_found: int = Field(default=0, ge=0)
    continuity_achieved: float | None = Field(default=None, ge=0.0, le=1.0)
    best_projected_width_fraction: float | None = Field(default=None, ge=0.0)
    best_visible_fraction: float | None = Field(default=None, ge=0.0, le=1.0)

    #: Pourquoi cet état. Obligatoire pour un besoin déclaré inatteignable :
    #: renoncer sans motif interdit d'y revenir.
    rationale: str | None = None

    #: Niveau atteint par la continuité, quand elle est exigée. Trois états,
    #: parce qu'ils ne s'établissent pas au même moment :
    #:
    #: ```text
    #: planned    positions, séquence, caps et champs de vision
    #: observed   recouvrement mesuré après acquisition
    #: verified   enregistrement géométrique, au Lot 2
    #: ```
    #:
    #: `planned` suffit à planifier une acquisition ; il ne suffira jamais à
    #: déclarer un besoin satisfait.
    continuity_level: str | None = None

    @model_validator(mode="after")
    def _unreachable_needs_a_reason(self) -> "DemandAssessment":
        if self.status is DemandStatus.UNREACHABLE and not (self.rationale or "").strip():
            raise ValueError(
                f"besoin {self.demand_id!r} déclaré inatteignable sans motif"
            )
        return self

    def meets(self, demand: "CaptureDemand") -> bool:
        """Ce corpus satisfait-il **ce** besoin ?

        Le nombre de points de vue ne suffit pas : un besoin exigeant de la
        continuité n'est satisfait que si elle a été **mesurée** et atteint le
        seuil. Une continuité planifiée dit qu'on l'a cherchée, non qu'on l'a
        obtenue — et un SfM ne se contente pas d'une intention.
        """
        if self.viewpoints_found < demand.viewpoints_required:
            return False
        if demand.continuity_required > 0:
            if self.continuity_achieved is None:
                return False
            if self.continuity_achieved < demand.continuity_required:
                return False
            if self.continuity_level == "planned":
                return False
        return True


class CaptureDemandManifest(BaseModel):
    """Les objectifs, et ce dont ils dérivent."""

    model_config = ConfigDict(extra="forbid")

    hotel_id: str
    built_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    demands: list[CaptureDemand] = Field(default_factory=list)

    site_manifest_digest: str | None = None
    spatial_manifest_digest: str | None = None
    policy_digest: str | None = None

    @model_validator(mode="after")
    def _unique_ids(self) -> "CaptureDemandManifest":
        ids = [d.demand_id for d in self.demands]
        if len(set(ids)) != len(ids):
            raise ValueError("identifiants de besoin dupliqués")
        return self

    def outstanding(self, assessments: "DemandAssessmentManifest | list[DemandAssessment]") -> list[CaptureDemand]:
        """Besoins qu'une acquisition peut encore servir.

        `met` est clos parce qu'il est atteint, `unreachable` parce qu'aucune
        acquisition ne le servira : les compter comme ouverts ferait chercher
        indéfiniment ce qui n'existe pas.
        """
        entries = (
            assessments.assessments
            if isinstance(assessments, DemandAssessmentManifest)
            else assessments
        )
        state = {a.demand_id: a.status for a in entries}
        closed = {DemandStatus.MET, DemandStatus.UNREACHABLE}
        return [d for d in self.demands if state.get(d.demand_id, DemandStatus.OPEN) not in closed]


class DemandAssessmentManifest(BaseModel):
    """L'état de tous les besoins, sur **un** corpus.

    Une liste nue laissait deux évaluations d'un même besoin s'écraser dans un
    dictionnaire, et permettait de mêler des états mesurés sur des corpus
    différents — « couvert » aurait alors désigné des instants distincts.
    """

    model_config = ConfigDict(extra="forbid")

    hotel_id: str
    corpus_digest: str = Field(min_length=1)
    demand_digest: str = Field(min_length=1)
    assessed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    assessments: list[DemandAssessment] = Field(default_factory=list)

    @model_validator(mode="after")
    def _one_assessment_per_demand_on_one_corpus(self) -> "DemandAssessmentManifest":
        ids = [a.demand_id for a in self.assessments]
        duplicated = sorted({i for i in ids if ids.count(i) > 1})
        if duplicated:
            raise ValueError(
                f"besoins évalués deux fois : {duplicated} — le dernier lu "
                "aurait silencieusement effacé l'autre"
            )
        divergent = sorted(
            {a.corpus_digest for a in self.assessments if a.corpus_digest != self.corpus_digest}
        )
        if divergent:
            raise ValueError(
                f"états mesurés sur d'autres corpus : {divergent} — un état de "
                "couverture ne se compose pas d'instants différents"
            )
        return self

    def bind(self, demands: CaptureDemandManifest) -> list[str]:
        """Confronte l'état au manifeste de besoins qu'il prétend décrire."""
        problems: list[str] = []
        if demands.hotel_id != self.hotel_id:
            problems.append(
                f"état de {self.hotel_id!r} confronté aux besoins de {demands.hotel_id!r}"
            )
        known = {d.demand_id for d in demands.demands}
        for assessment in self.assessments:
            if assessment.demand_id not in known:
                problems.append(f"état d'un besoin inconnu : {assessment.demand_id!r}")
        return problems


#: Secteurs qu'une cible ne peut pas désigner. `unknown` n'est pas un
#: objectif ; `context` et `transition` ont leur propre `TargetKind`, et les
#: accepter comme secteurs ferait exister deux vocabulaires pour la même chose.
_NON_TARGET_SECTORS = frozenset(
    {ViewSector.UNKNOWN.value, ViewSector.CONTEXT.value, ViewSector.TRANSITION.value}
)


def validate_targets(
    manifest: CaptureDemandManifest,
    site_object_ids: set[str],
    corridor_ids: set[str] | None = None,
    forbidden_zone_ids: set[str] | None = None,
) -> list[str]:
    """Confronte chaque référence au vocabulaire officiel et au site réel.

    Un registre **absent** et un registre **vide** ne disent pas la même
    chose : le premier signifie qu'on ne peut pas valider, le second que rien
    n'existe et donc que toute référence est fausse. Les confondre — par un
    simple `and corridors` — laissait passer sans contrôle toutes les
    références de corridor.
    """
    sectors = {s.value for s in ViewSector} - _NON_TARGET_SECTORS
    problems: list[str] = []

    for demand in manifest.demands:
        ref, kind = demand.target_ref, demand.target_kind

        if kind is TargetKind.SITE_OBJECT and ref not in site_object_ids:
            problems.append(f"{demand.demand_id} : objet de site inconnu {ref!r}")
        elif kind is TargetKind.VIEW_SECTOR and ref not in sectors:
            problems.append(
                f"{demand.demand_id} : secteur inconnu ou non ciblable {ref!r} ; "
                f"attendu l'un de {sorted(sectors)}"
            )
        elif kind in (TargetKind.CONTEXT_CORRIDOR, TargetKind.TRANSITION):
            if corridor_ids is None:
                problems.append(
                    f"{demand.demand_id} : cible {ref!r} invérifiable — aucun "
                    "registre de corridors fourni"
                )
            elif ref not in corridor_ids:
                problems.append(f"{demand.demand_id} : corridor inconnu {ref!r}")

        for zone in demand.forbidden_zone_refs:
            if forbidden_zone_ids is None:
                problems.append(
                    f"{demand.demand_id} : zone interdite {zone!r} invérifiable — "
                    "aucun registre de zones fourni"
                )
            elif zone not in forbidden_zone_ids:
                problems.append(f"{demand.demand_id} : zone interdite inconnue {zone!r}")

    return problems


def bind_evaluations(
    candidates: CandidateManifest, demands: CaptureDemandManifest
) -> list[str]:
    """Ferme la relation entre évaluations et besoins.

    Une évaluation qui cite un besoin inexistant, ou qui lui prête une autre
    intention, décrit un objectif que personne n'a formulé.
    """
    by_id = {d.demand_id: d for d in demands.demands}
    problems: list[str] = []

    if candidates.hotel_id != demands.hotel_id:
        problems.append(
            f"candidats de {candidates.hotel_id!r} confrontés aux besoins de "
            f"{demands.hotel_id!r}"
        )

    for evaluation in candidates.evaluations:
        demand = by_id.get(evaluation.demand_id)
        if demand is None:
            problems.append(
                f"{evaluation.candidate_id} : évalué pour un besoin inconnu "
                f"{evaluation.demand_id!r}"
            )
            continue
        if evaluation.intent is not demand.intent:
            problems.append(
                f"{evaluation.candidate_id}/{evaluation.demand_id} : intention "
                f"{evaluation.intent.value!r} ≠ {demand.intent.value!r} du besoin"
            )
    return problems


def bind_plan(
    plan: AcquisitionPlan,
    candidates: CandidateManifest,
    demands: CaptureDemandManifest,
) -> list[str]:
    """Ferme la relation entre le plan, les candidats et les besoins.

    Un plan est une promesse de dépense : chacun de ses éléments doit désigner
    un candidat qui existe, un besoin qui existe, une évaluation qui ne l'a pas
    rejeté, et une résolution que le fournisseur propose.
    """
    known_candidates = {c.candidate_id: c for c in candidates.candidates}
    known_demands = {d.demand_id for d in demands.demands}
    eligible: dict[str, set[str]] = {}
    for evaluation in candidates.evaluations:
        if evaluation.eligibility is not Eligibility.REJECTED:
            eligible.setdefault(evaluation.candidate_id, set()).add(evaluation.demand_id)

    problems: list[str] = []
    if plan.hotel_id != candidates.hotel_id:
        problems.append(
            f"plan de {plan.hotel_id!r} sur des candidats de {candidates.hotel_id!r}"
        )

    planned = {a.candidate_id for a in plan.acquisitions}
    for item in plan.acquisitions:
        candidate = known_candidates.get(item.candidate_id)
        if candidate is None:
            problems.append(f"{item.candidate_id} : au plan sans candidat correspondant")
            continue

        served = eligible.get(item.candidate_id, set())
        if not served:
            problems.append(
                f"{item.candidate_id} : retenu sans aucune évaluation favorable"
            )

        unknown = [d for d in item.serves_demands if d not in known_demands]
        if unknown:
            problems.append(f"{item.candidate_id} : besoins inconnus {sorted(unknown)}")

        not_eligible = [
            d for d in item.serves_demands if d in known_demands and d not in served
        ]
        if not_eligible:
            problems.append(
                f"{item.candidate_id} : prétend servir {sorted(not_eligible)}, "
                "dont l'évaluation l'a écarté"
            )

        # C'est la résolution **traduite** qu'on confronte : le plan parle
        # « 256 », le fournisseur « 256x256 », et comparer les deux
        # vocabulaires faisait refuser un plan parfaitement exécutable.
        # `provider_resolution` absente signifie qu'aucune traduction n'a eu
        # lieu — on retombe alors sur ce que le plan demande, faute de mieux.
        asked = item.provider_resolution or item.resolution
        if candidate.available_resolutions and asked not in candidate.available_resolutions:
            problems.append(
                f"{item.candidate_id} : résolution {asked!r} indisponible ; "
                f"le fournisseur propose {candidate.available_resolutions}"
            )

        missing_overlap = [c for c in item.overlap_with if c not in planned]
        if missing_overlap:
            problems.append(
                f"{item.candidate_id} : recouvrement annoncé avec {sorted(missing_overlap)}, "
                "absent(s) du plan"
            )

    return problems


# --- candidats ---------------------------------------------------------------


class ProjectionSupport(StrEnum):
    """Ce que le modèle de projection sait faire de cette optique.

    « Inconnu » et « connu mais non supporté » ne se corrigent pas de la même
    façon : le premier appelle une source de métadonnées, le second une
    validation du modèle. Les confondre — en mettant `None` partout — perdait
    l'information la plus utile : on **sait** que ces vues font 134,2°.
    """

    #: Optique dans le domaine où le cadrage rectiligne a été validé.
    SUPPORTED = "supported"

    #: Champ observé au-delà de ce que le modèle sait projeter. La vue est
    #: utilisable, son cadrage n'est pas calculable ainsi.
    UNSUPPORTED_FOV = "unsupported_fov"

    #: La source ne publie pas de quoi le déduire.
    UNKNOWN_INTRINSICS = "unknown_intrinsics"

    #: Sphérique : « cadrer » n'a pas de sens avant qu'un cap et une ouverture
    #: soient choisis.
    PANORAMIC_REQUIRES_EXTRACTION = "panoramic_requires_extraction"


class CaptureCandidate(BaseModel):
    """Une prise de vue possible : métadonnées fournisseur et caméra, rien de plus.

    Pas de fichier, pas d'empreinte, pas de rôle — et pas davantage de verdict :
    ce qu'elle vaut dépend du besoin, et se dit dans `CandidateEvaluation`.
    """

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    source: str

    #: Identifiant **stable** chez le fournisseur.
    provider_id: str

    #: Position interrogée et position réellement rendue. Street View rabat la
    #: requête sur le panorama le plus proche, parfois ailleurs ; ne garder que
    #: l'une des deux rend la sélection irreproductible.
    queried_lat: float | None = Field(default=None, ge=-90, le=90)
    queried_lon: float | None = Field(default=None, ge=-180, le=180)
    camera_lat: float | None = Field(default=None, ge=-90, le=90)
    camera_lon: float | None = Field(default=None, ge=-180, le=180)

    #: Cap rapporté par le fournisseur, et cap qu'il recalcule. Conserver les
    #: deux : le second est meilleur, le premier est la mesure.
    original_heading_deg: float | None = Field(default=None, ge=0, lt=360)
    computed_heading_deg: float | None = Field(default=None, ge=0, lt=360)
    heading_is_measured: bool = True

    #: Cadrage demandé pour une extraction panoramique : une vue Street View
    #: n'existe qu'au moment où on la cadre.
    requested_heading_deg: float | None = Field(default=None, ge=0, lt=360)
    #: Cadrage **exploitable** : renseigné seulement quand le modèle sait le
    #: projeter. Le plafond de 120° dit ce que le modèle a validé, non ce que
    #: l'optique vaut.
    requested_fov_deg: float | None = Field(default=None, gt=0, le=120)

    #: Champ horizontal **observé**, jusqu'à 360°. Une valeur hors du domaine
    #: supporté reste une mesure : la taire ferait passer un ultra-grand-angle
    #: pour une caméra sans métadonnées.
    observed_horizontal_fov_deg: float | None = Field(default=None, gt=0, le=360)

    projection_support: ProjectionSupport = ProjectionSupport.UNKNOWN_INTRINSICS

    #: Pourquoi ce statut, en clair.
    projection_note: str = ""
    requested_pitch_deg: float | None = Field(default=None, ge=-90, le=90)

    sequence_id: str | None = None
    panorama_id: str | None = None
    camera_type: str | None = None

    #: Dimensions **annoncées par le fournisseur**, à ne pas confondre avec
    #: celles du fichier acquis : elles seront mesurées, pas recopiées.
    advertised_width: int | None = Field(default=None, gt=0)
    advertised_height: int | None = Field(default=None, gt=0)

    captured_at: datetime | None = None
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0)

    #: Résolutions disponibles, par nom. Aucune adresse : elle se reconstruit
    #: au téléchargement à partir de `request_spec`.
    available_resolutions: list[str] = Field(default_factory=list)

    #: Paramètres non secrets nécessaires au résolveur d'URL.
    request_spec: dict[str, str] = Field(default_factory=dict)

    #: Preuve fournisseur que la vue est extérieure. Sans elle, l'asset restera
    #: `unknown` : déclarer `exterior` par défaut serait une mesure inventée.
    outdoor_evidence: str | None = None

    @model_validator(mode="after")
    def _no_urls_are_persisted(self) -> "CaptureCandidate":
        # Une URL de CDN expire, une URL signée porte la clé d'API : ni l'une
        # ni l'autre n'a sa place dans un manifeste conservé.
        for key, value in self.request_spec.items():
            text = str(value)
            if "://" in text or text.lower().startswith("www."):
                raise ValueError(
                    f"candidat {self.candidate_id!r} : {key!r} contient une URL ; "
                    "l'adresse se reconstruit au téléchargement"
                )
            if any(secret in key.lower() for secret in ("key", "token", "signature")):
                raise ValueError(
                    f"candidat {self.candidate_id!r} : {key!r} ressemble à un secret"
                )
        return self


class CandidateGeometry(BaseModel):
    """Ce que la géométrie dit d'un candidat **pour une cible donnée**.

    Tout est calculé sur des métadonnées : rien n'est mesuré sur l'image, qui
    n'a pas été acquise. Ces valeurs portent donc des espérances, et sont
    nommées comme telles.
    """

    model_config = ConfigDict(extra="forbid")

    distance_m: float | None = Field(default=None, ge=0)

    #: Intervalle angulaire occupé par l'empreinte depuis la caméra. Sur un
    #: bâtiment oblique, il ne se déduit ni du centroïde ni de la boîte.
    angular_span_deg: float | None = Field(default=None, ge=0, le=360)
    target_offset_deg: float | None = Field(default=None, ge=0, le=180)

    #: Taille projetée espérée : le critère décisif, la distance n'étant qu'un
    #: garde-fou extérieur. Non bornées à 1 et nommées `unclipped` : une cible
    #: plus large que le champ de vision déborde légitimement du cadre, et
    #: écrêter la mesure effacerait cette information.
    unclipped_width_fraction: float | None = Field(default=None, ge=0.0)
    unclipped_height_fraction: float | None = Field(default=None, ge=0.0)

    #: Ce que l'image contiendra **réellement**, une fois le débordement coupé.
    #: Distincte de la précédente, et il faut les deux : une cible deux fois
    #: plus large que le champ a une largeur non écrêtée de 2,0 — elle est donc
    #: énorme — et une part dans le cadre de 0,5, car la moitié en sort. Les
    #: confondre ferait accepter une vue dont il manque tout un pan.
    clipped_width_fraction: float | None = Field(default=None, ge=0.0, le=1.0)

    #: Part de la silhouette effectivement comprise dans le cadre. Répond à
    #: « la cible entre-t-elle dans l'image ? », que la taille apparente ne dit
    #: pas : une cible immense et à moitié hors champ paraît excellente sur la
    #: seule largeur.
    in_frame_fraction: float | None = Field(default=None, ge=0.0, le=1.0)

    #: Dimensions attendues de la cible dans l'image, en pixels. Deux mesures
    #: plutôt qu'une aire : « 40 000 pixels » ne dit pas si la façade fait
    #: 200×200 ou 800×50.
    expected_width_px: int | None = Field(default=None, ge=0)
    expected_height_px: int | None = Field(default=None, ge=0)

    #: Part de la silhouette utile non masquée, estimée par plusieurs rayons —
    #: jamais par le seul rayon vers le point le plus proche.
    visible_fraction: float | None = Field(default=None, ge=0.0, le=1.0)

    #: Obstacle de hauteur inconnue : le risque est signalé, il n'est pas
    #: transformé en certitude.
    occlusion_risk: bool = False
    occluded_by: list[str] = Field(default_factory=list)

    #: Contrôle 2,5D mené avec MNT et MNS. Sans lui, la visibilité reste plane,
    #: donc optimiste.
    used_elevation: bool = False
    elevation_provenance: str | None = None

    @model_validator(mode="after")
    def _elevation_is_stated_or_absent(self) -> "CandidateGeometry":
        # Un contrôle 2,5D sans provenance ne se rejoue pas ; une provenance
        # sans contrôle laisse croire qu'il a eu lieu.
        if self.used_elevation and not (self.elevation_provenance or "").strip():
            raise ValueError(
                "contrôle d'élévation déclaré sans provenance : on ne saurait "
                "pas de quel MNT ni de quel MNS il procède"
            )
        if self.elevation_provenance and not self.used_elevation:
            raise ValueError(
                "provenance d'élévation sans contrôle d'élévation"
            )
        return self

    #: Secteur du vocabulaire officiel, jamais une chaîne libre.
    view_sector: ViewSector | None = None

    #: Demi-ouverture effectivement appliquée pour juger le secteur. Inscrite
    #: à la mesure : un seuil qui ne figure pas dans ce qu'il a produit ne peut
    #: pas être confronté, et le modifier périmerait silencieusement candidats,
    #: évaluations et plans — sans toucher aux artefacts LiDAR, qui ne le lisent
    #: pas.
    sector_half_width_deg: float | None = Field(default=None, gt=0, le=180)

    #: Zones interdites aux plans rapprochés que ce candidat traverse. Le
    #: schéma les validait sans que rien ne les fasse agir.
    forbidden_zones_entered: list[str] = Field(default_factory=list)

    #: La caméra regarde-t-elle depuis un côté que le besoin n'accepte pas ?
    #: Une vue excellente prise de l'arrière ne montre pas la façade avant, et
    #: sa distance n'y change rien : le secteur se juge sur la position de
    #: l'observateur, pas sur la qualité de la vue.
    wrong_sector: bool = False

    #: Identifiant de la voie sur laquelle se trouve la caméra. `on_road`
    #: laissait croire à un booléen alors que le champ porte une référence.
    road_ref: str | None = None


class CandidateEvaluation(BaseModel):
    """Ce qu'un candidat vaut **pour un besoin**, et pourquoi.

    Un candidat peut cadrer la façade avant, manquer l'entrée et documenter la
    voie d'accès : trois verdicts pour une même image. Les réduire à un seul
    obligeait à choisir lequel taire.
    """

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    demand_id: str
    intent: CaptureIntent
    eligibility: Eligibility = Eligibility.ELIGIBLE
    geometry: CandidateGeometry = Field(default_factory=CandidateGeometry)

    #: Obligatoire dès qu'il y a rejet : un candidat écarté sans raison
    #: n'apprend rien à la recherche suivante.
    rejection_reason: str | None = None

    @model_validator(mode="after")
    def _rejection_needs_a_reason(self) -> "CandidateEvaluation":
        rejected = self.eligibility is Eligibility.REJECTED
        if rejected and not (self.rejection_reason or "").strip():
            raise ValueError(
                f"évaluation {self.candidate_id!r}/{self.demand_id!r} : rejet sans motif"
            )
        if not rejected and self.rejection_reason:
            raise ValueError(
                f"évaluation {self.candidate_id!r}/{self.demand_id!r} : motif de "
                "rejet sans rejet"
            )
        return self


class SourceRequestCounts(BaseModel):
    """Appels réellement émis vers une source, par étage.

    Distincts des candidats rendus : « 25 » pouvait signifier 25 appels ou 25
    images, et les deux nombres n'ont ni le même coût ni le même sens.
    """

    model_config = ConfigDict(extra="forbid")

    coarse_search: int = Field(default=0, ge=0)
    metadata_enrichment: int = Field(default=0, ge=0)
    sequence_expansion: int = Field(default=0, ge=0)

    @property
    def total(self) -> int:
        return self.coarse_search + self.metadata_enrichment + self.sequence_expansion

    def as_dict(self) -> dict:
        return {
            "coarse_search": self.coarse_search,
            "metadata_enrichment": self.metadata_enrichment,
            "sequence_expansion": self.sequence_expansion,
            "total": self.total,
        }


class SourceCandidateCounts(BaseModel):
    """Effectifs de candidats à chaque étage, pour une source.

    Sans eux, « zéro candidat arrière » ne distingue pas une zone vide d'une
    zone pleine de vues mal orientées.
    """

    model_config = ConfigDict(extra="forbid")

    returned: int = Field(default=0, ge=0)
    unique: int = Field(default=0, ge=0)
    enriched: int = Field(default=0, ge=0)
    expanded: int = Field(default=0, ge=0)
    recommended: int = Field(default=0, ge=0)
    rejected: int = Field(default=0, ge=0)


class RecommendationLevel(StrEnum):
    """Jusqu'où va ce qu'une recommandation autorise.

    `recommended_for_plan` confondait trois choses. Une vue dont on ignore la
    cible ou l'orientation pouvait entrer directement dans une acquisition
    complète, alors qu'on ne savait même pas ce qu'elle montre.
    """

    #: Vaut un appel de métadonnées supplémentaire, rien de plus.
    ENRICHMENT = "recommended_for_enrichment"

    #: Vaut un aperçu : ce qu'elle montre demande vérification humaine ou
    #: mesurée avant tout engagement.
    PREVIEW = "recommended_for_preview"

    #: Position **et** orientation établies, cible propre résolue : le plan
    #: peut l'envisager pour une acquisition complète. Il reste seul à décider.
    FULL_ACQUISITION = "eligible_for_full_acquisition"


class DemandRecommendation(BaseModel):
    """Ce que la recherche autorise, pour **un** couple candidat/besoin.

    Le niveau ne qualifie pas une image dans l'absolu : la même vue peut être
    pleinement acquérable pour documenter un stationnement et bornée à l'aperçu
    pour une façade dont la taille projetée n'est pas mesurée.
    """

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    demand_id: str

    #: Enum fermé, non une chaîne libre : un niveau inconnu — `banana` — était
    #: accepté, et rien en aval ne l'aurait reconnu comme une preview.
    level: RecommendationLevel

    #: Pourquoi ce niveau, et non un autre. **Obligatoire** : une autorisation
    #: sans motif ne se conteste pas, et c'est précisément ce qu'un relecteur
    #: doit pouvoir faire. `min_length` seul laissait passer une chaîne
    #: d'espaces, qui a la forme d'une explication sans en être une.
    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def _reason_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("motif vide : une autorisation muette ne se conteste pas")
        return value

    #: Exigences du besoin qu'aucune mesure de découverte n'établit.
    unmeasured_requirements: list[str] = Field(default_factory=list)


class DiscoveryMode(StrEnum):
    """Une découverte porte-t-elle sur tous les besoins, ou sur quelques-uns ?

    La distinction est décisive pour ce qui suit : un corpus rassemblé pour un
    seul besoin ne dit **rien** des autres. Le prendre pour un manifeste
    courant ferait lire l'absence de vues de façade comme un constat, alors
    qu'aucune façade n'a été cherchée.
    """

    #: Tous les besoins ouverts du manifeste canonique.
    FULL = "full"

    #: Un sous-ensemble explicitement nommé.
    TARGETED = "targeted"


class DiscoveryScope(BaseModel):
    """Sur quoi cette découverte a porté — et sur quoi elle ne dit rien.

    Sans portée inscrite, deux manifestes de contenus incomparables se
    ressembleraient : rien ne distinguerait « aucune vue trouvée pour ces sept
    besoins » de « un seul besoin a été cherché ».
    """

    model_config = ConfigDict(extra="forbid")

    mode: DiscoveryMode = DiscoveryMode.FULL

    #: Besoins réellement interrogés. Vide en mode `full` : la liste ferait
    #: doublon avec le manifeste, et deux sources de vérité divergeraient.
    demand_ids: tuple[str, ...] = ()

    #: Empreinte du manifeste de besoins **complet**, celui qui a servi à
    #: valider les identifiants. Une découverte ciblée reste rattachée aux
    #: besoins canoniques, sinon elle définirait son propre objectif.
    demand_manifest_digest: str = ""

    #: Corridor employé pour cadrer la recherche, quand le besoin en désigne un.
    corridor_ref: str = ""

    @model_validator(mode="after")
    def _a_targeted_scope_names_its_demands(self) -> "DiscoveryScope":
        if self.mode is DiscoveryMode.TARGETED and not self.demand_ids:
            raise ValueError(
                "portée ciblée sans besoin nommé : elle ne se distinguerait "
                "pas d'une découverte complète, et son manifeste partiel "
                "passerait pour un corpus entier"
            )
        if self.mode is DiscoveryMode.FULL and self.demand_ids:
            raise ValueError(
                "portée complète nommant des besoins : deux sources de vérité "
                "divergeraient sur ce qui a été cherché"
            )
        return self


class CandidateManifest(BaseModel):
    """Tout ce qui a été découvert, et ce qu'on en a pensé, besoin par besoin.

    Ne conserver que les retenus interdirait de comprendre une absence :
    « aucune vue de la façade arrière » ne se distinguerait pas de « aucune
    recherche de ce côté ».
    """

    model_config = ConfigDict(extra="forbid")

    hotel_id: str
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    candidates: list[CaptureCandidate] = Field(default_factory=list)
    evaluations: list[CandidateEvaluation] = Field(default_factory=list)

    #: Nombre d'entités rendues par source interrogée. Conservé pour les
    #: manifestes déjà écrits ; les deux champs ci-dessous le remplacent, en
    #: distinguant ce qu'il confondait.
    queries: dict[str, int] = Field(default_factory=dict)

    #: Appels émis, par source et par étage. « 25 » ne peut plus vouloir dire
    #: à la fois 25 requêtes et 25 images.
    requests_by_source: dict[str, SourceRequestCounts] = Field(default_factory=dict)

    #: Effectifs de candidats, par source et par étage.
    candidates_by_source: dict[str, SourceCandidateCounts] = Field(
        default_factory=dict
    )

    #: Trois niveaux, jamais un seul. `recommended_for_plan` confondait « vaut
    #: un appel de métadonnées », « vaut un aperçu » et « peut être acquise en
    #: entier » : une vue dont on ignorait la cible ou l'orientation pouvait
    #: entrer directement en acquisition complète.
    #:
    #: Recommander n'est jamais décider : `AcquisitionPlan` reste seul à
    #: trancher, et un candidat non recommandé demeure au manifeste — le
    #: retirer effacerait la trace de ce qui a été vu puis écarté.
    #: **L'autorité** : ce que la recherche autorise, couple par couple. Une
    #: vue recommandée pour le stationnement ne l'est pas pour la façade — et
    #: un niveau porté par le seul `candidate_id` laissait une autorisation
    #: obtenue pour un besoin en couvrir un autre qui ne l'avait jamais
    #: recommandée.
    recommendations: list["DemandRecommendation"] = Field(default_factory=list)

    recommended_for_enrichment: list[str] = Field(default_factory=list)
    recommended_for_preview: list[str] = Field(default_factory=list)
    eligible_for_full_acquisition: list[str] = Field(default_factory=list)

    #: Filiation avec le rapport de recherche qui l'a produit.
    adaptive_search_run_id: str | None = None
    adaptive_search_report_digest: str | None = None

    demand_digest: str | None = None
    policy_digest: str | None = None

    #: Ce que cette découverte a couvert. Par défaut complète : les manifestes
    #: écrits avant ce champ portaient bien sur tous les besoins.
    scope: DiscoveryScope = Field(default_factory=DiscoveryScope)

    @model_validator(mode="after")
    def _consistent(self) -> "CandidateManifest":
        ids = [c.candidate_id for c in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("identifiants de candidat dupliqués")

        # Recommander ce qui n'est pas au manifeste rendrait la recommandation
        # invérifiable : le plan citerait une vue qu'aucun rapport ne décrit.
        recommended = (
            set(self.recommended_for_enrichment)
            | set(self.recommended_for_preview)
            | set(self.eligible_for_full_acquisition)
        )
        unknown = sorted(recommended - set(ids))
        if unknown:
            raise ValueError(
                f"candidats recommandés absents du manifeste : {unknown}"
            )

        # Les niveaux se **succèdent** : ce qui est éligible à l'acquisition
        # complète ne peut pas être simultanément borné à l'aperçu. Les laisser
        # se chevaucher rendrait le plan libre de choisir la lecture qui
        # l'arrange.
        both = sorted(
            set(self.eligible_for_full_acquisition)
            & set(self.recommended_for_preview)
        )
        if both:
            raise ValueError(
                f"candidats à la fois preview-only et pleinement éligibles : {both}"
            )

        known = set(ids)

        # Les couples sont l'autorité ; les listes n'en sont qu'un résumé. Si
        # elles s'en écartent, un lecteur pressé lirait une autorisation que
        # personne n'a prononcée.
        # Une seule recommandation par couple. Deux niveaux contradictoires
        # pour un même couple laissaient l'**ordre du fichier** décider entre
        # 256 et 2048 : `_recommendation_levels` construit un dictionnaire, et
        # la dernière entrée l'emportait.
        seen_pairs: set[tuple[str, str]] = set()
        recommended_pairs: set[tuple[str, str]] = set()
        for entry in self.recommendations:
            if entry.candidate_id not in known:
                raise ValueError(
                    f"recommandation d'un candidat absent : {entry.candidate_id!r}"
                )
            pair = (entry.candidate_id, entry.demand_id)
            if pair in seen_pairs:
                raise ValueError(
                    f"deux recommandations pour le couple {pair} : le niveau "
                    "dépendrait de l'ordre du fichier"
                )
            seen_pairs.add(pair)
            recommended_pairs.add(pair)
        if self.recommendations:
            # Un même candidat peut être autorisé à deux niveaux, pour deux
            # besoins distincts : pleinement acquérable pour documenter un
            # stationnement, borné à l'aperçu pour une façade non mesurée. Le
            # résumé retient alors **le plus prudent** — promouvoir le fichier
            # entier parce qu'un seul besoin s'en contentait perdrait la
            # réserve de l'autre.
            rank = {
                "recommended_for_enrichment": 0,
                "recommended_for_preview": 1,
                "eligible_for_full_acquisition": 2,
            }
            safest: dict[str, int] = {}
            for entry in self.recommendations:
                if entry.level not in rank:
                    continue
                safest[entry.candidate_id] = min(
                    safest.get(entry.candidate_id, rank[entry.level]),
                    rank[entry.level],
                )
            derived: dict[str, set] = {name: set() for name in rank}
            reverse = {value: name for name, value in rank.items()}
            for candidate_id, value in safest.items():
                derived[reverse[value]].add(candidate_id)

            for name, expected in derived.items():
                if set(getattr(self, name)) != expected:
                    raise ValueError(
                        f"{name} ne résume pas les recommandations par besoin : "
                        f"{sorted(set(getattr(self, name)) ^ expected)}"
                    )

        pairs = set()
        rejected_pairs: set[tuple[str, str]] = set()
        for evaluation in self.evaluations:
            if evaluation.candidate_id not in known:
                raise ValueError(
                    f"évaluation d'un candidat absent : {evaluation.candidate_id!r}"
                )
            pair = (evaluation.candidate_id, evaluation.demand_id)
            if pair in pairs:
                raise ValueError(f"deux évaluations pour le couple {pair}")
            pairs.add(pair)
            if evaluation.eligibility is Eligibility.REJECTED:
                rejected_pairs.add(pair)

        # Recommander ce qu'on a rejeté est une contradiction : l'évaluation
        # dit « cette vue ne sert pas ce besoin », la recommandation dit
        # l'inverse, et le plan croirait la seconde.
        contradicted = sorted(recommended_pairs & rejected_pairs)
        if contradicted:
            raise ValueError(
                f"recommandation(s) portant sur une évaluation rejetée : "
                f"{contradicted}"
            )

        # Une recommandation sans évaluation ne s'appuie sur rien : le plan
        # citerait une autorisation qu'aucune mesure ne soutient.
        unsupported = sorted(recommended_pairs - pairs)
        if unsupported:
            raise ValueError(
                f"recommandation(s) sans évaluation correspondante : {unsupported}"
            )
        return self

    def eligible_for(self, demand_id: str) -> list[CandidateEvaluation]:
        return [
            e
            for e in self.evaluations
            if e.demand_id == demand_id and e.eligibility is not Eligibility.REJECTED
        ]

    def rejections_by_reason(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for evaluation in self.evaluations:
            if evaluation.eligibility is Eligibility.REJECTED:
                reason = evaluation.rejection_reason or "sans motif"
                counts[reason] = counts.get(reason, 0) + 1
        return dict(sorted(counts.items()))


# --- plan --------------------------------------------------------------------


class PlannedAcquisition(BaseModel):
    """Un candidat retenu, et ce qu'on s'engage à en télécharger.

    Une seule acquisition peut servir plusieurs besoins — cadrer la façade et
    documenter la voie d'accès. Ne porter qu'une intention obligeait à taire
    l'autre, et le contexte redevenait un sous-produit.
    """

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    intents: list[CaptureIntent] = Field(min_length=1)

    #: Celle qui classe l'acquisition quand il faut trancher.
    primary_intent: CaptureIntent | None = None

    #: Résolution demandée, dans le vocabulaire du **plan**. Une vérification
    #: de contenu se fait en 256 ou 1024 ; la pleine résolution ne vient
    #: qu'après sélection.
    resolution: str = "2048"

    #: Ce que le fournisseur comprendra — `thumb_256`, `256x256`. Renseignée à
    #: la résolution des requêtes : sans elle, le plan et le téléchargement
    #: parlaient deux langues sans que rien ne les confronte.
    provider_resolution: str | None = None

    #: Empreinte de la requête qui sera émise. C'est **elle** que le
    #: consentement verrouille : consentir à un candidat laisserait redéfinir
    #: ensuite ce qui est demandé pour lui.
    request_digest: str | None = None

    #: Niveau prononcé **par besoin**. Le fichier a une résolution unique, mais
    #: sert parfois deux besoins à deux niveaux : la façade en aperçu, le
    #: corridor pleinement. Les fondre perdrait la réserve du premier.
    demand_levels: dict[str, str] = Field(default_factory=dict)

    #: Volume attendu. `None` signifie **inconnu**, jamais zéro : compter une
    #: taille absente comme nulle annonçait un total « exact » faux.
    expected_bytes: int | None = Field(default=None, ge=0)

    #: Pourquoi ce candidat plutôt qu'un autre, et ce qu'il apporte au graphe :
    #: sa valeur propre et son recouvrement avec ses voisins.
    selection_rationale: str = Field(min_length=1)
    overlap_with: list[str] = Field(default_factory=list)

    #: Obligatoire : télécharger sans savoir quel besoin on sert, c'est
    #: revenir à collecter d'abord et justifier ensuite.
    serves_demands: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _intents_are_distinct(self) -> "PlannedAcquisition":
        if len(set(self.intents)) != len(self.intents):
            raise ValueError(
                f"acquisition {self.candidate_id!r} : intentions dupliquées"
            )
        if self.primary_intent and self.primary_intent not in self.intents:
            raise ValueError(
                f"acquisition {self.candidate_id!r} : intention principale "
                f"{self.primary_intent.value!r} absente de ses intentions"
            )
        return self


class AcquisitionPlan(BaseModel):
    """Ce qui sera téléchargé, arrêté avant tout appel réseau.

    Un plan exécutable porte **toutes** ses empreintes. Les rendre facultatives
    permettait de déclarer courant un plan sans lien avec le site, et donc
    d'acquérir des images choisies pour un autre état.
    """

    model_config = ConfigDict(extra="forbid")

    plan_id: str
    hotel_id: str
    status: PlanStatus = PlanStatus.DRAFT
    planned_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    acquisitions: list[PlannedAcquisition] = Field(default_factory=list)

    candidate_manifest_digest: str | None = None
    demand_digest: str | None = None
    policy_digest: str | None = None
    site_manifest_digest: str | None = None
    spatial_manifest_digest: str | None = None
    corpus_digest: str | None = None
    road_geometry_digest: str | None = None
    obstacle_geometry_digest: str | None = None

    #: Empreintes des **facettes** de politique que ce plan lit réellement.
    #: Distinctes de `policy_digest`, qui bouge dès qu'un seuil change où que
    #: ce soit : un réglage de terrain n'a rien à voir avec une sélection
    #: photographique, et le laisser périmer le plan obligerait à tout refaire.
    policy_dependency_digests: dict[str, str] = Field(default_factory=dict)

    #: Volume et statut **inscrits**, non seulement calculés. Un consentement
    #: doit reposer sur un artefact autonome : ces valeurs étaient des
    #: propriétés, donc absentes du document publié — un lecteur du fichier
    #: seul ne savait pas que le volume était complet.
    published_known_bytes: int | None = Field(default=None, ge=0)
    published_unknown_size_items: int | None = Field(default=None, ge=0)
    published_volume_status: VolumeStatus | None = None

    #: --- ce à quoi le consentement s'attache ------------------------------
    #: Plafond accepté, en octets. Le vérifier au moment d'acquérir ne suffit
    #: pas : un plan dont les acquisitions changeraient après coup porterait un
    #: consentement donné pour autre chose.
    consented_max_bytes: int | None = Field(default=None, ge=0)

    #: Empreintes des requêtes au moment du consentement. Elles ancrent
    #: l'accord à **ces** demandes-là : sans elles, réécrire une résolution
    #: après l'accord téléchargerait autre chose sous le même consentement.
    consented_request_digests: list[str] = Field(default_factory=list)

    #: Plan mesuré dont le volume a été montré. Consentir sans lui reviendrait
    #: à accepter un total qu'aucune mesure n'a établi.
    consented_from_plan_id: str | None = None

    #: Version du contrat de téléchargement en vigueur à l'accord. Ce que
    #: « télécharger » garantit fait partie de ce qui est consenti.
    consented_download_contract_version: int | None = Field(default=None, ge=1)

    @property
    def known_bytes(self) -> int:
        return sum(a.expected_bytes for a in self.acquisitions if a.expected_bytes is not None)

    @property
    def unknown_size_items(self) -> list[str]:
        return [a.candidate_id for a in self.acquisitions if a.expected_bytes is None]

    @property
    def volume_status(self) -> VolumeStatus:
        if not self.acquisitions:
            return VolumeStatus.EXACT
        if not self.unknown_size_items:
            return VolumeStatus.EXACT
        if len(self.unknown_size_items) == len(self.acquisitions):
            return VolumeStatus.UNKNOWN
        return VolumeStatus.PARTIAL

    def missing_digests(self) -> list[str]:
        return [name for name in REQUIRED_PLAN_DIGESTS if not getattr(self, name)]

    @model_validator(mode="after")
    def _published_volume_agrees(self) -> "AcquisitionPlan":
        """Les valeurs inscrites doivent dire la vérité sur les acquisitions.

        Un plan annonçant « exact » avec des tailles manquantes ferait consentir
        à un total dont une part est inconnue — précisément ce que le
        consentement exact interdit.
        """
        if self.published_known_bytes is not None:
            if self.published_known_bytes != self.known_bytes:
                raise ValueError(
                    f"volume publié {self.published_known_bytes} contre "
                    f"{self.known_bytes} réellement portés par les acquisitions"
                )
        if self.published_unknown_size_items is not None:
            if self.published_unknown_size_items != len(self.unknown_size_items):
                raise ValueError(
                    f"{self.published_unknown_size_items} taille(s) inconnue(s) "
                    f"publiée(s) contre {len(self.unknown_size_items)} réelle(s)"
                )
        if self.published_volume_status is not None:
            if self.published_volume_status is not self.volume_status:
                raise ValueError(
                    f"statut publié « {self.published_volume_status.value} » "
                    f"contre « {self.volume_status.value} » : "
                    "un plan ne peut pas se dire complet quand il ne l'est pas"
                )
        return self

    @model_validator(mode="after")
    def _executable_is_complete(self) -> "AcquisitionPlan":
        ids = [a.candidate_id for a in self.acquisitions]
        if len(set(ids)) != len(ids):
            raise ValueError("un candidat figure deux fois au plan")

        if self.status is PlanStatus.EXECUTABLE:
            missing = self.missing_digests()
            if missing:
                raise ValueError(
                    f"plan {self.plan_id!r} exécutable sans empreinte(s) : {missing} — "
                    "un plan qu'on ne peut pas rattacher à un état ne s'acquiert pas"
                )
            if not self.acquisitions:
                raise ValueError(f"plan {self.plan_id!r} exécutable et vide")
        return self


# --- provenance ---------------------------------------------------------------


class AcquisitionProvenance(BaseModel):
    """D'où vient réellement un fichier acquis.

    `source_url_or_id` recevait l'URL temporaire du CDN : l'identité durable de
    l'asset dépendait d'un lien qui expire, et les métadonnées du collecteur
    étaient perdues à la conversion.
    """

    model_config = ConfigDict(extra="forbid")

    provider_id: str
    plan_id: str
    plan_digest: str
    candidate_id: str
    intents: list[CaptureIntent] = Field(min_length=1)
    primary_intent: CaptureIntent | None = None

    queried_lat: float | None = Field(default=None, ge=-90, le=90)
    queried_lon: float | None = Field(default=None, ge=-180, le=180)
    returned_lat: float | None = Field(default=None, ge=-90, le=90)
    returned_lon: float | None = Field(default=None, ge=-180, le=180)

    original_heading_deg: float | None = Field(default=None, ge=0, lt=360)
    computed_heading_deg: float | None = Field(default=None, ge=0, lt=360)
    requested_heading_deg: float | None = Field(default=None, ge=0, lt=360)
    #: Cadrage **exploitable** : renseigné seulement quand le modèle sait le
    #: projeter. Le plafond de 120° dit ce que le modèle a validé, non ce que
    #: l'optique vaut.
    requested_fov_deg: float | None = Field(default=None, gt=0, le=120)

    #: Champ horizontal **observé**, jusqu'à 360°. Une valeur hors du domaine
    #: supporté reste une mesure : la taire ferait passer un ultra-grand-angle
    #: pour une caméra sans métadonnées.
    observed_horizontal_fov_deg: float | None = Field(default=None, gt=0, le=360)

    projection_support: ProjectionSupport = ProjectionSupport.UNKNOWN_INTRINSICS

    #: Pourquoi ce statut, en clair.
    projection_note: str = ""
    requested_pitch_deg: float | None = Field(default=None, ge=-90, le=90)

    sequence_id: str | None = None
    panorama_id: str | None = None
    camera_type: str | None = None
    #: Résolution **fournisseur** de ce qui a été téléchargé — `thumb_2048`,
    #: `640x640`. Distincte de ce que le plan demandait : inscrire le
    #: vocabulaire du plan décrirait un fichier qui n'est pas celui du disque.
    resolution: str | None = None

    #: Champ horizontal observé de l'optique, et ce que le modèle en fait.
    observed_horizontal_fov_deg: float | None = Field(default=None, gt=0, le=360)
    projection_support: ProjectionSupport = ProjectionSupport.UNKNOWN_INTRINSICS

    #: Besoins que cette acquisition devait servir, et à quel niveau chacun.
    #: Sans eux, un fichier téléchargé ne se rattachait à aucune exigence : la
    #: preview arrivait sans qu'on sache ce qu'elle venait vérifier, ni pour
    #: quel besoin le verdict comptera.
    serves_demands: list[str] = Field(default_factory=list)
    demand_levels: dict[str, str] = Field(default_factory=dict)

    #: Ce que le plan demandait, dans son propre vocabulaire.
    requested_resolution: str | None = None

    #: Empreinte de la requête effectivement émise. C'est elle que le
    #: consentement verrouille : changer la résolution change l'empreinte.
    request_digest: str | None = None
    acquired_at: datetime | None = None

    #: Dimensions annoncées par le fournisseur, conservées **à côté** de celles
    #: mesurées sur le fichier : les faire coïncider d'office masquerait un
    #: rendu tronqué ou redimensionné.
    advertised_width: int | None = Field(default=None, gt=0)
    advertised_height: int | None = Field(default=None, gt=0)

    #: Répertoire d'exécution où le fichier a été publié : les acquisitions ne
    #: se mélangent pas aux fichiers historiques.
    run_id: str | None = None


# --- identité ------------------------------------------------------------------


class IdentityStrategy(StrEnum):
    """Comment une source nomme ses prises de vue."""

    #: L'image existe telle quelle chez le fournisseur.
    PROVIDER_IMAGE = "provider_image"
    #: Le fournisseur rend une sphère : c'est le cadrage qui fait l'image.
    PANORAMA_FRAMING = "panorama_framing"


#: Stratégie par source. Une source inconnue ne se rabat pas silencieusement
#: sur `provider_image` : deux cadrages y porteraient le même identifiant, et
#: l'un écraserait l'autre.
IDENTITY_STRATEGIES: dict[str, IdentityStrategy] = {
    "mapillary": IdentityStrategy.PROVIDER_IMAGE,
    "commons": IdentityStrategy.PROVIDER_IMAGE,
    "flickr": IdentityStrategy.PROVIDER_IMAGE,
    "website": IdentityStrategy.PROVIDER_IMAGE,
    "places": IdentityStrategy.PROVIDER_IMAGE,
    "tripadvisor": IdentityStrategy.PROVIDER_IMAGE,
    "street_view": IdentityStrategy.PANORAMA_FRAMING,
}


def capture_identity(
    source: str,
    provider_id: str,
    *,
    heading_deg: float | None = None,
    fov_deg: float | None = None,
    pitch_deg: float | None = None,
    size: str | None = None,
    strategy: IdentityStrategy | None = None,
) -> str:
    """Identifiant durable d'une prise de vue, selon la nature de la source.

    Pour Mapillary, l'identifiant d'image suffit : elle existe telle quelle, et
    y adjoindre un cadrage fabriquerait une identité qui ne correspond à rien.
    Pour Street View, le panorama n'est pas une image mais une sphère : deux
    cadrages sont deux prises de vue, et les nommer pareil en écraserait une.

    Une source inconnue exige une stratégie déclarée. Le défaut implicite
    aurait été le plus dangereux des deux.
    """
    chosen = strategy or IDENTITY_STRATEGIES.get(source)
    if chosen is None:
        raise ValueError(
            f"source inconnue {source!r} : déclarez sa stratégie d'identité, "
            "elle ne peut pas être devinée"
        )

    framing = {"heading_deg": heading_deg, "fov_deg": fov_deg,
               "pitch_deg": pitch_deg, "size": size}

    if chosen is IdentityStrategy.PROVIDER_IMAGE:
        given = sorted(k for k, v in framing.items() if v is not None)
        if given:
            raise ValueError(
                f"{source} nomme ses images par identifiant : le cadrage "
                f"{given} n'entre pas dans leur identité"
            )
        return f"{source}-{provider_id}"

    missing = sorted(k for k, v in framing.items() if v is None)
    if missing:
        raise ValueError(
            f"{source} rend un panorama : son identité exige le cadrage complet, "
            f"or {missing} manque(nt)"
        )

    spec = "|".join(
        f"{value:.3f}" if isinstance(value, float) else str(value)
        for value in (heading_deg, fov_deg, pitch_deg, size)
    )
    digest = hashlib.sha256(f"{provider_id}|{spec}".encode("utf-8")).hexdigest()[:12]
    return f"{source}-{provider_id}-{digest}"


def validate_recommendation_demands(
    candidates: "CandidateManifest", demands: "CaptureDemandManifest",
) -> list[str]:
    """Confronte les recommandations aux besoins réellement déclarés.

    Vérification **inter-manifestes** : le validateur du manifeste de candidats
    ne connaît pas les besoins, et prétendre le contraire y ferait entrer une
    dépendance qu'il n'a pas. Elle se fait donc à la liaison, là où les deux
    documents se rencontrent.

    Une recommandation visant un besoin inexistant est invérifiable : le plan
    citerait une autorisation portant sur une exigence que personne n'a
    formulée.
    """
    declared = {demand.demand_id for demand in demands.demands}
    unknown = sorted(
        {
            entry.demand_id
            for entry in candidates.recommendations
            if entry.demand_id not in declared
        }
    )
    return [
        f"recommandation portant sur un besoin inconnu : {demand_id!r}"
        for demand_id in unknown
    ]
