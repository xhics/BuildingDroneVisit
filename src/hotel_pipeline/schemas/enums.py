"""Énumérations du domaine.

Les états sont des enums et non des chaînes libres : un objet `unresolved` ne
doit pas pouvoir franchir un Gate en silence (plan directeur §4).
"""

from __future__ import annotations

from enum import StrEnum


class ObjectState(StrEnum):
    """État d'un objet critique (plan directeur §4).

    `STALE` désigne un objet dont la décision reposait sur un artefact
    depuis remplacé. Sa décision antérieure et son motif sont conservés : la
    qualification n'est pas fausse, elle porte sur une production périmée.
    """

    CONFIRMED = "confirmed"
    INFERRED = "inferred"
    CONFLICTED = "conflicted"
    STALE = "stale"
    UNRESOLVED = "unresolved"


class Rights(StrEnum):
    """Droits d'usage d'un asset (plan directeur §9)."""

    OWNED = "owned"                    # fourni ou autorisé par l'hôtel
    LICENSED = "licensed"              # licence explicite couvrant l'usage
    OPEN_DATA = "open_data"            # licence ouverte vérifiée
    PUBLIC_UNCLEARED = "public_uncleared"  # indexée publiquement, droits non établis
    UNKNOWN = "unknown"


class AssetCategory(StrEnum):
    FACADE = "facade"
    ENTRANCE = "entrance"
    AERIAL = "aerial"
    PARKING = "parking"
    INTERIOR = "interior"
    SIGN = "sign"
    OTHER = "other"


class ExteriorInterior(StrEnum):
    EXTERIOR = "exterior"
    INTERIOR = "interior"
    UNKNOWN = "unknown"


class EntranceVersion(StrEnum):
    """Version de l'entrée, relative aux travaux déclarés au profil.

    Mélanger deux périodes sans contrôle est interdit (plan directeur §3).
    Non déductible visuellement sans référence datée : c'est un verrou humain.
    """

    BEFORE_RENOVATION = "before_renovation"
    AFTER_RENOVATION = "after_renovation"
    UNKNOWN = "unknown"


class Subject(StrEnum):
    """Ce que montre une image (Lot 1B §4).

    Multi-étiquette : une même photo peut montrer le bâtiment, le stationnement
    et l'enseigne. La catégorie unique ne doit plus porter toute la décision.
    """

    BUILDING = "building"
    ENTRANCE = "entrance"
    SIGN = "sign"
    PARKING = "parking"
    ROOF = "roof"
    GROUNDS = "grounds"
    ROAD = "road"
    NEIGHBOUR = "neighbour"
    INTERIOR = "interior"
    OTHER = "other"


class ViewSector(StrEnum):
    """Secteur du bâtiment observé (Lot 1B §4, §11)."""

    FRONT = "front"
    LEFT = "left"
    RIGHT = "right"
    REAR = "rear"
    ROOF = "roof"
    FRONT_LEFT_CORNER = "front_left_corner"
    FRONT_RIGHT_CORNER = "front_right_corner"
    REAR_LEFT_CORNER = "rear_left_corner"
    REAR_RIGHT_CORNER = "rear_right_corner"
    TRANSITION = "transition"
    CONTEXT = "context"
    UNKNOWN = "unknown"


class CaptureType(StrEnum):
    """Nature de la prise de vue (Lot 1B §4)."""

    STREET_IMAGERY = "street_imagery"
    TRAVELER = "traveler"
    PROMOTIONAL = "promotional"
    SOCIAL = "social"
    AERIAL_OBLIQUE = "aerial_oblique"
    ORTHOPHOTO = "orthophoto"
    LIDAR = "lidar"
    MUNICIPAL_DOCUMENT = "municipal_document"
    HOTEL_CAPTURE = "hotel_capture"
    UNKNOWN = "unknown"


class ReconstructionRole(StrEnum):
    """Usage prévu dans la reconstruction (Lot 1B §4).

    `REFERENCE_ONLY` est le défaut : un asset ne devient une source de
    géométrie que sur décision explicite, jamais par omission.
    """

    PHOTO_GEOMETRY = "photo_geometry"
    TEXTURE_REFERENCE = "texture_reference"
    GEO_GEOMETRY = "geo_geometry"
    CONTEXT_LOCK = "context_lock"
    IDENTITY_EVIDENCE = "identity_evidence"
    REFERENCE_ONLY = "reference_only"
    REJECT = "reject"


class TemporalStatus(StrEnum):
    """Position temporelle vis-à-vis des derniers travaux déclarés.

    Générique : les valeurs se rapportent à un `RenovationEvent` du profil,
    dont la date vit dans les données. La version précédente gravait
    `pre_2024` / `post_2024` dans le type — la rénovation d'un établissement
    précis promue au rang de vocabulaire.
    """

    BEFORE_EVENT = "before_event"
    AFTER_EVENT = "after_event"
    CURRENT_CONFIRMED = "current_confirmed"
    HISTORICAL = "historical"
    UNKNOWN = "unknown"


class ClusterRole(StrEnum):
    """Rôle d'un fichier au sein de son point de vue (Lot 1B §5, niveau 4).

    La déduplication ne supprime rien : elle hiérarchise. Le recouvrement utile
    entre images successives est ce qui rendra un SfM possible, et l'écraser
    serait contre-productif.
    """

    CANONICAL = "canonical"
    OVERLAP = "overlap"
    INACTIVE = "inactive"


class ReviewDecision(StrEnum):
    """Arbitrage humain sur la visibilité de la cible (Lot 1B §6).

    Distinct de `ReviewStatus`, qui décrit l'état du flux de revue. Ici il
    s'agit du **contenu** de la décision, qui prime sur toute déduction et
    n'est jamais recalculée.
    """

    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    UNRESOLVED = "unresolved"


class GeometrySuitability(StrEnum):
    """Ce que l'image apporte à la **structure**, distinct de ce qu'elle montre.

    Trois questions se posaient jusqu'ici en une seule, et se répondaient donc
    mal :

    ```text
    est-ce le bon bâtiment ?          → ReviewDecision
    y a-t-il assez de structure ?     → GeometrySuitability
    est-ce la meilleure du lot ?      → ClusterRole
    ```

    Confirmer l'identité d'une vue lointaine la promouvait immédiatement en
    porteuse de géométrie : la cible y est bien reconnaissable, sans que la
    façade y soit exploitable. Le critère n'est pas la distance — un téléobjectif
    à 117 m vaut mieux qu'un grand-angle à 40 — mais la part du cadre occupée
    par la cible, ses dimensions en pixels, la façade non masquée, la netteté
    sur la zone utile et les lignes raccordables.
    """

    #: Personne n'a encore répondu. Une vue non évaluée ne porte pas de
    #: géométrie : l'inverse ferait de l'absence d'examen une approbation.
    UNASSESSED = "unassessed"

    #: Structure exploitable : la vue peut fonder la reconstruction.
    PRIMARY = "primary"

    #: Utile pour relier le bâtiment à son environnement ou pour
    #: l'enregistrement, sans apporter de structure propre.
    AUXILIARY = "auxiliary"

    #: Rien d'exploitable pour la géométrie, quelle que soit l'identité.
    INSUFFICIENT = "insufficient"


class ReviewStatus(StrEnum):
    """Statut de revue d'une qualification (Lot 1B §4, §6)."""

    AUTOMATIC_ACCEPTED = "automatic_accepted"
    HUMAN_ACCEPTED = "human_accepted"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"


class PropertyMatchStatus(StrEnum):
    """L'asset représente-t-il bien la propriété visée ?"""

    MATCH = "match"
    MISMATCH = "mismatch"
    UNCERTAIN = "uncertain"


class RouterPath(StrEnum):
    """Routes de reconstruction (plan directeur §12)."""

    PATH_A_OPEN_3D = "path_a_open_3d"
    PATH_B_PHOTO_FIRST = "path_b_photo_first"
    PATH_C_GEO_FIRST = "path_c_geo_first"
    PATH_D_HYBRID = "path_d_hybrid"
    REJECT = "reject"


class Phase1Status(StrEnum):
    """Décisions finales possibles (plan directeur §23)."""

    ENVIRONMENT_3D_READY = "ENVIRONMENT_3D_READY"
    NEEDS_AUTHORIZED_CAPTURE = "NEEDS_AUTHORIZED_CAPTURE"
    NEEDS_MANUAL_CORRECTION = "NEEDS_MANUAL_CORRECTION"
    GEO_FIRST_PROXY_ONLY = "GEO_FIRST_PROXY_ONLY"
    REJECTED_PROPERTY_AMBIGUOUS = "REJECTED_PROPERTY_AMBIGUOUS"
    REJECTED_RIGHTS_INSUFFICIENT = "REJECTED_RIGHTS_INSUFFICIENT"
    REJECTED_DATA_INSUFFICIENT = "REJECTED_DATA_INSUFFICIENT"
