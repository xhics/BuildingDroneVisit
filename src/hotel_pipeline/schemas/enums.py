"""Énumérations du domaine.

Les états sont des enums et non des chaînes libres : un objet `unresolved` ne
doit pas pouvoir franchir un Gate en silence (plan directeur §4).
"""

from __future__ import annotations

from enum import StrEnum


class ObjectState(StrEnum):
    """État d'un objet critique (plan directeur §4)."""

    CONFIRMED = "confirmed"
    INFERRED = "inferred"
    CONFLICTED = "conflicted"
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
    """Version de l'entrée principale.

    La rénovation approuvée en 2024 interdit de fusionner les deux périodes
    sans contrôle (plan directeur §3). Non déductible visuellement sans
    référence datée : c'est un verrou humain.
    """

    PRE_2024 = "pre_2024"
    POST_2024 = "post_2024"
    UNKNOWN = "unknown"


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
