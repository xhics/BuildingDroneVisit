"""Schémas Pydantic du pipeline (plan directeur §4, §9, §18)."""

from .assets import PRODUCTION_RIGHTS, Asset, AssetManifest, GpsPoint
from .critical_objects import (
    EXCLUDED_OBJECTS,
    REQUIRED_OBJECTS,
    CriticalObject,
    CriticalObjectRegistry,
    HumanCorrection,
    SpatialRelation,
)
from .enums import (
    AssetCategory,
    EntranceVersion,
    ExteriorInterior,
    ObjectState,
    Phase1Status,
    PropertyMatchStatus,
    Rights,
    RouterPath,
)
from .project import BlockedState, ProjectManifest, StepRecord

__all__ = [
    "PRODUCTION_RIGHTS",
    "REQUIRED_OBJECTS",
    "EXCLUDED_OBJECTS",
    "Asset",
    "AssetCategory",
    "AssetManifest",
    "BlockedState",
    "CriticalObject",
    "CriticalObjectRegistry",
    "EntranceVersion",
    "ExteriorInterior",
    "GpsPoint",
    "HumanCorrection",
    "ObjectState",
    "Phase1Status",
    "ProjectManifest",
    "PropertyMatchStatus",
    "Rights",
    "RouterPath",
    "SpatialRelation",
    "StepRecord",
]
