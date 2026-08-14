"""Lecture des manifestes géométriques, anciens et nouveaux.

Le manifeste du pilote a été écrit avant que le référentiel de travail soit une
donnée : il ne porte ni `schema_version`, ni `working_crs`, ni
`spatial_context_digest`. Le nouveau schéma les exige, et c'est voulu.

Deux tentations à écarter, toutes deux fausses :

- lui donner `schema_version="1.0.0"` par défaut lui prêterait des garanties
  qu'il n'a jamais eues, et un fichier antérieur deviendrait indiscernable d'un
  fichier conforme ;
- le réécrire au passage modifierait un artefact publié, dont l'empreinte est
  citée par une vingtaine de rapports.

Un fichier sans `schema_version` est donc lu comme **legacy** : son référentiel
implicite est vérifié contre le contexte spatial courant, la liaison se fait en
mémoire, la lecture est autorisée, et rien n'est réécrit.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..logging import get_logger
from ..schemas.geometry import (
    GEOGRAPHIC_CRS,
    PROJECTED_CRS,
    CaptureGeometryManifest,
)

log = get_logger("geometry-loader")

#: Version des manifestes portant leur référentiel. Tout fichier qui ne déclare
#: aucune version lui est **antérieur**, par construction.
CURRENT_SCHEMA_VERSION = "2.0.0"

#: Référentiel implicite des manifestes antérieurs. Il n'est pas appliqué : il
#: est *vérifié* contre le contexte courant, et un désaccord arrête la lecture.
LEGACY_WORKING_CRS = PROJECTED_CRS


class LegacyManifestRefused(RuntimeError):
    """Le fichier antérieur ne peut pas être rattaché au contexte courant."""


def is_legacy(payload: dict) -> bool:
    """Un manifeste sans version déclarée est antérieur, sans exception."""
    return not payload.get("schema_version")


def load_capture_geometry(path: Path, spatial_reference) -> tuple[CaptureGeometryManifest, bool]:  # noqa: ANN001
    """Charge un manifeste géométrique, ancien ou nouveau.

    Rend le manifeste et un drapeau disant s'il a fallu le rattacher. Le
    manifeste rendu est **en mémoire** : le fichier n'est jamais réécrit.
    """
    payload = json.loads(path.read_text("utf-8"))

    if not is_legacy(payload):
        return CaptureGeometryManifest.model_validate(payload), False

    return bind_legacy(payload, spatial_reference), True


def bind_legacy(payload: dict, spatial_reference) -> CaptureGeometryManifest:  # noqa: ANN001
    """Rattache un manifeste antérieur au contexte spatial courant.

    Le rattachement n'est pas une conversion de complaisance : il **vérifie**
    que le référentiel implicite du fichier est bien celui du site aujourd'hui.
    Un manifeste québécois relu sous un contexte lyonnais est refusé — c'est
    exactement le cas qu'un défaut silencieux laisserait passer.
    """
    if spatial_reference is None:
        raise LegacyManifestRefused(
            "manifeste antérieur : aucun contexte spatial pour le rattacher. "
            "Lancez « geo reference », qui résout le référentiel du site."
        )

    working = getattr(spatial_reference, "working_crs", None)
    if working != LEGACY_WORKING_CRS:
        raise LegacyManifestRefused(
            f"manifeste antérieur écrit en {LEGACY_WORKING_CRS}, contexte "
            f"courant en {working!r} : les formes projetées qu'il contient ne "
            "sont pas celles de ce site. Rien n'est lu, rien n'est réécrit."
        )

    declared = {
        geometry.get("projected_crs")
        for geometry in payload.get("geometries", [])
        if geometry.get("resolution_status") == "resolved"
    }
    unexpected = declared - {LEGACY_WORKING_CRS}
    if unexpected:
        raise LegacyManifestRefused(
            f"manifeste antérieur portant des référentiels inattendus : "
            f"{sorted(unexpected)}"
        )

    bound = dict(payload)
    bound["schema_version"] = "1.0.0-legacy"
    bound["source_crs"] = payload.get("source_crs") or GEOGRAPHIC_CRS
    bound["working_crs"] = LEGACY_WORKING_CRS
    bound["spatial_context_digest"] = spatial_reference.context_digest()

    log.info(
        "manifeste antérieur rattaché en mémoire : %s, contexte %s — "
        "aucun fichier réécrit",
        LEGACY_WORKING_CRS, bound["spatial_context_digest"],
    )
    return CaptureGeometryManifest.model_validate(bound)
