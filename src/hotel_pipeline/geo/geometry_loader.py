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
    AccessStatus,
    CaptureFeasibilityAssessment,
    CaptureGeometryManifest,
    FeasibilityDetail,
    FeasibilityStatus,
    GEOGRAPHIC_CRS,
    GeometryResolutionStatus,
    GeometryRole,
    PROJECTED_CRS,
    ReachabilityStatus,
    RoadAccessGraph,
    RoadSegment,
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


# ---------------------------------------------------------------------------
# P4.1 — Graphe d'accessibilité routière
# ---------------------------------------------------------------------------


def build_road_access_graph(workspace, spatial_reference=None) -> RoadAccessGraph:  # noqa: ANN001
    """Construit le `RoadAccessGraph` depuis le manifeste géométrique du site.

    Les tronçons proviennent des `ResolvedGeometry` de rôle ACCESS_ROAD /
    ROAD_CANDIDATE (avec leur WKT projeté) et des `RoadCorridor` adjacents.
    Les évaluations de faisabilité sont laissées vides : elles sont produites
    par l'étape P4 une fois la première reconstruction mesurée.
    """
    from ..workspace import Workspace

    if not isinstance(workspace, Workspace):
        raise TypeError("build_road_access_graph attend un Workspace")

    segments: list[RoadSegment] = []
    geometry_path = workspace.path("06_geo", "capture_geometry.json")
    manifest: CaptureGeometryManifest | None = None
    if geometry_path.is_file():
        try:
            manifest, _ = load_capture_geometry(geometry_path, spatial_reference)
        except Exception as exc:  # noqa: BLE001 — manifeste absent ou illisible
            log.warning("graphe routier : manifeste géométrique illisible : %s", exc)
            manifest = None

    if manifest is not None:
        road_roles = {GeometryRole.ACCESS_ROAD, GeometryRole.ROAD_CANDIDATE}
        for geom in manifest.geometries:
            if (
                geom.role in road_roles
                and geom.resolution_status is GeometryResolutionStatus.RESOLVED
                and geom.projected_wkt
            ):
                segments.append(RoadSegment(
                    feature_id=geom.feature_id,
                    geometry_wkt=geom.projected_wkt,
                    access_status=AccessStatus.UNKNOWN,
                    reachability_status=ReachabilityStatus.UNKNOWN,
                ))
        for corridor in manifest.corridors:
            segments.append(RoadSegment(
                feature_id=corridor.feature_id,
                geometry_wkt="",
                access_status=corridor.access_status,
                camera_candidate=corridor.admissible_for_building,
                streetview_candidate=corridor.admissible_for_building,
            ))
        # Déduplique par feature_id (la géométrie résolue prime sur le corridor).
        by_id: dict[str, RoadSegment] = {}
        for seg in segments:
            by_id.setdefault(seg.feature_id, seg)
        segments = list(by_id.values())

    return RoadAccessGraph(hotel_id=workspace.hotel_id, road_segments=segments)


def build_capture_feasibility_assessment(
    target_id: str,
    *,
    remote_public: FeasibilityDetail | None = None,
    owner_assisted: FeasibilityDetail | None = None,
    professional_onsite: FeasibilityDetail | None = None,
    physically_impossible: FeasibilityDetail | None = None,
    evidence: list[str] | None = None,
) -> CaptureFeasibilityAssessment:
    """Assemble une `CaptureFeasibilityAssessment` et déduit le statut global.

    La chaîne de priorité : physiquement impossible > introuvable à distance >
    capture propriétaire requise > faisable. La preuve d'impossibilité prime
    toujours (le pipeline s'arrête honnêtement sur un MUST_SHOW impossible).
    """
    remote_public = remote_public or FeasibilityDetail()
    owner_assisted = owner_assisted or FeasibilityDetail()
    professional_onsite = professional_onsite or FeasibilityDetail()
    physically_impossible = physically_impossible or FeasibilityDetail()
    evidence = evidence or []

    if physically_impossible.status is FeasibilityStatus.INFEASIBLE_PROVEN:
        status = FeasibilityStatus.INFEASIBLE_PROVEN
    elif remote_public.status is FeasibilityStatus.FEASIBLE:
        status = FeasibilityStatus.FEASIBLE
    elif remote_public.status is FeasibilityStatus.NOT_FOUND_REMOTELY:
        status = FeasibilityStatus.NOT_FOUND_REMOTELY
    elif owner_assisted.status is FeasibilityStatus.OWNER_CAPTURE_REQUIRED:
        status = FeasibilityStatus.OWNER_CAPTURE_REQUIRED
    elif professional_onsite.status is FeasibilityStatus.FEASIBLE:
        status = FeasibilityStatus.FEASIBLE
    else:
        status = FeasibilityStatus.UNKNOWN

    return CaptureFeasibilityAssessment(
        target_id=target_id,
        status=status,
        remote_public=remote_public,
        owner_assisted=owner_assisted,
        professional_onsite=professional_onsite,
        physically_impossible=physically_impossible,
        evidence=evidence,
    )
