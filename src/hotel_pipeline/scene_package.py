"""Paquet 3D hybride, local et vérifiable pour un consommateur vidéo.

Le module ne lance ni SfM ni fournisseur vidéo. Il exporte la géométrie
disponible et publie séparément le verdict Phase 1 qui explique pourquoi ce
paquet ne vaut pas ``ENVIRONMENT_3D_READY``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from shapely import wkt
from shapely.geometry import Polygon
from shapely.ops import triangulate

from .context import PipelineContext
from .provenance import digest_of
from .schemas.enums import ObjectState, ReconstructionRole, Rights
from .schemas.scene import (
    CameraPose,
    EvidenceClass,
    GateCheck,
    GateState,
    PackageFile,
    Phase1Status,
    Phase1Verdict,
    ScenePackage,
    VirtualCameraPath,
)

SCENE_EXPORTER_VERSION = "hybrid-1.2.1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text("utf-8"))


def _zones_of(constraints: dict, rule: str) -> list[str]:
    """Objets nommés par une règle de contrainte caméra."""
    zones: set[str] = set()
    for row in constraints.get("constraints", []):
        if row.get("rule") != rule:
            continue
        zones.update(
            part for part in str(row.get("zone_ref", "")).split(",") if part
        )
    return sorted(zones)


def _forbidden_claims(constraints: dict) -> list[str]:
    """Ce dont rien ne peut être affirmé : l'existence même manque."""
    return _zones_of(constraints, "do_not_show_as_fact")


def _blind_visual_fields(constraints: dict) -> list[str]:
    """Objets réels que rien n'a photographiés — la caméra les contourne.

    Distincts des revendications interdites : ils existent, et les taire
    reviendrait à nier une preuve. C'est leur **apparence** qui manque, donc
    leur cadrage qui doit être évité.
    """
    return _zones_of(constraints, "avoid_framing_no_observed_appearance")


def _active_artifact(site, role: str):  # noqa: ANN001, ANN201
    found = [row for row in site.artifacts if row.is_active and row.role == role]
    if len(found) != 1:
        raise ValueError(
            f"attendu exactement un artefact actif {role!r}, obtenu {len(found)}"
        )
    artifact = found[0]
    path = Path(artifact.path)
    if not path.is_file():
        raise FileNotFoundError(f"artefact actif absent : {path}")
    actual = _sha256(path)
    if actual != artifact.sha256:
        raise ValueError(
            f"empreinte divergente pour {artifact.artifact_id}: "
            f"{actual} != {artifact.sha256}"
        )
    return artifact, path


def _target_polygon(capture: dict) -> Polygon:
    targets = [
        row for row in capture.get("geometries", [])
        if row.get("role") == "target_building"
        and row.get("resolution_status") == "resolved"
    ]
    if len(targets) != 1:
        raise ValueError("la scène exige exactement un bâtiment cible résolu")
    geometry = wkt.loads(targets[0]["projected_wkt"])
    if not isinstance(geometry, Polygon) or not geometry.is_valid:
        raise ValueError("empreinte projetée du bâtiment invalide")
    return geometry


def _extruded_obj(polygon: Polygon, ground_z: float, roof_z: float) -> str:
    """Volume fermé en coordonnées locales, explicitement classé proxy."""
    if polygon.interiors:
        raise ValueError(
            "l'exporteur proxy ne sait pas fermer une empreinte avec cour intérieure"
        )
    origin = polygon.centroid
    ring = list(polygon.exterior.coords)[:-1]
    if len(ring) < 3:
        raise ValueError("empreinte trop courte pour former un volume")
    lines = [
        "# BuildingDroneVisit hybrid proxy mesh",
        "mtllib environment.mtl",
        "o building_volume_proxy",
    ]
    for x, y in ring:
        lines.append(f"v {x-origin.x:.6f} {y-origin.y:.6f} {ground_z:.6f}")
    for x, y in ring:
        lines.append(f"v {x-origin.x:.6f} {y-origin.y:.6f} {roof_z:.6f}")

    lines.extend(["usemtl proxy_facade", "g facade_proxy"])
    count = len(ring)
    for index in range(count):
        nxt = (index + 1) % count
        a, b = index + 1, nxt + 1
        c, d = nxt + 1 + count, index + 1 + count
        lines.append(f"f {a} {b} {c} {d}")

    # Shapely triangule l'enveloppe de Delaunay ; le filtre conserve seulement
    # les triangles dont le centre appartient au polygone concave.
    bottom_by_xy = {
        (round(x, 9), round(y, 9)): idx + 1
        for idx, (x, y) in enumerate(ring)
    }
    top_by_xy = {
        (round(x, 9), round(y, 9)): idx + 1 + count
        for idx, (x, y) in enumerate(ring)
    }
    triangles = [triangle for triangle in triangulate(polygon) if polygon.covers(triangle)]
    if not triangles:
        raise ValueError("triangulation vide pour l'empreinte du bâtiment")
    lines.extend(["usemtl proxy_roof", "g flat_roof_proxy"])
    for triangle in triangles:
        indices: list[int] = []
        for x, y in list(triangle.exterior.coords)[:-1]:
            key = (round(x, 9), round(y, 9))
            if key not in top_by_xy:
                # Une triangulation de ce contour simple ne doit pas inventer
                # de sommet ; si GEOS le faisait, mieux vaut refuser le mesh.
                raise ValueError("triangulation du toit avec sommet non traçable")
            indices.append(top_by_xy[key])
        lines.append("f " + " ".join(str(value) for value in indices))

    # Un toit et quatre murs ne forment pas un volume fermé. Le fond est utile
    # même si la scène le pose normalement sur le DTM : plusieurs importeurs
    # et opérations booléennes exigent un maillage manifold. Les sommets sont
    # inversés pour orienter la normale vers le bas.
    lines.extend(["usemtl proxy_ground", "g ground_proxy"])
    for triangle in triangles:
        indices = [
            bottom_by_xy[(round(x, 9), round(y, 9))]
            for x, y in list(triangle.exterior.coords)[:-1]
        ]
        lines.append("f " + " ".join(str(value) for value in reversed(indices)))
    return "\n".join(lines) + "\n"


#: Façade regardée depuis une position, selon son écart au cap de façade.
#: L'observateur au sud-ouest d'un bâtiment orienté 227,89° voit la façade
#: principale ; à l'opposé, l'arrière.
_FACADE_BY_OFFSET = (
    (45.0, "FACADE_PRIMARY"),
    (135.0, "FACADE_LEFT"),
    (225.0, "FACADE_REAR"),
    (315.0, "FACADE_RIGHT"),
)


def _facade_faced(azimuth_deg: float, front_azimuth_deg: float) -> str:
    """Façade cadrée par une pose placée à cet azimut autour de l'emprise."""
    offset = (azimuth_deg - front_azimuth_deg) % 360.0
    for limit, kind in _FACADE_BY_OFFSET:
        if offset < limit:
            return kind
    return "FACADE_PRIMARY"


def _camera_path(  # noqa: PLR0913
    polygon: Polygon,
    height_m: float,
    fov_deg: float,
    front_azimuth_deg: float | None = None,
    observed_appearance: frozenset[str] = frozenset(),
) -> VirtualCameraPath:
    """Orbite virtuelle de cadrage, jamais une trajectoire de capture terrain.

    Chaque pose déclare la façade qu'elle regarde et si cette surface est un
    **champ visuel mort** — une géométrie dont aucune apparence n'a été
    observée. Les poses aveugles restent dans le chemin : les retirer
    masquerait la lacune au lieu de la déclarer.
    """
    radius_xy = max(
        math.hypot(x - polygon.centroid.x, y - polygon.centroid.y)
        for x, y in polygon.exterior.coords
    )
    # Rayon calculé pour laisser 20 % de marge autour du diamètre apparent.
    # C'est un paramètre de mise en scène, non une mesure du site.
    radius = (radius_xy * 1.2) / math.tan(math.radians(fov_deg / 2.0))
    look_z = height_m * 0.5
    elevation = 18.0
    camera_z = look_z + radius * math.tan(math.radians(elevation))
    poses = []
    aveugles = 0
    for index, azimuth in enumerate(range(0, 360, 30)):
        angle = math.radians(azimuth)
        faces = (
            _facade_faced(float(azimuth), front_azimuth_deg)
            if front_azimuth_deg is not None else None
        )
        # Sans orientation connue, aucune pose n'est déclarée aveugle : on ne
        # sait pas ce qu'elle regarde, et l'affirmer serait une invention.
        blind = bool(faces) and faces not in observed_appearance
        aveugles += int(blind)
        poses.append(
            CameraPose(
                frame=index * 60,
                position_local_m=(
                    round(radius * math.sin(angle), 4),
                    round(radius * math.cos(angle), 4),
                    round(camera_z, 4),
                ),
                look_at_local_m=(0.0, 0.0, round(look_z, 4)),
                azimuth_deg=float(azimuth),
                elevation_deg=elevation,
                distance_m=round(radius, 4),
                fov_horizontal_deg=fov_deg,
                faces=faces,
                blind_field=blind,
            )
        )
    derivation = (
        "12 poses à 30°, rayon = 1.2 × rayon d'emprise / tan(FOV/2); "
        "orbite virtuelle uniquement, sans affirmation d'accès physique"
    )
    if front_azimuth_deg is not None:
        derivation += (
            f"; façade cadrée déduite du cap {front_azimuth_deg:.2f}°, "
            f"{aveugles}/{len(poses)} pose(s) sur un champ visuel mort"
        )
    return VirtualCameraPath(
        path_id="virtual-context-orbit-v1",
        simulation_only=True,
        derivation=derivation,
        poses=poses,
    )


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _copy_atomic(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def _package_file(
    folder: Path, relative: str, role: str, evidence_class: EvidenceClass,
    source_refs: list[str],
) -> PackageFile:
    path = folder / relative
    return PackageFile(
        path=relative,
        sha256=_sha256(path),
        role=role,
        evidence_class=evidence_class,
        source_refs=source_refs,
    )


def _verify_existing(folder: Path, inputs: dict[str, str]) -> None:
    """Un paquet adressé par contenu est immuable et entièrement relu."""
    from .schemas.scene import Phase1Verdict, ScenePackage

    scene_path = folder / "scene.json"
    verdict_path = folder / "phase1_verdict.json"
    scene = ScenePackage.model_validate_json(scene_path.read_text("utf-8"))
    Phase1Verdict.model_validate_json(verdict_path.read_text("utf-8"))
    if scene.input_digests != inputs:
        raise ValueError("paquet existant sous le même identifiant avec d'autres entrées")
    for row in scene.files:
        path = folder / row.path
        if not path.is_file():
            raise FileNotFoundError(f"fichier déclaré absent du paquet : {row.path}")
        actual = _sha256(path)
        if actual != row.sha256:
            raise ValueError(
                f"fichier modifié dans le paquet {row.path}: {actual} != {row.sha256}"
            )


def _publish_pointer(workspace, folder: Path) -> Path:  # noqa: ANN001
    scene = folder / "scene.json"
    return workspace.write_json(
        "08_composite/scene_package_current.json",
        {
            "contract_version": 1,
            "hotel_id": workspace.hotel_id,
            "package_id": folder.name.removeprefix("scene_package_"),
            "manifest": str(scene.relative_to(workspace.root)),
            "manifest_sha256": _sha256(scene),
        },
    )


def _phase1_blocking_reasons(
    coverage: dict, checks: list[GateCheck], *, duplicate_files: int | None,
    asset_count: int, duplicate_robust: bool, exterior_count: int,
    geometry_with_quality: int, geometry_count: int, independent_viewpoints: int,
) -> list[str]:
    """Traduit les gates en motifs sans contredire un gate déjà franchi."""
    reasons = list(coverage.get("blocking_reasons") or [])
    by_gate = {check.gate_id: check for check in checks}
    if by_gate["G1_deduplication"].state is not GateState.PASSED:
        reasons.append(
            f"G1 non établi : {duplicate_files!r}/{asset_count} assets ; "
            f"preuve robuste courante : {duplicate_robust}"
        )
    if by_gate["G2_exterior"].state is not GateState.PASSED:
        reasons.append(f"G2 échoue avec {exterior_count} extérieurs exploitables")
    if by_gate["G3_quality"].state is not GateState.PASSED:
        reasons.append(
            f"G3 non établi : qualité mesurée sur "
            f"{geometry_with_quality}/{geometry_count} porteurs"
        )
    reasons.extend([
        f"G4 échoue avec {independent_viewpoints} point de vue indépendant",
        "G5 absent : aucune reconstruction SfM/LightGlue/pycolmap mesurée",
        "validation Blender non exécutée",
        "revue humaine finale absente",
    ])
    return list(dict.fromkeys(reasons))


def build(workspace) -> dict[str, Path]:  # noqa: ANN001
    """Construit le paquet hybride depuis les seules productions courantes."""
    context, warning = PipelineContext.for_workspace(workspace)
    if warning:
        raise ValueError(warning)
    site = workspace.read_site()
    assets = workspace.read_assets()
    spatial = workspace.read_spatial()
    if site is None or assets is None or spatial is None:
        raise FileNotFoundError("site, assets et manifeste spatial sont obligatoires")

    capture_path = workspace.path("06_geo", "capture_geometry.json")
    coverage_path = workspace.path("coverage", "coverage_report.json")
    constraints_path = workspace.path("coverage", "camera_constraints.json")
    confidence_path = workspace.path("coverage", "zone_confidence.geojson")
    capture = _json(capture_path)
    coverage = _json(coverage_path)
    constraints = _json(constraints_path)
    confidence = _json(confidence_path)
    polygon = _target_polygon(capture)

    dtm, dtm_path = _active_artifact(site, "dtm")
    roof, roof_path = _active_artifact(site, "dsm_roof")
    ndsm, ndsm_path = _active_artifact(site, "ndsm")
    if not (dtm.crs_horizontal == roof.crs_horizontal == ndsm.crs_horizontal):
        raise ValueError("les trois rasters ne partagent pas le même CRS horizontal")
    if not (dtm.crs_vertical == roof.crs_vertical == ndsm.crs_vertical):
        raise ValueError("les trois rasters ne partagent pas le même datum vertical")

    derivation = _json(workspace.path("06_geo", "derivation_report.json"))
    heights = derivation.get("metrics", {}).get("height_statistics", {})
    height_m = heights.get("median_m")
    if not isinstance(height_m, (int, float)) or height_m <= 0:
        raise ValueError("hauteur médiane absente du rapport de dérivation")

    # Le sol local est zéro : l'altitude absolue reste portée par l'origine et
    # le datum, sans injecter deux fois la même translation dans le mesh.
    try:
        import rasterio
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dépend de l'extra geo
        raise RuntimeError("l'extra geo (rasterio) est requis pour exporter la scène") from exc
    with rasterio.open(dtm_path) as dataset:
        values = dataset.read(1, masked=True)
        finite = values.compressed()
        if not len(finite):
            raise ValueError("DTM actif sans altitude exploitable")
        ground_altitude = float(np.median(finite))

    router_name = coverage.get("router", {}).get("file")
    router_path = workspace.path("10_validation", str(router_name))
    if not router_path.is_file():
        raise FileNotFoundError("décision Router citée par la couverture absente")

    inputs = {
        "site_manifest": _sha256(workspace.site_path),
        "asset_manifest": _sha256(workspace.assets_path),
        "spatial_manifest": _sha256(workspace.spatial_path),
        "capture_geometry": _sha256(capture_path),
        "coverage_report": _sha256(coverage_path),
        "camera_constraints": _sha256(constraints_path),
        "zone_confidence": _sha256(confidence_path),
        "router_decision": _sha256(router_path),
        "dtm": dtm.sha256,
        "dsm_roof": roof.sha256,
        "ndsm": ndsm.sha256,
        "scene_export_contract": digest_of(
            {"contract_version": 1, "exporter_version": SCENE_EXPORTER_VERSION}
        ),
    }
    package_id = digest_of(inputs)
    final_folder = workspace.path("08_composite", f"scene_package_{package_id}")
    if (final_folder / "scene.json").is_file():
        _verify_existing(final_folder, inputs)
        pointer = _publish_pointer(workspace, final_folder)
        return {
            "package": final_folder,
            "scene": final_folder / "scene.json",
            "verdict": final_folder / "phase1_verdict.json",
            "current": pointer,
        }
    if final_folder.exists():
        raise ValueError(f"publication partielle déjà présente : {final_folder}")
    folder = workspace.path("08_composite", f".scene_package_{package_id}.staging")
    if folder.exists():
        raise ValueError(f"staging antérieur à examiner avant reprise : {folder}")
    folder.mkdir(parents=True, exist_ok=False)

    obj_path = folder / "environment.obj"
    _write_atomic(obj_path, _extruded_obj(polygon, 0.0, float(height_m)))
    _write_atomic(
        folder / "environment.mtl",
        "newmtl proxy_facade\nKd 0.55 0.35 0.25\n"
        "newmtl proxy_roof\nKd 0.22 0.24 0.28\n"
        "newmtl proxy_ground\nKd 0.16 0.18 0.16\n",
    )
    _copy_atomic(dtm_path, folder / "dtm.tif")
    _copy_atomic(roof_path, folder / "dsm_roof.tif")
    _copy_atomic(ndsm_path, folder / "ndsm.tif")
    _write_atomic(
        folder / "zone_confidence.geojson",
        json.dumps(confidence, indent=2, ensure_ascii=False) + "\n",
    )
    _write_atomic(
        folder / "camera_constraints.json",
        json.dumps(constraints, indent=2, ensure_ascii=False) + "\n",
    )

    # Ce que la carte de confiance déclare réellement observé : une façade sans
    # apparence mesurée reste une forme, et son cadrage un champ visuel mort.
    observed = frozenset(
        feature["properties"]["kind"]
        for feature in confidence.get("features", [])
        if feature.get("properties", {}).get("appearance_coverage") in ("partial", "full")
    )
    path = _camera_path(
        polygon,
        float(height_m),
        float(context.policy.collection.image_fov_deg),
        front_azimuth_deg=getattr(spatial, "front_azimuth_deg", None),
        observed_appearance=observed,
    )
    _write_atomic(
        folder / "camera_path.json",
        json.dumps(path.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
    )
    prompts = {
        "contract_version": 1,
        "real_provider_call_performed": False,
        "generation_goal": "prévisualisation promotionnelle d'un volume hôtelier hybride",
        "required_inputs": ["environment.obj", "camera_path.json", "zone_confidence.geojson"],
        "positive_constraints": [
            "conserver exactement la silhouette et les proportions du mesh fourni",
            "conserver le contexte spatial ; ne déplacer ni routes ni bâtiments voisins",
            "utiliser uniquement les trajectoires virtuelles marquées simulation_only",
        ],
        "negative_constraints": [
            # Deux interdits distincts : ce dont l'existence manque, et ce qui
            # existe sans apparence observée. Les confondre ferait taire une
            # preuve ou, à l'inverse, autoriser une texture inventée.
            (
                "ne rien affirmer de : "
                + ", ".join(_forbidden_claims(constraints))
                + " — existence non établie"
            ),
            (
                "ne pas texturer ni cadrer en gros plan : "
                + ", ".join(_blind_visual_fields(constraints))
                + " — objets réels dont aucune apparence n'a été observée"
            ),
            "ne pas produire de gros plan sur les façades proxy ou les lacunes de toiture",
            "privilégier les poses dont blind_field vaut false dans camera_path.json",
            "ne pas présenter ce paquet comme un relevé ou une reconstruction photo-réaliste",
        ],
        "blind_field_poses": [
            pose.frame for pose in path.poses if pose.blind_field
        ],
    }
    _write_atomic(
        folder / "video_prompts.json",
        json.dumps(prompts, indent=2, ensure_ascii=False) + "\n",
    )
    _write_atomic(
        folder / "blender_import.py",
        "# Exécuter dans Blender: blender --background --python blender_import.py\n"
        "import bpy\nfrom pathlib import Path\n"
        "root = Path(__file__).resolve().parent\n"
        "bpy.ops.wm.obj_import(filepath=str(root / 'environment.obj'))\n"
        "bpy.ops.wm.save_as_mainfile(filepath=str(root / 'environment.blend'))\n",
    )
    _write_atomic(
        folder / "README.md",
        "# Paquet 3D hybride — WelcomINNS Boucherville\n\n"
        "Ce paquet est provider-agnostic et n'a appelé aucun service vidéo. "
        "Il décrit un volume proxy géoréférencé, les rasters LiDAR qualifiés, "
        "une orbite virtuelle et les claims interdits.\n\n"
        "## Utilisation\n\n"
        "- Lire `scene.json` avant tout autre fichier.\n"
        "- Importer `environment.obj` dans un outil 3D, ou exécuter "
        "`blender --background --python blender_import.py`.\n"
        "- Fournir `camera_path.json` et `video_prompts.json` au connecteur "
        "vidéo choisi.\n"
        "- Appliquer impérativement `camera_constraints.json` et "
        "`zone_confidence.geojson`.\n\n"
        "## Limite de statut\n\n"
        "`phase1_verdict.json` conclut `NEEDS_AUTHORIZED_CAPTURE`. Le paquet "
        "n'est ni une reconstruction photoréaliste, ni un relevé, ni une "
        "preuve de l'entrée ou du stationnement actuels.\n",
    )

    production_rights = {Rights.OWNED, Rights.LICENSED, Rights.OPEN_DATA}
    exterior_production = [
        asset for asset in assets.assets
        if asset.exterior_or_interior.value == "exterior"
        and asset.rights in production_rights
        and asset.reconstruction_role is not ReconstructionRole.REJECT
    ]
    geometry_assets = [
        asset for asset in assets.assets
        if asset.reconstruction_role is ReconstructionRole.PHOTO_GEOMETRY
    ]
    geometry_with_quality = [
        asset for asset in geometry_assets if asset.quality_score is not None
    ]
    duplicate_path = workspace.path("01_sources", "duplicate_report.json")
    duplicate_files = None
    duplicate_robust = False
    if duplicate_path.is_file():
        duplicate_files = _json(duplicate_path).get("files")
        from .lot1b_coverage import _robust_dedup_is_current

        duplicate_robust = _robust_dedup_is_current(workspace, context.policy)
    independent_viewpoints = int(
        coverage.get("demands", {}).get("independent_viewpoints", 0)
    )

    checks = [
        GateCheck(
            gate_id="G0_inventory",
            requirement="inventaire brut et métadonnées disponibles",
            state=GateState.PASSED,
            evidence=[f"asset_manifest.json : {len(assets.assets)} assets"],
        ),
        GateCheck(
            gate_id="G1_deduplication",
            requirement="déduplication perceptuelle sur le corpus courant",
            state=(
                GateState.PASSED
                if duplicate_files == len(assets.assets) and duplicate_robust
                else GateState.UNVERIFIED
            ),
            evidence=[
                f"duplicate_report.json : {duplicate_files!r} fichiers ; "
                f"manifeste courant : {len(assets.assets)} ; "
                f"recadrage/filigrane validé : {duplicate_robust}"
            ],
        ),
        GateCheck(
            gate_id="G2_exterior",
            requirement="au moins 15 extérieurs exploitables",
            state=(
                GateState.PASSED
                if len(exterior_production) >= 15
                else GateState.FAILED
            ),
            evidence=[
                f"{len(exterior_production)} extérieurs aux droits de production"
            ],
        ),
        GateCheck(
            gate_id="G3_quality",
            requirement="qualité mesurée sur tous les porteurs de géométrie",
            state=(
                GateState.PASSED
                if geometry_assets and len(geometry_with_quality) == len(geometry_assets)
                else GateState.UNVERIFIED
            ),
            evidence=[
                f"{len(geometry_with_quality)}/{len(geometry_assets)} "
                "porteurs de géométrie avec quality_score"
            ],
        ),
        GateCheck(
            gate_id="G4_diversity",
            requirement="diversité et connectivité probable suffisantes",
            state=GateState.FAILED,
            evidence=[f"{independent_viewpoints} point de vue indépendant"],
        ),
        GateCheck(
            gate_id="G5_sfm",
            requirement="reconstruction SfM sparse réelle mesurée",
            state=GateState.FAILED,
            evidence=["Lot 2 non exécuté ; aucun rapport hloc/LightGlue/pycolmap"],
        ),
        GateCheck(gate_id="inspectable", requirement="environnement 3D inspectable", state=GateState.PASSED, evidence=["environment.obj"]),
        GateCheck(gate_id="alignment", requirement="géoréférencement documenté", state=GateState.PASSED, evidence=[dtm.crs_horizontal, str(dtm.crs_vertical)]),
        GateCheck(gate_id="critical_objects", requirement="objets critiques établis", state=GateState.FAILED, evidence=coverage.get("unresolved_objects") or ["état inconnu"]),
        GateCheck(gate_id="current_entrance", requirement="entrée actuelle distinguée", state=GateState.FAILED, evidence=["ENTRANCE_MAIN_CURRENT unresolved"]),
        GateCheck(gate_id="rights", requirement="sources et droits tracés", state=GateState.PASSED, evidence=["coverage_report.json:rights"]),
        GateCheck(gate_id="route", requirement="route et gates enregistrés", state=GateState.PASSED, evidence=[router_name]),
        GateCheck(gate_id="confidence", requirement="carte de confiance par zone", state=GateState.PASSED, evidence=["zone_confidence.geojson"]),
        GateCheck(gate_id="blender", requirement="chargement Blender reproductible", state=GateState.UNVERIFIED, evidence=["blender_import.py produit; Blender non exécuté"]),
        GateCheck(gate_id="human_review", requirement="revue humaine du statut final", state=GateState.UNVERIFIED, evidence=["aucune approbation finale enregistrée"]),
    ]
    blocking_reasons = _phase1_blocking_reasons(
        coverage, checks, duplicate_files=duplicate_files,
        asset_count=len(assets.assets), duplicate_robust=duplicate_robust,
        exterior_count=len(exterior_production),
        geometry_with_quality=len(geometry_with_quality),
        geometry_count=len(geometry_assets),
        independent_viewpoints=independent_viewpoints,
    )

    verdict = Phase1Verdict(
        hotel_id=workspace.hotel_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=Phase1Status.NEEDS_AUTHORIZED_CAPTURE,
        router_decision_digest=inputs["router_decision"],
        input_digests=inputs,
        checks=checks,
        blocking_reasons=blocking_reasons,
    )
    verdict_path = folder / "phase1_verdict.json"
    _write_atomic(
        verdict_path,
        json.dumps(verdict.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
    )

    files = [
        _package_file(folder, "environment.obj", "building_volume", EvidenceClass.PROXY, ["BUILDING_MAIN", "ndsm"]),
        _package_file(folder, "environment.mtl", "proxy_materials", EvidenceClass.PROXY, ["camera_constraints.json"]),
        _package_file(folder, "dtm.tif", "terrain", EvidenceClass.INFERRED, [dtm.artifact_id]),
        _package_file(folder, "dsm_roof.tif", "roof_surface", EvidenceClass.MEASURED, [roof.artifact_id]),
        _package_file(folder, "ndsm.tif", "normalised_height", EvidenceClass.INFERRED, [ndsm.artifact_id]),
        _package_file(folder, "zone_confidence.geojson", "zone_confidence", EvidenceClass.INFERRED, ["coverage/zone_confidence.geojson"]),
        _package_file(folder, "camera_constraints.json", "camera_constraints", EvidenceClass.INFERRED, ["coverage/camera_constraints.json"]),
        _package_file(folder, "camera_path.json", "virtual_camera_path", EvidenceClass.PROXY, ["BUILDING_MAIN", "pipeline_policy.collection.image_fov_deg"]),
        _package_file(folder, "video_prompts.json", "video_prompt_contract", EvidenceClass.PROXY, ["camera_constraints.json", "zone_confidence.geojson"]),
        _package_file(folder, "blender_import.py", "blender_import", EvidenceClass.PROXY, ["environment.obj"]),
        _package_file(folder, "README.md", "integration_guide", EvidenceClass.PROXY, ["scene.json", "phase1_verdict.json"]),
        _package_file(folder, "phase1_verdict.json", "phase1_verdict", EvidenceClass.INFERRED, [router_name, "coverage/coverage_report.json"]),
    ]
    rights = coverage.get("rights", {}).get("by_status", {})
    manifest = ScenePackage(
        hotel_id=workspace.hotel_id,
        package_id=package_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        status="hybrid_proxy_package",
        horizontal_crs=dtm.crs_horizontal,
        vertical_datum=str(dtm.crs_vertical),
        local_origin_projected=(
            round(polygon.centroid.x, 4),
            round(polygon.centroid.y, 4),
            round(ground_altitude, 4),
        ),
        input_digests=inputs,
        phase1_verdict="phase1_verdict.json",
        files=files,
        camera_paths=[path],
        rights_summary={str(key): int(value) for key, value in rights.items()},
        # Lus des contraintes publiées, jamais figés : une liste en dur
        # continuait d'interdire des objets depuis établis, et taisait donc une
        # preuve au lieu de protéger contre une invention.
        forbidden_claims=_forbidden_claims(constraints),
        blind_visual_fields=_blind_visual_fields(constraints),
        limitations=[
            "volume de façade proxy sans texture de production",
            "toiture LiDAR observée à 96.9 %, lacunes conservées dans le raster",
            "terrain interpolé sous l'emprise, visual_proxy_not_survey",
            "aucun résultat SfM, splat Brush ou validation Blender exécutée",
            "paquet impropre à une affirmation photoréaliste de l'état courant",
        ],
        video_generation={
            **prompts,
            "provider": None,
            "output_video": None,
        },
    )
    scene_path = folder / "scene.json"
    _write_atomic(
        scene_path,
        json.dumps(manifest.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
    )
    # Le seul passage irréversible : le paquet complet paraît d'un coup.
    os.replace(folder, final_folder)
    pointer = _publish_pointer(workspace, final_folder)
    return {
        "package": final_folder,
        "scene": final_folder / "scene.json",
        "verdict": final_folder / "phase1_verdict.json",
        "current": pointer,
    }
