"""Localisation automatique guidée par un noyau d'images fiables.

Le module sépare volontairement trois choses que COLMAP mélange facilement :

* une reconstruction brute, qui n'est qu'une hypothèse ;
* un noyau d'ancres compatible avec les priors géographiques externes ;
* les images ajoutées ensuite par PnP, avec une preuve par image.

Une vue virtuelle ou feed-forward peut aider à proposer une pose, mais elle ne
devient jamais une mesure et ne compte jamais dans G5 sans validation PnP sur
les pixels d'une image réelle.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations
import json
import math
from pathlib import Path
import shutil
import sqlite3
from typing import Protocol, Sequence

import numpy as np

from .geometry_align import apply_sim3, umeyama_sim3
from .schemas import (
    AnchorCandidateEvidence,
    AnchorLocalizationPolicy,
    AnchorModelManifest,
    AnchorSelectionManifest,
    LocalizationAttempt,
    LocalizationDecision,
    LocalizationManifest,
    LocalizedPoseEvidence,
    PoseEvidenceClass,
)
from .schemas.assets import Asset, AssetManifest
from .workspace import Workspace


class AnchorLocalizationRefused(RuntimeError):
    """Le pipeline refuse de fabriquer une pose quand la preuve manque."""


@dataclass(frozen=True)
class ModelPose:
    image_name: str
    camera_center: np.ndarray
    world_from_camera_rotation: np.ndarray


@dataclass(frozen=True)
class LocalizationHypothesis:
    """Mesures brutes retournées par un backend PnP.

    La décision est prise ensuite par :func:`evaluate_localization_hypothesis`.
    Un backend n'a donc pas le pouvoir de s'auto-déclarer probant.
    """

    pose_world_from_camera: dict | None
    matches: int
    inliers: int
    reference_asset_ids: tuple[str, ...]
    reprojection_errors_px: tuple[float, ...] = ()
    positive_depth_ratio: float | None = None
    gps_residual_m: float | None = None
    gps_threshold_m: float | None = None
    heading_residual_deg: float | None = None
    heading_is_measured: bool = False
    stability_translation_m: float | None = None
    stability_rotation_deg: float | None = None
    derived_image_digest: str | None = None
    reasons: tuple[str, ...] = ()


class LocalizationBackend(Protocol):
    """Interface minimale d'un localiseur, remplaçable dans les tests."""

    def localize(
        self,
        asset: Asset,
        reference_asset_ids: Sequence[str],
        *,
        round_index: int,
        hop: int,
        retry_index: int,
        correction_level: str,
    ) -> LocalizationHypothesis | None: ...


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _digest_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_model(path: Path) -> str:
    """Empreinte stable d'un modèle COLMAP texte ou binaire."""

    digest = sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise FileNotFoundError(f"modèle COLMAP vide ou absent : {path}")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(_digest_file(item)))
    return digest.hexdigest()


def _qvec_to_rotation(qvec: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in qvec)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        raise ValueError("quaternion COLMAP nul")
    w, x, y, z = (v / norm for v in (w, x, y, z))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _load_text_model(path: Path) -> list[ModelPose]:
    images_path = path / "images.txt"
    if not images_path.is_file():
        raise FileNotFoundError(images_path)
    poses: list[ModelPose] = []
    data_lines = [
        line.strip()
        for line in images_path.read_text("utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    # Une image occupe deux lignes ; la seconde contient les observations 2D.
    for index in range(0, len(data_lines), 2):
        tokens = data_lines[index].split()
        if len(tokens) < 10:
            continue
        qvec = [float(value) for value in tokens[1:5]]
        translation = np.asarray([float(value) for value in tokens[5:8]])
        rotation_camera_from_world = _qvec_to_rotation(qvec)
        rotation_world_from_camera = rotation_camera_from_world.T
        center = -(rotation_world_from_camera @ translation)
        poses.append(
            ModelPose(
                image_name=" ".join(tokens[9:]),
                camera_center=center,
                world_from_camera_rotation=rotation_world_from_camera,
            )
        )
    return poses


def load_model_poses(path: Path) -> list[ModelPose]:
    """Lit les caméras enregistrées, sans inventer de modèle manquant."""

    if not path.is_dir():
        raise FileNotFoundError(f"modèle COLMAP absent : {path}")
    if (path / "images.txt").is_file():
        return _load_text_model(path)
    try:
        import pycolmap  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - dépend de l'extra SfM
        raise AnchorLocalizationRefused(
            "modèle binaire COLMAP fourni mais pycolmap n'est pas installé"
        ) from exc
    reconstruction = pycolmap.Reconstruction(str(path))
    poses: list[ModelPose] = []
    for image in reconstruction.images.values():
        if not image.has_pose:
            continue
        cam_from_world = image.cam_from_world
        rotation_camera_from_world = np.asarray(cam_from_world.rotation.matrix())
        rotation_world_from_camera = rotation_camera_from_world.T
        poses.append(
            ModelPose(
                image_name=image.name,
                camera_center=np.asarray(image.projection_center(), dtype=float),
                world_from_camera_rotation=rotation_world_from_camera,
            )
        )
    return poses


def _asset_local_name(asset: Asset) -> str | None:
    local_path = getattr(asset, "local_path", None)
    return Path(local_path).name if local_path else None


class AssetResolver:
    """Résout un nom COLMAP sans supposer qu'il égale l'identifiant asset."""

    def __init__(self, manifest: AssetManifest, image_dir: Path | None = None) -> None:
        self.assets = manifest.assets
        self.by_id = {asset.id: asset for asset in self.assets}
        self.image_dir = image_dir

    def resolve(self, image_name: str) -> Asset | None:
        path = Path(image_name)
        basename = path.name
        stem = path.stem
        direct = self.by_id.get(image_name) or self.by_id.get(stem)
        if direct is not None:
            return direct

        named = [asset for asset in self.assets if _asset_local_name(asset) == basename]
        if len(named) == 1:
            return named[0]

        if self.image_dir is not None:
            image_path = self.image_dir / image_name
            if image_path.is_file():
                checksum = _digest_file(image_path)
                checked = [asset for asset in self.assets if asset.checksum == checksum]
                if len(checked) == 1:
                    return checked[0]
                # Plusieurs entrées peuvent décrire le même octet. Préférer
                # celle dont le nom local confirme aussi le rapprochement.
                exact = [asset for asset in checked if _asset_local_name(asset) == basename]
                if len(exact) == 1:
                    return exact[0]

        suffix = [
            asset
            for asset in self.assets
            if asset.id.endswith(stem)
            or stem.endswith(asset.id)
            or (_asset_local_name(asset) or "").endswith(basename)
        ]
        return suffix[0] if len(suffix) == 1 else None


def _enu(points: Sequence[tuple[float, float]]) -> np.ndarray:
    """Projection locale suffisante à l'échelle d'un bâtiment."""

    lat0 = math.radians(sum(lat for lat, _ in points) / len(points))
    lon0 = math.radians(sum(lon for _, lon in points) / len(points))
    radius = 6_378_137.0
    result = []
    for lat, lon in points:
        east = radius * (math.radians(lon) - lon0) * math.cos(lat0)
        north = radius * (math.radians(lat) - lat0)
        result.append((east, north, 0.0))
    return np.asarray(result, dtype=float)


def _enu_one(lat: float, lon: float, lat0: float, lon0: float) -> np.ndarray:
    radius = 6_378_137.0
    return np.asarray(
        [
            radius * math.radians(lon - lon0) * math.cos(math.radians(lat0)),
            radius * math.radians(lat - lat0),
            0.0,
        ],
        dtype=float,
    )


def _angle_delta(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _heading_from_axis(axis: np.ndarray) -> float | None:
    horizontal = math.hypot(float(axis[0]), float(axis[1]))
    if horizontal < 1e-9:
        return None
    return math.degrees(math.atan2(float(axis[0]), float(axis[1]))) % 360.0


def _robust_sim3(
    source: np.ndarray,
    target: np.ndarray,
    *,
    threshold_m: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """RANSAC Sim(3) déterministe puis ré-estimation sur le consensus."""

    if len(source) < 3:
        raise AnchorLocalizationRefused("moins de trois correspondances géographiques")
    triplets = list(combinations(range(len(source)), 3))
    if len(triplets) > 10_000:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(triplets), size=10_000, replace=False)
        triplets = [triplets[int(index)] for index in chosen]

    best: tuple[int, float, np.ndarray, np.ndarray, float, np.ndarray] | None = None
    for indices in triplets:
        subset = np.asarray(indices, dtype=int)
        centered = source[subset] - source[subset].mean(axis=0)
        if np.linalg.matrix_rank(centered) < 2:
            continue
        rotation, translation, scale = umeyama_sim3(source[subset], target[subset])
        residuals = np.linalg.norm(apply_sim3(source, rotation, translation, scale) - target, axis=1)
        inliers = residuals <= threshold_m
        score = (int(inliers.sum()), -float(np.median(residuals[inliers])) if inliers.any() else -math.inf)
        if best is None or score > (best[0], best[1]):
            best = (score[0], score[1], rotation, translation, scale, inliers)
    if best is None or best[0] < 3:
        raise AnchorLocalizationRefused("aucun consensus Sim(3) non dégénéré")

    inliers = best[5]
    rotation, translation, scale = umeyama_sim3(source[inliers], target[inliers])
    residuals = np.linalg.norm(apply_sim3(source, rotation, translation, scale) - target, axis=1)
    inliers = residuals <= threshold_m
    if int(inliers.sum()) >= 3:
        rotation, translation, scale = umeyama_sim3(source[inliers], target[inliers])
    return rotation, translation, scale, inliers


def select_anchor_core(
    *,
    workspace: Workspace,
    reconstruction_input_id: str,
    source_model_path: Path,
    source_run_id: str | None = None,
    image_dir: Path | None = None,
    policy: AnchorLocalizationPolicy | None = None,
) -> AnchorSelectionManifest:
    """Sélectionne automatiquement les caméras cohérentes avec le terrain."""

    policy = policy or AnchorLocalizationPolicy()
    manifest = workspace.read_assets()
    if manifest is None:
        raise AnchorLocalizationRefused("asset_manifest.json absent")
    poses = load_model_poses(source_model_path)
    resolver = AssetResolver(manifest, image_dir)
    rows: list[tuple[ModelPose, Asset]] = []
    unresolved: list[str] = []
    for pose in poses:
        asset = resolver.resolve(pose.image_name)
        if asset is None:
            unresolved.append(pose.image_name)
            continue
        if asset.camera_lat is None or asset.camera_lon is None:
            continue
        rows.append((pose, asset))

    digest = digest_model(source_model_path)
    selection_id = f"anchors-{reconstruction_input_id}-{_utc_stamp()}"
    if len(rows) < 3:
        return AnchorSelectionManifest(
            anchor_selection_id=selection_id,
            reconstruction_input_id=reconstruction_input_id,
            source_run_id=source_run_id,
            source_model_path=str(source_model_path.resolve()),
            policy=policy,
            source_model_digest=digest,
            metrics={"registered": len(poses), "georeferenced": len(rows), "unresolved": unresolved},
            status="refused",
            refusal_reasons=["moins de trois poses enregistrées avec prior GPS"],
        )

    source = np.asarray([pose.camera_center for pose, _ in rows], dtype=float)
    geographic = [(asset.camera_lat, asset.camera_lon) for _, asset in rows]
    target = _enu(geographic)
    enu_origin_lat = sum(lat for lat, _ in geographic) / len(geographic)
    enu_origin_lon = sum(lon for _, lon in geographic) / len(geographic)
    try:
        rotation, translation, scale, inliers = _robust_sim3(
            source,
            target,
            threshold_m=policy.geo_inlier_threshold_m,
            seed=policy.random_seed,
        )
    except AnchorLocalizationRefused as exc:
        return AnchorSelectionManifest(
            anchor_selection_id=selection_id,
            reconstruction_input_id=reconstruction_input_id,
            source_run_id=source_run_id,
            source_model_path=str(source_model_path.resolve()),
            policy=policy,
            source_model_digest=digest,
            metrics={"registered": len(poses), "georeferenced": len(rows), "unresolved": unresolved},
            status="refused",
            refusal_reasons=[str(exc)],
        )

    transformed = apply_sim3(source, rotation, translation, scale)
    residuals = np.linalg.norm(transformed - target, axis=1)
    candidates: list[AnchorCandidateEvidence] = []
    heading_residuals: list[float] = []
    for index, ((pose, asset), is_inlier) in enumerate(zip(rows, inliers, strict=True)):
        optical_axis = rotation @ (pose.world_from_camera_rotation @ np.array([0.0, 0.0, 1.0]))
        predicted_heading = _heading_from_axis(optical_axis)
        heading_residual = None
        if predicted_heading is not None and asset.heading_deg is not None:
            heading_residual = _angle_delta(predicted_heading, asset.heading_deg)
        accepted_by_pose = bool(is_inlier)
        if (
            accepted_by_pose
            and asset.heading_is_measured
            and asset.heading_deg is not None
            and (
                heading_residual is None
                or heading_residual > policy.measured_heading_residual_max_deg
            )
        ):
            accepted_by_pose = False
        reasons: list[str] = []
        if not is_inlier:
            reasons.append("geo_residual_above_threshold")
        if bool(is_inlier) and not accepted_by_pose:
            reasons.append("measured_heading_residual_above_threshold")
        if accepted_by_pose and asset.heading_is_measured and heading_residual is not None:
            heading_residuals.append(heading_residual)
        candidates.append(
            AnchorCandidateEvidence(
                asset_id=asset.id,
                image_name=pose.image_name,
                source=asset.source_family or asset.source,
                reconstructed_center=tuple(float(v) for v in pose.camera_center),
                geographic_center_enu_m=tuple(float(v) for v in target[index]),
                position_residual_m=float(residuals[index]),
                heading_residual_deg=heading_residual,
                accepted=accepted_by_pose,
                reasons=reasons,
            )
        )

    # Le noyau cherché est un consensus, pas la totalité des inliers GPS.
    # Retirer automatiquement le pire cap mesuré est l'équivalent robuste du
    # trimming spatial RANSAC et évite qu'une seule métadonnée incohérente
    # fasse refuser un noyau par ailleurs suffisamment diversifié.
    assets_by_id = {asset.id: asset for _, asset in rows}
    while True:
        measured_candidates = [
            candidate
            for candidate in candidates
            if candidate.accepted
            and assets_by_id[candidate.asset_id].heading_is_measured
            and candidate.heading_residual_deg is not None
        ]
        values = [float(candidate.heading_residual_deg) for candidate in measured_candidates]
        median = float(np.median(values)) if values else None
        p90 = float(np.percentile(values, 90)) if values else None
        heading_ok = (
            median is None
            or (
                median <= policy.anchor_heading_median_max_deg
                and p90 is not None
                and p90 <= policy.anchor_heading_p90_max_deg
            )
        )
        if heading_ok or len(measured_candidates) <= 1:
            break
        worst = max(measured_candidates, key=lambda candidate: float(candidate.heading_residual_deg or 0.0))
        worst.accepted = False
        worst.reasons.append("trimmed_from_anchor_heading_consensus")

    accepted = [candidate for candidate in candidates if candidate.accepted]
    rmse = float(np.sqrt(np.mean([candidate.position_residual_m**2 for candidate in accepted])))
    sources = {candidate.source for candidate in accepted}
    heading_residuals = [
        float(candidate.heading_residual_deg)
        for candidate in accepted
        if assets_by_id[candidate.asset_id].heading_is_measured
        and candidate.heading_residual_deg is not None
    ]
    heading_median = float(np.median(heading_residuals)) if heading_residuals else None
    heading_p90 = float(np.percentile(heading_residuals, 90)) if heading_residuals else None
    refusal_reasons: list[str] = []
    if len(accepted) < policy.min_anchor_images:
        refusal_reasons.append(f"ancres insuffisantes: {len(accepted)} < {policy.min_anchor_images}")
    if len(sources) < policy.min_anchor_sources:
        refusal_reasons.append(f"sources indépendantes insuffisantes: {len(sources)} < {policy.min_anchor_sources}")
    if rmse > policy.anchor_rmse_max_m:
        refusal_reasons.append(f"RMSE géographique trop élevée: {rmse:.3f} m")
    if heading_median is not None and heading_median > policy.anchor_heading_median_max_deg:
        refusal_reasons.append(f"médiane du cap trop élevée: {heading_median:.3f} deg")
    if heading_p90 is not None and heading_p90 > policy.anchor_heading_p90_max_deg:
        refusal_reasons.append(f"P90 du cap trop élevé: {heading_p90:.3f} deg")

    return AnchorSelectionManifest(
        anchor_selection_id=selection_id,
        reconstruction_input_id=reconstruction_input_id,
        source_run_id=source_run_id,
        source_model_path=str(source_model_path.resolve()),
        policy=policy,
        source_model_digest=digest,
        candidates=candidates,
        anchor_asset_ids=[candidate.asset_id for candidate in accepted],
        rejected_asset_ids=[candidate.asset_id for candidate in candidates if not candidate.accepted],
        metrics={
            "registered": len(poses),
            "georeferenced": len(rows),
            "unresolved_image_names": unresolved,
            "anchor_count": len(accepted),
            "source_count": len(sources),
            "geo_rmse_m": rmse,
            "heading_measured_count": len(heading_residuals),
            "heading_median_deg": heading_median,
            "heading_p90_deg": heading_p90,
            "enu_origin_lat": enu_origin_lat,
            "enu_origin_lon": enu_origin_lon,
            "sim3": {
                "scale": float(scale),
                "rotation": rotation.tolist(),
                "translation": translation.tolist(),
            },
        },
        status="ready" if not refusal_reasons else "refused",
        refusal_reasons=refusal_reasons,
    )


def publish_anchor_selection(workspace: Workspace, selection: AnchorSelectionManifest) -> Path:
    relative = f"07_reconstruction/anchors/{selection.anchor_selection_id}.json"
    return workspace.write_json(relative, selection.model_dump(mode="json"))


def _filtered_anchor_database(
    *,
    source_database: Path,
    target_database: Path,
    resolver: AssetResolver,
    anchor_asset_ids: set[str],
) -> tuple[list[str], str]:
    """Copie puis réduit une base COLMAP aux seules ancres sélectionnées."""

    if not source_database.is_file():
        raise FileNotFoundError(source_database)
    target_database.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source_database, target_database)
    connection = sqlite3.connect(target_database)
    try:
        rows = connection.execute("SELECT image_id, name FROM images").fetchall()
        keep_ids: set[int] = set()
        keep_names: list[str] = []
        for image_id, name in rows:
            asset = resolver.resolve(str(name))
            if asset is not None and asset.id in anchor_asset_ids:
                keep_ids.add(int(image_id))
                keep_names.append(str(name))
        if len(keep_ids) < 3:
            raise AnchorLocalizationRefused(
                f"la base ne contient que {len(keep_ids)} ancres résolues"
            )
        drop_ids = [int(image_id) for image_id, _ in rows if int(image_id) not in keep_ids]
        for image_id in drop_ids:
            connection.execute("DELETE FROM keypoints WHERE image_id=?", (image_id,))
            connection.execute("DELETE FROM descriptors WHERE image_id=?", (image_id,))
            connection.execute("DELETE FROM pose_priors WHERE image_id=?", (image_id,))
            connection.execute("DELETE FROM images WHERE image_id=?", (image_id,))
        max_image_id = 2_147_483_647
        for table in ("matches", "two_view_geometries"):
            pair_ids = [int(row[0]) for row in connection.execute(f"SELECT pair_id FROM {table}")]
            for pair_id in pair_ids:
                image_id2 = pair_id % max_image_id
                image_id1 = (pair_id - image_id2) // max_image_id
                if image_id1 not in keep_ids or image_id2 not in keep_ids:
                    connection.execute(f"DELETE FROM {table} WHERE pair_id=?", (pair_id,))
        connection.execute(
            "DELETE FROM cameras WHERE camera_id NOT IN (SELECT DISTINCT camera_id FROM images)"
        )
        connection.commit()
    finally:
        connection.close()
    return sorted(keep_names), _digest_file(target_database)


def _model_pose_dict(path: Path) -> dict[str, np.ndarray]:
    return {pose.image_name: pose.camera_center for pose in load_model_poses(path)}


def _aligned_center_rmse_m(reference: Path, candidate: Path, metric_scale: float) -> tuple[float, int]:
    reference_centers = _model_pose_dict(reference)
    candidate_centers = _model_pose_dict(candidate)
    common = sorted(set(reference_centers) & set(candidate_centers))
    if len(common) < 3:
        return math.inf, len(common)
    source = np.asarray([candidate_centers[name] for name in common])
    target = np.asarray([reference_centers[name] for name in common])
    rotation, translation, scale = umeyama_sim3(source, target)
    aligned = apply_sim3(source, rotation, translation, scale)
    return (
        float(np.sqrt(np.mean(np.linalg.norm(aligned - target, axis=1) ** 2))) * metric_scale,
        len(common),
    )


def build_anchor_model(
    *,
    workspace: Workspace,
    selection: AnchorSelectionManifest,
    source_database: Path,
    image_dir: Path,
    stability_repeats: int = 3,
) -> AnchorModelManifest:
    """Reconstruit le noyau seul depuis une copie filtrée de la base.

    Le modèle brut n'est jamais simplement recopié : keypoints et matches
    sont conservés, mais le mapper repart de zéro avec uniquement les ancres.
    """

    if selection.status != "ready":
        raise AnchorLocalizationRefused(
            f"sélection d'ancres refusée: {selection.refusal_reasons}"
        )
    if stability_repeats < 2:
        raise ValueError("au moins deux reconstructions sont requises pour mesurer la stabilité")
    manifest = workspace.read_assets()
    if manifest is None:
        raise AnchorLocalizationRefused("asset_manifest.json absent")
    try:
        import pycolmap  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise AnchorLocalizationRefused("pycolmap requis pour reconstruire le noyau") from exc

    anchor_model_id = f"anchor-model-{selection.anchor_selection_id}-{_utc_stamp()}"
    root = workspace.path("07_reconstruction", "anchors", anchor_model_id)
    database_path = root / "database" / "database.db"
    resolver = AssetResolver(manifest, image_dir)
    image_names, database_digest = _filtered_anchor_database(
        source_database=source_database,
        target_database=database_path,
        resolver=resolver,
        anchor_asset_ids=set(selection.anchor_asset_ids),
    )

    successful_models: list[Path] = []
    run_records: list[dict] = []
    for index in range(stability_repeats):
        output = root / "stability" / f"run-{index + 1}"
        output.mkdir(parents=True, exist_ok=False)
        options = pycolmap.IncrementalPipelineOptions()
        options.min_num_matches = 15
        options.min_model_size = 3
        options.multiple_models = True
        options.max_num_models = 5
        options.num_threads = 1
        options.ba_refine_focal_length = False
        options.ba_refine_principal_point = False
        options.ba_refine_extra_params = False
        options.mapper.init_min_num_inliers = 30
        options.mapper.abs_pose_min_num_inliers = selection.policy.pnp_min_inliers
        options.mapper.abs_pose_min_inlier_ratio = selection.policy.pnp_min_inlier_ratio
        options.mapper.abs_pose_refine_focal_length = False
        options.mapper.abs_pose_refine_extra_params = False
        options.mapper.num_threads = 1
        pycolmap.set_random_seed(selection.policy.random_seed + index)
        reconstructions = pycolmap.incremental_mapping(
            database_path=str(database_path),
            image_path=str(image_dir),
            output_path=str(output),
            options=options,
        )
        if not reconstructions:
            run_records.append({"run_index": index + 1, "status": "failed", "models": 0})
            continue
        best_id, best = max(reconstructions.items(), key=lambda item: item[1].num_reg_images())
        model_path = output / str(best_id)
        # pycolmap écrit normalement déjà le modèle ; cette écriture rend
        # le contrat explicite et fonctionne aussi avec un backend mocké.
        model_path.mkdir(parents=True, exist_ok=True)
        best.write(str(model_path))
        successful_models.append(model_path)
        run_records.append(
            {
                "run_index": index + 1,
                "status": "completed",
                "model_path": str(model_path.resolve()),
                "registered_images": int(best.num_reg_images()),
                "points3D": int(best.num_points3D()),
            }
        )

    refusal_reasons: list[str] = []
    if len(successful_models) < 2:
        refusal_reasons.append("moins de deux reconstructions du noyau ont abouti")
    if not successful_models:
        # Un chemin non vide est requis par le schéma, mais il ne doit jamais
        # pointer vers un faux modèle : le dossier racine est un artefact refusé.
        return AnchorModelManifest(
            anchor_model_id=anchor_model_id,
            anchor_selection_id=selection.anchor_selection_id,
            reconstruction_input_id=selection.reconstruction_input_id,
            model_path=str(root.resolve()),
            model_digest=digest_model(root),
            anchor_asset_ids=selection.anchor_asset_ids,
            metrics={"database_digest": database_digest, "database_images": image_names},
            stability_runs=run_records,
            status="refused",
            refusal_reasons=refusal_reasons or ["aucun modèle d'ancre reconstruit"],
        )

    best_path = max(successful_models, key=lambda path: len(load_model_poses(path)))
    best_poses = load_model_poses(best_path)
    registered_assets = {
        asset.id
        for pose in best_poses
        if (asset := resolver.resolve(pose.image_name)) is not None
    }
    retained = sorted(registered_assets & set(selection.anchor_asset_ids))
    if len(retained) < selection.policy.min_anchor_images:
        refusal_reasons.append(
            f"noyau reconstruit insuffisant: {len(retained)} < {selection.policy.min_anchor_images}"
        )
    metric_scale = float(selection.metrics.get("sim3", {}).get("scale", 1.0))
    stability_rmse: list[float] = []
    for record, model_path in zip(
        [record for record in run_records if record["status"] == "completed"],
        successful_models,
        strict=True,
    ):
        rmse_m, common = _aligned_center_rmse_m(best_path, model_path, metric_scale)
        record["aligned_center_rmse_m"] = rmse_m
        record["common_images"] = common
        stability_rmse.append(rmse_m)
    finite_stability = [value for value in stability_rmse if math.isfinite(value)]
    worst_stability = max(finite_stability, default=math.inf)
    if worst_stability > selection.policy.pose_stability_translation_max_m:
        refusal_reasons.append(f"noyau instable entre runs: {worst_stability:.3f} m")

    reconstruction = pycolmap.Reconstruction(str(best_path))
    camera_parameters = {
        str(camera_id): [float(value) for value in camera.params]
        for camera_id, camera in reconstruction.cameras.items()
    }
    return AnchorModelManifest(
        anchor_model_id=anchor_model_id,
        anchor_selection_id=selection.anchor_selection_id,
        reconstruction_input_id=selection.reconstruction_input_id,
        model_path=str(best_path.resolve()),
        model_digest=digest_model(best_path),
        anchor_asset_ids=retained,
        camera_parameters=camera_parameters,
        metrics={
            "database_digest": database_digest,
            "database_images": image_names,
            "selected_anchor_images": len(selection.anchor_asset_ids),
            "registered_anchor_images": len(retained),
            "worst_stability_rmse_m": worst_stability,
        },
        stability_runs=run_records,
        status="ready" if not refusal_reasons else "refused",
        refusal_reasons=refusal_reasons,
    )


def publish_anchor_model(workspace: Workspace, model: AnchorModelManifest) -> Path:
    relative = f"07_reconstruction/anchors/{model.anchor_model_id}.json"
    return workspace.write_json(relative, model.model_dump(mode="json"))


class H5PnPLocalizationBackend:
    """Localisation 2D-3D depuis les features/matches HLoc déjà calculés.

    Ce backend appelle directement le PnP de pycolmap. Il n'utilise pas le
    fallback de ``hloc.localize_sfm.main`` qui peut copier la pose du voisin
    lorsqu'aucune pose n'a été estimée.
    """

    def __init__(
        self,
        *,
        workspace: Workspace,
        anchor_model: AnchorModelManifest,
        anchor_selection: AnchorSelectionManifest,
        source_database: Path,
        features_path: Path,
        matches_path: Path,
        image_dir: Path,
    ) -> None:
        if anchor_model.status != "ready" or anchor_selection.status != "ready":
            raise AnchorLocalizationRefused("noyau ou sélection d'ancres non validé")
        try:
            import h5py  # type: ignore[import-not-found]
            import pycolmap  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - extra sfm
            raise AnchorLocalizationRefused("les extras sfm pycolmap+h5py sont requis") from exc
        self.h5py = h5py
        self.pycolmap = pycolmap
        self.features_path = features_path
        self.matches_path = matches_path
        self.image_dir = image_dir
        self.variant_root = workspace.path("07_reconstruction", "localization", "variants")
        self._orb_reference_cache: dict[str, tuple[list, np.ndarray | None, list[int | None]]] = {}
        manifest = workspace.read_assets()
        if manifest is None:
            raise AnchorLocalizationRefused("asset_manifest.json absent")
        self.resolver = AssetResolver(manifest, image_dir)
        self.assets_by_id = {asset.id: asset for asset in manifest.assets}
        self.reconstruction = pycolmap.Reconstruction(anchor_model.model_path)
        self.reference_by_asset: dict[str, object] = {}
        for image in self.reconstruction.images.values():
            asset = self.resolver.resolve(image.name)
            if asset is not None:
                self.reference_by_asset[asset.id] = image

        database = pycolmap.Database()
        database.open(str(source_database))
        try:
            self.query_name_by_asset: dict[str, str] = {}
            self.camera_by_asset: dict[str, object] = {}
            for image in database.read_all_images():
                asset = self.resolver.resolve(image.name)
                if asset is None:
                    continue
                self.query_name_by_asset[asset.id] = image.name
                self.camera_by_asset[asset.id] = database.read_camera(image.camera_id)
        finally:
            database.close()
        sim3 = anchor_selection.metrics.get("sim3", {})
        self.geo_rotation = np.asarray(sim3.get("rotation"), dtype=float)
        self.geo_translation = np.asarray(sim3.get("translation"), dtype=float)
        self.geo_scale = float(sim3.get("scale", 1.0))
        origin_lat = anchor_selection.metrics.get("enu_origin_lat")
        origin_lon = anchor_selection.metrics.get("enu_origin_lon")
        if origin_lat is None or origin_lon is None:
            by_id = {asset.id: asset for asset in manifest.assets}
            geographic_assets = [
                by_id[candidate.asset_id]
                for candidate in anchor_selection.candidates
                if candidate.asset_id in by_id
                and by_id[candidate.asset_id].camera_lat is not None
                and by_id[candidate.asset_id].camera_lon is not None
            ]
            if not geographic_assets:
                raise AnchorLocalizationRefused("origine ENU absente de la sélection d'ancres")
            origin_lat = sum(float(asset.camera_lat) for asset in geographic_assets) / len(geographic_assets)
            origin_lon = sum(float(asset.camera_lon) for asset in geographic_assets) / len(geographic_assets)
        self.enu_origin_lat = float(origin_lat)
        self.enu_origin_lon = float(origin_lon)
        self.seed = anchor_selection.policy.random_seed

    @staticmethod
    def _pair_name(name0: str, name1: str, separator: str = "/") -> str:
        return separator.join((name0.replace("/", "-"), name1.replace("/", "-")))

    def _matches(self, name0: str, name1: str) -> tuple[np.ndarray, np.ndarray]:
        with self.h5py.File(str(self.matches_path), "r", libver="latest") as hfile:
            candidates = [
                (self._pair_name(name0, name1), False),
                (self._pair_name(name1, name0), True),
                (self._pair_name(name0, name1, "_"), False),
                (self._pair_name(name1, name0, "_"), True),
            ]
            for pair, reverse in candidates:
                if pair not in hfile:
                    continue
                matches0 = np.asarray(hfile[pair]["matches0"])
                scores0 = np.asarray(hfile[pair]["matching_scores0"])
                indices = np.where(matches0 != -1)[0]
                matches = np.stack([indices, matches0[indices]], axis=-1)
                if reverse:
                    matches = np.flip(matches, axis=-1)
                return matches.astype(int), scores0[indices]
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=float)

    def _keypoints(self, name: str) -> np.ndarray:
        with self.h5py.File(str(self.features_path), "r", libver="latest") as hfile:
            if name not in hfile:
                raise KeyError(f"features absentes pour {name}")
            # HLoc ajoute 0,5 pour passer du repère pixel OpenCV au centre du
            # pixel attendu par COLMAP.
            return np.asarray(hfile[name]["keypoints"], dtype=float) + 0.5

    def _correspondences(
        self, query_name: str, reference_asset_ids: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray, list[set[str]]]:
        votes: dict[int, list[tuple[int, float, str]]] = {}
        for reference_asset_id in reference_asset_ids:
            reference = self.reference_by_asset.get(reference_asset_id)
            if reference is None:
                continue
            matches, scores = self._matches(query_name, reference.name)
            for (query_index, reference_index), score in zip(matches, scores, strict=True):
                if reference_index >= len(reference.points2D):
                    continue
                point2D = reference.points2D[int(reference_index)]
                if not point2D.has_point3D():
                    continue
                votes.setdefault(int(query_index), []).append(
                    (int(point2D.point3D_id), float(score), reference_asset_id)
                )
        query_keypoints = self._keypoints(query_name)
        points2D: list[np.ndarray] = []
        points3D: list[np.ndarray] = []
        sources: list[set[str]] = []
        for query_index, candidates in sorted(votes.items()):
            if query_index >= len(query_keypoints):
                continue
            by_point: dict[int, tuple[float, set[str]]] = {}
            for point_id, score, reference_asset_id in candidates:
                previous_score, previous_sources = by_point.get(point_id, (0.0, set()))
                by_point[point_id] = (previous_score + score, previous_sources | {reference_asset_id})
            point_id, (_, point_sources) = max(by_point.items(), key=lambda item: item[1][0])
            if point_id not in self.reconstruction.points3D:
                continue
            point = self.reconstruction.points3D[point_id]
            points2D.append(query_keypoints[query_index])
            points3D.append(np.asarray(point.xyz, dtype=float))
            sources.append(point_sources)
        return np.asarray(points2D), np.asarray(points3D), sources

    def _orb_reference(self, reference_asset_id: str):  # noqa: ANN202
        cached = self._orb_reference_cache.get(reference_asset_id)
        if cached is not None:
            return cached
        import cv2

        reference = self.reference_by_asset[reference_asset_id]
        image = cv2.imread(str(self.image_dir / reference.name), cv2.IMREAD_GRAYSCALE)
        if image is None:
            result = ([], None, [])
            self._orb_reference_cache[reference_asset_id] = result
            return result
        orb = cv2.ORB_create(nfeatures=4000, fastThreshold=7)
        keypoints, descriptors = orb.detectAndCompute(image, None)
        mapped_point_ids: list[int | None] = []
        triangulated = [point for point in reference.points2D if point.has_point3D()]
        triangulated_xy = (
            np.asarray([point.xy for point in triangulated], dtype=float)
            if triangulated
            else np.empty((0, 2))
        )
        for keypoint in keypoints:
            if not len(triangulated_xy):
                mapped_point_ids.append(None)
                continue
            distances = np.linalg.norm(
                triangulated_xy - np.asarray(keypoint.pt, dtype=float), axis=1
            )
            nearest = int(np.argmin(distances))
            mapped_point_ids.append(
                int(triangulated[nearest].point3D_id) if distances[nearest] <= 3.0 else None
            )
        result = (keypoints, descriptors, mapped_point_ids)
        self._orb_reference_cache[reference_asset_id] = result
        return result

    def _corrected_orb_correspondences(
        self,
        asset: Asset,
        query_name: str,
        camera,
        reference_asset_ids: Sequence[str],
        correction_level: str,
    ) -> tuple[np.ndarray, np.ndarray, list[set[str]], object, str] | None:
        import cv2

        source_path = self.image_dir / query_name
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            return None
        output_camera = camera
        if correction_level == "photometric":
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lightness, a_channel, b_channel = cv2.split(lab)
            lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
            corrected = cv2.cvtColor(
                cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2BGR
            )
        elif correction_level == "deterministic_geometry":
            matrix = np.asarray(
                [
                    [camera.focal_length_x, 0.0, camera.principal_point_x],
                    [0.0, camera.focal_length_y, camera.principal_point_y],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
            # pycolmap fournit la conversion exacte quel que soit le modèle,
            # mais OpenCV ne partage pas tous ses modèles. SIMPLE_RADIAL et
            # RADIAL couvrent le corpus pilote ; ailleurs on refuse la variante.
            params = np.asarray(camera.params, dtype=float)
            camera_model_name = camera.model.name
            if camera_model_name == "SIMPLE_RADIAL":
                distortion = np.asarray([params[3], 0.0, 0.0, 0.0], dtype=float)
            elif camera_model_name == "RADIAL":
                distortion = np.asarray([params[3], params[4], 0.0, 0.0], dtype=float)
            elif camera_model_name in {"PINHOLE", "SIMPLE_PINHOLE"}:
                distortion = np.zeros(4, dtype=float)
            else:
                return None
            corrected = cv2.undistort(image, matrix, distortion)
            output_camera = self.pycolmap.Camera(
                model="PINHOLE",
                width=int(camera.width),
                height=int(camera.height),
                params=[
                    float(camera.focal_length_x),
                    float(camera.focal_length_y),
                    float(camera.principal_point_x),
                    float(camera.principal_point_y),
                ],
            )
        else:
            return None

        ok, encoded = cv2.imencode(".jpg", corrected, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            return None
        payload = bytes(encoded)
        derived_digest = sha256(payload).hexdigest()
        target = self.variant_root / asset.id / f"{correction_level}-{derived_digest}.jpg"
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)

        gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=4000, fastThreshold=7)
        query_keypoints, query_descriptors = orb.detectAndCompute(gray, None)
        if query_descriptors is None:
            return None
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        votes: dict[int, list[tuple[int, float, str]]] = {}
        for reference_asset_id in reference_asset_ids:
            if reference_asset_id not in self.reference_by_asset:
                continue
            _, reference_descriptors, point_ids = self._orb_reference(reference_asset_id)
            if reference_descriptors is None:
                continue
            pairs = matcher.knnMatch(query_descriptors, reference_descriptors, k=2)
            for pair in pairs:
                if len(pair) != 2 or pair[0].distance >= 0.78 * pair[1].distance:
                    continue
                match = pair[0]
                point_id = point_ids[match.trainIdx]
                if point_id is None:
                    continue
                votes.setdefault(match.queryIdx, []).append(
                    (point_id, 1.0 / (1.0 + float(match.distance)), reference_asset_id)
                )
        points2D: list[np.ndarray] = []
        points3D: list[np.ndarray] = []
        sources: list[set[str]] = []
        for query_index, candidates in sorted(votes.items()):
            by_point: dict[int, tuple[float, set[str]]] = {}
            for point_id, score, reference_asset_id in candidates:
                total, refs = by_point.get(point_id, (0.0, set()))
                by_point[point_id] = (total + score, refs | {reference_asset_id})
            point_id, (_, refs) = max(by_point.items(), key=lambda item: item[1][0])
            if point_id not in self.reconstruction.points3D:
                continue
            points2D.append(np.asarray(query_keypoints[query_index].pt, dtype=float))
            points3D.append(np.asarray(self.reconstruction.points3D[point_id].xyz, dtype=float))
            sources.append(refs)
        return np.asarray(points2D), np.asarray(points3D), sources, output_camera, derived_digest

    def _estimate(self, points2D: np.ndarray, points3D: np.ndarray, camera, seed: int):  # noqa: ANN001, ANN202
        self.pycolmap.set_random_seed(seed)
        options = self.pycolmap.AbsolutePoseEstimationOptions()
        options.ransac.max_error = 4.0
        options.ransac.min_inlier_ratio = 0.1
        options.ransac.confidence = 0.9999
        return self.pycolmap.estimate_and_refine_absolute_pose(
            points2D,
            points3D,
            camera,
            estimation_options=options,
        )

    @staticmethod
    def _rotation_delta_deg(a: np.ndarray, b: np.ndarray) -> float:
        cosine = (float(np.trace(a.T @ b)) - 1.0) / 2.0
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    def localize(
        self,
        asset: Asset,
        reference_asset_ids: Sequence[str],
        *,
        round_index: int,
        hop: int,
        retry_index: int,
        correction_level: str,
    ) -> LocalizationHypothesis | None:
        query_name = self.query_name_by_asset.get(asset.id)
        camera = self.camera_by_asset.get(asset.id)
        if query_name is None or camera is None:
            return None
        references = list(reference_asset_ids)
        if retry_index > 0 and asset.camera_lat is not None and asset.camera_lon is not None:
            def geographic_distance(reference_id: str) -> float:
                reference_asset = self.assets_by_id.get(reference_id)
                if (
                    reference_asset is None
                    or reference_asset.camera_lat is None
                    or reference_asset.camera_lon is None
                ):
                    return math.inf
                return float(
                    np.linalg.norm(
                        _enu_one(
                            reference_asset.camera_lat,
                            reference_asset.camera_lon,
                            asset.camera_lat,
                            asset.camera_lon,
                        )[:2]
                    )
                )

            references.sort(key=geographic_distance)
            references = references[: 6 if retry_index == 1 else 4]
        derived_digest = None
        if correction_level == "original":
            points2D, points3D, sources = self._correspondences(query_name, references)
        else:
            corrected = self._corrected_orb_correspondences(
                asset,
                query_name,
                camera,
                references,
                correction_level,
            )
            if corrected is None:
                return None
            points2D, points3D, sources, camera, derived_digest = corrected
        if len(points2D) < 4:
            return LocalizationHypothesis(
                pose_world_from_camera=None,
                matches=len(points2D),
                inliers=0,
                reference_asset_ids=(),
                derived_image_digest=derived_digest,
                reasons=("insufficient_2d3d_correspondences",),
            )
        attempt_seed = self.seed + round_index * 101 + hop + retry_index * 17
        result = self._estimate(points2D, points3D, camera, attempt_seed)
        second = self._estimate(points2D, points3D, camera, attempt_seed + 1)
        if result is None:
            return LocalizationHypothesis(
                pose_world_from_camera=None,
                matches=len(points2D),
                inliers=0,
                reference_asset_ids=(),
                derived_image_digest=derived_digest,
                reasons=("pnp_failed",),
            )

        cam_from_world = result["cam_from_world"]
        world_from_camera = cam_from_world.inverse()
        rotation_world_from_camera = np.asarray(world_from_camera.rotation.matrix())
        center = np.asarray(world_from_camera.translation, dtype=float)
        inlier_mask = np.asarray(result["inlier_mask"], dtype=bool)
        inlier_indices = np.where(inlier_mask)[0]
        inlier_sources: set[str] = set()
        for index in inlier_indices:
            inlier_sources.update(sources[int(index)])

        errors: list[float] = []
        positive = 0
        for index in inlier_indices:
            point_camera = cam_from_world * points3D[int(index)]
            if float(point_camera[2]) > 0:
                positive += 1
            projected = np.asarray(camera.img_from_cam(point_camera), dtype=float)
            errors.append(float(np.linalg.norm(projected - points2D[int(index)])))
        positive_ratio = positive / len(inlier_indices) if len(inlier_indices) else 0.0

        aligned_center = apply_sim3(
            center.reshape(1, 3), self.geo_rotation, self.geo_translation, self.geo_scale
        )[0]
        gps_residual = None
        gps_threshold = None
        if asset.camera_lat is not None and asset.camera_lon is not None:
            gps_target = _enu_one(
                asset.camera_lat,
                asset.camera_lon,
                self.enu_origin_lat,
                self.enu_origin_lon,
            )
            gps_residual = float(np.linalg.norm(aligned_center - gps_target))
            gps_threshold = max(10.0, 3.0 * 10.0)
        axis = self.geo_rotation @ (rotation_world_from_camera @ np.array([0.0, 0.0, 1.0]))
        predicted_heading = _heading_from_axis(axis)
        heading_residual = (
            _angle_delta(predicted_heading, asset.heading_deg)
            if predicted_heading is not None and asset.heading_deg is not None
            else None
        )

        stability_translation = None
        stability_rotation = None
        if second is not None:
            second_world = second["cam_from_world"].inverse()
            second_center = np.asarray(second_world.translation, dtype=float)
            stability_translation = float(np.linalg.norm(center - second_center)) * self.geo_scale
            stability_rotation = self._rotation_delta_deg(
                rotation_world_from_camera,
                np.asarray(second_world.rotation.matrix()),
            )
        return LocalizationHypothesis(
            pose_world_from_camera={
                "rotation": rotation_world_from_camera.tolist(),
                "translation": center.tolist(),
                "convention": "world_from_camera",
            },
            matches=len(points2D),
            inliers=int(inlier_mask.sum()),
            reference_asset_ids=tuple(sorted(inlier_sources)),
            reprojection_errors_px=tuple(errors),
            positive_depth_ratio=positive_ratio,
            gps_residual_m=gps_residual,
            gps_threshold_m=gps_threshold,
            heading_residual_deg=heading_residual,
            heading_is_measured=asset.heading_is_measured and asset.heading_deg is not None,
            stability_translation_m=stability_translation,
            stability_rotation_deg=stability_rotation,
            derived_image_digest=derived_digest,
        )


def evaluate_localization_hypothesis(
    *,
    asset: Asset,
    hypothesis: LocalizationHypothesis,
    policy: AnchorLocalizationPolicy,
    correction_level: str,
) -> tuple[LocalizationDecision, PoseEvidenceClass, list[str], dict[str, float | None]]:
    """Applique tous les seuils ; aucune valeur manquante ne vaut succès."""

    errors = np.asarray(hypothesis.reprojection_errors_px, dtype=float)
    reprojection_median = float(np.median(errors)) if errors.size else None
    reprojection_p95 = float(np.percentile(errors, 95)) if errors.size else None
    ratio = hypothesis.inliers / hypothesis.matches if hypothesis.matches else 0.0
    reasons = list(hypothesis.reasons)
    if hypothesis.pose_world_from_camera is None:
        reasons.append("pose_missing")
    if hypothesis.inliers < policy.pnp_min_inliers:
        reasons.append("pnp_inliers_below_threshold")
    if ratio < policy.pnp_min_inlier_ratio:
        reasons.append("pnp_inlier_ratio_below_threshold")
    if len(set(hypothesis.reference_asset_ids)) < policy.pnp_min_reference_images:
        reasons.append("reference_images_below_threshold")
    if reprojection_median is None:
        reasons.append("reprojection_metrics_missing")
    elif reprojection_median > policy.reprojection_median_max_px:
        reasons.append("reprojection_median_above_threshold")
    if reprojection_p95 is None:
        reasons.append("reprojection_metrics_missing")
    elif reprojection_p95 > policy.reprojection_p95_max_px:
        reasons.append("reprojection_p95_above_threshold")
    if hypothesis.positive_depth_ratio is None:
        reasons.append("positive_depth_ratio_missing")
    elif hypothesis.positive_depth_ratio < policy.positive_depth_ratio_min:
        reasons.append("positive_depth_ratio_below_threshold")
    if hypothesis.gps_threshold_m is not None:
        if hypothesis.gps_residual_m is None:
            reasons.append("gps_residual_missing")
        elif hypothesis.gps_residual_m > hypothesis.gps_threshold_m:
            reasons.append("gps_residual_above_threshold")
    if hypothesis.heading_is_measured:
        if hypothesis.heading_residual_deg is None:
            reasons.append("measured_heading_residual_missing")
        elif hypothesis.heading_residual_deg > policy.measured_heading_residual_max_deg:
            reasons.append("measured_heading_residual_above_threshold")
    if hypothesis.stability_translation_m is None or hypothesis.stability_rotation_deg is None:
        reasons.append("pose_stability_missing")
    else:
        if hypothesis.stability_translation_m > policy.pose_stability_translation_max_m:
            reasons.append("pose_translation_unstable")
        if hypothesis.stability_rotation_deg > policy.pose_stability_rotation_max_deg:
            reasons.append("pose_rotation_unstable")

    reasons = list(dict.fromkeys(reasons))
    virtual = correction_level in {"virtual", "feedforward"}
    if virtual:
        return (
            LocalizationDecision.INFERRED_ONLY if hypothesis.pose_world_from_camera else LocalizationDecision.REJECTED,
            PoseEvidenceClass.VIEW_INFERRED if hypothesis.pose_world_from_camera else PoseEvidenceClass.REJECTED,
            reasons + ["virtual_pose_is_non_probative"],
            {"inlier_ratio": ratio, "reprojection_median_px": reprojection_median, "reprojection_p95_px": reprojection_p95},
        )
    if reasons:
        return (
            LocalizationDecision.INSUFFICIENT_EVIDENCE,
            PoseEvidenceClass.REJECTED,
            reasons,
            {"inlier_ratio": ratio, "reprojection_median_px": reprojection_median, "reprojection_p95_px": reprojection_p95},
        )
    return (
        LocalizationDecision.ACCEPTED,
        PoseEvidenceClass.LOCALIZED_MEASURED,
        [],
        {"inlier_ratio": ratio, "reprojection_median_px": reprojection_median, "reprojection_p95_px": reprojection_p95},
    )


def run_anchor_localization(
    *,
    reconstruction_input_id: str,
    anchor_model_id: str,
    selected_assets: Sequence[Asset],
    anchor_poses: dict[str, dict],
    backend: LocalizationBackend,
    policy: AnchorLocalizationPolicy | None = None,
    raw_registered_images: int = 0,
    allow_virtual_suggestions: bool = False,
) -> LocalizationManifest:
    """Localise automatiquement les vues restantes, par vagues bornées.

    Une pose acceptée peut devenir une référence au tour suivant, mais le
    nombre de tours et de sauts est borné par la politique. Une pose rejetée ou
    seulement inférée n'entre jamais dans le noyau.
    """

    policy = policy or AnchorLocalizationPolicy()
    selected_by_id = {asset.id: asset for asset in selected_assets}
    unknown_anchors = set(anchor_poses) - set(selected_by_id)
    if unknown_anchors:
        raise AnchorLocalizationRefused(f"ancres hors snapshot: {sorted(unknown_anchors)}")
    references = list(anchor_poses)
    poses: dict[str, LocalizedPoseEvidence] = {
        asset_id: LocalizedPoseEvidence(
            asset_id=asset_id,
            evidence_class=PoseEvidenceClass.ANCHOR_MEASURED,
            decision=LocalizationDecision.ACCEPTED,
            hop=0,
            pose_world_from_camera=pose,
            reasons=["validated_anchor_core"],
        )
        for asset_id, pose in anchor_poses.items()
    }
    attempts: list[LocalizationAttempt] = []
    pending = [asset for asset in selected_assets if asset.id not in poses]
    levels = ["original", "photometric", "deterministic_geometry"]
    if allow_virtual_suggestions:
        levels.append("virtual")

    for round_index in range(policy.max_rounds):
        hop = min(round_index + 1, policy.max_hop)
        if hop > policy.max_hop or not pending:
            break
        newly_accepted: list[str] = []
        still_pending: list[Asset] = []
        for asset in pending:
            accepted_attempt: LocalizationAttempt | None = None
            inferred_attempt: LocalizationAttempt | None = None
            for level in levels:
                for retry in range(policy.max_attempts_per_level):
                    hypothesis = backend.localize(
                        asset,
                        tuple(references),
                        round_index=round_index,
                        hop=hop,
                        retry_index=retry,
                        correction_level=level,
                    )
                    if hypothesis is None:
                        break
                    decision, evidence, reasons, derived = evaluate_localization_hypothesis(
                        asset=asset,
                        hypothesis=hypothesis,
                        policy=policy,
                        correction_level=level,
                    )
                    attempt = LocalizationAttempt(
                        attempt_id=f"loc-{asset.id}-{round_index}-{level}-{retry}-{_utc_stamp()}",
                        asset_id=asset.id,
                        round_index=round_index,
                        hop=hop,
                        correction_level=level,
                        original_image_digest=asset.checksum,
                        derived_image_digest=hypothesis.derived_image_digest,
                        reference_asset_ids=list(hypothesis.reference_asset_ids),
                        matches=hypothesis.matches,
                        inliers=hypothesis.inliers,
                        inlier_ratio=float(derived["inlier_ratio"] or 0.0),
                        reprojection_median_px=derived["reprojection_median_px"],
                        reprojection_p95_px=derived["reprojection_p95_px"],
                        positive_depth_ratio=hypothesis.positive_depth_ratio,
                        gps_residual_m=hypothesis.gps_residual_m,
                        heading_residual_deg=hypothesis.heading_residual_deg,
                        stability_translation_m=hypothesis.stability_translation_m,
                        stability_rotation_deg=hypothesis.stability_rotation_deg,
                        pose_world_from_camera=hypothesis.pose_world_from_camera,
                        decision=decision,
                        evidence_class=evidence,
                        reasons=reasons,
                    )
                    attempts.append(attempt)
                    if evidence is PoseEvidenceClass.LOCALIZED_MEASURED:
                        accepted_attempt = attempt
                        break
                    if evidence is PoseEvidenceClass.VIEW_INFERRED:
                        inferred_attempt = attempt
                if accepted_attempt is not None:
                    break
            if accepted_attempt is not None:
                poses[asset.id] = LocalizedPoseEvidence(
                    asset_id=asset.id,
                    evidence_class=PoseEvidenceClass.LOCALIZED_MEASURED,
                    decision=LocalizationDecision.ACCEPTED,
                    hop=hop,
                    pose_world_from_camera=accepted_attempt.pose_world_from_camera,
                    accepted_attempt_id=accepted_attempt.attempt_id,
                    reasons=["validated_pnp"],
                )
                newly_accepted.append(asset.id)
            elif inferred_attempt is not None:
                poses[asset.id] = LocalizedPoseEvidence(
                    asset_id=asset.id,
                    evidence_class=PoseEvidenceClass.VIEW_INFERRED,
                    decision=LocalizationDecision.INFERRED_ONLY,
                    hop=hop,
                    pose_world_from_camera=inferred_attempt.pose_world_from_camera,
                    accepted_attempt_id=inferred_attempt.attempt_id,
                    reasons=["non_probative_virtual_suggestion"],
                )
            else:
                still_pending.append(asset)
        if not newly_accepted:
            pending = still_pending
            break
        references.extend(newly_accepted)
        pending = still_pending

    for asset in selected_assets:
        if asset.id not in poses:
            asset_attempts = [attempt for attempt in attempts if attempt.asset_id == asset.id]
            reasons = asset_attempts[-1].reasons if asset_attempts else ["no_localization_hypothesis"]
            poses[asset.id] = LocalizedPoseEvidence(
                asset_id=asset.id,
                evidence_class=PoseEvidenceClass.REJECTED,
                decision=LocalizationDecision.INSUFFICIENT_EVIDENCE,
                hop=policy.max_hop,
                reasons=reasons,
            )

    ordered = [poses[asset.id] for asset in selected_assets]
    anchor_count = sum(p.evidence_class is PoseEvidenceClass.ANCHOR_MEASURED for p in ordered)
    localized_count = sum(p.evidence_class is PoseEvidenceClass.LOCALIZED_MEASURED for p in ordered)
    inferred_count = sum(p.evidence_class is PoseEvidenceClass.VIEW_INFERRED for p in ordered)
    rejected_count = sum(p.evidence_class is PoseEvidenceClass.REJECTED for p in ordered)
    measured = anchor_count + localized_count
    total = len(ordered)
    return LocalizationManifest(
        localization_run_id=f"anchor-localization-{reconstruction_input_id}-{_utc_stamp()}",
        reconstruction_input_id=reconstruction_input_id,
        anchor_model_id=anchor_model_id,
        policy=policy,
        selected_asset_ids=[asset.id for asset in selected_assets],
        poses=ordered,
        attempts=attempts,
        raw_registered_images=raw_registered_images,
        measured_anchor_images=anchor_count,
        measured_localized_images=localized_count,
        inferred_images=inferred_count,
        rejected_images=rejected_count,
        validated_registration_rate=measured / total if total else 0.0,
        # Toute pose PnP acceptée est reliée à au moins trois références
        # du noyau ; le composant validé est donc unique par construction.
        validated_main_component_ratio=1.0 if measured else 0.0,
        status="completed",
    )


def publish_localization_manifest(workspace: Workspace, manifest: LocalizationManifest) -> Path:
    relative = f"07_reconstruction/localization/{manifest.localization_run_id}.json"
    return workspace.write_json(relative, manifest.model_dump(mode="json"))


def load_anchor_selection(path: Path) -> AnchorSelectionManifest:
    return AnchorSelectionManifest.model_validate_json(path.read_text("utf-8"))


def load_localization_manifest(path: Path) -> LocalizationManifest:
    return LocalizationManifest.model_validate_json(path.read_text("utf-8"))


def manifest_digest(manifest: LocalizationManifest | AnchorSelectionManifest) -> str:
    payload = json.dumps(manifest.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "AnchorLocalizationRefused",
    "AssetResolver",
    "build_anchor_model",
    "LocalizationBackend",
    "LocalizationHypothesis",
    "H5PnPLocalizationBackend",
    "ModelPose",
    "digest_model",
    "evaluate_localization_hypothesis",
    "load_anchor_selection",
    "load_localization_manifest",
    "load_model_poses",
    "manifest_digest",
    "publish_anchor_selection",
    "publish_anchor_model",
    "publish_localization_manifest",
    "run_anchor_localization",
    "select_anchor_core",
]
