"""Direct multi-view photographic fusion onto CanonicalSceneMesh surfaces.

Production charts and UVs come exclusively from canonical triangles and stable
surface IDs. Unobserved texels remain UNKNOWN; every measured texel retains its
source images, GSD, view count, confidence, and photometric variance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..geo.facade_visibility import LidarOcclusion, ProxyDepth, measure_facade_alignment
from ..geo.orthofacade import FacadePlane, plane_from_edge, rectify
from ..logging import get_logger
from ..workspace import Workspace
from .canonical_images import CanonicalImageTable
from .semantic_correspondence import _resolve_model_path
from .semantic_registered_support import transform_points
from .texture_masks import TextureViewMask, align_mask_to_image, load_texture_masks

log = get_logger("conditioning-facade-texture")
TEXTURE_ALGORITHM_VERSION = 11
REGISTRATION_HOLDOUT_MAX_P90_M = 3.0


def _read(path: Path) -> dict:
    return json.loads(path.read_text("utf-8"))


def _resolve_asset_path(workspace: Workspace, raw: str | None) -> Path | None:
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.is_file():
        return candidate
    normalized = raw.replace("\\", "/")
    marker = f"/work/{workspace.hotel_id}/"
    if marker in normalized:
        relocated = workspace.root / normalized.split(marker, 1)[1]
        if relocated.is_file():
            return relocated
    return None


def _eligible_images(workspace: Workspace, manifest: dict) -> list[tuple[str, Path]]:
    found = []
    seen = set()
    for asset in manifest.get("assets", []):
        if asset.get("exterior_or_interior") != "exterior":
            continue
        if not (asset.get("target_building_visible") or asset.get("contains_building")):
            continue
        path = _resolve_asset_path(workspace, asset.get("local_path"))
        if path is None:
            continue
        digest = str(asset.get("checksum") or hashlib.sha256(path.read_bytes()).hexdigest())
        if digest in seen:
            continue
        seen.add(digest)
        found.append((str(asset.get("id")), path))
    production = workspace.path("12_production", "scene_8s", "references")
    if production.is_dir():
        for path in sorted(production.glob("*.jpg")):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest not in seen:
                seen.add(digest)
                found.append((f"production-{path.stem}", path))
    web_research = workspace.path("02_images", "reference_only", "web_research")
    if web_research.is_dir():
        for path in sorted(web_research.glob("*.jpg")):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest not in seen:
                seen.add(digest)
                found.append((f"web-research-{path.stem}", path))
    return found


def _appearance_profile(images: list[tuple[str, Path]]) -> dict:
    brick, dark, green = [], [], []
    readable = 0
    for _asset_id, path in images:
        try:
            image = Image.open(path).convert("RGB")
            image.thumbnail((256, 256))
        except OSError:
            continue
        readable += 1
        pixels = np.asarray(image, dtype=np.uint8).reshape(-1, 3)
        r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]
        brick_mask = (r.astype(int) > g.astype(int) + 8) & (r.astype(int) > b.astype(int) + 12) & (r > 45) & (r < 225)
        dark_mask = (r < 105) & (g < 115) & (b < 125)
        green_mask = (g.astype(int) > r.astype(int) + 5) & (g.astype(int) > b.astype(int) + 3) & (g > 45) & (g < 190)
        for target, mask in ((brick, brick_mask), (dark, dark_mask), (green, green_mask)):
            if int(mask.sum()) >= 25:
                target.append(np.median(pixels[mask], axis=0))
    def colour(samples, fallback):
        value = np.median(np.asarray(samples), axis=0) if samples else np.asarray(fallback)
        return "#" + "".join(f"{round(channel):02x}" for channel in value)
    return {
        "method": "robust colour consensus across every readable exterior reference",
        "images_catalogued": len(images),
        "images_readable": readable,
        "brick": colour(brick, (112, 73, 62)),
        "glass": colour(dark, (38, 54, 61)),
        "roof": colour(dark, (70, 82, 78)),
        "vegetation": colour(green, (72, 112, 78)),
    }


def _contact_sheet(images: list[tuple[str, Path]], output: Path) -> None:
    if not images:
        return
    columns, cell_w, cell_h = 6, 180, 132
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), (17, 25, 35))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (asset_id, path) in enumerate(images):
        try:
            image = Image.open(path).convert("RGB")
            image.thumbnail((cell_w - 8, cell_h - 28))
        except OSError:
            continue
        x, y = (index % columns) * cell_w, (index // columns) * cell_h
        sheet.paste(image, (x + (cell_w - image.width) // 2, y + 4))
        draw.text((x + 5, y + cell_h - 19), asset_id[:27], fill=(225, 232, 239), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=88)


class _RegisteredCamera:
    def __init__(self, image, camera, transform: dict):
        self.image = image
        self.camera = camera
        self.transform = transform
        self.f = float(camera.params[0])
        self.position = transform_points(np.asarray([image.projection_center()], dtype=float), **transform)[0]
        cam_from_world = image.cam_from_world
        if callable(cam_from_world):
            cam_from_world = cam_from_world()
        self.rotation = np.asarray(cam_from_world.rotation.matrix(), dtype=float)
        self.translation = np.asarray(cam_from_world.translation, dtype=float)

    @property
    def R(self):
        return self.rotation

    @property
    def t(self):
        return self.translation

    @property
    def model(self):
        return str(self.camera.model_name if hasattr(self.camera, "model_name") else self.camera.model)

    @property
    def params(self):
        return np.asarray(self.camera.params, dtype=float)

    @property
    def near_m(self):
        return 0.05

    def img_from_cam(self, points_cam):
        return np.asarray(self.camera.img_from_cam(points_cam), dtype=float)

    def _to_colmap(self, points: np.ndarray) -> np.ndarray:
        tr = self.transform
        aligned = np.asarray(points, dtype=float) + tr["scene_origin_xyz"] - tr["registration_translation"]
        aligned[:, :2] -= np.asarray(tr["projected_origin_xy"], dtype=float)
        aligned -= tr["sim3_translation"]
        return (aligned @ tr["sim3_rotation"]) / tr["sim3_scale"]

    def project(self, points: np.ndarray):
        raw = self._to_colmap(points)
        camera_points = raw @ self.rotation.T + self.translation
        depth = camera_points[:, 2]
        normalized = camera_points[:, :2] / np.maximum(depth[:, None], 1e-8)
        screen = np.asarray(self.camera.img_from_cam(normalized), dtype=float)
        return screen, depth


class _ProjectOnlyCamera:
    """Force proxy rasterization through the registered world projection."""

    def __init__(self, camera) -> None:
        self.camera = camera

    def project(self, points: np.ndarray):
        return self.camera.project(points)


def prepare_view_masks(mask_info: TextureViewMask | None, image_shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray | None] | None:
    if mask_info is None or mask_info.building is None:
        return None
    building = align_mask_to_image(mask_info.building, image_shape[:2], mask_info.transform)
    if building is None:
        return None
    occluders = None
    if mask_info.occluders is not None:
        occluders = align_mask_to_image(mask_info.occluders, image_shape[:2], mask_info.transform)
        if occluders is None and mask_info.transform is None:
            occluders = None
    return building, occluders


def _canonical_mesh_pose_error(
    proxy: ProxyDepth, building_mask: np.ndarray,
    first_face: int, face_count: int, focal_px: float,
) -> tuple[float, float]:
    """Boundary residual of the exact canonical mesh against image evidence."""
    from scipy.ndimage import binary_erosion, distance_transform_edt

    face_map = proxy.face_id_map
    if face_map is None:
        return float("inf"), float("inf")
    rendered = (face_map >= first_face) & (face_map < first_face + face_count)
    if not rendered.any() or not building_mask.any():
        return float("inf"), float("inf")
    rendered_edge = rendered ^ binary_erosion(rendered)
    observed_edge = building_mask ^ binary_erosion(building_mask)
    distances = distance_transform_edt(~observed_edge)
    values = distances[rendered_edge]
    error_px = float(np.median(values)) if values.size else float("inf")
    depths = proxy.depth[rendered]
    depth = float(np.median(depths[np.isfinite(depths)]))
    error_m = error_px * depth / max(float(focal_px), 1e-6)
    return error_px, error_m


def _texture_registration_allowed(registration: dict) -> tuple[bool, str]:
    status = registration.get("status")
    if status != "accepted":
        return False, f"registration refusée ({status}) : pose non utilisée pour texturer"
    metrics = registration.get("metrics") or {}
    holdout = metrics.get("holdout") or metrics.get("fit") or {}
    p90 = holdout.get("p90_m")
    if p90 is not None and p90 > REGISTRATION_HOLDOUT_MAX_P90_M:
        return False, f"registration imprécise pour texture (holdout p90={p90:.2f} m > {REGISTRATION_HOLDOUT_MAX_P90_M} m)"
    return True, ""


def _facade_polygon_mask(camera, plane: FacadePlane, width: int, height: int):
    import cv2
    corners = np.array([plane.point(0.0, 0.0), plane.point(plane.length_m, 0.0), plane.point(plane.length_m, 1.0), plane.point(0.0, 1.0)], dtype=np.float64)
    screen, _depth = camera.project(corners)
    if screen is None:
        return None
    pts = np.round(screen).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def _canonical_triangles_with_surfaces(payload: dict) -> tuple[list[np.ndarray], list[int], dict[str, list[int]]]:
    """Read only authoritative canonical meshes; never re-extrude payload fields."""
    triangles: list[np.ndarray] = []
    face_ids: list[int] = []
    edge_face_ids: dict[str, list[int]] = {}
    fid = 0
    for volume_index, volume in enumerate(payload.get("volumes", [])):
        solid = volume.get("solid") or {}
        sv = solid.get("vertices") or []
        sf = solid.get("faces") or []
        kinds = solid.get("face_kind") or ["wall"] * len(sf)
        surface_ids = solid.get("surface_ids") or []
        if not sv or not sf:
            raise ValueError(f"volume {volume_index} has no canonical solid mesh")
        for face_index, face in enumerate(sf):
            tri = np.asarray([sv[idx] for idx in face[:3]], dtype=np.float64)
            if tri.shape != (3, 3):
                continue
            triangles.append(tri)
            face_ids.append(fid)
            if face_index < len(surface_ids):
                edge_face_ids.setdefault(str(surface_ids[face_index]), []).append(fid)
            if face_index < len(kinds) and kinds[face_index] == "wall":
                edge_face_ids.setdefault(f"{volume_index}:walls", []).append(fid)
            fid += 1
    return triangles, face_ids, edge_face_ids


def canonical_texture_triangles(mesh):
    """Strict texture-projector entry point for the One Reality Model."""
    from ..reality_contract import require_canonical_mesh

    receipt = require_canonical_mesh(mesh, "texture_projector")
    triangles = [triangle for triangle, _face_id in mesh.triangles()]
    return triangles, mesh.triangle_ids.copy(), receipt


def _build_canonical_surface_atlases(
    workspace: Workspace, payload: dict, meshes: list, target_index: int,
    views: list[tuple], output_dir: Path, input_digest: str,
    appearance: dict, identity_report: dict, references: list,
    rejected_views: list, view_ids: list, mesh_receipts: list,
) -> dict:
    """Production P0-4 path: exact triangles -> physical-surface UV charts."""
    from .canonical_texture import TextureObservation, texture_surface

    mesh = meshes[target_index]
    face_offset = sum(len(candidate.faces) for candidate in meshes[:target_index])
    observations = [
        TextureObservation(
            image_id=asset_id, image=rgb, camera=camera,
            valid_mask=combined_mask, proxy_depth=proxy,
            lidar_occlusion=lidar_occ, face_id_offset=face_offset,
            pose_error_m=pose_error_m,
        )
        for asset_id, rgb, camera, combined_mask, proxy, lidar_occ, pose_error_m in views
    ]
    textures: list[dict] = []
    for surface_id, surface in sorted(mesh.surface_catalog.items()):
        if surface.kind not in {"facade", "roof", "roof_step"}:
            continue
        atlas = texture_surface(mesh, surface_id, observations, texel_size_m=0.12)
        safe_name = hashlib.sha256(surface_id.encode("utf-8")).hexdigest()[:16]
        image_name = f"surface_{safe_name}.png"
        provenance_name = f"surface_{safe_name}_provenance.npz"
        Image.fromarray(np.flipud(atlas.rgba), "RGBA").save(
            output_dir / image_name, optimize=True
        )
        np.savez_compressed(
            output_dir / provenance_name,
            state=atlas.state,
            best_source=atlas.best_source,
            effective_gsd=atlas.effective_gsd,
            incidence_deg=atlas.incidence_deg,
            sharpness=atlas.sharpness,
            view_count=atlas.view_count,
            confidence=atlas.confidence,
            variance=atlas.variance,
            source_mask=atlas.source_mask,
            source_image_ids=np.asarray(atlas.source_image_ids),
            uv_vertices=atlas.chart.uv_vertices,
            uv_triangles=atlas.chart.uv_triangles,
            triangle_ids=np.asarray(atlas.chart.triangle_ids),
        )
        measured = atlas.state == "MEASURED"
        finite_gsd = atlas.effective_gsd[measured]
        from ..texture_reality import (
            CameraTextureDemand,
            TextureEvidence,
            TextureRealityLevel,
            evaluate_texture_reality,
        )

        median_sharpness = float(np.median(atlas.sharpness[measured])) if measured.any() else 0.0
        median_confidence = float(np.median(atlas.confidence[measured])) if measured.any() else 0.0
        median_variance = float(np.median(atlas.variance[measured])) if measured.any() else float("inf")
        pose_confidence = float(np.median([
            observation.pose_confidence
            * np.exp(-((max(0.0, observation.pose_error_m) / 0.22) ** 2))
            for observation in observations
        ])) if observations else 0.0
        evidence = TextureEvidence(
            float(np.median(finite_gsd)) if finite_gsd.size else None,
            atlas.coverage, median_sharpness,
            float(np.median(atlas.view_count[measured])) if measured.any() else 0.0,
            pose_confidence,
            float(np.median(atlas.incidence_deg[measured])) if measured.any() else 90.0,
            float(np.exp(-median_variance / 400.0)) if np.isfinite(median_variance) else 0.0,
            1.0 - atlas.coverage,
        )
        reality_tiles = []
        tile_px = 64
        for y0 in range(0, atlas.chart.height_px, tile_px):
            for x0 in range(0, atlas.chart.width_px, tile_px):
                ys = slice(y0, min(y0 + tile_px, atlas.chart.height_px))
                xs = slice(x0, min(x0 + tile_px, atlas.chart.width_px))
                tile_measured = measured[ys, xs]
                tile_coverage = float(np.mean(tile_measured))
                tile_gsd = atlas.effective_gsd[ys, xs][tile_measured]
                tile_incidence = atlas.incidence_deg[ys, xs][tile_measured]
                tile_variance = atlas.variance[ys, xs][tile_measured]
                centre_uv = np.array([
                    (x0 + (xs.stop - x0) / 2.0) * atlas.chart.texel_size_m,
                    (y0 + (ys.stop - y0) / 2.0) * atlas.chart.texel_size_m,
                ])
                centre_world = (
                    atlas.chart.origin_world
                    + centre_uv[0] * atlas.chart.basis_u
                    + centre_uv[1] * atlas.chart.basis_v
                )
                reality_tiles.append({
                    "tile": [x0 // tile_px, y0 // tile_px],
                    "centre_world": centre_world.tolist(),
                    "normal": atlas.chart.normal.tolist(),
                    "effective_gsd_m": float(np.median(tile_gsd)) if tile_gsd.size else None,
                    "coverage": tile_coverage,
                    "sharpness": float(np.median(atlas.sharpness[ys, xs][tile_measured])) if tile_measured.any() else 0.0,
                    "view_count": float(np.median(atlas.view_count[ys, xs][tile_measured])) if tile_measured.any() else 0.0,
                    "pose_confidence": pose_confidence,
                    "incidence_deg": float(np.median(tile_incidence)) if tile_incidence.size else 90.0,
                    "photometric_consistency": float(np.exp(-np.median(tile_variance) / 400.0)) if tile_variance.size else 0.0,
                    "unknown_fraction": 1.0 - tile_coverage,
                })
        reality_profiles = {}
        for label, width_px in (("1080p_60deg", 1920), ("4k_60deg", 3840)):
            reality_profiles[label] = {
                level.value: evaluate_texture_reality(
                    evidence, CameraTextureDemand(100.0, 60.0, width_px),
                    required_level=level,
                ).min_safe_distance_m
                for level in (
                    TextureRealityLevel.SAFE_FOR_CLOSEUP,
                    TextureRealityLevel.SAFE_FOR_NOVEL_VIEW,
                )
            }
        render_triangles = []
        for local_face, mesh_face_index, triangle_id in zip(
            atlas.chart.uv_triangles,
            atlas.chart.world_triangles,
            atlas.chart.triangle_ids,
        ):
            uv_m = atlas.chart.uv_vertices[local_face]
            render_triangles.append({
                "triangle_id": int(triangle_id),
                "vertices": mesh.vertices[mesh.faces[int(mesh_face_index)]].tolist(),
                # PNG rows are flipped when the atlas is written above.
                "uv_px": [
                    [
                        float(uv[0] / atlas.chart.texel_size_m),
                        float(
                            atlas.chart.height_px - 1
                            - uv[1] / atlas.chart.texel_size_m
                        ),
                    ]
                    for uv in uv_m
                ],
            })
        textures.append({
            "surface_id": surface_id,
            "path": f"facade_textures/{image_name}",
            "provenance_path": f"facade_textures/{provenance_name}",
            "width": atlas.chart.width_px,
            "height": atlas.chart.height_px,
            "texel_size_m": atlas.chart.texel_size_m,
            "triangle_ids": list(atlas.chart.triangle_ids),
            "render_triangles": render_triangles,
            "observed_fraction": atlas.coverage,
            "unknown_fraction": 1.0 - atlas.coverage,
            "median_effective_gsd_m": (
                float(np.median(finite_gsd)) if finite_gsd.size else None
            ),
            "p90_effective_gsd_m": (
                float(np.percentile(finite_gsd, 90)) if finite_gsd.size else None
            ),
            "median_view_count": (
                float(np.median(atlas.view_count[measured])) if measured.any() else 0.0
            ),
            "median_incidence_deg": (
                float(np.median(atlas.incidence_deg[measured])) if measured.any() else None
            ),
            "median_sharpness": median_sharpness,
            "median_confidence": median_confidence,
            "median_variance": median_variance if np.isfinite(median_variance) else None,
            "pose_confidence": pose_confidence,
            "photometric_consistency": evidence.photometric_consistency,
            "texture_reality": reality_profiles,
            "reality_tile_size_px": tile_px,
            "reality_tiles": reality_tiles,
            "low_confidence_fraction": float(np.mean(atlas.confidence < 0.4)),
            "best_sources": sorted({
                atlas.source_image_ids[int(index)]
                for index in np.unique(atlas.best_source[atlas.best_source >= 0])
            }),
            "rejection_counts": atlas.rejection_counts,
        })
    result = {
        "status": "ready" if textures else "unavailable",
        "input_digest": input_digest,
        "method": "direct CanonicalSceneMesh surface UV multi-view robust fusion",
        "facade_plane_proxy_usage": 0,
        "registered_images_used": len(views),
        "registered_asset_ids": sorted(view_ids),
        "views_rejected_no_building_mask": sorted(rejected_views),
        "canonical_images": identity_report,
        "reference_images_catalogued": len(references),
        "textures": textures,
        "appearance": appearance,
        "input_mesh_digests": [receipt.input_mesh_digest for receipt in mesh_receipts],
        "legacy_geometry_paths_used": 0,
    }
    payload["facade_textures"] = textures
    payload["appearance_profile"] = appearance
    payload["reference_fusion"] = result
    workspace.write_json("11_conditioning/facade_texture_audit.json", result)
    return result


def _build_triangles_from_payload(payload: dict) -> tuple[list[np.ndarray], list[int]]:
    """Backward-compatible public helper; edge ownership stays internal."""
    triangles, face_ids, _ = _canonical_triangles_with_surfaces(payload)
    return triangles, face_ids


def _pose_error_for_view(
    camera,
    plane: FacadePlane,
    building_mask: np.ndarray,
    facade_face_ids: set[int] | None = None,
) -> tuple[float, float, int]:
    return measure_facade_alignment(
        camera,
        plane,
        building_mask=building_mask,
        facade_face_ids=facade_face_ids,
    )


def _atlas_alpha(statuses: np.ndarray) -> np.ndarray:
    alpha = np.where(statuses, 255, 0).astype(np.uint8)
    return alpha


def build(workspace: Workspace, payload: dict) -> dict:
    asset_manifest_path = workspace.path("00_manifest", "asset_manifest.json")
    registration_path = workspace.path("11_conditioning", "vertical_registration.json")
    correspondence_path = workspace.path("11_conditioning", "semantic_correspondences.json")
    if not all(path.is_file() for path in (asset_manifest_path, registration_path, correspondence_path)):
        return {"status": "unavailable", "reason": "registered image inputs missing"}

    asset_manifest = _read(asset_manifest_path)
    references = _eligible_images(workspace, asset_manifest)
    appearance = _appearance_profile(references)
    output_dir = workspace.path("11_conditioning", "facade_textures")
    output_dir.mkdir(parents=True, exist_ok=True)
    _contact_sheet(references, output_dir / "reference_inventory_sheet.jpg")

    target = next((volume for volume in payload.get("volumes", []) if volume.get("target")), None)
    if not target:
        return {"status": "unavailable", "reason": "no target volume"}

    fp = target.get("fp") or []
    wh = target.get("wh") or []
    if len(fp) < 3:
        return {"status": "unavailable", "reason": "target footprint too small"}

    model_files_digest = b""
    selection_path = None
    model_path = None
    try:
        import pycolmap
        from pyproj import Transformer

        registration = _read(registration_path)
        allowed, refusal_reason = _texture_registration_allowed(registration)
        if not allowed:
            result = {"status": "unavailable", "reason": refusal_reason, "appearance": appearance}
            payload["appearance_profile"] = appearance
            return result
        correspondence = _read(correspondence_path)
        anchor_path = workspace.root / correspondence["sources"]["anchor_model_manifest"]
        anchor = _read(anchor_path)
        selection_path = workspace.path("07_reconstruction", "anchors", f"{anchor['anchor_selection_id']}.json")
        selection = _read(selection_path)
        model_path = _resolve_model_path(workspace, anchor_path, anchor)
        reconstruction = pycolmap.Reconstruction(str(model_path))

        model_files = [model_path / "cameras" / "cameras.txt", model_path / "images" / "images.txt", model_path / "points3D" / "points3D.txt"]
        for p in model_files:
            if p.is_file():
                model_files_digest += p.read_bytes()
    except (ImportError, KeyError, OSError, ValueError) as exc:
        result = {"status": "unavailable", "reason": str(exc), "appearance": appearance}
        payload["appearance_profile"] = appearance
        return result

    target_signature = json.dumps([{"fp": target.get("fp"), "h": target.get("h"), "wh": target.get("wh")}], sort_keys=True).encode("utf-8")
    reference_signature = "".join(hashlib.sha256(path.read_bytes()).hexdigest() for _asset_id, path in references).encode("ascii")
    selection_digest = selection_path.read_bytes() if selection_path and selection_path.is_file() else b""
    anchor_digest = anchor_path.read_bytes() if anchor_path and anchor_path.is_file() else b""
    texture_masks = load_texture_masks(workspace)
    masks_digest = hashlib.sha256()
    for k, v in sorted(texture_masks.items()):
        masks_digest.update(k.encode("utf-8"))
        masks_digest.update(v.fidelity.encode("utf-8") if v.fidelity else b"none")
        if v.building is not None:
            masks_digest.update(v.building.tobytes())
        if v.occluders is not None:
            masks_digest.update(v.occluders.tobytes())
    masks_digest_bytes = masks_digest.digest()

    policy_thresholds = json.dumps({
        "TEXEL_M_FACADE": 0.12,
        "MIN_PIXELS_PER_M": 2.0,
        "MAX_INCIDENCE_DEG": 65,
        "PROXY_DEPTH_TOLERANCE_M": 0.25,
        "LIDAR_CLASSES": [3, 4, 5],
        "LIDAR_OCCLUSION_MARGIN_M": 1.5,
        "POSE_MAX_ERROR_M": 0.5,
        "REGISTRATION_HOLDOUT_MAX_P90_M": 3.0,
        "MAD_OUTLIER_K": 2.5,
        "MIN_INLIERS_FOR_CONSENSUS": 2,
        "MAX_INLIER_SPREAD_DE": 12,
    }, sort_keys=True).encode("utf-8")

    input_digest = hashlib.sha256(
        asset_manifest_path.read_bytes() + registration_path.read_bytes() + correspondence_path.read_bytes() +
        target_signature + reference_signature + model_files_digest + selection_digest + anchor_digest +
        masks_digest_bytes + policy_thresholds + str(TEXTURE_ALGORITHM_VERSION).encode("ascii")
    ).hexdigest()

    audit_path = workspace.path("11_conditioning", "facade_texture_audit.json")
    if audit_path.is_file():
        cached = _read(audit_path)
        cached_textures = cached.get("textures") or []
        if cached.get("input_digest") == input_digest and cached_textures and all((workspace.path("11_conditioning") / item["path"]).is_file() for item in cached_textures):
            payload["facade_textures"] = cached_textures
            payload["appearance_profile"] = cached.get("appearance") or appearance
            payload["reference_fusion"] = cached
            return cached

    sim3 = selection["metrics"]["sim3"]
    origin_xy = Transformer.from_crs("EPSG:4326", registration["hypothesis"]["horizontal_crs"], always_xy=True).transform(float(selection["metrics"]["enu_origin_lon"]), float(selection["metrics"]["enu_origin_lat"]))
    transform = {
        "sim3_rotation": np.asarray(sim3["rotation"], dtype=float),
        "sim3_translation": np.asarray(sim3["translation"], dtype=float),
        "sim3_scale": float(sim3["scale"]),
        "projected_origin_xy": (float(origin_xy[0]), float(origin_xy[1])),
        "registration_translation": np.asarray(registration["hypothesis"]["translation_projected_m"], dtype=float),
        "scene_origin_xyz": np.asarray(registration["hypothesis"]["scene_origin_projected_xyz"], dtype=float),
    }

    views: list[tuple] = []
    view_ids: list[str] = []
    rejected_views: list[dict] = []
    triangles, face_ids, edge_face_ids = _canonical_triangles_with_surfaces(payload)
    from ..reality_contract import require_canonical_mesh
    from .canonical_mesh import CanonicalSceneMesh
    mesh_receipts = []
    canonical_meshes = []
    for volume in payload.get("volumes", []):
        mesh = CanonicalSceneMesh.from_dict(volume.get("solid") or {})
        canonical_meshes.append(mesh)
        mesh_receipts.append(require_canonical_mesh(mesh, "texture_projector"))

    target = next((volume for volume in payload.get("volumes", []) if volume.get("target")), None)
    target_volume_index = None
    for volume_index, volume in enumerate(payload.get("volumes", [])):
        if volume.get("target"):
            target_volume_index = volume_index
            break

    image_table = CanonicalImageTable.build(workspace, asset_manifest.get("assets", []), reconstruction, model_path)
    table_errors = image_table.validate()
    if table_errors:
        raise ValueError("table d'identite canonique invalide : " + "; ".join(table_errors))
    identity_report = {
        "resolved": sum(1 for record in image_table.records if record.colmap_image_id is not None),
        "assets_catalogued": len(image_table.records),
        "unresolved_colmap": list(image_table.unresolved_colmap),
        "ambiguous_assets": list(image_table.ambiguous_assets),
    }
    try:
        image_table.save(workspace)
    except OSError as exc:
        log.warning("table d'identite canonique non persistee : %s", exc)

    for model_image_id, model_image in sorted(reconstruction.images.items(), key=lambda item: int(item[0])):
        record = image_table.resolve_colmap(int(model_image_id))
        if record is None:
            continue
        path = workspace.root / record.normalized_path
        if not path.is_file():
            continue
        asset_id = record.asset_id
        try:
            rgb = np.asarray(Image.open(path).convert("RGB"))
        except OSError:
            continue
        camera = _RegisteredCamera(model_image, reconstruction.cameras[model_image.camera_id], transform)

        prepared = prepare_view_masks(texture_masks.get(asset_id), rgb.shape)
        if prepared is None:
            rejected_views.append((asset_id, "no_usable_building_mask"))
            continue
        building_mask, occluder_mask = prepared
        combined_mask = building_mask
        if occluder_mask is not None:
            combined_mask = building_mask & ~occluder_mask

        proxy = ProxyDepth.render(
            _ProjectOnlyCamera(camera), triangles, face_ids,
            rgb.shape[1], rgb.shape[0],
        )
        laz_path = workspace.path("06_geo", "site.laz")
        laz_occ = LidarOcclusion.from_window(None, camera, transform["scene_origin_xyz"], camera.f, rgb.shape[1], rgb.shape[0])
        if laz_path.is_file():
            from .laz_cache import read_window
            centre = tuple(float(v) for v in transform["scene_origin_xyz"][:2])
            window = read_window(laz_path, centre, 80.0)
            laz_occ = LidarOcclusion.from_window(window, camera, transform["scene_origin_xyz"], camera.f, rgb.shape[1], rgb.shape[0])

        target_face_offset = sum(
            len(candidate.faces) for candidate in canonical_meshes[:target_volume_index]
        )
        _pose_px, pose_m = _canonical_mesh_pose_error(
            proxy, building_mask, target_face_offset,
            len(canonical_meshes[target_volume_index].faces), camera.f,
        )
        views.append((asset_id, rgb, camera, combined_mask, proxy, laz_occ, pose_m))
        view_ids.append(asset_id)

    if target_volume_index is None or not canonical_meshes:
        raise ValueError("target canonical mesh is required for direct texturing")
    return _build_canonical_surface_atlases(
        workspace, payload, canonical_meshes, target_volume_index, views,
        output_dir, input_digest, appearance, identity_report, references,
        rejected_views, view_ids, mesh_receipts,
    )

    ring = fp
    signed_area = 0.5 * sum(ring[index][0] * ring[(index + 1) % len(ring)][1] - ring[(index + 1) % len(ring)][0] * ring[index][1] for index in range(len(ring)))
    outward_factor = 1.0 if signed_area >= 0 else -1.0

    textures = []
    status_images = []
    for edge_index, start in enumerate(ring):
        end = ring[(edge_index + 1) % len(ring)]
        length = float(np.hypot(end[0] - start[0], end[1] - start[1]))
        if length < 5.0:
            continue

        wh_len = len(wh)
        start_h = float(wh[edge_index]) if edge_index < wh_len else float(target.get("h") or 8.0)
        end_h = float(wh[(edge_index + 1) % wh_len]) if (edge_index + 1) % max(wh_len, 1) < wh_len else float(target.get("h") or 8.0)

        plane = plane_from_edge(np.asarray([start[0], start[1], 0.0]), np.asarray([end[0], end[1], 0.0]), max(start_h, end_h), f"EDGE_{edge_index:02d}", top_z_start_m=start_h, top_z_end_m=end_h)
        plane.normal *= outward_factor

        edge_key = f"{target_volume_index}:walls" if target_volume_index is not None else None
        facade_face_ids = set(edge_face_ids.get(edge_key, [])) if edge_key else None

        edge_views = []
        for asset_id, rgb, camera, combined_mask, proxy, laz_occ in views:
            if combined_mask is None:
                continue
            facade_mask = _facade_polygon_mask(camera, plane, rgb.shape[1], rgb.shape[0])
            if facade_mask is None:
                continue
            vis = facade_mask & combined_mask
            if not vis.any():
                continue
            _pose_px, pose_m, _cols_used = _pose_error_for_view(
                camera, plane, combined_mask, facade_face_ids=facade_face_ids
            )
            if pose_m > 0.5:
                continue
            edge_views.append((asset_id, rgb, camera, vis, proxy, laz_occ, pose_m))

        if not edge_views:
            continue

        ortho = rectify(plane, edge_views, texel_m=0.12)
        if ortho.image is None or ortho.observed_fraction < 0.01:
            continue

        observed = np.asarray([item.is_observed for item in ortho.support]).reshape(ortho.height_px, ortho.width_px)
        rgb_atlas = np.zeros((ortho.height_px, ortho.width_px, 3), dtype=np.uint8)
        rgb_atlas[observed] = ortho.image[observed]
        alpha = _atlas_alpha(observed)
        atlas = np.dstack((rgb_atlas, alpha))
        atlas = np.flipud(atlas)

        name = f"edge_{edge_index:02d}.png"
        image = Image.fromarray(atlas, "RGBA")
        if image.width > 2048:
            image.thumbnail((2048, 512))
        image.save(output_dir / name, optimize=True)

        status_name = f"edge_{edge_index:02d}_status.png"
        status_arr = np.zeros((ortho.height_px, ortho.width_px, 3), dtype=np.uint8)
        status_colours = {
            "OBSERVED_CONSENSUS": (82, 184, 92),
            "OBSERVED_SINGLE": (120, 200, 130),
            "REJECTED_DISAGREEMENT": (220, 90, 90),
            "REJECTED_OCCLUDED": (255, 159, 67),
            "REJECTED_SEMANTIC": (255, 176, 0),
            "REJECTED_POSE": (220, 90, 220),
            "REJECTED_RESOLUTION": (150, 150, 150),
            "non_observe": (30, 30, 30),
            "desaccord": (180, 80, 80),
            "vue_unique": (100, 160, 100),
            "accorde": (60, 120, 60),
        }
        for row in range(ortho.height_px):
            for col in range(ortho.width_px):
                texel = ortho.support[row * ortho.width_px + col]
                status_arr[row, col] = status_colours.get(texel.status, (50, 50, 50))
        status_img = Image.fromarray(np.flipud(status_arr), "RGB")
        if status_img.width > 2048:
            status_img.thumbnail((2048, 512))
        status_img.save(output_dir / status_name, optimize=True)
        status_images.append(status_name)

        report = ortho.as_dict()
        textures.append({
            "edge_index": edge_index,
            "path": f"facade_textures/{name}",
            "status_path": f"facade_textures/{status_name}",
            "width": image.width,
            "height": image.height,
            "observed_fraction": report["observed_fraction"],
            "disagreement_fraction": report["disagreement_fraction"],
            "views_used": report["provenance"].get("views_used", 0),
            "top_z_start_m": start_h,
            "top_z_end_m": end_h,
            "by_status": report.get("by_status", {}),
            "rejection_counts": report.get("provenance", {}).get("rejection_counts", {}),
        })

    result = {
        "status": "ready" if textures else "unavailable",
        "input_digest": input_digest,
        "method": "visibility-aware registered multi-view orthofacade",
        "registered_images_used": len(views),
        "registered_asset_ids": sorted(view_ids),
        "views_rejected_no_building_mask": sorted(rejected_views),
        "canonical_images": identity_report,
        "reference_images_catalogued": len(references),
        "textures": textures,
        "appearance": appearance,
        "contact_sheet": "facade_textures/reference_inventory_sheet.jpg",
        "status_images": status_images,
        "input_mesh_digests": [receipt.input_mesh_digest for receipt in mesh_receipts],
        "legacy_geometry_paths_used": 0,
    }
    payload["facade_textures"] = textures
    payload["appearance_profile"] = appearance
    payload["reference_fusion"] = result

    _apply_feature_coverage(payload, textures, workspace=workspace)
    workspace.write_json("11_conditioning/facade_texture_audit.json", result)
    return result


def _apply_feature_coverage(payload: dict, textures: list[dict], workspace=None) -> None:
    """Suppress synthetic detail only where its own footprint has photo support."""
    textures_by_edge = {item["edge_index"]: item for item in textures}
    target = next((v for v in payload.get("volumes", []) if v.get("target")), None) or {}
    ring = target.get("fp") or []
    for feature in payload.get("facade_features") or []:
        edge_index = feature.get("edge_index")
        if edge_index is None:
            continue
        texture = textures_by_edge.get(edge_index)
        if texture is None:
            feature["texture_coverage"] = 0.0
            feature["covered_by_photo"] = False
            continue
        vertices = feature.get("vertices") or []
        if len(vertices) < 3:
            feature["texture_coverage"] = 0.0
            feature["covered_by_photo"] = False
            continue
        feature_poly = np.asarray(vertices, dtype=float)
        observed = float(texture.get("observed_fraction", 0.0))
        area = 1.0
        if workspace is not None and 0 <= edge_index < len(ring):
            start = np.asarray(ring[edge_index][:2], dtype=float)
            end = np.asarray(ring[(edge_index + 1) % len(ring)][:2], dtype=float)
            tangent = end - start
            length = float(np.linalg.norm(tangent))
            if length > 1e-6:
                tangent /= length
                us = (feature_poly[:, :2] - start) @ tangent
                zs = feature_poly[:, 2] if feature_poly.shape[1] >= 3 else feature_poly[:, 1]
                min_u, max_u = float(us.min()), float(us.max())
                min_v, max_v = float(zs.min()), float(zs.max())
                area = (max_u - min_u) * (max_v - min_v)
                atlas_path = workspace.path("11_conditioning", *texture["path"].split("/"))
                if atlas_path.is_file():
                    alpha = np.asarray(Image.open(atlas_path).convert("RGBA"))[:, :, 3]
                    # Saved atlas is vertically flipped.
                    x0 = int(np.floor(min_u / max(length, 1e-9) * alpha.shape[1]))
                    x1 = int(np.ceil(max_u / max(length, 1e-9) * alpha.shape[1]))
                    height_m = max(float(texture.get("top_z_start_m", 0)), float(texture.get("top_z_end_m", 0)), 1e-9)
                    y0 = alpha.shape[0] - int(np.ceil(max_v / height_m * alpha.shape[0]))
                    y1 = alpha.shape[0] - int(np.floor(min_v / height_m * alpha.shape[0]))
                    crop = alpha[max(0, y0):min(alpha.shape[0], y1), max(0, x0):min(alpha.shape[1], x1)]
                    observed = float(np.mean(crop > 0)) if crop.size else 0.0
        if area <= 0.0:
            feature["texture_coverage"] = 0.0
            feature["covered_by_photo"] = False
            continue
        feature["texture_coverage"] = observed
        feature["covered_by_photo"] = observed >= 0.6


__all__ = ["build"]
