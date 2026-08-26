"""Fusion photographique multi-vues pour les façades du viewer.

Les images enregistrées par COLMAP sont reprojetées dans chaque plan de mur
afin de produire des orthofaçades. La visibilité est prouvée pixel par pixel :
un atlas troué est le résultat attendu, et le proxy mesuré reprend sa place
sous les trous.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..geo.facade_visibility import LidarOcclusion, ProxyDepth
from ..geo.orthofacade import FacadePlane, plane_from_edge, rectify
from ..geo.facade_visibility import LidarOcclusion, measure_facade_alignment
from ..workspace import Workspace
from .semantic_correspondence import _resolve_model_path
from .semantic_registered_support import transform_points
from .texture_masks import load_texture_masks, TextureViewMask


TEXTURE_ALGORITHM_VERSION = 9

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
    found: list[tuple[str, Path]] = []
    seen: set[str] = set()
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
        brick_mask = (
            (r.astype(int) > g.astype(int) + 8)
            & (r.astype(int) > b.astype(int) + 12)
            & (r > 45)
            & (r < 225)
        )
        dark_mask = (r < 105) & (g < 115) & (b < 125)
        green_mask = (
            (g.astype(int) > r.astype(int) + 5)
            & (g.astype(int) > b.astype(int) + 3)
            & (g > 45)
            & (g < 190)
        )
        for target, mask in ((brick, brick_mask), (dark, dark_mask), (green, green_mask)):
            if int(mask.sum()) >= 25:
                target.append(np.median(pixels[mask], axis=0))

    def colour(samples: list[np.ndarray], fallback: tuple[int, int, int]) -> str:
        value = np.median(np.asarray(samples), axis=0) if samples else np.asarray(fallback)
        return "#" + "".join(f"{int(round(channel)):02x}" for channel in value)

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
    def __init__(self, image, camera, transform: dict):  # noqa: ANN001
        self.image = image
        self.camera = camera
        self.transform = transform
        self.f = float(camera.params[0])
        self.position = transform_points(
            np.asarray([image.projection_center()], dtype=float), **transform
        )[0]
        cam_from_world = image.cam_from_world
        if callable(cam_from_world):
            cam_from_world = cam_from_world()
        self.rotation = np.asarray(cam_from_world.rotation.matrix(), dtype=float)
        self.translation = np.asarray(cam_from_world.translation, dtype=float)

    # ------------------------------------------------------------------
    # Contrat caméra : le moteur de rendu (clip near-plane, découpage au
    # bord, profondeurs Z espace caméra) consomme ces attributs standards.
    # ------------------------------------------------------------------
    @property
    def R(self) -> np.ndarray:
        return self.rotation

    @property
    def t(self) -> np.ndarray:
        return self.translation

    @property
    def model(self) -> str:
        return str(self.camera.model_name if hasattr(self.camera, "model_name") else self.camera.model)

    @property
    def params(self) -> np.ndarray:
        return np.asarray(self.camera.params, dtype=float)

    @property
    def near_m(self) -> float:
        return 0.05

    def img_from_cam(self, points_cam: np.ndarray) -> np.ndarray:
        return np.asarray(self.camera.img_from_cam(points_cam), dtype=float)

    def _to_colmap(self, points: np.ndarray) -> np.ndarray:
        tr = self.transform
        aligned = (
            np.asarray(points, dtype=float)
            + tr["scene_origin_xyz"]
            - tr["registration_translation"]
        )
        aligned[:, :2] -= np.asarray(tr["projected_origin_xy"], dtype=float)
        aligned -= tr["sim3_translation"]
        return (aligned @ tr["sim3_rotation"]) / tr["sim3_scale"]

    def project(self, points: np.ndarray):  # noqa: ANN201
        raw = self._to_colmap(points)
        camera_points = raw @ self.rotation.T + self.translation
        depth = camera_points[:, 2]
        normalized = camera_points[:, :2] / np.maximum(depth[:, None], 1e-8)
        screen = np.asarray(self.camera.img_from_cam(normalized), dtype=float)
        return screen, depth


def _asset_for_model_image(
    workspace: Workspace, assets: list[dict], image_name: str
) -> tuple[str, Path] | None:
    stem = Path(image_name).stem
    simplified = stem
    for prefix in ("mapillary-", "illary-", "llary-", "street_view-"):
        if simplified.startswith(prefix):
            simplified = simplified[len(prefix):]
    for asset in assets:
        asset_id = str(asset.get("id", ""))
        local = _resolve_asset_path(workspace, asset.get("local_path"))
        if local is None:
            continue
        if asset_id == stem or asset_id == simplified:
            return asset_id, local
        if local.stem == stem or local.stem == simplified:
            return asset_id, local
        asset_checksum = asset.get("checksum")
        if asset_checksum and _file_checksum(local) == asset_checksum:
            return asset_id, local
    return None


def _file_checksum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _texture_registration_allowed(registration: dict) -> tuple[bool, str]:
    status = registration.get("status")
    if status != "accepted":
        return False, f"registration refusée ({status}) : pose non utilisée pour texturer"
    metrics = registration.get("metrics") or {}
    holdout = metrics.get("holdout") or metrics.get("fit") or {}
    p90 = holdout.get("p90_m")
    if p90 is not None and p90 > REGISTRATION_HOLDOUT_MAX_P90_M:
        return (
            False,
            f"registration imprécise pour texture (holdout p90={p90:.2f} m > "
            f"{REGISTRATION_HOLDOUT_MAX_P90_M} m)",
        )
    return True, ""


def _facade_polygon_mask(camera, plane: FacadePlane, width: int, height: int):
    import cv2

    corners = np.array(
        [
            plane.point(0.0, 0.0),
            plane.point(plane.length_m, 0.0),
            plane.point(plane.length_m, 1.0),
            plane.point(0.0, 1.0),
        ],
        dtype=np.float64,
    )
    screen, _depth = camera.project(corners)
    if screen is None:
        return None
    # Aucun clamp au bord : fillPoly découpe le polygone au rectangle image,
    # une façade à moitié hors champ ne laisse pas de bande artificielle.
    pts = np.round(screen).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def _build_triangles_from_payload(payload: dict) -> tuple[list[np.ndarray], list[int]]:
    """Triangles du textureur : lus dans le maillage canonique, jamais refaits.

    Chaque volume porte son `solid` — l'instance canonique partagée avec le
    renderer, la collision et l'export. L'extrusion locale de l'emprise et
    la nappe `rv/rf` ne sont plus des sources : elles doublaient les
    surfaces et divergeaient du maillage rendu. Le repli historique ne sert
    qu'aux payloads antérieurs au contrat canonique.
    """
    triangles: list[np.ndarray] = []
    face_ids: list[int] = []
    fid = 0
    for volume in payload.get("volumes", []):
        solid = volume.get("solid") or {}
        sv = solid.get("vertices") or []
        sf = solid.get("faces") or []
        if sv and sf:
            for face in sf:
                if len(face) >= 3:
                    tri = np.asarray([sv[idx] for idx in face[:3]], dtype=np.float64)
                    if tri.shape == (3, 3):
                        triangles.append(tri)
                        face_ids.append(fid)
                        fid += 1
            continue

        # Repli legacy : payload sans maillage canonique.
        fp = volume.get("fp") or []
        wh = volume.get("wh") or []
        h_default = float(volume.get("h") or 8.0)
        if len(fp) >= 3:
            for i in range(len(fp)):
                j = (i + 1) % len(fp)
                a = np.array([fp[i][0], fp[i][1], 0.0], dtype=np.float64)
                b = np.array([fp[j][0], fp[j][1], 0.0], dtype=np.float64)
                h_i = float(wh[i]) if i < len(wh) else h_default
                h_j = float(wh[j]) if j < len(wh) else h_default
                c = np.array([fp[j][0], fp[j][1], h_j], dtype=np.float64)
                d = np.array([fp[i][0], fp[i][1], h_i], dtype=np.float64)
                triangles.extend([[a, b, c], [a, c, d]])
                face_ids.extend([fid, fid + 1])
                fid += 2
        rv = volume.get("rv") or []
        rf = volume.get("rf") or []
        if rv and rf:
            for face in rf:
                if len(face) >= 3:
                    tri = np.asarray([rv[idx] for idx in face[:3]], dtype=np.float64)
                    if tri.shape == (3, 3):
                        triangles.append(tri)
                        face_ids.append(fid)
                        fid += 1
    return triangles, face_ids


def _pose_error_for_view(camera, plane: FacadePlane, building_mask: np.ndarray) -> tuple[float, float, int]:
    return measure_facade_alignment(camera, plane, building_mask=building_mask)


def _atlas_alpha(statuses: np.ndarray) -> np.ndarray:
    alpha = np.where(statuses, 255, 0).astype(np.uint8)
    return alpha


def build(workspace: Workspace, payload: dict) -> dict:
    """Construit les atlas et enrichit le payload. Échec explicite et non fatal."""
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
            result = {
                "status": "unavailable",
                "reason": refusal_reason,
                "appearance": appearance,
            }
            payload["appearance_profile"] = appearance
            return result
        correspondence = _read(correspondence_path)
        anchor_path = workspace.root / correspondence["sources"]["anchor_model_manifest"]
        anchor = _read(anchor_path)
        selection_path = workspace.path(
            "07_reconstruction", "anchors", f"{anchor['anchor_selection_id']}.json"
        )
        selection = _read(selection_path)
        model_path = _resolve_model_path(workspace, anchor_path, anchor)
        reconstruction = pycolmap.Reconstruction(str(model_path))

        model_files = [
            model_path / "cameras" / "cameras.txt",
            model_path / "images" / "images.txt",
            model_path / "points3D" / "points3D.txt",
        ]
        for p in model_files:
            if p.is_file():
                model_files_digest += p.read_bytes()
    except (ImportError, KeyError, OSError, ValueError) as exc:
        result = {"status": "unavailable", "reason": str(exc), "appearance": appearance}
        payload["appearance_profile"] = appearance
        return result

    target_signature = json.dumps(
        [{"fp": target.get("fp"), "h": target.get("h"), "wh": target.get("wh")}],
        sort_keys=True,
    ).encode("utf-8")
    reference_signature = "".join(
        hashlib.sha256(path.read_bytes()).hexdigest() for _asset_id, path in references
    ).encode("ascii")
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

    policy_thresholds = json.dumps(
        {
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
        },
        sort_keys=True,
    ).encode("utf-8")

    input_digest = hashlib.sha256(
        asset_manifest_path.read_bytes()
        + registration_path.read_bytes()
        + correspondence_path.read_bytes()
        + target_signature
        + reference_signature
        + model_files_digest
        + selection_digest
        + anchor_digest
        + masks_digest_bytes
        + policy_thresholds
        + str(TEXTURE_ALGORITHM_VERSION).encode("ascii")
    ).hexdigest()

    audit_path = workspace.path("11_conditioning", "facade_texture_audit.json")
    if audit_path.is_file():
        cached = _read(audit_path)
        cached_textures = cached.get("textures") or []
        if cached.get("input_digest") == input_digest and cached_textures and all(
            (workspace.path("11_conditioning") / item["path"]).is_file()
            for item in cached_textures
        ):
            payload["facade_textures"] = cached_textures
            payload["appearance_profile"] = cached.get("appearance") or appearance
            payload["reference_fusion"] = cached
            return cached

    sim3 = selection["metrics"]["sim3"]
    origin_xy = Transformer.from_crs(
        "EPSG:4326", registration["hypothesis"]["horizontal_crs"], always_xy=True
    ).transform(
        float(selection["metrics"]["enu_origin_lon"]),
        float(selection["metrics"]["enu_origin_lat"]),
    )
    transform = {
        "sim3_rotation": np.asarray(sim3["rotation"], dtype=float),
        "sim3_translation": np.asarray(sim3["translation"], dtype=float),
        "sim3_scale": float(sim3["scale"]),
        "projected_origin_xy": (float(origin_xy[0]), float(origin_xy[1])),
        "registration_translation": np.asarray(
            registration["hypothesis"]["translation_projected_m"], dtype=float
        ),
        "scene_origin_xyz": np.asarray(
            registration["hypothesis"]["scene_origin_projected_xyz"], dtype=float
        ),
    }

    views: list[tuple] = []
    view_ids: list[str] = []
    triangles, face_ids = _build_triangles_from_payload(payload)

    for model_image in reconstruction.images.values():
        resolved = _asset_for_model_image(workspace, asset_manifest.get("assets", []), model_image.name)
        if resolved is None:
            continue
        asset_id, path = resolved
        try:
            rgb = np.asarray(Image.open(path).convert("RGB"))
        except OSError:
            continue
        camera = _RegisteredCamera(
            model_image, reconstruction.cameras[model_image.camera_id], transform
        )
        mask_info = texture_masks.get(asset_id)
        building_mask = mask_info.building if mask_info else None
        occluder_mask = mask_info.occluders if mask_info else None

        proxy = ProxyDepth.render(camera, triangles, face_ids, rgb.shape[1], rgb.shape[0])
        laz_path = workspace.path("06_geo", "site.laz")
        laz_occ = LidarOcclusion.from_window(
            None,
            camera,
            transform["scene_origin_xyz"],
            camera.f,
            rgb.shape[1],
            rgb.shape[0],
        )
        if laz_path.is_file():
            from .laz_cache import read_window
            centre = tuple(float(v) for v in transform["scene_origin_xyz"][:2])
            window = read_window(laz_path, centre, 80.0)
            laz_occ = LidarOcclusion.from_window(
                window, camera, transform["scene_origin_xyz"], camera.f,
                rgb.shape[1], rgb.shape[0],
            )

        combined_mask = building_mask
        if occluder_mask is not None and combined_mask is not None:
            combined_mask = combined_mask & ~occluder_mask
        elif occluder_mask is not None:
            combined_mask = ~occluder_mask

        views.append((asset_id, rgb, camera, combined_mask, proxy, laz_occ))
        view_ids.append(asset_id)

    ring = fp
    signed_area = 0.5 * sum(
        ring[index][0] * ring[(index + 1) % len(ring)][1]
        - ring[(index + 1) % len(ring)][0] * ring[index][1]
        for index in range(len(ring))
    )
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

        plane = plane_from_edge(
            np.asarray([start[0], start[1], 0.0]),
            np.asarray([end[0], end[1], 0.0]),
            max(start_h, end_h),
            f"EDGE_{edge_index:02d}",
            top_z_start_m=start_h,
            top_z_end_m=end_h,
        )
        plane.normal *= outward_factor

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
            pose_px, pose_m, cols_used = _pose_error_for_view(camera, plane, combined_mask)
            if pose_m > 0.5:
                continue
            edge_views.append((asset_id, rgb, camera, vis, proxy, laz_occ, pose_m))

        if not edge_views:
            continue

        ortho = rectify(plane, edge_views, texel_m=0.12)
        if ortho.image is None or ortho.observed_fraction < 0.01:
            continue

        observed = np.asarray([item.is_observed for item in ortho.support]).reshape(
            ortho.height_px, ortho.width_px
        )
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
        textures.append(
            {
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
            }
        )

    result = {
        "status": "ready" if textures else "unavailable",
        "input_digest": input_digest,
        "method": "visibility-aware registered multi-view orthofacade",
        "registered_images_used": len(views),
        "registered_asset_ids": sorted(view_ids),
        "reference_images_catalogued": len(references),
        "textures": textures,
        "appearance": appearance,
        "contact_sheet": "facade_textures/reference_inventory_sheet.jpg",
        "status_images": status_images,
    }
    payload["facade_textures"] = textures
    payload["appearance_profile"] = appearance
    payload["reference_fusion"] = result

    _apply_feature_coverage(payload, textures)
    workspace.write_json("11_conditioning/facade_texture_audit.json", result)
    return result


def _apply_feature_coverage(payload: dict, textures: list[dict]) -> None:
    textures_by_edge = {item["edge_index"]: item for item in textures}
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
        min_u = float(np.min(feature_poly[:, 0]))
        max_u = float(np.max(feature_poly[:, 0]))
        min_v = float(np.min(feature_poly[:, 1])) if feature_poly.shape[1] > 1 else 0.0
        max_v = float(np.max(feature_poly[:, 1])) if feature_poly.shape[1] > 1 else 1.0
        area = (max_u - min_u) * (max_v - min_v)
        observed = float(texture.get("observed_fraction", 0.0))
        if area <= 0.0:
            feature["texture_coverage"] = 0.0
            feature["covered_by_photo"] = False
            continue
        feature["texture_coverage"] = observed
        feature["covered_by_photo"] = observed > 0.1


__all__ = ["build"]
