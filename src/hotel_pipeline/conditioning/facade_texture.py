"""Fusion photographique multi-vues pour les façades du viewer.

Les images enregistrées par COLMAP sont reprojetées dans chaque plan de mur
afin de produire des orthofaçades. Les autres photographies extérieures sont
utilisées pour estimer une palette robuste et construire un inventaire visuel.
La sortie reste traçable : chaque atlas indique ses vues, sa couverture et son
désaccord, au lieu de masquer les zones non observées.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..workspace import Workspace
from ..geo.orthofacade import plane_from_edge, rectify
from .semantic_correspondence import _resolve_model_path
from .semantic_registered_support import transform_points


TEXTURE_ALGORITHM_VERSION = 8

#: Erreur de reprojection (p90, en mètres) au-delà de laquelle une registration,
#: même « accepted » géométriquement, n'est pas assez précise pour plaquer une
#: photo au pixel sur une façade. Le gate géométrique tolère ~2,5 m ; la texture
#: exige la même exigence, pas moins.
TEXTURE_REGISTRATION_MAX_P90_M = 2.5


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


def _semantic_building_masks(workspace: Workspace) -> dict[str, np.ndarray]:
    """Masque du bâtiment par image, dégagé de tout ce qui s'interpose.

    Le masque est plein (le bâtiment est un volume, pas un trou) ; les objets
    qui se projettent devant le mur — arbre, voiture, personne, clôture, poteau,
    mobilier — y sont soustraits de la même façon pleine. Une face texturée ne
    doit jamais incorporer le pixel d'un objet mobile vu depuis la rue.
    """
    import cv2

    path = workspace.path("11_conditioning", "semantic_observations.json")
    if not path.is_file():
        return {}
    payload = _read(path)
    by_asset: dict[str, list[dict]] = {}
    input_paths = {
        str(item.get("asset_id")): workspace.root / str(item.get("path"))
        for item in payload.get("inputs", [])
    }
    for observation in payload.get("observations", []):
        by_asset.setdefault(str(observation.get("asset_id")), []).append(observation)
    masks: dict[str, np.ndarray] = {}
    # Tout ce qui, devant le mur, ne fait pas partie de la façade texturée.
    occluders = {
        "tree_evergreen", "tree_deciduous", "lamp_post", "road_sign", "sign",
        "car", "truck", "bus", "person", "bicycle", "bush", "fence", "pole",
        "mobiliary", "hvac_unit",
    }
    for asset_id, observations in by_asset.items():
        image_path = input_paths.get(asset_id)
        if image_path is None or not image_path.is_file():
            continue
        with Image.open(image_path) as source:
            width, height = source.size
        building_polygons = [
            item.get("segmentation_2d", {}).get("points") or []
            for item in observations
            if item.get("class") == "building"
        ]
        if not any(len(points) >= 3 for points in building_polygons):
            continue
        mask = np.zeros((height, width), dtype=np.uint8)
        for points in building_polygons:
            if len(points) >= 3:
                pts = np.asarray([[point[:2] for point in points]], dtype=np.int32)
                cv2.fillPoly(mask, [pts], 1)
        for item in observations:
            if item.get("class") not in occluders:
                continue
            points = item.get("segmentation_2d", {}).get("points") or []
            if len(points) >= 3:
                pts = np.asarray([[point[:2] for point in points]], dtype=np.int32)
                cv2.fillPoly(mask, [pts], 0)
        masks[asset_id] = mask.astype(bool)
    return masks


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
            simplified = simplified[len(prefix) :]
    for asset in assets:
        asset_id = str(asset.get("id", ""))
        local = _resolve_asset_path(workspace, asset.get("local_path"))
        if local is None:
            continue
        if asset_id.endswith(stem) or asset_id.endswith(simplified) or local.stem in {stem, simplified}:
            return asset_id, local
    return None


def _fill_texture(image: np.ndarray, observed: np.ndarray, brick_hex: str) -> np.ndarray:
    import cv2

    rgb = np.asarray(image, dtype=np.uint8)
    mask = observed.astype(np.uint8)
    brick = np.asarray(
        [int(brick_hex[index : index + 2], 16) for index in (1, 3, 5)],
        dtype=np.uint8,
    )
    if mask.any() and (~observed).any():
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        near_observation = cv2.dilate(mask, np.ones((13, 13), np.uint8), iterations=1)
        small_holes = ((near_observation > 0) & (mask == 0)).astype(np.uint8) * 255
        bgr = cv2.inpaint(bgr, small_holes, 4, cv2.INPAINT_TELEA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb[near_observation == 0] = brick
    if not mask.any():
        rgb[:] = brick
    alpha = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1) * 255
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.4)
    return np.flipud(np.dstack((rgb, alpha)))


def _supplemental_facade(
    workspace: Workspace, edge_index: int, atlas: np.ndarray
) -> tuple[np.ndarray, str | None]:
    """Aucune tuile ne doit être fabriquée à partir d'une source non vérifiée.

    Une image web non enregistrée ne porte ni pose ni segmentation : la coller
    sur une façade y introduirait des arbres, des voitures ou des passants
    comme s'ils faisaient partie du bâtiment, et forcer l'opacité empêcherait
    le proxy de reprendre sa place. Le proxy mesuré est ici l'issue honnête ;
    une référence ne remplace une capture multi-vues validée que si elle est
    d'abord recalée et segmentée comme les autres vues.
    """
    return atlas, None


def _texture_registration_allowed(registration: dict) -> tuple[bool, str]:
    """La texture n'utilise une registration que si elle est à la fois acceptée
    et assez précise au pixel pour plaquer une photo sur une façade.

    Le pipeline possède déjà ce contrôle pour la géométrie de scène ; le réutiliser
    ici évite qu'une pose refusée (ou trop lâche) ne projette l'image à plusieurs
    mètres de la façade réelle.
    """
    status = registration.get("status")
    if status != "accepted":
        return False, f"registration refusée ({status}) : pose non utilisée pour texturer"
    p90 = (registration.get("metrics") or {}).get("fit", {}).get("p90_m")
    if p90 is not None and p90 > TEXTURE_REGISTRATION_MAX_P90_M:
        return (
            False,
            f"registration imprécise pour texture (p90={p90:.2f} m > "
            f"{TEXTURE_REGISTRATION_MAX_P90_M} m)",
        )
    return True, ""


def _edge_height(target: dict, edge_index: int) -> float:
    """Hauteur du mur le long d'une arête : moyenne des hauteurs de ses deux
    extrémités (wall_top_m), jamais une hauteur constante pour tout le mur."""
    wh = target.get("wh") or []
    if edge_index < len(wh) and len(wh) >= 2:
        following = (edge_index + 1) % len(wh)
        try:
            return 0.5 * (float(wh[edge_index]) + float(wh[following]))
        except (TypeError, ValueError):
            pass
    return float(target.get("h") or 10.0)


def _facade_polygon_mask(camera, plane, width: int, height: int):
    """Masque du seul mur visé, projeté dans l'image par la pose enregistrée.

    On ne garde que les pixels tombant sur le rectangle 3D de *ce* mur : un
    bâtiment voisin ou un autre pan ne peut plus contaminer la texture, et les
    vues sans détection sémantique restent utilisables (le polygone fait foi).
    """
    import cv2

    corners = np.array(
        [
            plane.point(0.0, 0.0),
            plane.point(plane.length_m, 0.0),
            plane.point(plane.length_m, plane.height_m),
            plane.point(0.0, plane.height_m),
        ],
        dtype=np.float64,
    )
    screen, _depth = camera.project(corners)
    if screen is None:
        return None
    xs = np.clip(np.round(screen[:, 0]).astype(np.int32), 0, width - 1)
    ys = np.clip(np.round(screen[:, 1]).astype(np.int32), 0, height - 1)
    pts = np.stack([xs, ys], axis=1).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def _views_for_edge(views, plane):
    """Restreint chaque vue au polygone de ce mur, et y soustrait les occulteurs."""
    edge_views = []
    for asset_id, rgb, camera, base_mask in views:
        facade_mask = _facade_polygon_mask(camera, plane, rgb.shape[1], rgb.shape[0])
        if facade_mask is None:
            continue
        vis = facade_mask & base_mask if base_mask is not None else facade_mask
        if not vis.any():
            continue
        edge_views.append((asset_id, rgb, camera, vis))
    return edge_views


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
    target_signature = json.dumps(
        [
            {"fp": volume.get("fp"), "h": volume.get("h"), "wh": volume.get("wh")}
            for volume in payload.get("volumes", [])
            if volume.get("target")
        ],
        sort_keys=True,
    ).encode("utf-8")
    reference_signature = "".join(
        hashlib.sha256(path.read_bytes()).hexdigest() for _asset_id, path in references
    ).encode("ascii")
    input_digest = hashlib.sha256(
        asset_manifest_path.read_bytes()
        + registration_path.read_bytes()
        + correspondence_path.read_bytes()
        + target_signature
        + reference_signature
        + str(TEXTURE_ALGORITHM_VERSION).encode("ascii")
        + (
            workspace.path("11_conditioning", "semantic_observations.json").read_bytes()
            if workspace.path("11_conditioning", "semantic_observations.json").is_file()
            else b""
        )
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
    except (ImportError, KeyError, OSError, ValueError) as exc:
        result = {"status": "unavailable", "reason": str(exc), "appearance": appearance}
        payload["appearance_profile"] = appearance
        return result

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

    views = []
    view_ids = []
    semantic_masks = _semantic_building_masks(workspace)
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
        # Toute vue résolue est conservée : le masque sémantique (s'il existe)
        # sert d'occulteur, mais le polygone de façade projeté borne la vue même
        # sans détection. On ne jette plus les poses COLMAP valides.
        views.append((asset_id, rgb, camera, semantic_masks.get(asset_id)))
        view_ids.append(asset_id)

    target = next((volume for volume in payload.get("volumes", []) if volume.get("target")), None)
    textures = []
    if target and len(target.get("fp", [])) >= 3 and views:
        ring = target["fp"]
        signed_area = 0.5 * sum(
            ring[index][0] * ring[(index + 1) % len(ring)][1]
            - ring[(index + 1) % len(ring)][0] * ring[index][1]
            for index in range(len(ring))
        )
        outward_factor = 1.0 if signed_area >= 0 else -1.0
        for edge_index, start in enumerate(ring):
            end = ring[(edge_index + 1) % len(ring)]
            length = float(np.hypot(end[0] - start[0], end[1] - start[1]))
            if length < 5.0:
                continue
            plane = plane_from_edge(
                np.asarray([start[0], start[1], 0.0]),
                np.asarray([end[0], end[1], 0.0]),
                _edge_height(target, edge_index),
                f"EDGE_{edge_index:02d}",
            )
            plane.normal *= outward_factor
            edge_views = _views_for_edge(views, plane)
            if not edge_views:
                continue
            ortho = rectify(plane, edge_views, texel_m=0.12)
            if ortho.image is None or ortho.observed_fraction < 0.01:
                continue
            observed = np.asarray([item.is_observed for item in ortho.support]).reshape(
                ortho.height_px, ortho.width_px
            )
            atlas = _fill_texture(ortho.image, observed, appearance["brick"])
            atlas, supplemental_reference = _supplemental_facade(
                workspace, edge_index, atlas
            )
            image = Image.fromarray(atlas, "RGBA")
            if image.width > 2048:
                image.thumbnail((2048, 512))
            name = f"edge_{edge_index:02d}.png"
            image.save(output_dir / name, optimize=True)
            report = ortho.as_dict()
            textures.append(
                {
                    "edge_index": edge_index,
                    "path": f"facade_textures/{name}",
                    "width": image.width,
                    "height": image.height,
                    "observed_fraction": report["observed_fraction"],
                    "disagreement_fraction": report["disagreement_fraction"],
                    "views_used": report["provenance"].get("views_used", 0),
                    "supplemental_reference": supplemental_reference,
                }
            )

    result = {
        "status": "ready" if textures else "unavailable",
        "input_digest": input_digest,
        "method": "registered multi-view orthofacade with best-incidence fusion",
        "registered_images_used": len(views),
        "registered_asset_ids": sorted(view_ids),
        "reference_images_catalogued": len(references),
        "textures": textures,
        "appearance": appearance,
        "contact_sheet": "facade_textures/reference_inventory_sheet.jpg",
    }
    payload["facade_textures"] = textures
    payload["appearance_profile"] = appearance
    payload["reference_fusion"] = result
    workspace.write_json("11_conditioning/facade_texture_audit.json", result)
    return result


__all__ = ["build"]
