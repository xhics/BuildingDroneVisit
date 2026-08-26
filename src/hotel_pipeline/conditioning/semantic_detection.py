"""Détection sémantique ouverte, guidée par texte et bornée par les poses.

Grounding DINO produit des boîtes et des classes candidates. Sur le poste Mac
Intel du pilote, SAM 2 ne peut pas être installé sans remplacer la version de
Torch verrouillée par le dépôt. Une segmentation GrabCut guidée par la boîte
fournit donc un contour local explicite ; un backend SAM 2 GPU pourra remplacer
ce contour sans changer le contrat d'observation.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..workspace import Workspace

DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
DEFAULT_SAM2_CHECKPOINT = "/workspace/.cache/sam2/sam2.1_hiera_large.pt"
DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_PROMPT = (
    "hotel building. hotel sign. road sign. traffic sign. entrance door. window. canopy. "
    "structural beam. support column. lamp post. air conditioning unit. "
    "balcony. gutter. deciduous tree. evergreen tree. "
    "car. truck. bus. person. bicycle. bush. hedge. fence. pole. planter. "
    "mobiliary. chair. table. trash can. flower pot."
)
TEXTURE_PROMPT = (
    "hotel building. hotel sign. road sign. traffic sign. entrance door. window. canopy. "
    "structural beam. support column. lamp post. air conditioning unit. "
    "balcony. gutter. deciduous tree. evergreen tree. "
    "car. truck. bus. person. bicycle. bush. hedge. fence. pole. planter. "
    "mobiliary. chair. table. trash can. flower pot."
)


class SemanticModelUnavailable(RuntimeError):
    """Le modèle ou ses poids ne sont pas accessibles selon la politique."""


@dataclass(frozen=True)
class SelectedImage:
    asset_id: str
    path: Path
    pose_evidence_class: str


def _read(path: Path) -> dict:
    return json.loads(path.read_text("utf-8"))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _latest_localization(workspace: Workspace) -> Path | None:
    folder = workspace.path("07_reconstruction", "localization")
    found = sorted(folder.glob("anchor-localization-*.json")) if folder.is_dir() else []
    return found[-1] if found else None


def _resolve_asset_path(workspace: Workspace, raw_path: str) -> Path | None:
    """Relocalise un chemin d'asset quand le workspace change de machine."""
    path = Path(raw_path)
    if path.is_file():
        return path
    if not path.is_absolute():
        candidate = workspace.root / path
        return candidate if candidate.is_file() else None
    parts = path.parts
    for index, part in enumerate(parts[:-1]):
        if part == "work" and parts[index + 1] == workspace.hotel_id:
            candidate = workspace.root.joinpath(*parts[index + 2 :])
            return candidate if candidate.is_file() else None
    return None


def resolve_device(requested: str = "auto") -> str:
    """Choisit CUDA seulement quand Torch confirme qu'il est utilisable."""
    value = requested.strip().lower()
    if value not in {"auto", "cpu", "cuda"}:
        raise ValueError("device doit valoir auto, cpu ou cuda")
    if value == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - extra optionnel
        if value == "cuda":
            raise SemanticModelUnavailable("Torch est absent : CUDA indisponible") from exc
        return "cpu"
    available = torch.cuda.is_available()
    if value == "cuda" and not available:
        raise SemanticModelUnavailable("CUDA demandé mais indisponible")
    return "cuda" if available else "cpu"


def select_validated_images(workspace: Workspace, limit: int = 4) -> list[SelectedImage]:
    """Croise identité positive et pose métriquement validée."""
    identity_path = workspace.path("09_confidence", "identity_screening.json")
    localization_path = _latest_localization(workspace)
    if not identity_path.is_file() or localization_path is None:
        return []
    identity = _read(identity_path)
    localization = _read(localization_path)
    poses = {
        str(item.get("asset_id")): item for item in localization.get("poses", [])
    }
    selected: list[SelectedImage] = []
    for asset in identity.get("assets", []):
        if asset.get("status") != "match" or not asset.get("path"):
            continue
        asset_id = str(asset.get("asset_id"))
        pose = poses.get(asset_id, {})
        if (
            pose.get("decision") != "accepted"
            or pose.get("evidence_class")
            not in {"anchor_measured", "measured_localized"}
        ):
            continue
        path = _resolve_asset_path(workspace, str(asset["path"]))
        if path is not None:
            selected.append(
                SelectedImage(
                    asset_id=asset_id,
                    path=path,
                    pose_evidence_class=str(pose["evidence_class"]),
                )
            )
        if len(selected) >= limit:
            break
    return selected


def canonical_class(label: str) -> str:
    """Ramène le texte libre du modèle au vocabulaire architectural."""
    value = label.lower().strip(" .")
    rules = (
        (("road sign", "traffic sign"), "road_sign"),
        (("evergreen tree", "conifer"), "tree_evergreen"),
        (("deciduous tree", "tree"), "tree_deciduous"),
        (("hotel building",), "building"),
        (("sign", "logo"), "sign"),
        (("door", "entrance"), "door"),
        (("window",), "window"),
        (("canopy", "awning"), "canopy"),
        (("beam",), "beam"),
        (("column", "pillar"), "column"),
        (("lamp", "light pole"), "lamp_post"),
        (("air conditioning", "air conditioner", "hvac"), "hvac_unit"),
        (("balcony",), "balcony"),
        (("gutter",), "gutter"),
        (("car", "vehicle", "automobile"), "car"),
        (("truck", "lorry", "van"), "truck"),
        (("person", "pedestrian", "man", "woman"), "person"),
        (("bicycle", "bike"), "bicycle"),
        (("bush", "shrub", "hedge"), "bush"),
        (("bus",), "bus"),
        (("fence", "railing"), "fence"),
        (("pole", "pylon"), "pole"),
        (("mobiliary", "chair", "table", "trash", "bench"), "mobiliary"),
        (("flower", "planter", "pot"), "flower_pot"),
    )
    for needles, found in rules:
        if any(needle in value for needle in needles):
            return found
    return "architectural_object"


def box_guided_mask(image: np.ndarray, box: list[float]) -> dict | None:
    """Segmente un objet dans sa boîte et retourne un contour simplifié.

    GrabCut est une approximation locale et le contrat le nomme comme telle.
    La boîte sémantique reste l'observation primaire ; le contour ne gagne
    aucune autorité géométrique supplémentaire.
    """
    import cv2

    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width - 1, x2), min(height - 1, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None

    max_side = 1024
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        work = cv2.resize(
            image,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        work = image
    sx1, sy1, sx2, sy2 = [
        int(round(value * scale)) for value in (x1, y1, x2, y2)
    ]
    rect = (sx1, sy1, max(2, sx2 - sx1), max(2, sy2 - sy1))
    mask = np.zeros(work.shape[:2], dtype=np.uint8)
    background = np.zeros((1, 65), dtype=np.float64)
    foreground = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(
            work,
            mask,
            rect,
            background,
            foreground,
            3,
            cv2.GC_INIT_WITH_RECT,
        )
    except cv2.error:
        return None
    binary = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    contours, _hierarchy = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour)) / max(scale * scale, 1e-9)
    if area < 12.0:
        return None
    epsilon = max(1.5, 0.01 * cv2.arcLength(contour, True))
    polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    points = np.round(polygon.astype(float) / scale, 1).tolist()
    return {
        "type": "polygon",
        "points": points,
        "area_px2": round(area, 1),
        "method": "box_guided_grabcut_v1",
        "confidence_role": "approximate_extent",
    }


def mask_to_polygon(
    binary: np.ndarray,
    *,
    method: str,
    score: float | None = None,
) -> dict | None:
    """Convertit un masque booléen en contour JSON compact et inspectable."""
    import cv2

    mask = np.asarray(binary, dtype=np.uint8) * 255
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < 12.0:
        return None
    epsilon = max(1.5, 0.006 * cv2.arcLength(contour, True))
    polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    has_holes = bool(hierarchy is not None and hierarchy[0, 0, 3] >= 0)
    payload: dict[str, object] = {
        "type": "polygon",
        "points": polygon.astype(float).tolist(),
        "area_px2": round(area, 1),
        "method": method,
        "confidence_role": "approximate_extent",
        "has_holes": has_holes,
    }
    if score is not None:
        payload["mask_score"] = round(float(score), 5)
    return payload


class Sam2BoxBackend:
    """SAM 2.1 guidé par les boîtes Grounding DINO, sur GPU Linux."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        device: str,
        model_config: str = DEFAULT_SAM2_CONFIG,
    ) -> None:
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:  # pragma: no cover - extra GPU optionnel
            raise SemanticModelUnavailable("SAM 2 n'est pas installé") from exc
        if not checkpoint.is_file():
            raise SemanticModelUnavailable(f"checkpoint SAM 2 absent : {checkpoint}")
        try:
            model = build_sam2(model_config, str(checkpoint), device=device)
        except (OSError, RuntimeError) as exc:
            raise SemanticModelUnavailable(f"chargement SAM 2 impossible : {exc}") from exc
        self.predictor = SAM2ImagePredictor(model)
        self.device = device
        self.checkpoint = checkpoint
        self.model_config = model_config

    def set_image(self, image: np.ndarray) -> None:
        self.predictor.set_image(np.asarray(image).copy())

    def segment(self, box: list[float]) -> dict | None:
        import torch

        autocast_enabled = self.device.startswith("cuda")
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            masks, scores, _logits = self.predictor.predict(
                box=np.asarray(box, dtype=np.float32),
                multimask_output=False,
            )
        return mask_to_polygon(
            masks[0],
            method="sam2.1_hiera_large_box_prompt_v1",
            score=float(scores[0]),
        )


class GroundingDinoBackend:
    """Adaptateur minimal autour de l'API officielle Transformers."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        allow_download: bool = False,
        device: str = "cpu",
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:  # pragma: no cover - extra optionnel
            raise SemanticModelUnavailable(
                "installez l'extra `semantic-vision` pour Grounding DINO"
            ) from exc
        self.torch = torch
        self.device = device
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_id, local_files_only=not allow_download
            )
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_id, local_files_only=not allow_download
            ).to(device)
        except (OSError, RuntimeError) as exc:
            mode = "téléchargement autorisé" if allow_download else "cache local seulement"
            raise SemanticModelUnavailable(
                f"modèle {model_id!r} indisponible ({mode}) : {exc}"
            ) from exc
        self.model.eval()
        self.model_id = model_id

    def detect(
        self,
        image: Image.Image,
        prompt: str,
        threshold: float,
        text_threshold: float,
    ) -> list[dict]:
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.get("input_ids"),
            threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]
        labels = result.get("text_labels") or result.get("labels") or []
        return [
            {
                "label": str(label),
                "score": round(float(score), 5),
                "box_xyxy": [round(float(value), 2) for value in box],
            }
            for label, score, box in zip(labels, result["scores"], result["boxes"])
        ]


def _write_previews(
    workspace: Workspace,
    images: list[SelectedImage],
    observations: list[dict],
    run_id: str,
) -> list[str]:
    """Produit des images de contrôle sans altérer les sources."""
    folder = workspace.path("11_conditioning", "semantic_previews", run_id)
    folder.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for selected in images:
        image = Image.open(selected.path).convert("RGB")
        draw = ImageDraw.Draw(image)
        related = [o for o in observations if o["asset_id"] == selected.asset_id]
        for observation in related:
            x1, y1, x2, y2 = observation["geometry_2d"]["xyxy"]
            colour = "#00e5ff" if observation["segmentation_2d"] else "#ffb000"
            draw.rectangle((x1, y1, x2, y2), outline=colour, width=4)
            label = f"{observation['class']} {observation['detector_score']:.2f}"
            draw.text((x1 + 4, max(0, y1 - 16)), label, fill=colour)
            segmentation = observation["segmentation_2d"]
            if segmentation and len(segmentation.get("points", [])) >= 3:
                draw.line(
                    [tuple(point) for point in segmentation["points"]]
                    + [tuple(segmentation["points"][0])],
                    fill="#ff3b7f",
                    width=3,
                )
        target = folder / f"{selected.asset_id}.jpg"
        image.save(target, quality=91)
        outputs.append(str(target.relative_to(workspace.root)))
    return outputs


def run(
    workspace: Workspace,
    *,
    purpose: str = "observation",
    limit: int = 2,
    allow_download: bool = False,
    model_id: str = DEFAULT_MODEL_ID,
    prompt: str | None = None,
    threshold: float = 0.28,
    text_threshold: float = 0.25,
    device: str = "auto",
    segmentation: str = "auto",
    sam2_checkpoint: Path | None = None,
) -> tuple[Path, dict]:
    """Exécute le modèle sur les vues sélectionnées selon le purpose."""
    if purpose == "texture":
        images = _select_colmap_images(workspace)
        effective_prompt = prompt if prompt is not None else TEXTURE_PROMPT
    else:
        images = select_validated_images(workspace, limit=limit)
        effective_prompt = prompt if prompt is not None else DEFAULT_PROMPT

    if not images:
        raise SemanticModelUnavailable(
            "aucune image ne correspond au purpose demandé"
        )
    selected_device = resolve_device(device)
    backend = GroundingDinoBackend(
        model_id, allow_download=allow_download, device=selected_device
    )
    segmentation_choice = segmentation.strip().lower()
    if segmentation_choice not in {"auto", "sam2", "grabcut"}:
        raise ValueError("segmentation doit valoir auto, sam2 ou grabcut")
    checkpoint = sam2_checkpoint or Path(
        os.environ.get("SAM2_CHECKPOINT", DEFAULT_SAM2_CHECKPOINT)
    )
    sam_backend: Sam2BoxBackend | None = None
    if segmentation_choice in {"auto", "sam2"} and selected_device == "cuda":
        try:
            sam_backend = Sam2BoxBackend(checkpoint, device=selected_device)
        except SemanticModelUnavailable:
            if segmentation_choice == "sam2":
                raise
    elif segmentation_choice == "sam2":
        raise SemanticModelUnavailable("SAM 2 exige ici un device CUDA")
    segmentation_method = (
        "sam2.1_hiera_large_box_prompt_v1"
        if sam_backend is not None
        else "box_guided_grabcut_v1"
    )
    observations: list[dict] = []
    started = time.monotonic()
    mask_root = None
    texture_run_id = None
    if purpose == "texture":
        generated_at = datetime.now(UTC)
        run_id = f"texture-{generated_at.strftime('%Y%m%dT%H%M%SZ')}"
        texture_run_id = run_id
        mask_root = workspace.path("11_conditioning", "texture_view_masks", run_id)
        mask_root.mkdir(parents=True, exist_ok=True)

    for selected in images:
        pil_image = Image.open(selected.path).convert("RGB")
        cv_image = np.asarray(pil_image)[:, :, ::-1].copy()
        if sam_backend is not None:
            sam_backend.set_image(np.asarray(pil_image).copy())
        detections = backend.detect(
            pil_image, effective_prompt, threshold=threshold, text_threshold=text_threshold
        )
        for index, detection in enumerate(detections):
            mask = (
                sam_backend.segment(detection["box_xyxy"])
                if sam_backend is not None
                else box_guided_mask(cv_image, detection["box_xyxy"])
            )
            observation = {
                "observation_id": f"grounding-dino-{selected.asset_id}-{index:03d}",
                "asset_id": selected.asset_id,
                "class": canonical_class(detection["label"]),
                "raw_label": detection["label"],
                "geometry_2d": {
                    "type": "box",
                    "xyxy": detection["box_xyxy"],
                },
                "segmentation_2d": mask,
                "detector": "grounding_dino_transformers",
                "detector_model": model_id,
                "detector_score": detection["score"],
                "pose_status": "validated",
                "pose_evidence_class": getattr(selected, "pose_evidence_class", "anchor_measured"),
                "provenance_class": "SEMANTICALLY_CONSTRAINED",
                "geometry_3d": None,
                "triangulation_status": "blocked",
                "blockers": [
                    "cross-view instance correspondence not established"
                ],
            }
            observations.append(observation)

    generated_at = datetime.now(UTC)
    run_id = f"semantic-{generated_at.strftime('%Y%m%dT%H%M%SZ')}"
    previews = _write_previews(workspace, images, observations, run_id)
    payload = {
        "contract_version": 1,
        "run_id": run_id,
        "hotel_id": workspace.hotel_id,
        "generated_at": generated_at.isoformat(),
        "backend": "grounding_dino_transformers",
        "model_id": model_id,
        "device": selected_device,
        "prompt": effective_prompt,
        "threshold": threshold,
        "text_threshold": text_threshold,
        "segmentation_backend": segmentation_method,
        "sam2_status": "executed" if sam_backend is not None else "not_executed",
        "sam2_checkpoint": str(checkpoint) if sam_backend is not None else None,
        "inputs": [
            {
                "asset_id": item.asset_id,
                "path": str(item.path.relative_to(workspace.root)),
                "sha256": _digest(item.path),
                "pose_evidence_class": getattr(item, "pose_evidence_class", "anchor_measured"),
            }
            for item in images
        ],
        "summary": {
            "images": len(images),
            "observations": len(observations),
            "segmented": sum(o["segmentation_2d"] is not None for o in observations),
            "geometry_3d_created": 0,
            "runtime_seconds": round(time.monotonic() - started, 2),
        },
        "previews": previews,
        "observations": observations,
    }
    run_relative = f"11_conditioning/semantic_runs/{run_id}.json"
    payload["versioned_artifact"] = run_relative
    workspace.write_json(run_relative, payload)

    if purpose == "texture":
        texture_payload = {
            "contract_version": 1,
            "run_id": run_id,
            "hotel_id": workspace.hotel_id,
            "generated_at": generated_at.isoformat(),
            "purpose": "texture",
            "views": [],
            "inputs": payload["inputs"],
            "summary": payload["summary"],
        }
        view_index: dict[str, dict] = {}
        for obs in observations:
            aid = str(obs["asset_id"])
            if aid not in view_index:
                view_index[aid] = {
                    "asset_id": aid,
                    "image_path": str(next(
                        (i.path for i in images if i.asset_id == aid), ""
                    )),
                    "building_polygon": [],
                    "building_polygons": [],
                    "occluders_polygon": [],
                    "fidelity": "polygon_no_holes",
                    "classes_present": [],
                    "sign_regions": [],
                    "width": 0,
                    "height": 0,
                    "raster": "",
                }
            view = view_index[aid]
            cls = obs["class"]
            view["classes_present"].append(cls)
            seg = obs.get("segmentation_2d")
            if cls == "building" and seg:
                points = seg.get("points", [])
                if points:
                    view["building_polygons"].append(points)
                    view["building_polygon"] = points
                if seg.get("has_holes"):
                    view["fidelity"] = "polygon_with_occluders"
            elif cls in {"car", "truck", "bus", "person", "bicycle", "bush", "hedge",
                         "fence", "pole", "planter", "mobiliary", "flower_pot",
                         "tree_evergreen", "tree_deciduous"} and seg:
                if not view["occluders_polygon"]:
                    view["occluders_polygon"] = seg.get("points", [])
                if seg.get("has_holes"):
                    view["fidelity"] = "polygon_with_occluders"
            elif cls in {"sign", "logo"} and seg:
                view["sign_regions"].append({
                    "class": cls,
                    "points": seg.get("points", []),
                    "decision": "pending",
                })
        texture_payload["views"] = list(view_index.values())
        if purpose == "texture" and mask_root is not None:
            _persist_texture_rasters(
                workspace, mask_root, texture_run_id, images, texture_payload["views"]
            )
        tex_path = workspace.path("11_conditioning", "texture_view_masks.json")
        workspace.write_json(tex_path, texture_payload)
        return tex_path, texture_payload

    path = workspace.write_json(
        "11_conditioning/semantic_observations.json", payload
    )
    return path, payload


def _select_colmap_images(workspace: Workspace) -> list[SelectedImage]:
    """Sélectionne toutes les vues du modèle COLMAP d'ancre."""
    try:
        import pycolmap
    except ImportError:
        return []
    correspondence_path = workspace.path("11_conditioning", "semantic_correspondences.json")
    if not correspondence_path.is_file():
        return []
    correspondence = _read(correspondence_path)
    anchor_path = workspace.root / correspondence["sources"]["anchor_model_manifest"]
    if not anchor_path.is_file():
        return []
    anchor = _read(anchor_path)
    from .semantic_correspondence import _resolve_model_path
    model_path = _resolve_model_path(workspace, anchor_path, anchor)
    if not model_path or not model_path.is_dir():
        return []
    try:
        reconstruction = pycolmap.Reconstruction(str(model_path))
    except Exception:
        return []

    selected: list[SelectedImage] = []
    for img in reconstruction.images.values():
        name = Path(img.name).stem
        candidates = list(workspace.path("07_reconstruction").rglob(f"*{name}*"))
        if not candidates:
            candidates = list(workspace.path("00_manifest").rglob(f"*{name}*"))
        path = candidates[0] if candidates else None
        if path is None or not path.is_file():
            continue
        selected.append(SelectedImage(
            asset_id=name,
            path=path,
            pose_evidence_class="anchor_measured",
        ))
    return selected


def _rasterize_polygons(polygons: list[list[list[float]]], width: int, height: int) -> np.ndarray | None:
    """Rastérise une ou plusieurs polylignes de bâtiment en masque binaire (H, W)."""
    import cv2

    if not polygons or width <= 0 or height <= 0:
        return None
    mask = np.zeros((height, width), dtype=np.uint8)
    all_points: list[np.ndarray] = []
    for polygon in polygons:
        pts = np.asarray(polygon, dtype=np.int32)
        if pts.shape[0] >= 3:
            all_points.append(pts)
    if not all_points:
        return None
    cv2.fillPoly(mask, all_points, 1)
    return mask.astype(bool)


def _persist_texture_rasters(
    workspace: Workspace,
    mask_root: Path,
    texture_run_id: str | None,
    images: list[SelectedImage],
    views: list[dict],
) -> None:
    """Rasterise les masques de bâtiment avec les trous d'occulteurs.

    Écrit un PNG par vue dans ``mask_root/{asset_id}/building_mask.png`` où
    les pixels d'occulteurs (arbres, voitures, etc.) sont mis à 0,
    préservant ainsi les trous SAM originaux.
    """
    if not texture_run_id:
        return
    image_dims: dict[str, tuple[int, int]] = {}
    for selected in images:
        if selected.asset_id not in image_dims:
            try:
                with Image.open(selected.path) as src:
                    image_dims[selected.asset_id] = src.size
            except OSError:
                image_dims[selected.asset_id] = (0, 0)

    for view in views:
        asset_id = view.get("asset_id", "")
        dim = image_dims.get(asset_id, (0, 0))
        view["width"], view["height"] = dim
        if dim[0] <= 0 or dim[1] <= 0:
            continue
        polygons = view.get("building_polygons") or []
        occluder_polygons = view.get("occluders_polygon") or []
        if not polygons and not occluder_polygons:
            continue
        binary = _rasterize_polygons(polygons, dim[0], dim[1]) if polygons else None
        if occluder_polygons:
            occluder_mask = _rasterize_polygons([occluder_polygons], dim[0], dim[1])
            if occluder_mask is not None:
                if binary is None:
                    binary = np.zeros_like(occluder_mask)
                binary[occluder_mask] = False
        if binary is None:
            continue
        asset_dir = mask_root / asset_id
        asset_dir.mkdir(parents=True, exist_ok=True)
        raster_name = "building_mask.png"
        raster_path = asset_dir / raster_name
        Image.fromarray((binary * 255).astype(np.uint8), mode="L").save(raster_path)
        view["raster"] = f"texture_view_masks/{texture_run_id}/{asset_id}/{raster_name}"
        from ..integrity_digests import mask_raster_digest
        view["raster_digest"] = mask_raster_digest(
            binary.astype(np.uint8), asset_id=asset_id,
            pixel_transform=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            segmenter_version=str(view.get("segmenter_version") or "semantic-detection-v1"),
        )

