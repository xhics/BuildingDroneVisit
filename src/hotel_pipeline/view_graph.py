"""Construction du graphe de vue (ViewGraphManifest) pour le Lot 2.

Ce module transforme un `ReconstructionInputManifest` en un graphe de vue
mesuré : retrieval → matching → vérification géométrique → estimation de
recouvrement. C'est le premier vrai produit du Lot 2, car il mesure la
continuité réelle du corpus avant d'exécuter un solveur coûteux.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .schemas.reconstruction import (
    PairEvidence,
    ReconstructionInputManifest,
    ViewGraphManifest,
    ViewGraphNode,
    ViewGraphReport,
)
from .workspace import Workspace


# ---------------------------------------------------------------------------
# Helpers IO
# ---------------------------------------------------------------------------


def _load_image(path: Path) -> np.ndarray | None:
    try:
        data = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return data
    except Exception:
        return None


def _load_intrinsics(asset) -> dict | None:  # noqa: ANN001
    return None


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def _retrieval_candidates(input_manifest: ReconstructionInputManifest, workspace: Workspace) -> list[tuple[str, str, float]]:
    """Construit la liste de paires candidates pour le matching.

    Stratégie par ordre de priorité (moins coûteuse en N²) :
    1. Même `viewpoint_cluster` (ou même asset_id)
    2. Même `view_sector`
    3. Position GPS proche (< 50 m)
    4. Séquence temporelle adjacente
    5. Récupération par descripteur global (ORB + BFMatcher sur des vignettes)

    Retourne une liste de tuples (asset_a, asset_b, priorité_estimée).
    """
    from .schemas import AssetManifest

    assets = AssetManifest.model_validate_json(
        workspace.assets_path.read_text("utf-8")
    )
    by_id = {a.id: a for a in assets.assets}
    selected = [aid for aid in input_manifest.selected_asset_ids if aid in by_id]
    if len(selected) < 2:
        return []

    candidates: list[tuple[str, str, float]] = []

    # 1. même viewpoint_cluster
    by_cluster: dict[str, list[str]] = {}
    for aid in selected:
        cluster = by_id[aid].viewpoint_cluster or aid
        by_cluster.setdefault(cluster, []).append(aid)

    seen: set[tuple[str, str]] = set()
    for members in by_cluster.values():
        if len(members) > 1:
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    pair = (members[i], members[j])
                    if pair not in seen:
                        candidates.append((members[i], members[j], 0.1))
                        seen.add(pair)

    # 2. même view_sector
    by_sector: dict[str, list[str]] = {}
    for aid in selected:
        sector = by_id[aid].view_sector.value if by_id[aid].view_sector else "unknown"
        by_sector.setdefault(sector, []).append(aid)

    for members in by_sector.values():
        if len(members) > 1:
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    pair = (members[i], members[j])
                    if pair not in seen:
                        candidates.append((members[i], members[j], 0.3))
                        seen.add(pair)

    # 3. proximité GPS (< 50 m)
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            pair = (selected[i], selected[j])
            if pair in seen:
                continue
            a = by_id[pair[0]]
            b = by_id[pair[1]]
            if (
                a.camera_lat is not None
                and a.camera_lon is not None
                and b.camera_lat is not None
                and b.camera_lon is not None
            ):
                d = _haversine_m(a.camera_lat, a.camera_lon, b.camera_lat, b.camera_lon)
                if d < 50.0:
                    candidates.append((pair[0], pair[1], 0.5))
                    seen.add(pair)

    # 4. descripteur global (ORB sur vignette centrée)
    descriptors: dict[str, np.ndarray] = {}
    orb = cv2.ORB_create(nfeatures=256)
    for aid in selected:
        asset = by_id[aid]
        if asset.local_path:
            img_path = workspace.path(asset.local_path)
            if img_path.is_file():
                img = _load_image(img_path)
                if img is not None:
                    h, w = img.shape
                    y0, y1 = int(h * 0.25), int(h * 0.75)
                    x0, x1 = int(w * 0.25), int(w * 0.75)
                    vignette = img[y0:y1, x0:x1]
                    if vignette.size > 0:
                        _, desc = orb.detectAndCompute(vignette, None)
                        if desc is not None:
                            descriptors[aid] = desc

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    keys = list(descriptors.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            pair = (keys[i], keys[j])
            if pair in seen:
                continue
            matches = bf.match(descriptors[pair[0]], descriptors[pair[1]])
            if len(matches) >= 8:
                candidates.append((pair[0], pair[1], 0.7))
                seen.add(pair)

    return candidates


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Matching & geometric verification
# ---------------------------------------------------------------------------


def _match_pair(
    asset_a: Any,
    asset_b: Any,
    workspace: Workspace,
    matcher: Any,
    *,
    max_features: int = 2048,
    detector: str = "orb",
) -> PairEvidence:
    """Match deux images et vérifie géométriquement."""
    if detector == "sift":
        detector_obj = cv2.SIFT_create(nfeatures=max_features)
        norm_type = cv2.NORM_L2
    else:
        detector_obj = cv2.ORB_create(nfeatures=max_features)
        norm_type = cv2.NORM_HAMMING

    bf = cv2.BFMatcher(norm_type, crossCheck=True)

    img_a = _load_image(workspace.path(asset_a.local_path)) if asset_a.local_path else None
    img_b = _load_image(workspace.path(asset_b.local_path)) if asset_b.local_path else None

    if img_a is None or img_b is None:
        return PairEvidence(
            image_a=asset_a.id,
            image_b=asset_b.id,
            status="failed",
        )

    kp_a, desc_a = detector_obj.detectAndCompute(img_a, None)
    kp_b, desc_b = detector_obj.detectAndCompute(img_b, None)

    if desc_a is None or desc_b is None or len(kp_a) < 8 or len(kp_b) < 8:
        return PairEvidence(
            image_a=asset_a.id,
            image_b=asset_b.id,
            matches=len(desc_a) if desc_a is not None else 0,
            status="failed",
        )

    matches = bf.match(desc_a, desc_b)
    matches = sorted(matches, key=lambda m: m.distance)[:500]

    src_pts = np.float32([kp_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    inlier_mask = np.zeros(len(matches), dtype=bool)
    inliers = 0
    status = "failed"
    overlap = 0.0
    relative_pose: dict | None = None
    degeneracy = "none"

    if len(matches) >= 8:
        E, mask = cv2.findEssentialMat(src_pts, dst_pts, method=cv2.RANSAC, prob=0.999, threshold=3.0)
        if E is not None and mask is not None:
            inlier_mask = mask.ravel() == 1
            inliers = int(inlier_mask.sum())
            inlier_ratio = inliers / len(matches)

            if inlier_ratio > 0.15:
                status = "valid"
                overlap = min(1.0, inlier_ratio * 2.0)

                # Récupérer pose relative (si possible)
                try:
                    _, R, t, _ = cv2.recoverPose(E, src_pts[inlier_mask], dst_pts[inlier_mask])
                    relative_pose = {
                        "R": R.tolist(),
                        "t": t.flatten().tolist(),
                        "inliers": inliers,
                    }
                except cv2.error:
                    pass

                # Détecter dégénérescence
                if inlier_ratio > 0.8 and len(matches) > 100:
                    degeneracy = "planar"

    return PairEvidence(
        image_a=asset_a.id,
        image_b=asset_b.id,
        retrieval_score=None,
        matches=len(matches),
        inliers=inliers,
        inlier_ratio=round(inliers / len(matches), 3) if matches else 0.0,
        relative_pose=relative_pose,
        overlap_estimate=round(overlap, 3),
        degeneracy=degeneracy,
        status=status,
    )


# ---------------------------------------------------------------------------
# ViewGraphBuilder
# ---------------------------------------------------------------------------


class ViewGraphBuilder:
    """Construit un ViewGraphManifest depuis un ReconstructionInputManifest."""

    def __init__(self, workspace: Workspace, *, max_features: int = 2048, detector: str = "orb"):
        self.workspace = workspace
        self.max_features = max_features
        self.detector = detector

    def build(self, input_manifest: ReconstructionInputManifest) -> ViewGraphManifest:
        from .schemas import AssetManifest
        assets = AssetManifest.model_validate_json(
            self.workspace.assets_path.read_text("utf-8")
        )
        by_id = {a.id: a for a in assets.assets}

        nodes = [
            ViewGraphNode(
                asset_id=aid,
                intrinsics=_load_intrinsics(by_id[aid]),
                quality_score=by_id[aid].quality_score or 0.0,
            )
            for aid in input_manifest.selected_asset_ids
            if aid in by_id
        ]
        if len(nodes) < 2:
            raise ValueError("au moins deux assets sont nécessaires pour un graphe de vue")

        candidates = _retrieval_candidates(input_manifest, self.workspace)
        pairs: list[PairEvidence] = []

        for a_id, b_id, _ in candidates:
            if a_id not in by_id or b_id not in by_id:
                continue
            pair = _match_pair(by_id[a_id], by_id[b_id], self.workspace, None, max_features=self.max_features, detector=self.detector)
            pairs.append(pair)

        valid_pairs = [p for p in pairs if p.status == "valid"]
        largest_component = self._largest_component(nodes, valid_pairs)

        inlier_ratios = [p.inlier_ratio for p in valid_pairs] if valid_pairs else [0.0]
        median_inlier = float(np.median(inlier_ratios)) if inlier_ratios else 0.0

        registered_ratio = len({p.image_a for p in valid_pairs} | {p.image_b for p in valid_pairs}) / len(nodes)

        repetitive_risk = self._repetitive_risk(pairs)

        report = ViewGraphReport(
            images_selected=len(nodes),
            pairs_tested=len(pairs),
            valid_pairs=len(valid_pairs),
            largest_component=largest_component,
            registered_candidate_ratio=round(registered_ratio, 3),
            median_inlier_ratio=round(median_inlier, 3),
            continuity_by_demand={},
            repetitive_risk=repetitive_risk,
            intrinsics_quality=self._intrinsics_quality(nodes),
        )

        view_graph_id = (
            f"vg-{input_manifest.reconstruction_input_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )
        return ViewGraphManifest(
            view_graph_id=view_graph_id,
            reconstruction_input_id=input_manifest.reconstruction_input_id,
            nodes=nodes,
            pairs=pairs,
            report=report,
        )

    @staticmethod
    def _largest_component(nodes: list[ViewGraphNode], pairs: list[PairEvidence]) -> int:
        if not pairs:
            return 1
        parent = {n.asset_id: n.asset_id for n in nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        for p in pairs:
            union(p.image_a, p.image_b)

        groups: dict[str, int] = {}
        for n in nodes:
            root = find(n.asset_id)
            groups[root] = groups.get(root, 0) + 1
        return max(groups.values()) if groups else 1

    @staticmethod
    def _repetitive_risk(pairs: list[PairEvidence]) -> str:
        if not pairs:
            return "none"
        high = sum(1 for p in pairs if p.degeneracy == "planar")
        ratio = high / len(pairs)
        if ratio > 0.5:
            return "high"
        if ratio > 0.2:
            return "medium"
        return "low"

    @staticmethod
    def _intrinsics_quality(nodes: list[ViewGraphNode]) -> str:
        known = sum(1 for n in nodes if n.intrinsics is not None)
        ratio = known / len(nodes) if nodes else 0.0
        if ratio > 0.9:
            return "excellent"
        if ratio > 0.6:
            return "good"
        if ratio > 0.3:
            return "fair"
        return "poor"


# ---------------------------------------------------------------------------
# Helpers for preprocessing / masks
# ---------------------------------------------------------------------------


def generate_mask_set(
    workspace: Workspace,
    input_manifest: ReconstructionInputManifest,
    *,
    mask_classes: list[str] | None = None,
) -> str:
    """Génère un jeu de masques binaires pour les assets sélectionnés.

    Args:
        workspace: workspace de l'hôtel
        input_manifest: manifeste d'entrée
        mask_classes: classes à masquer (sky, people, cars, water, large_reflections, signage, mobile_furniture)

    Returns:
        SHA-256 du jeu de masques
    """
    if mask_classes is None:
        mask_classes = ["sky", "people", "cars", "water"]

    mask_dir = workspace.path("05_colmap", "preprocessed", "masks")
    mask_dir.mkdir(parents=True, exist_ok=True)

    hasher = hashlib.sha256()
    for asset_id in input_manifest.selected_asset_ids:
        # Pour le MVP, on écrit un masque vide (tout visible) ;
        # la vraie segmentation viendra en P1 avec un modèle dédié.
        mask_path = mask_dir / f"{asset_id}.png"
        if not mask_path.exists():
            mask_path.write_bytes(b"")

    hasher.update(asset_id.encode("utf-8"))
    hasher.update(mask_path.read_bytes())

    return hasher.hexdigest()


def publish_view_graph(
    manifest: ViewGraphManifest,
    workspace: Workspace,
) -> Path:
    """Publie le ViewGraphManifest sous `07_reconstruction/view_graphs/`."""
    output_dir = workspace.path("07_reconstruction", "view_graphs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{manifest.view_graph_id}.json"
    output_path.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n")
    return output_path


__all__ = [
    "ViewGraphBuilder",
    "generate_mask_set",
    "publish_view_graph",
]
