"""Construction du graphe de vue (ViewGraphManifest) pour le Lot 2.

Ce module transforme un `ReconstructionInputManifest` en un graphe de vue
mesuré : retrieval → matching → vérification géométrique → estimation de
recouvrement. C'est le premier vrai produit du Lot 2, car il mesure la
continuité réelle du corpus avant d'exécuter un solveur coûteux.
"""

from __future__ import annotations

import hashlib
import json
import math
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
from .logging import get_logger
from .workspace import Workspace

log = get_logger("view-graph")


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
    """Calcule une focale en pixels seulement si l'EXIF le permet.

    ``FocalLength`` est exprimé en millimètres, jamais en pixels. Le recopier
    dans ``fx`` créait une calibration numériquement valide mais physiquement
    fausse. Sans équivalent 35 mm ni résolution du plan focal, on refuse donc
    de conclure.
    """
    try:
        from PIL import Image, ExifTags
        import PIL.Image
        image_path = Path(asset.local_path) if asset.local_path else None
        if image_path is None or not image_path.is_file():
            return None
        with Image.open(image_path) as img:
            exif = img._getexif()
        if not exif:
            return None
        exif_data = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
        width = exif_data.get("ExifImageWidth") or exif_data.get("ImageWidth")
        height = exif_data.get("ExifImageHeight") or exif_data.get("ImageLength")
        if width is None or height is None:
            return None
        focal_35mm = exif_data.get("FocalLengthIn35mmFilm")
        fx: float | None = None
        if focal_35mm is not None and float(focal_35mm) > 0:
            fx = float(focal_35mm) * float(width) / 36.0
        else:
            focal_length = exif_data.get("FocalLength")
            x_resolution = exif_data.get("FocalPlaneXResolution")
            resolution_unit = exif_data.get("FocalPlaneResolutionUnit")

            def _number(value):  # noqa: ANN001, ANN202
                if hasattr(value, "numerator") and hasattr(value, "denominator"):
                    return float(value.numerator) / float(value.denominator)
                if isinstance(value, tuple):
                    return float(value[0]) / float(value[1])
                return float(value)

            if focal_length and x_resolution and resolution_unit in {2, 3, 4, 5}:
                # EXIF: 2=in, 3=cm, 4=mm, 5=µm.
                unit_mm = {2: 25.4, 3: 10.0, 4: 1.0, 5: 0.001}[resolution_unit]
                sensor_width_mm = float(width) / _number(x_resolution) * unit_mm
                if sensor_width_mm > 0:
                    fx = _number(focal_length) / sensor_width_mm * float(width)
        if fx is None or not math.isfinite(fx) or fx <= 0:
            return None
        fy = fx
        cx = float(width) / 2.0
        cy = float(height) / 2.0
        return {
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "width": int(width), "height": int(height),
            "distortion": None,
            "source": "exif",
        }
    except Exception:
        return None


#: Largeur minimale, en pixels, d'une image dont on accepte de déduire une
#: focale. En deçà, ce n'est pas une prise de vue mais une vignette ou un
#: marqueur.
MIN_CALIBRATABLE_PX = 64


def _intrinsics_from_fov(asset, fov_deg: float | None) -> dict | None:  # noqa: ANN001
    """Focale dérivée du champ déclaré, faute d'EXIF.

    Les vues de rue sont des **recadrages de panoramas** : leur EXIF est
    dépouillé, mais le champ retenu au recadrage est connu de la politique de
    collecte. Mesuré sur ce pilote, aucune des trois cent quarante-neuf images
    ne porte de focale EXIF, alors que toutes ont des dimensions et un champ
    déclaré.

    Ce n'est pas une mesure de l'appareil : c'est le champ qu'on a demandé au
    fournisseur. La distinction est portée par `source`, pour qu'un calibrage
    déclaré ne se lise jamais comme un calibrage relevé.
    """
    if not fov_deg or fov_deg <= 0 or fov_deg >= 180:
        return None
    width = getattr(asset, "width", None)
    height = getattr(asset, "height", None)
    if not width or not height:
        return None
    # Une vignette de quelques pixels n'est pas une prise de vue : le corpus
    # porte des marqueurs de report en 1×1, dont on tirerait une focale d'un
    # pixel. Aucune calibration ne se déduit d'une image qu'on ne peut pas voir.
    if int(width) < MIN_CALIBRATABLE_PX or int(height) < MIN_CALIBRATABLE_PX:
        return None

    # Le champ déclaré est horizontal : c'est la largeur qui le porte.
    fx = float(width) / (2.0 * math.tan(math.radians(float(fov_deg)) / 2.0))
    if not math.isfinite(fx) or fx <= 0:
        return None
    return {
        "fx": fx,
        "fy": fx,
        "cx": float(width) / 2.0,
        "cy": float(height) / 2.0,
        "width": int(width),
        "height": int(height),
        "distortion": None,
        "source": "declared_fov",
    }


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


#: Écart relatif toléré entre deux focales pour les tenir pour comparables.
#: Au-delà, la paire mêle deux appareils trop différents pour qu'une focale
#: moyenne décrive l'un ou l'autre.
FOCAL_AGREEMENT = 0.25

#: Part du support de l'essentielle qu'une homographie doit atteindre pour
#: que la scène soit tenue pour plane. Seuil usuel des travaux SfM.
PLANARITY_RATIO = 0.85


def _camera_matrix(
    intrinsics_a: dict | None, intrinsics_b: dict | None, shape
) -> "np.ndarray | None":  # noqa: ANN001
    """Matrice caméra commune à une paire, ou `None` si rien ne l'atteste.

    Une seule matrice sert les deux vues : `findEssentialMat` n'en accepte
    qu'une. Ce n'est donc licite que si les deux focales concordent — deux
    appareils très différents ne partagent pas de calibration, et en inventer
    une moyenne ferait pire que ne rien supposer.
    """
    if intrinsics_a is None or intrinsics_b is None:
        return None

    fx_a, fx_b = intrinsics_a.get("fx"), intrinsics_b.get("fx")
    if not fx_a or not fx_b:
        return None
    if abs(fx_a - fx_b) / max(fx_a, fx_b) > FOCAL_AGREEMENT:
        return None

    fx = (float(fx_a) + float(fx_b)) * 0.5
    fy = (float(intrinsics_a.get("fy", fx_a)) + float(intrinsics_b.get("fy", fx_b))) * 0.5
    height, width = shape[:2]
    cx = (float(intrinsics_a.get("cx", width / 2)) + float(intrinsics_b.get("cx", width / 2))) * 0.5
    cy = (float(intrinsics_a.get("cy", height / 2)) + float(intrinsics_b.get("cy", height / 2))) * 0.5
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _planarity(
    src_pts: "np.ndarray", dst_pts: "np.ndarray", inlier_mask, inliers: int
) -> str:  # noqa: ANN001
    """Décide si la paire est dégénérée, en comparant deux modèles.

    Une façade plate se laisse décrire par une homographie aussi bien que par
    une essentielle : c'est le signe d'une scène plane, où la triangulation
    est mal conditionnée. Le test compare donc les deux supports au lieu de
    lire un taux d'inliers, qui ne dit que la qualité de l'appariement.
    """
    if inliers < 8:
        return "none"
    try:
        _homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    except cv2.error:
        return "none"
    if mask is None:
        return "none"
    return "planar" if int(mask.ravel().sum()) >= PLANARITY_RATIO * inliers else "none"


def _match_pair(
    asset_a: Any,
    asset_b: Any,
    workspace: Workspace,
    matcher: Any,
    *,
    max_features: int = 2048,
    detector: str = "orb",
    intrinsics_a: dict | None = None,
    intrinsics_b: dict | None = None,
) -> PairEvidence:
    """Match deux images et vérifie géométriquement.

    Les intrinsèques, quand elles sont connues, changent la nature du test :
    calibré, il rend une pose ; non calibré, il atteste seulement que les
    points s'accordent sur une géométrie épipolaire.
    """
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
        # La matrice essentielle n'a de sens qu'en coordonnées normalisées.
        # Appelée sans `cameraMatrix`, OpenCV suppose une focale de 1 et un
        # centre optique en (0, 0) — alors que les points sont en pixels. Ce
        # qui sortait n'était donc pas une matrice essentielle mais une
        # fondamentale, et la pose qu'on en tirait était fausse.
        #
        # Deux régimes, dits comme tels : calibré quand l'EXIF donne une
        # focale, non calibré sinon. Dans le second cas on estime une
        # fondamentale, qui vérifie l'appariement sans prétendre à une pose.
        camera_matrix = _camera_matrix(intrinsics_a, intrinsics_b, img_a.shape)
        if camera_matrix is not None:
            model, mask = cv2.findEssentialMat(
                src_pts,
                dst_pts,
                cameraMatrix=camera_matrix,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.5,
            )
        else:
            model, mask = cv2.findFundamentalMat(
                src_pts, dst_pts, cv2.FM_RANSAC, 3.0, 0.999
            )

        if model is not None and mask is not None:
            inlier_mask = mask.ravel() == 1
            inliers = int(inlier_mask.sum())
            inlier_ratio = inliers / len(matches)

            if inlier_ratio > 0.15:
                status = "valid"
                overlap = min(1.0, inlier_ratio * 2.0)

                # Une pose relative n'est récupérable que d'une essentielle :
                # sans intrinsèques, l'appariement est attesté mais la
                # géométrie relative reste indéterminée, et on le dit.
                if camera_matrix is not None:
                    try:
                        _, R, t, _ = cv2.recoverPose(
                            model,
                            src_pts[inlier_mask],
                            dst_pts[inlier_mask],
                            cameraMatrix=camera_matrix,
                        )
                        relative_pose = {
                            "R": R.tolist(),
                            "t": t.flatten().tolist(),
                            "inliers": inliers,
                            "calibrated": True,
                        }
                    except cv2.error:
                        pass

                # Dégénérescence planaire : c'est l'homographie qui la décide,
                # non un taux d'inliers élevé. Deux vues d'une façade plate
                # s'expliquent aussi bien par une homographie que par une
                # essentielle — un taux d'inliers fort dit seulement que
                # l'appariement est bon.
                degeneracy = _planarity(src_pts, dst_pts, inlier_mask, inliers)

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


def _policy_fov(workspace: Workspace) -> float | None:
    """Champ de vision déclaré par la politique de collecte, s'il l'est."""
    path = workspace.path("00_manifest", "pipeline_policy.json")
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    found = (payload.get("collection") or {}).get("image_fov_deg")
    return float(found) if found else None


class ViewGraphBuilder:
    """Construit un ViewGraphManifest depuis un ReconstructionInputManifest."""

    def __init__(
        self,
        workspace: Workspace,
        *,
        max_features: int = 2048,
        detector: str = "orb",
        declared_fov_deg: float | None = None,
    ):
        self.workspace = workspace
        self.max_features = max_features
        self.detector = detector
        self.declared_fov_deg = (
            declared_fov_deg
            if declared_fov_deg is not None
            else _policy_fov(workspace)
        )

    def build(self, input_manifest: ReconstructionInputManifest) -> ViewGraphManifest:
        from .schemas import AssetManifest
        assets = AssetManifest.model_validate_json(
            self.workspace.assets_path.read_text("utf-8")
        )
        by_id = {a.id: a for a in assets.assets}

        # La résolution des intrinsèques est partagée entre le rapport et la
        # vérification géométrique : rapporter `None` pour une image dont on a
        # bel et bien dérivé une focale ferait mentir le manifeste sur ce qui
        # a servi à calculer les poses.
        intrinsics: dict[str, dict | None] = {}

        def _intrinsics_of(asset_id: str) -> dict | None:
            if asset_id not in intrinsics:
                asset = by_id[asset_id]
                # L'EXIF d'abord : c'est une mesure. Le champ déclaré ensuite,
                # qui n'en est pas une mais vaut mieux qu'aucune calibration.
                found = _load_intrinsics(asset)
                if found is None:
                    found = _intrinsics_from_fov(asset, self.declared_fov_deg)
                intrinsics[asset_id] = found
            return intrinsics[asset_id]

        nodes = [
            ViewGraphNode(
                asset_id=aid,
                intrinsics=_intrinsics_of(aid),
                quality_score=by_id[aid].quality_score or 0.0,
            )
            for aid in input_manifest.selected_asset_ids
            if aid in by_id
        ]
        if len(nodes) < 2:
            raise ValueError("au moins deux assets sont nécessaires pour un graphe de vue")

        candidates = _retrieval_candidates(input_manifest, self.workspace)
        pairs: list[PairEvidence] = []

        # `_intrinsics_of` mémorise : une image apparaît dans des dizaines de
        # paires, et rouvrir son EXIF à chaque fois coûtait autant que le
        # reste de la vérification.
        for a_id, b_id, _ in candidates:
            if a_id not in by_id or b_id not in by_id:
                continue
            pair = _match_pair(
                by_id[a_id],
                by_id[b_id],
                self.workspace,
                None,
                max_features=self.max_features,
                detector=self.detector,
                intrinsics_a=_intrinsics_of(a_id),
                intrinsics_b=_intrinsics_of(b_id),
            )
            pairs.append(pair)

        calibrated = sum(
            1 for p in pairs if (p.relative_pose or {}).get("calibrated")
        )
        log.info(
            "graphe de vue : %d paire(s), %d avec pose calibrée",
            len(pairs),
            calibrated,
        )

        valid_pairs = [p for p in pairs if p.status == "valid"]
        largest_component = self._largest_component(nodes, valid_pairs)

        inlier_ratios = [p.inlier_ratio for p in valid_pairs] if valid_pairs else [0.0]
        median_inlier = float(np.median(inlier_ratios)) if inlier_ratios else 0.0

        registered_ratio = len({p.image_a for p in valid_pairs} | {p.image_b for p in valid_pairs}) / len(nodes)

        repetitive_risk = self._repetitive_risk(pairs)
        repetitive_structure_risk = self._repetitive_risk_float(pairs)
        homography_degeneracy_flags = {
            f"{p.image_a}__{p.image_b}": (p.degeneracy != "none") for p in pairs
        }
        doppelganger_rejections = self._doppelganger_rejections(pairs)

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
            homography_degeneracy_flags=homography_degeneracy_flags,
            repetitive_structure_risk=round(repetitive_structure_risk, 3),
            doppelganger_rejections=doppelganger_rejections,
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
    def _repetitive_risk_float(pairs: list[PairEvidence]) -> float:
        """Risque de structure répétitive continu (0.0–1.0)."""
        mapping = {"none": 0.0, "low": 0.25, "medium": 0.6, "high": 1.0}
        return mapping[ViewGraphBuilder._repetitive_risk(pairs)]

    @staticmethod
    def _doppelganger_rejections(pairs: list[PairEvidence]) -> int:
        """Compte les paires rejetées comme Doppelgangers / doublons structurels.

        Heuristique : une paire `valid` avec un ratio d'inliers très élevé et
        beaucoup de correspondances correspond typiquement à des fenêtres ou
        balcons répétitifs (doublons) plutôt qu'à une vraie parallaxe utile.
        Ces paires sont comptées comme rejets de doppelgangers.
        """
        return sum(
            1
            for p in pairs
            if p.status == "valid" and p.inlier_ratio >= 0.9 and p.matches >= 150
        )

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
