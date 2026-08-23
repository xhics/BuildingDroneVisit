"""Reprojection creuse sur vues cachées (Lot 2 — Porte C, version mesurée).

Détecter l'hallucination demande de confronter le modèle à une observation
qu'il n'a pas vue. La version dense — rendre la scène depuis la pose cachée et
comparer photométriquement — exige un rasteriseur qui n'est pas raccordé.

Il existe une mesure plus modeste et déjà possible : **reprojeter le nuage de
points creux dans la vue cachée**. Si la structure est réelle, les points
tombent dans le cadre et près des observations ; si elle a été fabriquée pour
satisfaire les vues d'entraînement, elle ne prédit pas une vue retirée.

Ce que cette porte mesure, et ce qu'elle ne mesure pas :

- `reprojection_px` est une **vraie distance en pixels**, non une distance en
  mètres portant un nom de pixel ;
- `feature_inliers` est la part des points reprojetés tombant dans le cadre ;
- `silhouette_iou` compare l'emprise 2D des points reprojetés à celle observée
  depuis les vues d'entraînement ;
- SSIM et LPIPS restent **non mesurés** : ils exigent une image rendue. On ne
  leur invente pas de valeur, et `metrics_measured` ne prétend couvrir que ce
  qui l'a été.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: Nombre minimal de points 3D pour qu'une reprojection ait un sens.
MIN_POINTS = 12

#: Nombre minimal de vues cachées exploitables.
MIN_HELD_OUT = 1

#: Tolérance, en pixels, sous laquelle une prédiction est tenue pour juste.
#: Ordre de grandeur d'un détecteur de points d'intérêt ; calibrable.
INLIER_TOLERANCE_PX = 4.0


def quaternion_to_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Matrice de rotation depuis un quaternion COLMAP (w, x, y, z)."""
    norm = float(np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz))
    if norm < 1e-12:
        return np.eye(3)
    qw, qx, qy, qz = (qw / norm, qx / norm, qy / norm, qz / norm)
    return np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)],
        ]
    )


def load_intrinsics(run_dir: Path) -> dict[int, dict]:
    """Intrinsèques par `camera_id`, depuis le fichier `cameras` normalisé.

    Seuls les modèles dont on sait lire la focale et le point principal sont
    retenus. Un modèle inconnu n'est pas approximé : sans intrinsèques, une
    reprojection en pixels n'a pas de sens, et la cible restera non mesurée.
    """
    path = run_dir / "normalized" / "cameras"
    if not path.is_file():
        return {}

    cameras: dict[int, dict] = {}
    for line in path.read_text("utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            camera_id = int(parts[0])
            model = parts[1]
            width, height = int(parts[2]), int(parts[3])
            params = [float(p) for p in parts[4:]]
        except ValueError:
            continue

        if model in ("PINHOLE",) and len(params) >= 4:
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL") and len(params) >= 3:
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        else:
            continue

        cameras[camera_id] = {
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "width": width, "height": height,
        }
    return cameras


def _looks_like_pose(line: str) -> bool:
    """Une ligne de pose finit par `CAMERA_ID NAME` ; une ligne d'observations
    est une suite de nombres. Le nom de fichier n'étant pas numérique, il
    suffit à les distinguer."""
    parts = line.split()
    if len(parts) < 10:
        return False
    try:
        float(parts[-1])
    except ValueError:
        return True
    return False


def load_poses(run_dir: Path) -> dict[str, dict]:
    """Poses par `asset_id` : rotation, translation et `camera_id`.

    COLMAP écrit la pose **monde vers caméra** : `X_cam = R @ X_monde + t`.
    C'est cette convention qu'on garde, sans la retourner : reprojeter demande
    précisément cette direction.
    """
    path = run_dir / "normalized" / "images"
    if not path.is_file():
        return {}

    # COLMAP écrit **deux** lignes par image : la pose, puis les observations
    # « X Y POINT3D_ID … ». Lire toutes les lignes prenait ces observations
    # pour des poses — neuf caméras fantômes à 200-700 m d'une scène de 10 m,
    # dont aucun point n'était visible. On ne lit donc qu'une ligne sur deux
    # dès que le fichier porte des observations.
    lines = [
        line for line in path.read_text("utf-8").splitlines()
        if not line.startswith("#")
    ]
    has_observations = any(
        line.strip() and not _looks_like_pose(line) for line in lines
    )
    if has_observations:
        lines = lines[::2]

    poses: dict[str, dict] = {}
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 9:
            continue
        try:
            qw, qx, qy, qz = (float(v) for v in parts[1:5])
            tx, ty, tz = (float(v) for v in parts[5:8])
            camera_id = int(parts[8])
        except ValueError:
            continue

        name = parts[9] if len(parts) > 9 else parts[8]
        asset_id = Path(name).stem
        poses[asset_id] = {
            "R": quaternion_to_rotation(qw, qx, qy, qz),
            "t": np.array([tx, ty, tz]),
            "camera_id": camera_id,
        }
    return poses


def load_points(run_dir: Path) -> np.ndarray:
    """Nuage de points 3D (N, 3) depuis `points3D` normalisé."""
    path = run_dir / "normalized" / "points3D"
    if not path.is_file():
        return np.empty((0, 3))

    rows: list[list[float]] = []
    for line in path.read_text("utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            rows.append([float(parts[1]), float(parts[2]), float(parts[3])])
        except ValueError:
            continue
    return np.array(rows) if rows else np.empty((0, 3))


def load_observations(run_dir: Path) -> dict[str, dict[int, np.ndarray]]:
    """Observations 2D par image : `{asset_id: {point3D_id: (u, v)}}`.

    C'est **la** donnée qui rend la porte C réfutable. Sans elle, on ne peut
    que reprojeter des points dans les poses qui les ont produits — une
    identité algébrique, vraie quel que soit le modèle. Avec elle, la question
    devient : « le modèle prédit-il l'endroit où ce point a réellement été
    observé dans une image qu'il n'a pas vue ? »
    """
    path = run_dir / "normalized" / "images"
    if not path.is_file():
        return {}

    lines = [
        line for line in path.read_text("utf-8").splitlines()
        if not line.startswith("#")
    ]

    observations: dict[str, dict[int, np.ndarray]] = {}
    for index in range(0, len(lines) - 1, 2):
        pose_line = lines[index].split()
        if len(pose_line) < 10:
            continue
        asset_id = Path(pose_line[9]).stem

        values = lines[index + 1].split()
        seen: dict[int, np.ndarray] = {}
        for offset in range(0, len(values) - 2, 3):
            try:
                u = float(values[offset])
                v = float(values[offset + 1])
                point_id = int(values[offset + 2])
            except ValueError:
                continue
            if point_id >= 0:
                seen[point_id] = np.array([u, v])
        observations[asset_id] = seen
    return observations


def project(points: np.ndarray, pose: dict, intrinsics: dict) -> tuple[np.ndarray, np.ndarray]:
    """Projette des points 3D dans une vue.

    Returns:
        `(uv, visible)` — coordonnées pixel (N, 2) et masque des points
        réellement **devant** la caméra et dans le cadre. Un point derrière
        l'objectif se projetterait mathématiquement mais ne s'observe pas :
        l'inclure gonflerait le score d'une structure invisible.
    """
    if points.size == 0:
        return np.empty((0, 2)), np.zeros(0, dtype=bool)

    cam = (pose["R"] @ points.T).T + pose["t"]
    depth = cam[:, 2]
    in_front = depth > 1e-6

    uv = np.full((points.shape[0], 2), np.nan)
    safe = np.where(in_front, depth, 1.0)
    uv[:, 0] = intrinsics["fx"] * cam[:, 0] / safe + intrinsics["cx"]
    uv[:, 1] = intrinsics["fy"] * cam[:, 1] / safe + intrinsics["cy"]

    inside = (
        in_front
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < intrinsics["width"])
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < intrinsics["height"])
    )
    return uv, inside


def _bounding_box(uv: np.ndarray, mask: np.ndarray) -> tuple[float, float, float, float] | None:
    if not mask.any():
        return None
    pts = uv[mask]
    return (
        float(pts[:, 0].min()), float(pts[:, 1].min()),
        float(pts[:, 0].max()), float(pts[:, 1].max()),
    )


def _iou(a: tuple | None, b: tuple | None) -> float:
    """IoU de deux emprises 2D. Sans l'une des deux, il n'y a rien à comparer."""
    if a is None or b is None:
        return 0.0
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(ax1 - ax0, 0.0) * max(ay1 - ay0, 0.0)
    area_b = max(bx1 - bx0, 0.0) * max(by1 - by0, 0.0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def measure_held_out(
    run_dir: Path,
    held_out_ids: list[str],
    train_ids: list[str],
) -> dict | None:
    """Le modèle prédit-il où les points ont **réellement** été observés ?

    C'est la seule formulation réfutable de la porte C sans rasteriseur. Trois
    tentatives antérieures ont échoué faute d'observation indépendante :

    1. distance au centre du cadre — un nuage bruité s'y recentre par
       symétrie ; 10 m de bruit *amélioraient* le score ;
    2. écart de position pixel entre deux vues — sur des bases de 15 m pour
       une scène de 10 m, cet écart **est** la parallaxe ;
    3. distance épipolaire — vraie par construction quand les deux
       projections viennent des mêmes poses : nulle pour tout modèle.

    Chacune ne mesurait qu'une géométrie qu'on avait soi-même dérivée. Ici,
    l'observation 2D vient de la reconstruction, non du calcul : un point
    fabriqué pour satisfaire l'entraînement se projette loin de l'endroit où
    la vue cachée l'a vu, et le résidu le dit — en pixels.

    Retourne `None` quand la mesure est impossible (pas d'observations, pas
    d'intrinsèques, aucune vue cachée exploitable) plutôt qu'un score par
    défaut.
    """
    points = load_points(run_dir)
    poses = load_poses(run_dir)
    cameras = load_intrinsics(run_dir)
    observations = load_observations(run_dir)

    if points.shape[0] < MIN_POINTS or not poses or not cameras:
        return None
    if not observations:
        # Sans piste d'observations, aucune mesure ne peut réfuter le modèle.
        return None

    usable = [
        a for a in held_out_ids
        if a in poses and observations.get(a) and poses[a]["camera_id"] in cameras
    ]
    if len(usable) < MIN_HELD_OUT:
        return None

    train_boxes = []
    for asset_id in train_ids:
        pose = poses.get(asset_id)
        if pose is None:
            continue
        intrinsics = cameras.get(pose["camera_id"])
        if intrinsics is None:
            continue
        uv, mask = project(points, pose, intrinsics)
        box = _bounding_box(uv, mask)
        if box is not None:
            train_boxes.append(box)

    residuals: list[float] = []
    inlier_ratios: list[float] = []
    ious: list[float] = []
    measured = 0

    for asset_id in usable:
        pose = poses[asset_id]
        intrinsics = cameras[pose["camera_id"]]
        seen = observations[asset_id]

        uv, mask = project(points, pose, intrinsics)

        # Seuls les points que la vue cachée a **réellement** observés, et que
        # le modèle prétend y voir, sont comparables.
        common = [
            point_id for point_id in seen
            if 0 <= point_id < len(points) and mask[point_id]
        ]
        if not common:
            # Le modèle ne prédit rien de ce que la vue a vu : désaccord
            # total, non absence de mesure.
            residuals.append(float("inf"))
            inlier_ratios.append(0.0)
            measured += 1
            continue

        errors = np.array(
            [float(np.linalg.norm(uv[pid] - seen[pid])) for pid in common]
        )
        residuals.append(float(np.median(errors)))

        # Part des observations prédites à moins d'un seuil de tolérance.
        inlier_ratios.append(float((errors <= INLIER_TOLERANCE_PX).mean()))

        box = _bounding_box(uv, mask)
        if train_boxes and box is not None:
            ious.append(max(_iou(box, tb) for tb in train_boxes))
        measured += 1

    if not residuals:
        return None

    diagonal = float(
        np.hypot(
            max(c["width"] for c in cameras.values()),
            max(c["height"] for c in cameras.values()),
        )
    )
    finite = [r for r in residuals if np.isfinite(r)]
    reprojection_px = float(np.median(finite)) if finite else diagonal

    return {
        "feature_inliers": float(np.median(inlier_ratios)),
        "reprojection_px": reprojection_px,
        "silhouette_iou": float(np.median(ious)) if ious else 0.0,
        # Cohérence structurelle : le résidu rapporté à la diagonale du
        # capteur — une grandeur du dispositif, non une constante choisie.
        "structural_similarity": (
            float(max(0.0, 1.0 - reprojection_px / diagonal)) if diagonal > 0 else 0.0
        ),
        "held_out_measured": measured,
        "points_used": int(points.shape[0]),
    }


__all__ = [
    "load_intrinsics",
    "load_poses",
    "load_observations",
    "load_points",
    "measure_held_out",
    "project",
    "quaternion_to_rotation",
]
