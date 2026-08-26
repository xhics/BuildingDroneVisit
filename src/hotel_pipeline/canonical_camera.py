"""Le contrat caméra : une seule définition, du SfM au viewer.

Une caméra mal partagée coûte quelques pixels de décalage — invisibles dans
un masque, rédhibitoires pour une projection de texture à trente mètres.
Ce module fixe le contrat que COLMAP, la projection de textures, la
validation, le z-buffer et le viewer consomment tous :

- ``CanonicalCamera`` : modèle exact COLMAP (PINHOLE, SIMPLE_PINHOLE,
  SIMPLE_RADIAL, RADIAL, OPENCV, FULL_OPENCV, OPENCV_FISHEYE) avec tous les
  coefficients de distorsion, la pose R|t et les plans near/far ;
- ``ImageLineage`` : la chaîne explicite original → EXIF → crop → resize,
  propagée aux intrinsèques comme aux masques ;
- ``camera_group_key`` : l'identité d'une vraie caméra (capteur, résolution,
  focale, source) — deux appareils différents ne partagent jamais leurs
  intrinsèques sous prétexte d'un optionnel ``single_camera``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: Modèles supportés, avec la longueur de leur vecteur de paramètres COLMAP.
CAMERA_MODELS: dict[str, int] = {
    "SIMPLE_PINHOLE": 3,   # f, cx, cy
    "PINHOLE": 4,          # fx, fy, cx, cy
    "SIMPLE_RADIAL": 4,    # f, cx, cy, k
    "RADIAL": 5,           # f, cx, cy, k1, k2
    "OPENCV": 8,           # fx, fy, cx, cy, k1, k2, p1, p2
    "OPENCV_FISHEYE": 8,   # fx, fy, cx, cy, k1, k2, k3, k4
    "FULL_OPENCV": 12,     # fx, fy, cx, cy, k1..k6, p1, p2 (COLMAP: k1,k2,p1,p2,k3,k4,k5,k6)
}

#: Distance near par défaut du frustum viewer, en mètres.
DEFAULT_NEAR_M = 0.1

#: Distance far par défaut du frustum viewer, en mètres.
DEFAULT_FAR_M = 2000.0


def _radial_undistorted(x: np.ndarray, y: np.ndarray, coeffs: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Distorsion radiale polynomiale : 1 + k1 r² + k2 r⁴ + ..."""
    r2 = x * x + y * y
    factor = np.ones_like(r2)
    r_power = np.ones_like(r2)
    for coefficient in coeffs:
        r_power = r_power * r2
        factor = factor + coefficient * r_power
    return x * factor, y * factor


class CanonicalCamera:
    """La caméra canonique : intrinsèques exacts, distorsion, pose, frustum.

    ``project`` reproduit fidèlement la chaîne COLMAP : passage caméra par
    R|t, division perspective, distorsion en coordonnées normalisées, puis
    application des focales et du point principal. Un point proche du bord
    se projette à moins d'un dixième de pixel du résultat COLMAP.
    """

    def __init__(
        self,
        model: str,
        width: int,
        height: int,
        params: np.ndarray | list[float],
        rotation: np.ndarray | None = None,
        translation: np.ndarray | None = None,
        near_m: float = DEFAULT_NEAR_M,
        far_m: float = DEFAULT_FAR_M,
        camera_id: str | None = None,
        group: str | None = None,
    ) -> None:
        model = str(model).upper()
        if model not in CAMERA_MODELS:
            raise ValueError(
                f"modèle de caméra inconnu : {model} — supportés : {sorted(CAMERA_MODELS)}"
            )
        self.model = model
        self.width = int(width)
        self.height = int(height)
        self.params = np.asarray(params, dtype=np.float64).reshape(-1)
        expected = CAMERA_MODELS[model]
        if self.params.size != expected:
            raise ValueError(
                f"{model} attend {expected} paramètres, reçu {self.params.size}"
            )
        self.R = (
            np.eye(3)
            if rotation is None
            else np.asarray(rotation, dtype=np.float64).reshape((3, 3))
        )
        self.t = (
            np.zeros(3)
            if translation is None
            else np.asarray(translation, dtype=np.float64).reshape(3)
        )
        self.near_m = float(near_m)
        self.far_m = float(far_m)
        self.camera_id = camera_id
        self.group = group

    # ------------------------------------------------------------------
    # Paramètres usuels
    # ------------------------------------------------------------------
    @property
    def focal(self) -> tuple[float, float]:
        """Focales (fx, fy) quel que soit le modèle."""
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            f = float(self.params[0])
            return (f, f)
        return (float(self.params[0]), float(self.params[1]))

    @property
    def principal(self) -> tuple[float, float]:
        """Point principal (cx, cy)."""
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            return (float(self.params[1]), float(self.params[2]))
        return (float(self.params[2]), float(self.params[3]))

    @property
    def K(self) -> np.ndarray:
        """Matrice intrinsèque pinhole équivalente (sans distorsion)."""
        fx, fy = self.focal
        cx, cy = self.principal
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    # ------------------------------------------------------------------
    # Projection — formules exactes COLMAP
    # ------------------------------------------------------------------
    def _distort(self, xd: np.ndarray, yd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Applique le modèle de distorsion en coordonnées normalisées."""
        p = self.params
        model = self.model

        if model in ("SIMPLE_PINHOLE", "PINHOLE"):
            return xd, yd
        if model == "SIMPLE_RADIAL":
            return _radial_undistorted(xd, yd, [float(p[3])])
        if model == "RADIAL":
            return _radial_undistorted(xd, yd, [float(p[3]), float(p[4])])
        if model == "OPENCV":
            k1, k2, p1, p2 = (float(v) for v in p[4:8])
            r2 = xd * xd + yd * yd
            radial = 1.0 + k1 * r2 + k2 * r2 * r2
            dx = 2.0 * p1 * xd * yd + p2 * (r2 + 2.0 * xd * xd)
            dy = p1 * (r2 + 2.0 * yd * yd) + 2.0 * p2 * xd * yd
            return xd * radial + dx, yd * radial + dy
        if model == "FULL_OPENCV":
            # COLMAP stocke : fx fy cx cy k1 k2 p1 p2 k3 k4 k5 k6
            k1, k2, p1, p2, k3, k4, k5, k6 = (float(v) for v in p[4:12])
            r2 = xd * xd + yd * yd
            r4 = r2 * r2
            r6 = r4 * r2
            radial = (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (
                1.0 + k4 * r2 + k5 * r4 + k6 * r6
            )
            dx = 2.0 * p1 * xd * yd + p2 * (r2 + 2.0 * xd * xd)
            dy = p1 * (r2 + 2.0 * yd * yd) + 2.0 * p2 * xd * yd
            return xd * radial + dx, yd * radial + dy
        if model == "OPENCV_FISHEYE":
            k1, k2, k3, k4 = (float(v) for v in p[4:8])
            r = np.hypot(xd, yd)
            theta = np.arctan(r)
            theta2 = theta * theta
            theta_d = theta * (
                1.0 + theta2 * (k1 + theta2 * (k2 + theta2 * (k3 + theta2 * k4)))
            )
            scale = np.where(r > 1e-12, theta_d / np.maximum(r, 1e-12), 1.0)
            return xd * scale, yd * scale
        raise AssertionError(model)  # pragma: no cover - table exhaustive

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Projette des points monde : pixels (N,2) et profondeurs (N,).

        La chaîne est celle de COLMAP, au floatant près : R|t puis division
        perspective, distorsion normalisée, focales et point principal.
        """
        points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
        camera_points = points @ self.R.T + self.t
        depth = camera_points[:, 2]
        safe_depth = np.where(np.abs(depth) < 1e-9, 1e-9, depth)
        xn = camera_points[:, 0] / safe_depth
        yn = camera_points[:, 1] / safe_depth
        ud, vd = self._distort(xn, yn)

        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            f = float(self.params[0])
            fx = fy = f
        else:
            fx, fy = float(self.params[0]), float(self.params[1])
        cx, cy = self.principal
        u = fx * ud + cx
        v = fy * vd + cy
        return np.column_stack([u, v]), depth

    def position(self) -> np.ndarray:
        """Centre optique dans le monde : −Rᵀ t."""
        return -self.R.T @ self.t

    # ------------------------------------------------------------------
    # Lignée pixel : crop / resize / EXIF (P19)
    # ------------------------------------------------------------------
    def adapt_to(
        self, transform: np.ndarray, new_width: int | None = None, new_height: int | None = None
    ) -> "CanonicalCamera":
        """Retourne la caméra adaptée à une transformation affine 2×3 pixels.

        Une transformation canonique s'écrit ``p_canonical = A · p_source + b``
        (par bloc supérieur 2×2 et translation). Les intrinsèques suivent :
        K' = A K Aᵀ⁻¹ ramené à la forme diagonale — implémenté directement
        sur fx, fy, cx, cy pour rester exact :

            fx' = sx · fx, fy' = sy · fy,
            cx' = sx · cx + tx (+ couplage si rotation/pure),
            cy' = sy · cy + ty.

        Les coefficients de distorsion sont inchangés : ils vivent en
        coordonnées normalisées, indépendantes des pixels.
        """
        matrix = np.asarray(transform, dtype=np.float64).reshape((2, 3))
        linear = matrix[:, :2]
        offset = matrix[:, 2]
        # Les coefficients de distorsion sont inchangés : ils vivent en
        # coordonnées normalisées, indépendantes des pixels.
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            new_params = self.params.copy()
            f_index, cx_index, cy_index = 0, 1, 2
        else:
            new_params = self.params.copy()
            f_index, cx_index, cy_index = None, 2, 3

        if (
            abs(linear[0, 1]) <= 1e-12 and abs(linear[1, 0]) <= 1e-12
        ):
            # Échelle pure (resize / crop) : focales et point principal
            # suivent l'échelle et la translation.
            sx = float(linear[0, 0])
            sy = float(linear[1, 1])
            fx, fy = self.focal
            cx, cy = self.principal
            if f_index is not None:
                new_params[f_index] = sx * fx
                new_params[cx_index] = sx * cx + offset[0]
                new_params[cy_index] = sy * cy + offset[1]
            else:
                new_params[0] = sx * fx
                new_params[1] = sy * fy
                new_params[cx_index] = sx * cx + offset[0]
                new_params[cy_index] = sy * cy + offset[1]
        elif bool(
            np.allclose(linear @ linear.T, np.eye(2))
        ) and bool(np.all(np.isin(np.round(linear), (-1.0, 0.0, 1.0)))):
            # Permutation signée (EXIF transpose : rotation de 90° ou miroir)
            # : le repère caméra tourne avec les axes pixel. K suit par
            # conjugaison Q·K·Qᵀ et la pose par Q·R — les distorsions
            # radiales survivent à la rotation ; les tangentielles se
            # permutent avec les axes, ce que la conjugaison de K traduit
            # au niveau des paramètres OPENCV.
            q = linear
            fx_old, fy_old = self.focal
            cx_old, cy_old = self.principal
            rotated_block = q @ np.array([[fx_old, 0.0], [0.0, fy_old]]) @ q.T
            rotated_centre = q @ np.array([cx_old, cy_old])
            fx_new = float(rotated_block[0, 0])
            fy_new = float(rotated_block[1, 1])
            cx_new = float(rotated_centre[0]) + float(offset[0])
            cy_new = float(rotated_centre[1]) + float(offset[1])
            if self.model == "OPENCV":
                # P' = Q P Qᵀ échange p1 et p2 sous une rotation d'un quart
                # de tour ; identité pour une rotation d'un demi-tour.
                p1, p2 = float(self.params[6]), float(self.params[7])
                if abs(float(round(q[0, 1]))) > 0.5:
                    new_params = self.params.copy()
                    new_params[6], new_params[7] = p2, p1
            if f_index is None:
                new_params[0] = fx_new
                new_params[1] = fy_new
                new_params[2] = cx_new
                new_params[3] = cy_new
            else:
                new_params[f_index] = max(fx_new, fy_new)
                new_params[cx_index] = cx_new
                new_params[cy_index] = cy_new
            adapted = CanonicalCamera(
                self.model,
                new_width or int(round(abs(float(linear[0, 1])) * self.height + abs(float(linear[0, 0])) * self.width)),
                new_height or int(round(abs(float(linear[1, 0])) * self.width + abs(float(linear[1, 1])) * self.height)),
                new_params,
                np.block([
                    [q, np.zeros((2, 1))],
                    [np.zeros((1, 2)), np.ones((1, 1))],
                ])
                @ self.R,
                self.t.copy(),
                self.near_m,
                self.far_m,
                self.camera_id,
                self.group,
            )
            return adapted
        else:
            raise ValueError(
                "adaptation limitée aux échelles et aux transpositions "
                "d'axes : appliquez d'abord EXIF transpose"
            )

        return CanonicalCamera(
            self.model,
            new_width or int(round(sx * self.width)),
            new_height or int(round(sy * self.height)),
            new_params,
            self.R.copy(),
            self.t.copy(),
            self.near_m,
            self.far_m,
            self.camera_id,
            self.group,
        )

    # ------------------------------------------------------------------
    # Construction / sérialisation
    # ------------------------------------------------------------------
    @classmethod
    def from_colmap(cls, camera, image=None, **kwargs) -> "CanonicalCamera":  # noqa: ANN001
        """Depuis un objet pycolmap (et son image pour la pose)."""
        name = str(getattr(camera, "model_name", camera.model))
        rotation = None
        translation = None
        if image is not None:
            cam_from_world = image.cam_from_world
            if callable(cam_from_world):
                cam_from_world = cam_from_world()
            rotation = np.asarray(cam_from_world.rotation.matrix(), dtype=float)
            translation = np.asarray(cam_from_world.translation, dtype=float)
        return cls(
            name,
            camera.width,
            camera.height,
            np.asarray(camera.params, dtype=float),
            rotation,
            translation,
            **kwargs,
        )

    def as_dict(self) -> dict:
        return {
            "contract": "canonical_camera/1",
            "model": self.model,
            "width": self.width,
            "height": self.height,
            "params": [round(float(v), 10) for v in self.params],
            "rotation": [[round(float(v), 12) for v in row] for row in self.R],
            "translation": [round(float(v), 10) for v in self.t],
            "near_m": self.near_m,
            "far_m": self.far_m,
            "camera_id": self.camera_id,
            "group": self.group,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CanonicalCamera":
        return cls(
            payload["model"],
            payload["width"],
            payload["height"],
            np.asarray(payload["params"], dtype=float),
            np.asarray(payload.get("rotation"), dtype=float),
            np.asarray(payload.get("translation"), dtype=float),
            float(payload.get("near_m", DEFAULT_NEAR_M)),
            float(payload.get("far_m", DEFAULT_FAR_M)),
            payload.get("camera_id"),
            payload.get("group"),
        )


# ----------------------------------------------------------------------
# Identité d'actif : jamais le basename (P16)
# ----------------------------------------------------------------------
def asset_identity(path: Path, metadata: dict | None = None) -> str:
    """Identité d'un actif : empreinte du contenu, jamais son nom de fichier.

    Deux fichiers différents baptisés ``IMG_001.jpg`` produisent deux
    identités distinctes ; le même contenu copié sous deux noms produit la
    même identité — c'est le même actif.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    fingerprint = digest.hexdigest()[:16]
    if metadata:
        relevant = {
            key: metadata[key]
            for key in sorted(metadata)
            if key in ("source", "captured_at", "lat", "lon", "altitude")
        }
        if relevant:
            fingerprint += "_" + hashlib.sha256(
                repr(relevant).encode("utf-8")
            ).hexdigest()[:8]
    stem = "".join(ch if ch.isalnum() else "_" for ch in Path(path).stem)[:40]
    return f"{fingerprint}_{stem}"


# ----------------------------------------------------------------------
# Groupes de vraies caméras (P17)
# ----------------------------------------------------------------------
def camera_group_key(metadata: dict) -> tuple:
    """Clé de regroupement par vraie caméra, pas par commodité.

    Capteur, résolution, focale, recadrage déclaré, source et nature
    panorama participent. Mélanger iPhone, drone et Street View donne trois
    groupes — jamais des intrinsèques partagés par défaut.
    """
    resolution = (int(metadata.get("width") or 0), int(metadata.get("height") or 0))
    focal_mm = round(float(metadata.get("focal_length_mm") or 0.0), 2)
    sensor = str(metadata.get("sensor") or metadata.get("make_model") or "")
    source = str(metadata.get("source") or "")
    crop = metadata.get("crop")
    crop_key = (
        tuple(round(float(v), 3) for v in crop)
        if isinstance(crop, (list, tuple)) and len(crop) == 4
        else None
    )
    panorama = bool(metadata.get("panorama"))
    return (sensor, resolution, focal_mm, source, crop_key, panorama)


def group_by_camera(items: list[tuple[dict, object]]) -> dict[tuple, list]:
    """Groupe (metadata, charge utile) par clé de vraie caméra."""
    grouped: dict[tuple, list] = {}
    for metadata, payload in items:
        grouped.setdefault(camera_group_key(metadata), []).append(payload)
    return grouped


# ----------------------------------------------------------------------
# Lignée pixel : original → EXIF → crop → resize → canonique (P15/P19)
# ----------------------------------------------------------------------
#: Transformations pixel associées à chaque valeur d'orientation EXIF.
#: Convention continue (centres de pixels) ; w/h sont celles de l'image
#: stockée. Vérifiées empiriquement contre ``ImageOps.exif_transpose``.


def _exif_affine(orientation: int, width: int, height: int) -> np.ndarray:
    """Matrice 2×3 : pixel stocké → pixel canonique (redressé)."""
    w, h = float(width), float(height)
    matrices = {
        1: np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        2: np.array([[-1.0, 0.0, w], [0.0, 1.0, 0.0]]),          # miroir horizontal
        3: np.array([[-1.0, 0.0, w], [0.0, -1.0, h]]),           # 180°
        4: np.array([[1.0, 0.0, 0.0], [0.0, -1.0, h]]),          # miroir vertical
        5: np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),         # transpose
        6: np.array([[0.0, -1.0, h], [1.0, 0.0, 0.0]]),          # 90° horaire
        7: np.array([[0.0, -1.0, h], [-1.0, 0.0, w]]),           # anti-transpose
        8: np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, w]]),          # 90° anti-horaire
    }
    found = matrices.get(int(orientation))
    if found is None:
        return matrices[1]
    return found


@dataclass
class ImageLineage:
    """Chaîne complète original → canonique, transform affines comprises."""

    asset_id: str
    steps: list[dict] = field(default_factory=list)
    #: Composition des étapes : pixel original → pixel canonique.
    transform: np.ndarray | None = None
    original_size: tuple[int, int] = (0, 0)
    canonical_size: tuple[int, int] = (0, 0)

    def composed(self) -> np.ndarray:
        if self.transform is not None:
            return self.transform
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        return matrix

    def as_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "steps": list(self.steps),
            "transform_original_to_canonical": [
                [float(v) for v in row] for row in self.composed()
            ],
            "original_size": list(self.original_size),
            "canonical_size": list(self.canonical_size),
        }

    def apply_intrinsics(self, camera: CanonicalCamera) -> CanonicalCamera:
        """Propage la lignée aux intrinsèques (P19)."""
        return camera.adapt_to(self.composed(), *self.canonical_size)


@dataclass
class CanonicalImage:
    """Une image canonique : redressée EXIF, identifiée par contenu."""

    asset_id: str
    width: int
    height: int
    lineage: ImageLineage
    image: object | None = None  # PIL.Image en mémoire

    def as_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "width": self.width,
            "height": self.height,
            "lineage": self.lineage.as_dict(),
        }


def canonize_image(path: Path, metadata: dict | None = None) -> CanonicalImage:
    """Redresse une image selon son EXIF et fige sa lignée pixel.

    L'image canonique porte une orientation EXIF remise à 1 : toute la suite
    du pipeline — SfM, segmentation, projection — ne voit plus qu'une seule
    représentation par photo physique, quelle que soit la façon dont l'appareil
    l'a stockée.
    """
    from PIL import Image, ImageOps

    path = Path(path)
    asset_id = asset_identity(path, metadata)
    with Image.open(path) as source:
        source.load()
        original_width, original_height = source.size
        try:
            exif = source.getexif()
            orientation = int(exif.get(274, 1) or 1)
        except (AttributeError, ValueError):
            orientation = 1

        upright = ImageOps.exif_transpose(source)
        canonical = upright.copy()

    # Orientation canonique : EXIF remis à 1, quoi qu'il en coûtait avant.
    try:
        exif_payload = canonical.getexif()
        exif_payload[274] = 1
        canonical.info["exif"] = exif_payload.tobytes()
    except (AttributeError, ValueError):  # pragma: no cover - PNG sans EXIF
        pass

    transform = _exif_affine(orientation, original_width, original_height)
    lineage = ImageLineage(
        asset_id=asset_id,
        steps=[
            {
                "kind": "exif_transpose",
                "orientation_before": orientation,
                "orientation_after": 1,
            }
        ],
        transform=transform,
        original_size=(original_width, original_height),
        canonical_size=(canonical.size[0], canonical.size[1]),
    )
    return CanonicalImage(
        asset_id=asset_id,
        width=canonical.size[0],
        height=canonical.size[1],
        lineage=lineage,
        image=canonical,
    )


def transform_mask(mask: np.ndarray, lineage: ImageLineage) -> np.ndarray:
    """Transporte un masque du repère original vers le repère canonique (P19)."""
    from PIL import Image

    pil = Image.fromarray(np.asarray(mask))
    matrix = lineage.composed()
    # PIL applique la matrice sortie→entrée : il faut l'inverse.
    inverse = np.linalg.inv(
        np.vstack([matrix, [0.0, 0.0, 1.0]])
    )
    transformed = pil.transform(
        tuple(lineage.canonical_size),
        Image.AFFINE,
        (inverse[0, 0], inverse[0, 1], inverse[0, 2],
         inverse[1, 0], inverse[1, 1], inverse[1, 2]),
        resample=Image.NEAREST,
    )
    return np.asarray(transformed)
