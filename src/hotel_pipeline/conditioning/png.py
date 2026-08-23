"""Écriture PNG sans dépendance graphique.

Pillow n'est pas installé et n'a pas à l'être pour ce harnais : le PNG est un
format assez simple pour être écrit directement, et le rendre autonome garantit
que la comparaison A/B tourne partout.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def write_png(path: Path, image: np.ndarray) -> None:
    """Écrit un tableau uint8 en PNG. Accepte (H,W) gris ou (H,W,3) RGB."""
    arr = np.ascontiguousarray(image, dtype=np.uint8)
    if arr.ndim == 2:
        height, width = arr.shape
        colour_type = 0
        channels = 1
    elif arr.ndim == 3 and arr.shape[2] == 3:
        height, width, channels = arr.shape
        colour_type = 2
    else:
        raise ValueError(f"forme non supportée pour un PNG : {arr.shape}")

    stride = width * channels
    raw = bytearray()
    flat = arr.reshape(height, stride)
    for row in flat:
        raw.append(0)  # filtre "None", ligne par ligne
        raw.extend(row.tobytes())

    header = struct.pack(">IIBBBBB", width, height, 8, colour_type, 0, 0, 0)
    blob = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + _chunk(b"IEND", b"")
    )
    Path(path).write_bytes(blob)


def depth_to_png(depth: np.ndarray, near: float, far: float) -> np.ndarray:
    """Profondeur en gris, proche = clair. Le vide reste noir."""
    out = np.zeros(depth.shape, dtype=np.uint8)
    finite = np.isfinite(depth)
    if not finite.any():
        return out
    span = max(far - near, 1e-6)
    norm = np.clip((depth[finite] - near) / span, 0.0, 1.0)
    out[finite] = ((1.0 - norm) * 255).astype(np.uint8)
    return out


def normal_to_png(normal: np.ndarray) -> np.ndarray:
    """Normales encodées en RGB, convention habituelle des ControlNet."""
    return np.clip((normal * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)


def confidence_to_png(confidence: np.ndarray) -> np.ndarray:
    """Crédit géométrique : vert = attesté, rouge = supposé, noir = rien.

    Le noir n'est pas une valeur basse, c'est une absence : aucun volume ne s'y
    projette et le générateur y est libre.
    """
    h, w = confidence.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    seen = confidence > 0
    if seen.any():
        value = confidence[seen]
        rgb[..., 0][seen] = ((1.0 - value) * 255).astype(np.uint8)
        rgb[..., 1][seen] = (value * 255).astype(np.uint8)
    return rgb


#: Niveau de gris attribué à chaque nature dans la carte de silhouette. Les
#: valeurs sont espacées pour rester distinctes après compression, et le sol
#: reste sombre : il ne masque rien, il situe.
SILHOUETTE_LEVELS: dict[int, int] = {
    1: 90,    # obstacle bâti
    2: 255,   # bâtiment cible
    3: 150,   # végétation
    4: 200,   # mobilier urbain
    5: 45,    # sol végétal
    6: 25,    # sol minéral
    7: 35,    # sol posé sans nature établie
}


def silhouette_to_png(silhouette: np.ndarray) -> np.ndarray:
    """Chaque nature reçoit son niveau : cible en blanc, sol au plus sombre.

    Les natures ajoutées après coup — végétation, mobilier, sol — ressortaient
    en noir tant qu'elles n'étaient pas listées ici, et une carte pourtant
    correcte paraissait vide.
    """
    out = np.zeros(silhouette.shape, dtype=np.uint8)
    for value, level in SILHOUETTE_LEVELS.items():
        out[silhouette == value] = level
    return out
