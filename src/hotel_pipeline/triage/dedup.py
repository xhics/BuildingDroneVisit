"""Déduplication perceptuelle (plan directeur §11, G1).

`imagehash` suffit et s'installe partout : le pHash regroupe les quasi-doublons
par distance de Hamming. `imagededup` (CNN) reste disponible dans la couche
vision pour les cas que le pHash rate — recadrages agressifs, changements
d'exposition marqués — mais il est inutile de l'imposer par défaut.
"""

from __future__ import annotations

from pathlib import Path

from ..logging import get_logger

log = get_logger("dedup")

#: Deux images dont les pHash diffèrent d'au plus ce nombre de bits sont
#: considérées comme le même cliché.
HAMMING_THRESHOLD = 6


def phash(image_path: Path) -> str:
    import imagehash
    from PIL import Image

    with Image.open(image_path) as image:
        return str(imagehash.phash(image.convert("RGB")))


def crop_resistant_hash(image_path: Path) -> str:
    """Signature multi-segments résistante aux recadrages et filigranes."""
    import imagehash
    from PIL import Image

    with Image.open(image_path) as image:
        return str(imagehash.crop_resistant_hash(image.convert("RGB")))


def group_duplicates(hashes: dict[str, str], threshold: int = HAMMING_THRESHOLD) -> dict[str, str]:
    """Associe chaque identifiant d'asset à un identifiant de groupe.

    Regroupement glouton : le premier membre rencontré nomme le groupe. Suffit
    à la volumétrie visée et reste explicable, ce qu'un clustering ne serait pas.
    """
    import imagehash

    groups: dict[str, str] = {}
    representatives: list[tuple[str, object]] = []

    for asset_id, hash_text in sorted(hashes.items()):
        if not hash_text:
            continue
        value = imagehash.hex_to_hash(hash_text)

        match = next(
            (rep_id for rep_id, rep_hash in representatives if (value - rep_hash) <= threshold),
            None,
        )
        if match is None:
            representatives.append((asset_id, value))
            groups[asset_id] = asset_id
        else:
            groups[asset_id] = match

    duplicates = len(groups) - len(representatives)
    log.info("déduplication : %d groupe(s), %d doublon(s)", len(representatives), duplicates)
    return groups
