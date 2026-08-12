"""Qualité d'image (plan directeur §11, G3).

`cleanvision` couvre presque tout le G3 tel quel — flou, sous- et
surexposition, images sans information, quasi-doublons — sur un répertoire
entier et sans configuration. C'est un audit de jeu de données, pas un score
par image, ce qui correspond exactement au besoin.

Repli sans dépendance : variance du laplacien pour le flou et moyenne de
luminance pour l'exposition. Grossier mais suffisant pour écarter le pire, et
toujours disponible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..logging import get_logger

log = get_logger("quality")

#: Sous ce seuil de variance du laplacien, l'image est jugée floue.
BLUR_VARIANCE_THRESHOLD = 60.0
DARK_THRESHOLD = 40.0
BRIGHT_THRESHOLD = 215.0


@dataclass
class QualityIssues:
    """Problèmes détectés, par identifiant d'image."""

    blurry: set[str] = field(default_factory=set)
    dark: set[str] = field(default_factory=set)
    light: set[str] = field(default_factory=set)
    low_information: set[str] = field(default_factory=set)

    def flagged(self) -> set[str]:
        return self.blurry | self.dark | self.light | self.low_information


def audit_with_cleanvision(image_dir: Path) -> QualityIssues:
    """Audit complet du répertoire par cleanvision."""
    from cleanvision import Imagelab

    lab = Imagelab(data_path=str(image_dir))
    lab.find_issues()

    issues = QualityIssues()
    frame = lab.issues
    mapping = {
        "is_blurry_issue": issues.blurry,
        "is_dark_issue": issues.dark,
        "is_light_issue": issues.light,
        "is_low_information_issue": issues.low_information,
    }
    for column, target in mapping.items():
        if column in frame.columns:
            target.update(Path(p).name for p in frame.index[frame[column]])

    log.info("cleanvision : %d image(s) signalée(s)", len(issues.flagged()))
    return issues


def basic_scores(image_path: Path) -> dict[str, float]:
    """Netteté et luminance, sans dépendance à la couche vision."""
    import numpy as np
    from PIL import Image

    with Image.open(image_path) as image:
        grey = np.asarray(image.convert("L"), dtype=float)

    # Laplacien 3x3 par différences finies.
    laplacian = (
        -4 * grey[1:-1, 1:-1]
        + grey[:-2, 1:-1]
        + grey[2:, 1:-1]
        + grey[1:-1, :-2]
        + grey[1:-1, 2:]
    )
    return {"sharpness": float(laplacian.var()), "brightness": float(grey.mean())}


def audit_basic(image_paths: list[Path]) -> QualityIssues:
    issues = QualityIssues()
    for path in image_paths:
        scores = basic_scores(path)
        if scores["sharpness"] < BLUR_VARIANCE_THRESHOLD:
            issues.blurry.add(path.name)
        if scores["brightness"] < DARK_THRESHOLD:
            issues.dark.add(path.name)
        elif scores["brightness"] > BRIGHT_THRESHOLD:
            issues.light.add(path.name)
    return issues


def normalised_quality(scores: dict[str, float]) -> float:
    """Ramène la netteté à [0, 1] pour classer les images entre elles."""
    sharpness = scores.get("sharpness", 0.0)
    return max(0.0, min(1.0, sharpness / (sharpness + BLUR_VARIANCE_THRESHOLD)))
