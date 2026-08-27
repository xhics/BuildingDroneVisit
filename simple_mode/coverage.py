"""Où avons-nous de la donnée réelle, et où n'en avons-nous pas.

Principe directeur : **ne jamais soumettre au générateur une zone couverte
par une donnée réelle exploitable.** On préserve partout où l'on possède du
réel ; on ne génère que les trous.

Cela supprime la contradiction entre « ajouter de la vie » et « ne rien
inventer » : la vie s'ajoute exactement là où il n'y a rien à abîmer — le
ciel, le sol nu, les abords lointains — et le bâtiment reste intact parce
qu'il est, lui, couvert par la mesure.

La couverture se lit dans la **profondeur** du rendu, pas dans un jugement
esthétique :

- profondeur infinie → ciel → aucune donnée → génération libre ;
- profondeur faible → sujet proche, tuiles au meilleur détail → intouchable ;
- profondeur forte → arrière-plan, tuiles grossières → liberté croissante.

C'est aussi ce qui explique, rétrospectivement, pourquoi l'intérieur
fonctionnait quand l'aérien échouait : à l'intérieur on ne demandait au
moteur que de combler des trous entre photos réelles, tandis qu'à
l'extérieur on lui demandait de refaire ce qu'on possédait déjà.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageFilter

#: Distance en deçà de laquelle la donnée est jugée fiable : tuiles au
#: meilleur niveau de détail, façades lisibles. Mesuré sur cette scène, la
#: photogrammétrie tient jusqu'à ~60 m et se délite au-delà.
NEAR_M = 70.0

#: Distance au-delà de laquelle la donnée ne vaut plus grand-chose : tuiles
#: grossières, arrière-plan de ville sans détail exploitable.
FAR_M = 320.0

#: Échelle d'encodage de la passe de profondeur (voir `cesium_scene.html`).
#: Un pixel blanc vaut cette distance ; le ciel, sans géométrie, est blanc.
DEPTH_SCALE_M = 800.0


@dataclass
class CoverageMask:
    """Masque de régénération : blanc = à générer, noir = à préserver."""

    path: Path
    #: Part de l'image laissée au générateur, entre 0 et 1.
    free_ratio: float

    def describe_fr(self) -> str:
        return (
            f"{self.free_ratio * 100:.0f}% de l'image laissée au générateur, "
            f"{100 - self.free_ratio * 100:.0f}% préservés"
        )


def thresholds_for_subject(subject_distance_m: float) -> tuple[float, float]:
    """Seuils de préservation, dérivés de la distance au sujet.

    Des seuils fixes n'ont pas de sens : un plan large à 200 m et un plan
    serré à 60 m ne placent pas le bâtiment à la même profondeur. On ancre
    donc la zone préservée sur la distance réelle du sujet — tout ce qui est
    nettement derrière lui devient de l'arrière-plan, donc régénérable.
    """
    near = max(40.0, subject_distance_m * 1.35)
    far = max(near * 2.5, subject_distance_m * 3.5)
    return near, far


def _coverage_curve(depth_norm: float, near_m: float, far_m: float) -> float:
    """Liberté accordée au générateur, pour une profondeur normalisée.

    0 = préserver strictement, 1 = régénérer librement. La transition est
    progressive : un basculement net dessinerait une découpe visible entre
    la zone conservée et la zone régénérée.
    """
    near = near_m / DEPTH_SCALE_M
    far = far_m / DEPTH_SCALE_M
    if depth_norm <= near:
        return 0.0
    if depth_norm >= 1.0 - 1e-6:
        return 1.0  # ciel : aucune géométrie, donc aucune donnée
    if depth_norm >= far:
        return 1.0
    return (depth_norm - near) / max(1e-6, far - near)


def build_mask(
    depth_image: str | Path,
    out_path: str | Path,
    *,
    subject_distance_m: float | None = None,
    feather_px: int = 12,
    protect_floor: float = 0.0,
) -> CoverageMask:
    """Construit le masque de régénération à partir d'une passe de profondeur.

    ``feather_px`` adoucit la frontière : sans ce fondu, la limite entre zone
    préservée et zone régénérée se voit comme une découpe.

    ``protect_floor`` permet de garder un minimum de retouche partout (par
    exemple 0.15 pour laisser le moteur unifier la lumière) sans jamais
    l'autoriser à refaire la géométrie.
    """
    near_m, far_m = (
        thresholds_for_subject(subject_distance_m)
        if subject_distance_m is not None
        else (NEAR_M, FAR_M)
    )
    depth = Image.open(depth_image).convert("L")
    lut = [
        int(255 * max(protect_floor, _coverage_curve(value / 255.0, near_m, far_m)))
        for value in range(256)
    ]
    mask = depth.point(lut)
    if feather_px > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather_px))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(out_path)

    histogram = mask.histogram()
    total = sum(histogram)
    free = sum(i * count for i, count in enumerate(histogram)) / (255 * total)
    return CoverageMask(path=out_path, free_ratio=free)


def summarize_depth(depth_image: str | Path) -> dict:
    """Répartition de la couverture, pour vérifier un masque avant de payer."""
    depth = Image.open(depth_image).convert("L")
    histogram = depth.histogram()
    total = sum(histogram)

    near = NEAR_M / DEPTH_SCALE_M * 255
    far = FAR_M / DEPTH_SCALE_M * 255
    buckets = {"proche_preserve": 0, "intermediaire": 0, "lointain_libre": 0, "ciel": 0}
    for value, count in enumerate(histogram):
        if value >= 254:
            buckets["ciel"] += count
        elif value <= near:
            buckets["proche_preserve"] += count
        elif value >= far:
            buckets["lointain_libre"] += count
        else:
            buckets["intermediaire"] += count
    return {k: v / total for k, v in buckets.items()}


__all__ = [
    "DEPTH_SCALE_M",
    "FAR_M",
    "NEAR_M",
    "CoverageMask",
    "build_mask",
    "summarize_depth",
    "thresholds_for_subject",
]
