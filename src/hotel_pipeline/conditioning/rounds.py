"""Rondes de comparaison : confronter le rendu aux photographies, et corriger.

La géométrie est mesurée, mais rien ne vérifiait qu'elle **ressemble** au
bâtiment vu depuis la rue. Un volume peut avoir la bonne hauteur et la bonne
emprise tout en produisant une silhouette que la photographie dément — trop
massive, mal proportionnée, ou vue d'un cadrage qui n'existe pas.

Le module compare des **profils verticaux** : la part de bâti par bande
horizontale, lue d'un côté dans la photographie par un modèle à vocabulaire
ouvert, de l'autre dans la silhouette rendue. Mesuré sur ce pilote, ce signal
discrimine nettement — 0,88 de corrélation au bon cadrage contre 0,50 à
soixante mètres de trop.

Chaque ronde ajuste un paramètre, mesure, et garde ce qui rapproche. La
convergence s'arrête d'elle-même quand plus rien ne gagne.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..logging import get_logger

log = get_logger("conditioning-rounds")

#: Hauteur d'œil d'une prise de vue au sol, en mètres. Street View et les
#: contributions photographiques sont captées depuis un véhicule ou un piéton.
EYE_HEIGHT_M = 2.5

#: Distances explorées lors de la recherche de cadrage, en mètres.
DISTANCE_CANDIDATES: tuple[float, ...] = (20.0, 30.0, 40.0, 55.0, 75.0, 100.0)

#: Champs de vision explorés, en degrés.
FOV_CANDIDATES: tuple[float, ...] = (45.0, 60.0, 75.0)

#: En deçà, la ressemblance de profil ne prouve rien : deux images vides
#: corrèlent parfaitement.
MIN_PROFILE_MASS = 0.02


@dataclass
class RoundResult:
    """Ce qu'une ronde a essayé, et ce qu'elle a obtenu."""

    asset_id: str
    bearing_deg: float
    distance_m: float
    fov_deg: float
    correlation: float
    coverage: float

    def as_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "bearing_deg": round(self.bearing_deg, 1),
            "distance_m": round(self.distance_m, 1),
            "fov_deg": round(self.fov_deg, 1),
            "correlation": round(self.correlation, 4),
            "coverage": round(self.coverage, 4),
        }


@dataclass
class ComparisonReport:
    """Bilan des rondes sur un site."""

    hotel_id: str
    rounds: list[RoundResult] = field(default_factory=list)
    best: list[RoundResult] = field(default_factory=list)

    @property
    def mean_correlation(self) -> float:
        if not self.best:
            return 0.0
        return float(np.mean([r.correlation for r in self.best]))

    def verdict(self) -> str:
        """Ce que la comparaison établit sur la ressemblance du volume."""
        score = self.mean_correlation
        if not self.best:
            return "non_comparable"
        if score >= 0.75:
            return "silhouette_conforme"
        if score >= 0.45:
            return "silhouette_approchante"
        return "silhouette_dementie"

    def as_dict(self) -> dict:
        return {
            "hotel_id": self.hotel_id,
            "compared_at": datetime.now(timezone.utc).isoformat(),
            "verdict": self.verdict(),
            "mean_correlation": round(self.mean_correlation, 4),
            "views_compared": len(self.best),
            "rounds_run": len(self.rounds),
            "best": [r.as_dict() for r in self.best],
            "caveats": [
                "la comparaison porte sur la répartition verticale du bâti, "
                "non sur son apparence : une façade de la bonne silhouette "
                "peut avoir la mauvaise couleur, les mauvaises ouvertures",
                "la pose de la prise de vue est **recherchée**, non connue : "
                "un bon accord peut venir d'un cadrage bien deviné autant que "
                "d'un volume juste",
                "un profil presque vide corrèle avec n'importe quoi : ces "
                "vues sont écartées plutôt que comptées comme conformes",
            ],
        }


def _profile_from_photo(silhouette_map, classes: tuple[str, ...]) -> np.ndarray:
    """Part des classes visées par bande horizontale d'une photographie."""
    indices = [
        silhouette_map.classes.index(name)
        for name in classes
        if name in silhouette_map.classes
    ]
    if not indices:
        return np.zeros(silhouette_map.labels.shape[0])
    return np.isin(silhouette_map.labels, indices).mean(axis=1)


def _profile_from_render(frame, values: tuple[int, ...]) -> np.ndarray:
    """Part des natures visées par bande horizontale d'un rendu."""
    mask = np.isin(frame.silhouette, values)
    return mask.mean(axis=1)


def _agreement(photo: np.ndarray, render: np.ndarray) -> float:
    """Ressemblance de deux profils, ramenés à la même longueur.

    La corrélation seule récompense deux profils vides : la masse de bâti est
    donc exigée des deux côtés avant de l'accorder.
    """
    if photo.size < 4 or render.size < 4:
        return 0.0
    if photo.sum() < MIN_PROFILE_MASS * photo.size:
        return 0.0
    if render.sum() < MIN_PROFILE_MASS * render.size:
        return 0.0

    resampled = np.interp(
        np.linspace(0.0, 1.0, photo.size),
        np.linspace(0.0, 1.0, render.size),
        render,
    )
    if np.std(photo) < 1e-9 or np.std(resampled) < 1e-9:
        return 0.0
    return float(np.clip(np.corrcoef(photo, resampled)[0, 1], -1.0, 1.0))


def _render_at(scene, environment, bearing_deg, distance_m, fov_deg, height, width):  # noqa: ANN001
    """Rendu depuis une pose au sol, à l'azimut d'une photographie."""
    from .render import Camera, render_frame

    cx, cy = scene.centre
    theta = math.radians(bearing_deg)
    target = scene.target
    look_z = (target.height_m * 0.5) if target else 5.0
    camera = Camera(
        position=np.array([
            cx + distance_m * math.sin(theta),
            cy + distance_m * math.cos(theta),
            EYE_HEIGHT_M,
        ]),
        target=np.array([cx, cy, look_z]),
        fov_deg=fov_deg,
        width=width,
        height=height,
    )
    return render_frame(scene, camera, environment)


def compare_view(
    scene,  # noqa: ANN001
    environment,  # noqa: ANN001
    asset_id: str,
    photo_profile: np.ndarray,
    bearing_deg: float,
    distances: tuple[float, ...] = DISTANCE_CANDIDATES,
    fovs: tuple[float, ...] = FOV_CANDIDATES,
) -> tuple[RoundResult | None, list[RoundResult]]:
    """Cherche la pose dont la silhouette rendue épouse le profil de la photo.

    La pose exacte d'une contribution photographique n'est pas connue : seul
    l'azimut est mesuré. Distance et champ sont donc **recherchés**, et le
    meilleur accord retenu — ce qui mesure la ressemblance du volume à cadrage
    optimal, non l'exactitude d'une pose devinée.
    """
    rows = max(int(photo_profile.size), 8)
    attempts: list[RoundResult] = []

    for distance in distances:
        for fov in fovs:
            frame = _render_at(
                scene, environment, bearing_deg, distance, fov, rows, rows * 2
            )
            render_profile = _profile_from_render(frame, (1, 2))
            attempts.append(
                RoundResult(
                    asset_id=asset_id,
                    bearing_deg=bearing_deg,
                    distance_m=distance,
                    fov_deg=fov,
                    correlation=_agreement(photo_profile, render_profile),
                    coverage=float(frame.target_coverage),
                )
            )

    usable = [a for a in attempts if a.correlation > 0.0]
    best = max(usable, key=lambda a: a.correlation) if usable else None
    return best, attempts


def run(
    scene,  # noqa: ANN001
    environment,  # noqa: ANN001
    views: list[tuple[str, Path, float]],
    embedder=None,  # noqa: ANN001
) -> ComparisonReport:
    """Confronte le volume rendu à chaque vue de référence.

    Une vue par ronde, chacune explorant les cadrages plausibles. Le rapport
    dit ensuite si la silhouette du volume tient devant les photographies, ou
    si elle les dément.
    """
    from .silhouette import read_image

    from ..identity.embedding import ImageEmbedder

    embedder = embedder or ImageEmbedder()
    report = ComparisonReport(hotel_id=scene.hotel_id)

    for asset_id, path, bearing in views:
        if bearing is None:
            continue
        silhouette_map = read_image(embedder, Path(path), asset_id, bearing)
        if silhouette_map is None:
            continue

        photo_profile = _profile_from_photo(silhouette_map, ("batiment",))
        best, attempts = compare_view(
            scene, environment, asset_id, photo_profile, float(bearing)
        )
        report.rounds.extend(attempts)
        if best is not None:
            report.best.append(best)

    log.info(
        "rondes : %d vue(s), accord moyen %.3f — %s",
        len(report.best),
        report.mean_correlation,
        report.verdict(),
    )
    return report
