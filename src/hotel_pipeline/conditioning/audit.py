"""Audit de fidélité : ce que la scène établit, ce qu'elle suppose, ce qu'elle ignore.

Un modèle 3D se juge mal à l'œil. Un volume peut paraître convaincant tout en
reposant sur des hauteurs inventées, et un autre sembler grossier alors que
chacune de ses faces est mesurée. Le module chiffre cette distinction, poste
par poste, et projette ce que chaque source de données fermerait comme écart.

Trois questions, dans cet ordre :

1. **Qu'est-ce qui est mesuré ?** — part de la géométrie adossée à un relevé,
   pondérée par la surface visible plutôt que par le nombre d'objets : un
   voisin supposé de deux cents mètres carrés pèse davantage qu'un arbuste.
2. **Qu'est-ce qui est supposé ?** — et de quelle hypothèse précisément.
3. **Que gagnerait-on ?** — chaque source non exploitée est chiffrée en points
   de fidélité, pour que l'effort suivant se décide sur un écart mesuré.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from ..logging import get_logger

log = get_logger("conditioning-audit")

#: Crédit de fidélité par nature de source, de 0 à 1. Ce n'est pas une
#: probabilité : c'est le rang de confiance qu'un relevé mérite face à une
#: hypothèse, et il sert à comparer des postes entre eux.
SOURCE_FIDELITY: dict[str, float] = {
    "lidar_ndsm": 1.00,      # relevé aérien qualifié, surface triangulée
    "lidar_cloud": 0.90,     # nuage brut, hauteur par percentile
    "lidar_derived": 0.75,   # déduit du nuage — végétation, sol, terrain
    "image_inferred": 0.50,  # lu dans les photographies au sol
    "cadastral": 0.60,       # emprise OSM confirmée par un opérateur
    "assumed": 0.15,         # convention déclarée, aucune mesure
}


@dataclass
class FidelityItem:
    """Un poste du modèle, sa source et ce qu'il pèse."""

    poste: str
    source: str
    #: Surface approximative que ce poste occupe dans la scène, en m².
    surface_m2: float
    detail: str
    gap: str | None = None

    @property
    def fidelity(self) -> float:
        return SOURCE_FIDELITY.get(self.source, 0.5)

    def as_dict(self) -> dict:
        return {
            "poste": self.poste,
            "source": self.source,
            "fidelity": self.fidelity,
            "surface_m2": round(self.surface_m2, 1),
            "detail": self.detail,
            "gap": self.gap,
        }


@dataclass
class Projection:
    """Ce qu'une source non exploitée fermerait comme écart."""

    levier: str
    cout: str
    postes: list[str] = field(default_factory=list)
    fidelite_visee: float = 0.0
    gain_points: float = 0.0
    #: Distribution du gain, quand la projection a été simulée.
    distribution: dict | None = None

    def as_dict(self) -> dict:
        payload = {
            "levier": self.levier,
            "cout": self.cout,
            "postes": self.postes,
            "fidelite_visee": round(self.fidelite_visee, 2),
            "gain_points": round(self.gain_points, 1),
        }
        if self.distribution:
            payload["distribution"] = self.distribution
        return payload


@dataclass
class FidelityAudit:
    """Bilan de fidélité d'une scène, et ce qui l'améliorerait."""

    hotel_id: str
    items: list[FidelityItem] = field(default_factory=list)
    projections: list[Projection] = field(default_factory=list)

    @property
    def total_surface(self) -> float:
        return sum(i.surface_m2 for i in self.items)

    @property
    def score(self) -> float:
        """Fidélité globale, pondérée par la surface de chaque poste."""
        total = self.total_surface
        if total <= 0:
            return 0.0
        return sum(i.fidelity * i.surface_m2 for i in self.items) / total

    def by_source(self) -> dict[str, float]:
        """Part de surface portée par chaque nature de source."""
        total = self.total_surface
        if total <= 0:
            return {}
        shares: dict[str, float] = {}
        for item in self.items:
            shares[item.source] = shares.get(item.source, 0.0) + item.surface_m2
        return {k: round(v / total, 4) for k, v in sorted(shares.items())}

    def gaps(self) -> list[FidelityItem]:
        """Postes dont l'écart est déclaré, du plus lourd au plus léger."""
        found = [i for i in self.items if i.gap]
        found.sort(key=lambda i: i.surface_m2 * (1.0 - i.fidelity), reverse=True)
        return found

    def as_dict(self) -> dict:
        return {
            "hotel_id": self.hotel_id,
            "audited_at": datetime.now(timezone.utc).isoformat(),
            "score": round(self.score, 4),
            "surface_m2": round(self.total_surface, 1),
            "by_source": self.by_source(),
            "items": [i.as_dict() for i in self.items],
            "gaps": [i.as_dict() for i in self.gaps()],
            "projections": [p.as_dict() for p in self.projections],
            "caveats": [
                "le score pondère par la surface visible, non par le nombre "
                "d'objets : un voisin supposé pèse plus qu'un arbuste mesuré",
                "les crédits de fidélité classent des sources entre elles ; "
                "ce ne sont pas des probabilités d'exactitude",
                "un gain projeté suppose que la source visée couvre bien le "
                "poste — la découverte le dit, le téléchargement le prouve",
            ],
        }


def _prism_surface(prism) -> float:  # noqa: ANN001
    """Surface visible d'un volume : murs et toiture."""
    from shapely.geometry import Polygon

    polygon = Polygon(prism.footprint)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return float(polygon.area + polygon.length * prism.height_m)


def audit(
    scene,  # noqa: ANN001
    environment=None,  # noqa: ANN001
    support=None,  # noqa: ANN001
    yields: dict | None = None,
    coverage_bias: float = 1.0,
) -> FidelityAudit:
    """Chiffre la fidélité d'une scène et projette ce qui l'améliorerait."""
    result = FidelityAudit(hotel_id=scene.hotel_id)

    for prism in scene.prisms:
        surface = _prism_surface(prism)
        role = "cible" if prism.is_target else "voisin"

        if prism.roof_measured:
            source, detail = (
                "lidar_ndsm",
                f"toit triangulé, {len(prism.roof_faces)} faces, "
                f"hauteur {prism.height_m:.2f} m",
            )
            gap = "façade sans relief : ni ouverture, ni renfoncement, ni colonne"
        elif not prism.height_assumed:
            source, detail = (
                "lidar_cloud",
                f"hauteur {prism.height_m:.2f} m au percentile du nuage",
            )
            gap = "toiture fermée par un cône : sa forme n'est pas relevée"
        else:
            source = "assumed"
            detail = f"hauteur conventionnelle de {prism.height_m:.1f} m"
            gap = prism.height_source

        result.items.append(
            FidelityItem(
                poste=f"{role} {prism.feature_id}",
                source=source,
                surface_m2=surface,
                detail=detail,
                gap=gap,
            )
        )

    if environment is not None:
        patches = getattr(environment, "patches", []) or []
        if patches:
            surface = sum(
                float(np.pi * p.radius_m * (p.radius_m + p.height_m)) for p in patches
            )
            shaped = sum(1 for p in patches if getattr(p, "shape", None))
            result.items.append(
                FidelityItem(
                    poste="végétation",
                    source="lidar_derived",
                    surface_m2=surface,
                    detail=(
                        f"{len(patches)} couronnes segmentées, "
                        f"{shaped} profilées d'après les images"
                    ),
                    gap="volume opaque : ni feuillage, ni transparence, ni espèce",
                )
            )

        furniture = getattr(environment, "furniture", []) or []
        if furniture:
            result.items.append(
                FidelityItem(
                    poste="mobilier",
                    source="lidar_derived",
                    surface_m2=sum(
                        float(2 * np.pi * f.radius_m * f.height_m) for f in furniture
                    ),
                    detail=f"{len(furniture)} mâts isolés par leur continuité",
                    gap="cylindre nu : ni luminaire, ni panneau, ni bras",
                )
            )

        ground = getattr(environment, "ground_patches", []) or []
        if ground:
            result.items.append(
                FidelityItem(
                    poste="sol",
                    source="lidar_derived",
                    surface_m2=sum(p.area_m2() for p in ground),
                    detail=f"{len(ground)} plages détourées, nature par intensité",
                    gap=(
                        None
                        if getattr(environment, "terrain", None) is not None
                        else "posé à plat : le relief du terrain n'est pas porté"
                    ),
                )
            )

    result.projections = _project(result, scene, support, yields, coverage_bias)
    log.info(
        "audit : score %.3f sur %.0f m², %d écart(s) déclaré(s)",
        result.score,
        result.total_surface,
        len(result.gaps()),
    )
    return result


def _expected_coverage(scene, assumed: list[FidelityItem]) -> float:  # noqa: ANN001
    """Part des volumes conventionnels qu'une tuile voisine devrait contenir.

    Estimée sur la distance au centre du site : plus un volume est loin, plus
    il risque de tomber hors des tuiles adjacentes. C'est une borne, pas une
    certitude — d'où la simulation qui l'entoure d'un intervalle.
    """
    if not assumed:
        return 0.0

    centre = np.asarray(scene.centre)
    names = {i.poste.split()[-1] for i in assumed}
    distances = [
        float(np.linalg.norm(p.footprint.mean(axis=0) - centre))
        for p in scene.prisms
        if p.feature_id in names
    ]
    if not distances:
        return 0.7

    # Une tuile fait mille mètres de côté : au-delà de cinq cents mètres du
    # centre, un volume sort des quatre tuiles adjacentes.
    inside = sum(1 for d in distances if d <= 500.0)
    return float(np.clip(inside / len(distances), 0.1, 0.95))


def _project(  # noqa: ANN001
    audit_result: FidelityAudit,
    scene,
    support,
    yields: dict | None = None,
    coverage_bias: float = 1.0,
) -> list[Projection]:
    """Chiffre ce que chaque source non exploitée fermerait."""
    projections: list[Projection] = []
    total = audit_result.total_surface or 1.0

    # 1. Les volumes encore conventionnels : une tuile qui les couvre les
    #    ferait passer d'une hypothèse à une mesure.
    assumed = [i for i in audit_result.items if i.source == "assumed"]
    if assumed:
        surface = sum(i.surface_m2 for i in assumed)
        # La couverture n'est pas certaine : la découverte raisonne sur des
        # boîtes englobantes, et un volume à cheval sur deux tuiles peut
        # échapper à celle qu'on télécharge. Elle est donc simulée.
        from .projection import simulate

        gain = simulate(
            "tuiles LiDAR couvrant les volumes hors emprise",
            surface_concernee=surface,
            surface_totale=total,
            fidelite_actuelle=SOURCE_FIDELITY["assumed"],
            fidelite_visee=SOURCE_FIDELITY["lidar_cloud"],
            coverage=min(_expected_coverage(scene, assumed) * coverage_bias, 0.99),
            source="lidar_cloud",
            yields=yields,
        )
        projections.append(
            Projection(
                levier=gain.levier,
                cout="téléchargement, quelques centaines de mégaoctets",
                postes=[i.poste for i in assumed],
                fidelite_visee=SOURCE_FIDELITY["lidar_cloud"],
                gain_points=gain.median,
                distribution=gain.as_dict(),
            )
        )

    # 2. Les façades : aucun relevé aérien ne les décrit, seule une
    #    reconstruction depuis les images au sol y accède.
    facades = [
        i for i in audit_result.items
        if i.gap and "façade" in i.gap
    ]
    if facades:
        surface = sum(i.surface_m2 for i in facades) * 0.6
        gain = (SOURCE_FIDELITY["image_inferred"] - 0.0) * (surface / total) * 0.5
        projections.append(
            Projection(
                levier="reconstruction feed-forward depuis les vues au sol",
                cout="GPU, une heure environ",
                postes=[i.poste for i in facades],
                fidelite_visee=SOURCE_FIDELITY["image_inferred"],
                gain_points=gain * 100,
            )
        )

    # 3. L'appui photographique : il ne change pas la géométrie, il conditionne
    #    ce que le générateur pourra habiller sans inventer.
    if support is not None and len(support):
        gap_deg = support.widest_gap()
        if gap_deg > 90:
            projections.append(
                Projection(
                    levier="captation des secteurs sans référence",
                    cout="visite sur place, ou source d'imagerie complémentaire",
                    postes=[f"arc non couvert de {gap_deg:.0f}°"],
                    fidelite_visee=SOURCE_FIDELITY["image_inferred"],
                    gain_points=gap_deg / 360.0 * 100 * 0.4,
                )
            )

    projections.sort(key=lambda p: p.gain_points, reverse=True)
    return projections
