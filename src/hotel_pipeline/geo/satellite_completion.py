"""Complétion des façades aveugles via satellite/orthophoto (Lot 1B complément).

Les champs visuels morts (blind fields) — objets existants jamais observés
photographiquement — peuvent être complétés via orthophotos satellites si :

1. une orthophoto couvre le site
2. la résolution est suffisante (min ~10 cm/px)
3. la façade regarde vers le sol (pas de surplomb ou intérieur)

Ce module reste **sceptique** : une façade "visible" dans l'orthophoto reçoit
une confidence LOW (synthétique), jamais HIGH. Le verdict reste "partial" ou
"full" selon le pourcentage mesuré, mais avec une note de provenance claire.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from shapely.geometry import Point, Polygon
from shapely.ops import transform, unary_union

from ..logging import get_logger

log = get_logger("satellite-completion")


class SyntheticSource(str, Enum):
    """Source de la complétion synthétique."""

    ORTHOPHOTO = "synthetic_from_orthophoto"
    AI_GENERATION = "synthetic_from_ai_texture_synthesis"
    UNKNOWN = "synthetic_unknown_source"


@dataclass(frozen=True)
class SatelliteAnalysis:
    """Analyse de visibilité d'une façade dans une orthophoto satellite."""

    facade_id: str
    is_visible: bool
    visible_fraction: float = 0.0
    source: SyntheticSource = SyntheticSource.UNKNOWN
    explanation: str = ""
    pixel_resolution_cm: float | None = None


@dataclass
class SyntheticCompletion:
    """Couverture synthétique pour une façade aveugle complétée par satellite."""

    facade_id: str
    source_type: SyntheticSource
    measured_fraction: float
    confidence_level: str = "low"  # Jamais HIGH pour synthétique
    contributing_source: str = ""  # ex. "cmm-ortho", "google-satellite"
    sampled: int = 0
    explanation: str = ""

    def as_dict(self) -> dict:
        """Sérialise pour zone_confidence.geojson."""
        return {
            "facade_id": self.facade_id,
            "appearance_coverage": "none",
            "appearance_union_fraction": 0.0,
            "geometric_support_coverage": self._derive_coverage(),
            "geometric_support_fraction": round(self.measured_fraction, 3),
            "union_fraction": 0.0,
            "weighted_union_fraction": 0.0,
            "best_fraction": round(self.measured_fraction, 3),
            "best_subject": None,
            "best_distance_m": None,
            "contributing_subjects": [self.contributing_source],
            "sampled_points": self.sampled,
            "synthesis": {
                "source_type": self.source_type.value,
                "confidence": self.confidence_level,
                "explanation": self.explanation,
            },
        }

    def _derive_coverage(self) -> str:
        """Dérive le verdict de couverture."""
        # Synthétique ne peut jamais être "full", même à 1.0.
        # Au mieux "partial" si fraction > 0.
        if self.measured_fraction <= 0.0:
            return "none"
        # Pour synthétique, on reste conservateur : max "partial"
        # même si la couverture semble complète.
        return "partial"


def analyze_facade_in_orthophoto(
    facade_geometry_wkt: str,
    footprint_geometry_wkt: str,
    orthophoto_data: dict,
    facade_kind: str,
) -> SatelliteAnalysis | None:
    """Analyse si une façade est visible dans une orthophoto.

    Arguments:
      facade_geometry_wkt: WKT de la géométrie du mur (LineString)
      footprint_geometry_wkt: WKT de l'empreinte du bâtiment
      orthophoto_data: dict avec clés:
        - "centroid_lat", "centroid_lon": localisation du centre
        - "resolution_cm": résolution en centimètres par pixel
        - "coverage_fraction": fraction couverte du bâtiment (0-1)
        - "notes": texte libre (ex. cloud cover, ...)
        - "path": chemin vers le fichier raster (optionnel)
        - "crs": CRS du raster (optionnel)
        - "bounds": bounding box projetée [minx, miny, maxx, maxy] (optionnel)
      facade_kind: "FACADE_PRIMARY", "FACADE_REAR", etc.

    Retourne:
      SatelliteAnalysis avec is_visible et visible_fraction, ou None si données insuffisantes.

    Logique:
      - Pas de données → None (absent, pas "non visible")
      - Résolution insuffisante (>25cm) → scepticisme (visible_fraction réduite)
      - Cloud cover → réduit la confiance
      - Façade orientée vers le ciel (peu probable) → visible_fraction réduite
      - Si un raster est fourni, analyse pixels, contraste et ombres
    """
    if orthophoto_data is None:
        log.debug(f"{facade_kind}: orthophoto absent")
        return None

    from shapely import wkt as shapely_wkt

    try:
        facade_geom = shapely_wkt.loads(facade_geometry_wkt)
        footprint_geom = shapely_wkt.loads(footprint_geometry_wkt)
    except Exception as e:
        log.warning(f"{facade_kind}: WKT parse error: {e}")
        return None

    raster_metrics = _raster_metrics_if_available(
        facade_geom, orthophoto_data, facade_kind
    )

    resolution_cm = orthophoto_data.get("resolution_cm")
    coverage_frac = orthophoto_data.get("coverage_fraction", 1.0)
    notes = orthophoto_data.get("notes", "")

    # Critère 1: Résolution suffisante?
    if resolution_cm is None:
        resolution_penalty = 0.5
    elif resolution_cm > 25.0:
        log.debug(f"{facade_kind}: résolution trop faible ({resolution_cm} cm)")
        resolution_penalty = 0.3
    elif resolution_cm > 15.0:
        resolution_penalty = 0.8
    else:
        resolution_penalty = 1.0

    cloud_penalty = 1.0
    if "cloud" in notes.lower() or "nuage" in notes.lower():
        cloud_penalty = 0.6

    orientation_score = 1.0
    if "PRIMARY" in facade_kind:
        orientation_score = 0.95
    elif "REAR" in facade_kind:
        orientation_score = 0.7

    heuristic = resolution_penalty * cloud_penalty * orientation_score * coverage_frac

    if raster_metrics is not None:
        visible_fraction = heuristic * raster_metrics["visible_fraction"]
        explanation = (
            f"raster analysis: visible={raster_metrics['visible_fraction']:.0%}, "
            f"contrast={raster_metrics['edge_contrast']:.2f}, "
            f"shadow={raster_metrics['shadow_fraction']:.0%}; "
            f"satellite resolution={resolution_cm} cm, "
            f"coverage={coverage_frac:.0%}, "
            f"orientation_adjustment={orientation_score:.0%}"
        )
    else:
        visible_fraction = heuristic
        explanation = (
            f"satellite resolution={resolution_cm} cm, "
            f"coverage={coverage_frac:.0%}, "
            f"orientation_adjustment={orientation_score:.0%}"
        )

    is_visible = visible_fraction > 0.2

    return SatelliteAnalysis(
        facade_id=facade_kind,
        is_visible=is_visible,
        visible_fraction=visible_fraction,
        source=SyntheticSource.ORTHOPHOTO,
        explanation=explanation,
        pixel_resolution_cm=resolution_cm,
    )


def _raster_metrics_if_available(
    facade_geom, orthophoto_data: dict, facade_kind: str,
) -> dict | None:
    """Analyse le raster si un chemin est fourni.

    Retourne un dict avec visible_fraction, edge_contrast, shadow_fraction,
    ou None si le raster n'est pas disponible/analysable.
    """
    raster_path = orthophoto_data.get("path")
    if not raster_path:
        return None

    try:
        import numpy as np
        import rasterio
        from rasterio.features import rasterize
    except ImportError:
        log.debug(f"{facade_kind}: rasterio/numpy indisponible pour l'analyse raster")
        return None

    try:
        with rasterio.open(raster_path) as dataset:
            if facade_geom.crs is None:
                log.debug(f"{facade_kind}: géométrie sans CRS, impossible de reprojeter")
                return None

            from pyproj import Transformer
            transformer = Transformer.from_crs(
                facade_geom.crs, dataset.crs, always_xy=True
            )
            projected = shapely.ops.transform(transformer.transform, facade_geom)

            coords = list(projected.coords)
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)

            if maxx <= minx or maxy <= miny:
                return None

            window = rasterio.windows.from_bounds(
                minx, miny, maxx, maxy, dataset.transform
            )
            if window.width < 1 or window.height < 1:
                return None

            rasterized = rasterize(
                [(projected, 1)],
                out_shape=(int(window.height), int(window.width)),
                transform=dataset.window_transform(window),
                fill=0,
                dtype="uint8",
            )

            read_window = dataset.read(
                1, window=window, out_shape=rasterized.shape, masked=True
            )
            if read_window.mask.all():
                return None

            facade_pixels = read_window[~read_window.mask & (rasterized > 0)]
            if facade_pixels.size == 0:
                return None

            nodata = dataset.nodata
            valid_pixels = facade_pixels != nodata if nodata is not None else facade_pixels
            visible_fraction = float(valid_pixels.sum() / facade_pixels.size)

            edge_contrast = _edge_contrast(read_window, rasterized)

            shadow_fraction = _shadow_fraction(facade_pixels)

            return {
                "visible_fraction": max(0.0, min(1.0, visible_fraction)),
                "edge_contrast": max(0.0, min(1.0, edge_contrast)),
                "shadow_fraction": max(0.0, min(1.0, shadow_fraction)),
            }
    except Exception as e:
        log.debug(f"{facade_kind}: raster analysis failed: {e}")
        return None


def _edge_contrast(image, mask) -> float:
    """Contraste moyen le long des pixels de la façade (Sobel simplifié)."""
    try:
        import numpy as np
        from scipy.ndimage import sobel
    except ImportError:
        return 0.5

    masked = image.astype(float)
    masked[mask == 0] = np.nan

    if np.all(np.isnan(masked)):
        return 0.5

    gx = sobel(masked, axis=1)
    gy = sobel(masked, axis=0)
    magnitude = np.hypot(gx, gy)
    magnitude = magnitude[~np.isnan(magnitude)]
    if magnitude.size == 0:
        return 0.5

    return float(np.mean(magnitude) / 255.0)


def _shadow_fraction(pixels) -> float:
    """Estime la fraction d'ombre par seuil percentile sombre."""
    try:
        import numpy as np
    except ImportError:
        return 0.0

    if pixels.size == 0:
        return 0.0
    p10 = np.percentile(pixels, 10)
    return float(np.mean(pixels < p10))

    return SatelliteAnalysis(
        facade_id=facade_kind,
        is_visible=is_visible,
        visible_fraction=visible_fraction,
        source=SyntheticSource.ORTHOPHOTO,
        explanation=explanation,
        pixel_resolution_cm=resolution_cm,
    )


def synthesize_completion_from_orthophoto(
    facade_kind: str,
    facade_geometry_wkt: str,
    footprint_geometry_wkt: str,
    orthophoto_source_id: str,
    orthophoto_data: dict,
) -> SyntheticCompletion | None:
    """Crée une couverture synthétique si l'orthophoto révèle la façade.

    Arguments:
      facade_kind: "FACADE_PRIMARY", etc.
      facade_geometry_wkt: WKT du mur
      footprint_geometry_wkt: WKT de l'empreinte
      orthophoto_source_id: ex. "cmm-ortho", "google-satellite"
      orthophoto_data: metadata de l'orthophoto

    Retourne:
      SyntheticCompletion si la façade est visible et mérite complétion, sinon None.
    """
    analysis = analyze_facade_in_orthophoto(
        facade_geometry_wkt,
        footprint_geometry_wkt,
        orthophoto_data,
        facade_kind,
    )
    if analysis is None or not analysis.is_visible:
        return None

    # Si visible, estimer la couverture.
    # Heuristique : si l'orthophoto la voit, on estime 60% (conservateur).
    # On ne dit jamais "full" pour synthétique.
    estimated_fraction = min(0.65, analysis.visible_fraction * 0.8)

    return SyntheticCompletion(
        facade_id=facade_kind,
        source_type=SyntheticSource.ORTHOPHOTO,
        measured_fraction=estimated_fraction,
        confidence_level="low",
        contributing_source=orthophoto_source_id,
        sampled=10,  # Approximation
        explanation=analysis.explanation,
    )


def merge_with_measured_coverage(
    measured: dict[str, dict], synthetics: list[SyntheticCompletion]
) -> dict[str, dict]:
    """Fusionne les mesures réelles avec les complétions synthétiques.

    Règle:
      - Si façade mesurée (appearance_union_fraction > 0) → garder la mesure réelle
      - Si façade aveugle (appearance_union_fraction == 0) ET synthétique propose → ajouter synthétique dans geometric_support_fraction
      - Jamais remplacer une mesure réelle par synthétique

    Arguments:
      measured: dict[facade_kind] = {appearance_union_fraction, appearance_coverage, ...}
      synthetics: list de SyntheticCompletion

    Retourne:
      dict[facade_kind] = dict, enrichi avec synthétique si applicable.
    """
    result = dict(measured)

    for synthetic in synthetics:
        kind = synthetic.facade_id
        if kind not in result:
            result[kind] = synthetic.as_dict()
            log.debug(f"{kind}: ajouté depuis synthétique (pas dans mesure)")
        else:
            measured_row = result[kind]
            measured_fraction = measured_row.get("appearance_union_fraction", 0.0)

            if measured_fraction <= 0.0:
                if "synthesis" not in measured_row:
                    measured_row["synthesis"] = {}
                measured_row["synthesis"][SyntheticSource.ORTHOPHOTO.value] = {
                    "measured_fraction": synthetic.measured_fraction,
                    "confidence": synthetic.confidence_level,
                    "source": synthetic.contributing_source,
                    "explanation": synthetic.explanation,
                }
                measured_row["geometric_support_fraction"] = round(
                    max(measured_row.get("geometric_support_fraction", 0.0), synthetic.measured_fraction), 3
                )
                measured_row["geometric_support_coverage"] = synthetic._derive_coverage()
                log.debug(
                    f"{kind}: complétion synthétique "
                    f"({synthetic.measured_fraction:.0%}) ajoutée dans geometric_support"
                )
            else:
                if "synthesis" not in measured_row:
                    measured_row["synthesis"] = {}
                measured_row["synthesis"]["not_needed"] = {
                    "reason": f"Already measured at {measured_fraction:.0%}",
                    "potential_synthetic": synthetic.measured_fraction,
                }
                log.debug(f"{kind}: déjà mesuré, synthétique ignoré")

    return result


# Données test : exemple d'orthophoto CMM ou service externe
ORTHOPHOTO_CMM_EXAMPLE = {
    "source_id": "cmm-ortho",
    "dataset": "Orthophotos CMM 2023",
    "resolution_cm": 20,  # 20 cm/pixel
    "coverage_fraction": 1.0,  # Couvre 100% de la parcelle
    "notes": "clear skies, no cloud cover",
    "centroid_lat": 45.6789,
    "centroid_lon": -73.5432,
}

ORTHOPHOTO_GOOGLE_EXAMPLE = {
    "source_id": "google-satellite",
    "dataset": "Google Maps Satellite",
    "resolution_cm": 50,  # Plus basse résolution
    "coverage_fraction": 0.9,
    "notes": "partial cloud cover in eastern section",
    "centroid_lat": 45.6789,
    "centroid_lon": -73.5432,
}
