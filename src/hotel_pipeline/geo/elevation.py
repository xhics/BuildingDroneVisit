"""Verticales mesurées, jamais supposées (Lot 1B V2, étape 3 bis).

Le premier passage du moteur a montré que les 92 lignes de vue à risque le
sont pour six données manquantes à la fois. Trois d'entre elles sont pourtant
déjà acquises ou dérivables :

```text
terrain et sommet de la cible   → rasters qualifiés TERRAIN_MAIN / ROOFLINE_MAIN
terrain et sommet des obstacles → nuage LAZ déjà téléchargé
terrain sous la caméra          → même nuage
hauteur d'œil de la caméra      → nulle part
```

La dernière ne se déduit d'aucune source : ni Mapillary ni Street View ne
publient la hauteur de leur capteur. Elle reste donc inconnue, et c'est
précisément l'intérêt de l'enrichissement — ramener six inconnues à une seule,
et savoir laquelle chercher.

La cible est échantillonnée **par rayon**, au point réellement visé : une
hauteur médiane écraserait un bâtiment dont un corps est plus bas qu'un autre.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..logging import get_logger

log = get_logger("elevation")

#: Codes ASPRS employés par la tuile.
GROUND = 2
BUILDING = 6

#: Rayon de recherche du sol autour d'une position caméra. Assez large pour
#: trouver des points au bord d'une chaussée, assez étroit pour ne pas
#: emprunter le terrain d'un talus voisin.
CAMERA_GROUND_RADIUS_M = 8.0

#: En deçà, l'échantillon n'est pas représentatif et le terrain reste inconnu.
MIN_GROUND_POINTS = 8


@dataclass(frozen=True)
class Sample:
    """Une mesure verticale et sa provenance."""

    value_m: float
    provenance: str
    points: int


class RasterSampler:
    """Lecture ponctuelle des rasters qualifiés du bâtiment cible.

    Les deux artefacts sont cités par leur identifiant : la mesure porte donc
    la trace de la dérivation qui l'a produite, et deviendra périmée avec elle.
    """

    def __init__(self, dtm_path: Path, roof_path: Path, provenance: str) -> None:
        import rasterio

        self.provenance = provenance
        self._dtm = rasterio.open(dtm_path)
        self._roof = rasterio.open(roof_path)

    def close(self) -> None:
        self._dtm.close()
        self._roof.close()

    def at(self, x: float, y: float) -> tuple[Sample | None, Sample | None]:
        """Terrain et sommet de toiture au point visé, s'ils y sont définis."""
        import math

        ground = self._read(self._dtm, x, y)
        roof = self._read(self._roof, x, y)
        return (
            Sample(ground, f"{self.provenance}:dtm", 1) if ground is not None else None,
            Sample(roof, f"{self.provenance}:dsm_roof", 1) if roof is not None else None,
        )

    @staticmethod
    def _read(dataset, x: float, y: float) -> float | None:  # noqa: ANN001
        import math

        try:
            value = next(dataset.sample([(x, y)]))[0]
        except (StopIteration, IndexError, ValueError):
            return None
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        if dataset.nodata is not None and value == dataset.nodata:
            return None
        return float(value)


@dataclass
class CloudSampler:
    """Mesures verticales tirées du nuage LiDAR déjà acquis.

    Le nuage est chargé une fois : vingt millions de points relus pour chaque
    voisin coûteraient plus que toute la chaîne.
    """

    x: object
    y: object
    z: object
    codes: object
    provenance: str

    @classmethod
    def load(cls, laz_path: Path, provenance: str) -> "CloudSampler":
        import laspy
        import numpy as np

        with laspy.open(str(laz_path)) as reader:
            points = reader.read()
        log.info("nuage chargé : %d points depuis %s", len(points.x), laz_path.name)
        return cls(
            np.asarray(points.x), np.asarray(points.y), np.asarray(points.z),
            np.asarray(points.classification), provenance,
        )

    def within(self, shape) -> tuple[Sample | None, Sample | None]:  # noqa: ANN001
        """Terrain et sommet d'une emprise : médiane du sol, p95 du bâti.

        Le p95 plutôt que le maximum : une cheminée ou un mât d'antenne ne dit
        pas la hauteur du volume qui masque.
        """
        import numpy as np
        from shapely import contains_xy

        minx, miny, maxx, maxy = shape.bounds
        window = (self.x >= minx) & (self.x <= maxx) & (self.y >= miny) & (self.y <= maxy)
        if not window.any():
            return None, None

        wx, wy = self.x[window], self.y[window]
        inside = contains_xy(shape, wx, wy)
        if not inside.any():
            return None, None

        wz, codes = self.z[window][inside], self.codes[window][inside]
        ground_points = wz[codes == GROUND]
        roof_points = wz[codes == BUILDING]

        ground = (
            Sample(float(np.median(ground_points)), f"{self.provenance}:class2",
                   int(ground_points.size))
            if ground_points.size >= MIN_GROUND_POINTS
            else None
        )
        top = (
            Sample(float(np.percentile(roof_points, 95)), f"{self.provenance}:class6_p95",
                   int(roof_points.size))
            if roof_points.size >= MIN_GROUND_POINTS
            else None
        )
        return ground, top

    def ground_near(self, x: float, y: float, radius_m: float = CAMERA_GROUND_RADIUS_M):
        """Terrain sous une position caméra, s'il est couvert par la tuile.

        Hors tuile, la valeur reste inconnue : extrapoler le sol d'un
        kilomètre plus loin serait une invention.
        """
        import numpy as np

        window = (
            (self.x >= x - radius_m) & (self.x <= x + radius_m)
            & (self.y >= y - radius_m) & (self.y <= y + radius_m)
        )
        if not window.any():
            return None

        wz = self.z[window][self.codes[window] == GROUND]
        if wz.size < MIN_GROUND_POINTS:
            return None
        return Sample(float(np.median(wz)), f"{self.provenance}:class2_voisinage",
                      int(wz.size))
