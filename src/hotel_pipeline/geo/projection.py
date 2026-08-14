"""Service de projection, partagé par la géométrie et la visibilité.

La même opération était écrite deux fois. `capture_geometry` vérifiait
l'emprise avant de projeter ; `visibility_run` appelait `Transformer.from_crs`
en littéral et ne vérifiait rien. Une seule des deux était protégée, et c'était
la moins critique : le moteur de visibilité calcule distances, azimuts et
occlusions sur les coordonnées projetées.

Tout passe désormais par ce service, qui refuse avant de calculer plutôt que de
rendre des nombres finis et faux. Trois contrôles, dans cet ordre :

```text
emprise      toutes les géométries, pas seulement le point central
finitude     une projection hors domaine peut rendre inf ou nan
aller-retour reprojeter et retrouver le point de départ
```

Le troisième attrape ce que les deux premiers laissent passer : une emprise
déclarée trop large, ou un axe inversé qui reste dans les bornes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..logging import get_logger
from ..schemas.spatial_reference import SpatialReferenceContext

log = get_logger("projection")

#: Écart maximal toléré sur un aller-retour, en degrés. Un ordre de grandeur
#: sous le centimètre à ces latitudes ; au-delà, la transformation est suspecte.
ROUNDTRIP_TOLERANCE_DEG = 1e-7


class ProjectionRefused(RuntimeError):
    """Rien n'a été projeté. Le message dit pourquoi, et où."""


@dataclass
class ProjectionService:
    """Projette depuis le référentiel source vers celui du site.

    Construit à partir du contexte spatial, jamais d'une constante : c'est ce
    qui rend le pipeline portable, et ce qui empêche un second site d'hériter
    du fuseau du premier.
    """

    reference: SpatialReferenceContext

    def __post_init__(self) -> None:
        if not self.reference.is_resolved:
            raise ProjectionRefused(
                f"contexte spatial non résolu (territoire "
                f"{self.reference.territory_state.value}) : aucun référentiel "
                "de travail n'a été choisi, et projeter reviendrait à en "
                "supposer un"
            )
        self._forward = None
        self._inverse = None

    @property
    def working_crs(self) -> str:
        return self.reference.working_crs  # type: ignore[return-value]

    def _transformers(self):  # noqa: ANN202
        if self._forward is None:
            from pyproj import Transformer

            # `always_xy=True` explicite : l'inverse échange latitude et
            # longitude sans rien signaler, et la forme part à des milliers de
            # kilomètres — sans erreur, ce qui est le pire mode de défaillance.
            self._forward = Transformer.from_crs(
                self.reference.source_crs, self.working_crs, always_xy=True
            )
            self._inverse = Transformer.from_crs(
                self.working_crs, self.reference.source_crs, always_xy=True
            )
        return self._forward, self._inverse

    # -- contrôles ---------------------------------------------------------

    def check_within_area(self, points: list[tuple[float, float]], label: str) -> None:
        """Toutes les positions sont-elles dans l'emprise du référentiel ?

        `points` en (lat, lon). Contrôler le seul point central laisserait
        passer un obstacle ou un corridor sortant du fuseau — et c'est
        justement là que la déformation devient grande.
        """
        outside = [
            (lat, lon) for lat, lon in points if not self.reference.contains(lat, lon)
        ]
        if outside:
            west, south, east, north = self.reference.working_area_of_use  # type: ignore[misc]
            first = outside[0]
            raise ProjectionRefused(
                f"{len(outside)} position(s) de « {label} » hors de l'emprise de "
                f"{self.working_crs} (lon {west}..{east}, lat {south}..{north}) — "
                f"première : lat {first[0]:.5f}, lon {first[1]:.5f}. "
                "Le calcul aurait été fini et faux."
            )

    def check_roundtrip(self, lat: float, lon: float) -> float:
        """Projette, reprojette, et rend l'écart constaté.

        Une emprise déclarée trop large, ou des axes inversés restant dans les
        bornes, ne se voient qu'ici.
        """
        forward, inverse = self._transformers()
        x, y = forward.transform(lon, lat)
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ProjectionRefused(
                f"projection non finie en lat {lat:.5f}, lon {lon:.5f} vers "
                f"{self.working_crs} : la position est hors du domaine de "
                "définition de la transformation"
            )
        back_lon, back_lat = inverse.transform(x, y)
        deviation = max(abs(back_lat - lat), abs(back_lon - lon))
        if deviation > ROUNDTRIP_TOLERANCE_DEG:
            raise ProjectionRefused(
                f"aller-retour de projection instable en lat {lat:.5f}, "
                f"lon {lon:.5f} : écart de {deviation:.3e}° via "
                f"{self.working_crs}, au-delà de {ROUNDTRIP_TOLERANCE_DEG:.0e}°"
            )
        return deviation

    def verify(self, points: list[tuple[float, float]], label: str) -> dict:
        """Les trois contrôles, dans l'ordre, avant tout calcul."""
        self.check_within_area(points, label)
        deviations = [self.check_roundtrip(lat, lon) for lat, lon in points]
        return {
            "label": label,
            "positions": len(points),
            "working_crs": self.working_crs,
            "max_roundtrip_deviation_deg": max(deviations) if deviations else 0.0,
        }

    # -- projection --------------------------------------------------------

    def point(self, lat: float, lon: float) -> tuple[float, float]:
        """Une position, contrôlée puis projetée."""
        self.check_within_area([(lat, lon)], "position")
        self.check_roundtrip(lat, lon)
        forward, _ = self._transformers()
        return forward.transform(lon, lat)

    def geometry(self, shape, label: str = "géométrie"):  # noqa: ANN001, ANN201
        """Une forme shapely en WGS84, contrôlée sur son emprise puis projetée."""
        from shapely.ops import transform

        minx, miny, maxx, maxy = shape.bounds
        self.check_within_area([(miny, minx), (maxy, maxx)], label)

        forward, _ = self._transformers()
        return transform(lambda xs, ys, zs=None: forward.transform(xs, ys), shape)

    def as_provenance(self) -> dict:
        """Ce qu'un rapport doit dire du référentiel qu'il a utilisé."""
        return {
            "working_crs": self.working_crs,
            "source_crs": self.reference.source_crs,
            "unit": self.reference.working_unit,
            "axes": self.reference.working_axes,
            "area_of_use": self.reference.working_area_of_use,
            "selection_method": self.reference.selection_method,
            "always_xy": True,
        }
