"""Contexte de référence spatiale d'un site (portabilité, commit 2).

Le référentiel de calcul était une constante de module : `EPSG:2950`, le fuseau
MTM 8 du Québec. Rien ne l'empêchait de servir ailleurs — pyproj projette hors
emprise sans lever, et Lyon se plaçait en `x=5 637 219 m, y=8 760 910 m` sans
une erreur. Distances, azimuts et occlusions se calculaient là-dessus, et le
rapport avait l'air normal.

Le CRS résolu est un **fait spatial du site**, non un seuil : il n'a rien à
faire dans une politique, où il deviendrait réglable. Il vit ici, versionné,
et les calculs le citent au lieu de le supposer.

Trois états territoriaux, et l'ignorance en est un :

```text
resolved      des juridictions établies, avec leur preuve
unsupported   territoire connu, mais hors de ce que le catalogue couvre
unknown       rien n'est établi — et surtout pas « QC par défaut »
```
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TerritoryState(StrEnum):
    """Ce qu'on sait du territoire. `UNKNOWN` est une réponse, pas un vide."""

    RESOLVED = "resolved"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class HeightType(StrEnum):
    """Ce que mesure une altitude. Les mélanger est l'erreur qu'on prévient."""

    ELLIPSOIDAL = "ellipsoidal"
    ORTHOMETRIC = "orthometric"
    ABOVE_GROUND = "above_ground"
    UNKNOWN = "unknown"


class VerticalTransform(BaseModel):
    """Comment passer d'un référentiel vertical à un autre.

    Deux référentiels différents ne sont pas nécessairement incompatibles : ils
    le sont tant que rien ne dit comment passer de l'un à l'autre. Une
    transformation est donc admise, à condition d'être déclarée et vérifiable.

    Une transformation supposée serait pire que l'absence : elle donnerait à un
    écart de référentiel l'apparence d'une mesure.
    """

    model_config = ConfigDict(extra="forbid")

    source_crs: str
    target_crs: str
    source_height_type: HeightType
    target_height_type: HeightType
    unit: str = "m"

    #: L'opération appliquée, nommée. « conversion » n'en est pas une.
    operation: str

    #: Modèle de géoïde et sa version, quand l'opération en utilise un.
    geoid_model: str | None = None
    geoid_version: str | None = None

    #: Précision annoncée **par l'opération**, non celle qu'on espère.
    accuracy_m: float | None = Field(default=None, ge=0)

    provenance: str

    @model_validator(mode="after")
    def _an_orthometric_conversion_names_its_geoid(self) -> "VerticalTransform":
        crossing = {self.source_height_type, self.target_height_type} == {
            HeightType.ELLIPSOIDAL, HeightType.ORTHOMETRIC
        }
        if crossing and not self.geoid_model:
            raise ValueError(
                "passer d'une hauteur ellipsoïdale à une hauteur orthométrique "
                "exige un modèle de géoïde : sans lui, l'écart — plusieurs "
                "dizaines de mètres — serait appliqué au jugé"
            )
        if HeightType.UNKNOWN in (self.source_height_type, self.target_height_type):
            raise ValueError(
                "transformation entre un type de hauteur inconnu : une "
                "transformation ne se déclare pas sur ce qu'on ignore"
            )
        return self


class VerticalReference(BaseModel):
    """Le référentiel vertical du site, ou l'aveu qu'on ne le connaît pas."""

    model_config = ConfigDict(extra="forbid")

    crs: str | None = None
    height_type: HeightType = HeightType.UNKNOWN
    unit: str = "m"
    provenance: str | None = None

    #: Transformations disponibles depuis d'autres référentiels verticaux.
    transforms: list[VerticalTransform] = Field(default_factory=list)

    @property
    def is_known(self) -> bool:
        return self.crs is not None and self.height_type is not HeightType.UNKNOWN

    def transform_from(self, other: str | None) -> VerticalTransform | None:
        """Comment ramener une mesure de `other` vers ce référentiel.

        Rend `None` quand rien ne le dit — et c'est alors au moteur de rendre
        `unknown`, jamais de soustraire deux nombres au motif qu'ils sont des
        mètres.
        """
        if other is None:
            return None
        if other == self.crs:
            return None  # identité : rien à déclarer
        return next(
            (
                item for item in self.transforms
                if item.source_crs == other and item.target_crs == self.crs
            ),
            None,
        )

    def comparable_with(self, other: str | None) -> bool:
        """Deux mesures sont-elles comparables sans supposition ?"""
        if other is None or self.crs is None:
            return False
        return other == self.crs or self.transform_from(other) is not None


class SpatialReferenceContext(BaseModel):
    """Référentiels et territoire d'un site, résolus et opposables.

    Versionné : un artefact cite la version qui l'a produit, et un changement
    de référentiel périme ce qui en dépend — sans périmer ce qui n'en dépend
    pas. C'est pourquoi `dependency_digests` est distinct de la provenance
    générale : un digest présent dans la provenance n'est pas automatiquement
    une dépendance de péremption.
    """

    model_config = ConfigDict(extra="forbid")

    version: str = "1.0.0"
    hotel_id: str
    resolved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    #: Position de référence ayant servi à résoudre le territoire et le CRS.
    reference_lat: float = Field(ge=-90, le=90)
    reference_lon: float = Field(ge=-180, le=180)

    territory_state: TerritoryState = TerritoryState.UNKNOWN

    #: Juridictions établies, du plus large au plus fin. Vide si non résolu.
    jurisdictions: list[str] = Field(default_factory=list)

    #: Comment elles ont été établies : source, méthode, empreinte.
    territory_evidence: list[str] = Field(default_factory=list)

    #: Référentiel des données d'entrée, et celui des calculs.
    source_crs: str = "EPSG:4326"
    working_crs: str | None = None

    working_unit: str | None = None
    working_axes: str | None = None
    working_area_of_use: list[float] | None = None

    #: Pourquoi ce CRS de travail plutôt qu'un autre.
    selection_method: str | None = None

    vertical: VerticalReference = Field(default_factory=VerticalReference)

    #: Empreintes des données ayant **réellement servi** à la résolution.
    #: Distinctes de la provenance générale : ce sont elles qui périment.
    dependency_digests: dict[str, str] = Field(default_factory=dict)

    @property
    def is_resolved(self) -> bool:
        return (
            self.territory_state is TerritoryState.RESOLVED
            and self.working_crs is not None
        )

    @property
    def vertical_is_usable(self) -> bool:
        """Peut-on qualifier une hauteur avec ce contexte ?

        Mesurer reste possible sans référentiel vertical ; **qualifier** ne
        l'est pas, car le seuil porterait sur une origine supposée.
        """
        return self.vertical.is_known

    @model_validator(mode="after")
    def _a_working_crs_declares_what_it_is(self) -> "SpatialReferenceContext":
        if self.working_crs is None:
            if self.territory_state is TerritoryState.RESOLVED:
                raise ValueError(
                    "territoire résolu sans référentiel de travail : la "
                    "résolution n'est pas terminée"
                )
            return self

        missing = [
            name for name, value in (
                ("working_unit", self.working_unit),
                ("working_axes", self.working_axes),
                ("working_area_of_use", self.working_area_of_use),
                ("selection_method", self.selection_method),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                f"référentiel de travail {self.working_crs!r} déclaré sans "
                f"{', '.join(missing)} — un CRS dont on ne peut pas dire "
                "l'unité, les axes, l'emprise ni pourquoi il a été choisi "
                "n'est pas opposable"
            )
        if self.territory_state is TerritoryState.UNKNOWN:
            raise ValueError(
                "référentiel de travail choisi sur un territoire inconnu : "
                "c'est exactement le défaut corrigé — le fuseau du pilote "
                "s'appliquait partout, parce que le territoire avait une "
                "valeur de repli"
            )
        return self

    def context_digest(self) -> str:
        """Empreinte de ce qui décide d'un calcul, et de rien d'autre.

        Trois champs seulement : les référentiels et le territoire. Ni la date
        de résolution, ni les preuves textuelles, ni les empreintes de
        dépendances n'y entrent — un contexte relu ou re-résolu à l'identique
        doit rendre la même empreinte, sans quoi tout manifeste deviendrait
        périmé au premier `geo reference --force`.

        C'est aussi ce qui distingue cette empreinte de la provenance générale :
        elle ne couvre que les entrées dont dépend réellement la géométrie.
        """
        import hashlib
        import json

        payload = json.dumps(
            {
                "source_crs": self.source_crs,
                "working_crs": self.working_crs,
                "jurisdictions": self.jurisdictions,
                "vertical_crs": self.vertical.crs,
                "height_type": self.vertical.height_type.value,
                "vertical_unit": self.vertical.unit,
                "vertical_transforms": [
                    item.model_dump(mode="json") for item in self.vertical.transforms
                ],
            },
            sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def contains(self, lat: float, lon: float) -> bool:
        """Ce point est-il dans l'emprise du référentiel de travail ?"""
        if not self.working_area_of_use:
            return False
        west, south, east, north = self.working_area_of_use
        return west <= lon <= east and south <= lat <= north

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "hotel_id": self.hotel_id,
            "resolved_at": self.resolved_at.isoformat(),
            "reference_position": {
                "lat": self.reference_lat, "lon": self.reference_lon
            },
            "territory": {
                "state": self.territory_state.value,
                "jurisdictions": self.jurisdictions,
                "evidence": self.territory_evidence,
            },
            "horizontal": {
                "source_crs": self.source_crs,
                "working_crs": self.working_crs,
                "unit": self.working_unit,
                "axes": self.working_axes,
                "area_of_use": self.working_area_of_use,
                "selection_method": self.selection_method,
            },
            "vertical": {
                "crs": self.vertical.crs,
                "height_type": self.vertical.height_type.value,
                "unit": self.vertical.unit,
                "provenance": self.vertical.provenance,
                "usable_for_qualification": self.vertical_is_usable,
                "transforms": [
                    item.model_dump(mode="json") for item in self.vertical.transforms
                ],
            },
            "dependency_digests": self.dependency_digests,
        }
