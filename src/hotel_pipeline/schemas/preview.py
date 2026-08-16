"""Ce qu'un aperçu établit, **pour un besoin précis** (collecte V2).

Une preview est téléchargée pour vérifier ce qu'une vue montre. Sans trace
structurée, ce constat se perdait : `review geometry --measure` conservait ses
mesures dans `GeometryEntry`, `demands assess` lisait les champs plats de
l'asset, et rien ne reliait l'un à l'autre. Après téléchargement, le pipeline
ne savait donc pas transformer l'aperçu en preuve ou en rejet.

Le couple `(asset_id, demand_id)` est l'unité, non l'asset seul. Une même
acquisition sert souvent deux besoins — cadrer la façade et documenter
l'enseigne — et le verdict n'est pas le même :

```text
asset A / façade avant     part dans le cadre 0,42 → insuffisant
asset A / enseigne         l'enseigne est lisible → établi
```

Conclure de l'un à l'autre ferait porter à la façade une mesure prise sur
l'enseigne. C'est la contamination que ce module interdit.

Append-only : une évaluation ne se corrige pas, elle se complète. Un verdict
qui change sans trace se relit comme s'il avait toujours été celui-là.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PreviewVerdict(StrEnum):
    """Ce que l'aperçu établit pour ce besoin."""

    #: La vue montre ce que le besoin demande, dans les proportions exigées.
    ESTABLISHED = "established"

    #: Elle ne le montre pas — et l'on sait pourquoi.
    REFUTED = "refuted"

    #: Vue examinée, verdict impossible : une métrique exigée n'a pas pu être
    #: mesurée. Distinct de `refuted` — l'un dit « non », l'autre « on ne sait
    #: toujours pas ».
    INCONCLUSIVE = "inconclusive"


class PreviewAssessment(BaseModel):
    """Le constat d'un aperçu, pour **un** couple asset/besoin.

    Porte de quoi se rattacher à ce qui l'a produit — plan, requête, fichier —
    sans quoi un verdict flotterait sans état vérifiable.
    """

    model_config = ConfigDict(extra="forbid")

    asset_id: str
    demand_id: str

    #: --- filiation : à quoi ce constat se rapporte -------------------------
    plan_id: str = Field(min_length=1)

    #: Empreinte de la requête qui a produit le fichier examiné. Deux
    #: résolutions du même candidat sont deux fichiers, et un constat pris sur
    #: l'un ne vaut pas pour l'autre.
    request_digest: str = Field(min_length=1)

    #: Empreinte du fichier lui-même : ce qu'on a réellement regardé.
    checksum: str = Field(min_length=1)

    #: Ce que la vue devait montrer, tel que le besoin le désigne.
    target_ref: str = ""

    #: --- ce qui a été observé ----------------------------------------------
    #: Emprise de la cible dans l'image, en pixels : `[x0, y0, x1, y1]` ou un
    #: polygone. Rendre la mesure relisible sans rouvrir le fichier.
    observed_geometry: list[float] = Field(default_factory=list)
    observed_polygon_wkt: str | None = None

    #: Métriques du besoin. `None` signifie **non mesurée**, jamais zéro : un
    #: zéro affirmerait qu'on a regardé et rien trouvé.
    in_frame_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    projected_width_fraction: float | None = Field(default=None, ge=0.0)
    visible_fraction: float | None = Field(default=None, ge=0.0, le=1.0)

    verdict: PreviewVerdict = PreviewVerdict.INCONCLUSIVE

    #: Ce qui reste inconnu après examen. Vide quand le verdict est établi.
    unmeasured: list[str] = Field(default_factory=list)

    #: Pourquoi ce verdict. Obligatoire : un constat sans motif ne se conteste
    #: pas, et c'est précisément ce qu'un relecteur doit pouvoir faire.
    rationale: str = Field(min_length=1)

    #: Qui l'a prononcé — un opérateur nommé, ou l'outil de mesure.
    assessed_by: str = Field(min_length=1)
    assessed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    #: Chemins ou identifiants de ce qui étaye le constat : capture annotée,
    #: rapport de mesure. Une affirmation sans pièce reste une opinion.
    evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _verdict_is_supported(self) -> "PreviewAssessment":
        if self.verdict is PreviewVerdict.ESTABLISHED and self.unmeasured:
            raise ValueError(
                f"{self.asset_id}/{self.demand_id} : verdict « établi » avec "
                f"{sorted(self.unmeasured)} non mesuré(s) — ce qu'on ignore ne "
                "peut pas fonder ce qu'on affirme"
            )
        if self.verdict is PreviewVerdict.INCONCLUSIVE and not self.unmeasured:
            raise ValueError(
                f"{self.asset_id}/{self.demand_id} : verdict « non concluant » "
                "sans rien d'inconnu — il manque la raison de ne pas conclure"
            )
        return self


class PreviewAssessmentLog(BaseModel):
    """Histoire append-only des constats d'aperçu.

    Un couple peut être réexaminé — meilleure mesure, second avis — et les deux
    constats restent. Le plus récent fait foi ; l'écraser effacerait ce qui a
    été vu la première fois, et la révision se lirait comme une certitude.
    """

    model_config = ConfigDict(extra="forbid")

    hotel_id: str
    entries: list[PreviewAssessment] = Field(default_factory=list)

    def latest_for(self, asset_id: str, demand_id: str) -> PreviewAssessment | None:
        """Dernier constat de ce couple, ou `None`."""
        matching = [
            entry for entry in self.entries
            if entry.asset_id == asset_id and entry.demand_id == demand_id
        ]
        return max(matching, key=lambda e: e.assessed_at) if matching else None

    def established_for(self, demand_id: str) -> set[str]:
        """Assets dont l'aperçu **établit** ce besoin.

        Un constat pris sur un autre besoin n'y figure pas : c'est ce qui
        empêche une mesure d'enseigne de créditer une façade.
        """
        established: set[str] = set()
        for entry in self.entries:
            if entry.demand_id != demand_id:
                continue
            latest = self.latest_for(entry.asset_id, demand_id)
            if latest is not None and latest.verdict is PreviewVerdict.ESTABLISHED:
                established.add(entry.asset_id)
        return established


def promotable(
    assessment: "PreviewAssessment | None", demand,  # noqa: ANN001
) -> tuple[bool, list[str]]:
    """Un aperçu autorise-t-il d'acquérir en pleine résolution ?

    Toutes les métriques **exigées par ce besoin** doivent être établies. Une
    seule inconnue suffit à s'en tenir à l'aperçu : dépenser la pleine
    résolution sur une vue dont on ignore encore ce qu'elle montre, c'est
    payer pour découvrir ce qu'un aperçu disait déjà.

    Rend aussi ce qui manque, pour que le refus s'explique.
    """
    if assessment is None:
        return False, ["aucun aperçu examiné pour ce besoin"]

    if assessment.verdict is not PreviewVerdict.ESTABLISHED:
        return False, [
            f"aperçu {assessment.verdict.value} : {assessment.rationale}"
        ]

    missing: list[str] = []
    if getattr(demand, "min_projected_width_fraction", 0.0) > 0:
        if assessment.projected_width_fraction is None:
            missing.append("largeur projetée non mesurée")
        elif (
            assessment.projected_width_fraction
            < demand.min_projected_width_fraction
        ):
            missing.append(
                f"largeur projetée {assessment.projected_width_fraction:.3f} "
                f"sous le minimum {demand.min_projected_width_fraction:.3f}"
            )

    if getattr(demand, "min_visible_fraction", 0.0) > 0:
        if assessment.visible_fraction is None:
            missing.append("fraction visible non mesurée")
        elif assessment.visible_fraction < demand.min_visible_fraction:
            missing.append(
                f"fraction visible {assessment.visible_fraction:.3f} sous le "
                f"minimum {demand.min_visible_fraction:.3f}"
            )

    return not missing, missing
