"""Récupérer auprès de Mapillary ce que l'ingestion n'a pas conservé.

Deux champs manquent au manifeste alors que la source les publie : le cap
**recalculé** par le fournisseur, et la séquence d'acquisition. Le premier
n'est pas un doublon du cap déclaré — mesuré sur ce pilote, il en diffère de
huit à vingt-deux degrés et rapproche chaque vue de la direction réelle du
bâtiment. Le second permet de retrouver les images voisines d'un même parcours.

**Le cap déclaré n'est pas écrasé.** Les deux valeurs disent des choses
différentes : l'une est ce que l'appareil a enregistré, l'autre ce que le
fournisseur déduit de la structure de la séquence. Remplacer la première par la
seconde effacerait la trace de ce qui a été mesuré ; les consommateurs
choisissent, en connaissance de cause.

L'enrichissement ne porte que sur les images Mapillary : les autres sources
n'exposent pas ces champs, et leur inventer un équivalent reviendrait à
fabriquer une donnée.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .logging import get_logger

log = get_logger("mapillary-enrich")

#: Images demandées par requête. L'API accepte des lots ; les interroger une à
#: une multiplierait les appels sans rien gagner.
BATCH_SIZE = 50

#: Champs demandés. `computed_compass_angle` est le cap recalculé, `sequence`
#: l'identifiant de parcours.
FIELDS = "id,sequence,compass_angle,computed_compass_angle,is_pano"

#: Écart, en degrés, au-delà duquel la divergence entre cap déclaré et cap
#: calculé mérite d'être signalée. Un désaccord franc peut révéler une image
#: mal géoréférencée autant qu'un cap déclaré fautif.
DIVERGENCE_NOTICE_DEG = 30.0


@dataclass
class EnrichedAsset:
    """Ce que la source ajoute à un asset, et ce qu'elle n'ajoute pas."""

    asset_id: str
    computed_heading_deg: float | None = None
    sequence_id: str | None = None
    declared_heading_deg: float | None = None
    reason: str = ""

    @property
    def divergence_deg(self) -> float | None:
        """Écart entre le cap déclaré et le cap calculé, s'ils existent tous deux."""
        if self.computed_heading_deg is None or self.declared_heading_deg is None:
            return None
        gap = abs(self.computed_heading_deg - self.declared_heading_deg) % 360.0
        return min(gap, 360.0 - gap)

    def as_dict(self) -> dict:
        divergence = self.divergence_deg
        return {
            "asset_id": self.asset_id,
            "computed_heading_deg": (
                round(self.computed_heading_deg, 3)
                if self.computed_heading_deg is not None
                else None
            ),
            "sequence_id": self.sequence_id,
            "divergence_deg": round(divergence, 1) if divergence is not None else None,
            "reason": self.reason,
        }


@dataclass
class EnrichmentReport:
    enriched: list[EnrichedAsset] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    @property
    def with_heading(self) -> list[EnrichedAsset]:
        return [e for e in self.enriched if e.computed_heading_deg is not None]

    @property
    def with_sequence(self) -> list[EnrichedAsset]:
        return [e for e in self.enriched if e.sequence_id]

    def sequences(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for entry in self.with_sequence:
            grouped.setdefault(entry.sequence_id, []).append(entry.asset_id)
        return grouped

    def diverging(self) -> list[EnrichedAsset]:
        return [
            entry
            for entry in self.with_heading
            if (entry.divergence_deg or 0.0) >= DIVERGENCE_NOTICE_DEG
        ]

    def as_dict(self) -> dict:
        divergences = [
            e.divergence_deg for e in self.with_heading if e.divergence_deg is not None
        ]
        return {
            "requested": len(self.enriched),
            "with_computed_heading": len(self.with_heading),
            "with_sequence": len(self.with_sequence),
            "sequences": {k: len(v) for k, v in self.sequences().items()},
            "median_divergence_deg": (
                round(sorted(divergences)[len(divergences) // 2], 1)
                if divergences
                else None
            ),
            "diverging_count": len(self.diverging()),
            "enriched": [e.as_dict() for e in self.enriched],
            "provenance": self.provenance,
            "caveats": [
                "le cap calculé ne remplace pas le cap déclaré : les deux sont "
                "conservés, et le consommateur choisit",
                "un écart important peut venir du cap déclaré comme du "
                "géoréférencement — il signale, il ne tranche pas",
            ],
        }


def _normalise(value) -> float | None:  # noqa: ANN001
    """Ramène un cap dans [0, 360). Mapillary publie parfois des négatifs."""
    if value is None:
        return None
    try:
        return float(value) % 360.0
    except (TypeError, ValueError):
        return None


def enrich(assets, fetch) -> EnrichmentReport:  # noqa: ANN001
    """Complète les assets Mapillary depuis la source.

    `fetch` reçoit une liste d'identifiants de la source et rend un
    dictionnaire indexé par ces identifiants. L'injecter permet de tester sans
    réseau, et de remplacer le transport sans toucher à la logique.
    """
    report = EnrichmentReport()
    targets = [a for a in assets if str(a.id).startswith("mapillary-")]
    if not targets:
        log.info("aucun asset Mapillary à enrichir")
        return report

    by_source_id = {str(a.id).split("-", 1)[1]: a for a in targets}
    payloads: dict[str, dict] = {}
    identifiers = list(by_source_id)
    for start in range(0, len(identifiers), BATCH_SIZE):
        chunk = identifiers[start : start + BATCH_SIZE]
        try:
            payloads.update(fetch(chunk) or {})
        except Exception as exc:  # noqa: BLE001 - une panne réseau n'arrête pas le lot
            log.warning("lot %d non récupéré (%s)", start // BATCH_SIZE, exc)

    for source_id, asset in by_source_id.items():
        payload = payloads.get(source_id)
        if payload is None:
            report.enriched.append(
                EnrichedAsset(
                    asset_id=asset.id,
                    declared_heading_deg=asset.heading_deg,
                    reason="la source ne rend rien pour cette image",
                )
            )
            continue

        computed = _normalise(payload.get("computed_compass_angle"))
        sequence = payload.get("sequence")
        report.enriched.append(
            EnrichedAsset(
                asset_id=asset.id,
                computed_heading_deg=computed,
                sequence_id=str(sequence) if sequence else None,
                declared_heading_deg=asset.heading_deg,
                reason="enrichi" if computed is not None else "cap calculé absent",
            )
        )

    report.provenance = {
        "fields": FIELDS,
        "batch_size": BATCH_SIZE,
        "assets_targeted": len(targets),
    }
    log.info(
        "enrichissement : %d cap(s) calculé(s), %d séquence(s) sur %d image(s)",
        len(report.with_heading),
        len(report.sequences()),
        len(targets),
    )
    return report


def apply(assets, report: EnrichmentReport) -> int:
    """Pose les champs enrichis sur les assets. Rend le nombre modifié."""
    by_id = {a.id: a for a in assets}
    touched = 0
    for entry in report.enriched:
        asset = by_id.get(entry.asset_id)
        if asset is None:
            continue
        changed = False
        if entry.computed_heading_deg is not None:
            asset.computed_heading_deg = entry.computed_heading_deg
            changed = True
        if entry.sequence_id:
            asset.sequence_id = entry.sequence_id
            changed = True
        touched += int(changed)
    return touched


__all__ = [
    "BATCH_SIZE",
    "DIVERGENCE_NOTICE_DEG",
    "FIELDS",
    "EnrichedAsset",
    "EnrichmentReport",
    "apply",
    "enrich",
]
