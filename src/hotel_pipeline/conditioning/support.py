"""Appui photographique d'une trajectoire : chaque angle a-t-il une référence ?

Une frame peut être géométriquement irréprochable et n'être appuyée par aucune
photographie. Le générateur y invente alors la façade, sans que rien ne le
signale — mesuré sur le pilote, une orbite traversant 150°–260° était annoncée
`condition_strongly` sur 100 % des frames alors qu'aucune référence retenue ne
couvrait ce secteur, le plus grand trou angulaire atteignant 161°.

La géométrie et la photographie sont deux appuis distincts, et le second ne se
déduit pas du premier. Ce module mesure le second.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..logging import get_logger

log = get_logger("conditioning-support")

#: Écart angulaire en deçà duquel une référence appuie pleinement une vue.
#: Une façade reste reconnaissable sur une trentaine de degrés de rotation.
FULL_SUPPORT_DEG = 25.0

#: Au-delà, la référence ne montre plus la même face du bâtiment.
NO_SUPPORT_DEG = 70.0


@dataclass
class ReferenceView:
    """Une référence retenue, et l'angle sous lequel elle voit le bâtiment."""

    asset_id: str
    bearing_deg: float
    quality: float = 1.0

    def as_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "bearing_deg": round(self.bearing_deg, 1),
            "quality": round(self.quality, 3),
        }


@dataclass
class SupportMap:
    """Ce que les références attestent, angle par angle."""

    references: list[ReferenceView] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.references)

    def support_at(self, bearing_deg: float) -> tuple[float, str | None]:
        """Appui photographique d'un azimut, et la référence qui l'explique.

        L'appui décroît avec l'écart angulaire plutôt que de basculer d'un
        coup : une vue à 30° d'une référence en montre encore beaucoup, une vue
        à 65° presque plus.
        """
        if not self.references:
            return 0.0, None

        best_score, best_id = 0.0, None
        for reference in self.references:
            delta = abs((bearing_deg - reference.bearing_deg + 180.0) % 360.0 - 180.0)
            if delta <= FULL_SUPPORT_DEG:
                score = 1.0
            elif delta >= NO_SUPPORT_DEG:
                score = 0.0
            else:
                span = NO_SUPPORT_DEG - FULL_SUPPORT_DEG
                score = 1.0 - (delta - FULL_SUPPORT_DEG) / span
            score *= reference.quality
            if score > best_score:
                best_score, best_id = score, reference.asset_id
        return best_score, best_id

    def widest_gap(self) -> float:
        """Plus grand secteur angulaire sans aucune référence, en degrés."""
        if len(self.references) < 2:
            return 360.0
        bearings = sorted(r.bearing_deg % 360.0 for r in self.references)
        gaps = [
            (bearings[(i + 1) % len(bearings)] - bearings[i]) % 360.0
            for i in range(len(bearings))
        ]
        return float(max(gaps))

    def as_dict(self) -> dict:
        return {
            "count": len(self.references),
            "widest_gap_deg": round(self.widest_gap(), 1),
            "full_support_deg": FULL_SUPPORT_DEG,
            "no_support_deg": NO_SUPPORT_DEG,
            "references": [r.as_dict() for r in self.references],
        }


def _bearing_of(asset: dict) -> float | None:
    """Azimut sous lequel un asset voit le bâtiment, s'il est mesuré."""
    raw = asset.get("bearing_from_building_deg")
    if raw in (None, "None", ""):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value % 360.0


def _resolve_asset(asset_id: str, by_id: dict) -> dict | None:
    """Retrouve l'asset source d'une image, recadrages compris.

    Un recrop est nommé d'après le panorama dont il provient — par exemple
    ``SECT225_zj6pG6EOemMZ7d_54h_51f`` pour l'asset
    ``street_view-zj6pG6EOemMZ7dPlDXJeMA``. Sans cette résolution, aucune des
    meilleures références du pilote ne portait d'azimut, et la carte d'appui
    restait vide alors que les données existaient.

    Le suffixe ``_54h_`` du nom est le **cap de la caméra**, non l'azimut sous
    lequel la vue voit le bâtiment : les deux diffèrent de plus de cent degrés
    sur ce corpus. Seul l'asset source porte la mesure utile.
    """
    direct = by_id.get(asset_id)
    if direct is not None:
        return direct

    parts = asset_id.split("_")
    for token in parts[1:]:
        if len(token) < 10:
            continue
        for key, asset in by_id.items():
            if token in key:
                return asset
    return None


def from_screening(
    screening_path: Path,
    asset_manifest_path: Path,
    min_reference_score: float = 0.2,
) -> SupportMap:
    """Construit la carte d'appui depuis un dépistage d'identité.

    Seules les images dont l'identité est établie comptent : une photographie
    du bâtiment voisin n'appuie aucun angle de *ce* bâtiment-ci. L'azimut vient
    du manifeste, où il est mesuré — une référence qui n'en porte pas ne peut
    pas être placée sur le cercle, et ne compte donc pas.
    """
    screening = json.loads(Path(screening_path).read_text(encoding="utf-8"))
    manifest = json.loads(Path(asset_manifest_path).read_text(encoding="utf-8"))
    by_id = {a["id"]: a for a in manifest.get("assets", [])}

    references: list[ReferenceView] = []
    placed = unplaced = 0
    for entry in screening.get("assets", []):
        if entry.get("status") != "match":
            continue
        if float(entry.get("reference_score", 0.0)) < min_reference_score:
            continue
        asset = _resolve_asset(entry["asset_id"], by_id)
        bearing = _bearing_of(asset) if asset else None
        if bearing is None:
            unplaced += 1
            continue
        placed += 1
        references.append(
            ReferenceView(
                asset_id=entry["asset_id"],
                bearing_deg=bearing,
                quality=min(1.0, float(entry.get("reference_score", 0.0)) / 0.6),
            )
        )

    if unplaced:
        log.info(
            "%d référence(s) sans azimut mesuré : non placées sur le cercle",
            unplaced,
        )
    log.info("carte d'appui : %d référence(s) placée(s)", placed)
    return SupportMap(references=references)
