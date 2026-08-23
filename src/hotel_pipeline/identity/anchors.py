"""Les ancres : ce qui atteste que le bâtiment est bien celui-ci.

Tout le jugement d'identité repose sur elles, donc leur provenance est la
question centrale. Une ancre fausse ne dégrade pas le tri : elle l'inverse
proprement, en validant le voisin et en rejetant la cible.

Trois origines sont admises, par ordre de force :

1. **confirmée par un opérateur** — un humain a dit « c'est bien l'hôtel » ;
2. **lue sur l'enseigne** — un modèle d'OCR a lu le nom de l'établissement ;
3. **issue du site officiel** — l'établissement publie ses propres photos.

Rien d'autre. En particulier, jamais une image simplement *proche* du centroïde
géographique : c'est cette confusion qui a fait entrer le 1205 pour le 1195.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..logging import get_logger
from .embedding import EmbeddingIndex, cosine

log = get_logger("identity-anchors")

ANCHOR_FILE = "identity_anchors.json"

#: Une ancre lue sur enseigne vaut moins qu'une ancre confirmée à la main :
#: l'OCR peut lire l'enseigne d'un commerce voisin dans le même cadre.
ORIGIN_WEIGHT: dict[str, float] = {
    "operator_confirmed": 1.0,
    "sign_ocr": 0.85,
    "official_website": 0.9,
}


@dataclass
class Anchor:
    """Une image dont on tient pour établi qu'elle montre l'établissement."""

    asset_id: str
    path: Path
    origin: str
    evidence: str

    @property
    def weight(self) -> float:
        return ORIGIN_WEIGHT.get(self.origin, 0.5)

    def as_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "path": str(self.path),
            "origin": self.origin,
            "evidence": self.evidence,
            "weight": self.weight,
        }


@dataclass
class AnchorSet:
    """Les ancres d'un établissement, et leur cohérence mutuelle."""

    hotel_id: str
    anchors: list[Anchor] = field(default_factory=list)
    _vectors: list[np.ndarray] = field(default_factory=list, repr=False)

    def __len__(self) -> int:
        return len(self.anchors)

    def embed(self, index: EmbeddingIndex) -> None:
        self._vectors = [index.vector_of(a.path) for a in self.anchors]

    def coherence(self) -> float:
        """Ressemblance moyenne des ancres entre elles.

        Des ancres qui ne se ressemblent pas ne décrivent pas le même bâtiment :
        l'une d'elles est fausse, et le jugement qui en découle ne vaut rien.
        """
        if len(self._vectors) < 2:
            return 1.0
        scores = [
            cosine(self._vectors[i], self._vectors[j])
            for i in range(len(self._vectors))
            for j in range(i + 1, len(self._vectors))
        ]
        return float(np.mean(scores))

    def similarity(self, vector: np.ndarray) -> tuple[float, str | None]:
        """Score d'appartenance d'une image, et l'ancre qui l'explique.

        Le maximum pondéré, pas la moyenne : une photographie de l'arrière du
        bâtiment ne ressemble à aucune vue de face, et devrait quand même être
        reconnue dès qu'une seule ancre la couvre.
        """
        if not self._vectors:
            return 0.0, None
        scored = [
            (cosine(vector, vec) * anchor.weight, anchor.asset_id)
            for vec, anchor in zip(self._vectors, self.anchors)
        ]
        best = max(scored, key=lambda item: item[0])
        return float(best[0]), best[1]

    def as_dict(self) -> dict:
        return {
            "hotel_id": self.hotel_id,
            "count": len(self.anchors),
            "coherence": round(self.coherence(), 4),
            "anchors": [a.as_dict() for a in self.anchors],
        }


def load_anchors(workspace, index: EmbeddingIndex | None = None) -> AnchorSet:  # noqa: ANN001
    """Relit les ancres déclarées, et vérifie que les fichiers existent."""
    path = workspace.path("00_manifest", ANCHOR_FILE)
    if not path.is_file():
        return AnchorSet(hotel_id=workspace.hotel_id)

    payload = json.loads(path.read_text(encoding="utf-8"))
    anchors: list[Anchor] = []
    for entry in payload.get("anchors", []):
        image = Path(entry["path"])
        if not image.is_absolute():
            image = workspace.path(*Path(entry["path"]).parts)
        if not image.is_file():
            log.info("ancre ignorée, fichier absent : %s", image)
            continue
        anchors.append(
            Anchor(
                asset_id=str(entry["asset_id"]),
                path=image,
                origin=str(entry.get("origin", "operator_confirmed")),
                evidence=str(entry.get("evidence", "")),
            )
        )

    anchor_set = AnchorSet(hotel_id=str(payload.get("hotel_id", workspace.hotel_id)),
                           anchors=anchors)
    if index is not None and anchors:
        anchor_set.embed(index)
    return anchor_set


def write_anchors(workspace, anchor_set: AnchorSet) -> Path:  # noqa: ANN001
    """Publie les ancres, chemins relatifs au workspace quand c'est possible."""
    root = workspace.path()
    entries = []
    for anchor in anchor_set.anchors:
        try:
            relative = anchor.path.relative_to(root)
        except ValueError:
            relative = anchor.path
        entry = anchor.as_dict()
        entry["path"] = str(relative)
        entries.append(entry)

    payload = {
        "hotel_id": anchor_set.hotel_id,
        "coherence": round(anchor_set.coherence(), 4),
        "anchors": entries,
    }
    return workspace.write_json(f"00_manifest/{ANCHOR_FILE}", payload)
