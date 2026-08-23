"""Encodage d'images par un modèle entraîné, et cache sur disque.

Le choix structurant : la ressemblance entre deux photographies n'est pas
calculée par des règles mais **mesurée dans l'espace d'un modèle**. Ce qu'une
heuristique sur les métadonnées ne peut pas faire — deux images prises au même
endroit à la même heure peuvent montrer deux bâtiments différents — un
embedding le fait sans qu'on lui décrive le bâtiment.

L'import d'`open_clip` reste différé : le reste du pipeline doit s'importer
sans la couche vision.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..logging import get_logger

log = get_logger("identity-embedding")

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

#: Version du cache. Un changement de modèle doit invalider les vecteurs
#: existants, sinon des embeddings de deux espaces différents se comparent.
CACHE_VERSION = f"{MODEL_NAME}/{PRETRAINED}/v1"


class VisionUnavailable(RuntimeError):
    """La couche vision manque : aucun jugement d'identité n'est possible."""


@dataclass
class ImageEmbedder:
    """Encode une image dans l'espace d'un modèle pré-entraîné."""

    model_name: str = MODEL_NAME
    pretrained: str = PRETRAINED
    _model: object | None = field(default=None, repr=False)
    _preprocess: object | None = field(default=None, repr=False)

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import open_clip
            import torch
        except ImportError as exc:  # pragma: no cover - dépend de l'installation
            raise VisionUnavailable(
                "open_clip et torch sont requis pour juger l'identité d'un "
                "bâtiment — installer l'extra 'vision'"
            ) from exc

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._torch = torch
        log.info("modèle chargé : %s / %s", self.model_name, self.pretrained)

    def encode(self, image_path: Path) -> np.ndarray:
        """Vecteur unitaire d'une image. Deux images proches ont un cosinus élevé."""
        from PIL import Image

        self.load()
        with Image.open(image_path) as raw:
            tensor = self._preprocess(raw.convert("RGB")).unsqueeze(0)
        with self._torch.no_grad():
            vector = self._model.encode_image(tensor)
        vector = vector / vector.norm(dim=-1, keepdim=True)
        return vector.squeeze(0).cpu().numpy().astype(np.float32)

    def encode_text(self, phrases: list[str]) -> np.ndarray:
        """Encode des descriptions, pour juger des attributs sans les coder."""
        import open_clip

        self.load()
        tokens = open_clip.tokenize(phrases)
        with self._torch.no_grad():
            vectors = self._model.encode_text(tokens)
        vectors = vectors / vectors.norm(dim=-1, keepdim=True)
        return vectors.cpu().numpy().astype(np.float32)


@dataclass
class EmbeddingIndex:
    """Vecteurs mémorisés, indexés par empreinte de fichier.

    L'empreinte porte sur le **contenu**, pas sur le chemin : une image
    renommée ou recadrée ailleurs n'est pas ré-encodée pour rien, et une image
    modifiée l'est forcément.
    """

    path: Path
    embedder: ImageEmbedder = field(default_factory=ImageEmbedder)
    _vectors: dict[str, np.ndarray] = field(default_factory=dict)
    _dirty: bool = False

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.path.is_file():
            payload = np.load(self.path, allow_pickle=False)
            if str(payload.get("version", "")) == CACHE_VERSION:
                keys = [str(k) for k in payload["keys"]]
                self._vectors = dict(zip(keys, payload["values"]))
                log.info("index rechargé : %d vecteurs", len(self._vectors))
            else:
                log.info("index ignoré : encodé par un autre modèle")

    @staticmethod
    def digest(image_path: Path) -> str:
        return hashlib.sha256(Path(image_path).read_bytes()).hexdigest()[:32]

    def vector_of(self, image_path: Path) -> np.ndarray:
        key = self.digest(image_path)
        cached = self._vectors.get(key)
        if cached is not None:
            return cached
        vector = self.embedder.encode(image_path)
        self._vectors[key] = vector
        self._dirty = True
        return vector

    def save(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self._vectors)
        np.savez(
            self.path,
            version=np.array(CACHE_VERSION),
            keys=np.array(keys),
            values=np.stack([self._vectors[k] for k in keys]),
        )
        self._dirty = False
        log.info("index écrit : %d vecteurs dans %s", len(keys), self.path)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Similarité de deux vecteurs unitaires."""
    return float(np.dot(a, b))
