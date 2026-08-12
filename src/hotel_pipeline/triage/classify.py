"""Classification zero-shot par OpenCLIP (plan directeur §11, G2).

Aucun entraînement, aucun jeu annoté : les catégories sont décrites en langage
naturel et le modèle les note. C'est exactement le besoin du G2, et cela évite
d'écrire puis de calibrer un classifieur maison.

L'import d'`open_clip` est différé : la couche vision vit dans l'image GPU,
et le reste du pipeline doit rester importable sans elle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..logging import get_logger
from ..schemas import AssetCategory, ExteriorInterior

log = get_logger("classify")

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

#: Descriptions en langage naturel. Les intitulés comptent : ils sont le
#: « modèle », et se règlent en les relisant, pas en réentraînant.
EXTERIOR_PROMPTS: dict[ExteriorInterior, list[str]] = {
    ExteriorInterior.EXTERIOR: [
        "the exterior facade of a hotel building seen from outside",
        "an outdoor view of a building, sky and parking lot visible",
        "a street view of a commercial building",
    ],
    ExteriorInterior.INTERIOR: [
        "the interior of a hotel room with a bed",
        "an indoor swimming pool",
        "a hotel lobby, breakfast room or corridor seen from inside",
        "a bathroom interior",
    ],
}

CATEGORY_PROMPTS: dict[AssetCategory, list[str]] = {
    AssetCategory.ENTRANCE: ["the main entrance doors of a hotel, with a canopy or porch"],
    AssetCategory.FACADE: ["the full facade of a multi-storey hotel building"],
    AssetCategory.SIGN: ["a large hotel sign or illuminated lettering on a pole"],
    AssetCategory.PARKING: ["a parking lot with parked cars in front of a building"],
    AssetCategory.AERIAL: ["an aerial or drone view of a building seen from above"],
    AssetCategory.INTERIOR: ["an indoor room inside a hotel"],
}


@dataclass
class Classification:
    exterior_or_interior: ExteriorInterior
    exterior_confidence: float
    category: AssetCategory
    category_confidence: float


class Classifier:
    """Enveloppe OpenCLIP. Charge le modèle une seule fois."""

    def __init__(self, model_name: str = MODEL_NAME, pretrained: str = PRETRAINED) -> None:
        import open_clip  # import différé : dépendance de la couche vision
        import torch

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("chargement d'OpenCLIP %s (%s) sur %s", model_name, pretrained, self.device)

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self._exterior = self._encode_groups(EXTERIOR_PROMPTS)
        self._category = self._encode_groups(CATEGORY_PROMPTS)

    def _encode_groups(self, groups: dict) -> tuple[list, "object"]:
        """Encode chaque groupe de descriptions en un vecteur moyen normalisé."""
        torch = self._torch
        labels = list(groups)
        vectors = []
        with torch.no_grad():
            for label in labels:
                tokens = self.tokenizer(groups[label]).to(self.device)
                features = self.model.encode_text(tokens)
                features /= features.norm(dim=-1, keepdim=True)
                vectors.append(features.mean(dim=0))
        matrix = torch.stack(vectors)
        matrix /= matrix.norm(dim=-1, keepdim=True)
        return labels, matrix

    def classify(self, image_path: Path) -> Classification:
        from PIL import Image

        torch = self._torch
        image = Image.open(image_path).convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(tensor)
            features /= features.norm(dim=-1, keepdim=True)

            ext_labels, ext_matrix = self._exterior
            ext_scores = (100.0 * features @ ext_matrix.T).softmax(dim=-1)[0]
            ext_index = int(ext_scores.argmax())

            cat_labels, cat_matrix = self._category
            cat_scores = (100.0 * features @ cat_matrix.T).softmax(dim=-1)[0]
            cat_index = int(cat_scores.argmax())

        return Classification(
            exterior_or_interior=ext_labels[ext_index],
            exterior_confidence=float(ext_scores[ext_index]),
            category=cat_labels[cat_index],
            category_confidence=float(cat_scores[cat_index]),
        )
