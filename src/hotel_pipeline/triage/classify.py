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


#: Chaque sujet est jugé **indépendamment**, par opposition à une scène
#: alternative. Une softmax sur ce couple donne une probabilité par sujet, au
#: lieu d'un argmax qui désigne toujours un vainqueur même quand toutes les
#: hypothèses sont faibles.
#:
#: **Le terme opposé décrit une scène concrète, jamais une négation.** CLIP ne
#: traite pas la négation : « a photo with no building visible » contient le
#: mot « building » et l'emporte donc sur une image de bâtiment. Ce piège a
#: mesuré 0 hôtel sur 118 vues Street View dont la première montrait clairement
#: le WelcomINNS ; corriger l'opposé a fait passer le score de 0,01 à 0,60.
SUBJECT_PROMPTS: dict[str, tuple[str, str]] = {
    "building": (
        "a multi-storey hotel, motel or apartment building seen from outside",
        "an empty road, parking lot, sky, trees or open grass field",
    ),
    "entrance": (
        "the main entrance of a hotel, with glass doors under a canopy or porch",
        "a long blank facade, an empty road or a stretch of lawn",
    ),
    "sign": (
        "large lettering, an illuminated hotel sign or a branded pylon sign",
        "a plain brick wall, bare asphalt or a row of trees",
    ),
    "parking": (
        "a parking lot with several parked cars",
        "an empty lawn, a building interior or a clear stretch of asphalt",
    ),
    "roof": (
        "a rooftop seen from above, showing roof surfaces and their edges",
        "a ground-level view of walls, road surface and trees",
    ),
    "grounds": (
        "landscaped grounds with lawn, hedges, shrubs or planted beds",
        "bare asphalt pavement, a building interior or a blank wall",
    ),
    "road": (
        "a road, street or highway with lane markings and asphalt",
        "a building interior, a rooftop view or a dense lawn",
    ),
    "interior": (
        "an indoor room, corridor, lobby, bathroom or swimming pool",
        "an outdoor street scene with open sky above",
    ),
}

#: Au-dessus, le sujet est retenu ; en dessous du seuil bas, il est écarté ;
#: entre les deux, la décision est incertaine et appelle une revue.
SUBJECT_ACCEPT = 0.70
SUBJECT_REJECT = 0.40


@dataclass
class MultiLabelResult:
    """Probabilité par sujet, sans vainqueur imposé."""

    scores: dict[str, float]

    def accepted(self, threshold: float = SUBJECT_ACCEPT) -> list[str]:
        return sorted([s for s, p in self.scores.items() if p >= threshold])

    def uncertain(
        self, low: float = SUBJECT_REJECT, high: float = SUBJECT_ACCEPT
    ) -> list[str]:
        return sorted([s for s, p in self.scores.items() if low <= p < high])

    def confidence(self) -> float:
        """Netteté globale de la décision.

        Un jeu de scores tous proches de 0,5 traduit une image que le modèle
        ne comprend pas : la confiance doit alors être basse, même si un sujet
        franchit de justesse le seuil.
        """
        if not self.scores:
            return 0.0
        return max(abs(p - 0.5) * 2 for p in self.scores.values())


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
        self._subject_pairs = self._encode_pairs(SUBJECT_PROMPTS)

    def _encode_pairs(self, pairs: dict[str, tuple[str, str]]):
        """Encode chaque couple description positive / contraire."""
        torch = self._torch
        encoded = {}
        with torch.no_grad():
            for subject, (positive, negative) in pairs.items():
                tokens = self.tokenizer([positive, negative]).to(self.device)
                features = self.model.encode_text(tokens)
                features /= features.norm(dim=-1, keepdim=True)
                encoded[subject] = features
        return encoded

    def multi_label(self, image_path: Path) -> MultiLabelResult:
        """Juge chaque sujet indépendamment, sans désigner de vainqueur."""
        from PIL import Image

        torch = self._torch
        image = Image.open(image_path).convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        scores: dict[str, float] = {}
        with torch.no_grad():
            features = self.model.encode_image(tensor)
            features /= features.norm(dim=-1, keepdim=True)
            for subject, pair in self._subject_pairs.items():
                probabilities = (100.0 * features @ pair.T).softmax(dim=-1)[0]
                scores[subject] = float(probabilities[0])

        return MultiLabelResult(scores=scores)

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
