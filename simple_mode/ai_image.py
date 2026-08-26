"""Illustration artistique du trajet, générée par OpenAI.

Un visuel marketing **séparé** du PNG précis produit par ``render.render()``
(PIL, coordonnées exactes). Un modèle de génération d'image ne suit pas des
coordonnées pixel — le tracé qui apparaît ici est indicatif, pas mesuré.
``render.render()`` reste la seule source de vérité géométrique du trajet.
"""

from __future__ import annotations

import base64
import io

from openai import OpenAI
from PIL import Image

from .maneuvers import Maneuver

DEFAULT_IMAGE_MODEL = "gpt-image-1"


class AIIllustrationError(RuntimeError):
    """L'illustration n'a pas pu être générée."""


def generate_illustration(
    reference_image: Image.Image,
    maneuvers: list[Maneuver],
    address: str,
    api_key: str,
    *,
    model: str = DEFAULT_IMAGE_MODEL,
    size: str = "1024x1024",
) -> Image.Image:
    """Édite la photo satellite pour y superposer un rendu stylisé du trajet."""
    client = OpenAI(api_key=api_key)

    buf = io.BytesIO()
    reference_image.convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    buf.name = "satellite.png"  # aide le SDK à déduire le type MIME

    figures = "; ".join(f"{m.name_fr} ({m.purpose_fr})" for m in maneuvers)
    prompt = (
        f"Illustration cinématique, vue aérienne, du trajet d'un drone de "
        f"tournage autour du bâtiment visible sur cette photo satellite "
        f"({address}). Trace {len(maneuvers)} boucles de vol colorées et "
        f"distinctes autour du bâtiment, dans cet ordre : {figures}. Style "
        f"infographie technique élégante, fond satellite conservé, flèches "
        f"indiquant le sens de vol, légende discrète."
    )

    try:
        result = client.images.edit(
            model=model,
            image=buf,
            prompt=prompt,
            size=size,
            response_format="b64_json",
        )
    except Exception as exc:  # noqa: BLE001 — remonté tel quel à l'appelant
        raise AIIllustrationError(f"appel OpenAI (images.edit) échoué : {exc}") from exc

    if not result.data or not result.data[0].b64_json:
        raise AIIllustrationError("le modèle n'a renvoyé aucune image")

    image_bytes = base64.b64decode(result.data[0].b64_json)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


__all__ = ["AIIllustrationError", "DEFAULT_IMAGE_MODEL", "generate_illustration"]
