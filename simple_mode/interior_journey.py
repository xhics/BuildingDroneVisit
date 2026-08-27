"""Parcours intérieur généré à partir de photos traitées comme des références.

Le mode ``flf2v`` utilisé jusqu'ici **verrouille** l'image de départ et
celle d'arrivée : le modèle doit les reproduire exactement et se contente
d'interpoler entre elles. D'où un rendu trop littéral, où l'on reconnaît les
photos plutôt qu'un mouvement de caméra continu.

``minimax-h3-r2v`` fonctionne autrement : *« conditions on a whole reference
set rather than one or two locked frames »*, et *« has no frame anchors at
all »*. Les photos deviennent de la **matière** que le prompt distribue, pas
des images à recopier. Son contrat expose ``retention_analysis``, où l'on
déclare le degré de fidélité attendu pour chaque référence —
``weak_reference`` demandant explicitement une inspiration libre.

Le prompt suit le contrat officiel à six champs (MiniMax H3 Ref2VA). Ses
noms, son ordre et sa grammaire de références (``<Picture 1>``…) font partie
du contrat : les altérer invalide l'entrée.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Le checkpoint Ref2VA accepte au plus 9 images de référence.
MAX_REFERENCE_IMAGES = 9

#: Grille de durées du modèle : 124 + n*17 images à 24 i/s, soit 5,17 à 15,08 s.
MIN_DURATION_S = 6
MAX_DURATION_S = 15


@dataclass
class InteriorStop:
    """Une étape du parcours, adossée à une photo réelle."""

    photo: Path
    label_fr: str
    description_fr: str = ""


def _reference_label(index: int) -> str:
    return f"<Picture {index + 1}>"


def possessive_fr(name: str) -> str:
    """« de » + nom d'établissement, avec la bonne contraction.

    Les noms d'hôtels portent souvent leur article (« le Château Frontenac »,
    « les Jardins du Roy ») : concaténer naïvement produit « de le », fautif
    dans un prompt lu par un modèle francophone.
    """
    name = name.lstrip()
    lowered = name.lower()
    if lowered.startswith("le "):
        return "du " + name[3:]
    if lowered.startswith("les "):
        return "des " + name[4:]
    return "de " + name


def build_ref2va_prompt(
    stops: list[InteriorStop],
    *,
    entry_fr: str = "la façade vitrée du rez-de-chaussée",
    establishment_fr: str = "l'établissement",
    time_of_day_fr: str = "",
    fidelity: str = "weak_reference",
) -> str:
    """Rédige le prompt Ref2VA à six champs décrivant la traversée.

    ``fidelity`` porte l'intention de ton point 2 : ``weak_reference`` laisse
    le modèle recomposer librement à partir des photos, là où
    ``fully_preserved`` lui demanderait de les restituer telles quelles.

    Le parcours est raconté dans ``detailed_description`` — par où l'on
    entre, ce que l'on traverse, ce que l'on découvre — plutôt que d'être
    imposé image par image.
    """
    if not stops:
        raise ValueError("au moins une étape est nécessaire")

    belongs = possessive_fr(establishment_fr)
    subjects = "\n".join(
        f"- {_reference_label(i)} : {stop.label_fr.lower()} {belongs}"
        + (f", {stop.description_fr.rstrip('.').lower()}" if stop.description_fr else "")
        for i, stop in enumerate(stops)
    )
    retention = "\n".join(
        f"- {_reference_label(i)} : {fidelity}" for i in range(len(stops))
    )

    shots = []
    for index, stop in enumerate(stops):
        label = _reference_label(index)
        if index == 0:
            shots.append(
                f"[Shot 1] La caméra franchit {entry_fr} en avançant et débouche "
                f"dans un espace inspiré de {label}. Elle garde sa hauteur et "
                f"poursuit son mouvement vers l'avant."
            )
        else:
            shots.append(
                f"[Shot {index + 1}] Sans coupure, la caméra emprunte un couloir "
                f"puis découvre un espace inspiré de {label}. Le mouvement reste "
                f"continu, à vitesse régulière."
            )
    shots.append(
        f"[Shot {len(stops) + 1}] La caméra poursuit tout droit, retrouve une "
        f"ouverture vitrée et ressort du bâtiment vers l'extérieur."
    )

    ambience = time_of_day_fr or "lumière naturelle cohérente d'un espace à l'autre"

    return (
        "subject_definitions:\n"
        f"{subjects}\n\n"
        "summary:\n"
        "reference generation. Plan-séquence subjectif d'une caméra volante qui "
        f"traverse {establishment_fr} de part en part, en découvrant "
        "successivement plusieurs de ses espaces.\n\n"
        "retention_analysis:\n"
        f"{retention}\n"
        "Ces images sont des documents de référence, fournis uniquement pour "
        "montrer à quoi ressemblent réellement ces espaces — style, matériaux, "
        "ambiance. Elles ne font pas partie du plan : ne pas les insérer, ne pas "
        "les montrer telles quelles, ne pas reprendre leur cadrage ni les "
        "inscriptions, enseignes ou textes qu'elles contiennent. Le plan est un "
        "mouvement de caméra continu, jamais une succession de photographies.\n\n"
        "detailed_description:\n"
        + "\n".join(shots)
        + f"\nMouvement d'avance continu du début à la fin, hauteur d'œil "
        f"constante, aucune coupure ni marche arrière. {ambience}.\n\n"
        "overall_soundscape:\n"
        "Ambiance intérieure feutrée, réverbération douce des volumes, "
        "pas et voix lointaines très discrètes.\n\n"
        "non_diegetic_music:\n"
        "Nappe instrumentale discrète et continue, sans percussion marquée.\n"
    )


def build_stops(photos: list, *, limit: int = MAX_REFERENCE_IMAGES) -> list[InteriorStop]:
    """Convertit des ``hotel_sources.SourcePhoto`` classées en étapes.

    Deux photos de même catégorie qui se suivent produiraient une transition
    entre deux pièces presque identiques : on n'en garde qu'une.
    """
    stops: list[InteriorStop] = []
    seen_categories: set[str] = set()
    for photo in photos:
        category = getattr(photo, "category", "autre")
        if category in seen_categories:
            continue
        seen_categories.add(category)
        stops.append(
            InteriorStop(
                photo=Path(getattr(photo, "path", photo)),
                label_fr=getattr(photo, "label_fr", "espace intérieur"),
                description_fr=getattr(photo, "description_fr", ""),
            )
        )
        if len(stops) >= limit:
            break
    return stops


__all__ = [
    "MAX_DURATION_S",
    "MAX_REFERENCE_IMAGES",
    "MIN_DURATION_S",
    "InteriorStop",
    "build_ref2va_prompt",
    "build_stops",
]
