"""Plans aériens générés à partir du rendu 3D pris comme référence.

Le rendu Cesium est géométriquement juste — vraie géométrie, trajectoire
réelle, cadrage correct — mais il *sent la 3D* : images parfaitement nettes,
sans flou de mouvement, éclairage cuit dans les textures. Trois défauts que
la règle des 180° et l'étalonnage traitent en prise de vue réelle, et
qu'aucun réglage de moteur 3D ne corrigera ici.

D'où ce module : les images rendues ne sont plus le livrable mais des
**références** passées au générateur vidéo, en mode Ref2VA (aucune image
verrouillée). Le rendu 3D fixe *où* est la caméra et *ce qu'elle voit* ; le
générateur fournit la matière photographique — flou de mouvement, grain,
lumière. Il produit nativement du 24 i/s, ce qui aligne aussi la cadence.

Chaque figure de vol (``maneuvers.artistic_maneuvers``) porte déjà son nom,
son intention et sa technique : ce vocabulaire alimente directement le
prompt, au lieu d'une description générique.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Le checkpoint Ref2VA accepte 9 images ; en deçà de 3 le mouvement est mal
#: contraint et le générateur dérive de la trajectoire.
MIN_REFERENCES = 3
MAX_REFERENCES = 6

#: Durées acceptées par le modèle (grille 124 + n*17 à 24 i/s).
MIN_SEGMENT_S = 6
MAX_SEGMENT_S = 15


@dataclass
class AerialSegment:
    """Un plan aérien : ses images de référence et la figure qu'il exécute."""

    frames: list[Path]
    maneuver_id: str
    name_fr: str
    purpose_fr: str
    skill_fr: str
    duration_s: int


def select_segments(
    frames: list[Path],
    maneuvers: list,
    *,
    segment_seconds: int = 10,
    source_fps: int = 12,
) -> list[AerialSegment]:
    """Découpe la suite d'images rendues en plans, un par tranche de temps.

    Les références d'un plan sont prélevées **dans ce plan uniquement** :
    piocher sur tout le vol donnerait au générateur des points de vue
    incompatibles et il inventerait un mouvement qui n'est pas le nôtre.

    ``maneuvers`` sert à nommer chaque tranche par la figure qu'elle traverse,
    pour que le prompt décrive le mouvement réellement exécuté.
    """
    if not frames:
        return []

    per_segment = max(1, segment_seconds * source_fps)
    segments: list[AerialSegment] = []

    for start in range(0, len(frames), per_segment):
        chunk = frames[start : start + per_segment]
        if len(chunk) < 2:
            # Une queue trop courte se rattache au plan précédent plutôt que
            # de produire un plan d'une seconde.
            if segments:
                segments[-1].frames.extend(chunk)
            continue

        # Figure dominante de la tranche, au prorata de l'avancement du vol.
        position = (start + len(chunk) / 2) / len(frames)
        maneuver = maneuvers[min(len(maneuvers) - 1, int(position * len(maneuvers)))] if maneuvers else None

        step = max(1, len(chunk) // MAX_REFERENCES)
        picked = chunk[::step][:MAX_REFERENCES]
        if len(picked) < MIN_REFERENCES:
            picked = chunk[: MIN_REFERENCES] if len(chunk) >= MIN_REFERENCES else chunk

        segments.append(
            AerialSegment(
                frames=list(picked),
                maneuver_id=getattr(maneuver, "id", "vol"),
                name_fr=getattr(maneuver, "name_fr", "Plan aérien"),
                purpose_fr=getattr(maneuver, "purpose_fr", ""),
                skill_fr=getattr(maneuver, "skill_fr", ""),
                duration_s=max(
                    MIN_SEGMENT_S, min(MAX_SEGMENT_S, round(len(chunk) / source_fps))
                ),
            )
        )
    return segments


def build_aerial_prompt(
    segment: AerialSegment,
    *,
    establishment_fr: str,
    time_of_day_fr: str = "",
    off_centre: bool = True,
) -> str:
    """Prompt Ref2VA décrivant le plan aérien à produire.

    ``retention_analysis`` déclare ``attribute_transfer`` : le générateur doit
    conserver l'architecture et l'implantation vues dans les rendus, mais
    n'est pas tenu d'en reproduire le rendu — c'est précisément ce qu'on lui
    demande d'améliorer.

    ``off_centre`` demande un cadrage aux tiers plutôt qu'un sujet centré :
    la caméra 3D vise le centre géométrique, ce qui est mécanique et se voit.
    """
    from .interior_journey import possessive_fr

    belongs = possessive_fr(establishment_fr)
    labels = "\n".join(
        f"- <Picture {i + 1}> : vue aérienne {belongs}, "
        f"position {i + 1} sur {len(segment.frames)} de la trajectoire"
        for i in range(len(segment.frames))
    )
    retention = "\n".join(
        f"- <Picture {i + 1}> : attribute_transfer" for i in range(len(segment.frames))
    )

    framing = (
        "Le bâtiment est placé selon la règle des tiers, jamais exactement au "
        "centre du cadre ; un élément de premier plan entre parfois dans le "
        "champ pour donner l'échelle."
        if off_centre
        else "Le bâtiment reste au centre du cadre."
    )
    light = time_of_day_fr or "lumière naturelle douce et directionnelle"

    return (
        "subject_definitions:\n"
        f"{labels}\n"
        f"- Sujet : {establishment_fr}, filmé depuis un drone.\n\n"
        "summary:\n"
        f"reference generation. {segment.name_fr} : {segment.purpose_fr}\n\n"
        "retention_analysis:\n"
        f"{retention}\n"
        "Les images fournies donnent la trajectoire, la géométrie du bâtiment "
        "et son implantation ; elles proviennent d'un rendu de synthèse et ne "
        "doivent pas être imitées photographiquement.\n\n"
        "detailed_description:\n"
        f"[Shot 1] Prise de vue aérienne réelle, filmée au drone. {segment.name_fr} : "
        f"{segment.skill_fr}. La caméra suit exactement le déplacement montré par "
        f"les images de référence, dans leur ordre. {framing}\n"
        "Mouvement continu et régulier, sans à-coup ni changement de direction "
        "brusque. Rendu photographique : flou de mouvement naturel, profondeur "
        f"de champ, matières et reflets réalistes. {light}.\n\n"
        "overall_soundscape:\n"
        "Souffle d'air léger et ambiance extérieure lointaine, sans bruit de moteur.\n\n"
        "non_diegetic_music:\n"
        "Nappe instrumentale ample et discrète, sans percussion marquée.\n"
    )


def total_cost_hint(segments: list[AerialSegment]) -> str:
    """Résumé du volume à générer, pour annoncer le coût avant de lancer."""
    seconds = sum(s.duration_s for s in segments)
    return f"{len(segments)} plan(s) générés, {seconds} s au total"


__all__ = [
    "MAX_REFERENCES",
    "MAX_SEGMENT_S",
    "MIN_REFERENCES",
    "MIN_SEGMENT_S",
    "AerialSegment",
    "build_aerial_prompt",
    "select_segments",
    "total_cost_hint",
]
