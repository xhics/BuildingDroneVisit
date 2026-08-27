"""Distingue les images **qui deviennent la vidéo** de celles qui la guident.

Deux rôles très différents ont été confondus jusqu'ici :

- une image de **structure** est la matière même du plan : le rendu 3D passé
  en ``--ref-video`` est retexturé image par image, il *devient* la vidéo ;
- une image de **référence** ne doit jamais apparaître. Elle sert à ce que le
  moteur sache à quoi ressemble vraiment le bâtiment — sa brique, ses
  toitures, la lumière qui s'y pose — sans qu'aucun de ses pixels ne se
  retrouve à l'écran.

Confondre les deux produit exactement ce qu'on a observé : une photo au sol
comportant un massif floral « UNESCO » utilisée comme référence, dont le
lettrage s'est incrusté dans le plan aérien. La photo devait informer, pas
figurer.

Ce module rend le rôle explicite, le transporte jusqu'au prompt, et l'y
énonce noir sur blanc pour le moteur.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: Rôles possibles.
STRUCTURE = "structure"
APPARENCE = "apparence"
AMBIANCE = "ambiance"

ROLE_LABELS_FR = {
    STRUCTURE: "structure et mouvement",
    APPARENCE: "apparence réelle du sujet",
    AMBIANCE: "ambiance et lumière",
}


@dataclass
class Reference:
    """Une image fournie au moteur, avec son rôle et son statut de sortie."""

    path: Path
    role: str = APPARENCE
    #: Cette image apparaît-elle dans la vidéo finale ? Faux pour toute
    #: référence : elle informe le moteur et disparaît.
    appears_in_output: bool = False
    note_fr: str = ""

    @property
    def role_label_fr(self) -> str:
        return ROLE_LABELS_FR.get(self.role, self.role)


@dataclass
class ReferenceSet:
    """Les images d'une étape, réparties par rôle."""

    structure: Reference | None = None
    appearance: list[Reference] = field(default_factory=list)

    def reference_only(self) -> list[Reference]:
        """Images à fournir au moteur sans qu'elles figurent au montage."""
        return [r for r in self.appearance if not r.appears_in_output]

    def describe_fr(self) -> str:
        """Résumé lisible, pour annoncer ce qui part au moteur."""
        parts = []
        if self.structure is not None:
            parts.append(f"structure : {self.structure.path.name}")
        hidden = self.reference_only()
        if hidden:
            parts.append(
                f"{len(hidden)} référence(s) non montée(s) : "
                + ", ".join(r.path.name for r in hidden)
            )
        return " | ".join(parts) if parts else "aucune image"

    def prompt_clause_fr(self) -> str:
        """Phrase à insérer dans le prompt pour expliciter le rôle des images.

        Sans elle, le moteur peut traiter une référence comme un plan à
        insérer — il en recopie alors le cadrage, le texte ou le premier plan.
        """
        hidden = self.reference_only()
        if not hidden:
            return ""

        labels = ", ".join(f"<Picture {i + 1}>" for i in range(len(hidden)))
        roles = "; ".join(
            f"<Picture {i + 1}> : {r.role_label_fr}"
            + (f", {r.note_fr}" if r.note_fr else "")
            for i, r in enumerate(hidden)
        )
        return (
            f"{labels} sont des documents de référence, fournis uniquement pour "
            "montrer à quoi ressemble réellement le lieu ({roles}). "
            "Ils ne font pas partie du plan : ne pas les insérer, ne pas les "
            "montrer, ne pas reprendre leur cadrage, leur texte, leurs "
            "inscriptions ni leur premier plan. Seule la vidéo source fournit "
            "les images du plan."
        ).replace("{roles}", roles)


def build_aerial_references(
    *,
    rendered_chunk: Path,
    exterior_photos: list[Path] | None = None,
    street_view_photos: list[Path] | None = None,
    max_appearance: int = 3,
) -> ReferenceSet:
    """Compose le jeu d'images d'un plan aérien.

    Le rendu 3D porte la structure et le mouvement ; les photos réelles ne
    servent qu'à renseigner l'apparence. Les vues Street View sont préférées
    lorsqu'elles existent : prises depuis le sol autour du bâtiment, elles en
    montrent les matériaux de bien plus près que le survol.
    """
    appearance: list[Reference] = []
    for path in (street_view_photos or [])[:max_appearance]:
        appearance.append(
            Reference(
                path=Path(path),
                role=APPARENCE,
                appears_in_output=False,
                note_fr="vue réelle prise depuis la rue, matériaux et couleurs exacts",
            )
        )
    remaining = max_appearance - len(appearance)
    for path in (exterior_photos or [])[:remaining]:
        appearance.append(
            Reference(
                path=Path(path),
                role=APPARENCE,
                appears_in_output=False,
                note_fr="photographie officielle de l'établissement",
            )
        )

    return ReferenceSet(
        structure=Reference(
            path=Path(rendered_chunk),
            role=STRUCTURE,
            appears_in_output=True,
            note_fr="trajectoire et géométrie du plan",
        ),
        appearance=appearance,
    )


__all__ = [
    "AMBIANCE",
    "APPARENCE",
    "ROLE_LABELS_FR",
    "STRUCTURE",
    "Reference",
    "ReferenceSet",
    "build_aerial_references",
]
