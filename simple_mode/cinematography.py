"""Intention cinématographique : ce qu'on raconte, et ce qu'il faut ajouter.

Les modules précédents savaient produire un vol juste — géométrie exacte,
vitesse constante, altitude tenue. Juste, mais pas vendeur : un plan-séquence
de quatre-vingts secondes qui tourne autour d'un bâtiment n'est pas un film
promotionnel, c'est une démonstration technique.

Ce module ajoute la couche qui manquait, et qui décide de deux choses :

1. **Le découpage.** Une vidéo qui vend enchaîne des plans d'intentions
   différentes — situer, révéler, détailler, faire vivre — avec un rythme.
   Le plan unique est remplacé par une liste de plans, chacun avec sa durée
   et sa raison d'être.

2. **Ce que la génération doit inventer.** Le rendu 3D est vide : ni
   passants, ni voitures, ni reflets, ni atmosphère. Jusqu'ici on demandait
   au moteur de *préserver* la structure ; on lui demande maintenant de
   **compléter ce qui manque**, en gardant l'implantation comme cadre. Le
   rendu devient le décor, la génération y remet la vie.

Le traitement est écrit pour l'établissement concerné, à partir de ses
propres photos classées : un hôtel de bord de mer et un palace urbain
n'appellent ni les mêmes plans ni les mêmes ajouts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

#: Intentions de plan, dans l'ordre où elles servent un montage promotionnel.
SHOT_INTENTS = {
    "situer": "place le lieu dans son environnement, donne l'échelle",
    "reveler": "dévoile progressivement le bâtiment, crée l'attente",
    "detailler": "s'attarde sur une qualité précise — matière, ornement, angle",
    "faire_vivre": "montre l'usage : allées et venues, terrasses occupées",
    "signature": "le plan qu'on retient, celui de l'affiche",
    "conclure": "referme en s'éloignant, laisse une dernière impression",
}

#: Figures disponibles, reprises du vocabulaire de tournage par drone.
CAMERA_MOVES = {
    "reveal": "montée qui dévoile le sujet derrière un premier plan",
    "orbite_parallaxe": "orbite serrée, l'arrière-plan défile et donne la profondeur",
    "helix": "orbite qui descend en spirale, la façade défile de haut en bas",
    "push_in": "rapprochement franc, décéléré à l'arrivée",
    "pull_out": "recul qui ouvre sur le contexte",
    "survol": "passage au-dessus, le regard bascule vers l'arrière",
    "lateral": "travelling latéral qui longe la façade",
}


@dataclass
class Shot:
    """Un plan : son intention, son mouvement, et ce que l'IA doit y ajouter."""

    intent: str
    move: str
    duration_s: int
    #: Ce que le plan doit montrer, en une phrase.
    subject_fr: str
    #: Éléments absents du rendu 3D que la génération doit créer.
    add_fr: list[str] = field(default_factory=list)
    #: ``exterieur`` ou ``interieur``. Les deux ne se fabriquent pas de la
    #: même façon : un plan extérieur part du rendu 3D, qui donne le trajet
    #: réel ; un plan intérieur part des photos de l'établissement, faute de
    #: toute donnée 3D à l'intérieur des murs. Sans cette distinction, le
    #: traitement demande une orbite de drone autour d'une piscine couverte.
    scope: str = "exterieur"

    @property
    def is_interior(self) -> bool:
        return self.scope == "interieur"

    @property
    def move_label_fr(self) -> str:
        return CAMERA_MOVES.get(self.move, self.move)

    @property
    def intent_label_fr(self) -> str:
        return SHOT_INTENTS.get(self.intent, self.intent)


@dataclass
class Treatment:
    """Le traitement complet : l'intention d'ensemble et ses plans."""

    establishment_fr: str
    pitch_fr: str
    shots: list[Shot] = field(default_factory=list)

    @property
    def total_duration_s(self) -> int:
        return sum(s.duration_s for s in self.shots)

    def describe_fr(self) -> str:
        lines = [f"« {self.pitch_fr} »", ""]
        for index, shot in enumerate(self.shots, 1):
            lines.append(
                f"{index}. [{shot.scope[:3]}/{shot.intent}] {shot.move_label_fr} "
                f"— {shot.duration_s}s"
            )
            lines.append(f"   {shot.subject_fr}")
            if shot.add_fr:
                lines.append(f"   à ajouter : {', '.join(shot.add_fr)}")
        return "\n".join(lines)


#: Éléments que le rendu 3D ne contient jamais. Les nommer explicitement est
#: ce qui distingue une image habitée d'une maquette : la photogrammétrie est
#: capturée à des heures creuses et nettoyée de ses passants.
MISSING_BY_DEFAULT = [
    "des personnes en mouvement, à une échelle crédible pour la distance",
    "des véhicules et une circulation discrète aux abords",
    "des reflets et des jeux de lumière sur les vitrages",
    "une atmosphère : légère brume, particules dans les rayons rasants",
    "de la végétation qui bouge sous le vent",
]


_SYSTEM_PROMPT = (
    "Tu es directeur de la photographie, spécialisé dans les films "
    "promotionnels d'hôtellerie tournés au drone. Tu conçois des traitements "
    "courts, rythmés, qui donnent envie de réserver. Réponds uniquement en "
    "JSON valide, sans texte autour."
)


def build_treatment_prompt(
    establishment_fr: str,
    *,
    spaces_fr: list[str],
    time_of_day_fr: str,
    total_seconds: int,
) -> str:
    """Consigne d'écriture du traitement, adaptée à l'établissement."""
    moves = ", ".join(CAMERA_MOVES)
    intents = ", ".join(SHOT_INTENTS)
    spaces = ", ".join(spaces_fr) if spaces_fr else "non renseignés"
    return (
        f"Établissement : {establishment_fr}.\n"
        f"Espaces connus, d'après ses photos : {spaces}.\n"
        f"Ambiance lumineuse : {time_of_day_fr}.\n"
        f"Durée cible : {total_seconds} secondes au total.\n\n"
        "Écris le traitement d'un film promotionnel aérien. Découpe-le en 4 à 6 "
        "plans, chacun avec une intention distincte — ne répète pas deux fois la "
        "même. Le film doit avoir une progression : on situe, on approche, on "
        "détaille, on referme.\n\n"
        f"Intentions possibles : {intents}.\n"
        f"Mouvements de caméra possibles : {moves}.\n\n"
        "Pour chaque plan, précise ce qu'il faut AJOUTER à un rendu 3D vide "
        "pour qu'il paraisse filmé : présence humaine, circulation, reflets, "
        "atmosphère, mouvement de végétation. Sois concret et proportionné à "
        "la distance de la caméra.\n\n"
        "Indique aussi la portée de chaque plan : \"exterieur\" s'il est filmé "
        "au drone autour du bâtiment, \"interieur\" s'il montre un espace "
        "intérieur. Un plan intérieur ne peut pas recevoir de mouvement de "
        "drone : une caméra ne fait pas d'orbite autour d'une piscine "
        "couverte. Garde la majorité des plans en extérieur, et au plus deux "
        "plans intérieurs.\n\n"
        'Réponds ainsi : {"pitch": "<une phrase qui résume le film>", '
        '"shots": [{"intent": "<intention>", "move": "<mouvement>", '
        '"scope": "exterieur|interieur", "duration_s": <entier>, '
        '"subject": "<ce que montre le plan>", '
        '"add": ["<élément à ajouter>", ...]}]}'
    )


def parse_treatment(payload: dict, establishment_fr: str) -> Treatment:
    """Convertit la réponse du modèle en traitement exploitable."""
    shots = []
    for raw in payload.get("shots", []):
        move = str(raw.get("move", "orbite_parallaxe"))
        scope = "interieur" if str(raw.get("scope", "")) == "interieur" else "exterieur"
        shots.append(
            Shot(
                intent=str(raw.get("intent", "situer")),
                # Un plan intérieur ne suit aucune trajectoire de drone : son
                # mouvement est une avancée simple dans l'espace.
                move=move if (move in CAMERA_MOVES and scope == "exterieur") else (
                    "push_in" if scope == "interieur" else "orbite_parallaxe"
                ),
                duration_s=max(4, min(15, int(raw.get("duration_s", 8)))),
                subject_fr=str(raw.get("subject", "")),
                add_fr=[str(a) for a in raw.get("add", [])] or list(MISSING_BY_DEFAULT[:3]),
                scope=scope,
            )
        )
    return Treatment(
        establishment_fr=establishment_fr,
        pitch_fr=str(payload.get("pitch", "")),
        shots=shots,
    )


def author_treatment(
    establishment_fr: str,
    openai_key: str,
    *,
    spaces_fr: list[str],
    time_of_day_fr: str,
    total_seconds: int = 45,
    model: str = "gpt-4o-mini",
) -> Treatment:
    """Fait écrire le traitement par un modèle, pour cet établissement."""
    from openai import OpenAI

    client = OpenAI(api_key=openai_key)
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_treatment_prompt(
                    establishment_fr,
                    spaces_fr=spaces_fr,
                    time_of_day_fr=time_of_day_fr,
                    total_seconds=total_seconds,
                ),
            },
        ],
    )
    payload = json.loads(response.choices[0].message.content or "{}")
    return parse_treatment(payload, establishment_fr)


def build_completion_prompt(
    shot: Shot,
    *,
    establishment_fr: str,
    time_of_day_fr: str,
    reference_clause_fr: str = "",
) -> str:
    """Prompt demandant au moteur de **compléter** le rendu, non de le copier.

    Renversement par rapport aux essais précédents : on ne demande plus la
    préservation d'une structure, mais l'ajout de ce que le rendu ne contient
    pas. L'implantation reste le cadre — le bâtiment, sa masse, sa place —
    tandis que la matière, la lumière et la vie sont à créer.
    """
    additions = "; ".join(shot.add_fr) if shot.add_fr else "; ".join(MISSING_BY_DEFAULT[:3])
    return (
        f"{establishment_fr} — film promotionnel, prise de vue aérienne réelle.\n"
        f"Intention du plan : {shot.intent_label_fr}. {shot.subject_fr}\n"
        f"Mouvement : {shot.move_label_fr}.\n\n"
        "La vidéo source est un rendu de synthèse : elle donne l'implantation, "
        "les volumes et le déplacement de la caméra, rien de plus. Conserve "
        "cette organisation de l'espace, et complète tout le reste pour obtenir "
        "une image filmée.\n\n"
        f"À créer, absent du rendu : {additions}.\n"
        f"Lumière : {time_of_day_fr}.\n"
        "Traitement photographique : flou de mouvement naturel, profondeur de "
        "champ, matières lisibles, reflets, grain fin de capteur.\n"
        f"{reference_clause_fr}"
    )


__all__ = [
    "CAMERA_MOVES",
    "MISSING_BY_DEFAULT",
    "SHOT_INTENTS",
    "Shot",
    "Treatment",
    "author_treatment",
    "build_completion_prompt",
    "build_treatment_prompt",
    "parse_treatment",
]
