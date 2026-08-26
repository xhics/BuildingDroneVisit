"""Conception du plan de vol par un modèle OpenAI, dessiné ensuite précisément.

Le modèle ne propose que des **paramètres** (rayons, altitudes, virages,
intentions) : la géométrie réelle du tracé — chaque point de la trajectoire —
reste calculée localement par les mêmes fonctions que le mode par défaut
(``maneuvers._circle`` / ``_spiral`` / ``_line``), jamais par le modèle. Le
dessin final garde donc l'exactitude pixel du mode par défaut ; seuls le
choix des figures et leur mise en récit deviennent adaptatifs par adresse.
"""

from __future__ import annotations

import json
from pathlib import Path

from openai import OpenAI

from .maneuvers import Maneuver, _circle, _line, _spiral

DEFAULT_MODEL = "gpt-4o-mini"

_SYSTEM_PROMPT = (
    "Tu conçois un plan de vol de drone pour une courte vidéo promotionnelle "
    "d'un bâtiment, en tournage immobilier/hôtelier. Le drone évolue autour "
    "d'un point central (l'adresse), dans un repère local en mètres "
    "est/nord, altitude en mètres au-dessus du sol. Réponds uniquement en "
    "JSON valide, dans la structure exacte demandée — aucun champ en plus, "
    "aucun texte hors JSON."
)

_SCHEMA_HINT = """Réponds avec un objet JSON de cette forme exacte :
{
  "maneuvers": [
    {
      "id": "identifiant_court_snake_case",
      "name_fr": "Nom de la figure",
      "kind": "circle",
      "clockwise": true,
      "radius_m": 55.0,
      "altitude_m": 45.0,
      "turns": 1.0,
      "purpose_fr": "rôle de la figure dans le montage final",
      "skill_fr": "technique de pilotage que cette figure exige, et pourquoi"
    }
  ]
}

Trois figures ("kind") sont possibles :
- "circle" : champs radius_m, altitude_m, turns
- "spiral" : champs radius_start_m, radius_end_m, altitude_start_m, altitude_end_m, turns
- "line"   : champs start [east_m, north_m, altitude_m], end [east_m, north_m, altitude_m]

Chaque figure porte aussi id, name_fr, clockwise, purpose_fr, skill_fr.
Propose 3 à 5 figures, dans l'ordre du montage vidéo (reconnaissance,
approche, figure principale, passage final...). Varie les rayons (entre 8 et
70 m) et altitudes (entre 5 et 50 m) selon le rôle de chaque figure. N'ajoute
aucun champ hors de cette structure et ne renvoie rien d'autre que le JSON."""

_PALETTE: list[tuple[int, int, int]] = [
    (255, 159, 10),  # orange
    (64, 156, 255),  # bleu
    (255, 62, 87),  # rouge cramoisi
    (70, 214, 140),  # vert
    (191, 90, 242),  # violet
]


class AIPlanError(RuntimeError):
    """Le modèle n'a pas renvoyé un plan exploitable."""


def generate_plan(
    address: str,
    lat: float,
    lon: float,
    api_key: str,
    *,
    model: str = DEFAULT_MODEL,
) -> list[Maneuver]:
    """Demande à OpenAI de concevoir le plan de vol, puis le convertit en géométrie exacte."""
    client = OpenAI(api_key=api_key)
    user_prompt = f"Bâtiment à l'adresse : {address} ({lat:.5f}, {lon:.5f}).\n\n{_SCHEMA_HINT}"

    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as exc:  # noqa: BLE001 — remonté tel quel à l'appelant
        raise AIPlanError(f"appel OpenAI échoué : {exc}") from exc

    content = response.choices[0].message.content or ""
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise AIPlanError(f"réponse non-JSON du modèle : {exc}") from exc

    return plan_from_json(data)


def plan_from_json(data: dict) -> list[Maneuver]:
    """Convertit un objet JSON conforme au schéma en figures géométriques exactes.

    Factorisé hors de :func:`generate_plan` pour être réutilisable avec un
    JSON qui ne vient pas d'un appel réseau — voir :func:`load_plan_fixture`.
    """
    raw_maneuvers = data.get("maneuvers")
    if not raw_maneuvers:
        raise AIPlanError("le plan ne propose aucune figure de vol")

    maneuvers = []
    for i, raw in enumerate(raw_maneuvers):
        try:
            maneuvers.append(_build_maneuver(raw, color=_PALETTE[i % len(_PALETTE)]))
        except (KeyError, TypeError, ValueError) as exc:
            raise AIPlanError(f"figure {i} invalide dans le plan : {exc}") from exc
    return maneuvers


def load_plan_fixture(path: str | Path) -> list[Maneuver]:
    """Charge un plan JSON local (même schéma que la réponse OpenAI).

    Sert à tester `--ai-plan` sans clé API ni appel réseau : le JSON peut
    être écrit à la main ou capturé depuis une vraie réponse du modèle.
    """
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AIPlanError(f"fixture JSON invalide ({path}) : {exc}") from exc
    return plan_from_json(data)


def _build_maneuver(raw: dict, *, color: tuple[int, int, int]) -> Maneuver:
    kind = raw["kind"]
    clockwise = bool(raw.get("clockwise", True))
    turns = float(raw.get("turns", 1.0))

    if kind == "circle":
        waypoints = _circle(
            radius_m=float(raw["radius_m"]),
            altitude_m=float(raw["altitude_m"]),
            turns=turns,
            clockwise=clockwise,
        )
    elif kind == "spiral":
        waypoints = _spiral(
            r0=float(raw["radius_start_m"]),
            r1=float(raw["radius_end_m"]),
            a0=float(raw["altitude_start_m"]),
            a1=float(raw["altitude_end_m"]),
            turns=turns,
            clockwise=clockwise,
        )
    elif kind == "line":
        start = raw["start"]
        end = raw["end"]
        waypoints = _line(
            (float(start[0]), float(start[1]), float(start[2])),
            (float(end[0]), float(end[1]), float(end[2])),
        )
    else:
        raise ValueError(f"type de figure inconnu : {kind!r}")

    return Maneuver(
        id=str(raw.get("id", kind)),
        name_fr=str(raw.get("name_fr", kind)),
        color=color,
        waypoints=waypoints,
        skill_fr=str(raw.get("skill_fr", "")),
        purpose_fr=str(raw.get("purpose_fr", "")),
    )


__all__ = ["AIPlanError", "DEFAULT_MODEL", "generate_plan"]
