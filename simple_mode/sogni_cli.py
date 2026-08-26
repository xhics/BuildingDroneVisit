"""Pont vers la CLI officielle Sogni (`sogni-agent`) pour la génération vidéo.

**Chaîne de clips, pas un appel unique.** Le compte Sogni utilisé ici a un
forfait Unlimited qui ne couvre ni Seedance ni HappyHorse — l'approche à
appel unique multi-références (`seedance2-5`, voir git history de ce
module) coûterait donc à chaque essai. On découpe plutôt le trajet en
plusieurs clips **MiniMax H3** premier-plan/dernier-plan (``--ref``/
``--ref-end``), chacun démarrant exactement où le précédent finit — même
logique de reprise que ``maneuvers.chain_maneuvers`` au niveau de la
géométrie, appliquée ici au niveau du rendu vidéo — puis recollés en une
seule vidéo avec ``--concat-videos``. C'est le patron documenté par Sogni
eux-mêmes pour ce cas (voir ``references/loop-maker.md`` du dépôt
``sogni-creative-agent-skill``, qui décrit exactement « un clip par paire
d'images adjacentes, puis recollage »).

Le SDK JavaScript de Sogni gère l'upload de fichiers locaux en interne, mais
ce mécanisme n'est pas exposé par un endpoint REST documenté (voir
``sogni_client.py``). ``sogni-agent`` — le CLI officiel construit sur ce
même SDK — expose cette capacité directement via ``--ref``/``--ref-end``
(chemins locaux acceptés, upload automatique).

Installation (une fois, hors de ce module) ::

    npm install -g @sogni-ai/sogni-creative-agent-skill@latest
    winget install --id Gyan.FFmpeg   # requis par --concat-videos
    sogni-agent doctor   # vérifie SOGNI_API_KEY, ffmpeg, l'authentification

**Chaque clip déclenche une vraie génération, facturée sur le compte
Sogni.** Il n'existe aucun mode bac-à-sable — voir le docstring de
``sogni_client.py`` pour le contexte découvert en sondant l'API brute.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from .storyboard import ContinuousStoryboard, Keyframe

#: MiniMax H3, variante Turbo (4 étapes) : moins chère et plus rapide que la
#: variante standard, pour un usage par défaut raisonnable en coût.
DEFAULT_VIDEO_MODEL = "minimax-h3-flf2v-turbo"

#: Durée par clip de transition (exemples officiels : 8s en H3 standard).
#: Volontairement plus courte par défaut pour limiter le coût par essai —
#: ajuster via `clip_duration_s`.
DEFAULT_CLIP_DURATION_S = 5

#: Nombre de clips à générer par défaut. Avec N images-clés on obtient N-1
#: clips ; ce plafond ré-échantillonne le trajet dense du story-board
#: (jusqu'à `storyboard.MAX_KEYFRAMES` images) vers un nombre de paires
#: payantes plus raisonnable.
DEFAULT_CHAIN_KEYFRAMES = 6

DEFAULT_TIMEOUT_S = 600


class SogniCliError(RuntimeError):
    """L'appel à `sogni-agent` a échoué, ou l'outil est introuvable."""


def _find_sogni_agent() -> str | None:
    """Résout le chemin de `sogni-agent`, sans dépendre du PATH ambiant.

    `shutil.which` échoue selon le contexte de lancement (une session bash
    en arrière-plan n'hérite pas forcément du même PATH qu'une session
    interactive) alors que le binaire npm global, lui, ne bouge pas. On
    retombe donc sur l'emplacement npm global connu si `which` ne le trouve
    pas directement.
    """
    found = shutil.which("sogni-agent") or shutil.which("sogni-agent.cmd")
    if found:
        return found

    appdata = os.environ.get("APPDATA")
    if appdata:
        candidate = Path(appdata) / "npm" / "sogni-agent.cmd"
        if candidate.exists():
            return str(candidate)

    try:
        prefix = subprocess.run(
            ["npm", "config", "get", "prefix"], capture_output=True, text=True, timeout=15, check=False,
        )
        if prefix.returncode == 0:
            candidate = Path(prefix.stdout.strip()) / "sogni-agent.cmd"
            if candidate.exists():
                return str(candidate)
    except (subprocess.SubprocessError, OSError):
        pass

    return None


def is_available() -> bool:
    """`sogni-agent` est-il installé et localisable ?"""
    return _find_sogni_agent() is not None


def _find_ffmpeg() -> str | None:
    """Résout le chemin de `ffmpeg`, même absent du PATH ambiant (voir `_find_sogni_agent`)."""
    found = shutil.which("ffmpeg")
    if found:
        return found

    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        packages_dir = Path(local_appdata) / "Microsoft" / "WinGet" / "Packages"
        if packages_dir.exists():
            matches = sorted(packages_dir.glob("Gyan.FFmpeg*/**/bin/ffmpeg.exe"))
            if matches:
                return str(matches[0])

    return None


def _resample(keyframes: list[Keyframe], count: int) -> list[Keyframe]:
    """`count` images-clés régulièrement espacées parmi celles du story-board.

    Toujours la première et la dernière — le début et la fin du trajet ne
    doivent jamais être coupés par le ré-échantillonnage.
    """
    if count >= len(keyframes):
        return keyframes
    if count < 2:
        count = 2
    step = (len(keyframes) - 1) / (count - 1)
    indices = sorted({round(i * step) for i in range(count)})
    return [keyframes[i] for i in indices]


def _dedupe_adjacent_images(keyframes: list[Keyframe]) -> list[Keyframe]:
    """Retire une image-clé si elle pointe vers le même fichier que la précédente.

    La plupart des figures « aériennes » du story-board retombent sur la
    même photo satellite (pas de recadrage par position) : sans ce filtre,
    un ré-échantillonnage peut produire une paire départ=arrivée
    identique — une transition dégénérée que MiniMax H3 refuse.
    """
    deduped: list[Keyframe] = []
    for keyframe in keyframes:
        if deduped and deduped[-1].reference_image == keyframe.reference_image:
            continue
        deduped.append(keyframe)
    return deduped


def _pair_prompt(a: Keyframe, b: Keyframe) -> str:
    """Prompt de transition entre deux images-clés, en français.

    Volontairement simple et descriptif plutôt que de suivre le contrat de
    prompt en trois champs propre à MiniMax H3 (voir
    references/video-prompting.md du dépôt sogni-creative-agent-skill) — à
    affiner si la qualité des transitions le justifie.
    """
    same_figure = a.maneuver_id == b.maneuver_id
    if same_figure:
        return (
            "Plan de drone continu : la caméra poursuit son mouvement de "
            f"la figure « {a.maneuver_id} », de l'image de départ vers "
            "l'image d'arrivée, sans changement brusque de vitesse ni de "
            "cap. Mouvement fluide et stabilisé, cohérent avec les deux "
            "images fournies."
        )
    return (
        f"Plan de drone continu : transition entre les figures « {a.maneuver_id} » "
        f"et « {b.maneuver_id} », de l'image de départ vers l'image "
        "d'arrivée, sans coupure ni saut. Mouvement fluide et stabilisé, "
        "cohérent avec les deux images fournies."
    )


def _run(command: list[str], *, timeout: int, env: dict | None = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False, env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise SogniCliError(f"sogni-agent n'a pas terminé en {timeout}s : {' '.join(command)}") from exc
    except FileNotFoundError as exc:
        raise SogniCliError("sogni-agent introuvable dans le PATH") from exc


def generate_video_chain(
    storyboard: ContinuousStoryboard,
    *,
    out_path: str | Path,
    video_model: str = DEFAULT_VIDEO_MODEL,
    clip_duration_s: int = DEFAULT_CLIP_DURATION_S,
    chain_keyframes: int = DEFAULT_CHAIN_KEYFRAMES,
    timeout: int = DEFAULT_TIMEOUT_S,
) -> Path:
    """Rend un clip par paire d'images adjacentes, puis les recolle.

    Chaque clip ``i`` démarre exactement où le clip ``i-1`` finit (même
    image en dernier/premier plan) : la continuité vient de cette chaîne de
    reprises, pas d'un unique appel multi-références. Lève
    :class:`SogniCliError` si `sogni-agent`/`ffmpeg` sont absents, si une
    référence est introuvable, ou si un clip échoue — la chaîne s'arrête au
    premier échec plutôt que de continuer à facturer des clips sur un
    modèle ou des references déjà en défaut.
    """
    sogni_agent = _find_sogni_agent()
    if sogni_agent is None:
        raise SogniCliError(
            "sogni-agent introuvable. Installer avec : "
            "npm install -g @sogni-ai/sogni-creative-agent-skill@latest"
        )
    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        raise SogniCliError(
            "ffmpeg introuvable — requis par --concat-videos. "
            "Installer (ex: winget install --id Gyan.FFmpeg) puis relancer le terminal."
        )
    # Passé explicitement : le PATH hérité par ce sous-processus (notamment
    # en tâche de fond) ne contient pas toujours le dossier d'installation
    # de ffmpeg, même quand `_find_ffmpeg` l'a localisé par ailleurs.
    env = {**os.environ, "FFMPEG_PATH": ffmpeg}

    keyframes = _dedupe_adjacent_images(_resample(storyboard.keyframes, chain_keyframes))
    if len(keyframes) < 2:
        raise SogniCliError(
            "moins de 2 images de référence distinctes après déduplication — augmenter "
            "--sogni-chain-keyframes, ou varier les références du story-board (voir "
            "storyboard.py : la plupart des figures aériennes partagent la même photo satellite)"
        )

    missing = [k.reference_image for k in keyframes if not Path(k.reference_image).exists()]
    if missing:
        raise SogniCliError(f"image(s) de référence introuvable(s) sur disque : {missing}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = out_path.parent / f"{out_path.stem}_clips"
    work_dir.mkdir(parents=True, exist_ok=True)

    clip_paths: list[Path] = []
    for index, (a, b) in enumerate(zip(keyframes, keyframes[1:])):
        clip_path = work_dir / f"clip_{index:02d}.mp4"
        command = [
            sogni_agent, "--video", "-m", video_model,
            "--ref", str(a.reference_image), "--ref-end", str(b.reference_image),
            "--duration", str(clip_duration_s),
            "-o", str(clip_path), "--json",
            _pair_prompt(a, b),
        ]
        result = _run(command, timeout=timeout, env=env)
        if result.returncode != 0 or not clip_path.exists():
            log_path = work_dir / f"clip_{index:02d}.log"
            log_path.write_text(
                f"COMMAND: {command}\nRETURNCODE: {result.returncode}\n\n"
                f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n",
                encoding="utf-8",
            )
            raise SogniCliError(
                f"clip {index} ({a.maneuver_id} -> {b.maneuver_id}) échoué "
                f"(code {result.returncode}) — détail complet dans {log_path} :\n"
                f"{(result.stderr or result.stdout)[-2000:]}"
            )
        clip_paths.append(clip_path)

    concat_command = [sogni_agent, "--concat-videos", str(out_path), *[str(p) for p in clip_paths]]
    result = _run(concat_command, timeout=timeout, env=env)
    if result.returncode != 0 or not out_path.exists():
        raise SogniCliError(
            f"recollage des clips échoué (code {result.returncode}) :\n"
            f"{(result.stderr or result.stdout)[:2000]}"
        )

    return out_path


__all__ = [
    "DEFAULT_CHAIN_KEYFRAMES",
    "DEFAULT_CLIP_DURATION_S",
    "DEFAULT_TIMEOUT_S",
    "DEFAULT_VIDEO_MODEL",
    "SogniCliError",
    "generate_video_chain",
    "is_available",
]
