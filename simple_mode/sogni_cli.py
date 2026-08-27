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

#: La génération par références (Ref2VA) est nettement plus lente que les
#: clips de transition : mesuré à 9-10 minutes par plan de 10 s, ce qui
#: heurtait le délai de 600 s et faisait échouer le 4ᵉ plan d'une série
#: après trois réussites. Large marge pour absorber la variabilité du réseau
#: de calcul.
REFERENCE_TIMEOUT_S = 2400


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


def generate_transition_clip(
    from_image: str | Path,
    to_image: str | Path,
    prompt: str,
    *,
    out_path: str | Path,
    video_model: str = DEFAULT_VIDEO_MODEL,
    duration_s: int = DEFAULT_CLIP_DURATION_S,
    timeout: int = DEFAULT_TIMEOUT_S,
) -> Path:
    """Génère un unique clip entre deux images réelles (premier/dernier plan).

    C'est l'usage où ces modèles sont fiables : les deux bornes sont des
    images existantes et proches en point de vue, le modèle n'a qu'à
    inventer le court passage entre elles. À ne pas confondre avec le
    morphing entre deux points de vue incompatibles (vue satellite verticale
    vs photo de rue), qui produit des éléments incohérents en cours de plan.

    Sert notamment à combler la traversée du bâtiment, que le rendu 3D ne
    peut pas produire faute d'intérieur modélisé — voir
    ``cesium_render.build_traverse_poses``.
    """
    sogni_agent = _find_sogni_agent()
    if sogni_agent is None:
        raise SogniCliError(
            "sogni-agent introuvable. Installer avec : "
            "npm install -g @sogni-ai/sogni-creative-agent-skill@latest"
        )
    for image in (from_image, to_image):
        if not Path(image).exists():
            raise SogniCliError(f"image de référence introuvable : {image}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sogni_agent, "--video", "-m", video_model,
        "--ref", str(from_image), "--ref-end", str(to_image),
        "--duration", str(duration_s),
        "-o", str(out_path), "--json", prompt,
    ]
    result = _run(command, timeout=timeout, env={**os.environ})
    if result.returncode != 0 or not out_path.exists():
        log_path = out_path.with_suffix(".log")
        log_path.write_text(
            f"COMMAND: {command}\nRETURNCODE: {result.returncode}\n\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n",
            encoding="utf-8",
        )
        raise SogniCliError(
            f"clip de transition échoué (code {result.returncode}) — détail dans {log_path} :\n"
            f"{(result.stderr or result.stdout)[-1500:]}"
        )
    return out_path


#: Checkpoint « référence vers vidéo » : il ne verrouille aucune image, à la
#: différence de flf2v qui épingle la première et la dernière. Il doit être
#: demandé par son nom — il n'est jamais choisi automatiquement.
REFERENCE_VIDEO_MODEL = "minimax-h3-r2v"


def generate_from_references(
    images: list[str | Path],
    prompt: str,
    *,
    out_path: str | Path,
    duration_s: int = 8,
    video_model: str = REFERENCE_VIDEO_MODEL,
    width: int = 1344,
    height: int = 768,
    timeout: int = REFERENCE_TIMEOUT_S,
) -> Path:
    """Génère un plan à partir d'un **jeu de références**, sans image imposée.

    À utiliser quand les photos doivent inspirer le plan plutôt que d'en
    constituer les extrémités : le modèle recompose, au lieu d'interpoler
    entre deux images qu'il doit restituer à l'identique.

    Le prompt doit suivre le contrat Ref2VA à six champs (voir
    ``interior_journey.build_ref2va_prompt``) ; la première image est passée
    en ``--ref``, les suivantes en ``-c`` répété, et le prompt les désigne
    par ``<Picture 1>``, ``<Picture 2>``…
    """
    sogni_agent = _find_sogni_agent()
    if sogni_agent is None:
        raise SogniCliError(
            "sogni-agent introuvable. Installer avec : "
            "npm install -g @sogni-ai/sogni-creative-agent-skill@latest"
        )
    images = [Path(i) for i in images]
    if not images:
        raise SogniCliError("au moins une image de référence est nécessaire")
    missing = [str(i) for i in images if not i.exists()]
    if missing:
        raise SogniCliError(f"image(s) de référence introuvable(s) : {missing}")
    if len(images) > 9:
        raise SogniCliError(f"{len(images)} références : le checkpoint Ref2VA en accepte 9 au plus")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    command = [sogni_agent, "--video", "-m", video_model, "--ref", str(images[0])]
    for extra in images[1:]:
        command += ["-c", str(extra)]
    command += [
        "--duration", str(duration_s), "-w", str(width), "-h", str(height),
        "-o", str(out_path), "--json", prompt,
    ]

    result = _run(command, timeout=timeout, env={**os.environ})
    if result.returncode != 0 or not out_path.exists():
        log_path = out_path.with_suffix(".log")
        log_path.write_text(
            f"COMMAND: {command}\nRETURNCODE: {result.returncode}\n\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n",
            encoding="utf-8",
        )
        raise SogniCliError(
            f"génération par références échouée (code {result.returncode}) — "
            f"détail dans {log_path} :\n{(result.stderr or result.stdout)[-1500:]}"
        )
    return out_path


#: Modèle LTX 2.5 vidéo-vers-vidéo, qui accepte le conditionnement ControlNet.
V2V_MODEL = "ltx25-v2v"

#: Force du conditionnement structurel. Sogni recommande 0.85 pour
#: ``depth``/``canny``, mais l'essai sur le Château Frontenac a montré qu'à
#: cette valeur le modèle produit *un* château, pas *ce* château. À 0.95 avec
#: ``canny``, brique, toitures et tourelles sont conservées.
DEFAULT_CONTROL_STRENGTH = 0.95

#: ``canny`` (contours) plutôt que ``depth`` (volumes) : la profondeur ne
#: transmet que la masse, si bien que le modèle réinvente le style. Les
#: contours portent le dessin des tourelles, les lignes de toiture et le
#: rythme des fenêtres — c'est-à-dire l'identité du bâtiment.
DEFAULT_CONTROL_MODE = "canny"


#: LTX v2v traite des plans courts ; au-delà le worker refuse ou tronque.
#: On découpe donc le vol rendu avant de le retexturer, puis on recolle.
MAX_V2V_SECONDS = 15


def _video_duration_s(path: str | Path) -> float | None:
    """Durée d'une vidéo, via ffprobe ; ``None`` si elle n'est pas lisible."""
    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        return None
    ffprobe = str(Path(ffmpeg).with_name("ffprobe.exe"))
    if not Path(ffprobe).exists():
        ffprobe = "ffprobe"
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30, check=False,
        )
        return float(result.stdout.strip()) if result.returncode == 0 else None
    except (subprocess.SubprocessError, OSError, ValueError):
        return None


def split_video(
    source: str | Path, out_dir: str | Path, *, chunk_seconds: int = MAX_V2V_SECONDS
) -> list[Path]:
    """Découpe une vidéo en tronçons de durée fixe, sans ré-encoder l'image.

    Le découpage se fait sur les images-clés (``-c copy``) : rapide et sans
    perte, au prix de tronçons dont la durée exacte peut varier de quelques
    dixièmes de seconde.
    """
    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        raise SogniCliError("ffmpeg introuvable — requis pour découper la vidéo")

    source = Path(source)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / f"{source.stem}_%03d.mp4"

    command = [
        ffmpeg, "-y", "-v", "error", "-i", str(source),
        "-c", "copy", "-map", "0",
        "-segment_time", str(chunk_seconds), "-f", "segment",
        "-reset_timestamps", "1", str(pattern),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    chunks = sorted(out_dir.glob(f"{source.stem}_*.mp4"))
    if result.returncode != 0 or not chunks:
        raise SogniCliError(f"découpage échoué :\n{result.stderr[-1000:]}")
    return chunks


def restyle_video(
    source_video: str | Path,
    prompt: str,
    *,
    out_path: str | Path,
    control: str = DEFAULT_CONTROL_MODE,
    control_strength: float = DEFAULT_CONTROL_STRENGTH,
    appearance_image: str | Path | None = None,
    mask_image: str | Path | None = None,
    duration_s: int | None = None,
    video_model: str = V2V_MODEL,
    timeout: int = REFERENCE_TIMEOUT_S,
) -> Path:
    """Retexture une vidéo **sans toucher à sa géométrie**.

    C'est la réponse au décrochage observé en génération libre : là où
    ``flf2v`` ne fixe que deux images et ``r2v`` aucune, le conditionnement
    ControlNet contraint **chaque** image du plan. Le modèle ne peut alors
    plus reconstruire le sujet à sa façon — il n'a la main que sur la
    matière : lumière, reflets, flou de mouvement, grain.

    ``control`` : ``depth`` (volumes), ``canny`` (contours) ou ``detailer``
    (préservation maximale, sans changement de style). ``control_strength``
    règle le compromis fidélité / liberté.
    """
    sogni_agent = _find_sogni_agent()
    if sogni_agent is None:
        raise SogniCliError(
            "sogni-agent introuvable. Installer avec : "
            "npm install -g @sogni-ai/sogni-creative-agent-skill@latest"
        )
    source_video = Path(source_video)
    if not source_video.exists():
        raise SogniCliError(f"vidéo source introuvable : {source_video}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sogni_agent, "--video", "--workflow", "v2v", "-m", video_model,
        "--ref-video", str(source_video),
        "--controlnet-name", control,
        "--controlnet-strength", str(control_strength),
    ]
    # Photo réelle du sujet. Le conditionnement structurel n'impose que la
    # forme : sans référence d'apparence, le modèle invente les matériaux et
    # produit un bâtiment du bon genre mais pas le bon. La doc LTX 2.5 décrit
    # `--ref` comme la « subject appearance » aux côtés de `--ref-video`.
    if appearance_image is not None:
        appearance_image = Path(appearance_image)
        if not appearance_image.exists():
            raise SogniCliError(f"image d'apparence introuvable : {appearance_image}")
        command += ["--ref", str(appearance_image)]

    # Masque de couverture : blanc = zone sans donnée réelle, à régénérer ;
    # noir = zone couverte par la mesure, à préserver. Voir `coverage.py`.
    if mask_image is not None:
        mask_image = Path(mask_image)
        if not mask_image.exists():
            raise SogniCliError(f"masque introuvable : {mask_image}")
        if control != "inpaint":
            raise SogniCliError(
                f"un masque n'a de sens qu'en mode inpaint, pas en '{control}'"
            )
        command += ["--mask", str(mask_image)]
    # Sans durée explicite, le worker retombe sur sa valeur par défaut et
    # tronque : mesuré à 5 s de sortie pour 13 s de source.
    if duration_s is None:
        duration_s = _video_duration_s(source_video)
    if duration_s:
        command += ["--duration", str(round(duration_s))]
    command += ["-o", str(out_path), "--json", prompt]
    result = _run(command, timeout=timeout, env={**os.environ})
    if result.returncode != 0 or not out_path.exists():
        log_path = out_path.with_suffix(".log")
        log_path.write_text(
            f"COMMAND: {command}\nRETURNCODE: {result.returncode}\n\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n",
            encoding="utf-8",
        )
        raise SogniCliError(
            f"retexturation v2v échouée (code {result.returncode}) — "
            f"détail dans {log_path} :\n{(result.stderr or result.stdout)[-1500:]}"
        )
    return out_path


def generate_transition_chain(
    anchors: list[str | Path],
    prompts: list[str],
    *,
    out_dir: str | Path,
    prefix: str = "passage",
    duration_s: int = DEFAULT_CLIP_DURATION_S,
    video_model: str = DEFAULT_VIDEO_MODEL,
    timeout: int = DEFAULT_TIMEOUT_S,
    progress=None,  # noqa: ANN001 — callable(index, total) optionnel
) -> list[Path]:
    """``N`` images d'ancrage -> ``N-1`` clips courts consécutifs.

    Un passage long confié d'un bloc au générateur dérive : plus il invente
    longtemps sans référence, plus il s'éloigne: c'est ce qui produisait un
    intérieur incohérent. En le découpant, chaque clip repart d'une image
    donnée et vise une image donnée — le modèle n'improvise jamais plus de
    quelques secondes d'affilée.

    ``prompts`` doit contenir un texte par saut (``len(anchors) - 1``).
    """
    anchors = [Path(a) for a in anchors]
    if len(anchors) < 2:
        raise SogniCliError("au moins deux images d'ancrage sont nécessaires")
    if len(prompts) != len(anchors) - 1:
        raise SogniCliError(
            f"{len(prompts)} prompt(s) pour {len(anchors) - 1} saut(s) : il en faut un par saut"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clips: list[Path] = []
    for index, (start, end) in enumerate(zip(anchors, anchors[1:])):
        clip = generate_transition_clip(
            start, end, prompts[index],
            out_path=out_dir / f"{prefix}_{index:02d}.mp4",
            video_model=video_model, duration_s=duration_s, timeout=timeout,
        )
        clips.append(clip)
        if progress is not None:
            progress(index + 1, len(anchors) - 1)
    return clips


def concat_videos(clips: list[str | Path], out_path: str | Path, *, timeout: int = 300) -> Path:
    """Recolle des clips en une seule vidéo (via `sogni-agent --concat-videos`)."""
    sogni_agent = _find_sogni_agent()
    ffmpeg = _find_ffmpeg()
    if sogni_agent is None or ffmpeg is None:
        raise SogniCliError("sogni-agent et ffmpeg sont requis pour le recollage")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sogni_agent, "--concat-videos", str(out_path), *[str(c) for c in clips]]
    result = _run(command, timeout=timeout, env={**os.environ, "FFMPEG_PATH": ffmpeg})
    if result.returncode != 0 or not out_path.exists():
        raise SogniCliError(
            f"recollage échoué (code {result.returncode}) :\n{(result.stderr or result.stdout)[-1500:]}"
        )
    return out_path


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
    "concat_videos",
    "generate_transition_chain",
    "generate_transition_clip",
    "DEFAULT_CHAIN_KEYFRAMES",
    "DEFAULT_CLIP_DURATION_S",
    "DEFAULT_TIMEOUT_S",
    "DEFAULT_VIDEO_MODEL",
    "SogniCliError",
    "generate_video_chain",
    "is_available",
]
