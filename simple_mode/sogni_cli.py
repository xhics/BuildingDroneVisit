"""Bridge to the official Sogni CLI for reality-constrained video generation.

The deterministic 3D render is the geometric authority. Generative passes may
improve appearance, but their output is accepted only when structural QA shows
that roof lines, silhouettes and major facade edges still agree with the source.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from .storyboard import ContinuousStoryboard, Keyframe

DEFAULT_VIDEO_MODEL = "minimax-h3-flf2v-turbo"
DEFAULT_CLIP_DURATION_S = 5
DEFAULT_CHAIN_KEYFRAMES = 6
DEFAULT_TIMEOUT_S = 600
REFERENCE_TIMEOUT_S = 2400
REFERENCE_VIDEO_MODEL = "minimax-h3-r2v"
V2V_MODEL = "ltx25-v2v"
DEFAULT_CONTROL_STRENGTH = 0.95
DEFAULT_CONTROL_MODE = "canny"
MAX_V2V_SECONDS = 15


class SogniCliError(RuntimeError):
    """The Sogni CLI call failed or its output violated the reality contract."""


def _find_sogni_agent() -> str | None:
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
            ["npm", "config", "get", "prefix"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if prefix.returncode == 0:
            for name in ("sogni-agent.cmd", "sogni-agent"):
                candidate = Path(prefix.stdout.strip()) / name
                if candidate.exists():
                    return str(candidate)
    except (subprocess.SubprocessError, OSError):
        pass
    return None


def is_available() -> bool:
    return _find_sogni_agent() is not None


def _find_ffmpeg() -> str | None:
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
    if count >= len(keyframes):
        return keyframes
    count = max(2, count)
    step = (len(keyframes) - 1) / (count - 1)
    indices = sorted({round(i * step) for i in range(count)})
    return [keyframes[i] for i in indices]


def _dedupe_adjacent_images(keyframes: list[Keyframe]) -> list[Keyframe]:
    deduped: list[Keyframe] = []
    for keyframe in keyframes:
        if deduped and deduped[-1].reference_image == keyframe.reference_image:
            continue
        deduped.append(keyframe)
    return deduped


def _pair_prompt(a: Keyframe, b: Keyframe) -> str:
    same_figure = a.maneuver_id == b.maneuver_id
    if same_figure:
        return (
            "Plan de drone continu : la caméra poursuit son mouvement de "
            f"la figure « {a.maneuver_id} », de l'image de départ vers "
            "l'image d'arrivée, sans changement brusque de vitesse ni de cap. "
            "Mouvement fluide et stabilisé, cohérent avec les deux images fournies."
        )
    return (
        f"Plan de drone continu : transition entre les figures « {a.maneuver_id} » "
        f"et « {b.maneuver_id} », de l'image de départ vers l'image d'arrivée, "
        "sans coupure ni saut. Mouvement fluide et stabilisé, cohérent avec les "
        "deux images fournies."
    )


def _run(
    command: list[str], *, timeout: int, env: dict | None = None
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise SogniCliError(
            f"sogni-agent n'a pas terminé en {timeout}s : {' '.join(command)}"
        ) from exc
    except FileNotFoundError as exc:
        raise SogniCliError("sogni-agent introuvable dans le PATH") from exc


def _write_failure_log(
    out_path: Path, command: list[str], result: subprocess.CompletedProcess
) -> Path:
    log_path = out_path.with_suffix(".log")
    log_path.write_text(
        f"COMMAND: {command}\nRETURNCODE: {result.returncode}\n\n"
        f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n",
        encoding="utf-8",
    )
    return log_path


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
        sogni_agent,
        "--video",
        "-m",
        video_model,
        "--ref",
        str(from_image),
        "--ref-end",
        str(to_image),
        "--duration",
        str(duration_s),
        "-o",
        str(out_path),
        "--json",
        prompt,
    ]
    result = _run(command, timeout=timeout, env={**os.environ})
    if result.returncode != 0 or not out_path.exists():
        log_path = _write_failure_log(out_path, command, result)
        raise SogniCliError(
            f"clip de transition échoué (code {result.returncode}) — "
            f"détail dans {log_path} :\n{(result.stderr or result.stdout)[-1500:]}"
        )
    return out_path


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
    sogni_agent = _find_sogni_agent()
    if sogni_agent is None:
        raise SogniCliError(
            "sogni-agent introuvable. Installer avec : "
            "npm install -g @sogni-ai/sogni-creative-agent-skill@latest"
        )
    images = [Path(image) for image in images]
    if not images:
        raise SogniCliError("au moins une image de référence est nécessaire")
    missing = [str(image) for image in images if not image.exists()]
    if missing:
        raise SogniCliError(f"image(s) de référence introuvable(s) : {missing}")
    if len(images) > 9:
        raise SogniCliError(
            f"{len(images)} références : le checkpoint Ref2VA en accepte 9 au plus"
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sogni_agent,
        "--video",
        "-m",
        video_model,
        "--ref",
        str(images[0]),
    ]
    for extra in images[1:]:
        command += ["-c", str(extra)]
    command += [
        "--duration",
        str(duration_s),
        "-w",
        str(width),
        "-h",
        str(height),
        "-o",
        str(out_path),
        "--json",
        prompt,
    ]
    result = _run(command, timeout=timeout, env={**os.environ})
    if result.returncode != 0 or not out_path.exists():
        log_path = _write_failure_log(out_path, command, result)
        raise SogniCliError(
            f"génération par références échouée (code {result.returncode}) — "
            f"détail dans {log_path} :\n{(result.stderr or result.stdout)[-1500:]}"
        )
    return out_path


def _ffprobe_path(ffmpeg: str) -> str:
    candidate = str(Path(ffmpeg).with_name("ffprobe.exe"))
    if Path(candidate).exists():
        return candidate
    candidate = str(Path(ffmpeg).with_name("ffprobe"))
    if Path(candidate).exists():
        return candidate
    return "ffprobe"


def _video_duration_s(path: str | Path) -> float | None:
    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        return None
    try:
        result = subprocess.run(
            [
                _ffprobe_path(ffmpeg),
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return float(result.stdout.strip()) if result.returncode == 0 else None
    except (subprocess.SubprocessError, OSError, ValueError):
        return None


def split_video(
    source: str | Path,
    out_dir: str | Path,
    *,
    chunk_seconds: int = MAX_V2V_SECONDS,
) -> list[Path]:
    """Split V2V control video into short, bounded clips.

    The former stream-copy segmentation depended on source keyframes and could
    silently produce a segment longer than the model limit. Re-encoding at a
    visually lossless CRF creates deterministic cut points while preserving the
    geometric content used by ControlNet.
    """
    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        raise SogniCliError("ffmpeg introuvable — requis pour découper la vidéo")
    if chunk_seconds <= 0 or chunk_seconds > MAX_V2V_SECONDS:
        raise SogniCliError(
            f"chunk_seconds doit être entre 1 et {MAX_V2V_SECONDS} secondes"
        )

    source = Path(source)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob(f"{source.stem}_*.mp4"):
        stale.unlink()
    pattern = out_dir / f"{source.stem}_%03d.mp4"
    command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        "-force_key_frames",
        f"expr:gte(t,n_forced*{int(chunk_seconds)})",
        "-segment_time",
        str(chunk_seconds),
        "-f",
        "segment",
        "-reset_timestamps",
        "1",
        str(pattern),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    chunks = sorted(out_dir.glob(f"{source.stem}_*.mp4"))
    if result.returncode != 0 or not chunks:
        raise SogniCliError(f"découpage échoué :\n{result.stderr[-1000:]}")
    return chunks


def _auto_appearance_reference(source_video: Path) -> Path | None:
    try:
        from .reality_qa import select_appearance_reference

        return select_appearance_reference(source_video)
    except Exception:  # noqa: BLE001 — appearance selection is optional evidence
        return None


def _audit_v2v(source_video: Path, out_path: Path) -> dict:
    try:
        from .reality_qa import assess_video_structure

        assessment = assess_video_structure(source_video, out_path)
        payload = assessment.as_dict()
    except Exception as exc:  # noqa: BLE001 — never turn missing QA into a PASS
        payload = {
            "available": False,
            "accepted": True,
            "reason": f"reality QA unavailable: {exc}",
        }
    audit_path = out_path.with_suffix(".reality.json")
    audit_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def restyle_video(
    source_video: str | Path,
    prompt: str,
    *,
    out_path: str | Path,
    control: str = DEFAULT_CONTROL_MODE,
    control_strength: float = DEFAULT_CONTROL_STRENGTH,
    appearance_image: str | Path | None = None,
    duration_s: int | None = None,
    video_model: str = V2V_MODEL,
    timeout: int = REFERENCE_TIMEOUT_S,
    auto_appearance: bool = True,
    reality_qa: bool = True,
) -> Path:
    """Retexture a deterministic render while preserving its architecture.

    When no explicit appearance image is supplied, a real hotel photo is
    selected by local-feature agreement with the current shot. After generation,
    structural QA compares the output with the source control video. A drifted
    result raises ``SogniCliError``; callers can then fall back to the trusted 3D
    render instead of delivering a visually attractive but false building.
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

    selected_appearance: Path | None
    if appearance_image is not None:
        selected_appearance = Path(appearance_image)
        if not selected_appearance.exists():
            raise SogniCliError(
                f"image d'apparence introuvable : {selected_appearance}"
            )
    elif auto_appearance:
        selected_appearance = _auto_appearance_reference(source_video)
    else:
        selected_appearance = None

    command = [
        sogni_agent,
        "--video",
        "--workflow",
        "v2v",
        "-m",
        video_model,
        "--ref-video",
        str(source_video),
        "--controlnet-name",
        control,
        "--controlnet-strength",
        str(control_strength),
    ]
    if selected_appearance is not None:
        command += ["--ref", str(selected_appearance)]
    if duration_s is None:
        duration_s = _video_duration_s(source_video)
    if duration_s:
        command += ["--duration", str(round(duration_s))]
    command += ["-o", str(out_path), "--json", prompt]

    result = _run(command, timeout=timeout, env={**os.environ})
    if result.returncode != 0 or not out_path.exists():
        log_path = _write_failure_log(out_path, command, result)
        raise SogniCliError(
            f"retexturation v2v échouée (code {result.returncode}) — "
            f"détail dans {log_path} :\n{(result.stderr or result.stdout)[-1500:]}"
        )

    if reality_qa:
        audit = _audit_v2v(source_video, out_path)
        if audit.get("available") and not audit.get("accepted"):
            raise SogniCliError(
                "retexturation refusée par Reality QA : la géométrie générée "
                f"dérive du rendu source (median={audit.get('median_score')}, "
                f"p10={audit.get('p10_score')}). Audit : "
                f"{out_path.with_suffix('.reality.json')}"
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
    progress=None,
) -> list[Path]:
    anchors = [Path(anchor) for anchor in anchors]
    if len(anchors) < 2:
        raise SogniCliError("au moins deux images d'ancrage sont nécessaires")
    if len(prompts) != len(anchors) - 1:
        raise SogniCliError(
            f"{len(prompts)} prompt(s) pour {len(anchors) - 1} saut(s) : "
            "il en faut un par saut"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips: list[Path] = []
    for index, (start, end) in enumerate(zip(anchors, anchors[1:])):
        clip = generate_transition_clip(
            start,
            end,
            prompts[index],
            out_path=out_dir / f"{prefix}_{index:02d}.mp4",
            video_model=video_model,
            duration_s=duration_s,
            timeout=timeout,
        )
        clips.append(clip)
        if progress is not None:
            progress(index + 1, len(anchors) - 1)
    return clips


def concat_videos(
    clips: list[str | Path], out_path: str | Path, *, timeout: int = 300
) -> Path:
    sogni_agent = _find_sogni_agent()
    ffmpeg = _find_ffmpeg()
    if sogni_agent is None or ffmpeg is None:
        raise SogniCliError("sogni-agent et ffmpeg sont requis pour le recollage")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sogni_agent,
        "--concat-videos",
        str(out_path),
        *[str(clip) for clip in clips],
    ]
    result = _run(
        command,
        timeout=timeout,
        env={**os.environ, "FFMPEG_PATH": ffmpeg},
    )
    if result.returncode != 0 or not out_path.exists():
        raise SogniCliError(
            f"recollage échoué (code {result.returncode}) :\n"
            f"{(result.stderr or result.stdout)[-1500:]}"
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
    sogni_agent = _find_sogni_agent()
    if sogni_agent is None:
        raise SogniCliError(
            "sogni-agent introuvable. Installer avec : "
            "npm install -g @sogni-ai/sogni-creative-agent-skill@latest"
        )
    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        raise SogniCliError(
            "ffmpeg introuvable — requis par --concat-videos."
        )
    env = {**os.environ, "FFMPEG_PATH": ffmpeg}

    keyframes = _dedupe_adjacent_images(
        _resample(storyboard.keyframes, chain_keyframes)
    )
    if len(keyframes) < 2:
        raise SogniCliError(
            "moins de 2 images de référence distinctes après déduplication"
        )
    missing = [
        keyframe.reference_image
        for keyframe in keyframes
        if not Path(keyframe.reference_image).exists()
    ]
    if missing:
        raise SogniCliError(
            f"image(s) de référence introuvable(s) sur disque : {missing}"
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = out_path.parent / f"{out_path.stem}_clips"
    work_dir.mkdir(parents=True, exist_ok=True)
    clip_paths: list[Path] = []
    for index, (a, b) in enumerate(zip(keyframes, keyframes[1:])):
        clip_path = work_dir / f"clip_{index:02d}.mp4"
        command = [
            sogni_agent,
            "--video",
            "-m",
            video_model,
            "--ref",
            str(a.reference_image),
            "--ref-end",
            str(b.reference_image),
            "--duration",
            str(clip_duration_s),
            "-o",
            str(clip_path),
            "--json",
            _pair_prompt(a, b),
        ]
        result = _run(command, timeout=timeout, env=env)
        if result.returncode != 0 or not clip_path.exists():
            log_path = _write_failure_log(clip_path, command, result)
            raise SogniCliError(
                f"clip {index} ({a.maneuver_id} -> {b.maneuver_id}) échoué "
                f"(code {result.returncode}) — détail complet dans {log_path} :\n"
                f"{(result.stderr or result.stdout)[-2000:]}"
            )
        clip_paths.append(clip_path)

    concat_command = [
        sogni_agent,
        "--concat-videos",
        str(out_path),
        *[str(path) for path in clip_paths],
    ]
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
    "DEFAULT_CONTROL_MODE",
    "DEFAULT_CONTROL_STRENGTH",
    "DEFAULT_TIMEOUT_S",
    "DEFAULT_VIDEO_MODEL",
    "MAX_V2V_SECONDS",
    "REFERENCE_TIMEOUT_S",
    "SogniCliError",
    "V2V_MODEL",
    "concat_videos",
    "generate_from_references",
    "generate_transition_chain",
    "generate_transition_clip",
    "generate_video_chain",
    "is_available",
    "restyle_video",
    "split_video",
]
