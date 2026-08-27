"""Donne au rendu 3D l'aspect d'une prise de vue, sans rien réinventer.

Les tentatives de retexturation par IA achoppent toutes sur le même mur :
pour rendre l'image photographique, le modèle doit avoir la liberté de la
réécrire — et il en profite pour redessiner le bâtiment. Fidélité et
réalisme y sont antagonistes.

Ce module prend le problème autrement. Le rendu Cesium est déjà fidèle : sa
géométrie est mesurée et ses textures sont de vraies photographies aériennes.
Ce qui trahit la synthèse, ce n'est pas son contenu mais son **rendu** :

- **aucun flou de mouvement** — chaque image est parfaitement nette, alors
  qu'une caméra réelle expose pendant un cinquantième de seconde et fond le
  mouvement d'une image à l'autre (la « règle des 180° ») ;
- **une colorimétrie plate** — pas de courbe, pas de dominante, contraste
  linéaire ;
- **aucun grain** ni vignettage, que tout capteur produit.

Ces trois manques se corrigent en post-traitement, sans modèle génératif et
sans toucher à un pixel de géométrie. Le résultat reste **exactement** le
bâtiment filmé.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class FilmizeError(RuntimeError):
    """Le post-traitement a échoué."""


#: Angle d'obturation, en degrés. 180° est la convention cinéma : le temps de
#: pose vaut la moitié de la durée d'une image. Au-delà, le filé devient
#: pâteux ; en deçà, l'image redevient saccadée.
DEFAULT_SHUTTER_DEG = 180.0

#: Facteur d'interpolation avant le fondu. Fondre des images consécutives
#: d'un rendu saccadé donne un dédoublement, pas un flou : on interpole
#: d'abord des images intermédiaires, puis on les fond.
DEFAULT_INTERPOLATION = 4


def _find_ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    import os

    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        packages = Path(local_appdata) / "Microsoft" / "WinGet" / "Packages"
        if packages.exists():
            matches = sorted(packages.glob("Gyan.FFmpeg*/**/bin/ffmpeg.exe"))
            if matches:
                return str(matches[0])
    return None


def build_filter(
    *,
    fps: int = 24,
    shutter_deg: float = DEFAULT_SHUTTER_DEG,
    interpolation: int = DEFAULT_INTERPOLATION,
    warmth: float = 0.04,
    contrast: float = 1.04,
    saturation: float = 1.08,
    grain: float = 4.0,
    vignette: bool = True,
) -> str:
    """Chaîne de filtres ffmpeg reproduisant les défauts d'une vraie caméra.

    L'ordre compte : on interpole d'abord pour avoir de quoi fondre, on fond
    pour créer le flou, puis on étalonne et on ajoute le grain — un grain
    posé avant le fondu serait lissé par celui-ci.
    """
    stages = []

    if interpolation > 1:
        # Images intermédiaires par estimation de mouvement : sans elles, le
        # fondu superpose des positions distinctes et dédouble l'image.
        stages.append(
            f"minterpolate=fps={fps * interpolation}:mi_mode=mci"
            ":mc_mode=aobmc:vsbmc=1"
        )

    # Nombre d'images à fondre = fraction d'image pendant laquelle l'obturateur
    # est ouvert. À 180° et 4× d'interpolation : 4 × 0,5 = 2 images. Fondre
    # tout l'intervalle interpolé donnerait un filé six fois trop long — c'est
    # l'erreur du premier essai, où les arbres du premier plan bavaient.
    blur_frames = max(1, round(interpolation * (shutter_deg / 360.0)))
    if blur_frames > 1:
        stages.append(f"tmix=frames={blur_frames}")
    if interpolation > 1:
        stages.append(f"fps={fps}")

    # Étalonnage : une légère dominante chaude dans les hautes lumières et un
    # contraste en S, comme une courbe de laboratoire.
    if warmth:
        stages.append(
            f"curves=r='0/0 0.5/{0.5 + warmth} 1/1':b='0/0 0.5/{0.5 - warmth} 1/1'"
        )
    stages.append(f"eq=contrast={contrast}:saturation={saturation}")

    if vignette:
        stages.append("vignette=PI/5")
    if grain:
        # Bruit temporel : un grain figé se lirait comme une salissure d'objectif.
        stages.append(f"noise=alls={grain}:allf=t")

    return ",".join(stages)


def filmize(
    source: str | Path,
    out_path: str | Path,
    *,
    fps: int = 24,
    shutter_deg: float = DEFAULT_SHUTTER_DEG,
    interpolation: int = DEFAULT_INTERPOLATION,
    crf: int = 18,
    timeout: int = 3600,
    **grading,
) -> Path:
    """Applique le traitement et écrit la vidéo résultante.

    Ne fait appel à aucun modèle : la géométrie, les matériaux et le cadrage
    du rendu sont conservés au pixel près.
    """
    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        raise FilmizeError(
            "ffmpeg introuvable — installer avec : winget install --id Gyan.FFmpeg"
        )

    source = Path(source)
    if not source.exists():
        raise FilmizeError(f"vidéo source introuvable : {source}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vf = build_filter(
        fps=fps, shutter_deg=shutter_deg, interpolation=interpolation, **grading
    )
    command = [
        ffmpeg, "-y", "-v", "error", "-i", str(source),
        "-vf", vf,
        "-c:v", "libx264", "-crf", str(crf), "-preset", "slow", "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode != 0 or not out_path.exists():
        raise FilmizeError(f"post-traitement échoué :\n{result.stderr[-1500:]}")
    return out_path


__all__ = [
    "DEFAULT_INTERPOLATION",
    "DEFAULT_SHUTTER_DEG",
    "FilmizeError",
    "build_filter",
    "filmize",
]
