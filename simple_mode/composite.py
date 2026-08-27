"""Recompose le plan généré avec le rendu fidèle, image par image.

Plutôt que de demander au moteur de respecter une zone interdite — ce qu'il
fait mal, et ce que l'outil ne permet qu'avec **un seul** masque pour tout un
plan —, on le laisse travailler librement puis on remet en place ce qu'on
possédait déjà.

Trois conséquences :

- **un masque par image**, puisque la composition se fait ici et non chez le
  fournisseur ; la caméra peut bouger sans que le masque devienne faux ;
- **coût nul**, c'est du calcul local ;
- **réglable après coup** : le dosage s'ajuste en regardant le résultat, sans
  relancer la moindre génération.

Le masque cesse d'être une consigne qu'on espère voir respectée pour devenir
une opération mécanique : la zone couverte par la mesure est réinjectée,
quoi qu'ait fait le générateur.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


class CompositeError(RuntimeError):
    """La composition a échoué."""


@dataclass
class CompositeResult:
    path: Path
    frames: int
    #: Part moyenne de l'image provenant du rendu fidèle.
    preserved_ratio: float

    def describe_fr(self) -> str:
        return (
            f"{self.frames} images, {self.preserved_ratio * 100:.0f}% de pixels "
            f"repris du rendu fidèle"
        )


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


def _fit_to(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Ramène ``image`` au cadrage de ``size``, sans la déformer.

    Le générateur impose ses propres dimensions — LTX exige des multiples de
    64 et recadre en silence : 1280×720 est ressorti en 1920×1024. Un
    redimensionnement direct étirerait alors l'image et décalerait chaque
    élément par rapport au rendu fidèle. On recadre donc au centre pour
    retrouver le même champ avant de mettre à l'échelle.
    """
    target_ratio = size[0] / size[1]
    ratio = image.width / image.height
    if abs(ratio - target_ratio) > 1e-3:
        if ratio > target_ratio:
            width = int(image.height * target_ratio)
            left = (image.width - width) // 2
            image = image.crop((left, 0, left + width, image.height))
        else:
            height = int(image.width / target_ratio)
            top = (image.height - height) // 2
            image = image.crop((0, top, image.width, top + height))
    return image.resize(size, Image.LANCZOS)


def _extract(video: str | Path, out_dir: Path, fps: int) -> list[Path]:
    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        raise CompositeError("ffmpeg introuvable")
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-y", "-v", "error", "-i", str(video),
        "-vf", f"fps={fps}", str(out_dir / "f_%05d.png"),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    frames = sorted(out_dir.glob("f_*.png"))
    if result.returncode != 0 or not frames:
        raise CompositeError(f"extraction échouée :\n{result.stderr[-800:]}")
    return frames


def composite_videos(
    faithful: str | Path,
    generated: str | Path,
    masks: list[str | Path],
    out_path: str | Path,
    *,
    fps: int = 24,
    strength: float = 1.0,
    work_dir: str | Path | None = None,
    keep_work: bool = False,
) -> CompositeResult:
    """Recompose ``generated`` sur ``faithful`` selon ``masks``.

    Convention du masque : **blanc = garder le généré**, noir = reprendre le
    rendu fidèle. C'est celle de ``coverage.build_mask``, où le blanc marque
    l'absence de donnée réelle.

    ``masks`` peut contenir une seule image — appliquée à tout le plan — ou
    autant d'images que de frames. ``strength`` atténue globalement l'apport
    du généré : à 0,8 le rendu fidèle transparaît partout, ce qui adoucit les
    écarts de teinte entre les deux sources.
    """
    faithful, generated = Path(faithful), Path(generated)
    for path in (faithful, generated):
        if not path.exists():
            raise CompositeError(f"vidéo introuvable : {path}")

    work = Path(work_dir) if work_dir else Path(out_path).parent / "_composite"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    faithful_frames = _extract(faithful, work / "faithful", fps)
    generated_frames = _extract(generated, work / "generated", fps)
    count = min(len(faithful_frames), len(generated_frames))
    if count == 0:
        raise CompositeError("aucune image commune entre les deux vidéos")

    mask_paths = [Path(m) for m in masks]
    if not mask_paths:
        raise CompositeError("au moins un masque est nécessaire")

    out_frames = work / "out"
    out_frames.mkdir()
    total_preserved = 0.0

    for index in range(count):
        base = Image.open(faithful_frames[index]).convert("RGB")
        # Appariement **proportionnel** et non un pour un : le générateur ne
        # rend pas exactement le même nombre d'images (233 pour 240 mesuré),
        # et un appariement par indice accumulerait un décalage croissant.
        source_index = min(
            len(generated_frames) - 1,
            round(index * (len(generated_frames) - 1) / max(1, count - 1)),
        )
        top = _fit_to(Image.open(generated_frames[source_index]).convert("RGB"), base.size)
        # Un masque unique s'applique à tout le plan ; sinon, chacun le sien.
        mask_path = mask_paths[min(index, len(mask_paths) - 1)]
        mask = Image.open(mask_path).convert("L").resize(base.size)

        if strength < 1.0:
            mask = mask.point(lambda v: int(v * strength))

        merged = Image.composite(top, base, mask)
        merged.save(out_frames / f"f_{index:05d}.png")

        histogram = mask.histogram()
        pixels = sum(histogram)
        total_preserved += 1.0 - sum(
            i * c for i, c in enumerate(histogram)
        ) / (255 * pixels)

    ffmpeg = _find_ffmpeg()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-y", "-v", "error", "-framerate", str(fps),
        "-i", str(out_frames / "f_%05d.png"),
        "-c:v", "libx264", "-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not out_path.exists():
        raise CompositeError(f"assemblage échoué :\n{result.stderr[-800:]}")

    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)

    return CompositeResult(
        path=out_path, frames=count, preserved_ratio=total_preserved / count
    )


__all__ = ["CompositeError", "CompositeResult", "composite_videos"]
