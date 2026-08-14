"""Lecture d'enseigne sur fichiers acquis (collecte V2, étape 4).

L'OCR vient **après** l'acquisition, jamais avant : à la découverte, aucune
image n'existe, et lire ce qui n'a pas été téléchargé n'a pas de sens.

Ce que ce module ajoute à `triage.sign_ocr`, qui savait déjà lire et comparer :
la **provenance**. `sign_text` disait ce qui avait été lu, sans dire par quel
moteur, dans quelles langues, sur quel fichier, ni quand. Une lecture sans
provenance ne se rejoue pas — et le verdict d'appartenance qu'elle fonde n'est
donc pas réfutable.

Trois refus tiennent le module :

```text
aucun fichier      un asset non acquis ne se lit pas
aucune langue      « fr, en » était le repli du pilote, pas une propriété du monde
fichier modifié    l'empreinte lue doit être celle qui a été jugée
```
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .logging import get_logger
from .schemas import PropertyMatchStatus
from .schemas.assets import Asset

log = get_logger("ocr")


class OcrRefused(RuntimeError):
    """Rien n'a été lu, et aucun asset n'a été modifié."""


@dataclass
class OcrReading:
    """Une lecture, et tout ce qu'il faut pour la rejouer.

    L'empreinte du fichier lu en fait partie : une lecture porte sur ce qui a
    été vu, et si le fichier change, la lecture ne le suit pas — même règle que
    pour une décision de revue.
    """

    asset_id: str
    text: str
    engine: str
    engine_version: str
    languages: list[str]
    read_at: datetime
    file_digest: str

    #: Termes reconnus, et verdict qui en découle. Le verdict est **dérivé**,
    #: pas saisi : le recalculer depuis le texte et le profil doit rendre le
    #: même résultat.
    status: PropertyMatchStatus = PropertyMatchStatus.UNCERTAIN
    matched_term: str | None = None

    def as_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "text": self.text,
            "engine": self.engine,
            "engine_version": self.engine_version,
            "languages": self.languages,
            "read_at": self.read_at.isoformat(),
            "file_digest": self.file_digest,
            "status": self.status.value,
            "matched_term": self.matched_term,
        }


@dataclass
class OcrReport:
    """Ce qui a été lu, ce qui ne l'a pas été, et pourquoi."""

    run_id: str = ""
    engine: str = ""
    languages: list[str] = field(default_factory=list)
    read: int = 0
    skipped: dict[str, str] = field(default_factory=dict)
    by_status: dict[str, int] = field(default_factory=dict)
    matched_terms: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "engine": self.engine,
            "languages": self.languages,
            "read": self.read,
            "skipped": self.skipped,
            "by_status": self.by_status,
            "matched_terms": self.matched_terms,
            "note": (
                "une lecture d'enseigne établit une appartenance, jamais une "
                "visibilité : lire le nom de l'établissement ne dit pas que le "
                "bâtiment est dans le cadre"
            ),
        }


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_asset(
    asset: Asset,
    reader,  # noqa: ANN001 — objet portant `.read(Path) -> str`
    languages: list[str],
    expected: list[str],
    excluded: list[str],
    engine: str = "easyocr",
    engine_version: str = "unknown",
    workspace_root: Path | None = None,
) -> OcrReading:
    """Lit l'enseigne d'un asset **acquis**, et trace ce qui l'a lue.

    Le verdict d'appartenance est dérivé du texte par `triage.evaluate`, qui
    fait déjà autorité : le refaire ici créerait deux règles pour une question.
    """
    from .triage import evaluate

    if not asset.local_path:
        raise OcrRefused(
            f"{asset.id} : aucun fichier local — un asset non acquis ne se lit pas"
        )
    if not languages:
        raise OcrRefused(
            f"{asset.id} : aucune langue d'OCR déclarée. Elles viennent du "
            "profil ; les supposer reviendrait à lire l'établissement comme "
            "s'il était ailleurs."
        )

    path = Path(asset.local_path)
    if workspace_root and not path.is_absolute():
        path = workspace_root / path
    if not path.is_file():
        raise OcrRefused(f"{asset.id} : fichier absent ({path})")

    digest = file_digest(path)
    if asset.checksum and digest != asset.checksum:
        raise OcrRefused(
            f"{asset.id} : le fichier a changé depuis son acquisition "
            f"({digest[:12]}… ≠ {asset.checksum[:12]}…) — la lecture porterait "
            "sur autre chose que ce qui a été mesuré"
        )

    text = reader.read(path)
    reading = evaluate(text, expected, excluded)

    return OcrReading(
        asset_id=asset.id, text=text, engine=engine, engine_version=engine_version,
        languages=list(languages), read_at=datetime.now(timezone.utc),
        file_digest=digest, status=reading.status, matched_term=reading.matched_term,
    )


def apply(asset: Asset, reading: OcrReading) -> Asset:
    """Reporte une lecture sur l'asset, sans écraser un verdict humain.

    Une revue humaine tranche l'appartenance ; l'OCR l'informe. Laisser une
    lecture automatique remplacer un verdict humain rejouerait le défaut déjà
    corrigé sur `review_status`.
    """
    updates = {"sign_text": (reading.text or None)}

    if asset.has_been_reviewed:
        log.info(
            "%s : lecture conservée, appartenance laissée à la revue humaine",
            asset.id,
        )
    else:
        updates["property_match_status"] = reading.status

    return asset.model_copy(update=updates)


def run(
    assets: list[Asset],
    reader,  # noqa: ANN001
    languages: list[str],
    expected: list[str],
    excluded: list[str],
    run_id: str | None = None,
    engine: str = "easyocr",
    engine_version: str = "unknown",
    workspace_root: Path | None = None,
) -> tuple[list[Asset], list[OcrReading], OcrReport]:
    """Lit tout ce qui est acquis, et dit pourquoi le reste ne l'est pas."""
    if not languages:
        raise OcrRefused(
            "aucune langue d'OCR déclarée : elles viennent du profil de "
            "l'établissement, et ne se supposent pas"
        )

    report = OcrReport(
        run_id=run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
        engine=engine, languages=list(languages),
    )
    readings: list[OcrReading] = []
    updated = list(assets)

    for index, asset in enumerate(assets):
        try:
            reading = read_asset(
                asset, reader, languages, expected, excluded,
                engine, engine_version, workspace_root,
            )
        except OcrRefused as exc:
            report.skipped[asset.id] = str(exc).split(" : ", 1)[-1]
            continue

        readings.append(reading)
        updated[index] = apply(asset, reading)
        report.read += 1
        report.by_status[reading.status.value] = (
            report.by_status.get(reading.status.value, 0) + 1
        )
        if reading.matched_term:
            report.matched_terms[reading.matched_term] = (
                report.matched_terms.get(reading.matched_term, 0) + 1
            )

    log.info(
        "OCR %s : %d lecture(s), %d ignoré(s), moteur %s (%s)",
        report.run_id, report.read, len(report.skipped), engine, engine_version,
    )
    return updated, readings, report
