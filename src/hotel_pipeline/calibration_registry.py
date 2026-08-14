"""Registre des calibrations, externe à la politique (portabilité).

`calibrated_on_sites=1` donne un nombre, et rien d'autre : ni quel site a
servi, ni sur quel corpus, ni par quelle méthode. Un lecteur de rapport ne peut
donc pas juger si la calibration s'applique à son cas.

Le registre vit **hors de la politique**, délibérément. L'y ajouter aurait
gonflé son dump, donc changé `policy_digest`, donc périmé une vingtaine de
rapports publiés — pour une information qui ne décide d'aucun seuil. La
politique garde `calibration_id` ; le registre dit ce que cet identifiant
recouvre, et l'attachement aux sites redevient vérifiable.

```text
politique   « ce seuil vient de la campagne X »
registre    « la campagne X, c'est ces sites, ces corpus, cette méthode »
```
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .logging import get_logger

log = get_logger("calibration-registry")

#: Emplacement par défaut, versionné avec le dépôt : une campagne est un fait
#: partagé entre projets, non un artefact d'un espace de travail.
DEFAULT_REGISTRY = Path("calibrations/registry.json")


class CalibrationEntry(BaseModel):
    """Une campagne de calibration, et ce qui la fonde."""

    model_config = ConfigDict(extra="forbid")

    calibration_id: str = Field(min_length=1)

    #: Sites ayant servi. Le nombre déclaré à la politique doit valoir la
    #: longueur de cette liste : deux comptes qui divergent, c'est un compte
    #: faux quelque part.
    site_ids: list[str] = Field(min_length=1)

    #: Empreintes des corpus mesurés, par site. Sans elles, « 36 images » est
    #: une affirmation invérifiable.
    corpus_digests: dict[str, str] = Field(default_factory=dict)

    method: str = Field(min_length=1)
    calibrated_on: date
    version: str = Field(min_length=1)

    notes: str | None = None

    @model_validator(mode="after")
    def _every_site_declares_its_corpus(self) -> "CalibrationEntry":
        missing = [site for site in self.site_ids if site not in self.corpus_digests]
        if missing:
            raise ValueError(
                f"campagne {self.calibration_id!r} : sites sans empreinte de "
                f"corpus {missing} — un site cité sans corpus ne prouve rien"
            )
        extra = sorted(set(self.corpus_digests) - set(self.site_ids))
        if extra:
            raise ValueError(
                f"campagne {self.calibration_id!r} : corpus déclarés pour des "
                f"sites absents de la campagne {extra}"
            )
        return self

    @property
    def site_count(self) -> int:
        return len(self.site_ids)


class CalibrationRegistry(BaseModel):
    """Toutes les campagnes connues, indexées par identifiant."""

    model_config = ConfigDict(extra="forbid")

    version: str = "1.0.0"
    entries: list[CalibrationEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _identifiers_are_unique(self) -> "CalibrationRegistry":
        seen = [entry.calibration_id for entry in self.entries]
        if len(set(seen)) != len(seen):
            raise ValueError("identifiants de campagne dupliqués")
        return self

    def get(self, calibration_id: str) -> CalibrationEntry | None:
        return next(
            (e for e in self.entries if e.calibration_id == calibration_id), None
        )


def load(path: Path | None = None) -> CalibrationRegistry:
    """Registre sur le disque, ou registre vide s'il n'existe pas encore.

    Un registre absent n'est pas une erreur : il signifie qu'aucune campagne
    n'est déclarée, ce qui est l'état d'un dépôt neuf. C'est *citer* une
    campagne introuvable qui en est une.
    """
    target = path or DEFAULT_REGISTRY
    if not target.is_file():
        return CalibrationRegistry()
    return CalibrationRegistry.model_validate_json(target.read_text("utf-8"))


def check(policy, registry: CalibrationRegistry) -> list[str]:  # noqa: ANN001
    """Les campagnes citées par la politique sont-elles vérifiables ?

    Trois désaccords possibles, et chacun compte : une campagne citée mais
    absente du registre, un nombre de sites qui diverge, ou une campagne
    déclarée sans qu'aucun site ne la porte.
    """
    from .schemas.policy import Calibrated

    problems: list[str] = []
    for name in ("model", "terrain", "qualification"):
        section = getattr(policy, name)
        if not isinstance(section, Calibrated) or not section.names_a_campaign:
            continue

        entry = registry.get(section.calibration_id)
        if entry is None:
            problems.append(
                f"{name}.calibration_id = {section.calibration_id!r} : campagne "
                "absente du registre — l'attachement aux sites est invérifiable"
            )
            continue
        if entry.site_count != section.calibrated_on_sites:
            problems.append(
                f"{name} : la politique déclare {section.calibrated_on_sites} "
                f"site(s), le registre en cite {entry.site_count} "
                f"({entry.site_ids})"
            )
    return problems


def describe(policy, registry: CalibrationRegistry) -> dict:  # noqa: ANN001
    """Ce qu'un rapport peut dire des calibrations qu'il applique."""
    from .schemas.policy import Calibrated

    described: dict[str, dict] = {}
    for name in ("model", "terrain", "qualification"):
        section = getattr(policy, name)
        if not isinstance(section, Calibrated):
            continue
        entry = registry.get(section.calibration_id)
        described[name] = {
            "calibration_id": section.calibration_id,
            "declared_sites": section.calibrated_on_sites,
            "is_calibrated": section.is_calibrated,
            "registry": (
                {
                    "site_ids": entry.site_ids,
                    "method": entry.method,
                    "calibrated_on": entry.calibrated_on.isoformat(),
                    "version": entry.version,
                    "corpus_digests": entry.corpus_digests,
                }
                if entry
                else None
            ),
        }
    return described


def write(registry: CalibrationRegistry, path: Path | None = None) -> Path:
    target = path or DEFAULT_REGISTRY
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            json.loads(registry.model_dump_json()), ensure_ascii=False, indent=2
        )
        + "\n",
        "utf-8",
    )
    log.info("registre de calibration écrit : %s", target)
    return target
