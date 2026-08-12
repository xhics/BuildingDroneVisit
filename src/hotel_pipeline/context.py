"""Contexte d'exécution — politique et profil, chargés une fois (Lot 1B).

Sans ce point de passage unique, chaque module retombe sur ses constantes :
la politique existe, le profil existe, et le pipeline n'en utilise ni l'un ni
l'autre. C'est exactement ce qui s'était produit — `resolve()` était appelé
sans profil, donc avec l'emprise de secours au lieu de celle dérivée des
116 chambres et des trois étages.

Le contexte porte aussi la provenance, pour qu'aucun rapport ne puisse être
écrit sans dire avec quels paramètres il a été produit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .logging import get_logger
from .provenance import provenance
from .schemas import DEFAULT_POLICY, PipelinePolicy, PropertyProfile

log = get_logger("context")

#: Répertoire des profils d'établissement, surchargeable par
#: ``HOTEL_PIPELINE_PROFILES``.
DEFAULT_PROFILES_DIR = Path("profiles")

#: Emplacement d'une politique non par défaut, propre à un projet.
POLICY_FILENAME = "pipeline_policy.json"


class ProfileNotFound(FileNotFoundError):
    pass


@dataclass(frozen=True)
class PipelineContext:
    """Ce que toute étape doit connaître, et rien de plus."""

    policy: PipelinePolicy
    profile: PropertyProfile | None = None

    @property
    def provenance(self) -> dict[str, str]:
        return provenance(self.policy, self.profile)

    # -- construction ----------------------------------------------------

    @classmethod
    def load(
        cls,
        property_id: str | None = None,
        profiles_dir: Path | None = None,
        policy_path: Path | None = None,
    ) -> "PipelineContext":
        """Charge la politique et, si demandé, le profil d'un établissement.

        Un profil absent est une erreur explicite : le pipeline tournerait
        sinon avec des valeurs de secours, silencieusement.
        """
        policy = DEFAULT_POLICY
        if policy_path and policy_path.is_file():
            policy = PipelinePolicy.model_validate_json(policy_path.read_text("utf-8"))
            log.info("politique chargée depuis %s (version %s)", policy_path, policy.version)

        profile = None
        if property_id:
            profile = cls._load_profile(property_id, profiles_dir)
            log.info(
                "profil %s chargé : %s, %s chambres",
                profile.property_id,
                profile.official_name,
                profile.room_count or "nombre inconnu de",
            )

        return cls(policy=policy, profile=profile)

    @staticmethod
    def _load_profile(property_id: str, profiles_dir: Path | None) -> PropertyProfile:
        import os

        directory = profiles_dir or Path(
            os.environ.get("HOTEL_PIPELINE_PROFILES", str(DEFAULT_PROFILES_DIR))
        )
        path = directory / f"{property_id}.json"
        if not path.is_file():
            raise ProfileNotFound(
                f"profil introuvable : {path}. Créez-le, ou lancez la commande "
                f"sans profil en acceptant les valeurs de secours."
            )
        return PropertyProfile.model_validate_json(path.read_text("utf-8"))

    @classmethod
    def load_lenient(
        cls,
        property_id: str,
        profiles_dir: Path | None = None,
        policy_path: Path | None = None,
    ) -> tuple["PipelineContext", str | None]:
        """Charge la politique, et le profil s'il existe.

        Retourne le contexte et un avertissement éventuel. Un profil manquant
        ne doit **pas** faire retomber la politique sur ses valeurs par défaut :
        c'était le cas, et une politique posée sur le disque était ignorée dès
        qu'aucun profil ne l'accompagnait.
        """
        policy = DEFAULT_POLICY
        if policy_path and policy_path.is_file():
            policy = PipelinePolicy.model_validate_json(policy_path.read_text("utf-8"))
            log.info("politique chargée depuis %s (version %s)", policy_path, policy.version)

        try:
            profile = cls._load_profile(property_id, profiles_dir)
        except ProfileNotFound as exc:
            return cls(policy=policy), str(exc)

        return cls(policy=policy, profile=profile), None

    @classmethod
    def for_workspace(cls, workspace) -> tuple["PipelineContext", str | None]:  # noqa: ANN001
        """Contexte d'un espace de travail, politique comprise.

        La politique est lue dans `00_manifest/`, jamais relativement au
        répertoire courant : sinon le même projet rendrait des résultats
        différents selon l'endroit d'où la commande est lancée.
        """
        return cls.load_lenient(workspace.hotel_id, policy_path=workspace.policy_path)

    @classmethod
    def default(cls) -> "PipelineContext":
        """Contexte sans profil, réservé aux traitements hors établissement."""
        return cls(policy=DEFAULT_POLICY)

    # -- raccourcis utiles ------------------------------------------------

    def identity_terms(self) -> list[str]:
        return self.profile.identity_terms() if self.profile else []

    def excluded_terms(self) -> list[str]:
        return self.profile.excluded_terms() if self.profile else []

    def ocr_languages(self) -> list[str]:
        return self.profile.ocr_languages if self.profile else ["fr", "en"]


def write_context_snapshot(path: Path, context: PipelineContext) -> None:
    """Consigne politique et profil effectifs à côté des résultats."""
    payload = {
        "provenance": context.provenance,
        "policy": json.loads(context.policy.model_dump_json()),
        "profile": (
            json.loads(context.profile.model_dump_json()) if context.profile else None
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
