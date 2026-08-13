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


def implicit_paths(model, raw: object, prefix: str = "") -> list[str]:  # noqa: ANN001
    """Chemins pointés des champs absents du fichier, à toute profondeur.

    Un contrôle au premier niveau ne voit qu'une section entière manquante.
    Le cas le plus probable est pourtant l'inverse : un seuil ajouté dans une
    section déjà présente. `qualification.terrain.max_new_metric` serait rempli
    par le code, dans un fichier qui paraît complet.
    """
    if not isinstance(raw, dict):
        return []

    missing: list[str] = []
    for name in type(model).model_fields:
        path = f"{prefix}{name}"
        if name not in raw:
            missing.append(path)
            continue
        value = getattr(model, name, None)
        if hasattr(type(value), "model_fields"):
            missing.extend(implicit_paths(value, raw[name], prefix=f"{path}."))
    return missing


def _read_policy(policy_path: Path | None) -> tuple[PipelinePolicy, tuple[str, ...]]:
    """Lit la politique et signale ce que le fichier ne contenait pas.

    Une politique gelée avant l'ajout d'une section se relit sans erreur : les
    champs manquants prennent les valeurs du code, tandis que le fichier
    continue d'annoncer son ancienne version. Le rapport afficherait alors des
    seuils qui ne figurent nulle part sur le disque.
    """
    if not (policy_path and policy_path.is_file()):
        return DEFAULT_POLICY, ()

    raw = json.loads(policy_path.read_text("utf-8"))
    policy = PipelinePolicy.model_validate(raw)
    filled = tuple(implicit_paths(policy, raw))
    log.info("politique chargée depuis %s (version %s)", policy_path, policy.version)
    if filled:
        log.warning(
            "politique %s : section(s) absente(s) du fichier, comblée(s) par les "
            "valeurs du code : %s",
            policy.version,
            ", ".join(filled),
        )
    return policy, filled


@dataclass(frozen=True)
class PipelineContext:
    """Ce que toute étape doit connaître, et rien de plus."""

    policy: PipelinePolicy
    profile: PropertyProfile | None = None

    #: Sections que la politique matérialisée ne contenait pas, et qui ont donc
    #: été comblées par les valeurs du code. Une politique gelée en 1.1.0 se
    #: relit sans erreur en 1.2.0 : Pydantic remplit les manques, la version
    #: écrite reste l'ancienne, et le rapport annonce des seuils que le fichier
    #: ne porte pas. Ce qui a été comblé doit donc être dit.
    policy_defaults_applied: tuple[str, ...] = ()

    def implicit_under(self, section: str) -> tuple[str, ...]:
        """Valeurs implicites dans une section, la section elle-même comprise."""
        return tuple(
            path
            for path in self.policy_defaults_applied
            if path == section or path.startswith(f"{section}.")
        )

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
        policy, filled = _read_policy(policy_path)

        profile = None
        if property_id:
            profile = cls._load_profile(property_id, profiles_dir)
            log.info(
                "profil %s chargé : %s, %s chambres",
                profile.property_id,
                profile.official_name,
                profile.room_count or "nombre inconnu de",
            )

        return cls(policy=policy, profile=profile, policy_defaults_applied=filled)

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
        policy, filled = _read_policy(policy_path)

        try:
            profile = cls._load_profile(property_id, profiles_dir)
        except ProfileNotFound as exc:
            return cls(policy=policy, policy_defaults_applied=filled), str(exc)

        return cls(policy=policy, profile=profile, policy_defaults_applied=filled), None

    @classmethod
    def for_workspace(cls, workspace) -> tuple["PipelineContext", str | None]:  # noqa: ANN001
        """Contexte d'un espace de travail, politique et profil compris.

        La politique est lue dans `00_manifest/`, jamais relativement au
        répertoire courant : sinon le même projet rendrait des résultats
        différents selon l'endroit d'où la commande est lancée.

        Le profil vient du manifeste de projet — `property_profile_id` — et non
        de l'identifiant d'hôtel. Les commandes autonomes et `run-phase1`
        chargeaient sinon deux profils différents pour le même projet.
        """
        property_id = workspace.hotel_id
        try:
            project = workspace.read_manifest()
        except FileNotFoundError:
            pass
        else:
            property_id = project.property_profile_id or project.hotel_id

        return cls.load_lenient(property_id, policy_path=workspace.policy_path)

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
