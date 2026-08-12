"""Configuration, secrets et santé des fournisseurs (plan directeur §6).

Les secrets sont injectés à l'exécution et ne sont jamais committés.
L'absence d'une source optionnelle ne provoque pas d'échec global : elle réduit
la couverture disponible et influencera le Preflight et le Router. Une source
obligatoire indisponible produit une erreur explicite et actionnable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Provider:
    """Un fournisseur de données et la variable d'environnement qui l'active."""

    name: str
    env_var: str | None  # None = aucune clé requise
    required: bool
    note: str = ""


PROVIDERS: tuple[Provider, ...] = (
    Provider("Google Places", "GOOGLE_PLACES_API_KEY", required=False),
    Provider("Street View", "GOOGLE_MAPS_API_KEY", required=False),
    Provider("Mapillary", "MAPILLARY_TOKEN", required=False),
    Provider("Vision", "VISION_API_KEY", required=False),
    Provider("OSM", None, required=True, note="aucune clé requise"),
    Provider("Overture", None, required=True, note="aucune clé requise"),
    Provider("RunPod", "RUNPOD_API_KEY", required=False, note="VM GPU"),
)


@dataclass(frozen=True)
class ProviderStatus:
    provider: Provider
    configured: bool

    @property
    def blocking(self) -> bool:
        """Une source obligatoire non configurée bloque l'exécution."""
        return self.provider.required and not self.configured

    @property
    def label(self) -> str:
        if self.provider.env_var is None:
            return self.provider.note or "aucune clé requise"
        if self.configured:
            return "configured"
        return f"missing {self.provider.env_var}"


def load_env(dotenv_path: Path | None = None) -> None:
    """Charge le .env sans écraser un environnement déjà injecté.

    L'environnement réel prime : sur la VM GPU, les secrets arrivent par
    variables d'environnement, pas par fichier.
    """
    load_dotenv(dotenv_path=dotenv_path, override=False)


def check_providers() -> list[ProviderStatus]:
    """État de configuration de chaque fournisseur."""
    statuses = []
    for provider in PROVIDERS:
        if provider.env_var is None:
            configured = True
        else:
            configured = bool(os.environ.get(provider.env_var, "").strip())
        statuses.append(ProviderStatus(provider=provider, configured=configured))
    return statuses


def secret(env_var: str) -> str:
    """Lit un secret obligatoire, avec une erreur actionnable s'il manque."""
    value = os.environ.get(env_var, "").strip()
    if not value:
        raise RuntimeError(
            f"{env_var} est absent. Ajoutez-le au fichier .env (voir .env.example) "
            f"ou injectez-le dans l'environnement."
        )
    return value
