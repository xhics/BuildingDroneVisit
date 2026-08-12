"""Journalisation sans secrets (plan directeur §18).

Aucun secret ne doit apparaître dans les journaux. Plutôt que de compter sur
la discipline des appelants, les valeurs de secrets connues sont expurgées au
niveau du filtre : même un log accidentel de la valeur brute ressort masqué.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Iterable

REDACTED = "***"

# Tout nom de variable d'environnement contenant l'un de ces fragments est
# considéré comme portant un secret.
_SECRET_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD")

# En deçà de cette longueur, une valeur est trop courte pour être expurgée sans
# risquer de mutiler du texte ordinaire ("1", "on", "true"...).
_MIN_REDACTABLE_LEN = 8


def secret_values(environ: dict[str, str] | None = None) -> list[str]:
    """Valeurs des variables d'environnement considérées comme secrètes."""
    env = os.environ if environ is None else environ
    values = []
    for name, value in env.items():
        if not value or len(value) < _MIN_REDACTABLE_LEN:
            continue
        if any(hint in name.upper() for hint in _SECRET_HINTS):
            values.append(value)
    return values


def redact(text: str, values: Iterable[str] | None = None) -> str:
    """Remplace toute occurrence d'un secret connu par ``***``."""
    secrets = list(values) if values is not None else secret_values()
    # Les plus longues d'abord : évite qu'un secret inclus dans un autre laisse
    # un fragment résiduel en clair.
    for value in sorted(secrets, key=len, reverse=True):
        text = text.replace(value, REDACTED)
    return text


def _redact_arg(value: object, secrets: list[str]) -> object:
    """Expurge un argument de log sans altérer son type.

    Convertir tous les arguments en chaînes casserait les formats numériques
    (`%d`, `%.6f`) dès qu'un secret existe dans l'environnement — c'est-à-dire
    en production, précisément là où la journalisation compte le plus. Seules
    les chaînes peuvent porter un secret ; le reste passe intact.
    """
    if isinstance(value, str):
        return redact(value, secrets)
    return value


class SecretRedactingFilter(logging.Filter):
    """Expurge les secrets du message et de ses arguments."""

    def filter(self, record: logging.LogRecord) -> bool:
        secrets = secret_values()
        if not secrets:
            return True

        record.msg = redact(str(record.msg), secrets)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: _redact_arg(v, secrets) for k, v in record.args.items()}
            else:
                record.args = tuple(_redact_arg(a, secrets) for a in record.args)
        return True


def configure(verbose: bool = False) -> None:
    """Installe la journalisation du pipeline sur stderr."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    handler.addFilter(SecretRedactingFilter())

    root = logging.getLogger("hotel_pipeline")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"hotel_pipeline.{name}")
