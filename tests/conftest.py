"""Garde-fous communs à toute la suite.

Le §17 du plan directeur exige des tests unitaires « rapides et sans réseau ».
Le mode hors ligne est donc imposé globalement : un test qui tenterait un appel
réseau échoue bruyamment au lieu de dépendre d'un service externe.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("HOTEL_PIPELINE_OFFLINE", "1")


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Chaque test a son propre cache : aucune fuite entre tests."""
    monkeypatch.setenv("HOTEL_PIPELINE_CACHE", str(tmp_path / "cache"))
    import hotel_pipeline.providers.cache as cache_module

    monkeypatch.setattr(cache_module, "_cache", None)


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def overpass_elements(fixtures_dir) -> list[dict]:
    """Extrait Overpass représentatif du site de Boucherville.

    Contient délibérément le piège du §3 : un bâtiment d'hôtel non étiqueté,
    un voisin commercial, un stationnement contigu et un parc-o-bus.
    """
    return json.loads((fixtures_dir / "overpass_boucherville.json").read_text("utf-8"))
