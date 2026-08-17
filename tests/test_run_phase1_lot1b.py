"""L'orchestrateur traverse le Lot 1B par les contrats existants (P2).

`run-phase1` ne doit pas devenir une troisième variante de collecte ou de
péremption : ces tests vérifient qu'il **délègue**, et qu'il s'arrête devant
les gates plutôt que de publier une route ou une campagne fabriquées.
"""

from __future__ import annotations

import pytest

from hotel_pipeline.steps import (
    STEP_ORDER,
    STEPS,
    StepBlocked,
    run_step,
)
from hotel_pipeline.workspace import Workspace


def test_lot1b_runs_after_collect_and_before_preflight() -> None:
    assert STEP_ORDER.index("collect") < STEP_ORDER.index("lot1b")
    assert STEP_ORDER.index("lot1b") < STEP_ORDER.index("preflight")
    assert STEPS["lot1b"].lot == "Lot 1B"


def _workspace(tmp_path):
    workspace = Workspace("hotel-test", root=tmp_path)
    workspace.create()
    return workspace


def test_lot1b_delegates_to_the_existing_contracts(tmp_path, monkeypatch) -> None:
    """L'étape appelle les mêmes fonctions que les sous-commandes modernes."""
    workspace = _workspace(tmp_path)
    workspace.write_json("10_validation/router_decision_x.json", {"path": "path_d"})
    called: list[str] = []

    class _Registry:
        closed_families = 15
        required_families = 15
        closure_complete = True

        @staticmethod
        def model_validate_json(_):
            return _Registry()

    def _registry(ws):
        called.append("sources")
        path = ws.write_json("00_manifest/source_registry.json", {})
        return path

    monkeypatch.setattr("hotel_pipeline.source_registry.build", _registry)
    monkeypatch.setattr("hotel_pipeline.source_registry.SourceRegistry", _Registry)
    monkeypatch.setattr(
        "hotel_pipeline.lot1b_coverage.build",
        lambda ws: called.append("coverage") or {"coverage_report": "p"},
    )
    monkeypatch.setattr(
        "hotel_pipeline.scene_package.build",
        lambda ws: called.append("scene") or {"scene": "p"},
    )

    run_step("lot1b", workspace)

    assert called == ["sources", "coverage", "scene"]


def test_lot1b_stops_when_no_router_decision_is_published(tmp_path, monkeypatch) -> None:
    """Sans route arrêtée, la couverture citerait une décision inexistante."""
    workspace = _workspace(tmp_path)

    class _Registry:
        closed_families = 15
        required_families = 15
        closure_complete = True

        @staticmethod
        def model_validate_json(_):
            return _Registry()

    monkeypatch.setattr(
        "hotel_pipeline.source_registry.build",
        lambda ws: ws.write_json("00_manifest/source_registry.json", {}),
    )
    monkeypatch.setattr("hotel_pipeline.source_registry.SourceRegistry", _Registry)

    built: list[str] = []
    monkeypatch.setattr(
        "hotel_pipeline.lot1b_coverage.build",
        lambda ws: built.append("coverage"),
    )

    with pytest.raises(StepBlocked, match="Router"):
        run_step("lot1b", workspace)

    # La couverture n'a pas été produite sur une route absente.
    assert built == []


def test_lot1b_blocks_on_an_incomplete_source_campaign(tmp_path, monkeypatch) -> None:
    """Le paquet est publié, mais la campagne incomplète reste un gate."""
    workspace = _workspace(tmp_path)
    workspace.write_json("10_validation/router_decision_x.json", {"path": "path_d"})

    class _Registry:
        closed_families = 2
        required_families = 15
        closure_complete = False

        @staticmethod
        def model_validate_json(_):
            return _Registry()

    monkeypatch.setattr(
        "hotel_pipeline.source_registry.build",
        lambda ws: ws.write_json("00_manifest/source_registry.json", {}),
    )
    monkeypatch.setattr("hotel_pipeline.source_registry.SourceRegistry", _Registry)
    monkeypatch.setattr("hotel_pipeline.lot1b_coverage.build", lambda ws: {})
    monkeypatch.setattr("hotel_pipeline.scene_package.build", lambda ws: {})

    with pytest.raises(StepBlocked, match="2/15"):
        run_step("lot1b", workspace)
