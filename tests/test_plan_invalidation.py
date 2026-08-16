"""Retirer un plan de la circulation sans effacer ce qu'il disait (collecte V2).

Trois brouillons précédaient le contrat `ResolvedAcquisitionRequest`. Invalider
le dernier seul aurait fait remonter l'avant-dernier par `_latest_plan` — qui
porte exactement le même défaut. C'est le piège que ces tests protègent.
"""

from __future__ import annotations

import json

import pytest

from hotel_pipeline.plan_invalidation import (
    InvalidationReason,
    InvalidationRefused,
    build,
    invalidated_plan_ids,
)

MOTIF = InvalidationReason.PRE_RESOLVED_ACQUISITION_REQUEST_CONTRACT
POURQUOI = "produit avant que la résolution planifiée n'atteigne la requête"


def _plan_file(directory, plan_id: str, acquisitions=1):
    path = directory / f"acquisition_plan_{plan_id}.json"
    path.write_text(
        json.dumps(
            {
                "plan_id": plan_id,
                "hotel_id": "essai",
                "status": "draft",
                "acquisitions": [{"candidate_id": f"c{i}"} for i in range(acquisitions)],
            },
            indent=2,
        ),
        "utf-8",
    )
    return path


def _commit(directory, event) -> None:
    """Publie l'événement comme le ferait la commande."""
    (directory / f"plan_invalidation_{event.invalidation_id}_committed.json").write_text(
        json.dumps(event.as_dict(state="committed"), indent=2, ensure_ascii=False),
        "utf-8",
    )


# --- l'événement nomme, il n'efface pas ---------------------------------------


def test_the_plans_are_named_with_their_digest(tmp_path) -> None:
    """Nommer un identifiant sans son SHA laisserait l'invalidation porter sur
    un fichier qu'on n'a pas vu."""
    from hotel_pipeline.transaction import sha_of_file

    first = _plan_file(tmp_path, "T1")
    second = _plan_file(tmp_path, "T2", acquisitions=3)

    event = build([first, second], MOTIF, POURQUOI)

    named = {plan.plan_id: plan.sha256 for plan in event.plans}
    assert named == {"T1": sha_of_file(first), "T2": sha_of_file(second)}
    assert event.reason is MOTIF


def test_the_files_are_untouched(tmp_path) -> None:
    """Supprimer effacerait ce qui a été planifié ; réécrire serait pire."""
    path = _plan_file(tmp_path, "T1")
    before = path.read_bytes()

    event = build([path], MOTIF, POURQUOI)
    _commit(tmp_path, event)

    assert path.is_file()
    assert path.read_bytes() == before, "pas un octet modifié"


def test_an_unknown_plan_is_refused(tmp_path) -> None:
    """On n'invalide pas ce qu'on n'a pas lu."""
    with pytest.raises(InvalidationRefused, match="introuvable"):
        build([tmp_path / "acquisition_plan_absent.json"], MOTIF, POURQUOI)


def test_an_invalidation_without_plans_is_refused(tmp_path) -> None:
    with pytest.raises(InvalidationRefused, match="n'invalide rien"):
        build([], MOTIF, POURQUOI)


def test_an_invalidation_without_a_rationale_is_refused(tmp_path) -> None:
    """Le code structuré dit la catégorie, non ce qu'un relecteur doit
    comprendre."""
    path = _plan_file(tmp_path, "T1")

    with pytest.raises(InvalidationRefused, match="sans motif lisible"):
        build([path], MOTIF, "   ")


# --- ce que la sélection retient ----------------------------------------------


def test_a_prepared_manifest_invalidates_nothing(tmp_path) -> None:
    """Il décrit une intention : rien ne dit encore qu'elle a abouti."""
    path = _plan_file(tmp_path, "T1")
    event = build([path], MOTIF, POURQUOI)

    (tmp_path / f"plan_invalidation_{event.invalidation_id}_prepared.json").write_text(
        json.dumps(event.as_dict(state="prepared")), "utf-8"
    )

    assert invalidated_plan_ids(tmp_path) == set(), (
        "un manifeste préparé ferait disparaître un plan qu'aucun événement "
        "n'a retiré"
    )

    _commit(tmp_path, event)
    assert invalidated_plan_ids(tmp_path) == {"T1"}


def test_invalidating_the_last_plan_does_not_promote_the_previous_one(
    tmp_path, monkeypatch
) -> None:
    """Le piège exact : trois brouillons portant le même défaut.

    Invalider le dernier seul ferait remonter l'avant-dernier, et le défaut
    reviendrait par la sélection.
    """
    from hotel_pipeline.cli import _latest_plan
    from hotel_pipeline.workspace import Workspace

    monkeypatch.setenv("HOTEL_PIPELINE_WORK", str(tmp_path / "travail"))
    workspace = Workspace("essai")
    sources = workspace.path("01_sources")
    sources.mkdir(parents=True, exist_ok=True)
    assert workspace.root.resolve().is_relative_to((tmp_path).resolve())

    first = _plan_file(sources, "20260815T161351329975Z")
    second = _plan_file(sources, "20260815T161740884597Z")
    third = _plan_file(sources, "20260815T162441387391Z")

    # Le dernier seul : l'avant-dernier remonte, avec le même défaut.
    _commit(sources, build([third], MOTIF, POURQUOI))
    assert _latest_plan(workspace).name.endswith("161740884597Z.json")

    # Les trois : plus rien en circulation.
    _commit(sources, build([first, second, third], MOTIF, POURQUOI))
    assert _latest_plan(workspace) is None, (
        "tous invalidés : pas de repli historique"
    )


def test_a_new_conforming_plan_becomes_the_current_one(tmp_path, monkeypatch) -> None:
    """Sans quoi l'invalidation bloquerait le pipeline au lieu de l'assainir."""
    from hotel_pipeline.cli import _latest_plan
    from hotel_pipeline.workspace import Workspace

    monkeypatch.setenv("HOTEL_PIPELINE_WORK", str(tmp_path / "travail"))
    workspace = Workspace("essai")
    sources = workspace.path("01_sources")
    sources.mkdir(parents=True, exist_ok=True)

    ancien = _plan_file(sources, "20260815T160000000000Z")
    _commit(sources, build([ancien], MOTIF, POURQUOI))
    assert _latest_plan(workspace) is None

    _plan_file(sources, "20260901T120000000000Z")
    latest = _latest_plan(workspace)

    assert latest is not None
    assert latest.name.endswith("20260901T120000000000Z.json")


def test_an_invalidation_names_no_wildcard(tmp_path) -> None:
    """Ce qui n'est pas nommé n'est pas invalidé.

    Une plage implicite retirerait des plans que personne n'a examinés.
    """
    first = _plan_file(tmp_path, "T1")
    _plan_file(tmp_path, "T2")

    _commit(tmp_path, build([first], MOTIF, POURQUOI))

    assert invalidated_plan_ids(tmp_path) == {"T1"}


def test_an_explicit_plan_option_cannot_bypass_the_invalidation(
    tmp_path, monkeypatch
) -> None:
    """`--plan` court-circuite la sélection.

    Sans ce contrôle, il suffisait de nommer le fichier pour exécuter ce qu'une
    invalidation avait écarté — et le refus doit tomber avant toute création de
    répertoire, lecture de cache ou appel réseau.
    """
    from typer.testing import CliRunner

    from hotel_pipeline.cli import app
    from hotel_pipeline.workspace import Workspace

    monkeypatch.setenv("HOTEL_PIPELINE_WORK", str(tmp_path / "travail"))
    monkeypatch.setenv("HOTEL_PIPELINE_OFFLINE", "1")

    runner = CliRunner()
    runner.invoke(app, [
        "init", "essai", "--address", "1 rue Test", "--name", "Essai",
        "--country", "CA", "--timezone", "America/Toronto",
        "--ocr-language", "fr", "--lat", "45.5", "--lon", "-73.4",
    ])

    workspace = Workspace("essai")
    assert workspace.root.resolve().is_relative_to(tmp_path.resolve()), (
        "l'espace de travail doit rester dans le répertoire temporaire"
    )
    sources = workspace.path("01_sources")
    sources.mkdir(parents=True, exist_ok=True)

    retired = _plan_file(sources, "20260815T162441387391Z")
    _commit(sources, build([retired], MOTIF, POURQUOI))

    result = runner.invoke(app, [
        "assets", "acquire", "essai", "--plan", str(retired),
    ])

    assert result.exit_code == 2, result.output
    assert "invalidé" in result.output
    assert "retiré de la circulation" in result.output
    # Rien n'a été téléchargé : le répertoire de destination n'existe même pas.
    assert not (workspace.path("03_images")).exists() or not list(
        workspace.path("03_images").iterdir()
    )
