"""Muter un fichier suivi, ou n'en rien faire (collecte V2).

Le défaut fermé ici s'est produit dans les deux sens. Un reçu écrit **après**
la mutation a manqué : la politique du pilote s'est retrouvée migrée sans
trace, relisible comme si elle avait toujours eu cette forme. L'inverse aurait
permis à un reçu d'affirmer une migration jamais faite.

Les deux ordres sont faux parce que le problème n'est pas l'ordre : il manque
un état intermédiaire. D'où le manifeste préparé, qui décrit une intention sans
rien changer, et l'empreinte du fichier qui tranche à la reprise.
"""

from __future__ import annotations

import json
import os

import pytest

from hotel_pipeline.transaction import (
    TransactionConflict,
    commit,
    pending,
    prepare,
    recover,
    sha_of,
    sha_of_file,
    write_atomic,
)


def _target(tmp_path, content="avant\n"):
    path = tmp_path / "suivi.json"
    path.write_text(content, "utf-8")
    return path


# --- écriture atomique --------------------------------------------------------


def test_an_interrupted_write_leaves_the_original_intact(tmp_path) -> None:
    """`write_text` tronque puis écrit : interrompue, elle laisse un fichier
    qui n'est aucun des deux états."""
    path = _target(tmp_path, "état initial")
    original = path.read_text("utf-8")

    class Boom(RuntimeError):
        pass

    def exploding_replace(*_args):
        raise Boom("interruption au pire moment")

    saved = os.replace
    os.replace = exploding_replace
    try:
        with pytest.raises(Boom):
            write_atomic(path, "état final")
    finally:
        os.replace = saved

    assert path.read_text("utf-8") == original
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".suivi")]
    assert leftovers == [], "aucun fichier temporaire ne subsiste"


def test_an_atomic_write_replaces_the_content(tmp_path) -> None:
    path = _target(tmp_path)
    write_atomic(path, "après\n")
    assert path.read_text("utf-8") == "après\n"


# --- les trois temps ----------------------------------------------------------


def test_the_prepared_manifest_changes_nothing(tmp_path) -> None:
    """Il décrit une intention : le fichier ne doit pas avoir bougé."""
    path = _target(tmp_path)
    before = path.read_text("utf-8")

    transaction = prepare(path, "après\n", kind="essai")

    assert path.read_text("utf-8") == before
    assert transaction.sha_before == sha_of(before)
    assert transaction.sha_after == sha_of("après\n")


def test_a_commit_publishes_prepared_then_mutates_then_receipts(tmp_path) -> None:
    path = _target(tmp_path)
    seen: list[tuple[str, str]] = []

    def prepared(payload):
        seen.append((payload["state"], path.read_text("utf-8")))

    def committed(payload):
        seen.append((payload["state"], path.read_text("utf-8")))

    transaction = prepare(path, "après\n", kind="essai")
    commit(transaction, "après\n", prepared, committed)

    assert seen == [("prepared", "avant\n"), ("committed", "après\n")], (
        "le manifeste précède la mutation, le reçu la suit"
    )


def test_a_file_changed_since_preparation_is_not_overwritten(tmp_path) -> None:
    """Quelqu'un est passé : écraser son travail serait pire que refuser."""
    path = _target(tmp_path)
    transaction = prepare(path, "après\n", kind="essai")

    path.write_text("modifié par un tiers\n", "utf-8")

    with pytest.raises(TransactionConflict, match="a changé depuis la préparation"):
        commit(transaction, "après\n", lambda _p: None, lambda _p: None)

    assert path.read_text("utf-8") == "modifié par un tiers\n"


# --- la reprise : l'empreinte tranche -----------------------------------------


def test_recovery_sees_an_unapplied_transaction(tmp_path) -> None:
    """Au SHA initial : la mutation n'a pas eu lieu, rien à défaire."""
    path = _target(tmp_path)
    transaction = prepare(path, "après\n", kind="essai")

    resolution = recover(transaction.as_dict(state="prepared"), path)

    assert resolution["state"] == "abandoned"
    assert resolution["recovered"] is True
    assert "n'a pas été appliquée" in resolution["resolution"]


def test_recovery_sees_an_applied_transaction_missing_its_receipt(tmp_path) -> None:
    """Au SHA final : c'est exactement ce qui est arrivé au pilote."""
    path = _target(tmp_path)
    transaction = prepare(path, "après\n", kind="essai")
    write_atomic(path, "après\n")  # la mutation a eu lieu, le reçu manque

    resolution = recover(transaction.as_dict(state="prepared"), path)

    assert resolution["state"] == "committed"
    assert resolution["recovered"] is True
    assert "seul son reçu manquait" in resolution["resolution"]


def test_recovery_refuses_a_third_state(tmp_path) -> None:
    """Ni avant, ni après : deviner serait pire que refuser."""
    path = _target(tmp_path)
    transaction = prepare(path, "après\n", kind="essai")
    path.write_text("un troisième état\n", "utf-8")

    with pytest.raises(TransactionConflict, match="quelqu'un est passé"):
        recover(transaction.as_dict(state="prepared"), path)


def test_a_prepared_manifest_without_a_receipt_is_pending(tmp_path) -> None:
    """C'est l'absence de reçu qui signale une transaction à reprendre."""
    (tmp_path / "essai_T1_prepared.json").write_text(
        json.dumps({"transaction_id": "T1"}), "utf-8"
    )
    (tmp_path / "essai_T2_prepared.json").write_text(
        json.dumps({"transaction_id": "T2"}), "utf-8"
    )
    (tmp_path / "essai_T2_committed.json").write_text("{}", "utf-8")

    outstanding = pending(tmp_path, "essai")

    assert [row["transaction_id"] for row in outstanding] == ["T1"]


def test_a_missing_target_has_no_prior_sha(tmp_path) -> None:
    """Créer un fichier est une mutation comme une autre : `None` avant."""
    path = tmp_path / "inexistant.json"
    assert sha_of_file(path) is None

    transaction = prepare(path, "neuf\n", kind="essai")
    assert transaction.sha_before is None

    commit(transaction, "neuf\n", lambda _p: None, lambda _p: None)
    assert path.read_text("utf-8") == "neuf\n"


# --- précondition des tests CLI ------------------------------------------------


def test_the_workspace_root_really_points_at_the_temporary_directory(
    tmp_path, monkeypatch
) -> None:
    """Un test CLI qui écrirait dans le vrai espace de travail le corromprait.

    La variable seule ne suffit pas : c'est ce que `Workspace` en fait qui
    compte, et une valeur relative ou ignorée passerait inaperçue.
    """
    from hotel_pipeline.workspace import Workspace

    monkeypatch.setenv("HOTEL_PIPELINE_WORK", str(tmp_path / "travail"))
    workspace = Workspace("essai")

    assert workspace.root.resolve().is_relative_to(tmp_path.resolve()), (
        f"l'espace de travail est {workspace.root}, hors du répertoire "
        "temporaire : un test écrirait dans le vrai dépôt"
    )
