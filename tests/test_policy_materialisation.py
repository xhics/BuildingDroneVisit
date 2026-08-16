"""Matérialisation de la politique (collecte V2).

Dix chemins venaient des valeurs par défaut du code : deux seuils `geometry`,
six paramètres `collection`, et les sections `coverage` et `adaptive_search`
entières. Le fichier paraissait complet ; il l'était pour la validation, pas
pour la lecture — un relecteur ne pouvait pas savoir sur quels seuils un
manifeste avait été produit.

Ce que ces tests protègent : la migration change la **représentation**, jamais
les valeurs. Une empreinte qui bougerait signalerait un recalibrage déguisé.
"""

from __future__ import annotations

import json

import pytest

from hotel_pipeline.context import implicit_paths
from hotel_pipeline.policy_materialise import (
    MaterialisationRefused,
    materialise,
)
from hotel_pipeline.provenance import policy_digest
from hotel_pipeline.schemas import DEFAULT_POLICY, PipelinePolicy

#: Les dix chemins de l'inventaire. `coverage` et `adaptive_search` sont des
#: sections entières ; les autres sont des champs isolés d'une section
#: présente — le cas le plus insidieux, puisque le fichier paraît complet.
IMPLICIT_PATHS = [
    "geometry.sector_observer_half_width_deg",
    "geometry.viewpoint_separation_m",
    "collection.framing_merge_bearing_deg",
    "collection.preview_resolution",
    "collection.full_resolution",
    "collection.sequence_enrichment_per_demand",
    "collection.sequence_expansion_max_members",
    "collection.sequence_expansion_max_distance_m",
    "coverage",
    "adaptive_search",
]


def _complete() -> dict:
    return json.loads(DEFAULT_POLICY.model_dump_json())


def _without(path: str) -> dict:
    """La politique complète, privée d'un chemin — comme un fichier ancien."""
    document = _complete()
    parts = path.split(".")
    target = document
    for part in parts[:-1]:
        target = target[part]
    target.pop(parts[-1], None)
    return document


def _write(tmp_path, document: dict):
    policy_path = tmp_path / "pipeline_policy.json"
    policy_path.write_text(json.dumps(document, indent=2, ensure_ascii=False), "utf-8")
    return policy_path


# --- chaque chemin, un par un -------------------------------------------------


@pytest.mark.parametrize("path", IMPLICIT_PATHS)
def test_each_implicit_path_is_materialised_without_changing_values(
    tmp_path, path
) -> None:
    """Retirer un seul chemin suffit à rendre la politique implicite.

    Et le remettre ne doit rien changer d'autre : c'est exactement la valeur
    que le code appliquait déjà.
    """
    partial = _without(path)
    before_digest = policy_digest(PipelinePolicy.model_validate(partial))
    policy_path = _write(tmp_path, partial)

    receipt = materialise(policy_path)

    assert receipt.materialised, f"{path} devait être matérialisé"
    assert receipt.digest_before == before_digest
    assert receipt.digest_after == before_digest, (
        "matérialiser ne recalibre rien : l'empreinte effective est la même"
    )
    assert receipt.values_unchanged
    assert receipt.altered == {} and receipt.removed == []
    assert receipt.version_before == receipt.version_after

    rewritten = json.loads(policy_path.read_text("utf-8"))
    assert implicit_paths(
        PipelinePolicy.model_validate(rewritten), rewritten
    ) == [], f"{path} reste implicite après migration"


def test_the_ten_paths_are_exactly_those_the_code_fills(tmp_path) -> None:
    """Le contrat porte sur ces dix chemins, ni plus ni moins.

    En ajouter un sans le déclarer laisserait une valeur venir du code sans que
    l'inventaire le dise.
    """
    stripped = _complete()
    for path in IMPLICIT_PATHS:
        parts = path.split(".")
        target = stripped
        for part in parts[:-1]:
            target = target[part]
        target.pop(parts[-1], None)

    detected = implicit_paths(
        PipelinePolicy.model_validate(stripped), stripped
    )
    assert sorted(detected) == sorted(IMPLICIT_PATHS)


# --- ce que la migration refuse ------------------------------------------------


def test_a_migration_that_would_change_a_value_is_refused(tmp_path, monkeypatch) -> None:
    """Écrire d'abord et vérifier ensuite laisserait un fichier faux sur le
    disque le temps de s'en apercevoir."""
    partial = _without("coverage")
    policy_path = _write(tmp_path, partial)
    original = policy_path.read_text("utf-8")

    # Une sérialisation qui déplacerait une valeur : le reçu doit le voir.
    # Capturé **avant** le patch, sinon le remplaçant s'appellerait lui-même.
    altered = _complete()
    altered["geometry"]["max_distance_m"] = 9999.0
    payload = json.dumps(altered)

    monkeypatch.setattr(
        PipelinePolicy, "model_dump_json", lambda self, **k: payload
    )

    with pytest.raises(MaterialisationRefused, match="changerait la politique"):
        materialise(policy_path)

    assert policy_path.read_text("utf-8") == original, (
        "rien n'est écrit quand la migration est refusée"
    )


def test_an_already_complete_policy_is_left_alone(tmp_path) -> None:
    """Rejouer la migration ne doit pas réécrire ni produire un reçu vide."""
    policy_path = _write(tmp_path, _complete())
    original = policy_path.read_text("utf-8")

    receipt = materialise(policy_path)

    assert receipt.materialised == {}
    assert policy_path.read_text("utf-8") == original


def test_the_receipt_is_written_before_the_policy(tmp_path) -> None:
    """Une interruption entre les deux laisserait une migration sans trace.

    C'est arrivé : une erreur après l'écriture du fichier a laissé la politique
    du pilote migrée et son reçu absent. Un reçu sans migration est visible et
    se corrige ; l'inverse ne se voit pas.
    """
    policy_path = _write(tmp_path, _without("adaptive_search"))
    order: list[str] = []

    def publish(_receipt) -> None:
        order.append("reçu")
        # Le fichier n'est pas encore réécrit à cet instant.
        assert "adaptive_search" not in json.loads(
            policy_path.read_text("utf-8")
        )

    materialise(policy_path, publish_receipt=publish)
    order.append("politique")

    assert order == ["reçu", "politique"]


# --- ce que la migration ne périme pas ----------------------------------------


def test_materialising_stales_no_production(tmp_path) -> None:
    """Les valeurs effectives sont inchangées : rien de ce qui a été produit
    sur elles ne devient caduc.

    Candidats, LiDAR, visibilité : tous jugés sur des seuils que la migration
    n'a pas touchés.
    """
    from hotel_pipeline.policy_facets import CONSUMERS, dependency_digests, stale_facets

    partial = _without("coverage")
    before = PipelinePolicy.model_validate(partial)
    policy_path = _write(tmp_path, partial)
    materialise(policy_path)
    after = PipelinePolicy.model_validate(json.loads(policy_path.read_text("utf-8")))

    for production in CONSUMERS:
        recorded = dependency_digests(before, production)
        assert stale_facets(recorded, after, production) == [], (
            f"{production} périmé par une migration de représentation"
        )
