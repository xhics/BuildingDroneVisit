"""Découverte ciblée sur un besoin nommé (collecte V2).

Le pilote a laissé un seul besoin responsable de `CAPTURE_REQUIRED` :
`ACCESS_ROAD_MAIN`. Relancer une collecte générale pour lui interrogerait les
sources sur six besoins déjà arbitrés, dépenserait le quota et produirait un
corpus qu'il faudrait à nouveau trier.

Ce que ces tests protègent, et pourquoi chacun existe :

```text
besoin inconnu refusé      avant cache et réseau — sinon la requête est déjà
                           partie quand on s'en aperçoit
aucun autre interrogé      un corpus ciblé ne dit rien des autres besoins
jamais manifeste courant   « aucune vue de façade » ne doit pas se confondre
                           avec « aucune façade cherchée »
scopes non mélangés        planifier hors portée lirait une absence comme un
                           constat
couple réfuté écarté       un examen humain a tranché ; le resservir ferait
                           refaire le même travail
aperçu obligatoire         payer la pleine résolution avant qu'un humain ait
                           vu l'image
```
"""

from __future__ import annotations

import json

import pytest
import typer
from typer.testing import CliRunner

from hotel_pipeline.schemas.acquisition import (
    CaptureDemand,
    CaptureDemandManifest,
    CaptureIntent,
    CandidateManifest,
    DiscoveryMode,
    DiscoveryScope,
    TargetKind,
)

runner = CliRunner()


def _demande(demand_id, target_ref, kind=TargetKind.SITE_OBJECT):
    return CaptureDemand(
        demand_id=demand_id, intent=CaptureIntent.BUILDING_CAPTURE,
        target_kind=kind, target_ref=target_ref, viewpoints_required=2,
    )


LES_SEPT = CaptureDemandManifest(
    hotel_id="essai",
    demands=[
        _demande("obligation:ACCESS_ROAD_MAIN", "ACCESS_ROAD_MAIN",
                 TargetKind.CONTEXT_CORRIDOR),
        _demande("obligation:FACADE_PRIMARY", "front", TargetKind.VIEW_SECTOR),
        _demande("obligation:FACADE_REAR", "rear", TargetKind.VIEW_SECTOR),
        _demande("obligation:PROPERTY_SIGN", "PROPERTY_SIGN"),
    ],
)


# --- un besoin inconnu est refusé avant tout appel ----------------------------


def test_an_unknown_demand_is_refused_before_any_source_is_touched() -> None:
    """Valider après l'appel aurait déjà émis la requête et dépensé le quota.

    Le refus doit donc précéder le cache **et** le réseau, non les suivre.
    """
    from hotel_pipeline import cli

    with pytest.raises(typer.Exit) as sortie:
        cli._discovery_scope(LES_SEPT, ["obligation:INEXISTANT"], "d" * 16)

    assert sortie.value.exit_code == 2


def test_the_refusal_names_every_unknown_demand() -> None:
    """N'en nommer qu'un ferait relancer la commande autant de fois qu'il y a
    de fautes de frappe."""
    from hotel_pipeline import cli

    with pytest.raises(typer.Exit):
        cli._discovery_scope(
            LES_SEPT, ["obligation:INCONNU_A", "obligation:INCONNU_B"], "d" * 16
        )


def test_a_known_demand_passes() -> None:
    from hotel_pipeline import cli

    scope, restreint = cli._discovery_scope(
        LES_SEPT, ["obligation:ACCESS_ROAD_MAIN"], "d" * 16
    )

    assert scope.mode is DiscoveryMode.TARGETED
    assert scope.demand_ids == ("obligation:ACCESS_ROAD_MAIN",)
    assert scope.demand_manifest_digest == "d" * 16


# --- aucun autre besoin n'est interrogé ni évalué ------------------------------


def test_no_other_demand_is_queried_or_assessed() -> None:
    """Filtrer plus loin laisserait les six autres être interrogés « pour
    rien », et le rapport dirait qu'on les a cherchés."""
    from hotel_pipeline import cli

    _scope, restreint = cli._discovery_scope(
        LES_SEPT, ["obligation:ACCESS_ROAD_MAIN"], "d" * 16
    )

    assert [d.demand_id for d in restreint.demands] == [
        "obligation:ACCESS_ROAD_MAIN"
    ], "le manifeste transmis aux sources ne porte que le besoin visé"
    assert len(LES_SEPT.demands) == 4, "le manifeste canonique n'est pas modifié"


def test_the_full_manifest_is_validated_against_not_a_subset() -> None:
    """Charger un sous-ensemble ne permettrait pas de dire qu'un identifiant
    est inconnu : il passerait pour un besoin d'un autre manifeste."""
    from hotel_pipeline import cli

    # `FACADE_REAR` existe au manifeste canonique : il doit être accepté même
    # si l'on ne cible que lui.
    scope, restreint = cli._discovery_scope(
        LES_SEPT, ["obligation:FACADE_REAR"], "d" * 16
    )
    assert scope.demand_ids == ("obligation:FACADE_REAR",)
    assert len(restreint.demands) == 1


def test_the_corridor_of_the_demand_is_recorded() -> None:
    """La recherche se cadre sur le corridor résolu, au lieu de balayer le
    rayon entier."""
    from hotel_pipeline import cli

    scope, _ = cli._discovery_scope(
        LES_SEPT, ["obligation:ACCESS_ROAD_MAIN"], "d" * 16
    )

    assert scope.corridor_ref == "ACCESS_ROAD_MAIN"


def test_two_corridors_leave_the_reference_empty() -> None:
    """Nommer l'un des deux ferait croire que la recherche s'y est cadrée."""
    from hotel_pipeline import cli

    manifeste = CaptureDemandManifest(
        hotel_id="essai",
        demands=[
            _demande("obligation:A", "ROUTE_A", TargetKind.CONTEXT_CORRIDOR),
            _demande("obligation:B", "ROUTE_B", TargetKind.CONTEXT_CORRIDOR),
        ],
    )
    scope, _ = cli._discovery_scope(
        manifeste, ["obligation:A", "obligation:B"], "d" * 16
    )

    assert scope.corridor_ref == ""


# --- la portée est inscrite, et vérifiable ------------------------------------


def test_a_targeted_scope_must_name_its_demands() -> None:
    """Sans besoin nommé, elle ne se distinguerait pas d'une découverte
    complète, et son corpus partiel passerait pour un corpus entier."""
    with pytest.raises(ValueError, match="sans besoin nommé"):
        DiscoveryScope(mode=DiscoveryMode.TARGETED)


def test_a_full_scope_cannot_name_demands() -> None:
    """Deux sources de vérité divergeraient sur ce qui a été cherché."""
    with pytest.raises(ValueError, match="deux sources de vérité"):
        DiscoveryScope(mode=DiscoveryMode.FULL, demand_ids=("obligation:X",))


def test_manifests_written_before_this_field_stay_readable() -> None:
    """Ils portaient bien sur tous les besoins : le défaut doit le dire."""
    manifeste = CandidateManifest(hotel_id="essai")

    assert manifeste.scope.mode is DiscoveryMode.FULL
    assert manifeste.scope.demand_ids == ()


# --- une découverte ciblée ne devient jamais le manifeste courant --------------


def test_a_targeted_discovery_never_becomes_the_current_manifest(tmp_path) -> None:
    """`_latest_candidates` ramasserait sinon un corpus qui ne dit rien des six
    autres besoins, et le plan acquerrait sur cette base."""
    from hotel_pipeline import cli
    from hotel_pipeline.workspace import Workspace

    workspace = Workspace("essai", root=tmp_path)
    sources = workspace.path("01_sources")
    sources.mkdir(parents=True, exist_ok=True)

    # Un manifeste courant, et un manifeste ciblé écrit **après** lui.
    (sources / "candidates_20260101T000000Z.json").write_text("{}", "utf-8")
    ciblé = sources / "targeted" / "20260816T999999Z"
    ciblé.mkdir(parents=True)
    (ciblé / "candidates_20260816T999999Z.json").write_text("{}", "utf-8")

    dernier = cli._latest_candidates(workspace)

    assert dernier is not None
    assert dernier.name == "candidates_20260101T000000Z.json", (
        "le manifeste ciblé, pourtant plus récent, ne doit pas être ramassé"
    )
    assert dernier.parent.name == "01_sources", (
        "le manifeste courant vit à la racine, non dans un dossier ciblé"
    )


def test_the_targeted_run_writes_under_its_own_run_directory() -> None:
    """Un dossier par exécution : deux découvertes ciblées successives ne
    doivent pas se recouvrir."""
    import inspect

    from hotel_pipeline import cli

    source = inspect.getsource(cli.assets_discover)

    assert 'f"01_sources/targeted/{report.run_id}"' in source, (
        "la publication ciblée doit porter le run_id, sinon deux exécutions "
        f"écriraient au même endroit — vu : {source[:0]!r}"
    )


# --- un plan ne mélange pas deux portées ---------------------------------------


def test_a_plan_cannot_mix_two_scopes() -> None:
    """Planifier les sept besoins sur un corpus ciblé lirait l'absence de vues
    des six autres comme un constat, alors qu'aucune n'a été cherchée."""
    from hotel_pipeline import cli

    ciblé = CandidateManifest(
        hotel_id="essai",
        scope=DiscoveryScope(
            mode=DiscoveryMode.TARGETED,
            demand_ids=("obligation:ACCESS_ROAD_MAIN",),
            demand_manifest_digest="d" * 16,
        ),
    )

    problems = cli._validate_manifest_pairing(
        "essai", ciblé, LES_SEPT, json.loads(LES_SEPT.model_dump_json())
    )

    assert any("n'ont pas été cherchés" in p for p in problems), (
        f"le mélange de portées doit être refusé — vu : {problems}"
    )


def test_a_plan_on_the_very_demand_it_targeted_is_accepted() -> None:
    """Sans quoi la découverte ciblée ne servirait à rien."""
    from hotel_pipeline import cli

    un_besoin = CaptureDemandManifest(
        hotel_id="essai",
        demands=[
            _demande("obligation:ACCESS_ROAD_MAIN", "ACCESS_ROAD_MAIN",
                     TargetKind.CONTEXT_CORRIDOR),
        ],
    )
    ciblé = CandidateManifest(
        hotel_id="essai",
        scope=DiscoveryScope(
            mode=DiscoveryMode.TARGETED,
            demand_ids=("obligation:ACCESS_ROAD_MAIN",),
            demand_manifest_digest="d" * 16,
        ),
    )

    problems = cli._validate_manifest_pairing(
        "essai", ciblé, un_besoin, json.loads(un_besoin.model_dump_json())
    )

    assert not any("n'ont pas été cherchés" in p for p in problems)


def test_a_full_manifest_still_plans_on_every_demand() -> None:
    """La règle ne doit pas gêner le chemin ordinaire."""
    from hotel_pipeline import cli

    complet = CandidateManifest(hotel_id="essai")

    problems = cli._validate_manifest_pairing(
        "essai", complet, LES_SEPT, json.loads(LES_SEPT.model_dump_json())
    )

    assert not any("n'ont pas été cherchés" in p for p in problems)


# --- le câblage réel : neutraliser le filtre doit se voir ----------------------


def test_the_filter_is_applied_before_any_source_is_queried() -> None:
    """Le test de câblage.

    Vérifier `_discovery_scope` isolément ne dit rien de ce que la commande en
    fait : si `assets discover` interrogeait les sources avec le manifeste
    complet, le filtre serait calculé puis ignoré, et sept besoins partiraient
    au réseau.
    """
    import inspect

    from hotel_pipeline import cli

    source = inspect.getsource(cli.assets_discover)

    portée = source.index("_discovery_scope(")
    interrogation = source.index("_query_sources(")

    assert portée < interrogation, (
        "la portée doit être arrêtée avant l'interrogation des sources : "
        "après, la requête serait déjà partie"
    )

    # Et le manifeste transmis doit être celui que le filtre a restreint.
    appel = source[interrogation:]
    appel = appel[: appel.index(")\n") + 1]
    assert "demands.demands" in appel, (
        f"les sources doivent recevoir le manifeste restreint — vu : {appel!r}"
    )
    assert "scope, demands = _discovery_scope" in source, (
        "le filtre doit remplacer le manifeste, non produire une valeur que "
        "la suite ignore"
    )


# --- un couple réfuté n'est pas reproposé --------------------------------------


def _candidat(candidate_id):
    from hotel_pipeline.schemas.acquisition import CaptureCandidate

    return CaptureCandidate(
        candidate_id=candidate_id, source="mapillary",
        provider_id=candidate_id.replace("cand-", ""),
    )


def _réfutation(asset_id, demand_id):
    from hotel_pipeline.schemas.preview import PreviewAssessment, PreviewVerdict

    return PreviewAssessment(
        asset_id=asset_id, demand_id=demand_id,
        plan_id="plan-essai", request_digest="r" * 16, checksum="c" * 64,
        verdict=PreviewVerdict.REFUTED,
        rationale="l'aperçu montre le bâtiment voisin, non la voie d'accès",
        assessed_by="opérateur",
    )


def _log(tmp_path, *entrées):
    from hotel_pipeline.schemas.preview import PreviewAssessmentLog
    from hotel_pipeline.workspace import Workspace

    workspace = Workspace("essai", root=tmp_path)
    sources = workspace.path("01_sources")
    sources.mkdir(parents=True, exist_ok=True)
    log = PreviewAssessmentLog(hotel_id="essai", entries=list(entrées))
    (sources / "preview_assessments.json").write_text(
        log.model_dump_json(), "utf-8"
    )
    return workspace


CIBLE = DiscoveryScope(
    mode=DiscoveryMode.TARGETED,
    demand_ids=("obligation:ACCESS_ROAD_MAIN",),
    demand_manifest_digest="d" * 16,
)


def test_a_refuted_couple_is_never_proposed_again(tmp_path) -> None:
    """Un examen humain a constaté que cette vue ne montre pas ce besoin.

    La resservir ferait refaire le même examen, et la réfutation n'aurait servi
    à rien.
    """
    from hotel_pipeline import cli

    workspace = _log(
        tmp_path, _réfutation("cand-A", "obligation:ACCESS_ROAD_MAIN")
    )
    manifeste = CandidateManifest(
        hotel_id="essai",
        candidates=[_candidat("cand-A"), _candidat("cand-B")],
        scope=CIBLE,
    )

    filtré = cli._targeted_manifest(workspace, manifeste, CIBLE)

    assert [c.candidate_id for c in filtré.candidates] == ["cand-B"]


def test_a_refutation_on_another_demand_does_not_exclude(tmp_path) -> None:
    """Un constat vaut pour le couple qu'il nomme.

    L'étendre écarterait une vue qui n'a jamais été examinée pour **ce**
    besoin — et le refus se propagerait sans que personne l'ait prononcé.
    """
    from hotel_pipeline import cli

    workspace = _log(
        tmp_path, _réfutation("cand-A", "obligation:FACADE_REAR")
    )
    manifeste = CandidateManifest(
        hotel_id="essai", candidates=[_candidat("cand-A")], scope=CIBLE,
    )

    filtré = cli._targeted_manifest(workspace, manifeste, CIBLE)

    assert [c.candidate_id for c in filtré.candidates] == ["cand-A"], (
        "réfuté pour la façade arrière, jamais examiné pour la voie d'accès"
    )


def test_an_absent_preview_log_excludes_nothing(tmp_path) -> None:
    """Aucun examen n'a eu lieu : tout reste proposable."""
    from hotel_pipeline import cli
    from hotel_pipeline.workspace import Workspace

    workspace = Workspace("essai", root=tmp_path)
    workspace.path("01_sources").mkdir(parents=True, exist_ok=True)
    manifeste = CandidateManifest(
        hotel_id="essai", candidates=[_candidat("cand-A")], scope=CIBLE,
    )

    filtré = cli._targeted_manifest(workspace, manifeste, CIBLE)

    assert len(filtré.candidates) == 1


# --- aucune pleine résolution avant aperçu établi ------------------------------


def test_no_full_resolution_is_plannable_before_a_preview(tmp_path) -> None:
    """Une acquisition ciblée porte sur un besoin qu'aucune vue ne couvre.

    Engager la pleine résolution reviendrait à payer pour une image dont
    personne n'a vérifié qu'elle montre ce qu'on cherche.
    """
    from hotel_pipeline import cli
    from hotel_pipeline.workspace import Workspace

    workspace = Workspace("essai", root=tmp_path)
    workspace.path("01_sources").mkdir(parents=True, exist_ok=True)
    manifeste = CandidateManifest(
        hotel_id="essai",
        candidates=[_candidat("cand-A")],
        # La recherche l'avait jugée éligible à la pleine résolution. Les
        # niveaux étant exclusifs, elle n'est pas simultanément « preview ».
        eligible_for_full_acquisition=["cand-A"],
        scope=CIBLE,
    )

    filtré = cli._targeted_manifest(workspace, manifeste, CIBLE)

    assert filtré.eligible_for_full_acquisition == [], (
        "l'aperçu et son examen humain sont le seul chemin vers l'acquisition "
        "complète d'une vue ciblée"
    )
    assert [c.candidate_id for c in filtré.candidates] == ["cand-A"], (
        "la vue reste au manifeste : l'aperçu est précisément ce qu'on veut "
        "obtenir d'elle"
    )


def test_the_excluded_candidates_leave_no_dangling_recommendation(tmp_path) -> None:
    """Recommander ce qui n'est plus au manifeste rendrait la recommandation
    invérifiable — et le schéma le refuse."""
    from hotel_pipeline import cli

    workspace = _log(
        tmp_path, _réfutation("cand-A", "obligation:ACCESS_ROAD_MAIN")
    )
    manifeste = CandidateManifest(
        hotel_id="essai",
        candidates=[_candidat("cand-A"), _candidat("cand-B")],
        recommended_for_preview=["cand-A", "cand-B"],
        recommended_for_enrichment=["cand-A"],
        scope=CIBLE,
    )

    filtré = cli._targeted_manifest(workspace, manifeste, CIBLE)

    assert filtré.recommended_for_preview == ["cand-B"]
    assert filtré.recommended_for_enrichment == []
    # Le manifeste doit rester valide au sens du schéma.
    CandidateManifest.model_validate(json.loads(filtré.model_dump_json()))
