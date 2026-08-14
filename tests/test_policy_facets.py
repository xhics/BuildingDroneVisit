"""Péremption par facettes de politique.

`policy_digest` bouge dès qu'un seuil change, où qu'il soit : la prendre pour
une dépendance périmerait le nuage LiDAR parce qu'une ouverture sectorielle a
bougé. Ce qui est éprouvé ici, ce sont les **non-péremptions** autant que les
péremptions — un contrat qui périme tout ne dit rien.
"""

from __future__ import annotations

import pytest

from hotel_pipeline import policy_facets as facets
from hotel_pipeline.policy_facets import (
    CONSUMERS,
    FACET_FIELDS,
    UNSCOPED_FIELDS,
    Facet,
    dependency_digests,
    facet_digest,
    stale_facets,
)
from hotel_pipeline.provenance import policy_digest
from hotel_pipeline.schemas import DEFAULT_POLICY, PipelinePolicy


def altered(**changes) -> PipelinePolicy:
    """Une politique dont un seul seuil a bougé."""
    policy = DEFAULT_POLICY.model_copy(deep=True)
    for path, value in changes.items():
        target = policy
        parts = path.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)
    return policy


def moved(production: str, **changes) -> list[str]:
    """Facettes périmées pour cette production, après ce changement."""
    before = dependency_digests(DEFAULT_POLICY, production)
    return stale_facets(before, altered(**changes), production)


# --- ce qui périme -------------------------------------------------------------


def test_the_sector_threshold_stales_evaluations_and_plans() -> None:
    change = {"geometry.sector_observer_half_width_deg": 45.0}

    assert moved("CandidateEvaluation", **change)
    assert moved("AcquisitionPlan", **change)


def test_the_viewpoint_separation_stales_the_plan() -> None:
    assert moved("AcquisitionPlan", **{"geometry.viewpoint_separation_m": 25.0})


def test_the_collection_radius_stales_the_candidates() -> None:
    """Puis le plan, par l'empreinte du manifeste — non par la politique.

    La dépendance transitive n'est pas répétée : le plan cite l'empreinte du
    manifeste de candidats, qui a changé. La redéclarer ici créerait deux
    chemins pour une même péremption, et ils divergeraient.
    """
    change = {"collection.radius_m": 900}

    assert moved("CandidateManifest", **change)
    assert moved("AcquisitionPlan", **change) == []


def test_a_visibility_setting_stales_the_run_and_the_plan() -> None:
    change = {"visibility.max_angular_step_deg": 1.0}

    assert moved("VisibilityRun", **change)
    assert moved("AcquisitionPlan", **change)


def test_an_adjacency_threshold_stales_the_capture_geometry() -> None:
    assert moved("CaptureGeometryManifest", **{"geometry.adjacency_max_m": 45.0})


def test_adjacency_and_framing_are_two_facets_despite_one_section() -> None:
    """La section `geometry` porte deux questions sans rapport.

    L'adjacence décide **quel bâtiment** est la cible ; l'ouverture sectorielle
    décide **quel candidat** sert quel besoin. Les regrouper — ce que la
    section invite à faire — ferait périmer une résolution de bâtiment parce
    qu'un cadrage a changé, et un plan parce qu'un stationnement a été rattaché
    autrement.
    """
    change = {"geometry.adjacency_max_m": 45.0}

    assert moved("CaptureGeometryManifest", **change)
    assert moved("CandidateEvaluation", **change) == []
    assert moved("AcquisitionPlan", **change) == []

    # Et réciproquement.
    framing = {"geometry.sector_observer_half_width_deg": 30.0}
    assert moved("CandidateEvaluation", **framing)
    assert moved("CaptureGeometryManifest", **framing) == []


def test_no_field_belongs_to_two_facets() -> None:
    """Un champ partagé périmerait des productions sans rapport entre elles."""
    seen: dict[str, str] = {}
    for facet, paths in FACET_FIELDS.items():
        for path in paths:
            assert path not in seen, (
                f"{path} figure dans {seen.get(path)} et {facet.value}"
            )
            seen[path] = facet.value


# --- ce qui ne périme pas, et c'est le cœur du contrat ------------------------


def test_a_qualification_threshold_never_stales_the_rasters() -> None:
    """Les seuils jugent la dérivation ; ils ne la produisent pas."""
    tightened = DEFAULT_POLICY.model_copy(deep=True)
    tightened.qualification.roofline.min_roof_observed = 0.99

    before = dependency_digests(DEFAULT_POLICY, "DerivedRaster")

    assert stale_facets(before, tightened, "DerivedRaster") == []
    # La qualification, elle, est bien périmée : c'est elle qui les lit.
    assert stale_facets(
        dependency_digests(DEFAULT_POLICY, "QualificationReport"),
        tightened, "QualificationReport",
    )


def test_a_terrain_threshold_stales_neither_candidates_nor_photo_visibility() -> None:
    change = {"terrain.ring_m": 35.0}

    assert moved("CandidateManifest", **change) == []
    assert moved("CandidateEvaluation", **change) == []
    assert moved("VisibilityRun", **change) == []
    # Mais bien la dérivation, qui en dépend.
    assert moved("DerivedRaster", **change)


def test_the_sector_threshold_never_stales_laz_dtm_or_roofline() -> None:
    """Le cas explicite : une ouverture sectorielle ne touche pas le sol."""
    change = {"geometry.sector_observer_half_width_deg": 30.0}

    assert moved("AcquiredLaz", **change) == []
    assert moved("DerivedRaster", **change) == []
    assert moved("QualificationReport", **change) == []


def test_an_acquired_file_depends_on_no_policy_facet() -> None:
    """Un fichier est identifié par son empreinte, pas par un seuil."""
    assert CONSUMERS["AcquiredLaz"] == ()
    assert dependency_digests(DEFAULT_POLICY, "AcquiredLaz") == {}
    assert dependency_digests(DEFAULT_POLICY, "AcquiredImage") == {}


def test_a_classification_threshold_stales_only_the_classification() -> None:
    change = {"model.subject_accept": 0.7}

    assert moved("ClassificationReport", **change)
    assert moved("CandidateEvaluation", **change) == []
    assert moved("DerivedRaster", **change) == []


def test_a_calibration_identifier_stales_nothing() -> None:
    """Nommer la campagne d'où viennent des seuils ne change aucun seuil."""
    renamed = DEFAULT_POLICY.model_copy(deep=True)
    renamed.model.calibration_id = "campagne-2027"
    renamed.model.calibrated_on_sites = 4

    for production in CONSUMERS:
        before = dependency_digests(DEFAULT_POLICY, production)
        assert stale_facets(before, renamed, production) == [], production


# --- les deux niveaux restent distincts ---------------------------------------


def test_the_full_digest_moves_where_a_facet_does_not() -> None:
    """C'est toute la raison d'être des deux niveaux."""
    tweaked = altered(**{"terrain.ring_m": 35.0})

    assert policy_digest(tweaked) != policy_digest(DEFAULT_POLICY)
    assert stale_facets(
        dependency_digests(DEFAULT_POLICY, "CandidateManifest"),
        tweaked, "CandidateManifest",
    ) == []


def test_every_report_still_carries_the_full_policy_digest() -> None:
    """La provenance reste complète : les facettes ne la remplacent pas."""
    described = facets.describe(DEFAULT_POLICY)

    assert described["policy_digest"] == policy_digest(DEFAULT_POLICY)
    assert set(described["facets"]) == {f.value for f in Facet}


def test_this_commit_changes_no_policy_value() -> None:
    """Décrire des dépendances ne règle rien : l'empreinte doit être stable."""
    from pathlib import Path

    path = Path("work/welcominns-boucherville/00_manifest/pipeline_policy.json")
    if not path.is_file():  # pragma: no cover — dépend du corpus local
        pytest.skip("corpus du pilote absent")

    loaded = PipelinePolicy.model_validate_json(path.read_text("utf-8"))

    assert policy_digest(loaded) == "9275a7e32eeb0431"


# --- la divergence se voit avant toute mutation -------------------------------


def test_a_missing_dependency_is_reported_not_assumed_valid() -> None:
    """Un artefact ancien reste lisible, sans passer pour à jour.

    Il a été écrit avant que la facette soit déclarée : affirmer qu'il la
    respecte serait lui prêter une garantie qu'il n'a jamais eue.
    """
    problems = stale_facets({}, DEFAULT_POLICY, "AcquisitionPlan")

    assert len(problems) == 2
    assert all("absente de la production" in problem for problem in problems)


def test_an_unchanged_policy_stales_nothing() -> None:
    for production in CONSUMERS:
        before = dependency_digests(DEFAULT_POLICY, production)
        assert stale_facets(before, DEFAULT_POLICY, production) == [], production


def test_an_undeclared_production_is_an_error_not_an_empty_set() -> None:
    """Le silence ferait d'un oubli une absence de dépendance."""
    with pytest.raises(KeyError, match="sans dépendances déclarées"):
        dependency_digests(DEFAULT_POLICY, "ProductionInconnue")


# --- aucun champ de politique n'échappe au contrat ----------------------------


def test_every_policy_field_belongs_to_a_facet_or_is_named_unscoped() -> None:
    """Un champ oublié ne périmerait rien, en silence."""
    scoped = {path for paths in FACET_FIELDS.values() for path in paths}
    known = scoped | set(UNSCOPED_FIELDS)

    missing = sorted(_leaf_paths(DEFAULT_POLICY) - known)

    assert missing == [], f"champs de politique sans facette : {missing}"


def _leaf_paths(model, prefix: str = "") -> set[str]:  # noqa: ANN001
    """Chemins pointés des champs, en s'arrêtant aux sous-tables déclarées."""
    paths: set[str] = set()
    for name in type(model).model_fields:
        path = f"{prefix}{name}"
        value = getattr(model, name)
        # Une sous-table citée telle quelle dans une facette est une feuille :
        # `qualification.terrain` y figure entière, et ses seuils avec elle.
        if hasattr(type(value), "model_fields") and path not in _DECLARED_SUBTABLES:
            paths |= _leaf_paths(value, prefix=f"{path}.")
        else:
            paths.add(path)
    return paths


_DECLARED_SUBTABLES = frozenset(
    {path for paths in FACET_FIELDS.values() for path in paths}
)


def test_no_facet_is_empty() -> None:
    """Une facette sans champ ne périmerait jamais rien."""
    for facet in Facet:
        assert FACET_FIELDS.get(facet), facet.value


def test_every_facet_has_at_least_one_consumer() -> None:
    """Une facette que personne ne lit n'aurait aucune raison d'exister."""
    consumed = {facet for facets_ in CONSUMERS.values() for facet in facets_}

    assert set(Facet) == consumed


# --- les deux niveaux atteignent le disque -----------------------------------


def test_a_written_report_carries_both_levels(tmp_path, monkeypatch) -> None:
    """Décrire les dépendances ne sert à rien si elles ne sont pas publiées."""
    import json

    from typer.testing import CliRunner

    from hotel_pipeline.cli import app
    from hotel_pipeline.schemas.acquisition import CandidateManifest, CaptureDemandManifest
    from hotel_pipeline.workspace import Workspace

    monkeypatch.setenv("HOTEL_PIPELINE_WORK", str(tmp_path / "work"))
    monkeypatch.setenv("HOTEL_PIPELINE_PROFILES", str(tmp_path / "profiles"))
    monkeypatch.setenv("HOTEL_PIPELINE_OFFLINE", "1")
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    runner.invoke(app, [
        "init", "hotel-test", "--address", "1 rue Test", "--name", "Hôtel Test",
        "--country", "CA", "--timezone", "America/Toronto", "--ocr-language", "fr",
        "--lat", "45.573", "--lon", "-73.443",
    ])

    from tests.test_plan import candidate, demand  # noqa: PLC0415

    workspace = Workspace("hotel-test")
    workspace.write_json(
        "01_sources/capture_demands.json",
        json.loads(
            CaptureDemandManifest(hotel_id="hotel-test", demands=[demand()])
            .model_dump_json()
        ),
    )
    workspace.write_json(
        "01_sources/candidates_20260814T000000000000Z.json",
        json.loads(
            CandidateManifest(hotel_id="hotel-test", candidates=[candidate()])
            .model_dump_json()
        ),
    )

    assert runner.invoke(app, ["assets", "plan", "hotel-test"]).exit_code == 0

    written = sorted(workspace.path("01_sources").glob("plan_report_*.json"))
    report = json.loads(written[-1].read_text("utf-8"))

    # L'empreinte complète, pour la provenance.
    assert report["provenance"]["policy_digest"]
    # Les facettes lues, pour la péremption — et **seulement** celles-là.
    dependencies = report["policy_dependency_digests"]
    assert set(dependencies) == {"candidate_geometry", "visibility"}
    assert "terrain_derivation" not in dependencies


def test_every_declared_production_is_actually_published() -> None:
    """Déclarer une dépendance sans jamais l'écrire ne périme rien.

    Six productions avaient leurs facettes déclarées et testées, sans que
    leurs commandes les publient : le contrat était exact et inopérant.
    """
    import re
    from pathlib import Path

    source = Path("src/hotel_pipeline/cli.py").read_text("utf-8")
    published = set(re.findall(r'production="([A-Za-z]+)"', source))

    # `AcquiredImage` est écrit par le rapport d'acquisition d'images ;
    # les autres se retrouvent dans leurs commandes respectives.
    missing = sorted(set(CONSUMERS) - published)

    assert missing == [], f"productions déclarées mais jamais publiées : {missing}"
