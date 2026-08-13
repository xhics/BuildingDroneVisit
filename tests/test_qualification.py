"""Qualification provisoire des objets dérivés (Lot 1B §9).

Trois familles y sont éprouvées, dans cet ordre :

- les seuils décident bien sur le **pire** essai, non sur une moyenne ;
- la plus grande composante observée est mesurée, non déduite d'un total ;
- un artefact remplacé fait retomber l'objet en `stale` **sans** effacer la
  décision antérieure.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pytest

from hotel_pipeline.geo import qualify
from hotel_pipeline.schemas import (
    DerivedArtifact,
    GeoSourceProvenance,
    ObjectState,
    PipelinePolicy,
    SiteManifest,
    SiteObject,
)

SOURCE_ID = "lidar-quebec-23_3095048F08_DC"


def policy() -> PipelinePolicy:
    return PipelinePolicy()


def metrics(**overrides) -> dict:
    """Les mesures réelles de la dérivation du WelcomINNS, sauf indication."""
    base = {
        "footprint_cells": 7341,
        "coverage": {
            "dtm_defined": 1.0,
            "roof_observed": 0.969,
            "class1_candidates": 0.012,
            "ndsm_valid": 0.969,
        },
        "roof_density_per_m2": 25.5,
        "roof_gaps": {
            "main_observed_cells": 7000,
            "main_observed_fraction": 0.9536,
            "missing_cells": 225,
            "missing_components": 18,
            "largest_gap_cells": 140,
            "largest_gap_grid_m2": 35.0,
            "largest_gap_fraction": 0.019,
        },
        "tin_vs_idw": {"mae_m": 0.0454},
        "extrapolation_rejected": {"fraction_of_footprint": 0.0},
        "support_distance_in_footprint": {"p50_m": 4.5, "p95_m": 9.66, "max_m": 12.5},
        "block_validation": {"rmse_m": 0.1607},
        "pseudo_footprint_validation": {
            "search_area_within_tile": True,
            "rejected_candidates": [],
            "trials": [
                {"rmse_m": 0.0676, "p95_m": 0.1445, "bias_m": 0.01},
                {"rmse_m": 0.1208, "p95_m": 0.2591, "bias_m": -0.02},
                {"rmse_m": 0.2463, "p95_m": 0.5272, "bias_m": 0.04},
            ],
        },
        "height_statistics": {"count": 7116, "median_m": 10.26, "negative_cells": 0},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return base


# --- seuils du terrain ------------------------------------------------------


def test_terrain_passes_on_real_measurements() -> None:
    verdict = qualify.evaluate_terrain(metrics(), policy().qualification.terrain)
    assert verdict.passed, verdict.failures
    assert verdict.confidence == "medium"


def test_worst_trial_decides_not_the_mean() -> None:
    """Un essai médiocre ne doit pas être dilué par deux bons.

    Moyenne des RMSE : 0,29 m — sous le seuil de 0,50 m. Le pire essai : 0,75 m.
    """
    bad = metrics(
        pseudo_footprint_validation={
            "trials": [
                {"rmse_m": 0.06, "p95_m": 0.14, "bias_m": 0.01},
                {"rmse_m": 0.07, "p95_m": 0.15, "bias_m": 0.01},
                {"rmse_m": 0.75, "p95_m": 0.95, "bias_m": 0.02},
            ],
            "search_area_within_tile": True,
            "rejected_candidates": [],
        }
    )
    assert np.mean([0.06, 0.07, 0.75]) < policy().qualification.terrain.max_worst_trial_rmse_m

    verdict = qualify.evaluate_terrain(bad, policy().qualification.terrain)
    assert not verdict.passed
    assert "worst_trial_rmse_m" in verdict.failures


def test_search_area_outside_tile_blocks_qualification() -> None:
    """Une zone de recherche débordant la tuile invalide les essais eux-mêmes."""
    verdict = qualify.evaluate_terrain(
        metrics(pseudo_footprint_validation={"search_area_within_tile": False}),
        policy().qualification.terrain,
    )
    assert "search_area_within_tile" in verdict.failures


def test_too_few_trials_blocks_qualification() -> None:
    verdict = qualify.evaluate_terrain(
        metrics(
            pseudo_footprint_validation={
                "trials": [{"rmse_m": 0.06, "p95_m": 0.14, "bias_m": 0.01}]
            }
        ),
        policy().qualification.terrain,
    )
    assert "accepted_trials" in verdict.failures


def test_terrain_reservations_always_state_the_interpolation() -> None:
    verdict = qualify.evaluate_terrain(metrics(), policy().qualification.terrain)
    joined = " ".join(verdict.reservations)
    assert "aucune cellule de terrain mesurée" in joined
    assert "un seul site" in joined
    # La validation par blocs est structurellement optimiste : elle informe,
    # elle ne décide pas.
    assert "diagnostique" in joined
    assert not any(c.name.startswith("block") for c in verdict.criteria)


# --- composantes de la toiture ---------------------------------------------


def test_main_component_is_measured_not_derived_from_the_total() -> None:
    """Deux moitiés observées séparées ne valent pas une surface continue.

    Le total observé est le même dans les deux cas ; la composante principale,
    non. C'est elle qui dit ce qu'une caméra peut suivre d'un seul tenant.
    """
    footprint = np.ones((10, 10), dtype=bool)
    split = np.ones((10, 10), dtype=bool)
    split[:, 5] = False  # une lacune traversante coupe la toiture en deux

    gaps = qualify.roof_gaps(split, footprint, cell_m=0.5)
    assert gaps["main_observed_fraction"] == pytest.approx(0.5)
    assert gaps["missing_cells"] == 10
    assert gaps["missing_components"] == 1

    scattered = np.ones((10, 10), dtype=bool)
    scattered[[0, 0, 4, 4, 8], [0, 4, 0, 8, 4]] = False  # cinq trous isolés
    dispersed = qualify.roof_gaps(scattered, footprint, cell_m=0.5)
    assert dispersed["missing_cells"] == 5
    assert dispersed["missing_components"] == 5
    assert dispersed["main_observed_fraction"] == pytest.approx(0.95)


def test_connectivity_is_taken_against_each_measure() -> None:
    """Deux cellules en diagonale : une seule lacune, mais pas une surface.

    Prendre la connexité dans l'autre sens flatterait les deux mesures — une
    toiture en damier passerait pour continue, et ses trous pour ponctuels.
    """
    footprint = np.ones((4, 4), dtype=bool)
    observed = np.zeros((4, 4), dtype=bool)
    observed[[0, 1, 2, 3], [0, 1, 2, 3]] = True  # une diagonale observée

    gaps = qualify.roof_gaps(observed, footprint, cell_m=1.0)
    # Quatre cellules observées en diagonale ne forment pas une surface.
    assert gaps["main_observed_cells"] == 1
    # Les douze cellules manquantes, elles, communiquent en diagonale de part
    # et d'autre de la diagonale observée : une seule lacune.
    assert gaps["missing_components"] == 1
    assert gaps["largest_gap_cells"] == 12


def test_split_roof_fails_even_with_high_total_coverage() -> None:
    """96,9 % de cellules vues, mais en deux surfaces : refus."""
    verdict = qualify.evaluate_roofline(
        metrics(roof_gaps={"main_observed_fraction": 0.49}),
        policy().qualification.roofline,
        terrain_passed=True,
    )
    assert not verdict.passed
    assert verdict.failures == ["main_observed_component"]


def test_unmeasured_main_component_fails_closed() -> None:
    """Une mesure absente ne vaut pas une mesure réussie."""
    payload = metrics()
    del payload["roof_gaps"]
    verdict = qualify.evaluate_roofline(
        payload, policy().qualification.roofline, terrain_passed=True
    )
    assert "main_observed_component" in verdict.failures


def test_roofline_requires_qualified_terrain() -> None:
    """Chaque hauteur est une différence au terrain : sans terrain, pas de nDSM."""
    verdict = qualify.evaluate_roofline(
        metrics(), policy().qualification.roofline, terrain_passed=False
    )
    assert not verdict.passed
    assert verdict.failures == ["qualified_terrain"]


def test_a_report_written_before_the_renaming_is_still_readable() -> None:
    """`largest_gap_m2` a été renommé en aire de grille ; les rapports déjà
    publiés restent jugeables, sans quoi une correction de vocabulaire
    invaliderait des dérivations correctes."""
    old = metrics()
    old["roof_gaps"]["largest_gap_m2"] = old["roof_gaps"].pop("largest_gap_grid_m2")
    verdict = qualify.evaluate_roofline(
        old, policy().qualification.roofline, terrain_passed=True
    )
    assert "35.0 m² d'aire de grille" in " ".join(verdict.reservations)


def test_gap_zones_are_recorded_as_forbidden_to_close_ups() -> None:
    verdict = qualify.evaluate_roofline(
        metrics(), policy().qualification.roofline, terrain_passed=True
    )
    assert verdict.passed, verdict.failures
    joined = " ".join(verdict.reservations)
    assert "35.0 m² d'aire de grille" in joined
    assert "cellules entières non découpées par le polygone" in joined
    assert "plans rapprochés" in joined
    assert "18 composante(s)" in joined


def test_density_is_measured_over_the_footprint() -> None:
    verdict = qualify.evaluate_roofline(
        metrics(roof_density_per_m2=4.0),
        policy().qualification.roofline,
        terrain_passed=True,
    )
    assert "class6_density_per_m2" in verdict.failures


# --- application au manifeste ----------------------------------------------


def source() -> GeoSourceProvenance:
    return GeoSourceProvenance(
        source_id=SOURCE_ID,
        dataset="Données LiDAR du Québec",
        vintage="2023",
        tile_id="23_3095048F08_DC",
        crs_horizontal="EPSG:2950",
        crs_vertical="CGVD 1928",
        carries_elevation=True,
        licence="CC BY 4.0",
        retrieved_at=datetime(2026, 8, 12, tzinfo=timezone.utc),
        file_digest="fc6407b2",
    )


def artifact(artifact_id: str, role: str, **overrides) -> DerivedArtifact:
    fields = dict(
        artifact_id=artifact_id,
        role=role,
        path=f"06_geo/derived/{artifact_id}.tif",
        format="GeoTIFF",
        sha256="a" * 64,
        crs_horizontal="EPSG:2950",
        crs_vertical="CGVD 1928",
        resolution_m=0.5,
        algorithm_id="tin-linear-v1",
        measured_fraction=0.0,
        interpolated_fraction=1.0,
        coverage_domain="footprint",
        derived_from_sources=[SOURCE_ID],
    )
    fields.update(overrides)
    return DerivedArtifact(**fields)


def manifest(artifacts: list[DerivedArtifact]) -> SiteManifest:
    return SiteManifest(
        hotel_id="welcominns-boucherville",
        geo_sources=[source()],
        artifacts=artifacts,
        objects=[
            SiteObject(object_id="TERRAIN_MAIN", kind="TERRAIN_MAIN"),
            SiteObject(object_id="ROOFLINE_MAIN", kind="ROOFLINE_MAIN"),
        ],
    )


def test_qualification_never_writes_confirmed() -> None:
    site = manifest([artifact("dtm@r1", "dtm"), artifact("dsm@r1", "dsm_roof"),
                     artifact("ndsm@r1", "ndsm")])
    mapping = qualify.select_artifacts(site)
    report = qualify.report(metrics(), policy(), digest="d1", artifacts=[])

    qualified = qualify.apply(site, report, mapping)

    assert sorted(qualified) == ["ROOFLINE_MAIN", "TERRAIN_MAIN"]
    for obj in site.objects:
        assert obj.state is ObjectState.INFERRED
        assert obj.qualification_rationale
        assert obj.qualification_confidence == "medium"
        assert obj.qualification_reservations
    assert site.objects[0].artifact_ids == ["dtm@r1"]
    assert site.objects[1].artifact_ids == ["dsm@r1", "ndsm@r1"]


def test_only_active_artifacts_are_selected() -> None:
    site = manifest(
        [
            artifact("dtm@r2", "dtm"),
            artifact("dtm@r1", "dtm", status="superseded", superseded_by="dtm@r2"),
            artifact("dsm@r1", "dsm_roof"),
            artifact("ndsm@r1", "ndsm"),
        ]
    )
    assert qualify.select_artifacts(site)["TERRAIN_MAIN"] == ["dtm@r2"]


def test_failed_object_keeps_no_artifact_and_says_which_threshold() -> None:
    site = manifest([artifact("dtm@r1", "dtm")])
    report = qualify.report(
        metrics(coverage={"dtm_defined": 0.40}), policy(), digest="d1", artifacts=[]
    )
    qualified = qualify.apply(site, report, qualify.select_artifacts(site))

    assert qualified == []
    terrain = site.objects[0]
    assert terrain.state is ObjectState.UNRESOLVED
    assert terrain.artifact_ids == []
    assert "dtm_defined" in terrain.unresolved_reason
    # Le terrain manque, donc la toiture aussi : la hauteur en dépend.
    assert site.objects[1].state is ObjectState.UNRESOLVED


def test_object_without_active_artifact_is_not_qualified() -> None:
    site = manifest([artifact("dsm@r1", "dsm_roof"), artifact("ndsm@r1", "ndsm")])
    report = qualify.report(metrics(), policy(), digest="d1", artifacts=[])
    qualified = qualify.apply(site, report, qualify.select_artifacts(site))

    assert "TERRAIN_MAIN" not in qualified
    assert site.objects[0].unresolved_reason == "aucun artefact actif ne porte cet objet"


# --- péremption -------------------------------------------------------------


def test_superseded_artifact_makes_the_object_stale_and_keeps_the_decision() -> None:
    site = manifest([artifact("dtm@r1", "dtm"), artifact("dsm@r1", "dsm_roof"),
                     artifact("ndsm@r1", "ndsm")])
    report = qualify.report(metrics(), policy(), digest="d1", artifacts=[])
    qualify.apply(site, report, qualify.select_artifacts(site))
    rationale = site.objects[0].qualification_rationale

    # Une nouvelle dérivation remplace le DTM cité.
    site.artifacts.append(artifact("dtm@r2", "dtm"))
    site.artifacts[0] = site.artifacts[0].model_copy(
        update={"status": "superseded", "superseded_by": "dtm@r2"}
    )

    marked = qualify.mark_stale(site)

    assert marked == ["TERRAIN_MAIN"]
    terrain = site.objects[0]
    assert terrain.state is ObjectState.STALE
    assert terrain.previous_state is ObjectState.INFERRED
    assert terrain.qualification_rationale == rationale
    assert "dtm@r1" in terrain.unresolved_reason
    # Le manifeste doit rester valide malgré la référence à un artefact remplacé.
    SiteManifest.model_validate(site.model_dump())


def test_stale_is_idempotent() -> None:
    site = manifest([artifact("dtm@r2", "dtm"), artifact("dsm@r1", "dsm_roof"),
                     artifact("ndsm@r1", "ndsm")])
    qualify.apply(site, qualify.report(metrics(), policy(), "d1", []),
                  qualify.select_artifacts(site))
    site.artifacts.append(artifact("dtm@r3", "dtm"))
    site.artifacts[0] = site.artifacts[0].model_copy(
        update={"status": "superseded", "superseded_by": "dtm@r3"}
    )

    assert qualify.mark_stale(site) == ["TERRAIN_MAIN"]
    # Un second passage ne doit pas écraser la décision conservée par « stale ».
    assert qualify.mark_stale(site) == []
    assert site.objects[0].previous_state is ObjectState.INFERRED


def test_inactive_reference_outside_stale_is_still_refused() -> None:
    """La tolérance ne vaut que pour `stale`, qui la déclare."""
    site = manifest([artifact("dtm@r2", "dtm"),
                     artifact("dtm@r1", "dtm", status="superseded", superseded_by="dtm@r2")])
    payload = site.model_dump()
    payload["objects"][0]["state"] = "inferred"
    payload["objects"][0]["artifact_ids"] = ["dtm@r1"]

    with pytest.raises(ValueError, match="artefacts non actifs"):
        SiteManifest.model_validate(payload)


def test_stale_requires_a_preserved_decision() -> None:
    with pytest.raises(ValueError, match="sans décision antérieure"):
        SiteObject(object_id="TERRAIN_MAIN", kind="TERRAIN_MAIN", state=ObjectState.STALE)


# --- politique gelée --------------------------------------------------------


def test_frozen_policy_missing_a_section_is_reported(tmp_path) -> None:
    """Une politique antérieure à la section se relit sans erreur — et ment.

    Pydantic comble les manques avec les valeurs du code, tandis que le fichier
    continue d'annoncer son ancienne version : le rapport afficherait des seuils
    qui ne figurent nulle part sur le disque. `geo qualify` s'y refuse.
    """
    from hotel_pipeline.context import PipelineContext

    frozen = json.loads(PipelinePolicy().model_dump_json())
    del frozen["qualification"]
    frozen["version"] = "1.1.0"
    path = tmp_path / "pipeline_policy.json"
    path.write_text(json.dumps(frozen), encoding="utf-8")

    context = PipelineContext.load(policy_path=path)

    assert context.policy.version == "1.1.0"
    assert context.policy_defaults_applied == ("qualification",)
    # Les seuils existent bel et bien en mémoire : c'est précisément le piège.
    assert context.policy.qualification.status == "provisional"


def test_complete_policy_reports_nothing_filled(tmp_path) -> None:
    path = tmp_path / "pipeline_policy.json"
    path.write_text(PipelinePolicy().model_dump_json(), encoding="utf-8")

    from hotel_pipeline.context import PipelineContext

    assert PipelineContext.load(policy_path=path).policy_defaults_applied == ()


def test_a_missing_nested_threshold_is_detected_too(tmp_path) -> None:
    """Le cas probable n'est pas la section absente, mais le seuil ajouté.

    Un fichier qui contient bien `qualification.terrain` mais pas l'un de ses
    seuils paraît complet ; Pydantic le remplit depuis le code, et le contrôle
    de premier niveau ne voit rien.
    """
    from hotel_pipeline.context import PipelineContext

    frozen = json.loads(PipelinePolicy().model_dump_json())
    del frozen["qualification"]["terrain"]["max_worst_trial_rmse_m"]
    del frozen["model"]["subject_accept"]
    path = tmp_path / "pipeline_policy.json"
    path.write_text(json.dumps(frozen), encoding="utf-8")

    context = PipelineContext.load(policy_path=path)

    assert "qualification.terrain.max_worst_trial_rmse_m" in context.policy_defaults_applied
    assert "model.subject_accept" in context.policy_defaults_applied
    # `implicit_under` isole ce qui concerne la qualification, sans y mêler le
    # reste de la politique.
    assert context.implicit_under("qualification") == (
        "qualification.terrain.max_worst_trial_rmse_m",
    )


# --- paramètres effectifs de l'artefact -------------------------------------


def synthetic_tile(path) -> tuple[object, dict]:
    """Un nuage minimal mais réaliste : sol dense au pourtour, toit plat.

    Les cellules font 2 m dans ces essais. À 0,5 m, la densité synthétique ne
    permettrait aucune couverture de sol crédible — c'était le piège d'une
    fixture antérieure, et le seuil, lui, était correct.
    """
    import laspy
    from shapely.geometry import box

    rng = np.random.default_rng(0)
    ground, roof = 120_000, 40_000
    gx = rng.uniform(900, 1160, ground)
    gy = rng.uniform(900, 1140, ground)
    gz = 10 + 0.01 * (gx - 900)
    rx = rng.uniform(1000, 1060, roof)
    ry = rng.uniform(1000, 1040, roof)

    x = np.concatenate([gx, rx])
    y = np.concatenate([gy, ry])
    z = np.concatenate([gz, np.full(roof, 22.0)])

    header = laspy.LasHeader(version="1.4", point_format=6)
    header.offsets = [x.min(), y.min(), z.min()]
    header.scales = [0.01, 0.01, 0.01]
    las = laspy.LasData(header)
    las.x, las.y, las.z = x, y, z
    las.classification = np.concatenate(
        [np.full(ground, 2), np.full(roof, 6)]
    ).astype(np.uint8)
    las.write(str(path))

    return box(1000, 1000, 1060, 1040), {
        "minx": 900, "miny": 900, "maxx": 1160, "maxy": 1140
    }


def test_artifacts_declare_the_parameters_actually_used(tmp_path) -> None:
    """L'anneau inscrit doit être celui du calcul, non celui du module.

    Avec un anneau porté à 25 m, la dérivation cherchait bien ses appuis à
    25 m tandis que l'artefact déclarait la constante du module — 20 m. Un
    raster décrit par de faux paramètres n'est pas reproductible, et rien dans
    le manifeste ne l'aurait démenti.
    """
    from hotel_pipeline.geo.derive import derive

    tuned = PipelinePolicy()
    tuned.terrain.cell_m = 2.0
    tuned.terrain.ring_m = 25.0
    tuned.terrain.search_radius_m = 80.0

    footprint, bounds = synthetic_tile(tmp_path / "tile.las")
    result = derive(
        tmp_path / "tile.las", footprint, tmp_path / "staging",
        crs="EPSG:2950", crs_vertical="CGVD 1928", source_id="s",
        policy=tuned, laz_bounds=bounds,
    )

    parameters = result.artifacts[0].parameters
    assert parameters["ring_m"] == "25.0"
    assert parameters["cell_m"] == "2.0"
    assert parameters["search_radius_m"] == "80.0"
    assert parameters["policy_version"] == tuned.version
    assert all(a.parameters == parameters for a in result.artifacts)


def test_artifacts_without_effective_parameters_are_refused() -> None:
    from hotel_pipeline.geo.derive import _artifacts

    with pytest.raises(ValueError, match="sans paramètres effectifs"):
        _artifacts(None, None, "EPSG:2950", "CGVD 1928", "s", 1, None, None)


# --- lien entre le rapport jugé et la série active --------------------------


def declared(*artifacts: DerivedArtifact) -> dict:
    return {"artifacts": [json.loads(a.model_dump_json()) for a in artifacts]}


def test_matching_series_raises_no_objection() -> None:
    dtm, dsm, ndsm = artifact("dtm@r1", "dtm"), artifact("dsm@r1", "dsm_roof"), artifact("ndsm@r1", "ndsm")
    site = manifest([dtm, dsm, ndsm])
    assert qualify.check_series(site, declared(dtm, dsm, ndsm), "r1") == []


def test_report_of_another_run_is_refused() -> None:
    """Le dernier rapport et la série active peuvent diverger sans le dire.

    Chacun est cohérent séparément : le rapport décrit une exécution complète,
    le manifeste une série entièrement active. Seul leur rapprochement révèle
    qu'ils ne parlent pas de la même production.
    """
    dtm, dsm, ndsm = artifact("dtm@r1", "dtm"), artifact("dsm@r1", "dsm_roof"), artifact("ndsm@r1", "ndsm")
    site = manifest([dtm, dsm, ndsm])
    other = declared(artifact("dtm@r2", "dtm"), artifact("dsm@r2", "dsm_roof"),
                     artifact("ndsm@r2", "ndsm"))

    problems = qualify.check_series(site, other, "r2")

    assert any("l'exécution r2" in p for p in problems)
    assert any("dtm@r1 ne figure pas dans le rapport jugé" in p for p in problems)


def test_mixed_run_suffixes_are_refused() -> None:
    dtm, dsm, ndsm = artifact("dtm@r1", "dtm"), artifact("dsm@r2", "dsm_roof"), artifact("ndsm@r2", "ndsm")
    site = manifest([dtm, dsm, ndsm])
    problems = qualify.check_series(site, declared(dtm, dsm, ndsm), None)
    assert any("plusieurs exécutions" in p for p in problems)


def test_two_active_artifacts_of_the_same_role_are_refused() -> None:
    """Deux DTM actifs : la citation serait arbitraire, donc irreproductible."""
    first, second = artifact("dtm@r1", "dtm"), artifact("dtm@r2", "dtm")
    site = manifest([first, second, artifact("dsm@r1", "dsm_roof"),
                     artifact("ndsm@r1", "ndsm")])
    problems = qualify.check_series(site, declared(first, second), "r1")
    assert any("2 artefact(s) actif(s) de rôle 'dtm'" in p for p in problems)


def test_no_active_artifact_of_a_required_role_is_refused() -> None:
    site = manifest([artifact("dsm@r1", "dsm_roof"), artifact("ndsm@r1", "ndsm")])
    problems = qualify.check_series(site, declared(artifact("dsm@r1", "dsm_roof")), "r1")
    assert any("0 artefact(s) actif(s) de rôle 'dtm'" in p for p in problems)


def test_digest_divergence_between_manifest_and_report_is_refused() -> None:
    """Même identifiant, même chemin, contenu différent : le pire des cas.

    Rien dans les noms ne le trahit ; seule l'empreinte le dit.
    """
    dtm = artifact("dtm@r1", "dtm")
    site = manifest([dtm, artifact("dsm@r1", "dsm_roof"), artifact("ndsm@r1", "ndsm")])
    tampered = declared(
        dtm.model_copy(update={"sha256": "b" * 64}),
        artifact("dsm@r1", "dsm_roof"),
        artifact("ndsm@r1", "ndsm"),
    )

    problems = qualify.check_series(site, tampered, "r1")
    assert any("empreintes divergentes" in p for p in problems)


# --- rapports versionnés ----------------------------------------------------


def test_report_name_carries_run_and_policy() -> None:
    """Une décision par exécution **et** par politique.

    Deux politiques appliquées aux mêmes mesures produisent deux décisions ;
    les écrire au même endroit effacerait la première, alors même que la
    dérivation et les artefacts qui la fondaient sont conservés.
    """
    first = qualify.report(metrics(), policy(), "d1", [], run_id="20260813T124251Z")

    stricter = policy()
    stricter.qualification.terrain.max_worst_trial_rmse_m = 0.10
    second = qualify.report(metrics(), stricter, "d1", [], run_id="20260813T124251Z")

    assert first.name.startswith("qualification_report_20260813T124251Z_")
    assert first.name != second.name
    assert not second.verdicts["TERRAIN_MAIN"].passed


def test_object_separates_the_two_digests() -> None:
    """`qualification_report` désignait en fait la dérivation : nom trompeur."""
    site = manifest([artifact("dtm@r1", "dtm"), artifact("dsm@r1", "dsm_roof"),
                     artifact("ndsm@r1", "ndsm")])
    report = qualify.report(metrics(), policy(), digest="deadbeef", artifacts=[], run_id="r1")

    qualify.apply(site, report, qualify.select_artifacts(site), report_digest="cafe1234")

    terrain = site.objects[0]
    assert terrain.qualification_report == report.name
    assert terrain.qualification_report_digest == "cafe1234"
    assert terrain.qualified_derivation_digest == "deadbeef"


# --- rapport ----------------------------------------------------------------


def test_report_carries_policy_and_derivation_digest() -> None:
    report = qualify.report(metrics(), policy(), digest="7f3a", artifacts=["dtm@r1"])
    payload = report.as_dict()

    assert payload["qualified_derivation_digest"] == "7f3a"
    assert payload["selected_artifacts"] == ["dtm@r1"]
    assert payload["policy"]["status"] == "provisional"
    assert payload["policy"]["intended_use"] == "visual_proxy_not_survey"
    assert payload["policy"]["calibrated_on_sites"] == 1

    terrain = payload["verdicts"]["TERRAIN_MAIN"]
    # Chaque seuil est écrit avec sa mesure : le rapport se relit sans le code.
    assert {c["criterion"] for c in terrain["criteria"]} >= {
        "dtm_defined", "worst_trial_rmse_m", "max_support_distance_m", "tin_idw_mae_m"
    }
    assert all(c["threshold"] and c["measured"] for c in terrain["criteria"])


# --- intégration : la CLI refuse une série incompatible ---------------------


def workspace_with(tmp_path, monkeypatch, artifacts, report_artifacts, run_id):
    """Un espace de travail où le rapport et la série sont posés à la main."""
    from typer.testing import CliRunner

    from hotel_pipeline.cli import app
    from hotel_pipeline.workspace import Workspace

    monkeypatch.setenv("HOTEL_PIPELINE_WORK", str(tmp_path / "work"))
    monkeypatch.setenv("HOTEL_PIPELINE_PROFILES", str(tmp_path / "profiles"))
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    runner.invoke(app, ["init", "hotel-test", "--address", "1 rue Test"])

    workspace = Workspace("hotel-test")
    site = manifest(artifacts)
    site.hotel_id = "hotel-test"
    workspace.write_site(site)
    workspace.policy_path.parent.mkdir(parents=True, exist_ok=True)
    workspace.policy_path.write_text(PipelinePolicy().model_dump_json(indent=2), "utf-8")
    workspace.write_json(
        f"06_geo/derivation_report_{run_id}.json",
        {"metrics": metrics(), "artifacts": [json.loads(a.model_dump_json())
                                             for a in report_artifacts]},
    )
    return runner, workspace


def test_cli_refuses_a_report_that_does_not_describe_the_active_series(
    tmp_path, monkeypatch
) -> None:
    """Le rapport décrit r2, le manifeste porte r1 : chacun est cohérent seul."""
    from hotel_pipeline.cli import app

    active = [artifact("dtm@r1", "dtm"), artifact("dsm@r1", "dsm_roof"),
              artifact("ndsm@r1", "ndsm")]
    described = [artifact("dtm@r2", "dtm"), artifact("dsm@r2", "dsm_roof"),
                 artifact("ndsm@r2", "ndsm")]
    runner, workspace = workspace_with(tmp_path, monkeypatch, active, described, "r2")

    result = runner.invoke(app, ["geo", "qualify", "hotel-test"])

    assert result.exit_code == 4
    assert "ne concordent pas" in result.output
    # Rien n'a été décidé, et aucun rapport n'a été publié.
    assert all(o.state is ObjectState.UNRESOLVED for o in workspace.read_site().objects)
    assert not list(workspace.path("06_geo").glob("qualification_report_*.json"))


def test_cli_publishes_one_report_per_run_and_policy(tmp_path, monkeypatch) -> None:
    from hotel_pipeline.cli import app

    series = [artifact("dtm@r1", "dtm"), artifact("dsm@r1", "dsm_roof"),
              artifact("ndsm@r1", "ndsm")]
    runner, workspace = workspace_with(tmp_path, monkeypatch, series, series, "r1")

    assert runner.invoke(app, ["geo", "qualify", "hotel-test"]).exit_code == 0

    published = list(workspace.path("06_geo").glob("qualification_report_*.json"))
    assert len(published) == 1
    assert published[0].name.startswith("qualification_report_r1_")

    terrain = next(o for o in workspace.read_site().objects if o.kind == "TERRAIN_MAIN")
    assert terrain.state is ObjectState.INFERRED
    assert terrain.qualification_report == published[0].name
    assert terrain.qualification_report_digest
    assert terrain.qualified_derivation_digest != terrain.qualification_report_digest

    # Une politique plus stricte publie une seconde décision, sans effacer la
    # première : les dérivations, elles, sont bien conservées.
    stricter = PipelinePolicy()
    stricter.qualification.terrain.max_worst_trial_rmse_m = 0.10
    workspace.policy_path.write_text(stricter.model_dump_json(indent=2), "utf-8")

    assert runner.invoke(app, ["geo", "qualify", "hotel-test"]).exit_code == 0
    assert len(list(workspace.path("06_geo").glob("qualification_report_*.json"))) == 2


def test_cli_refuses_an_implicit_threshold(tmp_path, monkeypatch) -> None:
    from hotel_pipeline.cli import app

    series = [artifact("dtm@r1", "dtm"), artifact("dsm@r1", "dsm_roof"),
              artifact("ndsm@r1", "ndsm")]
    runner, workspace = workspace_with(tmp_path, monkeypatch, series, series, "r1")

    frozen = json.loads(PipelinePolicy().model_dump_json())
    del frozen["qualification"]["roofline"]["min_main_component"]
    workspace.policy_path.write_text(json.dumps(frozen), "utf-8")

    result = runner.invoke(app, ["geo", "qualify", "hotel-test"])

    assert result.exit_code == 1
    assert "qualification.roofline.min_main_component" in result.output
