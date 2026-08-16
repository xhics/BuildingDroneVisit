"""Le calcul de visibilité des corridors (collecte V2).

`engine.assess()` exige un `crs` en argument obligatoire ; `_corridor` ne le
passait pas. Le chemin était donc cassé depuis l'ajout de ce paramètre, et rien
ne le signalait : les tests de corridors portaient sur le schéma, jamais sur
cette fonction.

Ce que ces tests protègent : une mesure sans référentiel ne se rattache à rien,
et le rendre facultatif laisserait ce chemin le deviner.
"""

from __future__ import annotations

import pytest
from shapely.geometry import LineString, Polygon

from hotel_pipeline.geo.visibility_run import _corridor
from hotel_pipeline.schemas import DEFAULT_POLICY


def _corridor_reel():
    """Le **vrai** corridor, non une doublure : une fixture approximative
    diverge du schéma et fait échouer le test pour une autre raison que celle
    qu'il éprouve."""
    from hotel_pipeline.schemas.geometry import (
        AccessStatus, CorridorClass, RoadCorridor,
    )

    return RoadCorridor(
        corridor_id="CORRIDOR_ESSAI", feature_id="ROAD_ESSAI",
        corridor_class=CorridorClass.ADJACENT_ROAD,
        access_status=AccessStatus.PUBLIC_CONFIRMED,
        rationale="voie d'essai passant devant la cible",
    )


def _report():
    """Le **vrai** rapport, non une doublure : une fixture approximative
    laisserait passer un champ que le code renseigne réellement."""
    from hotel_pipeline.geo.visibility_run import RunReport

    return RunReport()


#: Un bâtiment carré, et une voie qui passe devant à trente mètres.
CIBLE = Polygon([(0, 0), (30, 0), (30, 30), (0, 30), (0, 0)])
VOIE = LineString([(-20, -30), (50, -30)])


def _assess(crs: str = "EPSG:2950"):
    return _corridor(
        _corridor_reel(), VOIE, CIBLE, [], DEFAULT_POLICY.visibility,
        _report(), sectors=None, crs=crs,
    )


# --- le référentiel traverse jusqu'à la mesure --------------------------------


def test_the_corridor_assessment_runs_at_all() -> None:
    """Le chemin était cassé : `assess() missing 1 required keyword-only
    argument: 'crs'`. Aucun test ne l'exerçait."""
    résultat = _assess()

    assert résultat is not None
    assert résultat.corridor_id == "CORRIDOR_ESSAI"
    assert résultat.best_sample_id, "au moins un échantillon a été évalué"


def test_the_crs_reaches_the_underlying_assessment() -> None:
    """Une mesure sans référentiel ne se rattache à rien.

    Le vérifier sur le résultat, non sur l'appel : compter les arguments
    passés dirait que le code appelle, non que la valeur arrive.
    """
    from hotel_pipeline.geo import visibility_engine

    vus: list[str] = []
    vrai_assess = visibility_engine.assess

    def espion(*args, **kwargs):
        vus.append(kwargs.get("crs"))
        return vrai_assess(*args, **kwargs)

    visibility_engine.assess = espion
    try:
        _assess("EPSG:32188")
    finally:
        visibility_engine.assess = vrai_assess

    assert vus, "aucune évaluation n'a été demandée"
    assert set(vus) == {"EPSG:32188"}, (
        f"le référentiel du manifeste doit atteindre chaque mesure, vu : {set(vus)}"
    )


def test_a_missing_crs_is_not_silently_invented() -> None:
    """Le rendre facultatif laisserait ce chemin deviner un référentiel.

    Passer une chaîne vide reste possible — c'est l'appelant qui doit fournir
    celui du manifeste — mais la valeur transmise doit être **celle-là**, non
    une valeur de repli choisie ici.
    """
    from hotel_pipeline.geo import visibility_engine

    vus: list[str] = []
    vrai_assess = visibility_engine.assess

    def espion(*args, **kwargs):
        vus.append(kwargs.get("crs"))
        return vrai_assess(*args, **kwargs)

    visibility_engine.assess = espion
    try:
        # Le schéma refuse une mesure sans référentiel : c'est exactement ce
        # qu'on veut voir — l'omission ne passe pas en silence.
        with pytest.raises(ValueError, match="at least 1 character"):
            _corridor(
                _corridor_reel(), VOIE, CIBLE, [], DEFAULT_POLICY.visibility,
                _report(), sectors=None,
            )
    finally:
        visibility_engine.assess = vrai_assess

    assert set(vus) == {""}, (
        "sans référentiel fourni, aucun n'est inventé — et le schéma refuse "
        "ensuite la mesure, ce qui rend l'omission impossible à ignorer"
    )


def test_two_referentials_are_not_confused() -> None:
    """Deux mesures publiées sous des référentiels différents ne se comparent
    pas : leur confusion rendrait une distance en mètres illisible."""
    from hotel_pipeline.geo import visibility_engine

    vus: list[str] = []
    vrai_assess = visibility_engine.assess

    def espion(*args, **kwargs):
        vus.append(kwargs.get("crs"))
        return vrai_assess(*args, **kwargs)

    visibility_engine.assess = espion
    try:
        _assess("EPSG:2950")
        premier = set(vus)
        vus.clear()
        _assess("EPSG:32188")
    finally:
        visibility_engine.assess = vrai_assess

    assert premier == {"EPSG:2950"}
    assert set(vus) == {"EPSG:32188"}


# --- le manifeste antérieur au schéma courant ---------------------------------


def test_visibility_assess_loads_a_legacy_geometry_manifest(tmp_path, monkeypatch) -> None:
    """`visibility assess` validait le schéma directement.

    Le manifeste du pilote est antérieur : il ne porte ni `schema_version`, ni
    `working_crs`, ni `spatial_context_digest`. Le refuser bloquait tout
    recalcul, alors que `load_capture_geometry` sait le rattacher au
    référentiel du contexte — sans réécrire le fichier.
    """
    import json

    from hotel_pipeline.geo.geometry_loader import load_capture_geometry
    from hotel_pipeline.schemas.spatial_reference import (
        SpatialReferenceContext,
        TerritoryState,
    )

    legacy = {
        "hotel_id": "essai",
        "working_crs": "EPSG:2950",
        "built_at": "2026-08-13T00:00:00+00:00",
        "geometries": [],
        "corridors": [],
        "snapshots": [],
        "policy_digest": "d" * 16,
        "site_manifest_digest": "s" * 16,
        "spatial_manifest_digest": "p" * 16,
        "overpass_elements_digest": "o" * 16,
    }
    path = tmp_path / "capture_geometry.json"
    path.write_text(json.dumps(legacy), "utf-8")

    assert "schema_version" not in legacy, "c'est ce qui en fait un legacy"

    # `working_crs` doit correspondre à celui sous lequel le manifeste a été
    # écrit : un contexte différent ferait refuser le rattachement, et à juste
    # titre — les formes projetées ne seraient pas celles de ce site.
    # Un CRS déclaré doit être **opposable** : sans unité, axes, emprise ni
    # motif de sélection, on ne pourrait pas dire ce qu'une mesure signifie.
    reference = SpatialReferenceContext(
        hotel_id="essai", reference_lat=45.5, reference_lon=-73.4,
        # Un CRS ne se choisit pas sur un territoire inconnu : c'est ce qui
        # faisait appliquer le fuseau du pilote partout.
        territory_state=TerritoryState.RESOLVED,
        jurisdictions=["CA-QC"],
        working_crs="EPSG:2950", working_unit="metre",
        working_axes="easting,northing",
        working_area_of_use=[-73.5, 45.0, -73.0, 46.0],
        selection_method="essai",
    )
    manifeste, avertissement = load_capture_geometry(path, reference)

    assert manifeste is not None
    assert manifeste.working_crs == "EPSG:2950", (
        "le référentiel vient du contexte, non du fichier"
    )
    assert avertissement, "le rattachement se dit, il ne se tait pas"

    # Le fichier n'est pas réécrit : un manifeste qui change à la lecture ne se
    # relit plus comme ce qu'il fut.
    assert json.loads(path.read_text("utf-8")) == legacy


def test_the_run_passes_the_manifest_referential_to_every_corridor() -> None:
    """Appeler `_corridor` directement ne prouve rien sur ce que l'exécution
    lui donne.

    Sans ce test, remplacer `manifest.working_crs` par une constante passait
    inaperçu : les mesures auraient été publiées sous un référentiel qui n'est
    pas celui du manifeste.
    """
    import inspect

    from hotel_pipeline.geo import visibility_run

    source = inspect.getsource(visibility_run.run_assessment)
    appel = source[source.index("_corridor("):]
    appel = appel[: appel.index(")\n") + 1]

    assert "crs=manifest.working_crs" in appel, (
        "le référentiel transmis doit être celui du manifeste, non une valeur "
        f"choisie ici — vu : {appel!r}"
    )


def test_visibility_assess_does_not_validate_the_manifest_directly() -> None:
    """Le chemin du CLI, non seulement le chargeur.

    Vérifier `load_capture_geometry` isolément ne dit rien de ce que
    `visibility assess` emprunte : il validait le schéma directement, et
    refusait le manifeste du pilote.
    """
    import inspect

    from hotel_pipeline import cli

    source = inspect.getsource(cli.visibility_assess)

    assert "_capture_geometry_if_any(workspace, context)" in source, (
        "le manifeste doit passer par le chargeur tolérant"
    )
    assert "CaptureGeometryManifest.model_validate(raw)" not in source, (
        "le valider directement refuse tout manifeste antérieur au schéma"
    )
