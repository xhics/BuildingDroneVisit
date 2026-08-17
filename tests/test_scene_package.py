from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError
from shapely.geometry import Polygon
from typer.testing import CliRunner

from hotel_pipeline.cli import app
from hotel_pipeline.scene_package import (
    _camera_path,
    _extruded_obj,
    _phase1_blocking_reasons,
    _verify_existing,
)
from hotel_pipeline.schemas.scene import (
    EvidenceClass,
    GateCheck,
    GateState,
    PackageFile,
    Phase1Status,
    Phase1Verdict,
    ScenePackage,
)


def _check(state: GateState = GateState.PASSED) -> GateCheck:
    return GateCheck(
        gate_id="gate",
        requirement="preuve",
        state=state,
        evidence=["rapport.json"],
    )


def test_environment_ready_refuse_un_gate_incomplet() -> None:
    with pytest.raises(ValidationError, match="gates non franchis"):
        Phase1Verdict(
            hotel_id="hotel-test",
            generated_at=datetime.now(timezone.utc).isoformat(),
            status=Phase1Status.ENVIRONMENT_3D_READY,
            router_decision_digest="abc",
            input_digests={"router": "abc"},
            checks=[_check(GateState.UNVERIFIED)],
            blocking_reasons=[],
            human_review_approved=True,
        )


def test_environment_ready_exige_approbation_humaine() -> None:
    with pytest.raises(ValidationError, match="revue humaine"):
        Phase1Verdict(
            hotel_id="hotel-test",
            generated_at=datetime.now(timezone.utc).isoformat(),
            status=Phase1Status.ENVIRONMENT_3D_READY,
            router_decision_digest="abc",
            input_digests={"router": "abc"},
            checks=[_check()],
            blocking_reasons=[],
            human_review_approved=False,
        )


def test_verdict_capture_required_conserve_les_blocages() -> None:
    verdict = Phase1Verdict(
        hotel_id="hotel-test",
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=Phase1Status.NEEDS_AUTHORIZED_CAPTURE,
        router_decision_digest="abc",
        input_digests={"router": "abc"},
        checks=[_check(GateState.FAILED)],
        blocking_reasons=["capture manquante"],
    )
    assert verdict.blocking_reasons == ["capture manquante"]


def test_un_gate_g1_franchi_ne_reapparait_pas_comme_blocage() -> None:
    checks = [
        GateCheck(gate_id="G1_deduplication", requirement="dedup", state="passed", evidence=["ok"]),
        GateCheck(gate_id="G2_exterior", requirement="extérieur", state="passed", evidence=["ok"]),
        GateCheck(gate_id="G3_quality", requirement="qualité", state="passed", evidence=["ok"]),
    ]
    reasons = _phase1_blocking_reasons(
        {"blocking_reasons": ["capture manquante"]}, checks,
        duplicate_files=335, asset_count=335, duplicate_robust=True,
        exterior_count=189, geometry_with_quality=9, geometry_count=9,
        independent_viewpoints=1,
    )

    assert all("G1" not in reason for reason in reasons)
    assert "capture manquante" in reasons


def test_orbite_est_virtuelle_et_compte_douze_poses() -> None:
    polygon = Polygon([(0, 0), (20, 0), (20, 10), (0, 10)])
    path = _camera_path(polygon, height_m=10, fov_deg=80)
    assert path.simulation_only is True
    assert len(path.poses) == 12
    assert len({pose.frame for pose in path.poses}) == 12
    assert all(pose.distance_m > 0 for pose in path.poses)


def test_orbite_varie_avec_le_champ_de_vision_declare() -> None:
    polygon = Polygon([(0, 0), (20, 0), (20, 10), (0, 10)])
    étroite = _camera_path(polygon, height_m=10, fov_deg=50)
    large = _camera_path(polygon, height_m=10, fov_deg=100)
    assert étroite.poses[0].distance_m > large.poses[0].distance_m


def test_obj_est_un_volume_ferme_avec_groupes_proxy() -> None:
    polygon = Polygon([(0, 0), (20, 0), (20, 10), (10, 5), (0, 10)])
    obj = _extruded_obj(polygon, ground_z=0, roof_z=10)
    assert obj.count("\nv ") == 10
    assert "g facade_proxy" in obj
    assert "g flat_roof_proxy" in obj
    assert "g ground_proxy" in obj

    faces = [
        [int(token.split("/", 1)[0]) for token in line.split()[1:]]
        for line in obj.splitlines()
        if line.startswith("f ")
    ]
    undirected_edges: dict[tuple[int, int], int] = {}
    for face in faces:
        for index, start in enumerate(face):
            end = face[(index + 1) % len(face)]
            edge = tuple(sorted((start, end)))
            undirected_edges[edge] = undirected_edges.get(edge, 0) + 1
    assert undirected_edges
    assert set(undirected_edges.values()) == {2}, (
        "chaque arête d'un volume fermé doit appartenir à exactement deux faces"
    )


def test_obj_refuse_une_cour_interieure_qu_il_ne_sait_pas_fermer() -> None:
    polygon = Polygon(
        [(0, 0), (20, 0), (20, 20), (0, 20)],
        holes=[[(5, 5), (15, 5), (15, 15), (5, 15)]],
    )
    with pytest.raises(ValueError, match="cour intérieure"):
        _extruded_obj(polygon, ground_z=0, roof_z=10)


def _scene(real_call: bool = False) -> dict:
    path = _camera_path(Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]), 8, 80)
    return {
        "hotel_id": "hotel-test",
        "package_id": "0123456789abcdef",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "hybrid_proxy_package",
        "horizontal_crs": "EPSG:2950",
        "vertical_datum": "CGVD 1928",
        "local_origin_projected": (0, 0, 20),
        "input_digests": {"site": "abc"},
        "phase1_verdict": "phase1_verdict.json",
        "files": [
            PackageFile(
                path="environment.obj",
                sha256="a" * 64,
                role="volume",
                evidence_class=EvidenceClass.PROXY,
                source_refs=["BUILDING_MAIN"],
            )
        ],
        "camera_paths": [path],
        "rights_summary": {"open_data": 1},
        "forbidden_claims": ["ENTRANCE_MAIN_CURRENT"],
        "limitations": ["proxy"],
        "video_generation": {"real_provider_call_performed": real_call},
    }


def test_scene_refuse_un_appel_fournisseur_pretendu() -> None:
    with pytest.raises(ValidationError, match="fournisseur"):
        ScenePackage.model_validate(_scene(real_call=True))


def test_paquet_existant_est_relu_jusqu_au_contenu(
    tmp_path: Path,
) -> None:
    payload = _scene()
    verdict = Phase1Verdict(
        hotel_id="hotel-test",
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=Phase1Status.NEEDS_AUTHORIZED_CAPTURE,
        router_decision_digest="abc",
        input_digests={"site": "abc"},
        checks=[_check(GateState.FAILED)],
        blocking_reasons=["capture"],
    )
    (tmp_path / "environment.obj").write_text("mesh", encoding="utf-8")
    payload["files"][0] = payload["files"][0].model_copy(
        update={"sha256": __import__("hashlib").sha256(b"mesh").hexdigest()}
    )
    (tmp_path / "scene.json").write_text(
        ScenePackage.model_validate(payload).model_dump_json(), encoding="utf-8"
    )
    (tmp_path / "phase1_verdict.json").write_text(
        verdict.model_dump_json(), encoding="utf-8"
    )
    _verify_existing(tmp_path, {"site": "abc"})

    (tmp_path / "environment.obj").write_text("corrompu", encoding="utf-8")
    with pytest.raises(ValueError, match="fichier modifié"):
        _verify_existing(tmp_path, {"site": "abc"})


def test_cli_expose_scene_build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_build(_workspace):
        return {"package": tmp_path, "scene": tmp_path / "scene.json"}

    monkeypatch.setattr("hotel_pipeline.scene_package.build", fake_build)
    result = CliRunner().invoke(app, ["scene", "build", "hotel-test"])
    assert result.exit_code == 0
    assert "paquet 3D hybride publié" in result.stdout
    assert "NEEDS_AUTHORIZED_CAPTURE" in result.stdout


# --- champs visuels morts ---------------------------------------------------


def test_a_pose_facing_an_unobserved_facade_is_declared_blind() -> None:
    """Le cadrage d'une géométrie sans apparence mesurée est déclaré, non retiré."""
    from shapely.geometry import Polygon as _P

    from hotel_pipeline.scene_package import _camera_path

    square = _P([(0, 0), (20, 0), (20, 20), (0, 20)])
    path = _camera_path(
        square, height_m=10.0, fov_deg=80.0,
        front_azimuth_deg=0.0,
        observed_appearance=frozenset({"FACADE_PRIMARY"}),
    )

    faces = {pose.azimuth_deg: (pose.faces, pose.blind_field) for pose in path.poses}
    # Au cap de façade, l'observateur voit la façade principale : elle est vue.
    assert faces[0.0] == ("FACADE_PRIMARY", False)
    # À l'opposé, l'arrière — jamais photographié ici.
    assert faces[180.0] == ("FACADE_REAR", True)
    # Aucune pose n'est supprimée : la lacune se déclare, elle ne se masque pas.
    assert len(path.poses) == 12


def test_without_a_known_orientation_no_pose_claims_a_facade() -> None:
    """Ignorer ce qu'une pose regarde interdit d'affirmer qu'elle est aveugle."""
    from shapely.geometry import Polygon as _P

    from hotel_pipeline.scene_package import _camera_path

    square = _P([(0, 0), (20, 0), (20, 20), (0, 20)])
    path = _camera_path(square, height_m=10.0, fov_deg=80.0)

    assert all(pose.faces is None for pose in path.poses)
    assert not any(pose.blind_field for pose in path.poses)


def test_the_derivation_counts_the_blind_poses() -> None:
    from shapely.geometry import Polygon as _P

    from hotel_pipeline.scene_package import _camera_path

    square = _P([(0, 0), (20, 0), (20, 20), (0, 20)])
    path = _camera_path(
        square, height_m=10.0, fov_deg=80.0, front_azimuth_deg=227.89,
        observed_appearance=frozenset({"FACADE_PRIMARY"}),
    )

    assert "champ visuel mort" in path.derivation
    blind = sum(pose.blind_field for pose in path.poses)
    assert f"{blind}/12" in path.derivation


def test_forbidden_claims_and_blind_fields_come_from_the_constraints() -> None:
    """Une liste en dur interdisait des objets depuis établis."""
    from hotel_pipeline.scene_package import _blind_visual_fields, _forbidden_claims

    constraints = {"constraints": [
        {"rule": "do_not_show_as_fact", "zone_ref": "PROPERTY_PARCEL"},
        {"rule": "avoid_framing_no_observed_appearance",
         "zone_ref": "ENTRANCE_MAIN_CURRENT,PROPERTY_SIGN"},
        {"rule": "avoid_close_up", "zone_ref": "BUILDING_MAIN"},
    ]}

    assert _forbidden_claims(constraints) == ["PROPERTY_PARCEL"]
    assert _blind_visual_fields(constraints) == [
        "ENTRANCE_MAIN_CURRENT", "PROPERTY_SIGN",
    ]


def test_facade_faced_follows_the_canonical_sector_table() -> None:
    """Une règle propre au paquet avait inversé gauche et droite.

    `sectors.sector_for` classe les assets réels ; en déduire une façade par
    une seconde table indépendante plaçait `FACADE_RIGHT` là où le pipeline
    mesurait `left`, et le rapport aurait déclaré aveugle un mur observé.
    """
    from hotel_pipeline.scene_package import _facade_faced
    from hotel_pipeline.schemas.enums import ViewSector
    from hotel_pipeline.sectors import sector_for

    attendu = {
        ViewSector.FRONT: "FACADE_PRIMARY",
        ViewSector.FRONT_LEFT_CORNER: "FACADE_PRIMARY",
        ViewSector.FRONT_RIGHT_CORNER: "FACADE_PRIMARY",
        ViewSector.LEFT: "FACADE_LEFT",
        ViewSector.RIGHT: "FACADE_RIGHT",
        ViewSector.REAR: "FACADE_REAR",
        ViewSector.REAR_LEFT_CORNER: "FACADE_REAR",
        ViewSector.REAR_RIGHT_CORNER: "FACADE_REAR",
    }
    front = 227.89
    for bearing in range(0, 360, 5):
        secteur = sector_for(float(bearing), front)
        if secteur not in attendu:
            continue
        assert _facade_faced(float(bearing), front) == attendu[secteur], (
            f"azimut {bearing}° : secteur {secteur.value}"
        )


def test_the_side_facades_are_not_mirrored() -> None:
    """Contrôle direct de l'inversion : le côté gauche n'est pas le droit."""
    from hotel_pipeline.scene_package import _facade_faced
    from hotel_pipeline.sectors import sector_for
    from hotel_pipeline.schemas.enums import ViewSector

    front = 227.89
    gauche = next(
        b for b in range(360) if sector_for(float(b), front) is ViewSector.LEFT
    )
    droite = next(
        b for b in range(360) if sector_for(float(b), front) is ViewSector.RIGHT
    )

    assert _facade_faced(float(gauche), front) == "FACADE_LEFT"
    assert _facade_faced(float(droite), front) == "FACADE_RIGHT"
