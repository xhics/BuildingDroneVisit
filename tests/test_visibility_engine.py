"""Visibilité multi-rayons (Lot 1B V2, étape 3).

Toutes les scènes sont en projection métrique : la cible occupe un carré au
nord de la caméra, les obstacles s'intercalent ou non. Les distances sont donc
lisibles directement en mètres.
"""

from __future__ import annotations

import math

import pytest
from shapely.geometry import LineString, Polygon

from hotel_pipeline.geo.visibility_engine import (
    CameraVertical,
    Obstacle,
    TargetVertical,
    angular_span,
    assess,
    cells,
    frame_target,
    group_segments,
    sample_line,
    vertical_verdict,
)
from hotel_pipeline.schemas import DEFAULT_POLICY
from hotel_pipeline.schemas.visibility import (
    FramingAssessment,
    LineOfSightStatus,
    RayAssessment,
    RayPartition,
    VerticalVisibilityStatus,
    VisibilityAssessment,
)

POLICY = DEFAULT_POLICY.visibility
ORIGIN = (0.0, 0.0)

#: Cible carrée de 20 m, à 50 m au nord.
TARGET = Polygon([(-10, 50), (10, 50), (10, 70), (-10, 70)])


#: Référentiel des coordonnées de ces cas. Il est passé explicitement : le
#: moteur n'en suppose plus aucun.
CRS = "EPSG:2950"


def run(obstacles=(), **kwargs) -> VisibilityAssessment:
    kwargs.setdefault("crs", CRS)
    return assess(
        "a", "camera", "BUILDING_MAIN", ORIGIN, kwargs.pop("target", TARGET),
        list(obstacles), POLICY, **kwargs
    )


def wall(x0: float, y0: float, x1: float, y1: float, **kwargs) -> Obstacle:
    return Obstacle(
        kwargs.pop("feature_id", "OB"),
        Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]),
        **kwargs,
    )


# --- silhouette ---------------------------------------------------------------


def test_the_span_is_the_silhouette_not_the_box() -> None:
    """Sur une forme oblique, la boîte ajoute des degrés vides."""
    oblique = Polygon([(-10, 50), (10, 60), (0, 70), (-20, 60)])
    _, _, span, _ = angular_span(ORIGIN, oblique)
    _, _, box_span, _ = angular_span(ORIGIN, oblique.envelope)

    assert span < box_span


def test_a_span_crossing_north_is_handled() -> None:
    """Une cible à cheval sur 0° occupe un intervalle, pas son complément."""
    straddling = Polygon([(-10, 50), (10, 50), (10, 70), (-10, 70)])
    start, end, span, crosses = angular_span((0.0, 0.0), straddling)

    assert crosses is True
    assert start > end  # 348° → 11°
    assert span == pytest.approx(22.6, abs=0.2)
    # Le complément aurait donné 337° : c'est l'erreur qu'un tri naïf commet.
    assert span < 180


def test_a_concave_target_leaves_empty_directions_inside_its_span() -> None:
    """Un contour en U : certaines directions de l'intervalle ne touchent rien."""
    concave = Polygon(
        [(-20, 50), (-10, 50), (-10, 65), (10, 65), (10, 50), (20, 50), (20, 70), (-20, 70)]
    )
    assessment = run(target=concave)

    assert assessment.angular_span_deg > 0
    assert all(r.partition is RayPartition.CLEAR_2D for r in assessment.rays)


def test_the_cells_come_from_the_policy() -> None:
    fine = DEFAULT_POLICY.visibility.model_copy(update={"max_angular_step_deg": 0.1})
    coarse = DEFAULT_POLICY.visibility.model_copy(
        update={"max_angular_step_deg": 5.0, "min_angular_cells": 8}
    )

    assert len(cells(0.0, 40.0, fine)) > len(cells(0.0, 40.0, coarse))
    # Chaque cellule porte sa largeur, et leur somme fait l'intervalle.
    assert sum(width for _, width in cells(0.0, 40.0, fine)) == pytest.approx(40.0)


def test_rays_are_weighted_by_their_angular_width() -> None:
    """Compter les rayons à l'unité donnerait du poids au maillage.

    Une cellule de 2° et une de 0,1° ne disent pas la même chose de la façade.
    """
    assessment = run([wall(-12, 20, 0, 25)])
    widths = {round(r.angular_width_deg, 9) for r in assessment.rays}

    assert len(widths) == 1  # échantillonnage uniforme, ici
    covered = sum(
        r.angular_width_deg for r in assessment.rays
        if r.partition is RayPartition.RISK_UNKNOWN_HEIGHT
    )
    # Comparée à la précision publiée : `output_precision` arrondit les
    # fractions pour que deux exécutions identiques rendent le même rapport.
    assert assessment.risk_unknown_height_fraction == pytest.approx(
        covered / assessment.angular_span_deg, abs=1e-4
    )


# --- profondeur ----------------------------------------------------------------


def test_an_obstacle_behind_the_target_masks_nothing() -> None:
    assessment = run([wall(-12, 80, 12, 85)])

    assert assessment.proven_clear_fraction == 1.0
    assert assessment.status is LineOfSightStatus.CLEAR


def test_an_obstacle_in_front_creates_a_risk_when_heights_are_unknown() -> None:
    """Les 27 obstacles réels sont sans hauteur : ils ne prouvent rien."""
    assessment = run([wall(-12, 20, 12, 25)])

    assert assessment.risk_unknown_height_fraction == 1.0
    assert assessment.proven_blocked_fraction == 0.0
    assert assessment.status is LineOfSightStatus.AT_RISK
    assert assessment.visible_lower_bound == 0.0
    assert assessment.visible_upper_bound == 1.0


def test_a_partial_obstacle_leaves_a_clear_span() -> None:
    assessment = run([wall(-12, 20, 0, 25)])

    assert 0 < assessment.proven_clear_fraction < 1
    assert 0 < assessment.risk_unknown_height_fraction < 1
    assert assessment.status is LineOfSightStatus.PARTIAL
    assert assessment.largest_clear_span_deg > 0


def test_two_obstacles_on_one_ray_are_counted_once() -> None:
    """Tous sont conservés, la cellule n'est comptée qu'une fois."""
    assessment = run([
        wall(-12, 20, 12, 25, feature_id="PROCHE"),
        wall(-12, 30, 12, 35, feature_id="LOIN"),
    ])

    assert assessment.risk_unknown_height_fraction == 1.0
    assert sorted(assessment.obstacles_at_risk) == ["LOIN", "PROCHE"]
    crowded = [r for r in assessment.rays if len(r.obstacles) == 2]
    assert crowded
    # Le plus proche est cité en premier : l'ordre porte l'information de
    # profondeur.
    assert crowded[0].obstacles[0] == "PROCHE"


def test_the_three_fractions_always_partition_the_span() -> None:
    assessment = run([wall(-12, 20, 0, 25)])
    total = (
        assessment.proven_clear_fraction
        + assessment.risk_unknown_height_fraction
        + assessment.proven_blocked_fraction
    )
    assert total == pytest.approx(1.0, abs=1e-6)


# --- verticalité ----------------------------------------------------------------


def known_camera() -> CameraVertical:
    return CameraVertical(ground_m=10.0, height_above_ground_m=2.5, provenance="dtm+relevé")


def known_target() -> TargetVertical:
    return TargetVertical(ground_m=10.0, height_m=12.0, provenance="ndsm qualifié")


def test_a_tall_obstacle_with_known_heights_proves_a_block() -> None:
    assessment = run(
        [wall(-12, 20, 12, 25, feature_id="TOUR", height_m=30.0, ground_m=10.0)],
        camera=known_camera(), target_vertical=known_target(),
    )

    assert assessment.proven_blocked_fraction == 1.0
    assert assessment.status is LineOfSightStatus.BLOCKED
    assert assessment.obstacles_blocking == ["TOUR"]


def test_a_low_obstacle_with_known_heights_proves_nothing_blocked() -> None:
    """Un muret ne masque pas un bâtiment de douze mètres."""
    assessment = run(
        [wall(-12, 20, 12, 25, feature_id="MURET", height_m=1.5, ground_m=10.0)],
        camera=known_camera(), target_vertical=known_target(),
    )

    assert assessment.proven_clear_fraction == 1.0
    assert assessment.proven_blocked_fraction == 0.0


def test_one_missing_datum_keeps_the_ray_at_risk() -> None:
    """L'absence d'une seule valeur suffit : rien n'est extrapolé."""
    obstacle = wall(-12, 20, 12, 25, feature_id="OB", height_m=30.0, ground_m=10.0)

    blocked, status, missing = vertical_verdict(
        ORIGIN, obstacle, 22.0, 50.0, known_camera(), known_target()
    )
    assert blocked is True and status is VerticalVisibilityStatus.FULLY_KNOWN

    for camera, target, expected in [
        (CameraVertical(height_above_ground_m=2.5), known_target(), "terrain à la caméra"),
        (CameraVertical(ground_m=10.0), known_target(), "hauteur de caméra"),
        (known_camera(), TargetVertical(height_m=12.0), "terrain de la cible"),
        (known_camera(), TargetVertical(ground_m=10.0), "hauteur de la cible"),
    ]:
        blocked, status, missing = vertical_verdict(
            ORIGIN, obstacle, 22.0, 50.0, camera, target
        )
        assert blocked is False
        assert status is VerticalVisibilityStatus.INCOMPLETE
        assert expected in missing


def test_an_unknown_camera_elevation_prevents_any_proof() -> None:
    assessment = run(
        [wall(-12, 20, 12, 25, feature_id="TOUR", height_m=30.0, ground_m=10.0)],
        camera=CameraVertical(), target_vertical=known_target(),
    )

    assert assessment.proven_blocked_fraction == 0.0
    assert assessment.risk_unknown_height_fraction == 1.0
    assert "terrain à la caméra" in assessment.missing_vertical


def test_a_blocked_verdict_cannot_be_declared_without_full_knowledge() -> None:
    from hotel_pipeline.schemas.visibility import HitVerdict, ObstacleHit

    hit = ObstacleHit(
        obstacle_ref="OB", distance_m=10.0,
        vertical_status=VerticalVisibilityStatus.UNKNOWN,
        verdict=HitVerdict.UNDECIDABLE, missing_vertical=["hauteur de OB"],
    )
    with pytest.raises(ValueError, match="hauteur inconnue reste un risque"):
        RayAssessment(
            bearing_deg=0.0, angular_width_deg=1.0,
            partition=RayPartition.BLOCKED_2_5D, hits=[hit],
            vertical_status=VerticalVisibilityStatus.INCOMPLETE,
        )


def test_a_risk_without_an_undecidable_obstacle_is_refused() -> None:
    with pytest.raises(ValueError, match="risque annoncé sans obstacle"):
        RayAssessment(
            bearing_deg=0.0, angular_width_deg=1.0,
            partition=RayPartition.RISK_UNKNOWN_HEIGHT,
        )


# --- cadrage --------------------------------------------------------------------


def test_visibility_does_not_depend_on_the_frame() -> None:
    """Deux recadrages du même panorama voient la même scène.

    Faire varier la visibilité avec l'objectif reviendrait à déplacer les murs
    en tournant la caméra.
    """
    assert "frame" not in assess.__code__.co_varnames
    assert not hasattr(run(), "outside_frame_fraction")
    assert "OUT_OF_FRAME" not in {status.name for status in LineOfSightStatus}


def test_the_projection_is_tangential_and_off_axis_aware() -> None:
    """`span / fov` n'est exact que centré, et faux partout ailleurs."""
    centred = frame_target("a", "s", 345.0, 30.0, 0.0, 60.0, 640, 640, "requête", POLICY)
    off_axis = frame_target("b", "s", 10.0, 30.0, 0.0, 60.0, 640, 640, "requête", POLICY)

    naive = 30.0 / 60.0
    expected = math.tan(math.radians(15.0)) / math.tan(math.radians(30.0))

    assert centred.unclipped_width_fraction == pytest.approx(expected, abs=1e-4)
    assert centred.unclipped_width_fraction < naive
    # Hors axe, la même ouverture couvre **davantage** de largeur : c'est
    # l'inverse de l'erreur commise au centre.
    assert off_axis.unclipped_width_fraction > centred.unclipped_width_fraction


def test_the_frame_intersection_is_analytic() -> None:
    """Bornage des angles, non comptage d'échantillons."""
    partial = frame_target("a", "s", 10.0, 30.0, 0.0, 60.0, 640, 640, "requête", POLICY)

    # L'intervalle va de +10° à +40°, le cadre s'arrête à +30° : deux tiers.
    assert partial.target_in_frame_fraction == pytest.approx(2 / 3, abs=1e-4)
    assert partial.clipped_width_fraction < partial.unclipped_width_fraction


def test_a_target_outside_the_frame_has_nothing_in_it() -> None:
    framing = frame_target("a", "s", 100.0, 30.0, 0.0, 60.0, 640, 640, "requête", POLICY)

    assert framing.target_in_frame_fraction == 0.0
    assert framing.expected_width_px == 0
    # Ce n'est pas une occultation : la géométrie, elle, n'a pas changé.
    assert framing.horizontal_computable is True


def test_the_vertical_fov_comes_from_the_image_ratio() -> None:
    wide = frame_target("a", "s", 350.0, 20.0, 0.0, 90.0, 1280, 720, "requête", POLICY)
    square = frame_target("b", "s", 350.0, 20.0, 0.0, 90.0, 640, 640, "requête", POLICY)

    assert wide.vertical_fov_deg < wide.fov_deg
    assert square.vertical_fov_deg == pytest.approx(90.0, abs=1e-6)


def test_width_can_be_measured_while_height_stays_unknown() -> None:
    """Une visée réputée horizontale est une convention, pas une mesure."""
    framing = frame_target("a", "s", 345.0, 30.0, 0.0, 60.0, 640, 640, "requête", POLICY)

    assert framing.horizontal_computable is True
    assert framing.expected_width_px > 0
    assert framing.vertical_computable is False
    assert framing.expected_height_px is None
    assert "inclinaison" in framing.vertical_reason


def test_height_is_measured_once_pitch_and_extent_are_known() -> None:
    framing = frame_target(
        "a", "s", 345.0, 30.0, 0.0, 60.0, 640, 640, "requête", POLICY,
        pitch_deg=0.0, target_vertical_span_deg=14.0,
    )

    assert framing.vertical_computable is True
    assert framing.expected_height_px > 0


def test_mapillary_without_intrinsics_is_not_computable() -> None:
    """Aucun champ générique n'est supposé pour Mapillary."""
    framing = frame_target(
        "a", "mapillary-1", 340.0, 30.0, 45.0, None, 2048, 1536, None, POLICY,
        reason_if_absent="intrinsèques Mapillary absentes",
    )

    assert framing.horizontal_computable is False
    assert "intrinsèques Mapillary absentes" in framing.horizontal_reason
    assert framing.expected_width_px is None


def test_two_framings_of_one_panorama_share_the_geometry_and_differ_in_frame() -> None:
    east = frame_target("a", "sv", 80.0, 30.0, 90.0, 60.0, 640, 640, "requête", POLICY)
    west = frame_target("b", "sv", 80.0, 30.0, 270.0, 60.0, 640, 640, "requête", POLICY)

    assert east.target_in_frame_fraction > west.target_in_frame_fraction
    assert west.target_in_frame_fraction == 0.0


def test_a_computable_framing_must_carry_its_provenance() -> None:
    with pytest.raises(ValueError, match="provenance"):
        FramingAssessment(
            assessment_id="a", subject_ref="s", heading_deg=0.0, fov_deg=60.0,
            width_px=640, height_px=640, horizontal_computable=True,
            vertical_reason="sans objet",
        )


def test_an_uncomputable_framing_must_say_why() -> None:
    with pytest.raises(ValueError, match="largeur non calculable sans motif"):
        FramingAssessment(assessment_id="a", subject_ref="s")


def test_a_height_declared_computable_needs_a_pitch() -> None:
    with pytest.raises(ValueError, match="sans inclinaison"):
        FramingAssessment(
            assessment_id="a", subject_ref="s", horizontal_reason="sans objet",
            vertical_computable=True,
        )


# --- corridors --------------------------------------------------------------------


def test_a_loop_is_not_sampled_twice_at_its_closing_point() -> None:
    loop = LineString([(0, 0), (30, 0), (30, 30), (0, 30), (0, 0)])
    samples = sample_line(loop, 10.0)
    positions = [position for _, position in samples]

    assert len(positions) == len(set(positions))


def test_sampling_is_deterministic_and_stepped_by_the_policy() -> None:
    line = LineString([(0, 0), (100, 0)])

    assert sample_line(line, 10.0) == sample_line(line, 10.0)
    assert len(sample_line(line, 10.0)) > len(sample_line(line, 25.0))


def test_useful_samples_are_grouped_into_segments() -> None:
    """Vingt-cinq échantillons d'une route ne font pas vingt-cinq points de vue."""
    assert group_segments([0, 1, 2, 3]) == 1
    assert group_segments([0, 1, 5, 6, 7]) == 2
    assert group_segments([]) == 0


def test_a_corridor_assessment_has_no_frame_measures() -> None:
    """Sans caméra, `outside_frame` et les pixels n'ont pas de sens."""
    from hotel_pipeline.schemas.visibility import CorridorVisibilityAssessment

    fields = set(CorridorVisibilityAssessment.model_fields)
    assert "outside_frame_fraction" not in fields
    assert "expected_width_px" not in fields
    assert {"geometrically_useful", "access_status"} <= fields


def test_geometric_usefulness_never_grants_access() -> None:
    from hotel_pipeline.schemas.visibility import CorridorVisibilityAssessment

    from hotel_pipeline.schemas.visibility import UsefulnessVerdict

    corridor = CorridorVisibilityAssessment(
        assessment_id="a", corridor_id="CORRIDOR_1", feature_id="ROAD_1",
        geometrically_useful=UsefulnessVerdict.USEFUL, access_status="restricted",
        rationale="silhouette dégagée sur 40°",
    )

    assert corridor.geometrically_useful is UsefulnessVerdict.USEFUL
    # Avec des obstacles tous de hauteur inconnue, `false` serait aussi
    # injustifié que `true` : le défaut est donc `unknown`.
    assert CorridorVisibilityAssessment(
        assessment_id="b", corridor_id="c", feature_id="f"
    ).geometrically_useful is UsefulnessVerdict.UNKNOWN
    assert corridor.access_status == "restricted"
    assert "admissible_for_building" not in CorridorVisibilityAssessment.model_fields


# --- invariants du parcours -------------------------------------------------------


def test_the_partition_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="totalisent"):
        VisibilityAssessment(
            assessment_id="a", subject_ref="s", target_ref="t", crs=CRS,
            proven_clear_fraction=0.5, risk_unknown_height_fraction=0.2,
            rays=[RayAssessment(bearing_deg=0.0, angular_width_deg=1.0,
                                partition=RayPartition.CLEAR_2D)],
        )


def test_a_crossed_obstacle_is_not_a_responsible_one() -> None:
    """Un rayon peut croiser un mur bas prouvé inoffensif et un voisin inconnu.

    Les inscrire tous deux comme bloquants faisait porter le verdict par des
    obstacles dont rien ne l'établissait.
    """
    assessment = run(
        [
            wall(-12, 20, 12, 25, feature_id="MURET", height_m=1.0, ground_m=10.0),
            wall(-12, 30, 12, 35, feature_id="INCONNU"),
        ],
        camera=known_camera(), target_vertical=known_target(),
    )

    ray = next(r for r in assessment.rays if len(r.hits) == 2)
    verdicts = {hit.obstacle_ref: hit.verdict.value for hit in ray.hits}

    assert verdicts["MURET"] == "passes_under"
    assert verdicts["INCONNU"] == "undecidable"
    assert assessment.obstacles_at_risk == ["INCONNU"]
    assert assessment.obstacles_blocking == []


def test_a_hit_cannot_decide_without_full_knowledge() -> None:
    from hotel_pipeline.schemas.visibility import HitVerdict, ObstacleHit

    with pytest.raises(ValueError, match="sans données verticales complètes"):
        ObstacleHit(
            obstacle_ref="OB", distance_m=10.0,
            vertical_status=VerticalVisibilityStatus.INCOMPLETE,
            verdict=HitVerdict.BLOCKS,
        )
    with pytest.raises(ValueError, match="sans dire ce qui manque"):
        ObstacleHit(
            obstacle_ref="OB", distance_m=10.0,
            vertical_status=VerticalVisibilityStatus.UNKNOWN,
            verdict=HitVerdict.UNDECIDABLE,
        )


def test_the_engine_refuses_an_unsupported_setting() -> None:
    """Le maillage est uniforme : prétendre l'inverse serait une promesse vide."""
    from hotel_pipeline.geo.visibility_engine import check_supported

    assert check_supported(POLICY) == []
    exotic = POLICY.model_copy(update={"sampling_method": "adaptive_curvature"})
    assert any("non implémentée" in p for p in check_supported(exotic))


def test_the_ray_length_derives_from_the_target() -> None:
    """Un rayon fixe de mille mètres traversait des voisins hors sujet."""
    from hotel_pipeline.geo.visibility_engine import _reach

    near = _reach(ORIGIN, TARGET)
    far = _reach(ORIGIN, Polygon([(-10, 500), (10, 500), (10, 520), (-10, 520)]))

    assert near < 100 < far


def test_the_bounds_frame_the_uncertainty() -> None:
    assessment = run([wall(-12, 20, 0, 25)])

    assert assessment.visible_lower_bound == assessment.proven_clear_fraction
    assert assessment.visible_upper_bound == pytest.approx(
        assessment.proven_clear_fraction + assessment.risk_unknown_height_fraction
    )
    assert assessment.visible_lower_bound < assessment.visible_upper_bound


# --- verticale mesurée au point visé ---------------------------------------


def test_the_target_vertical_is_sampled_where_the_ray_lands() -> None:
    """Une hauteur médiane écraserait un bâtiment à corps inégaux.

    Le WelcomINNS varie de 3,3 m à 13 m selon l'endroit : c'est le point que
    le rayon touche qui décide.
    """
    seen: list[tuple[float, float]] = []

    def sampler(point):
        seen.append(point)
        return 10.0, 22.0

    assessment = run(
        [wall(-12, 20, 12, 25, feature_id="TOUR", height_m=30.0, ground_m=10.0)],
        camera=known_camera(),
        target_vertical=TargetVertical(sampler=sampler, provenance="rasters"),
    )

    assert assessment.proven_blocked_fraction == 1.0
    assert len(seen) >= len(assessment.rays)
    # Les points sondés se répartissent le long de la façade, non en un seul
    # centroïde.
    assert len({round(x, 1) for x, _ in seen}) > 1


def test_an_undefined_sample_keeps_the_ray_at_risk() -> None:
    """Une bordure sans valeur ne se remplace pas par une supposition."""
    assessment = run(
        [wall(-12, 20, 12, 25, feature_id="TOUR", height_m=30.0, ground_m=10.0)],
        camera=known_camera(),
        target_vertical=TargetVertical(sampler=lambda point: (None, None),
                                       provenance="rasters"),
    )

    assert assessment.proven_blocked_fraction == 0.0
    assert assessment.risk_unknown_height_fraction == 1.0
    assert "terrain de la cible" in assessment.missing_vertical


def test_the_sampler_is_retried_slightly_inside_the_footprint() -> None:
    """Le rayon touche l'emprise sur son bord, où les rasters sont souvent muets."""
    calls: list[tuple[float, float]] = []

    def edge_only(point):
        calls.append(point)
        # Défini seulement au-delà du premier impact.
        return (10.0, 22.0) if len(calls) % 4 != 1 else (None, None)

    assessment = run(
        [wall(-12, 20, 12, 25, feature_id="TOUR", height_m=30.0, ground_m=10.0)],
        camera=known_camera(),
        target_vertical=TargetVertical(sampler=edge_only, provenance="rasters"),
    )

    assert assessment.proven_blocked_fraction == 1.0


# --- provenance d'une exécution ----------------------------------------------


def visibility_run(**overrides):
    from hotel_pipeline.schemas.visibility import VisibilityRun

    fields = dict(
        run_id="r", hotel_id="h", engine_version="multiray-1.0.0",
        method="uniform_angular_cells", parameters={"max_angular_step_deg": "0.25"},
        capture_geometry_digest="a", policy_digest="b", site_manifest_digest="c",
        asset_files_digest="d1", asset_manifest_digest="d2", target_digest="e",
        obstacles_digest="f", road_geometry_digest="g",
        # Un run est lié à un contexte spatial, ou n'est pas constructible.
        spatial_context_digest="ctx0", crs=CRS,
    )
    fields.update(overrides)
    return VisibilityRun(**fields)


def test_a_run_without_its_digests_is_refused() -> None:
    with pytest.raises(ValueError):
        visibility_run(obstacles_digest="")


def test_the_files_digest_does_not_stand_for_the_manifest() -> None:
    """Une revue humaine ou un cap corrigé ne changent aucune image.

    Une empreinte unique laissait donc un run se croire courant après une
    décision qui, elle, avait tout changé.
    """
    from hotel_pipeline.schemas.visibility import VisibilityRun

    fields = set(VisibilityRun.model_fields)
    assert {"asset_files_digest", "asset_manifest_digest"} <= fields
    assert "assets_digest" not in fields

    with pytest.raises(ValueError):
        visibility_run(asset_manifest_digest="")


def test_a_run_without_parameters_is_refused() -> None:
    with pytest.raises(ValueError):
        visibility_run(parameters={})


def test_duplicate_framings_or_corridors_are_refused() -> None:
    from hotel_pipeline.schemas.visibility import (
        CorridorVisibilityAssessment,
        FramingAssessment,
    )

    assessment = VisibilityAssessment(
        assessment_id="vis-1", subject_ref="asset-1", target_ref="t", crs=CRS
    )
    framing = FramingAssessment(
        assessment_id="vis-1", subject_ref="asset-1",
        horizontal_reason="sans objet", vertical_reason="sans objet",
    )
    with pytest.raises(ValueError, match="cadrages dupliqués"):
        visibility_run(assessments=[assessment], framings=[framing, framing])

    corridor = CorridorVisibilityAssessment(
        assessment_id="c", corridor_id="c1", feature_id="f1"
    )
    with pytest.raises(ValueError, match="corridor dupliquées"):
        visibility_run(corridors=[corridor, corridor])


def test_two_assessments_of_one_asset_are_refused() -> None:
    """La correspondance évaluation ↔ asset doit rester univoque."""
    first = VisibilityAssessment(
        assessment_id="a", subject_ref="asset-1", target_ref="t", crs=CRS
    )
    second = VisibilityAssessment(
        assessment_id="b", subject_ref="asset-1", target_ref="t", crs=CRS
    )

    with pytest.raises(ValueError, match="même sujet"):
        visibility_run(assessments=[first, second])


def test_a_vertical_verdict_requires_a_cited_source() -> None:
    """Trancher grâce à une élévation sans dire laquelle interdit d'en douter."""
    decided = run(
        [wall(-12, 20, 12, 25, feature_id="TOUR", height_m=30.0, ground_m=10.0)],
        camera=known_camera(), target_vertical=known_target(),
    )

    from hotel_pipeline.schemas.visibility import ElevationSource

    with pytest.raises(ValueError, match="sans citer la moindre source"):
        visibility_run(assessments=[decided])

    source = ElevationSource(
        kind="raster", role="target_ground", artifact_id="dtm@20260813T124251Z",
        path="06_geo/derived/dtm.tif", sha256="a" * 64,
        horizontal_crs="EPSG:2950", vertical_crs="CGVD 1928",
        sampling_method="rasterio.sample",
    )
    assert visibility_run(assessments=[decided], elevation_sources=[source])


def test_an_elevation_source_must_be_verifiable() -> None:
    """« 9/27 mesurés dans le nuage » explique un calcul, ne le vérifie pas."""
    from hotel_pipeline.schemas.visibility import ElevationSource

    for missing in ("path", "sha256", "horizontal_crs", "sampling_method"):
        fields = dict(
            kind="point_cloud", role="obstacle_height", path="tuile.LAZ",
            sha256="b" * 64, horizontal_crs="EPSG:2950",
            sampling_method="médiane classe 2",
        )
        fields[missing] = ""
        with pytest.raises(ValueError):
            ElevationSource(**fields)


# --- provenance d'élévation vérifiable ---------------------------------------


def elevation_source(**overrides):
    from hotel_pipeline.schemas.visibility import ElevationSource

    fields = dict(
        kind="point_cloud", role="obstacle_height", tile_id="23_3095048F08_DC",
        path="06_geo/lidar_raw/23_3095048F08_DC.LAZ", sha256="f" * 64,
        horizontal_crs="EPSG:2950", vertical_crs="CGVD 1928",
        sampling_method="médiane classe 2 et p95 classe 6",
    )
    fields.update(overrides)
    return ElevationSource(**fields)


def test_an_elevation_source_must_be_identifiable() -> None:
    with pytest.raises(ValueError, match="sans identifiant d'artefact ni de tuile"):
        elevation_source(tile_id=None)


def test_an_elevation_without_vertical_datum_is_refused() -> None:
    """CGVD 1928 et NAD83 diffèrent de plusieurs mètres : sans datum, un
    nombre n'est pas une altitude."""
    with pytest.raises(ValueError, match="sans référentiel vertical"):
        elevation_source(vertical_crs=None)


def test_a_truncated_digest_is_refused() -> None:
    with pytest.raises(ValueError, match="qui n'est pas un SHA-256"):
        elevation_source(sha256="fc6407b2fa28")
    with pytest.raises(ValueError, match="qui n'est pas un SHA-256"):
        elevation_source(sha256="z" * 64)


def test_a_qualification_report_and_its_digest_go_together() -> None:
    with pytest.raises(ValueError, match="vont ensemble ou pas du tout"):
        elevation_source(qualification_report="qualification_report_x.json")
    with pytest.raises(ValueError, match="vont ensemble ou pas du tout"):
        elevation_source(qualification_digest="d0e78d7bc5edac57")


# --- empreinte de base et péremption ------------------------------------------


def asset_manifest(tmp_path, **overrides):
    from hotel_pipeline.schemas import Asset, AssetManifest

    fields = dict(
        id="mapillary-1", source="mapillary", source_url_or_id="1",
        rights="open_data", ai_eligible=False, confidence=0.5, category="facade",
        checksum="a" * 64, camera_lat=45.5730, camera_lon=-73.4433,
        heading_deg=45.0,
    )
    fields.update(overrides)
    return AssetManifest(hotel_id="h", assets=[Asset(**fields)])


def test_the_projected_fields_do_not_perish_their_own_run(tmp_path) -> None:
    """Sans normalisation, appliquer un run le périmerait aussitôt."""
    from hotel_pipeline.geo.visibility_run import base_manifest_digest

    before = asset_manifest(tmp_path)
    after = asset_manifest(
        tmp_path,
        visibility_run_id="r1", visibility_run_digest="d1",
        visibility_assessment_id="vis-mapillary-1", line_of_sight_status="clear",
        occlusion_risk_by=["OBSTACLE_1"], occlusion_blocked_by=[],
    )

    assert base_manifest_digest(before) == base_manifest_digest(after)


def test_everything_else_still_perishes_the_run(tmp_path) -> None:
    """Un cap corrigé ou une revue humaine doivent bien périmer la mesure."""
    from hotel_pipeline.geo.visibility_run import base_manifest_digest

    reference = base_manifest_digest(asset_manifest(tmp_path))

    assert base_manifest_digest(asset_manifest(tmp_path, heading_deg=200.0)) != reference
    assert base_manifest_digest(asset_manifest(tmp_path, camera_lat=45.58)) != reference
    assert base_manifest_digest(
        asset_manifest(tmp_path, geometry_suitability="primary",
                       geometry_history=[{
                           "suitability": "primary", "decided_by": "hm",
                           "rationale": "façade franche", "evidence": ["cadrage"],
                           "reviewed_checksum": "a" * 64,
                       }])
    ) != reference


def test_the_old_single_occlusion_field_is_now_projected(tmp_path) -> None:
    """`occluded_by` fait partie des champs écrits : le remettre à `None` ne
    doit pas non plus périmer le run qui l'a corrigé."""
    from hotel_pipeline.geo.visibility_run import base_manifest_digest

    occluded = asset_manifest(tmp_path, occluded_by="way/999")
    cleared = asset_manifest(tmp_path, occluded_by=None)

    assert base_manifest_digest(occluded) == base_manifest_digest(cleared)


# --- végétation ---------------------------------------------------------------


def tree(x0, y0, x1, y1, height_m=8.0, occlusion=0.5, **kwargs):
    return Obstacle(
        kwargs.pop("feature_id", "TREE"),
        Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]),
        height_m=height_m,
        category="vegetation",
        occlusion_fraction=occlusion,
        **kwargs,
    )


def test_vegetation_creates_risk_with_unknown_height() -> None:
    """Un arbre sans hauteur connue est un risque, pas un blocage."""
    assessment = run([tree(-12, 20, 12, 25)])

    assert assessment.risk_unknown_height_fraction > 0
    assert assessment.proven_blocked_fraction == 0.0
    assert assessment.status is LineOfSightStatus.AT_RISK
    assert assessment.vegetation_occlusion_fraction == pytest.approx(0.5, abs=0.01)


def test_dense_tree_has_higher_occlusion_than_shrub() -> None:
    """Un grand arbre (occlusion 0.7) contribue plus qu'un arbuste (0.3)."""
    dense = run([tree(-12, 20, 12, 25, height_m=15.0, occlusion=0.7)])
    shrub = run([tree(-12, 20, 12, 25, height_m=3.0, occlusion=0.3)])

    assert dense.vegetation_occlusion_fraction == pytest.approx(0.7, abs=0.01)
    assert shrub.vegetation_occlusion_fraction == pytest.approx(0.3, abs=0.01)


def test_vegetation_behind_target_does_not_occlude() -> None:
    """Un arbre derrière la cible ne contribue pas à l'occlusion végétale."""
    assessment = run([tree(-12, 80, 12, 85, occlusion=0.8)])

    assert assessment.vegetation_occlusion_fraction == 0.0
    assert assessment.proven_clear_fraction == 1.0
