"""Régressions sur la sémantique des portes et le masquage.

Chaque test correspond à un défaut réel : une porte absente comptée comme
échec démontré, des métriques non mesurées valant PASS, un masque « eau »
identique au masque « ciel », un digest insensible au contenu.
"""

from __future__ import annotations

import numpy as np
import pytest

from hotel_pipeline.fidelity_gate import evaluate_fidelity
from hotel_pipeline.reconstruction_preprocess import _mask_sky, _mask_water
from hotel_pipeline.schemas.reconstruction import (
    Criticality,
    GateResult,
    GeoAlignmentGate,
    GeoGateCriteria,
    HoldoutPlan,
    HoldoutStrategy,
    NovelViewCriteria,
    NovelViewValidationGate,
    ReconstructionTarget,
    ReconstructionTargetKind,
    SparseConsensusGate,
    SupportType,
)


def _target(
    criticality: Criticality, *, geometry_confirmed: bool = True
) -> ReconstructionTarget:
    """Cible d'essai. Géométrie confirmée par défaut : ces cas-là portent sur
    les seuils, non sur la localisation."""
    return ReconstructionTarget(
        target_id="FACADE_PRIMARY",
        kind=ReconstructionTargetKind.SURFACE,
        criticality=criticality,
        allowed_support=[SupportType.MEASURED_PHOTO],
        geometry_state="confirmed" if geometry_confirmed else "inferred",
        geometry_confirmed=geometry_confirmed,
    )


def _passing_novel_gate() -> NovelViewValidationGate:
    return NovelViewValidationGate(
        holdout_plan=HoldoutPlan(strategy=HoldoutStrategy.LEAVE_ONE_VIEWPOINT_OUT),
        feature_inliers=0.9,
        edge_alignment=0.9,
        silhouette_iou=0.9,
        lpips=0.1,
        ssim=0.9,
        reprojection_px=0.5,
        structural_similarity=0.9,
        pass_criteria=NovelViewCriteria(),
        metrics_measured=True,
    )


# ---------------------------------------------------------------------------
# Portes absentes : inconnu, pas échec
# ---------------------------------------------------------------------------


def test_missing_gates_are_inconclusive_not_failure() -> None:
    """Une porte absente sur MUST_SHOW ne vaut pas un échec géométrique.

    `any([None, None])` était toujours faux : le verdict tombait en FAIL,
    c'est-à-dire une contradiction démontrée, là où rien n'avait été mesuré.
    """
    result = evaluate_fidelity(_target(Criticality.MUST_SHOW))
    assert result.overall is GateResult.INSUFFICIENT_EVIDENCE


def test_should_show_without_geo_gate_is_inconclusive() -> None:
    result = evaluate_fidelity(
        _target(Criticality.SHOULD_SHOW),
        novel_view_gate=_passing_novel_gate(),
        geo_gate=None,
    )
    assert result.overall is GateResult.INSUFFICIENT_EVIDENCE


def test_unsupported_geometry_is_a_real_failure() -> None:
    """La géométrie non mesurée sur MUST_SHOW reste un échec, pas un inconnu."""
    result = evaluate_fidelity(
        _target(Criticality.MUST_SHOW), unsupported_geometry_gate=True
    )
    assert result.overall is GateResult.FAIL


# ---------------------------------------------------------------------------
# Métriques non mesurées
# ---------------------------------------------------------------------------


def test_unmeasured_novel_view_never_passes() -> None:
    """Les défauts du schéma passeraient les seuils permissifs sans rendu.

    Avec `feature_inliers_min=0` et `reprojection_px_max=inf`, une porte vide
    satisfait toutes les comparaisons. `metrics_measured` bloque ce faux PASS.
    """
    unmeasured = NovelViewValidationGate(
        holdout_plan=HoldoutPlan(strategy=HoldoutStrategy.K_FOLD),
        pass_criteria=NovelViewCriteria(),
        metrics_measured=False,
        unmeasured_reason="aucun moteur de rendu dense raccordé",
    )
    result = evaluate_fidelity(
        _target(Criticality.OPTIONAL), novel_view_gate=unmeasured
    )
    assert result.overall is not GateResult.PASS


def test_measured_and_passing_gates_do_pass() -> None:
    """Le durcissement ne doit pas rendre le PASS inatteignable."""
    result = evaluate_fidelity(
        _target(Criticality.MUST_SHOW),
        sparse_gate=SparseConsensusGate(
            registration_rate=0.95,
            validated_registration_rate=0.95,
            validated_main_component_ratio=0.90,
            external_pose_consistency=True,
            largest_component_size=12,
            median_reprojection_px=1.2,
        ),
        geo_gate=GeoAlignmentGate(
            alignment_rmse_m=0.4,
            footprint_error_m=0.3,
            pass_criteria=GeoGateCriteria(),
        ),
        novel_view_gate=_passing_novel_gate(),
    )
    assert result.overall is GateResult.PASS


def test_measured_but_below_threshold_is_failure() -> None:
    """Mesuré et insuffisant : un échec démontré, distinct de l'inconnu."""
    weak = _passing_novel_gate()
    weak = weak.model_copy(update={"ssim": 0.0, "structural_similarity": 0.0})
    result = evaluate_fidelity(
        _target(Criticality.MUST_SHOW),
        sparse_gate=SparseConsensusGate(
            registration_rate=0.95,
            validated_registration_rate=0.95,
            validated_main_component_ratio=0.90,
            external_pose_consistency=True,
            largest_component_size=12,
            median_reprojection_px=1.2,
        ),
        geo_gate=GeoAlignmentGate(
            alignment_rmse_m=9.0,
            footprint_error_m=9.0,
            pass_criteria=GeoGateCriteria(),
        ),
        novel_view_gate=weak,
    )
    assert result.overall is GateResult.FAIL


def test_forbidden_target_is_not_applicable() -> None:
    result = evaluate_fidelity(_target(Criticality.FORBIDDEN))
    assert result.overall is GateResult.NOT_APPLICABLE


# ---------------------------------------------------------------------------
# Masques
# ---------------------------------------------------------------------------


def _scene() -> np.ndarray:
    """Ciel clair en haut, façade brique au milieu, piscine en bas à gauche."""
    img = np.zeros((400, 600, 3), np.uint8)
    img[:150, :] = (235, 206, 135)   # BGR : bleu ciel
    img[150:300, :] = (60, 60, 160)  # brique
    img[300:, :200] = (180, 130, 40)  # eau saturée
    return img


def test_sky_and_water_masks_are_distinct() -> None:
    """Les deux masques partageaient la même plage HSV : ils étaient un seul."""
    img = _scene()
    sky = _mask_sky(img)
    water = _mask_water(img)

    assert int((sky[:150] > 0).sum()) > 0, "le ciel doit être masqué"
    assert int((water[300:, :200] > 0).sum()) > 0, "l'eau doit être masquée"
    # Aucun recouvrement : ce sont deux surfaces différentes.
    assert int(np.logical_and(sky > 0, water > 0).sum()) == 0


def test_facade_is_not_masked() -> None:
    """Masquer la façade supprimerait ce qu'on cherche à reconstruire."""
    img = _scene()
    sky = _mask_sky(img)
    water = _mask_water(img)
    facade = np.logical_or(sky[150:300] > 0, water[150:300] > 0)
    assert int(facade.sum()) == 0


def test_sky_mask_requires_contact_with_top_of_frame() -> None:
    """Une surface bleue isolée au milieu du cadre n'est pas le ciel."""
    img = np.zeros((300, 300, 3), np.uint8)
    img[:, :] = (60, 60, 160)          # façade partout
    img[120:180, 120:180] = (235, 206, 135)  # panneau bleu isolé
    assert int((_mask_sky(img) > 0).sum()) == 0


# ---------------------------------------------------------------------------
# Cascade : la mesure prime sur le modèle
# ---------------------------------------------------------------------------


def _positioned_asset(**kw):
    """Asset positionné, cap mesuré, prêt pour le test de cadrage."""
    from hotel_pipeline.schemas import Asset

    from hotel_pipeline.schemas import AssetCategory, Rights

    base = dict(
        id="a1",
        source="mapillary",
        source_url_or_id="mapillary-1",
        rights=Rights.UNKNOWN,
        ai_eligible=False,
        confidence=0.5,
        category=AssetCategory.FACADE,
        checksum="0" * 64,
        camera_lat=45.5735,
        camera_lon=-73.4435,
        heading_deg=200.0,
        heading_is_measured=True,
    )
    base.update(kw)
    return Asset(**base)


#: Empreinte carrée d'environ 40 m, au sud-ouest de la caméra ci-dessus.
_WKT = (
    "POLYGON((-73.4440 45.5730, -73.4436 45.5730, "
    "-73.4436 45.5734, -73.4440 45.5734, -73.4440 45.5730))"
)


def test_model_false_no_longer_proves_absence() -> None:
    """Un classifieur à 26 % de rappel ne prouve pas une absence.

    Sur le corpus pilote, ce verdict retirait 271 assets que personne n'avait
    regardés. Il doit produire « à établir », pas « prouvé absent ».
    """
    from hotel_pipeline.classify_cascade import _target_visibility

    visible, reason = _target_visibility(
        _positioned_asset(), model_contains_building=False, target_in_fov=False
    )
    assert visible is None
    assert "à établir" in (reason or "")


def test_measured_framing_outranks_the_model() -> None:
    """Cap mesuré + empreinte cadrée établissent la cible malgré le modèle."""
    from hotel_pipeline.classify_cascade import _target_visibility

    visible, reason = _target_visibility(
        _positioned_asset(),
        model_contains_building=False,
        target_in_fov=False,
        framed=True,
        framing_reason="empreinte cadrée par un cap mesuré",
    )
    assert visible is True
    assert "cadrée" in (reason or "")


def test_measured_framing_can_exclude() -> None:
    """Hors champ d'un cap mesuré : une mesure, donc une exclusion légitime."""
    from hotel_pipeline.classify_cascade import _target_visibility

    visible, _ = _target_visibility(
        _positioned_asset(),
        model_contains_building=True,
        target_in_fov=False,
        framed=False,
        framing_reason="hors champ",
    )
    assert visible is False


def test_framing_never_overrides_an_unresolved_human_review() -> None:
    """Après un `unresolved` humain, la déduction constate mais n'établit pas.

    Le schéma refuse `True` dans ce cas ; la cascade ne doit donc jamais le
    produire.
    """
    from hotel_pipeline.classify_cascade import _target_visibility
    from hotel_pipeline.schemas import ReviewDecision

    from hotel_pipeline.schemas.assets import ReviewEntry

    entry = ReviewEntry(
        decided_by="tester",
        rationale="vue ambiguë : ni confirmée ni réfutée",
        evidence=["inspection directe"],
        reviewed_checksum="0" * 64,
        decision=ReviewDecision.UNRESOLVED,
    )
    asset = _positioned_asset(
        target_visibility_decision=ReviewDecision.UNRESOLVED,
        review_status="human_unresolved",
        review_history=[entry],
        reviewer="tester",
        review_rationale="vue ambiguë : ni confirmée ni réfutée",
        review_evidence=["inspection directe"],
        checksum="0" * 64,
    )
    visible, _ = _target_visibility(
        asset,
        model_contains_building=True,
        target_in_fov=False,
        framed=True,
        framing_reason="empreinte cadrée par un cap mesuré",
    )
    assert visible is None


def test_framing_abstains_without_a_measured_heading() -> None:
    """Un cap que nous avons choisi ne prouve rien : le test doit s'abstenir."""
    from hotel_pipeline.classify_cascade import _framing
    from hotel_pipeline.schemas.policy import DEFAULT_POLICY

    asset = _positioned_asset(heading_is_measured=False)
    framed, reason = _framing(asset, _WKT, DEFAULT_POLICY)
    assert framed is None
    assert reason is None


def test_framing_abstains_without_a_footprint() -> None:
    from hotel_pipeline.classify_cascade import _framing
    from hotel_pipeline.schemas.policy import DEFAULT_POLICY

    framed, reason = _framing(_positioned_asset(), None, DEFAULT_POLICY)
    assert framed is None
    assert reason is None


# ---------------------------------------------------------------------------
# CLI : l'entrée du programme doit rester la dernière instruction
# ---------------------------------------------------------------------------


def test_cli_entrypoint_is_the_last_statement() -> None:
    """`app()` placé au milieu du module cassait toute commande définie après.

    Sous `python -m hotel_pipeline.cli`, l'exécution s'arrêtait à `app()` :
    les ~1500 lignes suivantes n'étaient jamais définies, et `assets discover`
    échouait en `NameError: _discovery_scope`. Les tests important le module
    au lieu de l'exécuter, la suite restait verte.
    """
    import ast
    from pathlib import Path

    import hotel_pipeline.cli as cli_module

    source = Path(cli_module.__file__).read_text("utf-8")
    tree = ast.parse(source)

    guards = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and ast.dump(node.test).find("__main__") != -1
    ]
    assert guards, "le garde `if __name__ == \"__main__\"` a disparu"

    last_guard = guards[-1]
    following = [
        node for node in tree.body
        if getattr(node, "lineno", 0) > getattr(last_guard, "end_lineno", 0)
    ]
    assert not following, (
        "des définitions suivent `app()` : elles n'existeront pas à "
        f"l'exécution ({[type(n).__name__ for n in following[:5]]})"
    )


def test_kartaview_has_an_identity_strategy() -> None:
    """Une source sans stratégie d'identité fait échouer toute découverte."""
    from hotel_pipeline.schemas.acquisition import (
        IDENTITY_STRATEGIES,
        IdentityStrategy,
    )

    assert IDENTITY_STRATEGIES["kartaview"] is IdentityStrategy.PROVIDER_IMAGE


def test_inferred_geometry_is_inconclusive_not_excused() -> None:
    """Une géométrie inférée ne dispense pas une cible MUST_SHOW.

    Le site ne confirme pas les façades une à une : elles sont dérivées de
    l'empreinte et de l'orientation. Rabattre pour autant leur criticité sur
    CONTEXT_ONLY faisait répondre NOT_APPLICABLE — la surface même que le
    produit doit montrer cessait d'être évaluée, et le silence passait pour
    un succès.
    """
    result = evaluate_fidelity(
        _target(Criticality.MUST_SHOW, geometry_confirmed=False),
        sparse_gate=SparseConsensusGate(
            registration_rate=0.95,
            validated_registration_rate=0.95,
            validated_main_component_ratio=0.90,
            external_pose_consistency=True,
            largest_component_size=12,
            median_reprojection_px=1.2,
        ),
        geo_gate=GeoAlignmentGate(
            alignment_rmse_m=0.4,
            footprint_error_m=0.3,
            pass_criteria=GeoGateCriteria(),
        ),
        novel_view_gate=_passing_novel_gate(),
    )
    assert result.overall is GateResult.INSUFFICIENT_EVIDENCE
    assert result.criticality is Criticality.MUST_SHOW


# ---------------------------------------------------------------------------
# Porte C creuse : mesurée sur observations, donc réfutable
# ---------------------------------------------------------------------------


def _synthetic_model(tmp_path, point_noise: float = 0.0):
    """Écrit un modèle COLMAP normalisé complet : poses, points, observations."""
    import numpy as np

    rng = np.random.default_rng(0)
    pts = rng.normal(size=(120, 3)) * 2.0
    fx = fy = 800.0
    cx, cy = 400.0, 300.0
    out = tmp_path / "normalized"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cameras").write_text("1 PINHOLE 800 600 800 800 400 300\n")

    up = np.array([0.0, 0.0, 1.0])
    image_lines, tracks = [], {i: [] for i in range(len(pts))}
    for idx in range(6):
        angle = 2 * np.pi * idx / 6
        centre = np.array([12 * np.cos(angle), 12 * np.sin(angle), 2.0])
        f = -centre / np.linalg.norm(centre)
        s = np.cross(f, up)
        s /= np.linalg.norm(s)
        u = np.cross(f, s)
        R = np.vstack([s, u, f])
        t = -R @ centre
        # Quaternion depuis R (trace positive garantie ici).
        trace = float(np.trace(R))
        sq = 2.0 * np.sqrt(max(trace + 1.0, 1e-9))
        quat = (
            0.25 * sq,
            (R[2, 1] - R[1, 2]) / sq,
            (R[0, 2] - R[2, 0]) / sq,
            (R[1, 0] - R[0, 1]) / sq,
        )
        image_lines.append(
            f"{idx} {quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f} "
            f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} 1 asset-{idx}"
        )
        cam = (R @ pts.T).T + t
        obs = []
        for pid in range(len(pts)):
            if cam[pid, 2] <= 1e-6:
                continue
            u_px = fx * cam[pid, 0] / cam[pid, 2] + cx
            v_px = fy * cam[pid, 1] / cam[pid, 2] + cy
            if 0 <= u_px < 800 and 0 <= v_px < 600:
                obs.append(f"{u_px:.2f} {v_px:.2f} {pid}")
                tracks[pid].append(idx)
        image_lines.append(" ".join(obs))

    (out / "images").write_text("\n".join(image_lines) + "\n")

    stored = pts + (
        rng.normal(scale=point_noise, size=pts.shape) if point_noise else 0.0
    )
    (out / "points3D").write_text(
        "\n".join(
            f"{i} {p[0]:.4f} {p[1]:.4f} {p[2]:.4f} 180 140 100 1.0"
            for i, p in enumerate(stored)
        )
        + "\n"
    )
    return tmp_path


def test_gate_c_is_near_zero_on_a_faithful_model(tmp_path) -> None:
    """Une structure exacte prédit ses propres observations."""
    from hotel_pipeline.sparse_reprojection import measure_held_out

    root = _synthetic_model(tmp_path)
    result = measure_held_out(root, ["asset-2"], [f"asset-{i}" for i in [0, 1, 3, 4, 5]])

    assert result is not None
    assert result["reprojection_px"] < 0.1
    assert result["feature_inliers"] > 0.99


def test_gate_c_detects_a_hallucinated_structure(tmp_path) -> None:
    """Une géométrie fabriquée ne prédit pas la vue retirée.

    Trois métriques antérieures échouaient ici — l'une notait *mieux* un nuage
    bruité. La différence tient à l'observation : elle vient de la
    reconstruction, non d'un calcul qu'on referait soi-même.
    """
    from hotel_pipeline.sparse_reprojection import measure_held_out

    root = _synthetic_model(tmp_path, point_noise=0.5)
    result = measure_held_out(root, ["asset-2"], [f"asset-{i}" for i in [0, 1, 3, 4, 5]])

    assert result is not None
    assert result["reprojection_px"] > 5.0
    assert result["feature_inliers"] < 0.5


def test_gate_c_is_unmeasurable_without_observations(tmp_path) -> None:
    """Sans piste d'observations, aucune mesure ne peut réfuter le modèle."""
    from hotel_pipeline.sparse_reprojection import measure_held_out

    root = _synthetic_model(tmp_path)
    lines = (root / "normalized" / "images").read_text().splitlines()
    # Poses seules, lignes d'observations vidées : le fichier reste lisible,
    # mais plus rien n'y atteste où un point a été vu.
    stripped = []
    for index, line in enumerate(lines):
        stripped.append(line if index % 2 == 0 else "")
    (root / "normalized" / "images").write_text("\n".join(stripped) + "\n")

    result = measure_held_out(root, ["asset-2"], ["asset-0", "asset-1"])
    assert result is None


# ---------------------------------------------------------------------------
# Confiance de couverture par surface
# ---------------------------------------------------------------------------


def test_occupied_arc_handles_the_wraparound() -> None:
    """Des vues groupées au nord occupent un arc étroit, non 344°."""
    from hotel_pipeline.surface_coverage_confidence import occupied_arc

    assert occupied_arc([350.0, 355.0, 5.0, 10.0]) == pytest.approx(20.0)


def test_many_collinear_views_do_not_buy_confidence() -> None:
    """Quinze vues confondues n'ont aucune parallaxe : rien à trianguler."""
    from hotel_pipeline.surface_coverage_confidence import SurfaceEvidence, assess

    result = assess(
        SurfaceEvidence("FACADE", [180.0] * 15, [50.0] * 15)
    )
    assert result.confidence == pytest.approx(0.0)
    assert result.limiting_factor == "parallaxe"


def test_a_well_observed_surface_is_reconstructible() -> None:
    from hotel_pipeline.surface_coverage_confidence import SurfaceEvidence, assess

    bearings = [150.0 + 8 * i for i in range(10)]
    result = assess(SurfaceEvidence("FACADE", bearings, [60.0] * 10))
    assert result.verdict == "reconstructible"
    assert result.confidence > 0.7


def test_a_surface_with_no_usable_view_is_unreachable() -> None:
    """Au-delà de la limite de proximité, une vue n'apporte pas de façade."""
    from hotel_pipeline.surface_coverage_confidence import SurfaceEvidence, assess

    result = assess(SurfaceEvidence("REAR", [10.0, 20.0], [400.0, 500.0]))
    assert result.verdict == "unreachable"
    assert result.n_usable == 0


# ---------------------------------------------------------------------------
# Prominence du sujet : mesurée sur les pixels, et distincte de l'identité
# ---------------------------------------------------------------------------


def _reading(score: float):
    from hotel_pipeline.subject_prominence import ProminenceReading, _verdict

    return ProminenceReading(
        asset_id="a1", score=score, verdict=_verdict(score)
    )


def test_prominence_and_identity_are_separate_questions() -> None:
    """« Un grand bâtiment remplit le cadre » ne dit pas **lequel**.

    Sur le pilote, les neuf vues les mieux notées comprenaient trois
    concessions automobiles voisines. Sans le prédicat d'identité, elles
    seraient devenues des références d'apparence de l'hôtel.
    """
    from hotel_pipeline.subject_prominence import is_reference_grade

    ok, reason = is_reference_grade(_reading(0.95), target_building_visible=None)
    assert ok is False
    assert "identité" in reason

    ok, _ = is_reference_grade(_reading(0.95), target_building_visible=True)
    assert ok is True


def test_an_incidental_subject_is_not_reference_grade() -> None:
    from hotel_pipeline.subject_prominence import is_reference_grade

    ok, reason = is_reference_grade(_reading(0.3), target_building_visible=True)
    assert ok is False
    assert "proéminent" in reason


def test_an_unreadable_image_is_unmeasured_not_zero() -> None:
    """Un fichier illisible n'est pas une image sans bâtiment."""
    from pathlib import Path

    from hotel_pipeline.subject_prominence import ProminenceReader

    reader = ProminenceReader.__new__(ProminenceReader)
    reader._classifier = None
    reader._text_features = None
    result = reader.read("a1", Path("/definitely/absent.jpg"))

    assert result.measured is False
    assert result.verdict == "unmeasured"


def test_measured_prominence_overrides_the_geometric_guess() -> None:
    """La preuve pixel remplace la prédiction, elle ne la pondère pas.

    La géométrie notait 1,0 une vue à 63 m où l'hôtel était un bloc lointain
    derrière un parc-o-bus : elle ignore ce qui se met devant.
    """
    from hotel_pipeline.appearance_quality import AppearanceEvidence, assess

    optimistic_geometry = AppearanceEvidence(
        asset_id="a1", sharpness=600.0, brightness=128.0, distance_m=63.0
    )
    assert assess(optimistic_geometry).prominence > 0.4

    with_pixels = AppearanceEvidence(
        asset_id="a1", sharpness=600.0, brightness=128.0, distance_m=63.0,
        measured_prominence=0.02,
    )
    result = assess(with_pixels)
    assert result.prominence == pytest.approx(0.02)
    assert result.verdict in ("weak", "unusable")


# ---------------------------------------------------------------------------
# Couverture par segment : l'union de vues partielles vaut mieux qu'un tri
# ---------------------------------------------------------------------------


def _segments(depths: list[int]):
    from hotel_pipeline.facade_segments import FacadeSegmentReport, SegmentCoverage

    return FacadeSegmentReport(
        facade_id="FACADE_PRIMARY",
        segments=[
            SegmentCoverage(index=i, views=[f"v{i}-{j}" for j in range(d)])
            for i, d in enumerate(depths)
        ],
    )


def test_partial_views_combine_into_full_coverage() -> None:
    """Deux vues ne montrant chacune qu'une moitié couvrent le mur entier.

    C'est ce qu'un classement par vue rate : chacune serait « partielle »,
    leur réunion est complète.
    """
    from hotel_pipeline.facade_segments import FacadeSegmentReport, SegmentCoverage

    left = ["a"] * 5 + [""] * 5
    report = FacadeSegmentReport(
        facade_id="F",
        segments=[
            SegmentCoverage(index=i, views=["a"] if i < 5 else ["b"])
            for i in range(10)
        ],
    )
    assert report.union_fraction == pytest.approx(1.0)
    # Aucune vue seule ne montre plus de la moitié.
    assert report.best_single_fraction == pytest.approx(0.5)


def test_unseen_segments_are_named_not_averaged() -> None:
    """Un trou se nomme : c'est lui qu'on demande à photographier."""
    from hotel_pipeline.facade_segments import capture_request

    report = _segments([3, 3, 0, 0, 3, 3, 1, 3, 3, 3])
    assert report.unseen() == [2, 3]
    assert report.thin() == [6]
    assert report.verdict() == "partial"

    request = capture_request(report)
    assert "2, 3" in request
    assert "6" in request


def test_a_fully_corroborated_facade_asks_for_nothing() -> None:
    from hotel_pipeline.facade_segments import capture_request

    report = _segments([4] * 10)
    assert report.verdict() == "corroborated"
    assert capture_request(report) is None


def test_a_single_view_everywhere_is_thin_not_corroborated() -> None:
    """Une seule vue par tronçon couvre sans corroborer : rien ne la vérifie."""
    report = _segments([1] * 10)
    assert report.union_fraction == pytest.approx(1.0)
    assert report.corroborated_fraction == pytest.approx(0.0)
    assert report.verdict() == "thin"


def test_an_unseen_facade_reports_zero_not_an_average() -> None:
    report = _segments([0] * 10)
    assert report.verdict() == "unseen"
    assert report.union_fraction == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Propagation : la grandeur mesurée survit au seuil qui l'a tranchée
# ---------------------------------------------------------------------------


def test_framing_strength_is_continuous_across_the_threshold() -> None:
    """44,9° et 45,1° donnaient des verdicts opposés, sans rien d'autre.

    Le booléen reste — il faut trancher — mais la force mesurée l'accompagne :
    en aval, un classement peut voir que les deux vues étaient presque
    identiques.
    """
    from hotel_pipeline.visibility import Visibility

    just_inside = Visibility(
        visible=True, distance_m=60.0, offset_deg=44.9, reason=""
    )
    just_outside = Visibility(
        visible=False, distance_m=60.0, offset_deg=45.1, reason=""
    )

    assert just_inside.visible is not just_outside.visible
    # Les forces, elles, sont voisines : l'information n'a pas été détruite.
    assert abs(just_inside.framing_strength - just_outside.framing_strength) < 0.01
    # Et l'axe optique reste franchement mieux noté qu'un bord de champ.
    on_axis = Visibility(visible=True, distance_m=60.0, offset_deg=0.0, reason="")
    assert on_axis.framing_strength == pytest.approx(1.0)


def test_reference_strength_keeps_the_score_the_boolean_discards() -> None:
    from hotel_pipeline.subject_prominence import ProminenceReading, reference_strength

    reading = ProminenceReading(asset_id="a", score=0.59, verdict="subject_incidental")
    strength, _ = reference_strength(reading, target_building_visible=True)
    assert strength == pytest.approx(0.59)


def test_reference_strength_distinguishes_unmeasured_from_absent() -> None:
    """`None` n'est pas 0,0 : l'un ignore, l'autre affirme l'absence."""
    from hotel_pipeline.subject_prominence import ProminenceReading, reference_strength

    unread = ProminenceReading(
        asset_id="a", score=0.0, verdict="unmeasured",
        measured=False, reason="fichier absent",
    )
    strength, _ = reference_strength(unread, target_building_visible=True)
    assert strength is None

    unidentified = ProminenceReading(asset_id="a", score=0.9, verdict="subject_prominent")
    strength, reason = reference_strength(unidentified, target_building_visible=None)
    assert strength == pytest.approx(0.0)
    assert "identité" in reason


# ---------------------------------------------------------------------------
# Composition : les marges se composent au lieu de se perdre dans un ET
# ---------------------------------------------------------------------------


def test_three_marginal_passes_differ_from_three_strong_ones() -> None:
    """Un `ET` booléen les confondait : tous « passés », donc identiques."""
    from hotel_pipeline.scene_confidence import compose

    marginal = compose("F", coverage=0.55, appearance=0.55, fidelity=0.55)
    strong = compose("F", coverage=0.95, appearance=0.95, fidelity=0.95)

    assert marginal.joint is not None and strong.joint is not None
    assert marginal.joint < strong.joint
    assert marginal.verdict == "marginal"
    assert strong.verdict == "carries_video"


def test_one_null_axis_annuls_the_joint_confidence() -> None:
    """Une surface bien couverte dont aucune image n'est exploitable ne porte
    pas de vidéo : la moyenne géométrique le dit, l'arithmétique non."""
    from hotel_pipeline.scene_confidence import compose

    result = compose("F", coverage=1.0, appearance=0.0, fidelity=1.0)
    assert result.joint == pytest.approx(0.0)
    assert result.limiting_factor == "apparence"


def test_a_missing_axis_is_unmeasured_not_zero() -> None:
    from hotel_pipeline.scene_confidence import compose

    result = compose("F", coverage=0.9, appearance=0.9, fidelity=None)
    assert result.joint is None
    assert result.verdict == "unmeasured"
    assert result.missing_axes == ["fidélité"]


def test_the_deliverable_is_bounded_by_its_weakest_required_surface() -> None:
    """Une vidéo n'est pas la moyenne de ses façades : elle bute sur la pire."""
    from hotel_pipeline.scene_confidence import compose, deliverable_confidence

    surfaces = [
        compose("FACADE_PRIMARY", coverage=0.95, appearance=0.95, fidelity=0.95),
        compose("ENTRANCE", coverage=0.45, appearance=0.45, fidelity=0.45),
    ]
    joint, reason = deliverable_confidence(surfaces)
    assert joint == pytest.approx(0.45, abs=0.01)
    assert "ENTRANCE" in reason


def test_an_unrequired_weak_surface_does_not_block_the_deliverable() -> None:
    from hotel_pipeline.scene_confidence import compose, deliverable_confidence

    surfaces = [
        compose("FACADE_PRIMARY", coverage=0.95, appearance=0.95, fidelity=0.95),
        compose("FACADE_REAR", coverage=0.0, appearance=0.0, fidelity=0.0),
    ]
    joint, _ = deliverable_confidence(surfaces, required={"FACADE_PRIMARY"})
    assert joint == pytest.approx(0.95, abs=0.01)


# ---------------------------------------------------------------------------
# Portabilité : rien du pilote ne doit être figé dans le code
# ---------------------------------------------------------------------------


def test_prominence_prompts_come_from_policy() -> None:
    """Un motel de banlieue ne se décrit pas comme un hôtel de centre-ville.

    Figées au module, les descriptions rendaient le portage impossible sans
    éditer le code.
    """
    from hotel_pipeline.schemas.policy import PipelinePolicy
    from hotel_pipeline.subject_prominence import ProminenceReader

    policy = PipelinePolicy()
    policy.geometry.subject_prompt = "a glass office tower seen from a plaza"
    policy.geometry.absence_prompts = ("a canal", "a market square")
    policy.geometry.prominence_accept = 0.8
    policy.geometry.prominence_partial = 0.3

    reader = ProminenceReader.__new__(ProminenceReader)
    ProminenceReader.__init__(reader, classifier=object(), policy=policy)

    assert reader.subject_prompts == ("a glass office tower seen from a plaza",)
    assert reader.absence_prompts == ("a canal", "a market square")
    assert reader.accept_threshold == pytest.approx(0.8)
    assert reader.partial_threshold == pytest.approx(0.3)


def test_verdict_thresholds_are_parameters_not_constants() -> None:
    from hotel_pipeline.subject_prominence import _verdict

    assert _verdict(0.5, accept=0.4, partial=0.1) == "subject_prominent"
    assert _verdict(0.5, accept=0.9, partial=0.1) == "subject_incidental"
    assert _verdict(0.05, accept=0.9, partial=0.1) == "subject_absent"


def test_selection_prefers_verified_prominence_over_proximity() -> None:
    """Classer par distance ramenait la rue arrière, bouchée par des pavillons
    absents du modèle d'obstacles."""
    from hotel_pipeline.recrop_opportunities import RecropOpportunity, select_minimal

    near_but_blocked = RecropOpportunity(
        panorama_id="near", asset_id="a", facade_id="F",
        heading_deg=0.0, distance_m=50.0, covers=[0, 1, 2],
        verified_prominence=0.02,
    )
    far_but_clear = RecropOpportunity(
        panorama_id="far", asset_id="b", facade_id="F",
        heading_deg=0.0, distance_m=150.0, covers=[0, 1, 2],
        verified_prominence=0.99,
    )
    chosen = select_minimal([near_but_blocked, far_but_clear], 3, max_requests=1)
    assert chosen[0].panorama_id == "far"


def test_unverified_candidates_do_not_outrank_verified_ones() -> None:
    """`None` n'est pas un bon score : jamais vérifié n'est pas jamais bon."""
    from hotel_pipeline.recrop_opportunities import RecropOpportunity, select_minimal

    unverified = RecropOpportunity(
        panorama_id="unknown", asset_id="a", facade_id="F",
        heading_deg=0.0, distance_m=40.0, covers=[0, 1],
    )
    verified = RecropOpportunity(
        panorama_id="seen", asset_id="b", facade_id="F",
        heading_deg=0.0, distance_m=200.0, covers=[0, 1],
        verified_prominence=0.9,
    )
    chosen = select_minimal([unverified, verified], 2, max_requests=1)
    assert chosen[0].panorama_id == "seen"


def test_fov_follows_distance() -> None:
    """Un champ fixe de 70° remplit l'image de stationnement dès 120 m."""
    from hotel_pipeline.recrop_opportunities import fov_for

    assert fov_for(50) > fov_for(150) > fov_for(300)
    # À 150 m, l'optimum mesuré sur le pilote était ~25°.
    assert 20.0 <= fov_for(150) <= 30.0


# ---------------------------------------------------------------------------
# Boucle de vérification : les pixels corrigent la géométrie, et ça persiste
# ---------------------------------------------------------------------------


def _opportunity(pano: str, heading: float, fov: float, covers, distance=100.0):
    from hotel_pipeline.recrop_opportunities import RecropOpportunity

    return RecropOpportunity(
        panorama_id=pano, asset_id=pano, facade_id="F",
        heading_deg=heading, distance_m=distance, covers=list(covers),
        fov_deg=fov,
    )


def test_the_register_keys_on_fov_not_only_heading() -> None:
    """Le même cap à 70° et à 25° ne montre pas la même chose : 0,396 contre
    0,997 sur le pilote."""
    from hotel_pipeline.recrop_verification import VerificationRegister

    register = VerificationRegister()
    register.record("p", 140.0, 70.0, score=0.30, verdict="subject_absent")
    register.record("p", 140.0, 25.0, score=0.99, verdict="subject_prominent")

    assert register.get("p", 140.0, 70.0)["score"] == pytest.approx(0.30)
    assert register.get("p", 140.0, 25.0)["score"] == pytest.approx(0.99)


def test_verification_persists_and_redirects_the_next_selection() -> None:
    """Sans écriture, la correction pixel était perdue et la sélection
    retombait sur la distance — celle qui proposait la rue bouchée."""
    from hotel_pipeline.recrop_opportunities import select_minimal
    from hotel_pipeline.recrop_verification import (
        VerificationRegister, apply_known,
    )

    near_blocked = _opportunity("near", 0.0, 50.0, [0, 1, 2], distance=50.0)
    far_clear = _opportunity("far", 0.0, 25.0, [0, 1, 2], distance=150.0)

    # Avant vérification : la distance décide, et elle se trompe.
    assert select_minimal(
        [near_blocked, far_clear], 3, max_requests=1
    )[0].panorama_id == "near"

    register = VerificationRegister()
    register.record("near", 0.0, 50.0, score=0.02, verdict="subject_absent")
    register.record("far", 0.0, 25.0, score=0.99, verdict="subject_prominent")

    known, unknown = apply_known([near_blocked, far_clear], register)
    assert (known, unknown) == (2, 0)
    assert select_minimal(
        [near_blocked, far_clear], 3, max_requests=1
    )[0].panorama_id == "far"


def test_a_known_recrop_is_never_refetched() -> None:
    """Chaque acquisition est facturée : un recadrage jugé ne se redemande pas."""
    from pathlib import Path

    from hotel_pipeline.recrop_verification import VerificationRegister, fetch_and_read

    register = VerificationRegister()
    register.record("p", 10.0, 30.0, score=0.5, verdict="subject_incidental")

    calls: list[str] = []

    def fetcher(opportunity):  # noqa: ANN001
        calls.append(opportunity.panorama_id)
        return b"x" * 6000

    read, skipped = fetch_and_read(
        [_opportunity("p", 10.0, 30.0, [0])], register,
        cache_dir=Path("/tmp/never-used"), fetcher=fetcher,
    )
    assert calls == []
    assert (read, skipped) == (0, 0)


def test_an_unfetchable_recrop_is_recorded_as_unmeasured_not_absent() -> None:
    """Une panne réseau n'est pas un cadrage vide : `None`, jamais 0,0."""
    import tempfile
    from pathlib import Path

    from hotel_pipeline.recrop_verification import VerificationRegister, fetch_and_read

    register = VerificationRegister()

    def failing(opportunity):  # noqa: ANN001
        raise RuntimeError("quota dépassé")

    read, skipped = fetch_and_read(
        [_opportunity("p", 10.0, 30.0, [0])], register,
        cache_dir=Path(tempfile.mkdtemp()), fetcher=failing,
    )
    assert (read, skipped) == (0, 1)
    row = register.get("p", 10.0, 30.0)
    assert row["score"] is None
    assert row["verdict"] == "unfetched"


# ---------------------------------------------------------------------------
# Trajectoire ancrée : les poses suivent ce qu'on a vu
# ---------------------------------------------------------------------------


class _FakeScene:
    """Scène minimale : un centroïde et des positions de panoramas."""

    def __init__(self, viewpoints):
        from shapely.geometry import Point, Polygon

        self.footprint = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
        self.viewpoints = viewpoints
        self._Point = Point


def _scene_with(anchors):
    """anchors: {panorama_id: (x, y)} en coordonnées projetées."""
    return _FakeScene([(f"a-{k}", k, v) for k, v in anchors.items()])


def test_a_pose_without_a_vantage_stays_unanchored() -> None:
    """Un arc sans référence reste déclaré, jamais interpolé : combler le trou
    ferait croire à une image que personne n'a."""
    from hotel_pipeline.camera_path_real import build

    scene = _scene_with({"north": (10.0, 200.0)})
    verifications = [
        {"panorama_id": "north", "heading_deg": 180.0, "fov_deg": 30.0,
         "pitch_deg": 0.0, "score": 0.9, "facade_id": "F"},
    ]
    path = build("h", scene, verifications, poses=8)

    assert any(p.anchored for p in path.poses)
    assert any(not p.anchored for p in path.poses)
    assert path.anchored_fraction < 1.0
    assert path.gaps, "un arc sans vantage doit être déclaré"


def test_altitude_descends_from_start_to_end() -> None:
    from hotel_pipeline.camera_path_real import END_ALTITUDE_M, START_ALTITUDE_M, build

    scene = _scene_with({"p": (10.0, 200.0)})
    path = build("h", scene, [], poses=10)

    assert path.poses[0].altitude_m == pytest.approx(START_ALTITUDE_M)
    assert path.poses[-1].altitude_m == pytest.approx(END_ALTITUDE_M)
    altitudes = [p.altitude_m for p in path.poses]
    assert altitudes == sorted(altitudes, reverse=True)


def test_a_weak_vantage_does_not_anchor_a_pose() -> None:
    """Sous le seuil partiel, la vue ne montre pas le sujet : elle n'atteste
    rien."""
    from hotel_pipeline.camera_path_real import build

    scene = _scene_with({"weak": (10.0, 200.0)})
    verifications = [
        {"panorama_id": "weak", "heading_deg": 180.0, "fov_deg": 30.0,
         "score": 0.01, "facade_id": "F"},
    ]
    path = build("h", scene, verifications, poses=4)
    assert path.anchored_fraction == pytest.approx(0.0)
    assert path.verdict() in ("unsupported", "mostly_unsupported")


def test_reference_requests_are_deduplicated() -> None:
    """Deux poses voisines partagent un vantage : le payer deux fois n'ajoute
    aucune référence."""
    from hotel_pipeline.camera_path_real import build, reference_requests

    scene = _scene_with({"p": (10.0, 200.0)})
    verifications = [
        {"panorama_id": "p", "heading_deg": 180.0, "fov_deg": 30.0,
         "pitch_deg": 0.0, "score": 0.9, "facade_id": "F"},
    ]
    path = build("h", scene, verifications, poses=12)
    anchored = sum(1 for p in path.poses if p.anchored)

    assert anchored > 1
    assert len(reference_requests(path)) == 1


def test_the_path_declares_that_altitude_is_not_attested() -> None:
    """L'imagerie de rue vit à ~2,5 m : aucune vue à 40 m n'est mesurée, et le
    paquet doit le dire plutôt que de laisser croire le contraire."""
    from hotel_pipeline.camera_path_real import build

    scene = _scene_with({"p": (10.0, 200.0)})
    payload = build("h", scene, [], poses=4).as_dict()
    assert any("40 m" in caveat for caveat in payload["caveats"])


def test_feedforward_never_fabricates_poses(tmp_path) -> None:
    """Un échec complet ne doit pas se présenter comme un succès parfait.

    Le repli écrivait une pose identique par image ; le parseur de métriques
    ne fait que compter les lignes, si bien que `registered_ratio` valait 1,0
    alors qu'aucune reconstruction n'avait eu lieu.
    """
    from hotel_pipeline.reconstruction_run import (
        ReconstructionRefused, _export_feed_forward_to_colmap,
    )

    with pytest.raises(ReconstructionRefused):
        _export_feed_forward_to_colmap(
            object(), tmp_path, ["a1", "a2", "a3"], {}
        )
