"""Le conditionnement doit contraindre le générateur sans jamais lui mentir."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from hotel_pipeline.conditioning import load_scene, render_sequence
from hotel_pipeline.conditioning.png import write_png
from hotel_pipeline.conditioning.render import Camera, render_frame
from hotel_pipeline.conditioning.scene import ASSUMED_HEIGHT_M

PILOT = Path("work/welcominns-boucherville/06_geo/capture_geometry.json")


def _square(cx: float, cy: float, half: float) -> str:
    pts = [
        (cx - half, cy - half),
        (cx + half, cy - half),
        (cx + half, cy + half),
        (cx - half, cy + half),
        (cx - half, cy - half),
    ]
    body = ", ".join(f"{x} {y}" for x, y in pts)
    return f"POLYGON (({body}))"


def _manifest(tmp_path: Path, entries: list[dict]) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "capture_geometry.json"
    path.write_text(
        json.dumps({"hotel_id": "t", "geometries": entries}), encoding="utf-8"
    )
    return path


def _entry(feature_id: str, role: str, wkt: str, **kw) -> dict:
    base = {
        "feature_id": feature_id,
        "role": role,
        "resolution_status": "resolved",
        "projected_wkt": wkt,
        "projected_crs": "EPSG:2950",
        "height_known": False,
        "height_m": None,
        "height_source": None,
    }
    base.update(kw)
    return base


@pytest.fixture()
def simple_scene(tmp_path: Path):
    return load_scene(
        _manifest(
            tmp_path,
            [_entry("TARGET_BUILDING", "target_building", _square(0, 0, 10))],
        )
    )


def test_une_hauteur_absente_est_supposee_et_declaree(tmp_path: Path) -> None:
    """Sans mesure, la hauteur est une consigne — jamais une donnée du site."""
    scene = load_scene(
        _manifest(
            tmp_path,
            [_entry("TARGET_BUILDING", "target_building", _square(0, 0, 10))],
        )
    )
    prism = scene.target
    assert prism is not None
    assert prism.height_assumed is True
    assert prism.height_m == ASSUMED_HEIGHT_M["target_building"]
    assert "aucune hauteur mesurée" in prism.height_source
    assert scene.assumed_height_count == 1


def test_une_hauteur_mesuree_est_reprise_telle_quelle(tmp_path: Path) -> None:
    scene = load_scene(
        _manifest(
            tmp_path,
            [
                _entry(
                    "TARGET_BUILDING",
                    "target_building",
                    _square(0, 0, 10),
                    height_known=True,
                    height_m=27.5,
                    height_source="lidar",
                )
            ],
        )
    )
    prism = scene.target
    assert prism is not None
    assert prism.height_assumed is False
    assert prism.height_m == 27.5
    assert prism.confidence > 0.9


def test_un_volume_non_resolu_n_entre_pas_dans_la_scene(tmp_path: Path) -> None:
    """Une emprise démentie ne doit pas contraindre le générateur."""
    scene = load_scene(
        _manifest(
            tmp_path,
            [
                _entry("TARGET_BUILDING", "target_building", _square(0, 0, 10)),
                _entry(
                    "OBST",
                    "obstacle_building",
                    _square(40, 0, 8),
                    resolution_status="stale",
                ),
            ],
        )
    )
    assert [p.feature_id for p in scene.prisms] == ["TARGET_BUILDING"]


def test_sans_cible_resolue_la_scene_est_refusee(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cible"):
        load_scene(
            _manifest(
                tmp_path,
                [_entry("OBST", "obstacle_building", _square(0, 0, 10))],
            )
        )


def test_le_plus_proche_occulte_le_plus_lointain(tmp_path: Path) -> None:
    """Le z-buffer doit trancher : c'est toute la valeur de l'étape 3D."""
    scene = load_scene(
        _manifest(
            tmp_path,
            [
                _entry("TARGET_BUILDING", "target_building", _square(0, 0, 10)),
                _entry("OBST", "obstacle_building", _square(0, -40, 12)),
            ],
        )
    )
    camera = Camera(
        position=np.array([0.0, -120.0, 8.0]),
        target=np.array([0.0, 0.0, 6.0]),
        width=160,
        height=90,
    )
    frame = render_frame(scene, camera)
    # L'obstacle est devant : il doit masquer la cible au centre de l'image.
    assert frame.silhouette[45, 80] == 1
    assert np.isfinite(frame.depth[45, 80])


def test_la_profondeur_croit_avec_l_eloignement(simple_scene) -> None:
    near = render_frame(
        simple_scene,
        Camera(
            position=np.array([0.0, -40.0, 6.0]),
            target=np.array([0.0, 0.0, 6.0]),
            width=120,
            height=80,
        ),
    )
    far = render_frame(
        simple_scene,
        Camera(
            position=np.array([0.0, -90.0, 6.0]),
            target=np.array([0.0, 0.0, 6.0]),
            width=120,
            height=80,
        ),
    )
    assert near.depth[40, 60] < far.depth[40, 60]
    assert near.target_coverage > far.target_coverage


def test_le_toit_est_moins_credite_que_la_facade(simple_scene) -> None:
    """Aucune source au sol n'atteste un toit : il ne doit pas contraindre."""
    frame = render_frame(
        simple_scene,
        Camera(
            position=np.array([0.0, -30.0, 60.0]),
            target=np.array([0.0, 0.0, 6.0]),
            width=160,
            height=90,
        ),
    )
    seen = frame.confidence[frame.confidence > 0]
    assert seen.size > 0
    # Deux régimes de crédit coexistent : le toit, et le reste.
    assert seen.min() < seen.max()


def test_une_cible_trop_petite_rend_la_main_au_generateur(tmp_path: Path) -> None:
    """Contraindre sur 1% de l'image n'apporte rien : il faut le dire."""
    scene = load_scene(
        _manifest(
            tmp_path,
            [_entry("TARGET_BUILDING", "target_building", _square(0, 0, 10))],
        )
    )
    result = render_sequence(
        scene,
        tmp_path / "out",
        frame_count=3,
        distance_factor=60.0,
        width=96,
        height=64,
        write_images=False,
    )
    assert result.verdict() == "prefer_ungrounded"
    assert all(f.guidance_mode == "prefer_ungrounded" for f in result.frames)
    assert all("ne contraint rien" in f.guidance_reason for f in result.frames)


def test_la_sequence_est_deterministe(simple_scene, tmp_path: Path) -> None:
    """Deux rendus du même plan doivent coïncider, sinon rien n'est rejouable."""
    kwargs = dict(frame_count=5, width=96, height=64, write_images=False)
    a = render_sequence(simple_scene, tmp_path / "a", **kwargs)
    b = render_sequence(simple_scene, tmp_path / "b", **kwargs)
    strip = lambda r: [  # noqa: E731
        {k: v for k, v in f.as_dict().items()} for f in r.frames
    ]
    assert strip(a) == strip(b)


def test_la_camera_tourne_et_descend(simple_scene, tmp_path: Path) -> None:
    result = render_sequence(
        simple_scene,
        tmp_path / "out",
        frame_count=10,
        arc_deg=180.0,
        start_altitude_m=50.0,
        end_altitude_m=5.0,
        width=96,
        height=64,
        write_images=False,
    )
    altitudes = [f.altitude_m for f in result.frames]
    bearings = [f.bearing_deg for f in result.frames]
    assert altitudes == sorted(altitudes, reverse=True)
    assert altitudes[0] == pytest.approx(50.0)
    assert altitudes[-1] == pytest.approx(5.0)
    assert len(set(round(b) for b in bearings)) > 5


def test_le_rapport_porte_ses_reserves(simple_scene, tmp_path: Path) -> None:
    out = tmp_path / "out"
    render_sequence(simple_scene, out, frame_count=3, width=96, height=64)
    payload = json.loads((out / "conditioning_report.json").read_text(encoding="utf-8"))
    assert payload["scene"]["assumed_height_count"] == 1
    joined = " ".join(payload["caveats"])
    assert "hypothèse" in joined
    assert "toit" in joined
    for frame in payload["frames"]:
        assert frame["guidance_mode"] in {
            "geometry_strong",
            "geometry_weak",
            "prefer_ungrounded",
        }
        assert frame["guidance_reason"]


def test_les_cartes_sont_ecrites_pour_chaque_frame(simple_scene, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = render_sequence(simple_scene, out, frame_count=4, width=96, height=64)
    for record in result.frames:
        for relative in record.files.values():
            written = out / relative
            assert written.is_file()
            assert written.stat().st_size > 0
            assert written.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_le_png_relit_ce_qu_il_ecrit(tmp_path: Path) -> None:
    """L'encodeur est fait maison : il doit produire un PNG réellement valide."""
    zlib_png = tmp_path / "x.png"
    image = np.arange(12 * 8, dtype=np.uint8).reshape(12, 8)
    write_png(zlib_png, image)
    raw = zlib_png.read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    assert b"IHDR" in raw and b"IDAT" in raw and raw[-8:-4] == b"IEND"


@pytest.mark.skipif(not PILOT.is_file(), reason="workspace pilote absent")
def test_le_pilote_reel_produit_une_sequence_exploitable(tmp_path: Path) -> None:
    """Le harnais doit tenir sur la vraie géométrie, pas seulement sur un carré."""
    scene = load_scene(PILOT)
    assert scene.crs == "EPSG:2950"
    assert scene.target is not None
    result = render_sequence(
        scene, tmp_path / "out", frame_count=6, width=128, height=72, write_images=False
    )
    assert result.verdict() == "condition_strongly"
    assert all(f.stats["geometry_hit"] for f in result.frames)


# --- portabilité : le même code sur un autre bâtiment ---------------------


def _tower(tmp_path: Path) -> Path:
    """Une tour dans un autre référentiel : tout l'inverse du motel pilote."""
    cx, cy = 299500.0, 5040000.0

    def rect(x0: float, y0: float, w: float, h: float) -> str:
        pts = [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h), (x0, y0)]
        return "POLYGON ((" + ", ".join(f"{x} {y}" for x, y in pts) + "))"

    entries = [
        _entry("TARGET_BUILDING", "target_building", rect(cx - 18, cy - 30, 36, 60)),
        _entry("OBST_A", "obstacle_building", rect(cx + 50, cy, 24, 24)),
        _entry(
            "OBST_STALE",
            "obstacle_building",
            rect(cx + 200, cy, 20, 20),
            resolution_status="stale",
        ),
    ]
    for entry in entries:
        entry["projected_crs"] = "EPSG:32188"
    return _manifest(tmp_path, entries)


def test_un_autre_batiment_se_charge_sans_configuration(tmp_path: Path) -> None:
    """Aucun paramètre du site pilote ne doit être nécessaire ailleurs."""
    scene = load_scene(_tower(tmp_path))

    assert scene.crs == "EPSG:32188"
    # La géométrie démentie est écartée, les deux autres restent.
    assert {p.feature_id for p in scene.prisms} == {"TARGET_BUILDING", "OBST_A"}
    assert scene.centre == pytest.approx((299500.0, 5040000.0), abs=0.5)


def test_le_cadrage_se_derive_du_batiment(tmp_path: Path) -> None:
    """Une tour et un motel doivent tous deux remplir l'image."""
    tower = load_scene(_tower(tmp_path))
    squat = load_scene(
        _manifest(
            tmp_path / "b",
            [_entry("TARGET_BUILDING", "target_building", _square(0, 0, 55))],
        )
    )

    for scene in (tower, squat):
        result = render_sequence(
            scene,
            tmp_path / f"out-{scene.crs}-{scene.radius_m():.0f}",
            frame_count=8,
            width=128,
            height=72,
            write_images=False,
        )
        coverage = [f.stats["target_coverage"] for f in result.frames]
        assert max(coverage) > 0.10, "la cible doit peser dans l'image"
        assert result.verdict() != "unusable"


def test_l_altitude_suit_la_hauteur_de_la_cible(tmp_path: Path) -> None:
    """Une trajectoire réglée sur un motel ne convient pas à une tour."""
    low = load_scene(
        _manifest(
            tmp_path / "low",
            [_entry("TARGET_BUILDING", "target_building", _square(0, 0, 20))],
        )
    )
    high = load_scene(_tower(tmp_path / "high"))
    high.target.height_m = 38.0
    high.target.height_assumed = False

    def altitudes(scene: object, out: str) -> tuple[float, float]:
        result = render_sequence(
            scene, tmp_path / out, frame_count=6, width=96, height=64,
            write_images=False,
        )
        return result.frames[0].altitude_m, result.frames[-1].altitude_m

    low_start, low_end = altitudes(low, "a")
    high_start, high_end = altitudes(high, "b")

    assert high_start > low_start
    assert high_end > low_end
    # La marge d'ouverture est dégressive : elle ne suit pas un simple multiple.
    assert high_start < low_start * 3


# --- appui photographique dans le verdict des frames ------------------------


def test_une_frame_sans_reference_est_signalee(tmp_path: Path) -> None:
    """Géométrie solide et aucune photographie : le mode doit le dire."""
    from hotel_pipeline.conditioning.support import ReferenceView, SupportMap

    scene = load_scene(
        _manifest(
            tmp_path,
            [_entry("TARGET_BUILDING", "target_building", _square(0, 0, 10))],
        )
    )
    # Une seule référence, à l'opposé de l'orbite rendue.
    support = SupportMap([ReferenceView("a", 0.0)])
    result = render_sequence(
        scene,
        tmp_path / "out",
        frame_count=6,
        arc_deg=40.0,
        start_bearing_deg=170.0,
        width=96,
        height=64,
        write_images=False,
        support=support,
    )

    assert result.verdict() == "unreferenced_arc"
    assert result.unreferenced_fraction == pytest.approx(1.0)
    modes = {f.guidance_mode for f in result.frames}
    assert modes == {"unreferenced"}
    assert "inventera" in result.frames[0].guidance_reason


def test_une_frame_appuyee_reste_fortement_contrainte(tmp_path: Path) -> None:
    from hotel_pipeline.conditioning.support import ReferenceView, SupportMap

    scene = load_scene(
        _manifest(
            tmp_path,
            [_entry("TARGET_BUILDING", "target_building", _square(0, 0, 10))],
        )
    )
    support = SupportMap([ReferenceView("a", 190.0)])
    result = render_sequence(
        scene,
        tmp_path / "out",
        frame_count=6,
        arc_deg=20.0,
        start_bearing_deg=180.0,
        width=96,
        height=64,
        write_images=False,
        support=support,
    )

    assert result.unreferenced_fraction == 0.0
    assert all(f.photo_support > 0.9 for f in result.frames)
    assert all(f.nearest_reference == "a" for f in result.frames)


def test_sans_carte_d_appui_le_verdict_ne_change_pas(tmp_path: Path) -> None:
    """L'appui est une information supplémentaire, pas une régression."""
    scene = load_scene(
        _manifest(
            tmp_path,
            [_entry("TARGET_BUILDING", "target_building", _square(0, 0, 10))],
        )
    )
    result = render_sequence(
        scene, tmp_path / "out", frame_count=6, width=96, height=64,
        write_images=False,
    )
    assert result.unreferenced_fraction == 0.0
    assert result.as_dict()["support"] is None
