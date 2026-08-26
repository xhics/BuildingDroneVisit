"""Tests obligatoires des problèmes 31 à 40."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from hotel_pipeline.conditioning.canonical_images import (
    CanonicalImageRecord,
    CanonicalImageTable,
)
from hotel_pipeline.conditioning.facade_texture import prepare_view_masks
from hotel_pipeline.conditioning.texture_masks import (
    TextureViewMask,
    align_mask_to_image,
    load_texture_masks,
    mask_checksum,
    save_texture_masks,
)
from hotel_pipeline.geo.facade_visibility import ProxyDepth, RegisteredView
from hotel_pipeline.geo.orthofacade import (
    Orthofacade,
    TexelCandidate,
    candidate_weight,
    fuse_texel_candidates,
    plane_from_edge,
    rectify,
)
from hotel_pipeline.geo.photometric import estimate_gain_bias
from hotel_pipeline.workspace import Workspace


class _Camera:
    width, height = 640, 480

    def __init__(self, position=(0.0, -30.0, 2.5), focal=400.0):
        self.position = np.asarray(position, dtype=float)
        self.f = focal
        self.fwd = np.array([0.0, 1.0, 0.0])
        self.right = np.array([1.0, 0.0, 0.0])
        self.up = np.array([0.0, 0.0, 1.0])

    def project(self, points):
        d = np.asarray(points, dtype=float) - self.position
        z = d @ self.fwd
        if np.all(z <= 0.5):
            return None, z
        safe = np.where(z > 1e-6, z, 1e-6)
        return (
            np.c_[
                self.width / 2 + self.f * (d @ self.right) / safe,
                self.height / 2 - self.f * (d @ self.up) / safe,
            ],
            z,
        )


class _OrientedCamera(_Camera):
    def __init__(self, position, fwd, right, focal=400.0):
        super().__init__(position=position, focal=focal)
        self.fwd = np.asarray(fwd, dtype=float)
        self.right = np.asarray(right, dtype=float)
        self.up = np.cross(self.right, self.fwd)

    def project(self, points):
        d = np.asarray(points, dtype=float) - self.position
        z = d @ self.fwd
        if np.all(z <= 0.5):
            return None, z
        safe = np.where(z > 1e-6, z, 1e-6)
        return (
            np.c_[
                self.width / 2 + self.f * (d @ self.right) / safe,
                self.height / 2 - self.f * (d @ self.up) / safe,
            ],
            z,
        )


def _wall(length=10.0, height=6.0):
    return plane_from_edge(
        np.array([-length / 2, 0.0, 0.0]),
        np.array([length / 2, 0.0, 0.0]),
        height,
        "MUR",
    )


def _image(colour=(120, 130, 140)):
    return np.full((480, 640, 3), colour, dtype=np.uint8)


def _full_mask():
    return np.ones((480, 640), dtype=bool)


def _proxy_disc(width=640, height=480, occ_x=320, occ_y=240, occ_radius=20, depth_m=5.0):
    depth = np.full((height, width), np.inf)
    face_map = np.full((height, width), -1, dtype=np.int32)
    for dy in range(-occ_radius, occ_radius + 1):
        for dx in range(-occ_radius, occ_radius + 1):
            if dx * dx + dy * dy <= occ_radius * occ_radius:
                py, px = occ_y + dy, occ_x + dx
                if 0 <= py < height and 0 <= px < width:
                    depth[py, px] = depth_m
                    face_map[py, px] = 99
    return ProxyDepth(width=width, height=height, depth=depth, face_id_map=face_map)


class TestP31FailClosedSansMasqueBuilding:
    def test_vue_sans_masque_contribution_zero_texel(self):
        view = RegisteredView(asset_id="A", camera=_Camera(), image=_image())
        found = rectify(_wall(), [view])
        assert found.observed_fraction == 0.0
        assert all(texel.contributing == 0 for texel in found.support)
        assert found.provenance["views_without_building_mask"] >= 1

    def test_masque_building_absent_prepare_view_masks_refuse(self):
        occluders = np.zeros((480, 640), dtype=bool)
        occluders[100:200, 100:200] = True
        mask_info = TextureViewMask(
            asset_id="A", building=None, occluders=occluders,
            fidelity="polygon_with_occluders", classes_present=[], sign_regions=[],
        )
        assert prepare_view_masks(mask_info, (480, 640, 3)) is None

    def test_aucun_masque_du_tout_refuse(self):
        assert prepare_view_masks(None, (480, 640, 3)) is None


class TestP32PersistanceMasques:
    @pytest.fixture
    def workspace(self, tmp_path):
        return Workspace("hotel-test", root=tmp_path)

    @pytest.fixture
    def masks(self):
        rng_pattern = (np.arange(64 * 96).reshape(64, 96) % 7) < 4
        building = np.zeros((64, 96), dtype=bool)
        building[8:56, 12:84] = rng_pattern[8:56, 12:84]
        occluders = np.zeros((64, 96), dtype=bool)
        occluders[20:40, 30:50] = True
        return {
            "asset-1": TextureViewMask(
                asset_id="asset-1",
                building=building,
                occluders=occluders,
                fidelity="polygon_with_occluders",
                classes_present=["building", "tree_evergreen"],
                sign_regions=[],
                width=96,
                height=64,
                image_checksum="sha256:image",
            )
        }

    def test_sauvegarder_recharger_bitwise_identique(self, workspace, masks):
        index_path = save_texture_masks(workspace, masks)
        assert index_path.is_file()
        payload = json.loads(index_path.read_text("utf-8"))
        entry = payload["views"][0]
        assert entry["width"] == 96 and entry["height"] == 64
        assert entry["building_checksum"] == mask_checksum(masks["asset-1"].building)
        assert entry["image_checksum"] == "sha256:image"

        reloaded = load_texture_masks(workspace)
        original = masks["asset-1"]
        restored = reloaded["asset-1"]
        assert restored.building.shape == original.building.shape
        assert np.array_equal(restored.building, original.building)
        assert np.array_equal(restored.occluders, original.occluders)
        assert mask_checksum(restored.building) == mask_checksum(original.building)

    def test_raster_corrompu_est_rejete_pas_charge(self, workspace, masks):
        save_texture_masks(workspace, masks)
        store = workspace.path("11_conditioning", "texture_view_masks_store")
        raster = store / "asset-1" / "building.png"
        corrupted = np.array(Image.open(raster).convert("L"), copy=True)
        corrupted[:10, :] = 255 - corrupted[:10, :]
        Image.fromarray(corrupted, mode="L").save(raster)

        reloaded = load_texture_masks(workspace)
        assert reloaded.get("asset-1") is None or reloaded["asset-1"].building is None


class TestP33AlignementGeometrie:
    def test_masque_1080p_jamais_applique_a_une_image_720p(self):
        mask = np.ones((1080, 1920), dtype=bool)
        assert align_mask_to_image(mask, (720, 1280, 3)) is None

    def test_transformation_explicite_permute_l_alignement(self):
        mask = np.ones((1080, 1920), dtype=bool)
        aligned = align_mask_to_image(
            mask,
            (720, 1280, 3),
            transform={"type": "resize", "source_width": 1920, "source_height": 1080},
        )
        assert aligned is not None
        assert aligned.shape == (720, 1280)

    def test_provenance_absente_rejete_la_vue(self):
        mask_info = TextureViewMask(
            asset_id="A",
            building=np.ones((1080, 1920), dtype=bool),
            occluders=None,
            fidelity="raster",
            classes_present=[],
            sign_regions=[],
        )
        assert prepare_view_masks(mask_info, (1280, 720, 3)) is None

    def test_tailles_egales_copie_directe(self):
        mask = np.ones((480, 640), dtype=bool)
        aligned = align_mask_to_image(mask, (480, 640, 3))
        assert aligned is not None and aligned.shape == (480, 640)


class _FakeColmapImage:
    def __init__(self, name: str):
        self.name = name


class _FakeReconstruction:
    def __init__(self, images: dict[int, _FakeColmapImage]):
        self.images = images


class TestP34IdentiteCanonique:
    @pytest.fixture
    def corpus(self, tmp_path):
        asset_a = tmp_path / "assets" / "photo-a.jpg"
        asset_b = tmp_path / "assets" / "photo-b.jpg"
        asset_a.parent.mkdir(parents=True)
        asset_a.write_bytes(b"contenu-image-A")
        asset_b.write_bytes(b"contenu-image-B")

        model_dir = tmp_path / "model"
        (model_dir / "images").mkdir(parents=True)
        (model_dir / "images" / "cam-001.jpg").write_bytes(asset_a.read_bytes())
        (model_dir / "images" / "inconnue.jpg").write_bytes(b"inconnu")

        assets = [
            {"id": "asset-a", "local_path": str(asset_a)},
            {"id": "asset-b", "local_path": str(asset_b)},
        ]
        return assets, model_dir

    def test_chaque_image_colmap_resout_vers_exact_un_asset_ou_aucun(self, tmp_path, corpus):
        assets, model_dir = corpus
        table = CanonicalImageTable.build(
            Workspace("hotel-test", root=tmp_path),
            assets,
            _FakeReconstruction({1: _FakeColmapImage("cam-001.jpg"), 2: _FakeColmapImage("inconnue.jpg")}),
            model_dir,
        )
        resolved = table.resolve_colmap(1)
        assert resolved is not None
        assert resolved.asset_id == "asset-a"
        assert resolved.checksum
        assert resolved.mask_id == "asset-a"
        assert table.resolve_colmap(2) is None
        assert table.validate() == []

    def test_deux_assets_meme_contenu_ne_creent_jamais_deux_resolutions(self, tmp_path):
        asset_a = tmp_path / "a.jpg"
        asset_dup = tmp_path / "dup.jpg"
        asset_a.write_bytes(b"same-bytes")
        asset_dup.write_bytes(b"same-bytes")
        table = CanonicalImageTable.build(
            Workspace("hotel-test", root=tmp_path),
            [
                {"id": "asset-a", "local_path": str(asset_a)},
                {"id": "asset-dup", "local_path": str(asset_dup)},
            ],
        )
        assert len(table.records) == 1
        assert "asset-dup" in table.ambiguous_assets

    def test_roundtrip_json(self, tmp_path):
        workspace = Workspace("hotel-test", root=tmp_path)
        table = CanonicalImageTable(records=[
            CanonicalImageRecord("ci-abc", "asset-a", "imgs/a.jpg", "sum-a", colmap_image_id=3, colmap_name="a.jpg", mask_id="asset-a"),
        ])
        table.save(workspace)
        loaded = CanonicalImageTable.from_workspace(workspace)
        assert loaded.resolve_colmap(3).asset_id == "asset-a"


class TestP35PasDeFallbackOccludeur:
    def test_occluder_seul_ne_devient_pas_facade(self):
        occluders = np.zeros((480, 640), dtype=bool)
        occluders[200:280, 280:360] = True
        mask_info = TextureViewMask(
            asset_id="A", building=None, occluders=occluders,
            fidelity="occluders_only", classes_present=[], sign_regions=[],
        )
        assert prepare_view_masks(mask_info, (480, 640, 3)) is None

    def test_ciel_et_route_hors_building_mask_ne_deviennent_jamais_facade(self):
        building = np.zeros((480, 640), dtype=bool)
        building[100:400, 150:500] = True
        mask_info = TextureViewMask(
            asset_id="A", building=building, occluders=None,
            fidelity="raster", classes_present=[], sign_regions=[],
        )
        building_aligned, occluders_aligned = prepare_view_masks(mask_info, (480, 640, 3))
        assert occluders_aligned is None
        assert not building_aligned[0, 0]
        assert not building_aligned[479, 639]
        assert building_aligned[250, 300]


class TestP36RejetParCandidat:
    def test_vue_occultee_plus_vue_propre_conservent_le_texel(self):
        proxy_occ = _proxy_disc()
        views = [
            ("A", _image((100, 110, 120)), _Camera(position=(-3.0, -30.0, 2.5)), _full_mask(), proxy_occ, None),
            ("B", _image((102, 108, 118)), _Camera(position=(3.0, -30.0, 2.5)), _full_mask(), None, None),
        ]
        found = rectify(_wall(), views)
        assert found.observed_fraction > 0.5
        occ_slot = int(240 / 480 * found.height_px) * found.width_px + int(320 / 640 * found.width_px)
        texel = found.support[occ_slot]
        assert texel.contributing >= 1
        assert any(c.asset_id == "B" for c in texel.candidates)

    def test_texel_entierement_filtre_porte_le_motif_par_candidat(self):
        proxy_occ = _proxy_disc()
        found = rectify(_wall(), [("A", _image(), _Camera(), _full_mask(), proxy_occ, None)])
        occ_slot = int(240 / 480 * found.height_px) * found.width_px + int(320 / 640 * found.width_px)
        texel = found.support[occ_slot]
        assert texel.contributing == 0
        assert texel.rejection_reason == "REJECTED_OCCLUDED"
        assert ("A", "REJECTED_OCCLUDED") in texel.rejected_candidates

    def test_une_seule_vue_valide_reste_observee(self):
        verdict = fuse_texel_candidates([np.array([100.0, 110.0, 120.0])])
        assert verdict.accepted
        assert verdict.status == "OBSERVED_SINGLE"
        assert verdict.colour == (100.0, 110.0, 120.0)


class TestP37FusionRobusteAvantDispersion:
    BRIQUE = (112, 73, 62)

    def test_brique_brique_feuille_donnera_brique_pas_rejet(self):
        colours = [
            np.array([112.0, 73.0, 62.0]),
            np.array([114.0, 75.0, 60.0]),
            np.array([30.0, 180.0, 70.0]),
        ]
        verdict = fuse_texel_candidates(colours)
        assert verdict.accepted
        assert verdict.inlier_count == 2
        consensus = np.asarray(verdict.colour)
        assert np.linalg.norm(consensus - np.array(self.BRIQUE)) < 15.0

    def test_niveau_rectify_le_texel_est_conserve(self):
        views = [
            ("A", _image((112, 73, 62)), _Camera(position=(-3.0, -30.0, 2.5)), _full_mask()),
            ("B", _image((113, 74, 61)), _Camera(position=(0.0, -30.0, 2.5)), _full_mask()),
            ("C", _image((30, 180, 70)), _Camera(position=(3.0, -30.0, 2.5)), _full_mask()),
        ]
        found = rectify(_wall(), views)
        assert found.by_status().get("REJECTED_DISAGREEMENT", 0) == 0
        assert found.observed_fraction > 0.3
        slot = found.width_px // 2 + (found.height_px // 2) * found.width_px
        consensus = np.asarray(found.support[slot].consensus_colour)
        assert np.linalg.norm(consensus - np.array(self.BRIQUE)) < 20.0

    def test_deux_couleurs_opposees_restent_rejetees(self):
        verdict = fuse_texel_candidates([
            np.array([0.0, 0.0, 0.0]),
            np.array([255.0, 255.0, 255.0]),
        ])
        assert not verdict.accepted
        assert verdict.reason == "inlier_spread"


class TestP38ConsensusEcritDansLatlas:
    def test_pixel_final_exactement_la_sortie_de_la_fusion(self):
        views = [
            ("A", _image((100, 110, 120)), _Camera(position=(-3.0, -30.0, 2.5)), _full_mask()),
            ("B", _image((104, 106, 118)), _Camera(position=(3.0, -30.0, 2.5)), _full_mask()),
        ]
        found = rectify(_wall(), views)
        checked = 0
        for slot, texel in enumerate(found.support):
            if not texel.candidates:
                continue
            row, col = divmod(slot, found.width_px)
            verdict = fuse_texel_candidates(
                [c.normalised_colour() for c in texel.candidates],
                [c.weight for c in texel.candidates],
            )
            expected = np.clip(verdict.colour, 0.0, 255.0).astype(np.uint8)
            assert np.array_equal(found.image[row, col], expected)
            checked += 1
        assert checked > 0

    def test_ponderation_angle_gsd_confiance(self):
        frontal = candidate_weight(incidence_deg=0.0, gsd_m=0.05)
        oblique = candidate_weight(incidence_deg=60.0, gsd_m=0.10)
        assert frontal > oblique
        assert candidate_weight(incidence_deg=0.0, gsd_m=0.05, pose_confidence=0.94) >                candidate_weight(incidence_deg=0.0, gsd_m=0.05, pose_confidence=0.50)


class TestP39NormalisationPhotometrique:
    def test_meme_mur_soliel_ombre_transition_homogene(self):
        base = _image((120, 130, 140))
        shaded = np.clip(base.astype(np.float64) * 1.5, 0, 255).astype(np.uint8)
        views = [
            ("A", base, _Camera(position=(-3.0, -30.0, 2.5)), _full_mask()),
            ("B", shaded, _Camera(position=(3.0, -30.0, 2.5)), _full_mask()),
        ]
        found = rectify(_wall(), views)
        normalization = found.provenance["photometric_normalization"]
        assert normalization["normalized_views"]

        slot = found.width_px // 2 + (found.height_px // 2) * found.width_px
        consensus = np.asarray(found.support[slot].consensus_colour)
        assert np.all(np.abs(consensus - np.array([120, 130, 140])) < 8.0)
        assert found.by_status().get("REJECTED_DISAGREEMENT", 0) == 0

    def test_gain_bias_estimateur_robuste(self):
        rng_values = np.linspace(80, 200, 64)
        reference = np.stack([rng_values] * 3, axis=1)
        other = np.stack([rng_values * 1.5] * 3, axis=1)
        model = estimate_gain_bias(reference, other)
        assert model is not None
        assert all(abs(g - 1 / 1.5) < 0.05 for g in model.gain)
        corrected = model.apply(other[10])
        assert np.allclose(corrected, reference[10], atol=2.0)

    def test_contenu_different_n_est_pas_normalise(self):
        reference = np.full((32, 3), 10.0)
        other = np.full((32, 3), 250.0)
        assert estimate_gain_bias(reference, other) is None


class TestP40GsdJacobian:
    WALL_CENTRE = np.array([0.0, 0.0, 3.0])
    DISTANCE_M = 25.0
    FOCAL_PX = 400.0

    def _frontal_camera(self):
        return _OrientedCamera(
            position=self.WALL_CENTRE - np.array([0.0, self.DISTANCE_M, 0.0]),
            fwd=(0.0, 1.0, 0.0),
            right=(1.0, 0.0, 0.0),
            focal=self.FOCAL_PX,
        )

    def _oblique_camera_60deg(self):
        theta = np.deg2rad(60.0)
        fwd = np.array([np.sin(theta), np.cos(theta), 0.0])
        right = np.array([np.cos(theta), -np.sin(theta), 0.0])
        return _OrientedCamera(
            position=self.WALL_CENTRE - fwd * self.DISTANCE_M,
            fwd=fwd,
            right=right,
            focal=self.FOCAL_PX,
        )

    def test_vue_a_60_degres_penalisee_face_a_une_vue_frontale(self):
        from hotel_pipeline.geo.facade_visibility import effective_gsd_m
        along = np.array([1.0, 0.0, 0.0])
        gsd_frontal = effective_gsd_m(self._frontal_camera(), self.WALL_CENTRE, along)
        gsd_oblique = effective_gsd_m(self._oblique_camera_60deg(), self.WALL_CENTRE, along)
        assert gsd_frontal is not None and gsd_oblique is not None
        ratio = gsd_oblique / gsd_frontal
        assert 1.7 < ratio < 2.4, f"ratio mesure {ratio:.2f}"

    def test_gsd_frontale_correspond_a_distance_sur_focale(self):
        from hotel_pipeline.geo.facade_visibility import effective_gsd_m
        gsd = effective_gsd_m(self._frontal_camera(), self.WALL_CENTRE, np.array([1.0, 0.0, 0.0]))
        attendu = self.DISTANCE_M / self.FOCAL_PX
        assert gsd == pytest.approx(attendu, rel=0.1)

    def test_point_non_projetable_retourne_none(self):
        from hotel_pipeline.geo.facade_visibility import effective_gsd_m
        camera = self._frontal_camera()
        point = camera.position - camera.fwd * 5.0
        assert effective_gsd_m(camera, point) is None
