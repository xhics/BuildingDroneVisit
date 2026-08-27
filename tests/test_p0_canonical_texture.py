import numpy as np

from hotel_pipeline.conditioning.build_canonical import build_canonical_building_mesh
from hotel_pipeline.conditioning.canonical_texture import (
    TextureObservation,
    build_surface_chart,
    observation_weight,
    texture_surface,
    trace_rendered_pixel,
)


class Camera:
    def __init__(self, position=(20.0, 3.0, 4.0), focal=100.0):
        self.position = np.asarray(position, float)
        self.f = float(focal)

    def project(self, points):
        points = np.asarray(points, float)
        screen = np.column_stack([10.0 * points[:, 1] + 10.0, 70.0 - 8.0 * points[:, 2]])
        return screen, self.position[0] - points[:, 0]

    def unproject(self, u, v, depth):
        return np.array([
            self.position[0] - depth,
            (u - 10.0) / 10.0,
            (70.0 - v) / 8.0,
        ])


class OccludedProxy:
    def hit(self, _x, _y):
        return 1.0, 999_999


def _mesh(split=False):
    footprint = (
        [[0, 0], [10, 0], [10, 3], [10, 6], [0, 6]]
        if split else [[0, 0], [10, 0], [10, 6], [0, 6]]
    )
    mesh = build_canonical_building_mesh(np.asarray(footprint, float), top_heights=8.0)
    mesh.assign_surface_ids("hotel", "main")
    return mesh


def _image(colour):
    return np.broadcast_to(np.asarray(colour, np.uint8), (80, 100, 3)).copy()


def test_uv_chart_comes_from_exact_canonical_surface_and_survives_retriangulation():
    first, second = _mesh(), _mesh(split=True)
    surface_id = "hotel/main/facade/east-01"
    a = build_surface_chart(first, surface_id, texel_size_m=0.5)
    b = build_surface_chart(second, surface_id, texel_size_m=0.5)
    assert a.surface_id == b.surface_id
    assert (a.width_px, a.height_px) == (b.width_px, b.height_px)
    assert np.allclose(a.uv_vertices.min(axis=0), b.uv_vertices.min(axis=0))
    assert np.allclose(a.uv_vertices.max(axis=0), b.uv_vertices.max(axis=0))
    assert len(b.triangle_ids) > len(a.triangle_ids)


def test_front_view_and_fine_gsd_outweigh_grazing_or_coarse_views():
    frontal = observation_weight(1.0, 0.02, 1.0, 1.0, 0.0, 1.0)
    grazing = observation_weight(np.cos(np.radians(80)), 0.02, 1.0, 1.0, 0.0, 1.0)
    coarse = observation_weight(1.0, 0.12, 1.0, 1.0, 0.0, 1.0)
    bad_pose = observation_weight(1.0, 0.02, 1.0, 1.0, 0.5, 1.0)
    assert frontal > grazing
    assert frontal > coarse
    assert bad_pose < frontal * 0.3


def test_mesh_occlusion_leaves_texels_unknown():
    mesh = _mesh()
    atlas = texture_surface(mesh, "hotel/main/facade/east-01", [
        TextureObservation(
            "tree-view", _image([20, 150, 30]), Camera(),
            valid_mask=np.ones((80, 100), bool), proxy_depth=OccludedProxy(),
            sharpness=1.0,
        )
    ], texel_size_m=1.0)
    assert atlas.coverage == 0.0
    assert np.all(atlas.state == "UNKNOWN")
    assert atlas.rejection_counts["occluded_by_mesh"] > 0


def test_robust_multiview_fusion_rejects_contradictory_red_car():
    mesh = _mesh()
    observations = [
        TextureObservation("clean-a", _image([110, 110, 110]), Camera(), sharpness=1.0),
        TextureObservation("clean-b", _image([114, 112, 110]), Camera(), sharpness=1.0),
        TextureObservation("red-car", _image([255, 0, 0]), Camera(), sharpness=1.0),
    ]
    atlas = texture_surface(
        mesh, "hotel/main/facade/east-01", observations, texel_size_m=1.0
    )
    measured = atlas.state == "MEASURED"
    assert measured.any()
    colour = np.median(atlas.rgba[measured, :3], axis=0)
    assert np.linalg.norm(colour - [112, 111, 110]) < 12
    assert np.max(atlas.view_count[measured]) == 2


def test_texel_trace_reaches_surface_uv_and_source_image():
    mesh = _mesh()
    atlas = texture_surface(mesh, "hotel/main/facade/east-01", [
        TextureObservation("img-023", _image([90, 100, 110]), Camera(), sharpness=1.0)
    ], texel_size_m=1.0)
    positions = np.argwhere(atlas.state == "MEASURED")
    assert len(positions)
    y, x = positions[0]
    provenance = atlas.texel_provenance(int(x), int(y))
    assert provenance["surface_id"] == "hotel/main/facade/east-01"
    assert provenance["best_source"] == "img-023"
    assert provenance["source_image_ids"] == ["img-023"]
    assert provenance["effective_gsd_m"] is not None
    assert provenance["incidence_deg"] is not None
    assert provenance["sharpness"] == 1.0


def test_recess_walls_remain_distinct_physical_texture_charts():
    footprint = np.asarray([
        [0, 0], [10, 0], [10, 2], [8, 2],
        [8, 4], [10, 4], [10, 6], [0, 6],
    ], float)
    mesh = build_canonical_building_mesh(footprint, top_heights=8.0)
    mesh.assign_surface_ids("hotel", "main")
    east = sorted(value for value in mesh.surface_catalog if "/facade/east-" in value)
    west = sorted(value for value in mesh.surface_catalog if "/facade/west-" in value)
    assert len(east) >= 2
    assert west  # rear wall of the recess
    assert set(build_surface_chart(mesh, east[0]).triangle_ids).isdisjoint(
        build_surface_chart(mesh, west[-1]).triangle_ids
    )


def test_synthetic_projection_reconstructs_known_image_samples_with_high_psnr():
    mesh = _mesh()
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    yy, xx = np.indices(image.shape[:2])
    image[..., 0] = xx
    image[..., 1] = yy
    image[..., 2] = (xx + yy) // 2
    atlas = texture_surface(mesh, "hotel/main/facade/east-01", [
        TextureObservation("gradient", image, Camera(), sharpness=1.0)
    ], texel_size_m=1.0)
    errors = []
    for y, x in np.argwhere(atlas.state == "MEASURED"):
        uv = np.array([(x + 0.5), (y + 0.5)])
        world = atlas.chart.origin_world + uv[0] * atlas.chart.basis_u + uv[1] * atlas.chart.basis_v
        screen, _ = Camera().project(world.reshape(1, 3))
        px, py = np.round(screen[0]).astype(int)
        errors.append(atlas.rgba[y, x, :3].astype(float) - image[py, px].astype(float))
    mse = float(np.mean(np.square(errors)))
    psnr = float("inf") if mse == 0 else 10 * np.log10(255**2 / mse)
    assert psnr > 45.0


def test_rendered_pixel_lookup_traces_to_atlas_provenance():
    mesh = _mesh()
    surface_id = "hotel/main/facade/east-01"
    camera = Camera()
    atlas = texture_surface(mesh, surface_id, [
        TextureObservation("img-trace", _image([80, 90, 100]), camera, sharpness=1.0)
    ], texel_size_m=1.0)
    y_tex, x_tex = np.argwhere(atlas.state == "MEASURED")[0]
    uv = np.array([x_tex + 0.5, y_tex + 0.5])
    world = atlas.chart.origin_world + uv[0] * atlas.chart.basis_u + uv[1] * atlas.chart.basis_v
    screen, depth = camera.project(world.reshape(1, 3))
    px, py = np.floor(screen[0]).astype(int)
    face_index = next(i for i, value in enumerate(mesh.surface_ids) if value == surface_id)

    class Frame:
        width, height = 100, 80
        triangle_id = np.full((80, 100), -1, np.int32)
        depth_z = np.full((80, 100), np.inf)

    frame = Frame()
    frame.camera = camera
    frame.triangle_id[py, px] = face_index
    frame.depth_z[py, px] = depth[0]
    trace = trace_rendered_pixel(frame, mesh, {surface_id: atlas}, int(px), int(py))
    assert trace is not None
    assert trace.surface_id == surface_id
    assert trace.provenance["best_source"] == "img-trace"
