import numpy as np

from hotel_pipeline.conditioning.facade_texture import canonical_camera_from_registration
from hotel_pipeline.conditioning.semantic_registered_support import transform_points


class _Rotation:
    def __init__(self, matrix):
        self._matrix = matrix

    def matrix(self):
        return self._matrix


class _Pose:
    def __init__(self, rotation, translation):
        self.rotation = _Rotation(rotation)
        self.translation = translation


class _Image:
    camera_id = 7

    def __init__(self, rotation, translation):
        self.cam_from_world = _Pose(rotation, translation)

    def projection_center(self):
        pose = self.cam_from_world
        return -pose.rotation.matrix().T @ pose.translation


class _Camera:
    camera_id = 7
    model_name = "SIMPLE_RADIAL"
    width = 1280
    height = 720
    params = np.array([910.0, 640.0, 360.0, -0.015])

    def img_from_cam(self, points):
        points = np.asarray(points)
        radius2 = np.sum(points * points, axis=1)
        distorted = points * (1.0 + self.params[3] * radius2[:, None])
        return distorted * self.params[0] + self.params[1:3]


def _rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_registered_camera_preserves_pixels_and_makes_depth_metric():
    rotation_colmap = _rotation_z(-0.17)
    translation_colmap = np.array([0.6, -0.3, 8.0])
    image = _Image(rotation_colmap, translation_colmap)
    source_camera = _Camera()
    transform = {
        "sim3_rotation": _rotation_z(0.31),
        "sim3_translation": np.array([2.0, -1.5, 0.4]),
        "sim3_scale": 2.75,
        "projected_origin_xy": (621000.0, 5041000.0),
        "registration_translation": np.array([3.0, 5.0, -0.2]),
        "scene_origin_xyz": np.array([621010.0, 5040994.0, 101.0]),
    }
    points_colmap = np.array([
        [-1.2, 0.4, 2.0],
        [0.5, -0.8, 4.0],
        [1.1, 0.9, 7.0],
    ])
    points_world = transform_points(points_colmap, **transform)

    old_camera_points = points_colmap @ rotation_colmap.T + translation_colmap
    old_pixels = source_camera.img_from_cam(
        old_camera_points[:, :2] / old_camera_points[:, 2, None]
    )
    camera = canonical_camera_from_registration(image, source_camera, transform)
    pixels, metric_depth = camera.project(points_world)

    np.testing.assert_allclose(pixels, old_pixels, atol=1e-9)
    np.testing.assert_allclose(
        metric_depth, transform["sim3_scale"] * old_camera_points[:, 2], atol=1e-9
    )
    np.testing.assert_allclose(
        camera.position(),
        transform_points(image.projection_center()[None, :], **transform)[0],
        atol=1e-9,
    )
