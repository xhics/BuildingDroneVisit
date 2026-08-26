from __future__ import annotations

import numpy as np

from hotel_pipeline.architectural_geometry import classify_sign, railing_mesh, tube_along_polyline
from hotel_pipeline.canonical_gltf import triangle_provenance
from hotel_pipeline.geo.crs_control import validate_control_points
from hotel_pipeline.integrity_digests import mask_raster_digest, reconstruction_digest
from hotel_pipeline.schemas.canonical_states import MeasurementState, TexelStatus
from hotel_pipeline.schemas.spatial_reference import HeightType, SpatialReferenceContext, VerticalReference


def test_gutter_centres_follow_sloped_roof_edge():
    edge = [(0,0,4), (2,0,5), (5,0,4.5)]
    mesh = tube_along_polyline(edge, .08)
    assert mesh["centreline"] == [list(p) for p in edge]


def test_railing_is_open_geometry_not_wall():
    mesh = railing_mesh((0,0,0), (3,0,0), 1.1, spacing_m=.3)
    assert 0 < mesh["coverage"] < .2
    assert len(mesh["faces"]) > 6


def test_sign_depth_selects_texture_or_occluding_geometry():
    assert classify_sign(.01) == "surface_sign"
    assert classify_sign(.30) == "projecting_sign"


def test_measured_triangle_without_local_sources_is_downgraded():
    payload = {"volumes": [{"state": "MEASURED", "solid": {"vertices": [[0,0,0],[1,0,0],[0,1,0]], "faces": [[0,1,2]]}}]}
    assert triangle_provenance(payload)[0]["state"] == MeasurementState.UNKNOWN


def test_vertical_reference_changes_context_digest():
    base = dict(hotel_id="h", reference_lat=45, reference_lon=-73)
    a = SpatialReferenceContext(**base, vertical=VerticalReference(crs="CGVD2013", height_type=HeightType.ORTHOMETRIC))
    b = SpatialReferenceContext(**base, vertical=VerticalReference(crs="NAVD88", height_type=HeightType.ORTHOMETRIC))
    assert a.context_digest() != b.context_digest()


def test_wrong_but_invertible_crs_fails_independent_controls():
    expected = np.array([[0,0,10], [100,0,11]], float)
    wrong = expected + np.array([500,200,30])
    result = validate_control_points(wrong, expected, max_horizontal_error_m=1, max_vertical_error_m=.2, max_azimuth_error_deg=1)
    assert not result["passed"]


def test_reconstruction_digest_changes_for_one_point(tmp_path):
    for name in ("cameras", "images", "points3D"):
        (tmp_path/name).write_text(name)
    before = reconstruction_digest(tmp_path, run_parameters={"matcher":"x"}, critical_versions={"colmap":"3.13"})
    (tmp_path/"points3D").write_text("changed one point")
    assert reconstruction_digest(tmp_path, run_parameters={"matcher":"x"}, critical_versions={"colmap":"3.13"}) != before


def test_mask_digest_changes_for_one_pixel():
    a = np.zeros((8,8), np.uint8); b = a.copy(); b[3,4] = 1
    kwargs = dict(asset_id="a", pixel_transform=(1,0,0,0,1,0), segmenter_version="sam2-x")
    assert mask_raster_digest(a, **kwargs) != mask_raster_digest(b, **kwargs)


def test_texel_status_is_the_single_canonical_enum():
    assert TexelStatus.OBSERVED_CONSENSUS.value == "OBSERVED_CONSENSUS"
    assert "accorde" not in {item.value for item in TexelStatus}
