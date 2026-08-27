import numpy as np
import pytest

from hotel_pipeline.camera_feasibility import CanonicalCollisionEngine
from hotel_pipeline.conditioning.build_canonical import build_canonical_building_mesh
from hotel_pipeline.conditioning.canonical_mesh import CanonicalSceneMesh


def _mesh(points):
    mesh = build_canonical_building_mesh(np.asarray(points, float), top_heights=8.0)
    mesh.assign_surface_ids("hotel-main", "main")
    return mesh


RECTANGLE = [[0, 0], [10, 0], [10, 6], [0, 6]]


def _surface_ids(mesh):
    return set(mesh.surface_ids)


def test_surface_ids_survive_cw_ccw_and_ring_rotation():
    reference = _surface_ids(_mesh(RECTANGLE))
    assert _surface_ids(_mesh(RECTANGLE[::-1])) == reference
    assert _surface_ids(_mesh(RECTANGLE[2:] + RECTANGLE[:2])) == reference


def test_collinear_vertex_does_not_split_a_physical_facade():
    split = [[0, 0], [5, 0], [10, 0], [10, 6], [0, 6]]
    original, densified = _mesh(RECTANGLE), _mesh(split)
    south = "hotel-main/main/facade/south-01"
    assert south in original.surface_catalog
    assert south in densified.surface_catalog
    assert len(densified.triangles_for_surface(south)) > len(original.triangles_for_surface(south))


def test_surface_matching_preserves_unchanged_facades_after_new_wing():
    old = _mesh(RECTANGLE)
    new = build_canonical_building_mesh(
        np.asarray([[0, 0], [10, 0], [10, 3], [14, 3], [14, 6], [0, 6]], float),
        top_heights=8.0,
    )
    previous = {key: value.as_dict() for key, value in old.surface_catalog.items()}
    new.assign_surface_ids("hotel-main", "main", previous_surfaces=previous)
    preserved = _surface_ids(old) & _surface_ids(new)
    assert "hotel-main/main/facade/south-01" in preserved
    assert "hotel-main/main/facade/west-01" in preserved
    assert _surface_ids(new) - _surface_ids(old)


def test_every_triangle_has_a_kind_consistent_surface_and_evidence():
    mesh = _mesh(RECTANGLE)
    mesh.provenance = [{"source_ids": ["lidar", "photo-01"]} for _ in mesh.faces]
    mesh.assign_surface_ids("hotel-main", "main")
    assert mesh.surface_audit()["passed"] is True
    assert all(record.surface_id for record in mesh.triangle_records())
    east = mesh.surface("hotel-main/main/facade/east-01")
    assert east.kind == "facade"
    assert east.source_ids == ("lidar", "photo-01")


def test_raycast_returns_triangle_and_physical_surface():
    mesh = _mesh(RECTANGLE)
    hit = mesh.raycast_hit(np.array([15.0, 3.0, 4.0]), np.array([-1.0, 0.0, 0.0]))
    assert hit is not None
    assert hit.surface_id == "hotel-main/main/facade/east-01"
    assert hit.triangle_id in mesh.surface(hit.surface_id).triangle_ids
    collision_hit = CanonicalCollisionEngine(mesh).raycast(
        np.array([15.0, 3.0, 4.0]), np.array([-1.0, 0.0, 0.0])
    )
    assert collision_hit == hit


def test_surface_ids_and_catalog_survive_serialization_exactly():
    original = _mesh(RECTANGLE)
    loaded = CanonicalSceneMesh.from_dict(original.as_dict())
    assert loaded.surface_ids == original.surface_ids
    assert loaded.surface_catalog == original.surface_catalog
    assert loaded.mesh_digest() == original.mesh_digest()


def test_surface_type_mismatch_is_rejected():
    mesh = _mesh(RECTANGLE)
    wall = mesh.face_kind.index("wall")
    mesh.surface_ids[wall] = "hotel-main/main/roof/plane-flat-01"
    with pytest.raises(ValueError, match="surface kind mismatch"):
        mesh.validate_triangle_metadata()
