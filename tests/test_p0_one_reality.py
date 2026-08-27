import numpy as np
import pytest

from hotel_pipeline.camera_feasibility import CanonicalCollisionEngine
from hotel_pipeline.canonical_gltf import export_canonical_mesh_gltf
from hotel_pipeline.conditioning.build_canonical import build_canonical_building_mesh
from hotel_pipeline.conditioning.facade_texture import canonical_texture_triangles
from hotel_pipeline.conditioning.render import _prism_faces
from hotel_pipeline.conditioning.scene import Prism
from hotel_pipeline.reality_contract import RealityContractError, require_canonical_mesh
from hotel_pipeline.reality_contract import audit_consumer_receipts


def _mesh(height=8.0, clockwise=False):
    footprint = np.array([[0, 0], [8, 0], [8, 5], [0, 5]], float)
    if clockwise:
        footprint = footprint[::-1]
    mesh = build_canonical_building_mesh(footprint, top_heights=height)
    mesh.assign_surface_ids("building", "main")
    return mesh


def test_same_mesh_digest_reaches_all_consumers(tmp_path):
    mesh = _mesh()
    canonical = mesh.mesh_digest()
    prism = Prism("building", "target_building", np.zeros((0, 2)), 8, False, "test", True)
    prism.canonical_mesh = mesh
    _prism_faces(prism)
    renderer = require_canonical_mesh(mesh, "renderer")
    _triangles, _ids, texture = canonical_texture_triangles(mesh)
    collision = CanonicalCollisionEngine(mesh)
    gltf = export_canonical_mesh_gltf(mesh, tmp_path / "scene.gltf")
    assert renderer.input_mesh_digest == canonical
    assert texture.input_mesh_digest == canonical
    assert collision.input_mesh_digest == canonical
    assert gltf["input_mesh_digest"] == canonical
    audit = audit_consumer_receipts(
        canonical, [renderer, texture, collision.receipt]
    )
    assert audit["passed"] is True
    assert audit["legacy_geometry_paths_used"] == 0


def test_consumers_refuse_footprint_height_fallback(tmp_path):
    prism = Prism("building", "target_building", np.array([[0, 0], [1, 0], [0, 1]]), 8, False, "test", True)
    with pytest.raises(RealityContractError):
        _prism_faces(prism)
    with pytest.raises(RealityContractError):
        CanonicalCollisionEngine({"footprint": prism.footprint, "height": 8})
    with pytest.raises(RealityContractError):
        export_canonical_mesh_gltf({"footprint": prism.footprint, "height": 8}, tmp_path / "bad.gltf")


def test_roof_change_invalidates_every_consumer_digest(tmp_path):
    before, after = _mesh(8.0), _mesh(8.5)
    assert before.mesh_digest() != after.mesh_digest()
    assert CanonicalCollisionEngine(after).input_mesh_digest == after.mesh_digest()
    assert canonical_texture_triangles(after)[2].input_mesh_digest == after.mesh_digest()
    assert export_canonical_mesh_gltf(after, tmp_path / "after.gltf")["input_mesh_digest"] == after.mesh_digest()


def test_surface_ids_are_stable_and_semantic():
    first, second = _mesh(), _mesh()
    assert first.surface_ids == second.surface_ids
    assert "building/main/roof/plane-flat-01" in first.surface_ids
    assert {
        f"building/main/facade/{side}-01"
        for side in ("north", "south", "east", "west")
    } <= set(first.surface_ids)


def test_clockwise_and_counterclockwise_inputs_are_canonically_equal():
    ccw, cw = _mesh(clockwise=False), _mesh(clockwise=True)
    assert ccw.mesh_digest() == cw.mesh_digest()
    assert sorted(ccw.surface_ids) == sorted(cw.surface_ids)
