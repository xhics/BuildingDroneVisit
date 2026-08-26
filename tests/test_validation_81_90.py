from __future__ import annotations
import numpy as np
import pytest

from hotel_pipeline.canonical_registration import CanonicalSim3, DegenerateRegistration, VerticalSourceTransform, fit_canonical_sim3, fit_vertical_rigid, CANONICAL_TO_THREE, adapt_direction
from hotel_pipeline.dense_registration import dense_registration_gate, fuse_surface
from hotel_pipeline.floating_origin import FloatingOrigin
from hotel_pipeline.geo.ridge_graph import RidgeEdge, RidgeGraph, RidgeNode, validate_roof_graph
from hotel_pipeline.schemas.canonical_states import MeasurementState


def test_known_colmap_point_reaches_world_coordinate():
    transform = CanonicalSim3(2.0, np.eye(3), np.array([10.,20.,30.]))
    np.testing.assert_allclose(transform.colmap_to_world([[1,2,3]])[0], [12,24,36])


def test_sim3_explicit_inverse_round_trip():
    angle=.4; r=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    transform=CanonicalSim3(1.7,r,np.array([4,-8,2.])); points=np.random.default_rng(4).normal(size=(30,3))
    np.testing.assert_allclose(transform.world_to_colmap(transform.colmap_to_world(points)),points,atol=1e-10)


def test_axis_adapter_preserves_north_and_down_semantics():
    north_down=np.array([0.,1.,-1.]); three=adapt_direction(north_down,CANONICAL_TO_THREE)
    np.testing.assert_allclose(three,[0.,-1.,-1.])


def test_one_degree_tilt_is_removed_without_distance_deformation():
    rng=np.random.default_rng(3); source=rng.normal(size=(20,3)); a=np.deg2rad(1); r=np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
    target=source@r.T+np.array([2,3,4]); fitted=fit_vertical_rigid(source,target); aligned=fitted.colmap_to_world(source)
    np.testing.assert_allclose(aligned,target,atol=1e-10)
    assert np.linalg.norm(aligned[0]-aligned[1]) == pytest.approx(np.linalg.norm(source[0]-source[1]))


def test_vertical_sources_converge_in_canonical_datum():
    ellipsoid=np.array([[1,2,70.]])
    converted=VerticalSourceTransform("gps","ELLIPSOID","CGVD2013",-30.,"geoid grid").apply(ellipsoid)
    np.testing.assert_allclose(converted,[[1,2,40.]])


def test_nearly_collinear_control_points_are_refused():
    x=np.linspace(0,10,5); source=np.c_[x,x*1e-7,x*2e-7]
    with pytest.raises(DegenerateRegistration): fit_canonical_sim3(source,source+1)


def test_dense_shifted_half_metre_is_unregistered():
    canonical=np.random.default_rng(2).normal(size=(100,3)); dense=canonical+np.array([.5,0,0])
    assert not dense_registration_gate(dense,canonical,reprojection_px=1.,normal_agreement=.95)["admitted"]


def test_noisy_dense_does_not_move_precise_lidar():
    lidar=np.array([[0.,0.,5.]]); dense=np.array([[.2,.1,5.3]])
    result=fuse_surface(lidar,dense,primary_state=MeasurementState.MEASURED,primary_sigma_m=.03,secondary_state=MeasurementState.MEASURED,secondary_sigma_m=.3)
    np.testing.assert_array_equal(result["xyz"],lidar)


def test_roof_graph_rejects_impossible_crossing():
    nodes=[RidgeNode(i,np.array(p,float)) for i,p in enumerate([(0,0,2),(2,2,2),(0,2,2),(2,0,2)])]
    edges=[RidgeEdge(0,0,0,1,3,45,"ridge"),RidgeEdge(1,1,2,3,3,135,"ridge")]
    graph=RidgeGraph(nodes,edges)
    assert not validate_roof_graph(graph)["passed"]


def test_floating_origin_preserves_geometry_at_large_coordinates():
    local=np.array([[0,0,0],[10,.25,4]],float); shift=np.array([600000.,5000000.,100.])
    a=FloatingOrigin(np.zeros(3)).to_render(local); b=FloatingOrigin(shift).to_render(local+shift)
    np.testing.assert_array_equal(a,b)
