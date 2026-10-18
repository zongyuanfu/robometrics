"""Self-contained CPU API contract: no Planigo, JAX, Torch or URDF dependencies."""
import gc
import sys
import numpy as np
import pytest
from robometrics.kinematics import KinematicModel
from robometrics.trajectory_evaluation import CpuTrajectoryEvaluator, build_info
from robometrics.esdf import EsdfContext


def model(**changes):
    origins=np.repeat(np.eye(4)[None],3,axis=0)
    origins[2,0,3]=1
    fields=dict(base=np.eye(4),origins=origins,parents=[-1,0,1],q_indices=[-1,0,-1],
        axes=[[0,0,1]]*3,ee_link=2,ee_offset=np.eye(4),sphere_centers=[[0,0,0]],
        radii=[.1],sphere_links=[2],pairs=np.empty((0,2),np.int32),padding=[0,0,0],
        lower=[-4],upper=[4],velocity=[1],source_fingerprint='synthetic')
    fields.update(changes)
    return KinematicModel(**fields)


def esdf(value=None):
    grid=np.arange(8,dtype=np.float32).reshape(2,2,2) if value is None else np.full((2,2,2),value,np.float32)
    return EsdfContext(grid, np.full(3,.5),1.,(2,2,2),np.zeros(3),np.full(3,2.))


def test_fixed_revolute_chain_and_lifetime():
    m=model(); evaluator=CpuTrajectoryEvaluator(m)
    q=np.array([[0],[np.pi/2]],np.float32)
    poses=evaluator.end_effector(q)
    np.testing.assert_allclose(poses[:,:3],[[1,0,0],[0,1,0]],atol=1e-6)
    del q,m,evaluator; gc.collect()
    np.testing.assert_allclose(poses[0,:3],[1,0,0],atol=1e-6)
    assert build_info()['schema_version']==1 and build_info()['build_type']=='Release'


def test_interpolation_pairs_bounds_and_map_isolation():
    origin=np.repeat(np.eye(4)[None],3,axis=0); origin[2,:3,3]=1.
    evaluator=CpuTrajectoryEvaluator(model(origins=origin))
    a=evaluator.geometry_profile(np.zeros((1,1,1)),esdf())
    assert a['min_clearance'][0,0]==pytest.approx(3.4,abs=1e-6)
    assert np.isposinf(a['min_self_clearance'][0,0])
    b=evaluator.geometry_profile(np.zeros((1,1,1)),esdf(-1))
    assert b['min_clearance'][0,0]==pytest.approx(-1.1,abs=1e-6)
    for pos,valid in ((0,True),(2,False),(1.999,True),(-.001,False)):
        origin[2,:3,3]=pos
        e=CpuTrajectoryEvaluator(model(origins=origin))
        p=e.geometry_profile(np.zeros((1,1,1)),esdf())
        assert bool(p['valid_sphere_count'][0,0])==valid
        if not valid:
            assert np.isposinf(p['min_clearance'][0,0]) and p['oob_count'][0,0]==1
    m=model(sphere_centers=[[0,0,0],[.5,0,0]],radii=[.1,.2],sphere_links=[2,2],pairs=[[0,1]],padding=[0,0,.05])
    p=CpuTrajectoryEvaluator(m).geometry_profile(np.zeros((1,1,1)),esdf())
    assert p['min_self_clearance'][0,0]==pytest.approx(.1,abs=1e-6)


def test_empty_noncontiguous_and_invalid_input():
    e=CpuTrajectoryEvaluator(model())
    assert e.evaluate_states(np.empty((0,1)),[0,0,0,0,0,0,1],esdf())['ee'].shape==(0,7)
    assert e.geometry_profile(np.empty((0,3,1)),esdf())['min_clearance'].shape==(0,3)
    q=np.zeros((3,2),np.float64)[:,::2]
    assert e.end_effector(q).shape==(3,7)
    for q in ([np.nan],np.zeros((1,2)),np.full((1,1),np.inf)):
        with pytest.raises(ValueError): e.end_effector(q)
    with pytest.raises(ValueError): e.geometry_profile(np.empty((1,0,1)),esdf())
    with pytest.raises(ValueError): e.evaluate_states([[0]],[0,0,0,0,0,0,np.nan],esdf())


@pytest.mark.parametrize('changes',[
    dict(parents=[-1,2,1]), dict(q_indices=[-1,1,-1]), dict(axes=[[0,0,1],[0,0,0],[0,0,1]]),
    dict(radii=[0]),dict(sphere_links=[9]),dict(pairs=[[0,1]]),dict(lower=[4]),
    dict(velocity=[-1]),dict(q_indices=[-1,.5,-1]),dict(padding=[0,-1,0]),
])
def test_invalid_models(changes):
    with pytest.raises(ValueError): model(**changes)


def test_model_hash_tracks_payload_and_owns_arrays():
    radii=np.array([.1],np.float32)
    a=model(radii=radii); radii[:]=.2
    b=model(radii=radii)
    assert a.payload_sha256!=b.payload_sha256
    assert model().payload_sha256==a.payload_sha256
    assert model(source_fingerprint='other').payload_sha256==a.payload_sha256
    with pytest.raises(ValueError): a.radii.setflags(write=True)


def test_import_isolation():
    import subprocess
    code = """
import sys
from robometrics.trajectory_evaluation import CpuTrajectoryEvaluator
assert not any(k in sys.modules for k in ('planigo','jax','torch','urchin','geometrout'))
"""
    subprocess.run([sys.executable, '-c', code], check=True)


def test_noninteger_extent_and_half_turn():
    origins=np.repeat(np.eye(4)[None],3,axis=0)
    origins[2,:3,3]=[1.,1.,1.]
    e=CpuTrajectoryEvaluator(model(origins=origins))
    pose=e.end_effector([[np.pi]])[0]
    assert abs(pose[5])==pytest.approx(1.,abs=1e-6)
    ctx=EsdfContext(np.full((2,2,2),2.,'f'),np.full(3,.5,'f'),1.,(2,2,2),np.zeros(3),np.full(3,1.7,'f'))
    assert e.geometry_profile(np.zeros((1,1,1)),ctx)['oob_count'][0,0]==0
    origins[2,0,3]=1.8
    e=CpuTrajectoryEvaluator(model(origins=origins))
    assert e.geometry_profile(np.zeros((1,1,1)),ctx)['oob_count'][0,0]==1


def test_trajectory_supplied_derivatives_and_empty_errors():
    e=CpuTrajectoryEvaluator(model())
    q=np.zeros((2,1),'f'); goal=e.end_effector(q)[-1]
    m=e.evaluate_trajectory(q,np.full_like(q,2),np.full_like(q,-17),np.full_like(q,-501),goal,esdf(),
        limits=dict(max_acceleration=15,max_jerk=500))
    assert m['max_velocity_violation']==1 and m['max_acceleration_violation']==2 and m['max_abs_jerk']==501
    assert np.isposinf(m['min_self_clearance']) and m['position_error']==0
    for values in ((np.empty((0,1)),)*4,(q,q,q,np.full_like(q,np.nan)),(q,q[:1],q,q)):
        with pytest.raises(ValueError):
            e.evaluate_trajectory(*values,goal,esdf(),limits=dict(max_acceleration=15,max_jerk=500))


def test_general_axis_base_and_ee_offset():
    origin=np.repeat(np.eye(4)[None],3,axis=0);origin[2,1,3]=1
    base=np.eye(4);base[0,3]=.5
    offset=np.eye(4);offset[1,3]=.5
    e=CpuTrajectoryEvaluator(model(origins=origin,axes=[[0,0,1],[1,0,0],[0,0,1]],base=base,ee_offset=offset))
    np.testing.assert_allclose(e.end_effector([[np.pi/2]])[0,:3],[.5,0,1.5],atol=1e-6)



@pytest.mark.parametrize('missing_goal,missing_map', [(True, False), (False, True), (True, True)])
def test_state_evaluation_requires_goal_and_map(missing_goal, missing_map):
    evaluator = CpuTrajectoryEvaluator(model())
    goal = None if missing_goal else [0, 0, 0, 0, 0, 0, 1]
    context = None if missing_map else esdf()
    with pytest.raises(ValueError, match='required'):
        evaluator.evaluate_states(np.zeros((1, 1)), goal, context)


def test_geometry_requires_map_but_fk_does_not():
    evaluator = CpuTrajectoryEvaluator(model())
    with pytest.raises(ValueError, match='esdf is required'):
        evaluator.geometry_profile(np.zeros((1, 2, 1)), None)
    np.testing.assert_allclose(evaluator.end_effector([[0]])[0, :3], [1, 0, 0])


def test_trajectory_requires_goal_and_map():
    evaluator = CpuTrajectoryEvaluator(model())
    q = np.zeros((2, 1))
    limits = dict(max_acceleration=15, max_jerk=500)
    with pytest.raises(ValueError, match='goal_pose is required'):
        evaluator.evaluate_trajectory(q, q, q, q, None, esdf(), limits=limits)
    with pytest.raises(ValueError, match='esdf is required'):
        evaluator.evaluate_trajectory(q, q, q, q, [0, 0, 0, 0, 0, 0, 1], None, limits=limits)
