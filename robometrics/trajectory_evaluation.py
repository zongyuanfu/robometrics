"""Single-thread C++ CPU geometry evaluation; no planner or GPU imports."""
import numpy as np
from .kinematics import KinematicModel
from .esdf import EsdfContext
from . import _evaluation_cpp

_PROFILE_FIELDS = ('min_clearance', 'min_self_clearance', 'oob_count',
                   'valid_sphere_count', 'max_joint_violation', 'normalized_joint_violation')


def build_info():
    return {name:getattr(_evaluation_cpp, name) for name in
            ('schema_version', 'compiler', 'build_type')}


def snapshot(esdf):
    """Reuse a validated context; mappings are deliberately copied and validated."""
    if esdf is None:
        raise ValueError('esdf is required for geometry evaluation')
    return esdf if isinstance(esdf, EsdfContext) else EsdfContext.from_mapping(esdf)


class CpuTrajectoryEvaluator:
    def __init__(self, model: KinematicModel):
        if not isinstance(model, KinematicModel):
            raise TypeError('model must be a KinematicModel')
        self.model = model
        self._native = _evaluation_cpp.Evaluator(model.as_mapping())

    def _states(self, q, goal=None, esdf=None):
        values = np.ascontiguousarray(q, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.model.dof or not np.isfinite(values).all():
            raise ValueError('q must have finite shape (N, D)')
        return self._native.states(values, goal, None if esdf is None else snapshot(esdf).as_mapping())

    def end_effector(self, q):
        return self._states(q)['ee']

    def evaluate_states(self, q, goal_pose, esdf):
        if goal_pose is None:
            raise ValueError('goal_pose is required for state evaluation')
        return self._states(q, goal_pose, snapshot(esdf))

    def geometry_profile(self, paths, esdf):
        values = np.ascontiguousarray(paths, dtype=np.float32)
        if values.ndim != 3 or values.shape[2] != self.model.dof or values.shape[1] == 0:
            raise ValueError('paths must have shape (N, T, D), T > 0')
        profile = self._states(values.reshape(-1, self.model.dof), None, snapshot(esdf))
        return {k:profile[k].reshape(values.shape[:2]) for k in _PROFILE_FIELDS}

    def evaluate_trajectory(self, q, dq, ddq, jerk, goal_pose, esdf, *, limits):
        if goal_pose is None:
            raise ValueError('goal_pose is required for trajectory evaluation')
        if set(limits) != {'max_acceleration', 'max_jerk'}:
            raise ValueError('limits must contain max_acceleration and max_jerk')
        arrays = [np.ascontiguousarray(v, dtype=np.float32) for v in (q,dq,ddq,jerk)]
        return self._native.trajectory(*arrays, np.asarray(goal_pose, dtype=np.float32),
            snapshot(esdf).as_mapping(), float(limits['max_acceleration']), float(limits['max_jerk']))
