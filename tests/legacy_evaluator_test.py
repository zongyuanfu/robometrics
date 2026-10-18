"""Regressions for the retained URDF evaluator's configuration and file handling."""
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

geometrout = pytest.importorskip('geometrout')
urchin = pytest.importorskip('urchin')
from robometrics.evaluator import Evaluator
from robometrics.robot import CollisionSphereConfig, Robot


@pytest.fixture
def urdf(monkeypatch):
    value = SimpleNamespace(links=[SimpleNamespace(name='base')], joints=[],
                            link_fk=lambda *args, **kwargs: {'base': np.eye(4)})
    monkeypatch.setattr(urchin.URDF, 'load', lambda *args, **kwargs: value)
    return value


def config(names):
    return CollisionSphereConfig(
        collision_spheres={name: [geometrout.Sphere(center=np.zeros(3), radius=.1)] for name in names},
        self_collision_ignore={name: [] for name in names},
        self_collision_buffer={name: 0. for name in names},
    )


def test_unknown_collision_link_rejects_entire_model(urdf):
    with pytest.raises(ValueError, match='typo'):
        Robot('unused.urdf', config(['base', 'typo']))


def test_missing_fk_link_is_not_silently_ignored(urdf):
    robot = Robot('unused.urdf', config(['base']))
    urdf.link_fk = lambda *args, **kwargs: {}
    with pytest.raises(ValueError, match='missing from FK: base'):
        robot._fk_sphere_info(np.empty(0))


def test_valid_link_preserves_collision_spheres(urdf):
    robot = Robot('unused.urdf', config(['base']))
    centers, radii = robot._fk_sphere_info(np.empty(0))
    np.testing.assert_array_equal(centers, np.zeros((1, 3)))
    np.testing.assert_allclose(radii, [.1])


def test_save_explicit_group_uses_requested_name_without_overwriting_current(tmp_path):
    evaluator = Evaluator(robot=None)
    evaluator.create_new_group('first')
    evaluator.current_group.append('first result')
    evaluator.create_new_group('second')
    evaluator.current_group.append('second result')
    evaluator.save_group(tmp_path, 'run')
    evaluator.save_group(tmp_path, 'run', key='first')
    for key in ('first', 'second'):
        with (tmp_path / f'run_{key}.pkl').open('rb') as stream:
            assert pickle.load(stream) == [f'{key} result']
    assert evaluator.current_group_key == 'second'
