"""Optional host integration: point ROBOMETRICS_PLANIGO_ROOT at a checkout."""
import os
from pathlib import Path

import numpy as np
import pytest

from robometrics.datasets import load_raw_dataset
from robometrics.esdf import build_esdf_from_scene


@pytest.fixture(scope="module")
def planigo():
    root = os.environ.get("ROBOMETRICS_PLANIGO_ROOT")
    if not root:
        pytest.skip("set ROBOMETRICS_PLANIGO_ROOT for checkout integration")
    assert (Path(root) / "planigo").is_dir()
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(root)
        from planigo.geometry import esdf_builder, primitives
        from planigo.geometry.context import EsdfContext
        yield esdf_builder, primitives, EsdfContext


@pytest.mark.parametrize("index", range(5))
def test_demo_host_mapping_accepted_by_planigo(planigo, index):
    builder, primitives, context_type = planigo
    problem = next(iter(load_raw_dataset("demo").values()))[index]
    scene = primitives.Scene.create(problem["obstacles"])
    bounds = dict(workspace_center=(0, 0, .34), workspace_extent=(2.4, 2.4, 2.42),
                  voxel_size=.05)
    actual = builder.build_esdf_from_scene(scene, **bounds)
    reference = build_esdf_from_scene(scene, **bounds).as_mapping()
    snapshot = context_type.from_mapping(actual)
    assert isinstance(actual["grid"], np.ndarray)
    assert actual["grid"].dtype == np.float32 and actual["grid"].flags.c_contiguous
    assert not actual["grid"].flags.writeable
    assert not np.shares_memory(snapshot.grid, actual["grid"])
    np.testing.assert_allclose(snapshot.grid, reference["grid"], atol=2e-7, rtol=0)
    for key in ("origin", "grid_shape", "voxel_size", "lower_bound", "upper_bound", "version"):
        np.testing.assert_array_equal(actual[key], reference[key])


def test_planigo_defaults_and_explicit_numpy(planigo):
    builder, primitives, context_type = planigo
    for backend in ("numpy", "cpp"):
        mapping = builder.build_esdf_from_scene(primitives.Scene(), backend=backend)
        snapshot = context_type.from_mapping(mapping)
        assert snapshot.grid_shape == (75, 70, 70)
        np.testing.assert_array_equal(snapshot.grid, 10)
        np.testing.assert_allclose(snapshot.lower_bound, (-.5, -.6, -.1), atol=1e-7)


def test_planigo_rejects_unknown_backend(planigo):
    builder, primitives, _ = planigo
    with pytest.raises(ValueError, match="backend"):
        builder.build_esdf_from_scene(primitives.Scene(), backend="invalid")
