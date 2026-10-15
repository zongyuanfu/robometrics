import numpy as np
import pytest

from robometrics.datasets import demo_raw
from robometrics.esdf import _cpp_obstacles, build_esdf_from_scene
from robometrics.geometry import Scene


native = pytest.importorskip("robometrics._esdf_cpp")
POSE = [0, 0, 0, 1, 0, 0, 0]


def compare(scene, centre=(.13, -.27, .31), extent=(1.3, .91, 1.43), voxel=.13):
    expected = build_esdf_from_scene(scene, centre, extent, voxel)
    actual = build_esdf_from_scene(scene, centre, extent, voxel, backend="cpp")
    # Double scalar reductions can differ from NumPy/BLAS before the final
    # float32 rounding. 2e-7 m is below two float32 ULPs at one metre.
    np.testing.assert_allclose(actual.grid, expected.grid, rtol=0, atol=2e-7)
    for name in ("origin", "lower_bound", "upper_bound", "voxel_size", "grid_shape", "version"):
        np.testing.assert_array_equal(actual.as_mapping()[name], expected.as_mapping()[name])
    assert actual.grid.dtype == np.float32 and actual.grid.flags.c_contiguous
    assert not actual.grid.flags.writeable
    return actual, expected


def test_empty_scene():
    actual, expected = compare(Scene())
    np.testing.assert_array_equal(actual.grid, expected.grid)


@pytest.mark.parametrize("kind,parameters", [
    ("cuboid", {"dims": [.7, .2, 1.3]}),
    ("sphere", {"radius": .23}),
    ("capsule", {"radius": .12, "base": [-.1, .2, -.4], "tip": [.2, -.1, .3]}),
    ("cylinder", {"radius": .21, "height": .73}),
])
@pytest.mark.parametrize("quaternion", [[1, 0, 0, 0], [.5, .5, .5, .5]])
def test_each_primitive_and_rotation(kind, parameters, quaternion):
    compare(Scene.create({kind: {"test": dict(parameters, pose=[.2, -.1, .3] + quaternion)}}))


def test_random_mixed_scenes():
    rng = np.random.default_rng(20261014)
    for _ in range(64):
        mapping = {}
        for kind in ("cuboid", "sphere", "capsule", "cylinder"):
            objects = {}
            for i in range(int(rng.integers(1, 4))):
                quaternion = rng.normal(size=4)
                quaternion /= np.linalg.norm(quaternion)
                obj = {"pose": rng.uniform(-.5, .5, size=3).tolist() + quaternion.tolist()}
                if kind == "cuboid":
                    obj["dims"] = rng.uniform(.01, 1, size=3).tolist()
                else:
                    obj["radius"] = float(rng.uniform(.01, .4))
                    if kind == "capsule":
                        obj["base"] = rng.uniform(-.5, 0, size=3).tolist()
                        obj["tip"] = rng.uniform(0, .5, size=3).tolist()
                    elif kind == "cylinder":
                        obj["height"] = float(rng.uniform(.01, 1))
                objects[str(i)] = obj
            mapping[kind] = objects
        actual, expected = compare(Scene.create(mapping))
        np.testing.assert_array_equal(actual.grid < 0, expected.grid < 0)
        np.testing.assert_array_equal(actual.grid < .01, expected.grid < .01)


def test_demo_raw_scenes():
    for problem in demo_raw()["bookshelf_small_panda"]:
        compare(Scene.create(problem["obstacles"]), (.25, .1, .6), (1.3, .91, 1.43), .05)


@pytest.mark.parametrize("radius", [.25 - 1e-7, .25, .25 + 1e-7, .24 - 1e-7, .24, .24 + 1e-7])
def test_surface_and_clearance_thresholds(radius):
    actual, expected = compare(Scene.create({"sphere": {"sphere": {"pose": POSE, "radius": radius}}}),
                               (.25, 0, 0), (1.5, 1.5, 1.5), .5)
    np.testing.assert_array_equal(actual.grid < 0, expected.grid < 0)
    np.testing.assert_array_equal(actual.grid < .01, expected.grid < .01)


def test_native_axis_dtype_strides_and_owned_output():
    source = np.linspace(-1, 1, 12, dtype=np.float64)
    axis = source[::2]
    records = _cpp_obstacles(Scene.create({"sphere": {"s": {"pose": POSE, "radius": .2}}}))
    grid = native.build(axis, axis, axis, records)
    centres = axis.astype(np.float32)
    xyz = np.stack(np.meshgrid(centres, centres, centres, indexing="ij"), axis=-1).astype(float)
    expected = np.linalg.norm(xyz, axis=-1) - .2
    np.testing.assert_allclose(grid, expected.astype(np.float32), rtol=0, atol=2e-7)
    source[:] = 100
    assert grid.flags.owndata and grid.flags.c_contiguous and grid.dtype == np.float32
    np.testing.assert_allclose(grid, expected.astype(np.float32), rtol=0, atol=2e-7)


@pytest.mark.parametrize("axis", [np.zeros((2, 2)), np.zeros(1), np.array([0, np.nan])])
def test_native_rejects_bad_axes(axis):
    with pytest.raises(ValueError, match="axes"):
        native.build(axis, np.zeros(2), np.zeros(2), [])


def test_native_profiling_and_single_thread_contract():
    timings = {}
    result = native.build(np.zeros(2), np.zeros(2), np.zeros(2), [], timings)
    assert result.shape == (2, 2, 2)
    assert timings["packing_s"] >= 0 and timings["compute_s"] >= 0
    assert native.threads == 1 and len(native.source_sha256) == 64
