import numpy as np
import pytest

from robometrics.esdf import (
    EsdfContext,
    _sdf_box,
    _sdf_capsule,
    _sdf_cylinder,
    _sdf_sphere,
    build_esdf_from_scene,
)
from robometrics.geometry import Scene


POSE = [0, 0, 0, 1, 0, 0, 0]


@pytest.mark.parametrize("distance,points,expected", [
    (lambda p: _sdf_box(p, [2, 2, 2]),
     [[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 2, 0]], [-1, 0, 1, np.sqrt(2)]),
    (lambda p: _sdf_sphere(p, 1),
     [[0, 0, 0], [1, 0, 0], [2, 0, 0]], [-1, 0, 1]),
    (lambda p: _sdf_capsule(p, 1, [0, 0, -1], [0, 0, 1]),
     [[0, 0, 0], [1, 0, 0], [0, 0, 3], [1, 0, 2]], [-1, 0, 1, np.sqrt(2) - 1]),
    (lambda p: _sdf_cylinder(p, 1, 2),
     [[0, 0, 0], [1, 0, 0], [0, 0, 2], [2, 0, 2]], [-1, 0, 1, np.sqrt(2)]),
])
def test_analytic_distances(distance, points, expected):
    np.testing.assert_allclose(distance(np.array(points, dtype=float)), expected, atol=1e-12)


def test_scene_preserves_names_and_raw_geometry():
    mapping = {"cuboid": {"wall": {"pose": POSE, "dims": [2, 1, 3]}}}
    scene = Scene.create(mapping)
    assert scene.cuboid[0].name == "wall"
    assert scene.cuboid[0].pose == POSE
    assert scene.cuboid[0].dims == [2, 1, 3]
    assert scene.sphere == scene.capsule == scene.cylinder == ()


@pytest.mark.parametrize("kind", ["mesh", "voxel", "unknown"])
def test_unsupported_geometry(kind):
    with pytest.raises(NotImplementedError, match="unsupported analytic"):
        Scene.create({kind: {}})


def test_empty_scene_and_positive_cap():
    empty = build_esdf_from_scene(Scene(), (0, 0, 0), (2, 2, 2), 1)
    assert empty.grid_shape == (2, 2, 2)
    np.testing.assert_array_equal(empty.grid, np.full((2, 2, 2), 10, dtype=np.float32))
    far = Scene.create({"sphere": {"far": {"pose": [100, 0, 0, 1, 0, 0, 0], "radius": 1}}})
    np.testing.assert_array_equal(build_esdf_from_scene(far, (0, 0, 0), (2, 2, 2), 1).grid, empty.grid)


def test_rotated_translated_box_and_signed_interior():
    scene = Scene.create({"cuboid": {"box": {
        "pose": [1, 2, 3, np.sqrt(.5), 0, 0, np.sqrt(.5)], "dims": [2, .5, 2],
    }}})
    result = build_esdf_from_scene(scene, (1, 2, 3), (3, 3, 3), 1)
    assert result.grid[1, 1, 1] == pytest.approx(-.25)
    assert result.grid[2, 1, 1] == pytest.approx(.75)
    assert result.grid[1, 2, 1] == pytest.approx(0, abs=1e-7)


def test_obstacle_union():
    left = {"pose": [-1, 0, 0, 1, 0, 0, 0], "radius": 1}
    right = {"pose": [1, 0, 0, 1, 0, 0, 0], "radius": 1}
    args = ((0, 0, 0), (4, 2, 2), .5)
    a = build_esdf_from_scene(Scene.create({"sphere": {"a": left}}), *args)
    b = build_esdf_from_scene(Scene.create({"sphere": {"b": right}}), *args)
    union = build_esdf_from_scene(Scene.create({"sphere": {"a": left, "b": right}}), *args)
    np.testing.assert_array_equal(union.grid, np.minimum(a.grid, b.grid))


def test_non_divisible_bounds_voxel_centres_and_axis_order():
    scene = Scene.create({"sphere": {"sphere": {"pose": POSE, "radius": .1}}})
    result = build_esdf_from_scene(scene, (.1, .2, .3), (1.3, .8, 1.1), .5)
    assert result.grid_shape == (3, 2, 3)
    np.testing.assert_allclose(result.origin, [-.3, .05, 0], atol=1e-7)
    np.testing.assert_allclose(result.lower_bound, [-.55, -.2, -.25], atol=1e-7)
    np.testing.assert_allclose(result.upper_bound, [.75, .6, .85], atol=1e-7)
    for x in range(3):
        for y in range(2):
            for z in range(3):
                point = [-.3 + .5*x, .05 + .5*y, .5*z]
                assert result.grid[x, y, z] == pytest.approx(np.linalg.norm(point) - .1, abs=1e-7)
    assert result.origin[2] + 2 * result.voxel_size > result.upper_bound[2]


def test_snapshot_owns_contiguous_readonly_data_and_metadata():
    original = (np.arange(128, dtype=np.float64).reshape(8, 4, 4) - 40)[::2, :, ::-1]
    assert not original.flags.c_contiguous and original.strides[-1] < 0
    origin, lower, upper = np.full(3, .5), np.zeros(3), np.full(3, 4.)
    result = EsdfContext(original, origin, 1, original.shape, lower, upper, version=7)
    expected = original.astype(np.float32)
    original[:] = 999
    origin[:] = 999
    upper[:] = 999
    assert result.grid.dtype == np.float32 and result.grid.flags.c_contiguous
    np.testing.assert_array_equal(result.grid, expected)
    assert result.grid.min() < 0
    np.testing.assert_array_equal(result.origin, [.5, .5, .5])
    np.testing.assert_array_equal(result.upper_bound, [4, 4, 4])
    for array in (result.grid, result.origin, result.lower_bound, result.upper_bound):
        assert not array.flags.writeable
    restored = EsdfContext.from_mapping(result.as_mapping())
    assert restored.version == 7 and restored.grid_shape == (4, 4, 4)
    assert not np.shares_memory(result.grid, restored.grid)
    np.testing.assert_array_equal(restored.grid, expected)


def test_fortran_order_snapshot():
    original = np.asfortranarray(np.arange(27).reshape(3, 3, 3))
    result = EsdfContext(original, [.5]*3, 1, original.shape, [0]*3, [3]*3)
    assert result.grid.flags.c_contiguous
    np.testing.assert_array_equal(result.grid, original)


@pytest.mark.parametrize("kwargs,match", [
    ({"workspace_center": [0, 0]}, "workspace_center"),
    ({"workspace_center": [0, 0, np.nan]}, "workspace_center"),
    ({"workspace_extent": [2, 2, 0]}, "workspace_extent"),
    ({"workspace_extent": [2, 2, np.inf]}, "workspace_extent"),
    ({"workspace_extent": [.5, 2, 2]}, "two cells"),
    ({"voxel_size": 0}, "voxel_size"),
    ({"voxel_size": np.nan}, "voxel_size"),
    ({"voxel_size": np.inf}, "voxel_size"),
])
def test_invalid_workspace(kwargs, match):
    args = dict(workspace_center=(0, 0, 0), workspace_extent=(2, 2, 2), voxel_size=1)
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        build_esdf_from_scene(Scene(), **args)


@pytest.mark.parametrize("mapping,match", [
    ({"sphere": {"s": {"pose": [0]*7, "radius": 1}}}, "unit wxyz"),
    ({"sphere": {"s": {"pose": [0, 0, np.nan, 1, 0, 0, 0], "radius": 1}}}, "pose"),
    ({"sphere": {"s": {"pose": POSE, "radius": -1}}}, "radius"),
    ({"cuboid": {"b": {"pose": POSE, "dims": [1, 0, 1]}}}, "dims"),
    ({"cylinder": {"c": {"pose": POSE, "radius": 1, "height": np.inf}}}, "height"),
    ({"capsule": {"c": {"pose": POSE, "radius": 1, "base": [0]*3, "tip": [0]*3}}}, "must differ"),
])
def test_invalid_obstacles(mapping, match):
    with pytest.raises(ValueError, match=match):
        build_esdf_from_scene(Scene.create(mapping), (0, 0, 0), (2, 2, 2), 1)


def test_direct_scene_rejects_meshes():
    with pytest.raises(NotImplementedError, match="mesh/voxel"):
        build_esdf_from_scene(Scene(mesh=(object(),)), (0, 0, 0), (2, 2, 2), 1)


@pytest.mark.parametrize("changes,match", [
    ({"grid_shape": (2, 2, 3)}, "grid_shape"),
    ({"grid": np.full((2, 2, 2), np.nan)}, "finite"),
    ({"voxel_size": 0}, "voxel_size"),
    ({"origin": [0, 0, 0]}, "disagree"),
    ({"lower_bound": [0, 0, np.nan]}, "lower_bound"),
    ({"upper_bound": [3, 2, 2]}, "disagree"),
    ({"upper_bound": [.9, 2, 2]}, "disagree"),
])
def test_invalid_snapshot(changes, match):
    values = dict(grid=np.zeros((2, 2, 2)), origin=[.5]*3, voxel_size=1,
                  grid_shape=(2, 2, 2), lower_bound=[0]*3, upper_bound=[2]*3)
    values.update(changes)
    with pytest.raises(ValueError, match=match):
        EsdfContext.from_mapping(values)


def test_snapshot_rejects_missing_fields_and_non_numpy_grid():
    with pytest.raises(ValueError, match="missing keys"):
        EsdfContext.from_mapping({})
    with pytest.raises(TypeError, match="host numpy"):
        EsdfContext([[[0]*2]*2]*2, [.5]*3, 1, (2, 2, 2), [0]*3, [2]*3)
