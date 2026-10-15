"""NumPy CPU ESDF with Planigo-compatible, owned host snapshots.

Distance expressions and grid construction follow Planigo's NumPy reference.
Workspace bounds are explicit; this module has no device or planner imports.
"""

from dataclasses import dataclass
from typing import Mapping, Tuple

import numpy as np


@dataclass(frozen=True)
class EsdfContext:
    grid: np.ndarray
    origin: np.ndarray
    voxel_size: float
    grid_shape: Tuple[int, int, int]
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    version: int = 0

    def __post_init__(self):
        if not isinstance(self.grid, np.ndarray):
            raise TypeError("ESDF snapshot requires a host numpy grid")
        grid = np.array(self.grid, dtype=np.float32, copy=True, order="C")
        shape = np.asarray(self.grid_shape)
        if (shape.shape != (3,) or not np.isfinite(shape).all()
                or np.any(shape != np.floor(shape)) or np.any(shape < 2)
                or grid.shape != tuple(shape)):
            raise ValueError("ESDF grid_shape must match a 3D grid with at least two cells per axis")
        if not np.isfinite(grid).all():
            raise ValueError("ESDF grid must be finite")
        voxel = float(self.voxel_size)
        if not np.isfinite(voxel) or voxel <= 0:
            raise ValueError("ESDF voxel_size must be finite and positive")
        for name in ("origin", "lower_bound", "upper_bound"):
            value = np.array(getattr(self, name), dtype=np.float32, copy=True)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f"ESDF {name} must contain three finite values")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        extent = self.upper_bound - self.lower_bound
        tolerance = max(1e-6, voxel * 1e-4)
        if (np.any(extent <= 0)
                or not np.allclose(self.origin, self.lower_bound + voxel / 2, rtol=0, atol=tolerance)
                or np.any(extent > shape * voxel + tolerance)
                or np.any(extent <= (shape - 1) * voxel - tolerance)):
            raise ValueError("ESDF bounds, voxel centers and grid_shape disagree")
        grid.setflags(write=False)
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "grid_shape", tuple(int(v) for v in shape))
        object.__setattr__(self, "voxel_size", voxel)

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]):
        names = ("grid", "origin", "voxel_size", "grid_shape", "lower_bound", "upper_bound")
        missing = set(names) - values.keys()
        if missing:
            raise ValueError(f"ESDF context is missing keys: {sorted(missing)}")
        return cls(**{name: values[name] for name in names}, version=values.get("version", 0))

    def as_mapping(self):
        """Return the host fields consumed by Planigo's CUDA upload boundary."""
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _vector(value, size, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return array


def _positive(value, name):
    array = np.asarray(value)
    if (array.ndim != 0 or array.dtype.kind not in "fiu"
            or not np.isfinite(array) or array <= 0):
        raise ValueError(f"{name} must be finite and positive")


def _validate_scene(scene):
    if scene.mesh or scene.voxel:
        raise NotImplementedError("analytic ESDF does not support mesh/voxel obstacles")
    for kind in ("cuboid", "sphere", "capsule", "cylinder"):
        for obj in getattr(scene, kind):
            pose = _vector(obj.pose, 7, f"{obj.name} pose")
            if not np.isclose(np.linalg.norm(pose[3:]), 1.0, rtol=0, atol=1e-5):
                raise ValueError(f"{obj.name} pose must contain a unit wxyz quaternion")
            if kind == "cuboid":
                if np.any(_vector(obj.dims, 3, f"{obj.name} dims") <= 0):
                    raise ValueError(f"{obj.name} dims must be positive")
            else:
                _positive(obj.radius, f"{obj.name} radius")
                if kind == "cylinder":
                    _positive(obj.height, f"{obj.name} height")
                elif kind == "capsule":
                    base = _vector(obj.base, 3, f"{obj.name} base")
                    tip = _vector(obj.tip, 3, f"{obj.name} tip")
                    if np.array_equal(base, tip):
                        raise ValueError(f"{obj.name} capsule base and tip must differ")


def _quat_to_rot_matrix(qw, qx, qy, qz):
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ])


def _world_to_local(points_world, pose):
    px, py, pz, qw, qx, qy, qz = pose
    return (points_world - np.array([px, py, pz])) @ _quat_to_rot_matrix(qw, qx, qy, qz)


def _sdf_box(points_local, dims):
    q = np.abs(points_local) - np.asarray(dims) / 2.0
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return outside + inside


def _sdf_sphere(points_local, radius):
    return np.linalg.norm(points_local, axis=-1) - radius


def _sdf_capsule(points_local, radius, base, tip):
    ba = np.asarray(tip) - np.asarray(base)
    pa = points_local - np.asarray(base)
    h = np.clip(np.dot(pa, ba) / np.dot(ba, ba), 0.0, 1.0)
    closest = np.asarray(base) + h[..., None] * ba
    return np.linalg.norm(points_local - closest, axis=-1) - radius


def _sdf_cylinder(points_local, radius, height):
    radial = np.linalg.norm(points_local[..., :2], axis=-1) - radius
    axial = np.abs(points_local[..., 2]) - height / 2.0
    q = np.stack([radial, axial], axis=-1)
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.maximum(radial, axial), 0.0)
    return outside + inside


def _cpp_obstacles(scene):
    """Preserve NumPy quaternion arithmetic while precomputing small rotations."""
    records = []
    for kind in ("cuboid", "sphere", "capsule", "cylinder"):
        for obj in getattr(scene, kind):
            record = dict(vars(obj), kind=kind)
            record["rotation"] = _quat_to_rot_matrix(*obj.pose[3:]).ravel()
            records.append(record)
    return records


def _load_cpp():
    try:
        from . import _esdf_cpp
    except ImportError as exc:
        raise RuntimeError(
            "C++ ESDF extension is unavailable; build/install it with CMake for this Python interpreter"
        ) from exc
    return _esdf_cpp


def _build_numpy(scene_cfg, xs, ys, zs):
    grid_xyz = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).astype(np.float32)
    esdf = np.full((len(xs), len(ys), len(zs)), 10.0, dtype=np.float32)
    for obj in scene_cfg.cuboid:
        esdf = np.minimum(esdf, _sdf_box(_world_to_local(grid_xyz, obj.pose), obj.dims))
    for obj in scene_cfg.sphere:
        esdf = np.minimum(esdf, _sdf_sphere(_world_to_local(grid_xyz, obj.pose), obj.radius))
    for obj in scene_cfg.capsule:
        sd = _sdf_capsule(_world_to_local(grid_xyz, obj.pose), obj.radius, obj.base, obj.tip)
        esdf = np.minimum(esdf, sd)
    for obj in scene_cfg.cylinder:
        sd = _sdf_cylinder(_world_to_local(grid_xyz, obj.pose), obj.radius, obj.height)
        esdf = np.minimum(esdf, sd)
    return esdf


def build_esdf_from_scene(scene_cfg, workspace_center, workspace_extent, voxel_size=0.02,
                          *, backend="numpy"):
    """Build a readonly host ESDF from four types of analytic obstacles.

    Distances are the pointwise minimum of primitive SDFs, capped above at
    10 metres. Empty scenes contain 10 everywhere. Grid axes are (x, y, z),
    and origin is the first voxel centre. Physical upper bounds remain those
    requested even when ceil(extent / voxel_size) extends the grid beyond them.
    Every axis must have at least two voxels, matching Planigo's host contract.
    NumPy is the default. Explicit C++ selection requires the optional extension
    and never silently falls back to NumPy.
    """
    if backend not in ("numpy", "cpp"):
        raise ValueError("backend must be 'numpy' or 'cpp'")
    _vector(workspace_center, 3, "workspace_center")
    extent = _vector(workspace_extent, 3, "workspace_extent")
    if np.any(extent <= 0):
        raise ValueError("workspace_extent must be positive")
    _positive(voxel_size, "voxel_size")
    _validate_scene(scene_cfg)

    cx, cy, cz = workspace_center
    ex, ey, ez = workspace_extent
    x_min, y_min, z_min = cx - ex / 2.0, cy - ey / 2.0, cz - ez / 2.0
    nx, ny, nz = (int(np.ceil(value / voxel_size)) for value in (ex, ey, ez))
    if min(nx, ny, nz) < 2:
        raise ValueError("ESDF requires at least two cells per axis")
    xs = np.linspace(x_min + voxel_size / 2.0, x_min + (nx - 0.5) * voxel_size, nx)
    ys = np.linspace(y_min + voxel_size / 2.0, y_min + (ny - 0.5) * voxel_size, ny)
    zs = np.linspace(z_min + voxel_size / 2.0, z_min + (nz - 0.5) * voxel_size, nz)
    if backend == "numpy":
        esdf = _build_numpy(scene_cfg, xs, ys, zs)
    else:
        esdf = _load_cpp().build(xs, ys, zs, _cpp_obstacles(scene_cfg))

    origin = np.array([x_min + voxel_size / 2.0, y_min + voxel_size / 2.0,
                       z_min + voxel_size / 2.0], dtype=np.float32)
    lower = np.array([x_min, y_min, z_min], dtype=np.float32)
    upper = lower + np.array([ex, ey, ez], dtype=np.float32)
    return EsdfContext(esdf, origin, np.float32(voxel_size), esdf.shape, lower, upper)
