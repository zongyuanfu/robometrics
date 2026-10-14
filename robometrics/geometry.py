"""Analytic obstacles in metres; poses are xyz + unit quaternion wxyz.

Adapted from Planigo's geometry/primitives.py. No planner or geometry framework
is needed to represent the original dataset obstacles.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class Cuboid:
    name: str
    pose: List[float]
    dims: List[float]


@dataclass(frozen=True)
class Sphere:
    name: str
    pose: List[float]
    radius: float


@dataclass(frozen=True)
class Capsule:
    name: str
    pose: List[float]
    radius: float
    base: List[float]
    tip: List[float]


@dataclass(frozen=True)
class Cylinder:
    name: str
    pose: List[float]
    radius: float
    height: float


@dataclass(frozen=True)
class Scene:
    cuboid: Tuple[Cuboid, ...] = ()
    sphere: Tuple[Sphere, ...] = ()
    capsule: Tuple[Capsule, ...] = ()
    cylinder: Tuple[Cylinder, ...] = ()
    mesh: tuple = ()
    voxel: tuple = ()

    @classmethod
    def create(cls, mapping):
        """Keep raw obstacle names, poses and dimensions without conversion."""
        kinds = dict(cuboid=Cuboid, sphere=Sphere, capsule=Capsule, cylinder=Cylinder)
        unknown = mapping.keys() - kinds.keys()
        if unknown:
            raise NotImplementedError(f"unsupported analytic obstacles: {sorted(unknown)}")
        return cls(**{
            name: tuple(kind(name=key, **value) for key, value in mapping.get(name, {}).items())
            for name, kind in kinds.items()
        })
