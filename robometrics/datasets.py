from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from robometrics.file_structure import get_dataset_path

_DATASET_FILES = {
    "demo": "demo_set.yaml",
    "mb": "mb_set.yaml",
    "mpinets": "mpinets_set.yaml",
}


def load_raw_dataset(
    name: str, data_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Read a fresh raw dataset from packaged YAML or an explicit directory.

    Values, quaternion conventions and problem ordering are left unchanged.
    Raw loading needs only PyYAML; numerical and geometry imports are deferred
    to the existing structured-data functions.
    """
    if name not in _DATASET_FILES:
        raise ValueError(
            "unknown dataset {!r}; choose one of: {}".format(
                name, ", ".join(sorted(_DATASET_FILES))
            )
        )
    directory = get_dataset_path() if data_dir is None else Path(data_dir)
    path = directory / _DATASET_FILES[name]
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    try:
        with path.open(encoding="utf-8") as stream:
            raw = yaml.load(stream, Loader=loader)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "dataset {!r} not found at {}; set data_dir to a directory "
            "containing {}".format(name, path, _DATASET_FILES[name])
        ) from exc
    if not isinstance(raw, dict):
        raise ValueError("dataset {} must contain a mapping".format(path))
    return raw


def list_problem_groups(
    name: str, data_dir: Optional[Union[str, Path]] = None
) -> Dict[str, int]:
    """Return group sizes in the dataset's original order."""
    raw = load_raw_dataset(name, data_dir)
    return {group: len(problems) for group, problems in raw.items()}


def structure_problems(problem_dict):
    try:
        import geometrout
    except ImportError as exc:
        raise AssertionError(
            "Optional package geometrout not installed; install "
            "robometrics[evaluator] to use structured datasets"
        ) from exc
    import numpy as np

    assert set(problem_dict.keys()) == set(
        [
            "collision_buffer_ik",
            "goal_ik",
            "goal_pose",
            "obstacles",
            "start",
            "world_frame",
        ]
    )
    assert set(problem_dict["obstacles"].keys()) == set(["cuboid", "cylinder"])
    obstacles = {
        "cylinder": {
            k: geometrout.Cylinder(
                height=v["height"],
                radius=v["radius"],
                center=np.array(v["pose"][:3]),
                quaternion=np.array(v["pose"][3:]),
            )
            for k, v in problem_dict["obstacles"]["cylinder"].items()
        },
        "cuboid": {
            k: geometrout.Cuboid(
                dims=np.array(v["dims"]),
                center=np.array(v["pose"][:3]),
                quaternion=np.array(v["pose"][3:]),
            )
            for k, v in problem_dict["obstacles"]["cuboid"].items()
        },
    }
    return {
        "collision_buffer_ik": problem_dict["collision_buffer_ik"],
        "goal_ik": [np.asarray(ik) for ik in problem_dict["goal_ik"]],
        "goal_pose_frame": problem_dict["goal_pose"]["frame"],
        "goal_pose": geometrout.SE3(
            pos=np.array(problem_dict["goal_pose"]["position_xyz"]),
            quaternion=np.array(problem_dict["goal_pose"]["quaternion_wxyz"]),
        ),
        "obstacles": obstacles,
        "start": np.asarray(problem_dict["start"]),
        "world_frame": problem_dict["world_frame"],
    }


def demo_raw() -> Dict[str, Any]:
    return load_raw_dataset("demo")


def structure_dataset(raw_dataset: Dict[str, Any]) -> Dict[str, Any]:
    for problem_set in raw_dataset.values():
        for ii, v in enumerate(problem_set):
            problem_set[ii] = structure_problems(v)
    return raw_dataset


def motion_benchmaker_raw() -> Dict[str, Any]:
    return load_raw_dataset("mb")


def motion_benchmaker() -> Dict[str, Any]:
    raw_data = motion_benchmaker_raw()
    return structure_dataset(raw_data)


def mpinets_raw() -> Dict[str, Any]:
    return load_raw_dataset("mpinets")


def mpinets() -> Dict[str, Any]:
    raw_data = mpinets_raw()
    return structure_dataset(raw_data)


def demo() -> Dict[str, Any]:
    raw_data = demo_raw()
    return structure_dataset(raw_data)
