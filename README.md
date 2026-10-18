# Robometrics

Robot motion datasets, CPU ESDF construction, and trajectory evaluation.

Based on [fishbotics/robometrics](https://github.com/fishbotics/robometrics) by
Adam Fishman and Balakumar Sundaralingam. **zongyuanfu** adds unified raw dataset
loading, standalone geometry, NumPy and C++ ESDF backends, and C++ CPU trajectory
evaluation. Existing structured dataset and evaluator interfaces are retained.

## Installation

Requires **Python 3.11**. Source installation also requires a **C++17 compiler**
and **Python development headers**; the C++ extensions are built automatically.
Runtime dependencies are NumPy and PyYAML.

```bash
python -m pip install .
# Or, in an activated environment:
uv pip install .
```

Or install from GitHub:

```bash
python -m pip install "git+https://github.com/zongyuanfu/robometrics.git"
```

The legacy structured datasets and evaluator additionally require:

```bash
python -m pip install 'geometrout>=0.1.0.10,<0.2' 'urchin>=0.0.26,<0.0.27'
```

These optional dependencies were verified with NumPy 1.26.4; their NumPy 2
compatibility is not established.

## Datasets

Bundled datasets: `demo` (5 problems), `mb` (800), and `mpinets` (1800).

```python
from robometrics.datasets import load_raw_dataset, list_problem_groups

print(list_problem_groups("mb"))
dataset = load_raw_dataset("mb")
problem = dataset["bookshelf_small_panda"][0]
start = problem["start"]
goal = problem["goal_pose"]  # position xyz, quaternion wxyz
```

Raw loading preserves the original fields and ordering, uses only PyYAML, and
requires no planner or GPU. Pass `data_dir="/path/to/datasets"` to load custom
YAML files. The existing `demo_raw()`, `motion_benchmaker_raw()`, and
`mpinets_raw()` functions remain available.

## CPU ESDF

Using `problem` from the example above:

```python
from robometrics.geometry import Scene
from robometrics.esdf import build_esdf_from_scene

context = build_esdf_from_scene(
    Scene.create(problem["obstacles"]),
    workspace_center=(0.25, 0.10, 0.60),
    workspace_extent=(1.30, 0.91, 1.43),
    voxel_size=0.02,
    backend="cpp",  # "numpy" is the default
)
grid = context.grid
host_fields = context.as_mapping()  # Planigo-compatible host fields
```

Choose workspace bounds for your robot. Both backends support cuboids, spheres,
capsules, and cylinders, and return a readonly `float32` grid with `(x, y, z)`
axes. Distances are in metres and negative inside obstacles. Mesh obstacles are
not supported. The C++ backend uses one CPU thread and releases the GIL.

## CPU trajectory evaluation

`CpuTrajectoryEvaluator` uses the compiled C++ backend and requires no planner,
GPU, or URDF library. Supply a `robometrics.kinematics.KinematicModel` containing
link transforms and topology, fixed/revolute joints, end-effector configuration,
collision spheres, self-collision pairs, and joint limits. There is no built-in
robot model or URDF importer.

Here, `D` is the number of joints, `N` the batch size, and `T` the number of
trajectory samples. Inputs must contain finite values.

| Method | Inputs | Outputs |
| --- | --- | --- |
| `end_effector` | Joint positions `(N, D)` | End-effector poses `(N, 7)` |
| `evaluate_states` | Joint positions `(N, D)`, goal pose, ESDF | Per-state poses, goal errors, clearances, joint-limit violations, and sphere counts |
| `geometry_profile` | Paths `(N, T, D)`, ESDF | Per-sample clearances, joint-limit violations, and sphere counts `(N, T)` |
| `evaluate_trajectory` | Positions and derivatives `(T, D)`, goal pose, ESDF, dynamics limits | End-effector poses, terminal goal errors, and trajectory geometry/dynamics summaries |

With a caller-supplied `model` and sampled arrays `q`, `dq`, `ddq`, and `jerk`:

```python
import numpy as np
from robometrics.trajectory_evaluation import CpuTrajectoryEvaluator

# Convert the dataset goal to evaluation format: position xyz, quaternion xyzw.
quaternion = np.asarray(goal["quaternion_wxyz"])
goal_pose = np.concatenate((goal["position_xyz"], quaternion[[1, 2, 3, 0]]))
evaluator = CpuTrajectoryEvaluator(model)
result = evaluator.evaluate_trajectory(
    q, dq, ddq, jerk,
    goal_pose=goal_pose,
    esdf=context,
    limits={"max_acceleration": 15.0, "max_jerk": 500.0},
)
print(result["position_error"], result["min_clearance"])
```

Positions and clearances use metres; joint angles use radians. Supply unit goal
quaternions: the evaluator does not normalize them. Orientation error is
`1 - min(1, abs(dot(q_ee, q_goal)))`.

Derivatives are used as supplied, without differentiation or resampling. Velocity
limits come from the model; acceleration and jerk limits are positive scalars.
Trajectory summaries include maximum joint, velocity, and acceleration violations,
maximum absolute jerk, minimum environment/self clearances, and the total count of
out-of-bounds spheres. Check sphere counts alongside clearances: no valid
environment spheres or no self-collision pairs yields positive infinity.
Success thresholds remain the caller's responsibility.

Reuse an `EsdfContext` to avoid copying its grid. Each evaluator uses one CPU
thread, releases the GIL, and supports one call at a time.

## Development

```bash
python -m pip install -e . pytest
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q
```

Reinstall after changing C++ sources. Planigo integration tests require an
external checkout and are skipped unless configured.
See [implementation and validation notes](docs/esdf-notes.md) for benchmarks,
Planigo checks, numerical conventions, and dataset hashes.

## Sources and license

- Original code: [fishbotics/robometrics](https://github.com/fishbotics/robometrics).
  Imported baseline: `d7ec62d`; the exact upstream commit was not recorded.
- Geometry and NumPy ESDF: adapted from
  [Planigo](https://github.com/zongyuanfu/planigo), commit
  `ba59e35adcfc4a6048fe27e9c03e26c8ec7516d0`.
- Code: [MIT](LICENSE), retaining the original copyright and adding
  zongyuanfu's 2026 modifications and additions.
- Bundled data: [third-party notices](Licenses) for Motion Policy Networks
  (MIT) and Motion Benchmaker (BSD 3-Clause). Both license files ship with
  source and wheel distributions.
