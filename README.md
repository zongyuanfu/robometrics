# Robometrics

Robot motion datasets, evaluation utilities, and CPU ESDF construction.

Based on [fishbotics/robometrics](https://github.com/fishbotics/robometrics) by
Adam Fishman and Balakumar Sundaralingam. **zongyuanfu** adds unified raw dataset
loading, standalone geometry, NumPy and C++ ESDF backends, automatic C++ builds,
and Planigo integration checks and benchmarks. Existing structured dataset and
evaluator interfaces are retained.

## Installation

Requires **Python 3.11**. Source installation also requires a **C++17 compiler**
and **Python development headers**; the C++ extension is built automatically.
Runtime dependencies are NumPy and PyYAML.

```bash
python -m pip install .
# Or, in an activated environment:
uv pip install .
```

Install from GitHub after this version is pushed:

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
axes. Distances are in metres and negative inside obstacles. Meshes and distance
queries are not supported. The C++ backend uses one CPU thread and releases the GIL.

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
