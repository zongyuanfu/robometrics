# Robometrics

Robometrics provides packaged manipulation datasets and robot motion metrics.
This checkout adds lightweight raw dataset access and NumPy CPU ESDF for planners such as Planigo,
while retaining the existing structured dataset and evaluator interfaces.
## Installation

Install from this checkout:

```bash
python -m pip install .
```

The currently supported Python version is 3.11. The core dependencies are NumPy and PyYAML. Raw dataset loading uses only PyYAML;
it does not import NumPy, geometrout, urchin, JAX, Torch, cuRobo or Planigo.
The NumPy requirement allows both 1.x and 2.x.

The core wheel and raw loaders were verified on Python 3.11 with PyYAML 6.0.3
and NumPy 1.26.4 / 2.4.6 in separate environments without optional dependencies.

The existing structured datasets and evaluator use optional geometry dependencies:

```bash
python -m pip install '.[evaluator]'
```

Raw loading compatibility does not imply NumPy 2 compatibility of those optional
dependencies. `poetry.lock` is synchronized with the current declarations.
The existing structured loaders were separately verified with NumPy 1.26.4 and
geometrout 0.1.3.

## Usage

```python
from robometrics.datasets import load_raw_dataset, list_problem_groups

groups = list_problem_groups("mb")  # Eight groups, 100 problems each.
dataset = load_raw_dataset("mb")
problem = dataset["bookshelf_small_panda"][0]
start = problem["start"]
goal = problem["goal_pose"]  # position_xyz and quaternion_wxyz, unchanged.

# Optional directory containing demo_set.yaml, mb_set.yaml, or mpinets_set.yaml.
custom_data = load_raw_dataset("demo", data_dir="/path/to/datasets")
```

Supported dataset names are `demo` (5 problems), `mb` (800), and `mpinets` (1800).
The returned dictionary preserves the original fields, group and problem order,
and quaternion conventions. Each call parses a fresh dictionary; mutations do
not affect later calls. Files are never downloaded automatically. Missing files
report the expected path and how to select a data directory.

The existing `demo_raw()`, `motion_benchmaker_raw()`, `mpinets_raw()`, and
`robometrics.file_structure.get_dataset_path()` remain available. This keeps the
existing Planigo resource lookup compatible with a local installation.

## CPU ESDF

```python
from robometrics.geometry import Scene
from robometrics.esdf import build_esdf_from_scene

scene = Scene.create(problem["obstacles"])
context = build_esdf_from_scene(
    scene,
    workspace_center=(0.25, 0.10, 0.60),
    workspace_extent=(1.30, 0.91, 1.43),
    voxel_size=0.02,
)
host_fields = context.as_mapping()
```

Supply bounds appropriate for your robot; the example bounds are explicit,
not an automatic Franka workspace. Cuboids, spheres, capsules and finite
cylinders use the raw xyz + unit quaternion wxyz convention. Sizes and radii
must be positive, and capsule endpoints must differ. Mesh and voxel obstacles
are unsupported.

The grid is an owned, readonly, C-contiguous float32 NumPy array with axes
`(x, y, z)`. Values are the minimum of analytic primitive distances, negative
inside obstacles, capped above at 10 metres; an empty scene contains 10.
`origin` is the first voxel centre. Shapes use `ceil(extent / voxel_size)`,
with at least two cells per axis; physical bounds remain the requested bounds
even when the grid extends beyond them. Query consumers must use those bounds
for OOB checks.

`as_mapping()` retains Planigo's `grid`, `origin`, `voxel_size`, `grid_shape`,
`lower_bound`, `upper_bound` and `version` fields for its existing host/CUDA
boundary. CPU construction imports no JAX, GPU runtime or planner. Distance
queries are not part of this stage. Optional single-thread C++ construction uses
the same context and host fields.

The numerical source is Planigo commit `ba59e35adcfc4a6048fe27e9c03e26c8ec7516d0`,
files `planigo/geometry/primitives.py`, `context.py`, and the NumPy computation
in `esdf_builder.py`. In NumPy 1.26.4 and 2.4.6, 90 scenes matched the original
float32 host reference voxel-for-voxel. This comparison does not run JAX.

## Optional C++ ESDF

Use a local project environment. A C++17 compiler, CMake 3.18+ and Python
headers are required for this optional build:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e . pytest 'pybind11>=2.12'
cmake -S . -B build/esdf -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build/esdf -j 2
cmake --install build/esdf --prefix .
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests
```

The install command above places the extension in the source package for editable
use. For a regular installed package, use its site-packages directory as the
CMake install prefix. The standard wheel remains pure Python; the sdist includes
CMake and C++ sources. Generated binaries, build files and `.venv` are ignored.

Select C++ explicitly:

```python
context = build_esdf_from_scene(
    scene, workspace_center=(0, 0, 0.34),
    workspace_extent=(2.4, 2.4, 2.42), voxel_size=0.02, backend="cpp",
)
```

NumPy remains the default. Selecting C++ without the extension raises a clear
error. C++ releases the GIL, uses one thread, float32 voxel coordinates/output and
double intermediate calculations, without fast-math. It packs obstacle constants
and computes distances directly into one grid. The tested mixed, rotated and
near-threshold scenes agree within 2e-7 metres; this is not a bitwise equivalence
claim. The extension built with pybind11 3.0.4 is tested with NumPy 1.26.4 / 2.4.6.

Measure fresh grid construction with separate processes for each backend:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python benchmarks/esdf.py \
    --dataset demo --case-limit 5 --repeat 3 --output build/esdf-benchmark-demo.json
```

The report includes obstacle packing, computation, snapshot and total times,
separate dataset loading/scene preparation times, peak RSS, CPU, compiler and
source metadata. Each process discards one warmup grid, then builds a new grid
for every repetition. Linux RSS uses the worker's `VmHWM`.

On AMD Ryzen 7 7700, GCC 11.4 Release, Python 3.11 and NumPy 2.4.6,
with a 120 × 120 × 121 grid (2 cm voxels):

| Sample | NumPy median / P90 | C++ median / P90 | Peak RSS NumPy / C++ |
|---|---|---|---|
| demo, all 5 problems | 771 / 779 ms | 62 / 64 ms | 275 / 50 MiB |
| MB, first problem in each of 8 groups | 1119 / 1596 ms | 78 / 136 ms | 315 / 50 MiB |

These are medians across case medians with three builds per case; RSS is the
maximum worker peak across sampled cases. The demo cases share geometry. These
small samples do not establish performance for all MB800 or MPINets problems.

## Local Planigo smoke

Integration checks can run from this checkout and its `.venv` without changing
Planigo's environment:

```bash
ROBOMETRICS_PLANIGO_ROOT=/path/to/planigo PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    .venv/bin/python -m pytest tests/planigo_test.py -q
.venv/bin/python benchmarks/planigo_smoke.py \
    --planigo-root /path/to/planigo --build-dir /path/to/planigo/igo_solver_pkg/build \
    --output build/planigo-smoke.json
```

The Planigo checkout must provide the thin forwarding ESDF entry point. Its CUDA
binary must match its current robot model. Host tests verify all five demo maps
and the legacy workspace defaults. The native GPU smoke uploads an obstacle map,
initializes/solves/executes, resets the task, then uploads an empty map and repeats.
It uses NumPy/PyYAML and the existing CUDA extension, without JAX or SciPy. This
checks native interface integration, not full planner convergence or evaluation.

On 2026-10-15, the local host integration and ESDF tests passed (68 tests), and the
Release/sm_120 native GPU smoke passed with one grid allocation for both maps.
The existing Planigo environment, lock synchronization, JAX query integration and
full benchmark validation remain separate steps.

## Data provenance

The bundled YAML files are unchanged from the local Git baseline `d7ec62d`
(tag `robometrics-baseline-20261014`). The upstream repository is
[fishbotics/robometrics](https://github.com/fishbotics/robometrics); its exact
source commit was not recorded in the imported checkout.

| File | SHA256 |
|---|---|
| `demo_set.yaml` | `712b9fc93802cc39ed769b32b1e3ad5006196db73962da59ebd68012c44ce74d` |
| `mb_set.yaml` | `5165ad4fb55f93c63dbbdda6f25a14f3ceb24ee3305e27ad317f38f1e0d6ed0d` |
| `mpinets_set.yaml` | `9189186d83e51600a2c768aa7657933aa052150501852771994757052b797bbf` |

The existing code and data notices remain in [LICENSE](LICENSE) and
[Licenses](Licenses), including the Motion Policy Networks and Motion Benchmaker
notices. Both notices are included in distribution archives.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Citation
If you find this work useful in your work, please cite:

	@inproceedings{fishman2022mpinets,
	  title={Motion Policy Networks},
	  author={Fishman, Adam and Murali, Adithyavairavan and Eppner, Clemens and Peele, Bryan and Boots, Byron and Fox, Dieter},
    booktitle = {Proceedings of the 6th Conference on Robot Learning (CoRL)},
	  year={2022}
	}
