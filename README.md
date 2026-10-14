# Robometrics

Robometrics provides packaged manipulation datasets and robot motion metrics.
This checkout adds lightweight raw dataset access and NumPy CPU ESDF for planners such as Planigo,
while retaining the existing structured dataset and evaluator interfaces.
## Installation

Install from this checkout:

```bash
python -m pip install .
```

The core dependencies are NumPy and PyYAML. Raw dataset loading uses only PyYAML;
it does not import NumPy, geometrout, urchin, JAX, Torch, cuRobo or Planigo.
The NumPy requirement allows both 1.x and 2.x.

The core wheel and raw loaders were verified on Python 3.11 with PyYAML 6.0.3
and NumPy 1.26.4 / 2.4.6 in separate environments without optional dependencies.

The existing structured datasets and evaluator use optional geometry dependencies:

```bash
python -m pip install '.[evaluator]'
```

Raw loading compatibility does not imply NumPy 2 compatibility of those optional
dependencies. The historical `poetry.lock` predates the current dependency
declarations and needs regeneration before using `poetry install`.
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
queries and C++ acceleration are not part of this stage.

The numerical source is Planigo commit `ba59e35adcfc4a6048fe27e9c03e26c8ec7516d0`,
files `planigo/geometry/primitives.py`, `context.py`, and the NumPy computation
in `esdf_builder.py`. In NumPy 1.26.4 and 2.4.6, 90 scenes matched the original
float32 host reference voxel-for-voxel. This comparison does not run JAX.

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
