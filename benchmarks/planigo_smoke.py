"""Run the existing Planigo CUDA binding against local Robometrics host maps.

Uses only NumPy/PyYAML and the compiled CUDA module, without JAX or SciPy.
This checks the native upload/solve/execution/reset boundary, not planner success.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from robometrics.datasets import load_raw_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planigo-root", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.planigo_root.resolve()))
    from planigo.backend.loading import configure_extension, load_extension, validate_extension_model
    from planigo.geometry.esdf_builder import build_esdf_from_scene
    from planigo.geometry.context import EsdfContext
    from planigo.geometry.primitives import Scene

    report = {"python": sys.executable, "numpy": np.__version__, "stages": []}
    try:
        configure_extension("planigo_cuda_bspline", args.build_dir)
        native = validate_extension_model(load_extension("planigo_cuda_bspline"))
        report.update(binary=native.__file__, model_sha256=native.MODEL_SHA256)
        report["stages"].append("native model validated")
        problem = next(iter(load_raw_dataset("demo").values()))[0]
        bounds = dict(workspace_center=(0, 0, .34), workspace_extent=(2.4, 2.4, 2.42),
                      voxel_size=.02)
        maps = [build_esdf_from_scene(Scene.create(problem["obstacles"]), **bounds),
                build_esdf_from_scene(Scene(), **bounds)]
        snapshots = [EsdfContext.from_mapping(value) for value in maps]
        assert not np.array_equal(snapshots[0].grid, snapshots[1].grid)
        report["stages"].append("two different host maps validated")
        session = native.Solver(40, 10, 8, 5, .2, 42)
        report["stages"].append("CUDA session created")
        for index, snapshot in enumerate(snapshots):
            if index:
                session.reset_task(43)
                report["stages"].append("task reset")
            session.upload_esdf(snapshot.grid)
            report["stages"].append(f"map {index} uploaded")
            context = native.Context()
            context.nx, context.ny, context.nz = snapshot.grid_shape
            context.origin, context.lower_bound, context.upper_bound = (
                snapshot.origin, snapshot.lower_bound, snapshot.upper_bound)
            context.voxel_size = snapshot.voxel_size
            q = np.asarray(problem["start"], dtype=np.float32)
            zero = np.zeros(7, dtype=np.float32)
            context.start_q, context.start_dq, context.start_ddq = q, zero, zero
            context.goal_q = np.asarray(problem["goal_ik"][0], dtype=np.float32)
            pose = problem["goal_pose"]
            w, x, y, z = pose["quaternion_wxyz"]
            context.goal_pose_xyzw = np.asarray([*pose["position_xyz"], x, y, z, w], np.float32)
            for name, value in dict(environment_clearance=.01, self_clearance=0.,
                                    path_position=1., path_orientation=1.).items():
                setattr(context.weights, name, value)
            reference = session.initialize_distribution(context, .1, .95)
            free = session.solve(context, 8)
            diagnostic = session.diagnostics(context, free)
            executed = session.execute(free, q, zero, zero)
            for array in (reference, free, *executed):
                assert np.isfinite(array).all()
            assert np.isfinite(diagnostic["cost"])
            assert all(np.asarray(array).shape == (7,) for array in executed)
            report["stages"].append(f"map {index} initialized, solved, diagnosed and executed")
        report["resources"] = dict(session.resource_stats())
        del session
        report["stages"].append("session released")
        report["passed"] = True
    except Exception as error:
        report.update(passed=False, error=f"{type(error).__name__}: {error}")
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
