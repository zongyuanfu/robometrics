"""ESDF construction timing and peak RSS; every backend runs in a fresh process."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import time
import traceback

import numpy as np

from robometrics import esdf
from robometrics.datasets import load_raw_dataset
from robometrics.geometry import Scene


def measure(connection, scene, backend, centre, extent, voxel, repeat):
    try:
        stages = {}

        def timed(name, function):
            def call(*args, **kwargs):
                started = time.perf_counter()
                value = function(*args, **kwargs)
                stages[name] = time.perf_counter() - started
                return value
            return call

        esdf.EsdfContext = timed("snapshot_s", esdf.EsdfContext)
        native_info = None
        if backend == "cpp":
            native = esdf._load_cpp()
            esdf._cpp_obstacles = timed("packing_python_s", esdf._cpp_obstacles)
            original = native.build

            def native_build(*args):
                timings = {}
                value = original(*args, timings)
                stages["packing_native_s"] = timings["packing_s"]
                stages["compute_s"] = timings["compute_s"]
                return value

            native.build = native_build
            native_info = {name: getattr(native, name) for name in (
                "compiler", "build_type", "source_sha256", "pybind11_version",
                "threads", "implementation_version")}
        else:
            esdf._build_numpy = timed("compute_s", esdf._build_numpy)
        # Warm imports/allocator once. This grid is discarded, never reused.
        warmup = esdf.build_esdf_from_scene(scene, centre, extent, voxel, backend=backend)
        shape = warmup.grid_shape
        del warmup
        samples = []
        for _ in range(repeat):
            stages.clear()
            started = time.perf_counter()
            output = esdf.build_esdf_from_scene(scene, centre, extent, voxel, backend=backend)
            total = time.perf_counter() - started
            sample = {key: value * 1000 for key, value in stages.items()}
            sample["total_s"] = total * 1000
            samples.append({key.replace("_s", "_ms"): value for key, value in sample.items()})
            del output
        # Linux ru_maxrss can retain the parent's pre-exec maximum after spawn.
        # VmHWM measures this worker's current address space instead.
        if Path("/proc/self/status").exists():
            status = Path("/proc/self/status").read_text().splitlines()
            peak_mb = int(next(line for line in status if line.startswith("VmHWM:")).split()[1]) / 1024
            peak_source = "worker VmHWM"
        else:
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak_mb = peak / (1024**2 if platform.system() == "Darwin" else 1024)
            peak_source = "worker ru_maxrss"
        connection.send({"backend": backend, "grid_shape": shape, "samples": samples,
                         "median_ms": statistics.median(s["total_ms"] for s in samples),
                         "peak_rss_mb": peak_mb, "peak_rss_source": peak_source,
                         "native_build": native_info})
    except Exception:
        connection.send({"error": traceback.format_exc()})
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("demo", "mb", "mpinets"), default="demo")
    parser.add_argument("--group")
    parser.add_argument("--case-limit", type=int, default=5, help="Total cases across selected groups")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--backend", choices=("numpy", "cpp", "both"), default="both")
    parser.add_argument("--voxel-size", type=float, default=.02)
    parser.add_argument("--workspace-center", nargs=3, type=float, default=(0, 0, .34))
    parser.add_argument("--workspace-extent", nargs=3, type=float, default=(2.4, 2.4, 2.42))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.case_limit <= 0 or args.repeat <= 0:
        parser.error("case-limit and repeat must be positive")
    loading_started = time.perf_counter()
    data = load_raw_dataset(args.dataset)
    dataset_loading_ms = (time.perf_counter() - loading_started) * 1000
    groups = list(data) if args.group is None else [args.group]
    if any(group not in data for group in groups):
        parser.error("unknown group")
    backends = ("numpy", "cpp") if args.backend == "both" else (args.backend,)
    ctx = mp.get_context("spawn")
    results = []
    # Sample groups in turn so a small MB run covers more than its first group.
    for index in range(max(len(data[group]) for group in groups)):
        if len(results) >= args.case_limit * len(backends):
            break
        for group in groups:
            if len(results) >= args.case_limit * len(backends):
                break
            if index >= len(data[group]):
                continue
            problem = data[group][index]
            scene_started = time.perf_counter()
            scene = Scene.create(problem["obstacles"])
            scene_preparation_ms = (time.perf_counter() - scene_started) * 1000
            case_id = f"{args.dataset}_{group}_{index:04d}"
            for backend in backends:
                receiver, sender = ctx.Pipe(duplex=False)
                worker = ctx.Process(target=measure, args=(sender, scene, backend,
                    args.workspace_center, args.workspace_extent, args.voxel_size, args.repeat))
                worker.start()
                sender.close()
                try:
                    result = receiver.recv()
                except EOFError:
                    worker.join()
                    raise RuntimeError(f"benchmark worker failed: exit code {worker.exitcode}")
                finally:
                    receiver.close()
                worker.join()
                if "error" in result:
                    raise RuntimeError(result["error"])
                if worker.exitcode:
                    raise RuntimeError(f"benchmark worker failed: exit code {worker.exitcode}")
                result["case_id"] = case_id
                result["scene_preparation_ms"] = scene_preparation_ms
                result["obstacles"] = sum(len(getattr(scene, kind)) for kind in
                                          ("cuboid", "sphere", "capsule", "cylinder"))
                results.append(result)
                print(f"{case_id} {backend}: {result['median_ms']:.1f} ms, "
                      f"peak RSS {result['peak_rss_mb']:.1f} MiB", flush=True)
    summary = {}
    for backend in backends:
        selected = [r for r in results if r["backend"] == backend]
        values = [r["median_ms"] for r in selected]
        summary[backend] = {"median_ms": statistics.median(values),
                            "p90_ms": float(np.percentile(values, 90)),
                            "peak_rss_mb": max(r["peak_rss_mb"] for r in selected)}
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                           capture_output=True, text=True).stdout.strip()
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), cpu)
    report = {"dataset": args.dataset, "voxel_size": args.voxel_size,
              "workspace_center": args.workspace_center, "workspace_extent": args.workspace_extent,
              "dataset_loading_ms": dataset_loading_ms,
              "repeat": args.repeat, "warmup_builds": 1, "map_cache": False,
              "python": platform.python_version(), "numpy": np.__version__, "cpu": cpu,
              "source_commit": commit, "source_dirty": bool(dirty),
              "thread_environment": {name: os.environ.get(name) for name in
                                     ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
              "summary": summary, "cases": results}
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
