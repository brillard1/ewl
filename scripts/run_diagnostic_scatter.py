#!/usr/bin/env python3
"""
Diagnostic run: save per-sample EMA/velocity/weight stats for scatter analysis.

Runs EWL with noise=20% (and optionally imbalance) on each dataset for 1 seed,
saving per-epoch .npz files to:
    outputs/vision/diagnostic/{dataset}/{condition}/sample_stats/

Usage:
    # Noise experiment (default)
    python scripts/run_diagnostic_scatter.py --gpus 0 1 2

    # Imbalance experiment
    python scripts/run_diagnostic_scatter.py --gpus 0 1 2 --mode imbalance

    # Single dataset
    python scripts/run_diagnostic_scatter.py --gpus 0 --datasets aircraft
"""

import argparse
import concurrent.futures
import queue
import subprocess
import sys
from pathlib import Path

DATASET_CONFIGS = {
    "aircraft":      "configs/vision/aircraft_vit_base.yaml",
    "cub200":        "configs/vision/cub200_vit_base.yaml",
    "stanford_dogs": "configs/vision/stanford_dogs_vit_base.yaml",
}

RANK        = 4
ALPHA       = 0.9
TEMPERATURE = 1.0
SEED        = 11
NOISE       = 0.20
IMBALANCE   = 0.10   # minority/majority ratio for imbalance mode

BASE_OUT = "outputs/vision/diagnostic"


def run_job(dataset, config, gpu, mode, no_wandb, extra_args):
    if mode == "noise":
        tag     = f"{dataset}/ewl_noise20pct"
        out_dir = str(Path(BASE_OUT) / dataset / "ewl_noise20pct")
        extra   = ["--noise_ratio", str(NOISE)]
    else:
        tag     = f"{dataset}/ewl_imbalance10pct"
        out_dir = str(Path(BASE_OUT) / dataset / "ewl_imbalance10pct")
        extra   = ["--imbalance_ratio", str(IMBALANCE)]

    log_file = f"{out_dir}/run.log"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "scripts/run_vision_experiment.py",
        "--config",     config,
        "--condition",  "ewl",
        "--lora_rank",  str(RANK),
        "--alpha",      str(ALPHA),
        "--temperature", str(TEMPERATURE),
        "--seed",       str(SEED),
        "--output_dir", out_dir,
        "--gpu",        str(gpu),
        "--save_sample_stats",
    ] + extra + extra_args

    if no_wandb:
        cmd += ["--no_wandb"]

    print(f"[START] {tag}  gpu={gpu}")
    with open(log_file, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    status = "OK" if result.returncode == 0 else f"FAIL (exit {result.returncode})"
    print(f"[{status}] {tag}" + (f"  — see {log_file}" if result.returncode != 0 else ""))
    return tag, result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+",
                        choices=list(DATASET_CONFIGS.keys()),
                        default=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--mode", choices=["noise", "imbalance"], default="noise",
                        help="noise: label noise=20%; imbalance: IF=10 long-tail (default: noise)")
    parser.add_argument("--no_wandb", action="store_true")
    args, extra = parser.parse_known_args()

    jobs = [(ds, DATASET_CONFIGS[ds]) for ds in args.datasets]
    gpu_pool = queue.Queue()
    for g in args.gpus:
        gpu_pool.put(g)

    print(f"\nMode: {args.mode}  |  Datasets: {args.datasets}  |  Seed: {SEED}")
    print(f"GPUs: {args.gpus}  |  Output: {BASE_OUT}/\n")

    def worker(job):
        ds, cfg = job
        gpu = gpu_pool.get()
        try:
            return run_job(ds, cfg, gpu, args.mode, args.no_wandb, extra)
        finally:
            gpu_pool.put(gpu)

    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = {pool.submit(worker, j): j for j in jobs}
        for fut in concurrent.futures.as_completed(futures):
            tag, code = fut.result()
            if code != 0:
                failed.append(tag)

    print("\n" + "=" * 60)
    print(f"DONE  ({len(jobs) - len(failed)}/{len(jobs)} succeeded)")
    if failed:
        print(f"Failed: {failed}")
    print(f"Stats under: {BASE_OUT}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
