#!/usr/bin/env python3
"""
Baseline sweep: run the LoRA SFT baseline on ALL datasets for 3 seeds.

Covers both the original datasets (CUB-200, Aircraft, Stanford Dogs, Food-101)
and the newly added datasets (Flowers-102, Oxford-IIIT Pets, Caltech-101).

All datasets use the same uniform 80/10/10 stratified split, so results are
directly comparable across datasets.

Runs per configuration:
  7 datasets × 3 seeds = 21 runs

Each run saves to:
  outputs/vision/baseline_sweep/{dataset}/baseline_s{seed}/
    run.log
    results_lora_sft_*.json

Usage:
  # All datasets, GPUs 0–3
  python scripts/sweep_baseline_all_datasets.py --gpus 0 1 2 3

  # Specific datasets only
  python scripts/sweep_baseline_all_datasets.py --gpus 0 1 --datasets cub200 flowers102

  # 2 concurrent jobs per GPU
  python scripts/sweep_baseline_all_datasets.py --gpus 0 1 2 3 --jobs_per_gpu 2

  # Smoke test: 2 epochs, no wandb
  python scripts/sweep_baseline_all_datasets.py --gpus 0 --epochs 2 --no_wandb

  # Custom seeds
  python scripts/sweep_baseline_all_datasets.py --gpus 0 1 --seeds 11 22 33
"""

import argparse
import concurrent.futures
import queue
import subprocess
import sys
from pathlib import Path

# ── Fixed hyperparameters ─────────────────────────────────────────────────────
RANK  = 4
SEEDS = [11, 22, 33]

# ── Dataset → config file mapping (all 7 datasets) ───────────────────────────
DATASET_CONFIGS = {
    "cub200":        "configs/vision/cub200_vit_base.yaml",
    "aircraft":      "configs/vision/aircraft_vit_base.yaml",
    "stanford_dogs": "configs/vision/stanford_dogs_vit_base.yaml",
    "food101":       "configs/vision/food101_vit_base.yaml",
    "flowers102":    "configs/vision/flowers102_vit_base.yaml",
    "oxford_pets":   "configs/vision/oxford_pets_vit_base.yaml",
    "caltech101":    "configs/vision/caltech101_vit_base.yaml",
}

BASE_OUT = "outputs/vision/baseline_sweep"


def _run(cmd, tag, log_file):
    """Execute cmd, stream stdout+stderr to log_file, print status."""
    print(f"[START] {tag}  →  {log_file}")
    with open(log_file, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    status = "OK" if result.returncode == 0 else f"FAIL (exit {result.returncode})"
    print(f"[{status}] {tag}")
    return tag, result.returncode


def run_baseline(dataset, config, seed, gpu, no_wandb, extra_args):
    tag     = f"{dataset}/baseline_s{seed}"
    out_dir = Path(BASE_OUT) / dataset / f"baseline_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "scripts/run_vision_experiment.py",
        "--config",     config,
        "--condition",  "lora_sft",
        "--lora_rank",  str(RANK),
        "--seed",       str(seed),
        "--output_dir", str(out_dir),
        "--gpu",        str(gpu),
    ]
    if no_wandb:
        cmd += ["--no_wandb"]
    cmd += extra_args
    return _run(cmd, tag, str(out_dir / "run.log"))


def build_jobs(datasets, seeds):
    return [(ds, DATASET_CONFIGS[ds], s) for ds in datasets for s in seeds]


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--datasets", nargs="+",
        choices=list(DATASET_CONFIGS.keys()),
        default=list(DATASET_CONFIGS.keys()),
        help="Datasets to sweep (default: all 7).",
    )
    parser.add_argument(
        "--gpus", nargs="+", type=int, required=True,
        help="GPU indices, e.g. --gpus 0 1 2 3",
    )
    parser.add_argument(
        "--jobs_per_gpu", type=int, default=1,
        help="Concurrent jobs per GPU (default: 1).",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS,
        help=f"Random seeds (default: {SEEDS}).",
    )
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging.")
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override num_epochs (e.g. --epochs 2 for a smoke test).",
    )
    args, extra = parser.parse_known_args()

    if args.epochs:
        extra += ["--epochs", str(args.epochs)]

    jobs = build_jobs(args.datasets, args.seeds)
    total_workers = len(args.gpus) * args.jobs_per_gpu

    gpu_pool = queue.Queue()
    for _ in range(args.jobs_per_gpu):
        for g in args.gpus:
            gpu_pool.put(g)

    print(f"\nDatasets : {args.datasets}")
    print(f"Seeds    : {args.seeds}")
    print(f"Total    : {len(jobs)} runs  ({len(args.datasets)} datasets × {len(args.seeds)} seeds)")
    print(f"GPUs     : {args.gpus}  |  jobs/GPU: {args.jobs_per_gpu}  |  workers: {total_workers}")
    print(f"LoRA rank: {RANK}")
    print(f"Output   : {BASE_OUT}/\n")

    def worker(job):
        dataset, config, seed = job
        gpu = gpu_pool.get()
        try:
            return run_baseline(dataset, config, seed, gpu, args.no_wandb, extra)
        finally:
            gpu_pool.put(gpu)

    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=total_workers) as pool:
        futures = {pool.submit(worker, j): j for j in jobs}
        for fut in concurrent.futures.as_completed(futures):
            tag, code = fut.result()
            if code != 0:
                failed.append(tag)

    print("\n" + "=" * 70)
    print(f"DONE  ({len(jobs) - len(failed)}/{len(jobs)} succeeded)")
    if failed:
        print(f"Failed runs:")
        for f in failed:
            print(f"  {f}")
    print(f"Results under: {BASE_OUT}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
