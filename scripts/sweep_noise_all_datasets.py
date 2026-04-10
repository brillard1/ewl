#!/usr/bin/env python3
"""
Noise ablation sweep across all vision datasets.

Runs EWL and SFT baseline at each noise level for each dataset.
Noise and imbalance are studied independently (never combined).
Fixed hyperparams: rank=4, alpha=0.9, temperature=1.0

Each run saves to:
    outputs/vision/noise_all_datasets/{dataset}/{condition}_noise{pct}pct_s{seed}/
        run.log
        results_*.json

Usage:
    # All datasets, GPUs 0–3
    python scripts/sweep_noise_all_datasets.py --gpus 0 1 2 3

    # Single dataset
    python scripts/sweep_noise_all_datasets.py --datasets cub200 --gpus 0 1

    # 2 jobs per GPU
    python scripts/sweep_noise_all_datasets.py --gpus 0 1 2 3 --jobs_per_gpu 2

    # Smoke test (2 epochs)
    python scripts/sweep_noise_all_datasets.py --gpus 0 1 --epochs 2 --no_wandb
"""

import argparse
import concurrent.futures
import queue
import subprocess
import sys
from pathlib import Path

# ── Fixed hyperparams ────────────────────────────────────────────────────────
RANK        = 4
ALPHA       = 0.9
TEMPERATURE = 1.0
SEEDS       = [11, 22, 33]

# ── Noise sweep grid ─────────────────────────────────────────────────────────
NOISE_SWEEP = [0.0, 0.1, 0.2, 0.3, 0.4]   # fraction of labels flipped

# ── Dataset → config file mapping ────────────────────────────────────────────
DATASET_CONFIGS = {
    "aircraft":      "configs/vision/aircraft_vit_base.yaml",
    "cub200":        "configs/vision/cub200_vit_base.yaml",
    "stanford_dogs": "configs/vision/stanford_dogs_vit_base.yaml",
    # "food101":       "configs/vision/food101_vit_base.yaml",
}

BASE_OUT = "outputs/vision/noise_all_datasets"


def _base_cmd(dataset, condition, out_dir, gpu, no_wandb, extra_args):
    cmd = [
        sys.executable, "scripts/run_vision_experiment.py",
        "--config",     DATASET_CONFIGS[dataset],
        "--condition",  condition,
        "--lora_rank",  str(RANK),
        "--output_dir", out_dir,
        "--gpu",        str(gpu),
    ]
    if no_wandb:
        cmd += ["--no_wandb"]
    cmd += extra_args
    return cmd


def run_ewl(dataset, noise, seed, gpu, no_wandb, extra_args):
    pct     = int(noise * 100)
    tag     = f"{dataset}/ewl_noise{pct}pct_s{seed}"
    out_dir = str(Path(BASE_OUT) / dataset / f"ewl_noise{pct}pct_s{seed}")
    log_file = f"{out_dir}/run.log"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    cmd = _base_cmd(dataset, "ewl", out_dir, gpu, no_wandb, extra_args) + [
        "--alpha",       str(ALPHA),
        "--temperature", str(TEMPERATURE),
        "--seed",        str(seed),
        "--noise_ratio", str(noise),
    ]

    print(f"[START] {tag}  gpu={gpu}")
    with open(log_file, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    status = "OK" if result.returncode == 0 else f"FAIL (exit {result.returncode})"
    print(f"[{status}] {tag}" + (f" — see {log_file}" if result.returncode != 0 else ""))
    return tag, result.returncode


def run_baseline(dataset, noise, seed, gpu, no_wandb, extra_args):
    pct     = int(noise * 100)
    tag     = f"{dataset}/sft_noise{pct}pct_s{seed}"
    out_dir = str(Path(BASE_OUT) / dataset / f"sft_noise{pct}pct_s{seed}")
    log_file = f"{out_dir}/run.log"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    cmd = _base_cmd(dataset, "lora_sft", out_dir, gpu, no_wandb, extra_args) + [
        "--seed",        str(seed),
        "--noise_ratio", str(noise),
    ]

    print(f"[START] {tag}  gpu={gpu}")
    with open(log_file, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    status = "OK" if result.returncode == 0 else f"FAIL (exit {result.returncode})"
    print(f"[{status}] {tag}" + (f" — see {log_file}" if result.returncode != 0 else ""))
    return tag, result.returncode


def build_run_list(datasets, seeds):
    """Return list of (kind, dataset, noise, seed) tuples."""
    runs = []
    for ds in datasets:
        for s in seeds:
            for v in NOISE_SWEEP:
                runs.append(("ewl",      ds, v, s))
                runs.append(("baseline", ds, v, s))
    return runs


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--datasets", nargs="+",
        choices=list(DATASET_CONFIGS.keys()),
        default=list(DATASET_CONFIGS.keys()),
        help="Datasets to run (default: all four).",
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
        help=f"Seeds (default: {SEEDS}).",
    )
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override num_epochs (e.g. --epochs 2 for smoke test).",
    )
    args, extra = parser.parse_known_args()

    if args.epochs:
        extra += ["--epochs", str(args.epochs)]

    all_runs = build_run_list(args.datasets, args.seeds)
    total_workers = len(args.gpus) * args.jobs_per_gpu

    gpu_pool = queue.Queue()
    for _ in range(args.jobs_per_gpu):
        for g in args.gpus:
            gpu_pool.put(g)

    n_ewl  = sum(1 for r in all_runs if r[0] == "ewl")
    n_sft  = sum(1 for r in all_runs if r[0] == "baseline")
    print(f"\nDatasets:  {args.datasets}")
    print(f"Noise levels: {NOISE_SWEEP}  |  Seeds: {args.seeds}")
    print(f"EWL runs: {n_ewl}  |  SFT runs: {n_sft}  |  Total: {len(all_runs)}")
    print(f"GPUs: {args.gpus}  |  jobs/GPU: {args.jobs_per_gpu}  |  workers: {total_workers}")
    print(f"Fixed: rank={RANK}, alpha={ALPHA}, temperature={TEMPERATURE}")
    print(f"Output: {BASE_OUT}/\n")

    def worker(job):
        kind, dataset, noise, seed = job
        gpu = gpu_pool.get()
        try:
            if kind == "ewl":
                return run_ewl(dataset, noise, seed, gpu, args.no_wandb, extra)
            else:
                return run_baseline(dataset, noise, seed, gpu, args.no_wandb, extra)
        finally:
            gpu_pool.put(gpu)

    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=total_workers) as pool:
        futures = {pool.submit(worker, j): j for j in all_runs}
        for fut in concurrent.futures.as_completed(futures):
            tag, code = fut.result()
            if code != 0:
                failed.append(tag)

    print("\n" + "=" * 70)
    print(f"DONE  ({len(all_runs) - len(failed)}/{len(all_runs)} succeeded)")
    if failed:
        print(f"Failed: {failed}")
    print(f"Results under: {BASE_OUT}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
