#!/usr/bin/env python3
"""
Cross-dataset sweep: run SFT baseline, EWL, and EWL (no grad proxy) on all
four vision datasets for 3 seeds.

Conditions per dataset × seed:
  baseline          — LoRA SFT, no EWL
  ewl               — EWL with grad proxy
  ewl_no_proxy      — EWL without grad proxy (use_lora_proxy=False)

Fixed hyperparameters: rank=4, alpha=0.9, temperature=1.0

Each run saves to:
  outputs/vision/all_datasets/{dataset}/{condition}_s{seed}/
    run.log
    results_*.json

Usage:
  # All datasets, GPUs 0–3
  python scripts/sweep_all_datasets.py --gpus 0 1 2 3

  # Single dataset, e.g. food101
  python scripts/sweep_all_datasets.py --gpus 0 1 --datasets food101

  # 2 jobs per GPU
  python scripts/sweep_all_datasets.py --gpus 0 1 2 3 --jobs_per_gpu 2

  # Smoke test: 2 epochs
  python scripts/sweep_all_datasets.py --gpus 0 1 --epochs 2 --no_wandb
"""

import argparse
import concurrent.futures
import queue
import subprocess
import sys
from pathlib import Path

# ── Fixed hyperparameters ────────────────────────────────────────────────────
RANK        = 4
ALPHA       = 0.9
TEMPERATURE = 1.0
SEEDS       = [11, 22, 33]

# ── Dataset → config file mapping ────────────────────────────────────────────
DATASET_CONFIGS = {
    "aircraft":      "configs/vision/aircraft_vit_base.yaml",
    "cub200":        "configs/vision/cub200_vit_base.yaml",
    "stanford_dogs": "configs/vision/stanford_dogs_vit_base.yaml",
    # "food101":       "configs/vision/food101_vit_base.yaml",
}

BASE_OUT = "outputs/vision/all_datasets"


def _run(cmd, tag, log_file):
    """Execute cmd, stream output to log_file, print status."""
    print(f"[START] {tag}  log={log_file}")
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
        "--config",    config,
        "--condition", "lora_sft",
        "--lora_rank", str(RANK),
        "--seed",      str(seed),
        "--output_dir", str(out_dir),
        "--gpu",       str(gpu),
    ]
    if no_wandb:
        cmd += ["--no_wandb"]
    cmd += extra_args
    return _run(cmd, tag, str(out_dir / "run.log"))


def run_ewl(dataset, config, seed, gpu, no_wandb, extra_args, no_proxy=False):
    cond    = "ewl_no_proxy" if no_proxy else "ewl"
    tag     = f"{dataset}/{cond}_s{seed}"
    out_dir = Path(BASE_OUT) / dataset / f"{cond}_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "scripts/run_vision_experiment.py",
        "--config",      config,
        "--condition",   "ewl",
        "--lora_rank",   str(RANK),
        "--alpha",       str(ALPHA),
        "--temperature", str(TEMPERATURE),
        "--seed",        str(seed),
        "--output_dir",  str(out_dir),
        "--gpu",         str(gpu),
    ]
    if no_proxy:
        cmd += ["--no_proxy"]
    if no_wandb:
        cmd += ["--no_wandb"]
    cmd += extra_args
    return _run(cmd, tag, str(out_dir / "run.log"))


def build_jobs(datasets, seeds):
    """Return list of (kind, dataset, config, seed, no_proxy) tuples."""
    jobs = []
    for ds in datasets:
        cfg = DATASET_CONFIGS[ds]
        for s in seeds:
            jobs.append(("baseline",      ds, cfg, s, False))
            jobs.append(("ewl",           ds, cfg, s, False))
            jobs.append(("ewl_no_proxy",  ds, cfg, s, True))
    return jobs


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=__doc__)
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

    jobs = build_jobs(args.datasets, args.seeds)
    total_workers = len(args.gpus) * args.jobs_per_gpu

    gpu_pool = queue.Queue()
    for _ in range(args.jobs_per_gpu):
        for g in args.gpus:
            gpu_pool.put(g)

    n_baseline = sum(1 for j in jobs if j[0] == "baseline")
    n_ewl      = sum(1 for j in jobs if j[0] == "ewl")
    n_noproxy  = sum(1 for j in jobs if j[0] == "ewl_no_proxy")

    print(f"\nDatasets: {args.datasets}  |  Seeds: {args.seeds}")
    print(f"baseline: {n_baseline}  |  ewl: {n_ewl}  |  ewl_no_proxy: {n_noproxy}  |  total: {len(jobs)}")
    print(f"GPUs: {args.gpus}  |  jobs/GPU: {args.jobs_per_gpu}  |  workers: {total_workers}")
    print(f"Fixed: rank={RANK}, alpha={ALPHA}, temperature={TEMPERATURE}")
    print(f"Output: {BASE_OUT}/\n")

    def worker(job):
        kind, dataset, config, seed, no_proxy = job
        gpu = gpu_pool.get()
        try:
            if kind == "baseline":
                return run_baseline(dataset, config, seed, gpu, args.no_wandb, extra)
            else:
                return run_ewl(dataset, config, seed, gpu, args.no_wandb, extra,
                               no_proxy=no_proxy)
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
        print(f"Failed: {failed}")
    print(f"Results under: {BASE_OUT}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
