#!/usr/bin/env python3
"""
Ablation sweep: LoRA rank, EWL alpha, EWL temperature on FGVC-Aircraft.

Runs jobs in parallel across a user-specified set of GPUs, with a configurable
number of concurrent jobs per GPU.

Each run saves to its own directory:
    outputs/vision/aircraft_ablations/r{rank}_a{alpha}_t{temp}/
        run.log                   ← full stdout+stderr
        results_ewl_*.json        ← epoch-wise + final test metrics

Default values for non-swept params: rank=4, alpha=0.9, temperature=2.0

Usage:
    # All sweeps, GPUs 0–3, 1 job per GPU (4 parallel jobs)
    python scripts/sweep_aircraft_ablations.py --gpus 0 1 2 3

    # GPUs 0–7, 2 jobs per GPU (16 parallel jobs)
    python scripts/sweep_aircraft_ablations.py --gpus 0 1 2 3 4 5 6 7 --jobs_per_gpu 2

    # Only rank sweep, GPU 0 only (sequential)
    python scripts/sweep_aircraft_ablations.py --sweep rank --gpus 0

    # Smoke test: 2 epochs per run
    python scripts/sweep_aircraft_ablations.py --gpus 0 1 --epochs 2 --no_wandb
"""

import argparse
import queue
import subprocess
import sys
import concurrent.futures
from pathlib import Path

# ── Default hyperparameters ──────────────────────────────────────────────────
DEFAULT_RANK        = 4
DEFAULT_ALPHA       = 0.9
DEFAULT_TEMPERATURE = 1.0

# ── Sweep grids ──────────────────────────────────────────────────────────────
RANK_SWEEP        = [2, 4, 8, 16]
ALPHA_SWEEP       = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
TEMPERATURE_SWEEP = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
SEEDS             = [11, 22, 33]

CONFIG   = "configs/vision/aircraft_vit_base.yaml"
BASE_OUT = "outputs/vision/aircraft_ablations"


def run_baseline(rank, seed, gpu, no_wandb, extra_args):
    """Launch a lora_sft baseline run."""
    tag = f"baseline_r{rank}_s{seed}"
    out_dir = f"{BASE_OUT}/{tag}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    log_file = f"{out_dir}/run.log"

    cmd = [
        sys.executable, "scripts/run_vision_experiment.py",
        "--config",    CONFIG,
        "--condition", "lora_sft",
        "--lora_rank", str(rank),
        "--seed",      str(seed),
        "--output_dir", out_dir,
        "--gpu",       str(gpu),
    ]
    if no_wandb:
        cmd += ["--no_wandb"]
    cmd += extra_args

    print(f"[START] {tag}  gpu={gpu}  log={log_file}")

    with open(log_file, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        print(f"[FAIL]  {tag}  (exit {result.returncode}) — see {log_file}")
    else:
        print(f"[OK]    {tag}")

    return tag, result.returncode


def run(rank, alpha, temperature, seed, gpu, no_wandb, extra_args):
    """Launch a single experiment subprocess and capture its output to run.log."""
    tag = f"r{rank}_a{alpha}_t{temperature}_s{seed}"
    out_dir = f"{BASE_OUT}/{tag}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    log_file = f"{out_dir}/run.log"

    cmd = [
        sys.executable, "scripts/run_vision_experiment.py",
        "--config",      CONFIG,
        "--condition",   "ewl",
        "--lora_rank",   str(rank),
        "--alpha",       str(alpha),
        "--temperature", str(temperature),
        "--seed",        str(seed),
        "--output_dir",  out_dir,
        "--gpu",         str(gpu),
    ]
    if no_wandb:
        cmd += ["--no_wandb"]
    cmd += extra_args

    print(f"[START] {tag}  gpu={gpu}  log={log_file}")

    with open(log_file, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        print(f"[FAIL]  {tag}  (exit {result.returncode}) — see {log_file}")
    else:
        print(f"[OK]    {tag}")

    return tag, result.returncode


def build_run_list(sweeps, seeds):
    """Return two lists:
      ewl_runs:      (rank, alpha, temperature, seed)
      baseline_runs: (rank, seed)   — one SFT run per rank × seed
    """
    ewl_runs, baseline_runs = [], []
    ewl_seen, base_seen = set(), set()

    for s in seeds:
        if "rank" in sweeps:
            for r in RANK_SWEEP:
                key = (r, DEFAULT_ALPHA, DEFAULT_TEMPERATURE, s)
                if key not in ewl_seen:
                    ewl_seen.add(key); ewl_runs.append(key)
                bkey = (r, s)
                if bkey not in base_seen:
                    base_seen.add(bkey); baseline_runs.append(bkey)
        if "alpha" in sweeps:
            for a in ALPHA_SWEEP:
                key = (DEFAULT_RANK, a, DEFAULT_TEMPERATURE, s)
                if key not in ewl_seen:
                    ewl_seen.add(key); ewl_runs.append(key)
            # baseline at default rank for alpha sweep (if not already added)
            bkey = (DEFAULT_RANK, s)
            if bkey not in base_seen:
                base_seen.add(bkey); baseline_runs.append(bkey)
        if "temperature" in sweeps:
            for t in TEMPERATURE_SWEEP:
                key = (DEFAULT_RANK, DEFAULT_ALPHA, t, s)
                if key not in ewl_seen:
                    ewl_seen.add(key); ewl_runs.append(key)
            bkey = (DEFAULT_RANK, s)
            if bkey not in base_seen:
                base_seen.add(bkey); baseline_runs.append(bkey)

    return ewl_runs, baseline_runs


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--sweep", nargs="+",
        choices=["rank", "alpha", "temperature"],
        default=["rank", "alpha", "temperature"],
        help="Which parameter(s) to sweep (default: all three).",
    )
    parser.add_argument(
        "--gpus", nargs="+", type=int, required=True,
        help="GPU indices to use, e.g. --gpus 0 1 2 3",
    )
    parser.add_argument(
        "--jobs_per_gpu", type=int, default=1,
        help="Max concurrent jobs per GPU (default: 1).",
    )
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS,
        help=f"Seeds to run each config with (default: {SEEDS}).",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override num_epochs (e.g. --epochs 2 for a smoke test).",
    )
    args, extra = parser.parse_known_args()

    if args.epochs:
        extra += ["--epochs", str(args.epochs)]

    ewl_runs, baseline_runs = build_run_list(args.sweep, args.seeds)
    all_runs = [("ewl", r) for r in ewl_runs] + [("baseline", r) for r in baseline_runs]
    total_workers = len(args.gpus) * args.jobs_per_gpu

    # GPU slot pool: each GPU index appears jobs_per_gpu times
    gpu_pool = queue.Queue()
    for _ in range(args.jobs_per_gpu):
        for g in args.gpus:
            gpu_pool.put(g)

    print(f"\nSweep: {args.sweep}  |  Seeds: {args.seeds}")
    print(f"EWL runs: {len(ewl_runs)}  |  Baseline runs: {len(baseline_runs)}  |  Total: {len(all_runs)}")
    print(f"GPUs: {args.gpus}  |  jobs/GPU: {args.jobs_per_gpu}  |  parallel workers: {total_workers}")
    print(f"Output base: {BASE_OUT}/\n")

    def worker(job):
        kind, run_args = job
        gpu = gpu_pool.get()
        try:
            if kind == "baseline":
                rank, seed = run_args
                return run_baseline(rank, seed=seed,
                                    gpu=gpu, no_wandb=args.no_wandb, extra_args=extra)
            else:
                rank, alpha, temperature, seed = run_args
                return run(rank, alpha, temperature, seed=seed,
                           gpu=gpu, no_wandb=args.no_wandb, extra_args=extra)
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
    print(f"SWEEP COMPLETE  ({len(all_runs) - len(failed)}/{len(all_runs)} succeeded)")
    if failed:
        print(f"Failed runs: {failed}")
    print(f"Results under: {BASE_OUT}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
