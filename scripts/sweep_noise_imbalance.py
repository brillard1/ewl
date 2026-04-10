#!/usr/bin/env python3
"""
Ablation sweep: label noise and class imbalance on FGVC-Aircraft.

Noise and imbalance are studied independently (never combined).
Fixed hyperparams: rank=4, alpha=0.9, temperature=2.0
Seeds: 11, 22, 33

Each run saves to its own directory:
    outputs/vision/aircraft_noise_imbalance/<tag>/
        run.log               ← full stdout+stderr
        results_*.json        ← epoch-wise + final test metrics

Usage:
    # All sweeps, GPUs 0–3
    python scripts/sweep_noise_imbalance.py --gpus 0 1 2 3

    # Noise sweep only
    python scripts/sweep_noise_imbalance.py --sweep noise --gpus 0 1

    # Imbalance sweep only
    python scripts/sweep_noise_imbalance.py --sweep imbalance --gpus 0 1

    # Smoke test (2 epochs)
    python scripts/sweep_noise_imbalance.py --gpus 0 1 --epochs 2 --no_wandb
"""

import argparse
import queue
import subprocess
import sys
import concurrent.futures
from pathlib import Path

# ── Fixed hyperparams ────────────────────────────────────────────────────────
RANK        = 4
ALPHA       = 0.9
TEMPERATURE = 1.0

# ── Sweep grids ──────────────────────────────────────────────────────────────
# Label noise: fraction of training labels randomly flipped
NOISE_SWEEP      = [0, 0.1, 0.2, 0.3, 0.4]

# Imbalance ratio: n_min / n_max  (IF = 1/ratio, so 0.1→IF=10, 0.05→IF=20)
IMBALANCE_SWEEP  = [1, 0.5, 0.2, 0.1, 0.05]

SEEDS   = [11]
CONFIG  = "configs/vision/aircraft_vit_base.yaml"
BASE_OUT = "outputs/vision/aircraft_noise_imbalance"


def _base_cmd(condition, tag_args, out_dir, gpu, no_wandb, extra_args):
    """Build the base command list shared by all run types."""
    cmd = [
        sys.executable, "scripts/run_vision_experiment.py",
        "--config",      CONFIG,
        "--condition",   condition,
        "--lora_rank",   str(RANK),
        "--output_dir",  out_dir,
        "--gpu",         str(gpu),
    ]
    cmd += tag_args
    if no_wandb:
        cmd += ["--no_wandb"]
    cmd += extra_args
    return cmd


def run_ewl(sweep_type, value, seed, gpu, no_wandb, extra_args):
    """Launch one EWL run for either noise or imbalance."""
    if sweep_type == "noise":
        tag     = f"ewl_noise{int(value*100)}pct_s{seed}"
        arg_key = "--noise_ratio"
    else:
        tag     = f"ewl_imb{int(value*100)}_s{seed}"
        arg_key = "--imbalance_ratio"

    out_dir  = f"{BASE_OUT}/{tag}"
    log_file = f"{out_dir}/run.log"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    cmd = _base_cmd(
        "ewl",
        [
            "--alpha",       str(ALPHA),
            "--temperature", str(TEMPERATURE),
            "--seed",        str(seed),
            arg_key,         str(value),
        ],
        out_dir, gpu, no_wandb, extra_args,
    )

    print(f"[START] {tag}  gpu={gpu}  log={log_file}")
    with open(log_file, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    status = "OK" if result.returncode == 0 else f"FAIL (exit {result.returncode})"
    print(f"[{status}] {tag}" + (f" — see {log_file}" if result.returncode != 0 else ""))
    return tag, result.returncode


def run_baseline(sweep_type, value, seed, gpu, no_wandb, extra_args):
    """Launch one SFT baseline at the same noise / imbalance level."""
    if sweep_type == "noise":
        tag     = f"sft_noise{int(value*100)}pct_s{seed}"
        arg_key = "--noise_ratio"
    else:
        tag     = f"sft_imb{int(value*100)}_s{seed}"
        arg_key = "--imbalance_ratio"

    out_dir  = f"{BASE_OUT}/{tag}"
    log_file = f"{out_dir}/run.log"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    cmd = _base_cmd(
        "lora_sft",
        ["--seed", str(seed), arg_key, str(value)],
        out_dir, gpu, no_wandb, extra_args,
    )

    print(f"[START] {tag}  gpu={gpu}  log={log_file}")
    with open(log_file, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    status = "OK" if result.returncode == 0 else f"FAIL (exit {result.returncode})"
    print(f"[{status}] {tag}" + (f" — see {log_file}" if result.returncode != 0 else ""))
    return tag, result.returncode


def build_run_list(sweeps, seeds):
    """Return list of (kind, sweep_type, value, seed) tuples.

    For each (sweep_type, value, seed) pair we schedule:
      - one EWL run
      - one SFT baseline at the same perturbation level
    Noise and imbalance runs are never mixed.
    """
    runs = []
    for s in seeds:
        if "noise" in sweeps:
            for v in NOISE_SWEEP:
                runs.append(("ewl",      "noise",     v, s))
                runs.append(("baseline", "noise",     v, s))
        if "imbalance" in sweeps:
            for v in IMBALANCE_SWEEP:
                runs.append(("ewl",      "imbalance", v, s))
                runs.append(("baseline", "imbalance", v, s))
    return runs


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--sweep", nargs="+",
        choices=["noise", "imbalance"],
        default=["noise", "imbalance"],
        help="Which ablation(s) to run (default: both).",
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

    all_runs = build_run_list(args.sweep, args.seeds)
    total_workers = len(args.gpus) * args.jobs_per_gpu

    # GPU slot pool
    gpu_pool = queue.Queue()
    for _ in range(args.jobs_per_gpu):
        for g in args.gpus:
            gpu_pool.put(g)

    noise_runs = sum(1 for r in all_runs if r[1] == "noise")
    imb_runs   = sum(1 for r in all_runs if r[1] == "imbalance")
    print(f"\nSweep: {args.sweep}  |  Seeds: {args.seeds}")
    print(f"Noise runs: {noise_runs}  |  Imbalance runs: {imb_runs}  |  Total: {len(all_runs)}")
    print(f"GPUs: {args.gpus}  |  jobs/GPU: {args.jobs_per_gpu}  |  parallel workers: {total_workers}")
    print(f"Fixed params: rank={RANK}, alpha={ALPHA}, temperature={TEMPERATURE}")
    print(f"Output base: {BASE_OUT}/\n")

    def worker(job):
        kind, sweep_type, value, seed = job
        gpu = gpu_pool.get()
        try:
            if kind == "ewl":
                return run_ewl(sweep_type, value, seed, gpu, args.no_wandb, extra)
            else:
                return run_baseline(sweep_type, value, seed, gpu, args.no_wandb, extra)
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
