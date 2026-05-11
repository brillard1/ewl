#!/usr/bin/env python3
"""
Vision experiment runner for CUB-200 fine-tuning with EWL.

Usage:
    python scripts/run_vision_experiment.py --config configs/vision/cub200_vit_base.yaml --condition lora_sft
    python scripts/run_vision_experiment.py --config configs/vision/cub200_vit_base.yaml --condition ewl --temperature 0.5
"""

import argparse
import os
import sys
import yaml
import torch
import wandb
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vision.data.pipeline import prepare_vision_dataset, create_vision_dataloaders
from src.vision.models.setup import setup_vit_model
from src.vision.training.train import (
    train_vision_vanilla,
    train_vision_ewl,
    evaluate_vision_accuracy,
    evaluate_vision_metrics,
)
from src.ewl.ema_weighter import EMAWeighter


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_optimizer_scheduler(model, config, num_training_steps):
    import math
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR

    base_lr = config["training"]["learning_rate"]
    head_lr = config["training"].get("head_learning_rate", base_lr * 5)

    head_params  = [p for n, p in model.named_parameters() if "head" in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if "head" not in n and p.requires_grad]

    optimizer = AdamW(
        [
            {"params": other_params, "lr": base_lr},
            {"params": head_params,  "lr": head_lr},
        ],
        weight_decay=config["training"]["weight_decay"],
    )

    warmup_steps = config["training"]["warmup_steps"]

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, num_training_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def run_lora_sft(
    model,
    train_loader,
    val_loader,
    device,
    config,
    wandb_run,
):
    print("\n" + "=" * 70)
    print("RUNNING STANDARD LORA SFT (VISION)")
    print("=" * 70)

    num_training_steps = len(train_loader) * config["training"]["num_epochs"]
    optimizer, scheduler = setup_optimizer_scheduler(model, config, num_training_steps)

    save_dir = (
        os.path.join(config["output"]["base_dir"], "lora_sft")
        if config["output"]["save_checkpoints"]
        else None
    )
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    history = train_vision_vanilla(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        wandb_run=wandb_run,
    )

    final_acc = evaluate_vision_accuracy(model, val_loader, device)

    if wandb_run:
        wandb_run.summary["final/val_accuracy"] = final_acc

    return {"val_accuracy": final_acc, "history": history}


def run_ewl(
    model,
    train_loader,
    val_loader,
    train_dataset_size,
    device,
    config,
    wandb_run,
    condition="ewl",
    sample_stats_dir=None,
):
    use_lora_proxy = config["ewl"].get("use_lora_proxy", True)

    print("\n" + "=" * 70)
    print(
        f"RUNNING EWL (VISION) - condition={condition} "
        f"temperature={config['ewl']['temperature']} alpha={config['ewl']['alpha']}"
    )
    print("=" * 70)

    ema_weighter = EMAWeighter(
        num_samples=train_dataset_size,
        alpha=config["ewl"]["alpha"],
        temperature=config["ewl"]["temperature"],
        warmup_steps=config["ewl"]["warmup_steps"],
        use_lora_proxy=use_lora_proxy,
    )

    ewl_logger = None

    num_training_steps = len(train_loader) * config["training"]["num_epochs"]
    optimizer, scheduler = setup_optimizer_scheduler(model, config, num_training_steps)

    cond_tag = condition if condition != "ewl" else "ewl_combined"
    save_dir = (
        os.path.join(
            config["output"]["base_dir"],
            f"{cond_tag}_t{config['ewl']['temperature']}_a{config['ewl']['alpha']}_r{config['lora']['r']}",
        )
        if config["output"]["save_checkpoints"]
        else None
    )
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    history = train_vision_ewl(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        ema_weighter=ema_weighter,
        wandb_run=wandb_run,
        condition=condition,
        ewl_logger=ewl_logger,
        sample_stats_dir=sample_stats_dir,
    )

    final_acc = evaluate_vision_accuracy(model, val_loader, device)

    if wandb_run:
        wandb_run.summary["final/val_accuracy"] = final_acc

    return {"val_accuracy": final_acc, "history": history}


def main():
    parser = argparse.ArgumentParser(description="Run vision EWL experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument(
        "--condition",
        type=str,
        required=True,
        choices=["lora_sft", "ewl"],
        help="lora_sft: SFT baseline; ewl: EWL with progress signal and grad proxy",
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="Override EWL temperature"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Override output directory"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override number of training epochs"
    )
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument(
        "--lora_rank", type=int, default=None, help="Override LoRA rank (lora.r)"
    )
    parser.add_argument(
        "--alpha", type=float, default=None, help="Override EWL alpha (ewl.alpha)"
    )
    parser.add_argument(
        "--label_noise",
        type=float,
        default=None,
        help="Fraction of training labels to randomly flip (0.0–1.0)",
    )
    parser.add_argument(
        "--rank", type=int, default=None, help="Override LoRA rank (Prediction 3 sweep)"
    )
    parser.add_argument(
        "--noise_ratio", type=float, default=None, help="Label noise fraction (Prediction 1)"
    )
    parser.add_argument(
        "--no_proxy", action="store_true", help="Disable LoRA grad proxy (ablation)"
    )
    parser.add_argument(
        "--gpu", type=int, default=None, help="GPU index to use (e.g. --gpu 1). Defaults to cuda:0."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Override dataset/training seed"
    )
    parser.add_argument(
        "--imbalance_ratio",
        type=float,
        default=None,
        help="Long-tail imbalance ratio: minority/majority class size (0–1). "
             "Class i retains imbalance_ratio^(i/C) * n_max samples. (Prediction 2 ablation)",
    )
    parser.add_argument(
        "--save_sample_stats",
        action="store_true",
        help="Save per-sample EMA/velocity/weight stats to .npz files each epoch (diagnostic).",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    if args.temperature is not None:
        config["ewl"]["temperature"] = args.temperature
    if args.epochs is not None:
        config["training"]["num_epochs"] = args.epochs
    if args.output_dir:
        config["output"]["base_dir"] = args.output_dir
    if args.rank is not None:
        config["lora"]["r"] = args.rank
    if args.no_proxy:
        config["ewl"]["use_lora_proxy"] = False

    if args.lora_rank is not None:
        config["lora"]["r"] = args.lora_rank
        config["lora"]["lora_alpha"] = args.lora_rank * 2  # keep alpha = 2×r convention

    if args.alpha is not None:
        config["ewl"]["alpha"] = args.alpha

    if args.seed is not None:
        config["dataset"]["seed"] = args.seed

    if args.label_noise is not None:
        config["dataset"]["label_noise"] = args.label_noise
    if args.noise_ratio is not None:
        config["dataset"]["label_noise"] = args.noise_ratio
    if args.imbalance_ratio is not None:
        config["dataset"]["imbalance_ratio"] = args.imbalance_ratio

    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Set seed
    torch.manual_seed(config["dataset"]["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["dataset"]["seed"])

    # Initialize wandb
    wandb_run = None
    if not args.no_wandb:
        is_ewl_cond = args.condition == "ewl"
        temp_str = f"_t{config['ewl']['temperature']}" if is_ewl_cond else ""
        alpha_str = f"_a{config['ewl']['alpha']}" if is_ewl_cond else ""
        rank_str = f"_r{config['lora']['r']}" if is_ewl_cond else ""
        noise_str = (
            f"_noise{int(config['dataset'].get('label_noise', 0)*100)}pct"
            if config["dataset"].get("label_noise", 0) > 0
            else ""
        )
        imb_ratio = config["dataset"].get("imbalance_ratio", 0.0)
        imb_str = f"_imb{int(imb_ratio*100)}" if imb_ratio and imb_ratio > 0 else ""
        cond_str = "baseline" if args.condition == "lora_sft" else args.condition
        run_name = (
            f"{config.get('experiment_name', 'vision')}_{cond_str}"
            f"{temp_str}{alpha_str}{rank_str}{noise_str}{imb_str}"
        )

        wandb_run = wandb.init(
            project=config["wandb"]["project"],
            entity=config["wandb"].get("entity"),
            name=run_name,
            tags=config["wandb"].get("tags", []) + [args.condition],
            config={
                "condition": args.condition,
                **config["dataset"],
                **config["model"],
                **config["lora"],
                **config["training"],
                **config["ewl"],
            },
        )

    # Prepare data
    print(f"\nPreparing {config['dataset']['name']} dataset...")
    train_dataset, val_dataset, test_dataset = prepare_vision_dataset(config)

    train_loader, val_loader, test_loader = create_vision_dataloaders(
        train_dataset, val_dataset, config, test_dataset=test_dataset
    )
    print(
        f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    # Setup model
    model = setup_vit_model(config)
    model = model.to(device)

    # Sample-stats dir (EWL only, when --save_sample_stats is set)
    sample_stats_dir = None
    if args.save_sample_stats and args.condition == "ewl":
        import numpy as np
        from src.vision.data.pipeline import NoisyLabelWrapper
        sample_stats_dir = os.path.join(config["output"]["base_dir"], "sample_stats")
        os.makedirs(sample_stats_dir, exist_ok=True)

        # Save noisy mask if training set has label noise applied
        ds = train_dataset
        if isinstance(ds, NoisyLabelWrapper):
            np.save(os.path.join(sample_stats_dir, "noisy_mask.npy"), ds.noisy_mask)
            print(f"[Diag] Saved noisy_mask.npy  ({ds.noisy_mask.sum()} noisy / {len(ds.noisy_mask)} total)")
        else:
            # No noise — save all-False mask so notebook code is uniform
            n = len(train_dataset)
            np.save(os.path.join(sample_stats_dir, "noisy_mask.npy"), np.zeros(n, dtype=bool))
            print(f"[Diag] No NoisyLabelWrapper found — saved all-False noisy_mask ({n} samples)")

    # Run experiment
    if args.condition == "lora_sft":
        results = run_lora_sft(
            model,
            train_loader,
            val_loader,
            device,
            config,
            wandb_run,
        )
    elif args.condition == "ewl":
        results = run_ewl(
            model,
            train_loader,
            val_loader,
            len(train_dataset),
            device,
            config,
            wandb_run,
            sample_stats_dir=sample_stats_dir,
        )

    # Final evaluation on held-out test set (run once at the end)
    test_metrics = evaluate_vision_metrics(model, test_loader, device)
    test_acc = test_metrics["accuracy"]
    test_f1  = test_metrics["weighted_f1"]

    if wandb_run:
        wandb_run.summary["final/test_accuracy"]    = test_acc
        wandb_run.summary["final/test_weighted_f1"] = test_f1

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)
    print(f"Condition:     {args.condition}")
    print(f"Val  Accuracy: {results['val_accuracy']:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}   Test W-F1: {test_f1:.4f}   Test Loss: {test_metrics['avg_loss']:.4f}")
    print("=" * 70)

    # Save results JSON
    import json, datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = config["output"]["base_dir"]
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, f"results_{args.condition}_{timestamp}.json")
    json_payload = {
        "condition": args.condition,
        "dataset": config["dataset"]["name"],
        "timestamp": timestamp,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "test_loss": test_metrics["avg_loss"],
        "history": results.get("history", []),
    }
    with open(results_path, "w") as f:
        json.dump(json_payload, f, indent=2)
    print(f"Results saved → {results_path}")

    if wandb_run:
        wandb.finish()


if __name__ == "__main__":
    main()
