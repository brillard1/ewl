#!/usr/bin/env python3
"""
Universal experiment runner for LLM fine-tuning with EWL.

Usage:
    python scripts/run_experiment.py --config configs/llm/alpaca_llama32_1b_t05.yaml --condition lora_sft
    python scripts/run_experiment.py --config configs/llm/alpaca_llama32_1b_t05.yaml --condition ewl
"""

import argparse
import os
import sys
import yaml
import torch
import wandb
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.data import prepare_dataset, create_dataloaders
from src.llm.models.setup import (
    setup_model_for_training,
    get_lora_config,
    print_model_info,
)
from src.ewl.ema_weighter import EMAWeighter
from src.llm.training.train_vanilla import train_vanilla
from src.llm.training.train_ewl import train_ewl
from src.llm.training.evaluate import run_full_evaluation, evaluate_mmlu


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_optimizer_scheduler(model, config, num_training_steps):
    """Create optimizer and learning rate scheduler"""
    import math
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR

    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
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


def run_base_eval(model, tokenizer, val_loader, raw_val_dataset, device, config):
    """Run evaluation on base model (no training)"""
    print("\n" + "=" * 70)
    print("RUNNING BASE MODEL EVALUATION (NO TRAINING)")
    print("=" * 70)

    results = run_full_evaluation(
        model,
        tokenizer,
        val_loader,
        raw_val_dataset,
        device,
        rouge_num_samples=config["evaluation"].get("rouge_num_samples", 200),
        rouge_max_new_tokens=config["evaluation"].get("rouge_max_new_tokens", 256),
    )

    wandb.summary["final/val_loss"] = results["val_loss"]
    wandb.summary["final/rouge_l"] = results["rouge_l"]

    return results


def run_lora_sft(
    model, tokenizer, train_loader, val_loader, raw_val_dataset, device, config
):
    """Run standard LoRA SFT training"""
    print("\n" + "=" * 70)
    print("RUNNING STANDARD LORA SFT")
    print("=" * 70)

    num_training_steps = (
        len(train_loader)
        // config["training"]["gradient_accumulation_steps"]
        * config["training"]["num_epochs"]
    )

    optimizer, scheduler = setup_optimizer_scheduler(model, config, num_training_steps)

    save_dir = (
        os.path.join(config["output"]["base_dir"], "lora_sft")
        if config["output"]["save_checkpoints"]
        else None
    )
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    history = train_vanilla(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config["training"]["num_epochs"],
        device=device,
        grad_accum_steps=config["training"]["gradient_accumulation_steps"],
        save_dir=save_dir,
    )

    print("\nRunning final evaluation...")
    results = run_full_evaluation(
        model,
        tokenizer,
        val_loader,
        raw_val_dataset,
        device,
        rouge_num_samples=config["evaluation"].get("rouge_num_samples", 200),
        rouge_max_new_tokens=config["evaluation"].get("rouge_max_new_tokens", 256),
    )

    wandb.summary["final/val_loss"] = results["val_loss"]
    wandb.summary["final/rouge_l"] = results["rouge_l"]

    return results


def run_ewl(
    model,
    tokenizer,
    train_loader,
    val_loader,
    raw_val_dataset,
    train_dataset_size,
    device,
    config,
):
    """Run EWL training"""
    print("\n" + "=" * 70)
    print("RUNNING EWL TRAINING")
    print("=" * 70)

    ema_weighter = EMAWeighter(
        num_samples=train_dataset_size,
        alpha=config["ewl"]["alpha"],
        temperature=config["ewl"]["temperature"],
        warmup_steps=config["ewl"]["warmup_steps"],
        use_lora_proxy=config["ewl"].get("use_lora_proxy", True),
    )

    ewl_logger = None

    num_training_steps = (
        len(train_loader)
        // config["training"]["gradient_accumulation_steps"]
        * config["training"]["num_epochs"]
    )

    optimizer, scheduler = setup_optimizer_scheduler(model, config, num_training_steps)

    save_dir = (
        os.path.join(config["output"]["base_dir"], "ewl")
        if config["output"]["save_checkpoints"]
        else None
    )
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    history = train_ewl(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        ema_weighter=ema_weighter,
        num_epochs=config["training"]["num_epochs"],
        device=device,
        grad_accum_steps=config["training"]["gradient_accumulation_steps"],
        save_dir=save_dir,
        ewl_logger=ewl_logger,
    )

    print("\nRunning final evaluation...")
    results = run_full_evaluation(
        model,
        tokenizer,
        val_loader,
        raw_val_dataset,
        device,
        rouge_num_samples=config["evaluation"].get("rouge_num_samples", 200),
        rouge_max_new_tokens=config["evaluation"].get("rouge_max_new_tokens", 256),
    )

    wandb.summary["final/val_loss"] = results["val_loss"]
    wandb.summary["final/rouge_l"] = results["rouge_l"]

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run EWL experiments with any dataset/model"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (e.g., configs/llm/alpaca_llama32_1b_t05.yaml)",
    )
    parser.add_argument(
        "--condition",
        type=str,
        required=True,
        choices=["base", "lora_sft", "ewl"],
        help="Experimental condition",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Override output directory"
    )
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    # Ablation flags
    parser.add_argument(
        "--rank", type=int, default=None,
        help="Override LoRA rank (lora.r). Used for rank-sweep ablation (Prediction 3).",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Override EWL temperature. Used for temperature-sweep ablation.",
    )
    parser.add_argument(
        "--no_proxy", action="store_true",
        help="Disable LoRA grad proxy (ablation: relative progress only).",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    if args.output_dir:
        config["output"]["base_dir"] = args.output_dir
    if args.rank is not None:
        config["lora"]["r"] = args.rank
    if args.temperature is not None:
        config["ewl"]["temperature"] = args.temperature
    if args.no_proxy:
        config["ewl"]["use_lora_proxy"] = False

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    # Initialize wandb
    if not args.no_wandb:
        exp_name = (
            config.get("experiment_name")
            or f"{config['dataset']['name']}_{args.condition}"
        )
        run_name = f"{exp_name}_{args.condition}"

        wandb.init(
            project=config["wandb"]["project"],
            entity=config["wandb"].get("entity"),
            name=run_name,
            tags=config["wandb"].get("tags", []) + [args.condition],
            notes=config.get("description", ""),
            config={
                "condition": args.condition,
                "dataset": config["dataset"]["name"],
                **config["dataset"],
                **config["model"],
                **config["lora"],
                **config["training"],
                **config["ewl"],
            },
        )

    # Set seed
    torch.manual_seed(config["dataset"]["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["dataset"]["seed"])

    # Load model
    apply_lora = args.condition != "base"
    lora_config = get_lora_config(**config["lora"]) if apply_lora else None

    model, tokenizer = setup_model_for_training(
        model_id=config["model"]["model_id"],
        use_qlora=config["model"]["use_qlora"],
        apply_lora_adapters=apply_lora,
        lora_config=lora_config,
    )

    print_model_info(model, config["model"]["model_id"])

    # Prepare data (works with any dataset!)
    print(f"\nPreparing {config['dataset']['name']} dataset...")
    train_dataset, val_dataset, raw_val_dataset = prepare_dataset(
        dataset_name=config["dataset"]["name"],
        tokenizer=tokenizer,
        max_len=config["dataset"]["max_seq_len"],
        test_size=config["dataset"]["val_split"],
        seed=config["dataset"]["seed"],
        mask_prompt=config["dataset"].get("mask_prompt", False),
    )

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        batch_size=config["training"]["micro_batch_size"],
    )

    # Run experiment based on condition
    if args.condition == "base":
        results = run_base_eval(
            model, tokenizer, val_loader, raw_val_dataset, device, config
        )
    elif args.condition == "lora_sft":
        results = run_lora_sft(
            model, tokenizer, train_loader, val_loader, raw_val_dataset, device, config
        )
    elif args.condition == "ewl":
        results = run_ewl(
            model,
            tokenizer,
            train_loader,
            val_loader,
            raw_val_dataset,
            len(train_dataset),
            device,
            config,
        )

    # Print MMLU evaluation command if needed
    if args.condition != "base" and config["evaluation"].get("run_mmlu", False):
        adapter_path = os.path.join(
            config["output"]["base_dir"],
            args.condition,
            f"epoch_{config['training']['num_epochs']}",
        )
        evaluate_mmlu(
            config["model"]["model_id"],
            adapter_path,
            f"results/mmlu_{config['dataset']['name']}_{args.condition}",
        )

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Model: {config['model']['model_id']}")
    print(f"Condition: {args.condition}")
    print(f"Val Loss: {results['val_loss']:.4f}")
    print(f"ROUGE-L: {results['rouge_l']:.4f}")
    print("=" * 70)

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
