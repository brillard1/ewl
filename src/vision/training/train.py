"""
Training loops for vision classification tasks
"""

import time

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

try:
    from sklearn.metrics import f1_score as _sk_f1
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


def compute_lora_grad_param_cosine_sim(model):
    """Cosine similarity between LoRA gradient vector and LoRA parameter vector."""
    grad_vecs, param_vecs = [], []
    for name, param in model.named_parameters():
        if "lora" in name.lower() and param.grad is not None:
            grad_vecs.append(param.grad.detach().view(-1))
            param_vecs.append(param.data.detach().view(-1))
    if not grad_vecs:
        return 0.0
    g = torch.cat(grad_vecs)
    p = torch.cat(param_vecs)
    denom = (g.norm() * p.norm()).clamp(min=1e-8)
    return (g @ p / denom).item()


@torch.no_grad()
def evaluate_vision_metrics(model, dataloader, device):
    """
    Evaluate accuracy, weighted F1, and average cross-entropy loss.

    Returns:
        dict with keys: accuracy, weighted_f1, avg_loss
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        images = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(images)
        loss = F.cross_entropy(outputs, labels, reduction="sum")
        total_loss += loss.item()

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    n = len(all_labels)
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / n if n > 0 else 0.0
    avg_loss = total_loss / n if n > 0 else 0.0

    if _SKLEARN_AVAILABLE:
        weighted_f1 = _sk_f1(all_labels, all_preds, average="weighted", zero_division=0)
    else:
        weighted_f1 = float("nan")

    return {"accuracy": accuracy, "weighted_f1": weighted_f1, "avg_loss": avg_loss}


@torch.no_grad()
def evaluate_vision_accuracy(model, dataloader, device):
    """Thin wrapper kept for backward compatibility — returns accuracy only."""
    return evaluate_vision_metrics(model, dataloader, device)["accuracy"]


def train_vision_vanilla(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    config,
    wandb_run=None,
):
    """
    Standard LoRA fine-tuning for vision tasks (uniform weights)

    Args:
        model: LoRA-adapted vision model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device
        config: Configuration dictionary
        wandb_run: Optional wandb run for logging
        early_stopping_patience: Stop after this many epochs without val accuracy
            improvement.  ``None`` disables early stopping.

    Returns:
        history: Training history
    """
    num_epochs = config["training"]["num_epochs"]
    log_interval = config["training"].get("log_interval", 10)

    history = []
    global_step = 0
    peak_vram_gb = 0.0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    print("\n" + "=" * 70)
    print("TRAINING: Standard LoRA Fine-tuning (Vision)")
    print("=" * 70)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        all_train_preds, all_train_labels = [], []
        step_times = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            t0 = time.perf_counter()

            images = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(images)
            per_sample_loss = F.cross_entropy(outputs, labels, reduction="none")

            # Uniform weighting
            loss = per_sample_loss.mean()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if config["training"].get("max_grad_norm", 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["training"]["max_grad_norm"]
                )

            optimizer.step()
            scheduler.step()

            step_times.append((time.perf_counter() - t0) * 1000)  # ms

            # Track metrics
            epoch_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)
            epoch_correct += (predictions == labels).sum().item()
            epoch_total += labels.size(0)
            all_train_preds.extend(predictions.cpu().tolist())
            all_train_labels.extend(labels.cpu().tolist())

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{epoch_correct/epoch_total:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

            global_step += 1

        # Epoch metrics
        train_loss = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total
        train_f1 = _sk_f1(all_train_labels, all_train_preds, average="weighted", zero_division=0) if _SKLEARN_AVAILABLE else float("nan")

        # System metrics
        mean_step_ms = float(np.mean(step_times)) if step_times else 0.0
        batch_size = config["training"].get("micro_batch_size", 64)
        throughput = batch_size / (mean_step_ms / 1000.0) if mean_step_ms > 0 else 0.0
        if torch.cuda.is_available():
            peak_vram_gb = torch.cuda.max_memory_allocated(device) / 1e9

        # Evaluate on validation set
        val_metrics = evaluate_vision_metrics(model, val_loader, device)
        val_acc  = val_metrics["accuracy"]
        val_f1   = val_metrics["weighted_f1"]
        val_loss = val_metrics["avg_loss"]

        print(
            f"[Epoch {epoch+1:>3}] "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
            f"train_f1={train_f1:.4f}  val_f1={val_f1:.4f}  "
            f"step={mean_step_ms:.1f}ms  vram={peak_vram_gb:.2f}GB"
        )

        # Log to wandb
        if wandb_run:
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "train/weighted_f1": train_f1,
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                    "val/weighted_f1": val_f1,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "system/peak_vram_gb": peak_vram_gb,
                    "system/step_time_ms": mean_step_ms,
                    "system/throughput_samples_per_sec": throughput,
                }
            )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_weighted_loss": None,
                "train_acc": train_acc,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "weight_entropy": None,
                "ema_mean": None,
                "ema_std":  None,
                "ema_min":  None,
                "ema_max":  None,
                "peak_vram_gb": peak_vram_gb,
                "step_time_ms": mean_step_ms,
                "throughput_samples_per_sec": throughput,
            }
        )

    if wandb_run:
        wandb_run.summary["system/peak_vram_gb"] = peak_vram_gb
        wandb_run.summary["system/mean_step_time_ms"] = mean_step_ms
        wandb_run.summary["system/throughput_samples_per_sec"] = throughput

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    return history


def train_vision_ewl(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    config,
    ema_weighter,
    wandb_run=None,
    condition="ewl",
    ewl_logger=None,
    sample_stats_dir=None,
):
    """
    EWL training for vision tasks (learning-progress-based weighting)

    Args:
        model: LoRA-adapted vision model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device
        config: Configuration dictionary
        ema_weighter: EMAWeighter instance
        wandb_run: Optional wandb run for logging

    Returns:
        history: Training history
    """
    num_epochs = config["training"]["num_epochs"]
    log_interval = config["training"].get("log_interval", 10)

    history = []
    global_step = 0
    peak_vram_gb = 0.0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    # EWL CPU state size (loss_ema + seen tensors, lives entirely on CPU)
    ewl_cpu_state_kb = (
        ema_weighter.loss_ema.nbytes + ema_weighter.seen.nbytes
    ) / 1024.0

    print("\n" + "=" * 70)
    print("TRAINING: EWL (Vision - Learning-Progress-Based Weighting)")
    print(f"EWL CPU state: {ewl_cpu_state_kb:.1f} KB  (loss_ema + seen, not on GPU)")
    print("=" * 70)

    for epoch in range(num_epochs):
        model.train()

        epoch_loss = 0.0
        epoch_weighted_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        epoch_grad_proxy_sum = 0.0
        epoch_grad_proxy_count = 0
        step_times = []

        # Per-sample accumulator for diagnostic .npz files
        epoch_sample_stats = []

        # EWL diagnostics
        epoch_diagnostics = {
            "mean_weight": 0.0,
            "max_weight": 0.0,
            "min_weight": 1.0,
            "weight_entropy": 0.0,
            "mean_signal": 0.0,
            "cosine_sim": 0.0,
        }
        diag_count = 0
        all_weights_epoch = []  # collect for histogram at early/late epochs
        is_early = (epoch + 1) <= 5
        is_late = (epoch + 1) >= (num_epochs - 4)
        all_train_preds, all_train_labels = [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            t0 = time.perf_counter()

            images = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            sample_ids = batch["sample_id"].to(device)

            # Forward pass
            outputs = model(images)
            per_sample_loss = F.cross_entropy(outputs, labels, reduction="none")

            # Compute EMA-based weights
            weights = ema_weighter.compute_weights(sample_ids, per_sample_loss)

            # Weighted loss
            loss = (weights * per_sample_loss).sum()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Cosine similarity between LoRA gradients and LoRA params (after backward)
            lora_cosine_sim = compute_lora_grad_param_cosine_sim(model)

            # Update LoRA gradient proxy (lagged gradients for next step)
            ema_weighter.update_grad_proxy(model)
            epoch_grad_proxy_sum += ema_weighter.grad_proxy
            epoch_grad_proxy_count += 1

            # Per-sample logging (for Figure 1 & 2)
            if ewl_logger is not None:
                ewl_logger.log_step(global_step, epoch + 1, ema_weighter)

            # Accumulate per-sample stats for diagnostic npz (skip warmup batches)
            if sample_stats_dir is not None:
                b = ema_weighter._last_batch
                if b is not None and b["signal"] is not None:
                    epoch_sample_stats.append({k: v.copy() for k, v in b.items()})

            # Gradient clipping
            if config["training"].get("max_grad_norm", 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["training"]["max_grad_norm"]
                )

            optimizer.step()
            scheduler.step()

            step_times.append((time.perf_counter() - t0) * 1000)  # ms

            # Track metrics
            epoch_loss += per_sample_loss.mean().item() * labels.size(0)
            epoch_weighted_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)
            epoch_correct += (predictions == labels).sum().item()
            epoch_total += labels.size(0)
            all_train_preds.extend(predictions.cpu().tolist())
            all_train_labels.extend(labels.cpu().tolist())

            # Accumulate diagnostics
            if ema_weighter.global_step > ema_weighter.warmup_steps:
                epoch_diagnostics["mean_weight"] += weights.mean().item()
                epoch_diagnostics["max_weight"] = max(
                    epoch_diagnostics["max_weight"], weights.max().item()
                )
                epoch_diagnostics["min_weight"] = min(
                    epoch_diagnostics["min_weight"], weights.min().item()
                )
                # Weight entropy
                weight_entropy = (
                    -(weights * torch.log(weights.clamp(min=1e-10))).sum().item()
                )
                epoch_diagnostics["weight_entropy"] += weight_entropy
                # Get EMA stats for signal
                ema_stats = ema_weighter.get_statistics()
                epoch_diagnostics["mean_signal"] += ema_stats.get(
                    "grad_proxy", 0.0
                )
                epoch_diagnostics["cosine_sim"] += lora_cosine_sim
                # Collect weights for histogram at early/late epochs
                if is_early or is_late:
                    all_weights_epoch.append(weights.detach().cpu())
                diag_count += 1

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{per_sample_loss.mean().item():.4f}",
                    "w_loss": f"{loss.item():.4f}",
                    "acc": f"{epoch_correct/epoch_total:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

            global_step += 1

        # Epoch metrics
        train_loss = epoch_loss / epoch_total
        train_weighted_loss = epoch_weighted_loss / epoch_total
        train_acc = epoch_correct / epoch_total
        train_f1 = _sk_f1(all_train_labels, all_train_preds, average="weighted", zero_division=0) if _SKLEARN_AVAILABLE else float("nan")

        # Save per-sample stats for this epoch
        if sample_stats_dir is not None and epoch_sample_stats:
            Path(sample_stats_dir).mkdir(parents=True, exist_ok=True)
            all_ids  = np.concatenate([b["ids"]      for b in epoch_sample_stats])
            all_loss = np.concatenate([b["loss"]     for b in epoch_sample_stats])
            all_pema = np.concatenate([b["prev_ema"] for b in epoch_sample_stats])
            all_cema = np.concatenate([b["loss_ema"] for b in epoch_sample_stats])
            all_sig  = np.concatenate([b["signal"]   for b in epoch_sample_stats])
            all_wt   = np.concatenate([b["weight"]   for b in epoch_sample_stats])
            velocity = (all_pema - all_loss) / (np.abs(all_pema) + 1e-8)
            np.savez_compressed(
                Path(sample_stats_dir) / f"sample_stats_ep{epoch+1:02d}.npz",
                ids=all_ids, loss=all_loss, prev_ema=all_pema, curr_ema=all_cema,
                signal=all_sig, weight=all_wt, velocity=velocity,
            )

        # Average diagnostics
        if diag_count > 0:
            for key in ["mean_weight", "weight_entropy", "mean_signal", "cosine_sim"]:
                epoch_diagnostics[key] /= diag_count

        mean_grad_proxy = (
            epoch_grad_proxy_sum / epoch_grad_proxy_count
            if epoch_grad_proxy_count > 0 else 0.0
        )

        # EMA state snapshot at epoch end
        ema_stats = ema_weighter.get_statistics()

        # System metrics
        mean_step_ms = float(np.mean(step_times)) if step_times else 0.0
        batch_size = config["training"].get("micro_batch_size", 64)
        throughput = batch_size / (mean_step_ms / 1000.0) if mean_step_ms > 0 else 0.0
        if torch.cuda.is_available():
            peak_vram_gb = torch.cuda.max_memory_allocated(device) / 1e9

        # Evaluate on validation set
        val_metrics = evaluate_vision_metrics(model, val_loader, device)
        val_acc  = val_metrics["accuracy"]
        val_f1   = val_metrics["weighted_f1"]
        val_loss = val_metrics["avg_loss"]

        entropy = epoch_diagnostics["weight_entropy"] if diag_count > 0 else float("nan")
        print(
            f"[Epoch {epoch+1:>3}] "
            f"train_loss={train_loss:.4f}  train_w_loss={train_weighted_loss:.4f}  val_loss={val_loss:.4f}  "
            f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
            f"train_f1={train_f1:.4f}  val_f1={val_f1:.4f}  "
            f"entropy={entropy:.3f}  step={mean_step_ms:.1f}ms  vram={peak_vram_gb:.2f}GB"
        )

        # Log to wandb
        if wandb_run:
            log_dict = {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/weighted_loss": train_weighted_loss,
                "train/accuracy": train_acc,
                "train/weighted_f1": train_f1,
                "train/weighted_loss": train_weighted_loss,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "val/weighted_f1": val_f1,
                "learning_rate": scheduler.get_last_lr()[0],
                "ewl/grad_proxy": mean_grad_proxy,
            }
            # Add EWL diagnostics if available
            if diag_count > 0:
                log_dict.update(
                    {
                        "ewl/mean_weight": epoch_diagnostics["mean_weight"],
                        "ewl/max_weight": epoch_diagnostics["max_weight"],
                        "ewl/min_weight": epoch_diagnostics["min_weight"],
                        "ewl/weight_entropy": epoch_diagnostics["weight_entropy"],
                        "ewl/cosine_sim_grad_param": epoch_diagnostics["cosine_sim"],
                        "ema/mean": ema_stats["mean_ema"],
                        "ema/std":  ema_stats["std_ema"],
                        "ema/min":  ema_stats["min_ema"],
                        "ema/max":  ema_stats["max_ema"],
                        "ema/num_seen": ema_stats["num_seen"],
                    }
                )
                # Weight distribution histogram at early/late epochs
                if (is_early or is_late) and all_weights_epoch:
                    try:
                        import wandb as _wandb
                        all_w = torch.cat(all_weights_epoch).numpy()
                        phase = "early" if is_early else "late"
                        log_dict[f"ewl/weight_dist_{phase}_epoch{epoch+1}"] = (
                            _wandb.Histogram(all_w)
                        )
                    except Exception:
                        pass
            log_dict.update(
                {
                    "system/peak_vram_gb": peak_vram_gb,
                    "system/step_time_ms": mean_step_ms,
                    "system/throughput_samples_per_sec": throughput,
                    "system/ewl_cpu_state_kb": ewl_cpu_state_kb,
                }
            )
            wandb_run.log(log_dict)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_weighted_loss": train_weighted_loss,
                "train_acc": train_acc,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "weight_entropy": epoch_diagnostics["weight_entropy"] if diag_count > 0 else None,
                "ewl_grad_proxy": mean_grad_proxy,
                "ema_mean": ema_stats["mean_ema"],
                "ema_std":  ema_stats["std_ema"],
                "ema_min":  ema_stats["min_ema"],
                "ema_max":  ema_stats["max_ema"],
                "peak_vram_gb": peak_vram_gb,
                "step_time_ms": mean_step_ms,
                "throughput_samples_per_sec": throughput,
                "ewl_cpu_state_kb": ewl_cpu_state_kb,
            }
        )

    if wandb_run:
        wandb_run.summary["system/peak_vram_gb"] = peak_vram_gb
        wandb_run.summary["system/mean_step_time_ms"] = mean_step_ms
        wandb_run.summary["system/throughput_samples_per_sec"] = throughput
        wandb_run.summary["system/ewl_cpu_state_kb"] = ewl_cpu_state_kb

    if ewl_logger is not None:
        ewl_logger.finish()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    return history
