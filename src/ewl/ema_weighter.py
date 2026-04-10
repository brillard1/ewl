"""
EWL: Example Weighting via Learning Progress

Algorithm (paper method, use_lora_proxy=True):
  1. Track per-sample loss EMA:   μ_i ← α·μ_i + (1-α)·ℓ_i
  2. Velocity × magnitude signal:
       velocity_i = (μ_i_prev - ℓ_i) / (μ_i_prev + ε)   [relative progress, +ve if improving]
       s_i        = velocity_i × μ_i                      [amplify by current loss level]
  3. Modulate by lagged LoRA grad proxy (global scalar from previous step):
                                  s_i ← s_i · g_prev
  4. Softmax weights:             w_i = softmax(s_i / τ)

  g_prev acts as an adaptive temperature: large grad norms (early training, rapid
  adaptation) amplify signals → sharper weights; decaying grad norms (convergence)
  shrink signals → flatter weights. This is a free automatic curriculum with no
  extra hyperparameter.

  NOTE: z-normalisation is intentionally omitted. z-norm would cancel the
  grad_proxy modulation since for any positive scalar g,
  (g·x − mean(g·x)) / std(g·x) = (x − mean(x)) / std(x).

Ablation (use_lora_proxy=False):
  Steps 2, 4 only — velocity × magnitude signal, no grad modulation.

Usage:
    weighter = EMAWeighter(num_samples=N, alpha=0.9, temperature=1.5)

    # Each training step:
    weights = weighter.compute_weights(sample_ids, per_sample_losses)
    loss = (weights * per_sample_losses).sum()
    loss.backward()
    weighter.update_grad_proxy(model)   # call after backward, before optimizer.step
    optimizer.step()
"""

import torch
import torch.nn.functional as F


class EMAWeighter:

    def __init__(
        self,
        num_samples,
        alpha=0.9,
        temperature=1.5,
        warmup_steps=100,
        use_lora_proxy=True,
    ):
        """
        Args:
            num_samples:    Total number of training samples.
            alpha:          EMA decay for per-sample loss tracking.
            temperature:    Softmax temperature τ. Higher → more uniform weights.
                            Healthy range [1.5, 2.0]. τ→∞ recovers SFT baseline.
            warmup_steps:   Steps with uniform weights before EWL activates.
            use_lora_proxy: True  = paper method (relative progress × grad proxy).
                            False = ablation (absolute progress only).
        """
        self.loss_ema = torch.zeros(num_samples, dtype=torch.float32)
        self.seen = torch.zeros(num_samples, dtype=torch.bool)

        self.alpha = alpha
        self.temperature = temperature
        self.warmup_steps = warmup_steps
        self.use_lora_proxy = use_lora_proxy
        self.global_step = 0

        # Lagged global LoRA gradient proxy; updated after each backward pass.
        # Initialised to 1.0 (neutral) so the first weighted step is well-defined.
        self.grad_proxy = 1.0

        # Set by compute_weights each step; read by EWLLogger.
        self._last_batch = None

    # ------------------------------------------------------------------

    def compute_weights(self, sample_ids, per_sample_losses):
        """
        Compute per-sample importance weights.

        Args:
            sample_ids:         (B,) integer tensor of dataset indices.
            per_sample_losses:  (B,) tensor of per-sample CE loss (detached).

        Returns:
            weights: (B,) tensor on the same device as per_sample_losses.
        """
        device = per_sample_losses.device
        losses = per_sample_losses.detach().cpu().float()
        ids = sample_ids.cpu()

        # Seed EMA on first visit so prev_ema is meaningful next step
        new = ~self.seen[ids]
        if new.any():
            self.loss_ema[ids[new]] = losses[new]
            self.seen[ids[new]] = True

        prev_ema = self.loss_ema[ids].clone()

        # EMA update
        self.loss_ema[ids] = self.alpha * self.loss_ema[ids] + (1.0 - self.alpha) * losses

        self.global_step += 1

        if self.global_step <= self.warmup_steps:
            B = len(sample_ids)
            uniform = torch.ones(B, device=device) / B
            self._last_batch = {
                "ids": ids.numpy(), "loss": losses.numpy(),
                "prev_ema": prev_ema.numpy(),
                "loss_ema": self.loss_ema[ids].numpy(),
                "signal": None, "weight": uniform.cpu().numpy(),
            }
            return uniform

        # Velocity × magnitude signal.
        # velocity: relative per-step progress (+ve = loss improving)
        # curr_ema: smoothed loss level (high = persistently difficult)
        # Product: hard-clean (velocity>0, moderate ema) → positive signal
        #          noisy/flat (velocity≈0, high ema)     → near-zero signal
        curr_ema = self.loss_ema[ids]
        velocity = (prev_ema - losses) / (prev_ema.abs() + 1e-8)
        signal = velocity * curr_ema

        if self.use_lora_proxy:
            signal = signal * self.grad_proxy

        raw_signal = signal.clone()

        weights = F.softmax(signal / self.temperature, dim=0)

        self._last_batch = {
            "ids": ids.numpy(), "loss": losses.numpy(),
            "prev_ema": prev_ema.numpy(),
            "loss_ema": self.loss_ema[ids].numpy(),
            "signal": raw_signal.numpy(),
            "weight": weights.numpy(),
        }

        return weights.to(device)

    def update_grad_proxy(self, model):
        """
        Compute and cache the global LoRA gradient proxy.

        Call AFTER loss.backward() and BEFORE optimizer.step().
        The cached value is used in the NEXT call to compute_weights (lagged by 1 step).

        Computes: g = mean over all LoRA layers of (‖∇B‖_F + ‖∇A‖_F)
        """
        norms = []
        for name, param in model.named_parameters():
            if "lora" in name.lower() and param.grad is not None:
                norms.append(param.grad.norm().item())

        self.grad_proxy = sum(norms) / len(norms) if norms else 1.0

    # ------------------------------------------------------------------

    def get_statistics(self):
        """Return diagnostic stats for logging."""
        if not self.seen.any():
            return {"mean_ema": 0.0, "std_ema": 0.0, "min_ema": 0.0, "max_ema": 0.0,
                    "num_seen": 0, "grad_proxy": self.grad_proxy}
        seen = self.loss_ema[self.seen]
        return {
            "mean_ema": seen.mean().item(),
            "std_ema":  seen.std().item(),
            "min_ema":  seen.min().item(),
            "max_ema":  seen.max().item(),
            "num_seen": self.seen.sum().item(),
            "grad_proxy": self.grad_proxy,
        }

    def weight_entropy_ratio(self, weights):
        """H(w) / log(B) — 1.0 = uniform, 0.0 = degenerate. Healthy: 0.4–0.7."""
        B = len(weights)
        if B < 2:
            return 1.0
        entropy = -(weights * (weights + 1e-12).log()).sum().item()
        return entropy / torch.tensor(float(B)).log().item()

    def save_state(self, path):
        torch.save({
            "loss_ema": self.loss_ema,
            "seen": self.seen,
            "global_step": self.global_step,
            "alpha": self.alpha,
            "temperature": self.temperature,
            "warmup_steps": self.warmup_steps,
            "use_lora_proxy": self.use_lora_proxy,
            "grad_proxy": self.grad_proxy,
        }, path)

    def load_state(self, path):
        state = torch.load(path, map_location="cpu")
        # "loss_ema" is the current key; older checkpoints used "signal_ema"
        self.loss_ema = state.get("loss_ema", state.get("signal_ema"))
        if self.loss_ema is None:
            raise KeyError(f"Checkpoint at {path} contains neither 'loss_ema' nor 'signal_ema'")
        self.seen = state["seen"]
        self.global_step = state["global_step"]
        self.alpha = state["alpha"]
        self.temperature = state["temperature"]
        self.warmup_steps = state["warmup_steps"]
        self.use_lora_proxy = state["use_lora_proxy"]
        self.grad_proxy = state.get("grad_proxy", 1.0)
