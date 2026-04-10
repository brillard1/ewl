# EWL: Example Weighting via Learning Progress for LoRA Fine Tuning

## Overview

This repository implements **EWL (Example Weighting via Learning Progress)**, a training method that improves LoRA fine-tuning of vision transformers by dynamically reweighting training samples based on how fast each sample's loss is decreasing.

Standard LoRA SFT treats all samples equally — but not all samples are equally useful. Easy samples are already mastered and contribute little; noisy samples have wrong labels and actively hurt training; only hard-but-learnable samples drive meaningful adaptation. EWL identifies this *active learning zone* automatically, without any extra backward passes, clean validation set, or task-specific tuning.

It does this by maintaining a per-sample exponential moving average (EMA) of the loss and computing a *velocity* signal — how much each sample is currently improving relative to its history. Samples with high persistent loss and positive velocity (hard-clean) are upweighted; samples with high loss but no improvement (noisy) are suppressed. A lagged LoRA gradient proxy scales the signal globally, acting as an adaptive temperature that sharpens weights during rapid early adaptation and flattens them near convergence.

The method adds negligible overhead over standard LoRA SFT and requires no hyperparameter tuning per dataset.

## Installation

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```


## Usage

```bash
# SFT baseline
python scripts/run_vision_experiment.py \
    --config configs/vision/aircraft_vit_base.yaml \
    --condition lora_sft --no_wandb

# EWL
python scripts/run_vision_experiment.py \
    --config configs/vision/aircraft_vit_base.yaml \
    --condition ewl --no_wandb

# EWL with label noise
python scripts/run_vision_experiment.py \
    --config configs/vision/aircraft_vit_base.yaml \
    --condition ewl --noise_ratio 0.2 --seed 11 --no_wandb

# EWL with long-tail imbalance (IF=10)
python scripts/run_vision_experiment.py \
    --config configs/vision/aircraft_vit_base.yaml \
    --condition ewl --imbalance_ratio 0.1 --seed 11 --no_wandb
```


## Project Structure

```
ewl-lora-sft/
├── src/
│   ├── ewl/
│   │   ├── ema_weighter.py           # Core EWL algorithm
│   │   └── logger.py                 # Per-step diagnostic logging
│   ├── vision/
│   │   ├── data/
│   │   │   ├── aircraft_dataset.py   # FGVC-Aircraft (official 3-way split)
│   │   │   ├── cub_dataset.py        # CUB-200-2011 (80/20 stratified)
│   │   │   ├── dogs_dataset.py       # Stanford Dogs (80/20 stratified)
│   │   │   ├── food101_dataset.py    # Food-101 (80/20 stratified)
│   │   │   ├── noise.py              # Label noise injection
│   │   │   └── pipeline.py           # Dataset & DataLoader factory
│   │   ├── models/setup.py           # ViT-Base + LoRA initialisation
│   │   └── training/train.py         # SFT and EWL training loops
│   └── llm/                          # LLM experiments (Alpaca, Dolly, OASST1)
│
├── configs/vision/                   # Per-dataset YAML configs
├── scripts/                          # Experiment runners and sweep scripts
└── notebooks/                        # Analysis and figures
```


## Datasets

| Dataset | Classes | Train | Val | Test | Split |
|---------|--------:|------:|----:|-----:|-------|
| FGVC-Aircraft | 100 | 6,667 | 3,333 | 3,333 | Official |
| CUB-200-2011 | 200 | 4,795 | 1,199 | 5,794 | 80/20 stratified |
| Stanford Dogs | 120 | 9,600 | 2,400 | 8,580 | 80/20 stratified |
| Food-101 | 101 | 60,600 | 15,150 | 25,250 | 80/20 stratified |

---

## Configuration

Key hyperparameters (`configs/vision/*.yaml`):

```yaml
lora:
  r: 4
  lora_alpha: 8
  target_modules: [qkv, proj, fc1, fc2]

ewl:
  alpha: 0.9          # EMA smoothing
  temperature: 1.0    # Softmax temperature
  warmup_steps: 100   # Uniform weights before EWL activates
  use_lora_proxy: true
```

---

## License

MIT
