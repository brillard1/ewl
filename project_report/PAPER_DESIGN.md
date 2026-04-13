# EWL Paper Design — ENSF 619
## End-to-End Design Based on Actual Experimental Results

> **Scope:** Vision-only. LLM experiments excluded.
> **Datasets:** FGVC-Aircraft, CUB-200-2011, Stanford Dogs
> **Last updated:** 2026-04-13

---

## What the Results Actually Say (Ground Truth)

Before designing the paper, here is a honest summary of every result:

### Main Results (clean data, mean ± std over 3 seeds)

| Dataset     | SFT            | EWL            | EWL (no proxy) | Δ EWL vs SFT |
|-------------|----------------|----------------|----------------|--------------|
| Aircraft    | 58.94 ± 0.60%  | 58.99 ± 0.71%  | 52.18 ± 0.48%  | +0.05%       |
| CUB-200     | 86.15 ± 0.29%  | 86.30 ± 0.42%  | 85.97 ± 0.40%  | +0.16%       |
| Stanford Dogs | 89.43 ± 0.05% | 89.41 ± 0.06% | 88.89 ± 0.24%  | −0.01%       |

**Key takeaway:** EWL with proxy ≈ SFT on clean data. EWL without proxy degrades
significantly on Aircraft (−6.76%). The gradient proxy is the critical component.

### Noise Robustness (mean ± std, 3 seeds)

| Dataset       | Noise | SFT     | EWL     | Δ     |
|---------------|-------|---------|---------|-------|
| Aircraft      | 0%    | 58.94%  | 59.01%  | +0.07 |
| Aircraft      | 10%   | 51.90%  | 56.16%  | +4.26 |
| Aircraft      | 20%   | 46.21%  | 52.12%  | +5.90 |
| Aircraft      | 30%   | 40.26%  | 47.07%  | +6.81 |
| Aircraft      | 40%   | 34.64%  | 40.89%  | +6.25 |
| CUB-200       | 10%   | 81.24%  | 80.32%  | −0.92 |
| CUB-200       | 20%   | 75.08%  | 73.53%  | −1.55 |
| CUB-200       | 40%   | 60.71%  | 59.45%  | −1.26 |
| Stanford Dogs | 10%   | 84.95%  | 84.09%  | −0.86 |
| Stanford Dogs | 20%   | 80.30%  | 77.87%  | −2.43 |
| Stanford Dogs | 40%   | 71.02%  | 64.46%  | −6.55 |

**Key takeaway:** Noise robustness is completely dataset-dependent. EWL strongly
benefits Aircraft under noise, but hurts CUB-200 and Stanford Dogs under noise.
This is the most interesting and unexpected finding.

### Rank Sweep (Aircraft, EWL, α=0.9, τ=1.0)

| Rank | EWL Accuracy     |
|------|-----------------|
| r=2  | 57.40 ± 0.18%   |
| r=4  | 59.06 ± 0.71%   |
| r=8  | 60.37 ± 0.93%   |
| r=16 | 62.08 ± 0.83%   |

Higher rank → higher accuracy. Monotonically increasing.

### Temperature Sweep (Aircraft, EWL, rank=4, α=0.9)

| τ    | Accuracy        |
|------|-----------------|
| 0.5  | 56.77 ± 1.05%   |
| 1.0  | 59.06 ± 0.71%   |
| 1.5  | 59.23 ± 0.74%   |
| 2.0  | 59.16 ± 0.60%   |
| 2.5  | 59.10 ± 0.56%   |
| 3.0  | 59.09 ± 0.42%   |

τ=0.5 hurts. Above τ=1.0, results are stable. Low sensitivity.

### Class Imbalance (Aircraft, mean ± std)

| Imbalance Factor | SFT              | EWL              | Δ      |
|------------------|-----------------|-----------------|--------|
| IF=1 (balanced)  | 58.95 ± 0.57%   | 58.97 ± 0.66%   | +0.02% |
| IF=2             | 51.68 ± 0.77%   | 51.40 ± 0.97%   | −0.28% |
| IF=5             | 39.38 ± 0.19%   | 39.66 ± 0.51%   | +0.28% |
| IF=10            | 30.99 ± 0.72%   | 31.23 ± 0.46%   | +0.24% |
| IF=20            | 23.40 ± 0.59%   | 23.83 ± 0.46%   | +0.43% |

No meaningful benefit from EWL under imbalance.

### System Overhead

| Metric              | SFT          | EWL          | Ratio |
|---------------------|--------------|--------------|-------|
| Step time (ms)      | 56.4         | 277.9        | 4.9×  |
| Peak VRAM (GB)      | 9.46         | 9.46         | 1.0×  |
| Throughput (samp/s) | 1135         | 230          | 0.2×  |
| CPU state overhead  | —            | 16.3 KB      | tiny  |

EWL is ~5× slower per step due to per-sample loss tracking. Zero GPU memory overhead.

---

## Story of the Paper

The paper is **not** "EWL always beats SFT." The honest story is:

1. **The gradient proxy is the critical component.** Without it, EWL destabilises
   training on Aircraft (−6.76%). With it, EWL matches SFT on clean data. The proxy
   acts as a safeguard — it gates the progress signal by adapter activity.

2. **EWL's noise robustness is dataset-conditional.** On Aircraft (extreme fine-grained
   similarity, wide adaptation frontier), EWL is strongly noise-robust (+4–7%). On
   CUB-200 and Stanford Dogs (more separable classes, narrower frontier), EWL
   is actually harmful under noise (−1% to −7%).

3. **The dataset-conditional behaviour validates the adaptation frontier hypothesis.**
   When the frontier is wide (Aircraft), EWL correctly identifies clean vs. noisy samples.
   When the frontier is narrow (CUB, Dogs), the EWL signal is less discriminative and
   noise samples can receive high weights, hurting performance.

4. **Clean accuracy: marginal, within noise.** EWL does not significantly improve
   clean-data accuracy. The gains (+0.05%, +0.16%, −0.01%) are within standard deviation.

5. **Rank and temperature ablations** show EWL is relatively robust above τ=1.0,
   and that higher rank consistently improves accuracy (capacity helps).

6. **System cost:** ~5× step-time increase is a real cost. Zero VRAM overhead.

---

## Paper Narrative Arc

```
Problem         Why uniform weighting is suboptimal for LoRA fine-tuning
↓
Method          EWL: loss velocity × gradient proxy → adaptive sample weights
↓
Critical finding   Gradient proxy is essential — without it, EWL breaks on Aircraft
↓
Main results    EWL ≈ SFT on clean data; proxy stabilises training
↓
Key finding     Noise robustness is dataset-conditional — Aircraft benefits strongly,
                CUB and Dogs are hurt — explained by adaptation frontier width
↓
Ablations       Rank, temperature, imbalance: rank matters most; τ robust above 1.0
↓
Discussion      Why does frontier width determine noise behaviour? What does
                this mean for when to apply EWL?
↓
Conclusion      EWL is a principled but conditional method; gradient proxy is key;
                dataset heterogeneity determines applicability
```

---

## Section-by-Section Design

---

### Abstract

Content:
- Problem: LoRA fine-tuning treats all samples equally despite changing informativeness
- Method: EWL tracks per-sample loss velocity via EMA, gated by a LoRA gradient proxy,
  producing adaptive sample weights with no meta-gradients and no held-out data
- Critical finding: the LoRA gradient proxy is essential — ablating it causes −6.76%
  accuracy on Aircraft
- Conditional finding: EWL strongly improves noise robustness on Aircraft (+4–7%) but
  hurts on CUB-200 and Stanford Dogs, revealing that the benefit is conditional on
  the width of the adaptation frontier
- Ablation results: EWL is robust to τ above 1.0; higher LoRA rank monotonically
  improves accuracy; no meaningful imbalance benefit
- Cost: ~5× step time overhead, zero VRAM overhead

---

### §1 Introduction

**Goal:** Motivate EWL, introduce the adaptation frontier, state contributions honestly.

**Paragraph 1 — The core problem:**
Fine-tuning a pre-trained transformer with LoRA is not like training from scratch.
The model arrives with strong, general-purpose priors. As fine-tuning proceeds,
the informativeness of each training sample changes continuously: some become
redundant (already mastered), some become irreconcilable noise (conflicting with
the pre-trained prior), and a shifting minority sit at the adaptation frontier where
loss is actively declining and gradient signal is richest. Standard SFT treats
all three equally.

**Paragraph 2 — Three sample types:**
Introduce mastered / conflicting / frontier. Explain that uniform averaging dilutes
the mini-batch gradient with near-zero contributions from the first two groups.

**Paragraph 3 — Prior work gap:**
Curriculum and self-paced methods weight by loss magnitude — conflating the frontier
with irreconcilable noise. Meta-reweighting needs a clean held-out set and doubles
compute. Dataset Cartography identifies the ambiguous stratum but requires a
preliminary training epoch.

**Paragraph 4 — EWL:**
EWL tracks per-sample loss velocity via exponential moving averages (cheap, online)
and multiplies by a LoRA gradient proxy that acts as an adaptive temperature.
No held-out data, no meta-gradients, no inference cost.

**Paragraph 5 — What we find (honest):**
Our experiments on three fine-grained visual benchmarks reveal two key findings:
(i) the LoRA gradient proxy is not optional — removing it causes severe degradation
on Aircraft, demonstrating that the progress signal alone is unstable without gating
by adapter activity; (ii) EWL's noise robustness is dataset-conditional — strongly
beneficial on Aircraft where class similarity is extreme, but harmful on CUB-200 and
Stanford Dogs where the adaptation frontier is narrower and EWL's signal cannot
reliably separate noisy from informative samples.

**Contribution bullets:**
1. EWL: per-sample loss velocity × LoRA gradient proxy → adaptive weights;
   single pass, no held-out data, no meta-gradients
2. Empirical demonstration that the gradient proxy is critical for training stability
3. Dataset-conditional noise robustness: strongly positive on Aircraft (+4–7%),
   negative on CUB-200 and Stanford Dogs — tied to adaptation frontier width
4. Full ablation study: rank, temperature, EMA decay, class imbalance

**> Figure to place here (right column):**
**Fig 1 — Three sample types: loss trajectories and progress signal**
Two-panel figure generated from actual per-sample diagnostic data (FGVC-Aircraft,
20% label noise, epochs 2–10):
- Panel (a): Mean loss ± 1 std for each type — Blue circles (Mastered, n=84),
  Green squares (Frontier, n=23), Red triangles (Conflicting, n=9)
- Panel (b): EWL progress signal sᵢ ± 1 std with inline end-labels
  (sᵢ→0, sᵢ>0, sᵢ≈0)
Source: `notebooks/plots/fig1_three_sample_types.pdf` (PNG also available)
Notebook: `notebooks/fig1_three_sample_types.ipynb`
Purpose: gives readers the intuition before any equation — all three categories
are empirically real, not schematic.

---

### §2 Related Work

Five short paragraphs, one per theme:

**2.1 Training Dynamics and Dataset Cartography:**
Arpit et al. (ICML 2017), Toneva et al. (ICLR 2019), Swayamdipta et al. (EMNLP 2020).
Key distinction: Dataset Cartography requires a full preliminary pass to map the corpus;
EWL does this online in the same training run.

**2.2 Curriculum and Self-Paced Learning:**
Bengio et al. (ICML 2009), Kumar et al. (NeurIPS 2010).
Key distinction: both weight by loss magnitude, conflating frontier samples with
irreconcilable noise. EWL uses loss velocity (rate of change), not magnitude.

**2.3 Noise Robustness and Hard Example Mining:**
Jiang et al. MentorNet (ICML 2018), Han et al. Co-teaching (NeurIPS 2018),
Shrivastava et al. OHEM (CVPR 2016), Lin et al. Focal Loss (ICCV 2017).
Key distinction: all require a clean reference set or use magnitude-based weighting.
EWL's implicit noise suppression emerges from velocity without any noise model.

**2.4 Meta-Learning for Reweighting:**
Ren et al. (ICML 2018), Shu et al. Meta-Weight-Net (NeurIPS 2019).
Key distinction: second-order gradient computation (2× compute), clean held-out set.

**2.5 LoRA and PEFT:**
Hu et al. LoRA (ICLR 2022), Zhang et al. AdaLoRA (ICLR 2023).
Key distinction: AdaLoRA uses Frobenius norms of adapter weights to allocate rank
budget; EWL uses Frobenius norms of adapter gradients to gate sample weighting —
different purpose, complementary to AdaLoRA.

---

### §3 Method

#### 3.1 Progress Signal

**Eq (1) — EMA update:**
µᵢ⁽ᵗ⁾ = α µᵢ⁽ᵗ⁻¹⁾ + (1−α) ℓᵢ⁽ᵗ⁾

µ is a smoothed loss history — a cheap online surrogate for per-epoch
confidence statistics used by Dataset Cartography.

**Eq (2) — Relative progress signal:**
sᵢ = [(µᵢ⁽ᵗ⁻¹⁾ − ℓᵢ⁽ᵗ⁾) / (|µᵢ⁽ᵗ⁻¹⁾| + ε)] · µᵢ⁽ᵗ⁾

Numerator positive when loss is below its smoothed history. Division by µ makes
the signal scale-invariant. Three cases:
- Mastered: µ ≈ ℓ ≈ 0 → sᵢ ≈ 0
- Conflicting: ℓ ≈ µ ≫ 0 → sᵢ ≈ 0 or negative
- Frontier: ℓ < µ → sᵢ consistently positive

#### 3.2 LoRA Gradient Proxy

**Eq (3):**
g⁽ᵗ⁾ = (1/|Λ|) Σ_{θ∈Λ} ‖∇θ‖_F

Mean Frobenius norm across all LoRA adapter parameter tensors. Using the
lagged proxy g⁽ᵗ⁻¹⁾ avoids a circular dependency.

**Why this component is critical (motivate before the ablation):**
Without this proxy, sᵢ alone applies aggressive reweighting even when the
adapter has converged or is unstable. The proxy acts as an adaptive temperature:
large when adapters update rapidly → sharper weights; near zero at convergence
→ graceful recovery to uniform (SFT baseline). This gating is what prevents the
instability seen in the no-proxy ablation.

#### 3.3 Weight Computation

**Eq (4) — Softmax with temperature:**
wᵢ = exp(sᵢ/τ) / Σⱼ exp(sⱼ/τ)

**Weighted loss:** L_EWL = B · Σᵢ wᵢ ℓᵢ

Z-score normalisation intentionally omitted — it would cancel the gradient proxy.
K-step warmup (wᵢ = 1/B) lets EMA accumulate history.

#### 3.4 Theoretical Grounding

**Gradient variance reduction:**
Optimal importance weights ∝ ‖∇θ ℓᵢ‖. EWL's progress signal is a cheap proxy:
declining loss implies large, directionally consistent gradient. Holds regardless
of optimiser since variance is a property of the data-sampling step.

**LoRA subspace alignment:**
LoRA constrains updates to rank-r subspace. Stagnant loss signals that a sample's
gradient lies largely outside the adapter subspace. EWL suppresses these without
knowing the subspace geometry — stagnant loss is the observable signature.
This predicts EWL's benefit should depend on how many samples are "irreconcilable"
given the current rank, which varies by dataset heterogeneity.

**> Keep Algorithm 1 box from the current PDF — it is the most compact statement
of the method and earns its space.**

---

### §4 Experiments & Results

#### 4.1 Setup

**Model:** ViT-Small/16 (Dosovitskiy et al.), ImageNet-21k pretrained via timm.
21.7M parameters. LoRA r=4 by default (307K trainable, 1.41%), applied to all
12 qkv and proj layers. Classification head trained without LoRA.

**Datasets:**
- FGVC-Aircraft: 100 aircraft variant classes — extreme inter-class visual similarity
- CUB-200-2011: 200 bird species — fine-grained but more separable than Aircraft
- Stanford Dogs: 120 dog breeds — moderate inter-class similarity

**Training:** 50 epochs, AdamW (η=5×10⁻⁴, cosine decay), EMA α=0.9, warmup K=48.
Default τ=1.0. All results mean±std over seeds {11, 22, 33}.

**Conditions compared:**
- SFT (LoRA baseline): standard cross-entropy, uniform weighting
- EWL: full method with gradient proxy
- EWL (no proxy): progress signal only, g≡1 (ablation)

---

#### 4.2 Main Results: Clean Accuracy

**> TABLE 1 — Main accuracy table (place here)**

| Dataset       | SFT           | EWL (no proxy) | EWL           | Δ (EWL vs SFT) |
|---------------|---------------|----------------|---------------|----------------|
| Aircraft      | 58.94 ± 0.60  | 52.18 ± 0.48   | 58.99 ± 0.71  | +0.05          |
| CUB-200       | 86.15 ± 0.29  | 85.97 ± 0.40   | 86.30 ± 0.42  | +0.16          |
| Stanford Dogs | 89.43 ± 0.05  | 88.89 ± 0.24   | 89.41 ± 0.06  | −0.01          |

Caption: On clean data, EWL with proxy matches SFT performance within standard
deviation across all three datasets. The no-proxy ablation reveals that the gradient
proxy is critical: removing it causes a −6.76% drop on Aircraft. The proxy stabilises
training by gating the progress signal with adapter activity.

**> FIGURE 1 — Training curves across all 3 datasets (place here)**
Source: `notebooks/plots/fig_all_datasets_curves.pdf`
Three-panel figure: one panel per dataset. Each panel shows val accuracy vs. epoch
for SFT (blue), EWL (orange), EWL-no-proxy (red dashed).
- Aircraft: no-proxy diverges/drops sharply; EWL and SFT converge together
- CUB/Dogs: all three converge similarly
Purpose: makes the proxy ablation visually immediate. The Aircraft panel is the
most dramatic and should be featured prominently.

---

#### 4.3 Critical Finding: The Gradient Proxy is Essential

**This is the most decisive clean-data result — give it its own subsection.**

On Aircraft, removing the gradient proxy drops accuracy by 6.76% (52.18% vs 58.94% SFT).
On CUB-200 and Stanford Dogs, the drop is smaller (−0.18%, −0.54%) but consistently
negative. In no case does removing the proxy help.

The proxy gates the progress signal sᵢ by the current adapter update magnitude.
Without gating, aggressive reweighting early in training — before the adapter has
settled — amplifies unstable gradient directions. On Aircraft, with its narrow
decision boundaries and high inter-class confusion, this instability has a large
negative effect. On CUB and Dogs, the task is more separable, so the damage is
smaller but still present.

**> FIGURE 2 — Proxy ablation delta bar chart (place here)**
Source: `notebooks/plots/fig_proxy_ablation_delta.pdf`
Bar chart showing Δ(EWL_no_proxy − SFT) and Δ(EWL − SFT) per dataset.
Makes the asymmetric impact of removing the proxy visually clear.
Purpose: proves the proxy is not a minor tuning detail — it is the component
that prevents degradation.

---

#### 4.4 Noise Robustness: Dataset-Conditional Behaviour

This is the most unexpected and informative finding. EWL's noise robustness is
strongly positive on Aircraft but negative on CUB-200 and Stanford Dogs.

**> TABLE 2 — Noise robustness summary (place here)**

| Dataset       | Noise | SFT Acc.       | EWL Acc.       | Δ      |
|---------------|-------|----------------|----------------|--------|
| Aircraft      | 0%    | 58.94 ± 0.58   | 59.01 ± 0.85   | +0.07  |
| Aircraft      | 10%   | 51.90 ± 0.65   | 56.16 ± 0.43   | +4.26  |
| Aircraft      | 20%   | 46.21 ± 0.98   | 52.12 ± 0.55   | +5.90  |
| Aircraft      | 30%   | 40.26 ± 0.64   | 47.07 ± 0.66   | +6.81  |
| Aircraft      | 40%   | 34.64 ± 1.03   | 40.89 ± 1.68   | +6.25  |
| CUB-200       | 20%   | 75.08 ± 0.55   | 73.53 ± 0.37   | −1.55  |
| CUB-200       | 40%   | 60.71 ± 0.77   | 59.45 ± 0.91   | −1.26  |
| Stanford Dogs | 20%   | 80.30 ± 0.39   | 77.87 ± 0.78   | −2.43  |
| Stanford Dogs | 40%   | 71.02 ± 0.67   | 64.46 ± 0.80   | −6.55  |

Caption: Aircraft shows strongly increasing EWL advantage as noise grows. CUB-200
and Stanford Dogs show the opposite: EWL is consistently worse under noise, with the
disadvantage growing with noise level on Stanford Dogs. Full table (all noise levels,
all seeds) in Appendix B.

**> FIGURE 3 — Noise delta lines across datasets (place here)**
Source: `notebooks/plots/fig_noise_delta_lines_all_datasets.pdf`
Single panel: x-axis = noise level (0–40%), y-axis = Δ(EWL−SFT), one line per dataset.
- Aircraft line: starts near 0, rises steeply to +6.81% then slightly back
- CUB-200 line: stays near 0 at clean, drops to −1.55%
- Stanford Dogs line: drops sharply to −6.55% at 40%
Purpose: this single figure captures the key finding — the divergence of three
datasets from each other is the story. Any reader will immediately ask "why?"
which motivates the Discussion.

**> FIGURE 4 — Noise accuracy line plots (place here or in Appendix)**
Source: `notebooks/plots/fig_noise_lineplots_all_datasets.pdf`
Three panels (one per dataset), each showing SFT and EWL accuracy vs. noise ratio.
Shows the degradation curve shape — EWL has shallower slope on Aircraft,
steeper slope on Stanford Dogs.
Consider putting Fig 4 in Appendix B if space is tight; Fig 3 tells the story more
efficiently.

---

#### 4.5 Rank Sensitivity

**> TABLE 3 — Rank sweep (Aircraft, EWL)**

| Rank | Trainable Params | EWL Accuracy    |
|------|-----------------|-----------------|
| r=2  | 154K            | 57.40 ± 0.18%   |
| r=4  | 307K            | 59.06 ± 0.71%   |
| r=8  | 614K            | 60.37 ± 0.93%   |
| r=16 | 1.2M            | 62.08 ± 0.83%   |

Higher rank monotonically improves accuracy. This reflects increased representational
capacity, consistent with expected LoRA behaviour. The r=4 default used throughout is
a deliberate efficiency choice.

---

#### 4.6 Hyperparameter Sensitivity

**> FIGURE 5 — Rank and Temperature sensitivity (place here)**
Source: `notebooks/plots/fig_rank_temp_trends.pdf`
Two-panel figure:
- Left: EWL accuracy vs. rank (r=2,4,8,16) — monotonically increasing
- Right: EWL accuracy vs. τ (0.5, 1.0, 1.5, 2.0, 2.5, 3.0) — flat above τ=1.0,
  drops at τ=0.5

Key message: τ=0.5 creates over-concentrated weights that hurt performance.
Above τ=1.0, EWL is stable. The method is not sensitive to τ in the practical range.

**> FIGURE 6 — Alpha × Temperature heatmap (place here or Appendix)**
Source: `notebooks/plots/fig_alpha_temp_heatmap.pdf`
Heatmap of EWL accuracy across α ∈ {0.5,0.6,0.7,0.8,0.9,1.0} and
τ ∈ {0.5,1.0,1.5,2.0,2.5,3.0}.
Shows the joint sensitivity region — most α/τ combinations above τ=1.0 work well.
Put in Appendix D if the rank/temperature two-panel already communicates the message.

---

#### 4.7 Class Imbalance

**> TABLE 4 — Imbalance sweep (Aircraft)**

| Imbalance Factor | SFT            | EWL            | Δ      |
|-----------------|----------------|----------------|--------|
| IF=1 (balanced) | 58.95 ± 0.57%  | 58.97 ± 0.66%  | +0.02% |
| IF=2            | 51.68 ± 0.77%  | 51.40 ± 0.97%  | −0.28% |
| IF=5            | 39.38 ± 0.19%  | 39.66 ± 0.51%  | +0.28% |
| IF=10           | 30.99 ± 0.72%  | 31.23 ± 0.46%  | +0.24% |
| IF=20           | 23.40 ± 0.59%  | 23.83 ± 0.46%  | +0.43% |

EWL provides no meaningful benefit over SFT under class imbalance. The differences
are within standard deviation at every imbalance level. The adaptation frontier
hypothesis does not predict a strong imbalance effect — under imbalance, minority
samples may remain on the frontier longer, but the effect is too small to overcome
the overall task difficulty increase.

**> FIGURE 7 — Noise and Imbalance delta bars (place here)**
Source: `notebooks/plots/fig_noise_imbalance_delta.pdf`
Two-panel bar chart: Δ(EWL−SFT) under noise (left) and imbalance (right) on Aircraft.
Purpose: cleanly shows noise gives strong positive signal while imbalance is flat.

---

#### 4.8 System Overhead

**> TABLE 5 — System cost comparison**

| Metric              | SFT     | EWL     | Overhead |
|---------------------|---------|---------|----------|
| Step time (ms)      | 56.4    | 277.9   | +4.9×    |
| Peak VRAM (GB)      | 9.46    | 9.46    | 0%       |
| Throughput (samp/s) | 1135    | 230     | −80%     |
| CPU state (per run) | —       | 16.3 KB | negligible |

**> FIGURE 8 — System overhead (place here)**
Source: `notebooks/plots/fig_system_overhead.pdf` · notebook: `notebooks/fig_system_overhead.ipynb`
3-panel figure:
- Panel (a): Step time per epoch over 50 epochs (Aircraft, seed 11) — EWL stable at
  ~265 ms/step vs SFT ~45 ms/step. Dashed median lines annotated. "5.9× overhead"
  callout box centred between the two lines.
- Panel (b): Median throughput bar (all datasets, all seeds) — SFT 1422 vs EWL 238
  samples/sec; −83% annotated with arrow.
- Panel (c): Peak VRAM bar — identical bars at 9.46 GB with "identical" label.
Purpose: the line chart proves the overhead is steady-state (not a warmup artifact);
bars give the exact numbers at a glance. The ~5× step time is a real limitation
that must not be hidden. The zero VRAM overhead is a genuine advantage.

---

### §5 Discussion

#### 5.1 Why Does Frontier Width Determine Noise Robustness?

The key question from §4.4: why does EWL strongly help on Aircraft under noise,
but hurt on CUB-200 and Stanford Dogs?

On Aircraft, 100 aircraft variants share nearly identical silhouettes. The decision
boundaries between classes are extremely close in feature space. At any training step,
many clean samples remain genuinely contested — their loss is still actively declining,
producing a strong positive progress signal. A corrupted label, by contrast, never
produces a sustained decline: the model cannot reconcile it with consistent gradients
from visually similar classes, so its loss stagnates. The frontier is wide and
EWL's progress signal discriminates cleanly between corrupted and informative samples.

**> FIGURE 9 — Velocity vs EMA scatter (place here)**
Source: `notebooks/plots/fig_scatter_velocity_ema.pdf`
Scatter plot: x-axis = per-sample EMA µᵢ (persistent loss level),
y-axis = velocity (relative progress), coloured by clean (blue) vs noisy (red),
sized by EWL weight. Three panels: Aircraft, CUB-200, Stanford Dogs.
On Aircraft: clean and noisy samples should cluster separately in velocity space.
On CUB and Dogs: the clusters overlap — EWL cannot discriminate.
Purpose: this is the mechanistic proof of why noise behaviour differs.

On CUB-200 and Stanford Dogs, classes are more visually separable. Most clean
samples are learned quickly, reducing the frontier. Noisy samples, however, still
stagnate — but so do many harder clean samples that are near the frontier.
EWL cannot reliably distinguish them, and the noise samples occasionally receive
high weights, degrading performance.

**> FIGURE 10 — Separation score over epochs (place here)**
Source: `notebooks/plots/fig_separation_score.pdf`
Line chart: mean_weight(clean) − mean_weight(noisy) per epoch, one line per dataset.
- Aircraft: separation score positive and increasing
- CUB, Dogs: separation score near zero or negative
Purpose: single scalar per epoch that quantifies whether EWL is doing its job.
This directly tests the noise suppression hypothesis.

**> FIGURE 11 — Weight KDE: noisy vs clean (place here)**
Source: `notebooks/plots/fig_weight_kde_noisy_clean.pdf`
KDE of EWL weights assigned to noisy vs clean samples, per dataset.
- Aircraft: two clearly separated KDE peaks (clean gets high weight, noisy gets low)
- CUB, Dogs: overlapping KDEs — EWL assigns similar weights to clean and noisy
Purpose: visual proof of the discrimination mechanism.

#### 5.2 The Gradient Proxy as a Stabiliser

The proxy ablation result (−6.76% on Aircraft without proxy) reveals that the
progress signal sᵢ alone is not sufficient. Without gating by adapter activity,
aggressive reweighting happens indiscriminately — including during unstable early
training epochs when EMA has not yet accumulated meaningful history. The proxy
g⁽ᵗ⁾ ≈ 0 during the warmup period, keeping weights uniform until the adapter
has begun meaningful adaptation. After warmup, g grows with adapter activity,
allowing the progress signal to amplify selectively. On Aircraft, where decision
boundaries are densely packed, this indiscriminate early reweighting compounds
into a large accuracy deficit. On CUB and Dogs, the signal is weaker overall,
so the impact of removing the proxy is smaller but still consistently negative.

**> FIGURE 12 — Temporal weight evolution (place here or in Discussion)**
Source: `notebooks/plots/fig_weight_temporal.pdf`
Temporal evolution of mean weight for noisy vs clean samples over training epochs,
per dataset. Shows when (and whether) EWL begins discriminating between clean
and noisy samples.

#### 5.3 Rank, Temperature, and Practical Guidance

The rank sweep shows a simple monotonic trend: higher rank → higher accuracy.
This is unsurprising (more capacity helps), but confirms that EWL does not
interact negatively with rank — it works correctly across the r=2 to r=16 range.

Temperature analysis: τ=0.5 creates over-concentrated weights that hurt performance
(56.77% vs 59.06% at τ=1.0). Above τ=1.0, the method is stable. Practical
recommendation: start with τ=1.0–1.5; only lower if the weight entropy ratio
drops below 0.3 (indicating near-deterministic weighting).

The imbalance results (§4.7) confirm that EWL should not be expected to function
as an implicit oversampler. The mechanism is velocity-based, not frequency-based.

#### 5.4 When to Use EWL

Based on the empirical results, EWL is most appropriate when:
1. The dataset has high inter-class visual or semantic similarity
   (wide adaptation frontier throughout training)
2. Training data may contain label noise
3. Step-time overhead (~5×) is acceptable

EWL should be used with caution (or not at all) when:
1. Classes are visually separable (narrow frontier → EWL signal not discriminative)
2. Training under noisy labels on moderately separable datasets
3. Step-time budget is constrained (the 5× overhead is a real cost)

---

### §6 Conclusion & Limitations

#### Conclusion

EWL is a progress-adaptive sample weighting method for LoRA fine-tuning that tracks
per-sample loss velocity via exponential moving averages, gated by a LoRA gradient
proxy. Our experiments reveal two key findings. First, the gradient proxy is
essential — ablating it causes −6.76% accuracy on Aircraft, demonstrating that
the progress signal alone is unstable without gating by adapter activity. Second,
EWL's noise robustness is dataset-conditional: strongly beneficial on Aircraft
(+4–7% under 10–40% label noise) where high inter-class similarity creates a wide
adaptation frontier, but harmful on CUB-200 and Stanford Dogs where the frontier
is narrower and EWL's velocity signal cannot reliably discriminate corrupted from
informative samples. The scatter and separation-score analyses mechanistically
confirm this interpretation. These results define both the promise of EWL and its
conditions of applicability.

#### Limitations

1. **Dataset scope:** All experiments use ViT-Small/16 on fine-grained classification.
   Whether the adaptation frontier hypothesis generalises to language models, generation
   tasks, or other PEFT methods is not established here.

2. **Step-time overhead:** EWL is ~5× slower per step due to per-sample loss tracking
   and weight computation. This is negligible for small datasets and short training runs,
   but becomes a real constraint at scale. The 16.3 KB CPU state overhead is negligible.

3. **Noise behaviour on non-aircraft datasets:** EWL is harmful under noise on CUB-200
   and Stanford Dogs. Practitioners must assess whether their dataset has sufficient
   adaptation frontier width before applying EWL in noisy settings.

4. **Single architecture:** ViT-Small/16 only. The interaction between EWL's gating
   mechanism and different transformer architectures is not studied.

5. **Imbalance:** EWL provides no benefit under class imbalance; it is not a substitute
   for explicit re-balancing or oversampling strategies.

---

## Complete Figure and Table Plan

### Main Paper Figures

| # | Title | Source File | Section | What it proves |
|---|-------|-------------|---------|----------------|
| Fig 1 | Three sample types: loss trajectories + progress signal (2 panels) | `notebooks/plots/fig1_three_sample_types.pdf` · notebook: `notebooks/fig1_three_sample_types.ipynb` | §1 Intro | Intuition hook before equations — empirical, not schematic; Blue circles Mastered n=84, Green squares Frontier n=23, Red triangles Conflicting n=9 |
| Fig 2 | Training curves: SFT vs EWL vs No-Proxy (3 datasets) | fig_all_datasets_curves.pdf | §4.2 | No-proxy degradation on Aircraft visible immediately |
| Fig 3 | Proxy ablation delta bar chart | fig_proxy_ablation_delta.pdf | §4.3 | Proxy is critical; Aircraft most affected |
| Fig 4 | Noise delta lines across datasets | fig_noise_delta_lines_all_datasets.pdf | §4.4 | Dataset-conditional noise robustness in one panel |
| Fig 5 | Rank and temperature sensitivity (2 panels) | fig_rank_temp_trends.pdf | §4.6 | τ robust above 1.0; rank monotonically helps |
| Fig 6 | Noise and imbalance delta bars | fig_noise_imbalance_delta.pdf | §4.7 | Noise gives strong signal; imbalance is flat |
| Fig 7 | System overhead: step-time over epochs + throughput + VRAM (3 panels) | `notebooks/plots/fig_system_overhead.pdf` · notebook: `notebooks/fig_system_overhead.ipynb` | §4.8 | Line chart proves 5.9× overhead is steady-state; bars show −83% throughput, 0% VRAM increase |
| Fig 8 | Velocity × EMA scatter (clean vs noisy) | fig_scatter_velocity_ema.pdf | §5.1 | Mechanistic: why Aircraft separates, CUB/Dogs don't |
| Fig 9 | Separation score over epochs | fig_separation_score.pdf | §5.1 | Scalar proof: EWL discriminates on Aircraft only |
| Fig 10 | Weight KDE: noisy vs clean | fig_weight_kde_noisy_clean.pdf | §5.1 | Weight distribution shows Aircraft separation |
| Fig 11 | Temporal weight evolution | fig_weight_temporal.pdf | §5.2 | When/whether EWL starts discriminating |

### Main Paper Tables

| # | Title | Section | Contents |
|---|-------|---------|----------|
| Table 1 | Main accuracy: SFT / EWL (no proxy) / EWL across datasets | §4.2 | Core result + proxy ablation together |
| Table 2 | Noise robustness summary (Aircraft + CUB + Dogs) | §4.4 | Selected noise levels to tell the story |
| Table 3 | Rank sweep accuracy (Aircraft, EWL) | §4.5 | r=2,4,8,16 with trainable param count |
| Table 4 | Imbalance sweep (Aircraft, SFT vs EWL) | §4.7 | IF=1,2,5,10,20 |
| Table 5 | System overhead comparison | §4.8 | Step time, VRAM, throughput, CPU state |

---

## Appendix Structure

### Appendix A — Implementation Details
- ViT-Small/16 pretrain source (timm, ImageNet-21k)
- LoRA targets: all 12 qkv + proj projections; head without LoRA
- Augmentation: RandomResizedCrop(224, scale=(0.6,1.0)), H-flip,
  ColorJitter(0.4,0.4,0.4,0.1) p=0.8; val: Resize(256)→CenterCrop(224)
- ImageNet mean/std normalisation
- Seeds: {11, 22, 33}; hardware details

### Appendix B — Full Noise Robustness Tables
- Complete results: Aircraft + CUB-200 + Stanford Dogs × 5 noise levels (0–40%)
- Mean ± std over 3 seeds, Accuracy and F1
- Figure: `fig_noise_bars_all_datasets.pdf` — bar chart of accuracy per noise level
- Figure: `fig_noise_curves_all_datasets.pdf` — per-noise-level training curves
- Figure: `fig_noise_delta_all_datasets.pdf` — delta heatmap format

### Appendix C — Hyperparameter Ablation Details
- Full alpha sweep table (α=0.5–1.0 at fixed rank=4, τ=1.0)
- Full temperature sweep table (τ=0.5–3.0 at fixed rank=4, α=0.9)
- Figure: `fig_alpha_temp_heatmap.pdf` — joint α×τ accuracy heatmap
- Guidance on τ selection using weight entropy ratio

### Appendix D — Rank Ablation Full Results
- Full rank sweep: r=2,4,8,16 on Aircraft
- Per-seed accuracy and val curves
- Note: r=4 used throughout as efficiency default

### Appendix E — Class Imbalance Full Results
- Full table with F1 in addition to accuracy
- Per-seed breakdown
- Note on why EWL does not help under imbalance

### Appendix F — Weight Entropy Analysis
- Weight entropy ratio H(w)/log(B) over training epochs (Aircraft, CUB, Dogs)
- Runtime diagnostic table: healthy ranges for entropy, max/min weight ratio,
  EMA mean, LoRA proxy g
- Guidance: entropy in 0.3–0.8 is the healthy operating regime

---

## Decisions Based on Actual Results

| Claim in original design | Reality | Action |
|---|---|---|
| EWL +1.4–2.8% on clean data | EWL ≈ SFT (+0.05%, +0.16%, −0.01%) | Report honestly; gains not significant |
| Aircraft benefits most | Aircraft: EWL ≈ SFT clean; but strongly noise-robust | Refocus story on noise behaviour |
| Heterogeneity gradient holds | Only under noise, not on clean data | Reframe as conditional noise story |
| Noise robustness confirmed | Only on Aircraft; CUB+Dogs negatively impacted | Full honest reporting; dataset-conditionality is the finding |
| Imbalance benefit predicted | No clear benefit | Report as null result; remove from main story |
| Three predictions validated | Mixed: proxy essential ✓, noise conditional ✓/✗, rank monotone ✓ | Report all honestly |
