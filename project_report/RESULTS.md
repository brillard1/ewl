# EWL Experimental Results — Full Report
> ENSF 619 | All 231 runs complete | Seeds: 11, 22, 33 | Date: 2026-04-13

---

## Summary of Findings

| Finding | Result | Implication |
|---|---|---|
| EWL vs SFT (clean data) | Marginal: +0.05–+0.15% on CUB/Dogs, +0.05% on Aircraft | EWL does not meaningfully improve over SFT on clean data |
| EWL vs SFT (noisy data, Aircraft) | +4.3% to +6.8% at 10–40% noise | Strong noise robustness on fine-grained, high-confusion dataset |
| EWL vs SFT (noisy data, CUB-200) | **−0.9% to −1.6%** (EWL worse) | Noise robustness does not generalise to CUB-200 |
| EWL vs SFT (noisy data, Stanford Dogs) | **−0.9% to −6.6%** (EWL worse) | EWL hurts under noise on Stanford Dogs |
| EWL vs EWL-no-proxy (clean) | No-proxy is −6.8% on Aircraft, −0.2% on CUB/Dogs | LoRA grad proxy is critical for Aircraft; less so for easier datasets |
| Class imbalance | +0.02% to +0.43% (marginal) | No meaningful benefit from EWL under imbalance |
| VRAM overhead | **0 bytes** (identical 9.456 GB) | EWL adds zero GPU memory overhead |
| Compute overhead | **5.9× slower** per step (265ms vs 45ms) | Significant wall-clock cost despite zero VRAM cost |

---

## 1. Main Results: SFT vs EWL vs EWL-no-proxy (Clean Data)

All runs: ViT-Base with LoRA (rank=4, alpha=0.9, temp=1.0), 10 epochs, 3 seeds.

### Table 1 — Top-1 Test Accuracy on 3 Fine-Grained Visual Benchmarks

| Dataset | SFT (mean ± std) | EWL-no-proxy (mean ± std) | EWL (mean ± std) | EWL vs SFT Δ |
|---|---|---|---|---|
| FGVC-Aircraft | 58.94% ± 0.60% | 52.18% ± 0.48% | **58.99% ± 0.71%** | **+0.05%** |
| CUB-200-2011 | 86.15% ± 0.29% | 85.97% ± 0.40% | **86.30% ± 0.42%** | **+0.15%** |
| Stanford Dogs | **89.43% ± 0.05%** | 88.89% ± 0.24% | 89.41% ± 0.06% | **−0.02%** |

Individual seed results:

**FGVC-Aircraft**
- SFT: 58.15%, 59.08%, 59.59% → mean=58.94%
- EWL-no-proxy: 51.73%, 51.97%, 52.84% → mean=52.18%
- EWL: 58.00%, 59.35%, 59.62% → mean=58.99%

**CUB-200-2011**
- SFT: 85.74%, 86.30%, 86.40% → mean=86.15%
- EWL-no-proxy: 85.42%, 86.16%, 86.33% → mean=85.97%
- EWL: 85.73%, 86.47%, 86.71% → mean=86.30%

**Stanford Dogs**
- SFT: 89.37%, 89.42%, 89.49% → mean=89.43%
- EWL-no-proxy: 88.71%, 88.72%, 89.23% → mean=88.89%
- EWL: 89.35%, 89.41%, 89.49% → mean=89.41%

### Key Observations

1. **EWL ≈ SFT on clean data.** The gains are statistically negligible (+0.05% to +0.15%). Given std of 0.4–0.7%, these differences are within noise across all three datasets.

2. **EWL-no-proxy collapses on Aircraft (−6.8%).** Removing the LoRA gradient proxy causes a catastrophic drop on FGVC-Aircraft but only marginal degradation on CUB-200 (−0.18%) and Stanford Dogs (−0.54%). This suggests the grad proxy is essential for the high-confusion Aircraft dataset and acts as a stabiliser — without it, the progress-only signal gives pathological weights on a 100-class near-identical-silhouette dataset.

3. **Heterogeneity gradient is not confirmed on clean data.** The paper design predicted Aircraft > CUB > Dogs ordering in EWL gains. On clean data, all gains are near zero — there is no ordering to confirm or deny.

---

## 2. Noise Robustness Ablation

Fixed: rank=4, alpha=0.9, temp=1.0. Label noise applied uniformly to training set only. 3 seeds each.

### Table 2 — FGVC-Aircraft: EWL vs SFT under Label Noise

| Noise | SFT (mean ± std) | EWL (mean ± std) | **Δ (EWL − SFT)** |
|---|---|---|---|
| 0% | 58.94% ± 0.22% | 59.01% ± 0.17% | +0.07% |
| 10% | 51.90% ± 0.75% | 56.16% ± 0.50% | **+4.26%** |
| 20% | 46.21% ± 0.99% | 52.12% ± 0.60% | **+5.90%** |
| 30% | 40.26% ± 0.77% | 47.07% ± 0.79% | **+6.81%** |
| 40% | 34.64% ± 1.06% | 40.89% ± 1.78% | **+6.25%** |

**Finding:** EWL provides strong and growing noise robustness on Aircraft. The advantage widens from +0.07% at 0% noise to +6.81% at 30% noise, then slightly narrows at 40% (likely because both methods are severely degraded). This is the strongest positive result for EWL.

### Table 3 — CUB-200-2011: EWL vs SFT under Label Noise

| Noise | SFT (mean ± std) | EWL (mean ± std) | **Δ (EWL − SFT)** |
|---|---|---|---|
| 0% | 86.30% ± 0.29% | 86.34% ± 0.38% | +0.03% |
| 10% | 81.24% ± 0.39% | 80.32% ± 0.37% | **−0.92%** |
| 20% | 75.08% ± 0.61% | 73.53% ± 0.58% | **−1.55%** |
| 30% | 68.07% ± 0.69% | 67.00% ± 0.60% | **−1.07%** |
| 40% | 60.71% ± 0.55% | 59.45% ± 0.84% | **−1.26%** |

**Finding:** EWL performs *worse* than SFT under noise on CUB-200. The disadvantage is small but consistent across all noise levels and all 3 seeds.

### Table 4 — Stanford Dogs: EWL vs SFT under Label Noise

| Noise | SFT (mean ± std) | EWL (mean ± std) | **Δ (EWL − SFT)** |
|---|---|---|---|
| 0% | 89.48% ± 0.06% | 89.45% ± 0.07% | −0.03% |
| 10% | 84.95% ± 0.31% | 84.09% ± 0.44% | **−0.86%** |
| 20% | 80.30% ± 0.26% | 77.87% ± 0.92% | **−2.43%** |
| 30% | 75.95% ± 0.36% | 71.55% ± 0.90% | **−4.40%** |
| 40% | 71.02% ± 0.35% | 64.46% ± 0.36% | **−6.55%** |

**Finding:** EWL performs substantially worse than SFT under noise on Stanford Dogs, with the gap worsening as noise increases. At 40% noise, EWL is −6.55% behind SFT. This is the most unexpected result of the entire study.

### Interpretation of Dataset-Dependent Noise Results

The noise robustness of EWL appears to be **dataset-specific, not general**:

- **Aircraft (100 near-identical classes):** High inter-class confusion means corrupted labels produce genuinely inconsistent gradients → loss on noisy samples stagnates → EWL correctly suppresses them. The confusion is the key enabler.
- **CUB-200 (200 bird species):** Moderate fine-grainedness. EWL may be down-weighting hard-but-clean samples (species with similar plumage) along with noisy ones, net negative effect.
- **Stanford Dogs (120 breeds):** More inter-class separability. EWL appears to over-suppress hard clean samples under noise, actively hurting generalisation. The progress signal may conflate "hard sample" with "noisy sample" when classes are visually distinct enough for most samples to make progress.

**Hypothesis:** EWL's noise robustness requires that clean hard samples show detectable progress (declining loss) while noisy samples stagnate. On Aircraft, near-identical classes create enough confusion that even clean hard samples are hard to distinguish from noisy ones early on — so the EMA history correctly identifies stagnant patterns. On Stanford Dogs, most clean samples show clear progress even under heavy noise, making noisy samples stand out — but EWL's progress signal is calibrated to the *whole batch*, and the relative weighting may suppress valuable high-loss clean samples.

---

## 3. Aircraft Hyperparameter Ablations

Fixed: FGVC-Aircraft, 3 seeds (11, 22, 33) per configuration.

### Table 5 — LoRA Rank Ablation (alpha=0.9, temp=1.0)

| Rank | EWL Test Acc (mean ± std) | n |
|---|---|---|
| 2 | 57.40% ± 0.18% | 3 |
| 4 | 59.06% ± 0.71% | 3 |
| 8 | 60.37% ± 0.93% | 3 |
| 16 | 62.08% ± 0.83% | 3 |

**Finding:** Accuracy increases monotonically with rank. Each doubling of rank yields roughly +1–1.7% accuracy. This is consistent with higher-rank adapters having more expressive capacity. Note: this is EWL accuracy only — we do not have a SFT rank sweep to measure whether EWL's *advantage* diminishes with rank (as the theory predicts). Based on clean-data results (EWL ≈ SFT at rank=4), the rank sweep mostly measures expressivity, not EWL-specific benefit.

### Table 6 — EMA Alpha Ablation (rank=4, temp=1.0)

| Alpha | EWL Test Acc (mean ± std) | n |
|---|---|---|
| 0.5 | 58.18% ± 0.40% | 3 |
| 0.6 | 58.18% ± 0.54% | 3 |
| 0.7 | 58.21% ± 0.85% | 3 |
| 0.8 | 58.81% ± 0.83% | 3 |
| 0.9 | 59.06% ± 0.71% | 3 |
| 1.0 | 59.21% ± 0.85% | 3 |

**Finding:** EWL is robust to alpha. Performance increases gently from alpha=0.5 to 1.0 (less than 1 percentage point total spread). Higher alpha (slower EMA decay, longer memory) is slightly better, with alpha=1.0 best. This makes intuitive sense: longer-horizon memory provides a more stable estimate of loss velocity. Alpha=0.9 is a reasonable default.

### Table 7 — Temperature Ablation (rank=4, alpha=0.9)

| Temperature | EWL Test Acc (mean ± std) | n |
|---|---|---|
| 0.5 | 56.77% ± 1.05% | 3 |
| 1.0 | 59.06% ± 0.71% | 3 |
| 1.5 | 59.23% ± 0.74% | 3 |
| 2.0 | 59.16% ± 0.60% | 3 |
| 2.5 | 59.10% ± 0.56% | 3 |
| 3.0 | 59.09% ± 0.42% | 3 |

**Finding:** Temperature is critical only at the low end. temp=0.5 significantly hurts (−2.3% vs temp=1.0), likely because weights collapse to near-one-hot, destabilising training. Above temp=1.0, performance is essentially flat (59.06–59.23%) — EWL is robust to temperature in the range [1.0, 3.0]. The paper's recommendation of temp=1.0 is safe and conservative.

---

## 4. Noise + Class Imbalance Ablation (Aircraft Only)

Fixed: rank=4, alpha=0.9, temp=1.0, 3 seeds.

### Table 8 — Noise Sweep (replicated on Aircraft, consistent with §2)

| Noise | SFT (mean ± std) | EWL (mean ± std) | Δ |
|---|---|---|---|
| 0% | 58.83% ± 0.63% | 59.02% ± 0.70% | +0.19% |
| 10% | 51.72% ± 0.63% | 55.99% ± 0.47% | **+4.27%** |
| 20% | 46.32% ± 0.93% | 52.21% ± 0.49% | **+5.88%** |
| 30% | 40.27% ± 0.70% | 47.25% ± 0.75% | **+6.98%** |
| 40% | 34.58% ± 0.89% | 41.00% ± 1.61% | **+6.42%** |

Consistent with the noise_all_datasets sweep — confirms Aircraft noise robustness across both sweep setups.

### Table 9 — Class Imbalance Sweep (Aircraft, noise=0%)

Imbalance ratio = fraction of majority-class samples available for minority classes (1.0=balanced, 0.05=20:1 imbalance).

| Imbalance Ratio | SFT (mean ± std) | EWL (mean ± std) | Δ |
|---|---|---|---|
| 1.0 (balanced) | 58.95% ± 0.57% | 58.97% ± 0.66% | +0.02% |
| 0.5 (2:1) | 51.68% ± 0.77% | 51.40% ± 0.97% | −0.28% |
| 0.2 (5:1) | 39.38% ± 0.19% | 39.66% ± 0.51% | +0.28% |
| 0.1 (10:1) | 30.99% ± 0.72% | 31.23% ± 0.46% | +0.24% |
| 0.05 (20:1) | 23.40% ± 0.59% | 23.83% ± 0.46% | +0.43% |

**Finding:** EWL provides no meaningful benefit under class imbalance. Deltas range from −0.28% to +0.43%, all within the noise floor given ±0.6–1.0% std. The theoretical mechanism (minority samples stay on the frontier longer) does not translate to measurable accuracy improvement. This section should be moved to the appendix only — it does not support a main-text claim.

---

## 5. System Metrics

All measured on FGVC-Aircraft, epoch 10, batch size=64.

### Table 10 — VRAM, Step Time, Throughput

| Method | Peak VRAM | Step Time | Throughput | EWL CPU State |
|---|---|---|---|---|
| SFT (lora_sft) | 9.456 GB | 45.1 ms | ~1,420 samples/sec | — |
| EWL | 9.456 GB | 264.7 ms | ~242 samples/sec | 16.3 KB |
| EWL-no-proxy | 9.456 GB | 264.9 ms | ~241 samples/sec | 16.3 KB |

### Key Points

1. **Zero VRAM overhead.** EWL and SFT use identical peak GPU memory (9.456 GB). All EWL state (loss EMA + seen mask) lives on CPU as 16.3 KB tensors — negligible even at scale. This is a genuine advantage over methods that store gradients or model copies on GPU.

2. **~5.9× compute overhead per step.** EWL's step time (265ms) is nearly 6× slower than SFT (45ms). Both use batch size 64. The overhead comes from per-sample loss reduction (no `.mean()` — must compute N individual losses), EMA updates, and weight computation. This is a real and significant cost that should be reported clearly.

3. **Effective throughput: ~242 vs ~1,420 samples/sec.** For a training run that would take 1 hour with SFT, EWL requires ~6 hours. At the 10-epoch scale used here, Aircraft EWL takes ~6 minutes vs ~1 minute for SFT.

4. **EWL and EWL-no-proxy are identical in system cost.** The grad proxy computation (Frobenius norm of existing `.grad` tensors) adds negligible overhead on top of the per-sample loss computation.

---

## 6. Summary of Paper Design vs Actual Results

| Claim in PAPER_DESIGN.md | Actual Result | Verdict |
|---|---|---|
| "+1.7–2.8% accuracy across 3 benchmarks" | +0.05% to +0.15% on clean data | **Not confirmed** |
| "EWL advantage widens under noise" | Only on Aircraft (+6.8%); CUB-200 and Dogs show EWL *worse* | **Partially confirmed (Aircraft only)** |
| "Heterogeneity gradient: Aircraft > CUB > Dogs" | On noise: Aircraft ✓, but CUB/Dogs are negative — ordering is reversed | **Not confirmed as stated** |
| "Dataset Cartography alignment (r=0.73)" | Not yet measured from these runs | **Pending analysis** |
| "LoRA rank dependence — gain shrinks with rank" | Rank sweep only has EWL; no SFT baseline for comparison | **Cannot confirm without SFT rank sweep** |
| "Imbalance: EWL benefits minority classes" | +0.02% to +0.43%, not significant | **Not confirmed** |
| "EWL-no-proxy catastrophic on Aircraft" | −6.8% vs SFT (52.18% vs 58.94%) | **Confirmed** |
| "Zero VRAM overhead" | Confirmed: identical 9.456 GB | **Confirmed** |

---

## 7. Recommendations for Paper

### What to include in main text

1. **Aircraft noise robustness** is the strongest result. Table showing +4.3% to +6.8% improvement at 10–40% noise with all 3 seeds is clean and compelling. This belongs in the main paper as the primary contribution evidence.

2. **EWL-no-proxy ablation** shows the grad proxy is essential on Aircraft (−6.8%). Include in main text as ablation validating the proxy mechanism.

3. **System metrics table**: zero VRAM overhead + 5.9× step-time cost should be stated honestly. It's a genuine engineering trade-off.

4. **Temperature ablation**: the flat response above temp=1.0 is a positive robustness result — EWL is not sensitive to this hyperparameter over a wide range.

### What to move to appendix or reframe

1. **Clean-data accuracy gains (+0.05–+0.15%)** are not the story. The method is noise-robust, not universally better. Reframe the narrative: EWL is a noise-robust training method, not a universal accuracy booster.

2. **CUB-200 and Stanford Dogs noise results are negative.** These must be reported honestly. Possible framing: EWL's noise robustness depends on inter-class visual similarity — it works when corrupted labels produce genuinely stagnant training dynamics (high-confusion datasets), and may backfire when clean hard samples are difficult to distinguish from noisy ones.

3. **Class imbalance results** (marginal, non-significant) → Appendix only.

4. **The "+1.7–2.8%" claim in the abstract needs revision.** The actual clean-data results do not support these numbers. The noise-condition results for Aircraft do support large gains, but only under noise.

### Open questions for further analysis

- **Why does EWL hurt on CUB-200 and Dogs under noise?** Needs investigation of per-sample weight trajectories — are clean hard samples being suppressed?
- **Dataset Cartography alignment** (Pearson r between σᵢ and w̄ᵢ) — still needs to be computed from saved training histories.
- **SFT rank sweep** — currently only EWL was swept over rank. Need SFT baselines at rank=2,8,16 to verify the "EWL gain shrinks with rank" hypothesis.
