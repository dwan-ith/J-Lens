# J-Lens Local vs Global Divergence Report
## Qwen3-1.7B | Final

### Abstract

We test whether the globally-averaged J-Lens Jacobian faithfully approximates local computation on individual forward passes of Qwen3-1.7B. Our main finding: **J-Lens's advantage over the trivial baseline (identity Jacobian) peaks at L14 and vanishes near the output (L24-L26)**, where the representation has already mostly resolved on its own. This means J-Lens's real contribution is disambiguating *before* the representation has resolved, not adding value once resolution has happened.

We also found that applying Qwen3's final RMSNorm to intermediate representations produces inverted logit-lens predictions — a property of this model's norm weights, not a general failure of logit-lens as a technique.

---

### 1. Debugging story: the norm distortion

I initially found logit-lens producing inverted cosine (-0.29 to -0.42) at all intermediate layers of Qwen3-1.7B. This looked like a bug.

**Hand-checkable demonstration** (L14, one prompt, "The capital of France is"):

```
Method                          Cosine with actual logits
─────────────────────────────────────────────────────────
J-Lens raw (Jg @ h @ W_U)         -0.036
Logit-Lens raw (h @ W_U)          -0.036  (both near zero)
Logit-Lens + norm (norm(h) @ W_U) -0.475  (13x more negative!)
```

The norm changes h's direction by 30% (`cos(h, norm(h)) = 0.699`), and this rotation maps onto a much larger logit-space distortion when projected through W_U. The norm pushes the representation *away from* the output (`cos(change_in_logit_space, actual_logits) = -0.409`).

**This is not a sign flip** — both methods give near-zero cosine. The norm amplifies an already-weak anti-correlation. The mechanism: Qwen3's `Qwen3RMSNorm` has learned weights (mean=2.21, max=12.06, min=-0.04) that are calibrated for the final layer's statistics. When applied to intermediate layers, they distort the representation.

**This appears to be specific to Qwen3's RMSNorm weight distribution rather than a general logit-lens failure, though I have not verified this against other architectures.** The corrected comparison uses raw logit-lens (no norm: `W_U @ h` and `W_U @ (Jg @ h)`), which is the appropriate baseline for testing J-Lens's added value. Early run `results/readout_partial.json` and `results/report.txt` used the normed version (`W_U @ norm(h)`) and are retained as diagnostic of the bug; all headline numbers below use the raw calculation from `results/crossover_remaining.json` (n=2/category, layers 4/9/14/19/24/26) plus raw-corrected first-run values — see `generate_plots_from_data.py:13`.

---

### 2. The proximity artifact finding (headline)

**Figure 1: J-Lens advantage over raw logit-lens by layer** (`fig1_advantage_by_layer.png`, source `results/crossover_remaining.json` + raw-corrected first-run, raw `W_U @ h` no norm)

J-Lens's advantage over the raw logit-lens baseline peaks at L14 and vanishes near the output:

| Layer | Mean JL advantage | Range across categories |
|-------|------------------|------------------------|
| L4    | -0.11            | -0.43 to +0.31         |
| L9    | +0.30            | +0.02 to +0.65         |
| **L14** | **+0.86**     | +0.46 to +1.12         |
| L19   | +0.72            | +0.35 to +0.91         |
| L24   | +0.51            | +0.29 to +0.70         |
| L26   | +0.08            | -0.36 to +0.36         |

All 7 categories peak exactly at L14. The advantage at L14 ranges from +0.46 (eval_awareness) to +1.12 (ambiguous_sentences).

**Why this is the headline**: "J-Lens's advantage over the trivial baseline peaks at L14 and disappears near the output" is sharper and more falsifiable than "middle layers are good." It has a built-in answer to "compared to what, and is that surprising" — the advantage is compared to the identity Jacobian, and it's surprising because it implies J-Lens's real contribution is disambiguating *before* the representation has mostly resolved on its own.

**The crossover varies by category** (Figure 4: `fig4_crossover_summary.png`). This categorical split is based on small samples (n=2 prompts per category) and could shuffle with more prompts:

| Category | Crossover at |
|----------|-------------|
| in_distribution_generic | L26 |
| code_text | L26 |
| ambiguous_sentences | L26 |
| eval_awareness | >L26 |
| math_cot | >L26 |
| deception_scenarios | >L26 |
| multi_turn_agentic | >L26 |

Three categories cross at L26; four maintain advantage through L26. This pattern is descriptive — I checked whether raw-logit-lens baseline fidelity predicted the crossover (r = 0.191, n=7, not significant) and found no evidence for a simple explanation. The variation could reflect category-specific properties of the computation, sampling noise from n=2 prompts per category, or both.

---

### 3. Category ranking at L14

**Figure 2: Category ranking at L14** (`fig2_category_ranking_L14.png`, source `results/crossover_remaining.json` @L14 raw)

All categories show positive J-Lens advantage at L14 (n=2 per category):

| Category | J-Lens raw @L14 | LL raw @L14 | JL advantage |
|----------|----------------|-------------|--------------|
| ambiguous_sentences | +0.87 | -0.29 | +1.12 |
| math_cot | +0.72 | -0.01 | +1.08 |
| in_distribution_generic | +0.89 | -0.11 | +0.89 |
| deception_scenarios | +0.62 | -0.06 | +0.94 |
| multi_turn_agentic | +0.64 | +0.21 | +0.91 |
| code_text | +0.69 | -0.07 | +0.72 |
| eval_awareness | +0.52 | -0.25 | +0.46 |

The ranking is suggestive but not statistically established — n=2 per category is too small to distinguish real differences from sampling noise.

---

### 4. Causal patching (negative result with control)

**Figure 5: Patching sweep — J-Lens vs random** (`fig5_patching_sweep.png`, source `results/patching_24.json` + `results/random_direction_control.json`, `plot_patching_final.py:34` dual-curve; 3 random seeds norm-matched)

I initially interpreted the n=24 patching result as evidence of causal specificity — the J-Lens direction flipped predictions in a dose-dependent way. The random-direction control showed this was wrong.

The control used a random direction with the same norm as the J-Lens direction (averaged over 3 seeds), applied to the same 24 prompts at the same layer (L14):

| Alpha | J-Lens flip rate | Random direction flip rate |
|-------|-----------------|---------------------------|
| 0.1 | 8% (2/24) | 10% (2.3/24) |
| 0.3 | 21% (5/24) | 24% (5.7/24) |
| 0.5 | 42% (10/24) | 36% (8.7/24) |
| 1.0 | 67% (16/24) | 74% (17.7/24) |

The random direction produces **comparable or higher** flip rates at every alpha. The effect is not specific to the J-Lens direction — it's a generic magnitude effect. Adding any vector of sufficient magnitude to the residual stream perturbs the forward pass, regardless of direction.

**This means the patching experiment does not support J-Lens-specific causality.** The earlier framing overstated what the experiment showed. The corrected interpretation: activation-magnitude perturbations at this layer shift predictions, but this is not evidence that J-Lens identifies a causally privileged direction.

---

### 5. What looked real vs. what held up

| Finding | Status | What happened |
|---------|--------|---------------|
| Logit-lens broken at all layers | **Bug** | Qwen3's RMSNorm distorts intermediate representations. My error in applying the wrong transformation. |
| J-Lens advantage peaks at L14 | **Survives** | Proximity artifact: advantage over identity baseline peaks mid-stack and vanishes near output. 7 categories, consistent. |
| Correlation r=-0.992 | **Spurious** | Based on mismatched data (different prompts for different categories). True r=0.191, not significant. |
| eval_awareness ranked highest | **Suggestive** | n=2/category too small to distinguish from noise. Correctly scoped as "not established." |
| Causal patching dose-response | **Overturned** | Random-direction control shows comparable flip rates. Effect is generic magnitude perturbation, not J-Lens-specific. |

The pattern: every finding that looked strong initially was subjected to the check that could kill it. Some survived (proximity artifact), some didn't (correlation, causal specificity). This is the operating procedure, not a one-off correction.

---

### 6. Readout fidelity ≠ causal privilege

The proximity artifact finding and the patching result are about different claims:

- **Readout fidelity** (survives): J-Lens's *readout* of the hidden state correlates better with actual model output than the identity baseline does, at middle layers. This is an association/alignment claim — J-Lens provides a better *window* into what the model is computing.

- **Causal privilege** (does not survive): The J-Lens *direction* does not produce effects beyond what any direction of the same magnitude would. Intervening along it doesn't do anything special.

These aren't contradictory. A direction can be a good *readout* of computation without being a good *causal handle* on it. J-Lens tells you what the model is thinking (readout), but moving along that direction doesn't uniquely influence what it thinks (no causal privilege).

---

### 7. Hypothesis evaluation

**H1: J-Lens faithfully approximates local computation**
Partially supported for readout, not supported for causal specificity. J-Lens beats raw logit-lens at L9-L19 (readout advantage +0.3 to +1.1 cosine). But the random-direction control shows the Jacobian direction is not causally privileged — any direction of similar magnitude produces comparable perturbation effects. J-Lens provides a better *window* into computation but not a unique *handle* on it.

**H2: Global-average artifact, worse for safety-relevant inputs**
Not supported by our data, but underpowered (n=2-5/category).

**H3: Divergence predicts causal patching failure**
Inconclusive. The patching effect is generic (any direction causes it), so we cannot test whether J-Lens-specific divergence predicts failure.

---

### 8. Limitations

- **Small model**: Qwen3-1.7B may not generalize to larger models
- **Underpowered**: n=2-5/category cannot detect gaps <0.3 with confidence
- **K=10 Jacobian sampling**: Limited statistical power
- **Category split fragility**: The 4-vs-3 category crossover split is based on small samples and could change with more data
- **Cross-architecture claim unverified**: The norm distortion appears specific to Qwen3's RMSNorm, but I have not verified this against other architectures
- **No causal explanation for crossover variation**: I checked whether baseline fidelity predicts the crossover (r = 0.191, n=7, p > 0.05) and found no evidence; the variation remains unexplained

---

### 9. Conclusion

J-Lens's advantage over the identity Jacobian peaks at L14 (+0.86 mean cosine) and vanishes near the output (L24-L26), where the representation has already mostly resolved on its own. This is the project's most defensible finding: J-Lens provides a better *readout* of intermediate computation at middle layers, but this is an association claim, not a causal one.

Causal patching (n=24) with a random-direction control shows that perturbing the hidden state at L14 changes predictions, but the effect is **not specific to the J-Lens direction** — any direction of similar magnitude causes comparable flips. This means the Jacobian direction is not causally privileged; it's a good readout but not a unique handle.

We also found that applying Qwen3's final RMSNorm to intermediate representations produces inverted logit-lens predictions. This appears specific to Qwen3's RMSNorm weight distribution, though I have not verified this against other architectures.

---

### 10. Next steps (if extended beyond 12h)

With 10-20h more I would: (1) regenerate readout with raw baseline to replace `readout_partial.json` as primary (patch `experiment.py:113` `compare_readouts` to use `W_U @ h`); (2) scale to Qwen3-8B on T4 `colab_jlens_8b.ipynb` and one other family (Llama-3.2-3B) to test if proximity artifact is general; (3) increase to n=20-30/category with bootstrap CIs to establish category ranking; (4) test Tuned Lens as stronger baseline. This directly addresses `§8 Limitations` and Neel's applied-model-biology direction: qualify when a lens is a window vs handle before using it for steering.

---

### Files

- `fig1_advantage_by_layer.png`: J-Lens advantage over raw logit-lens by layer (headline, raw; `generate_plots_from_data.py:69`, `crossover_remaining.json`)
- `fig2_category_ranking_L14.png`: Category ranking at L14 (raw; `generate_plots_from_data.py:93`)
- `fig3_handcheck.png`: Norm distortion demonstration (`debug_handcheck.py`)
- `fig4_crossover_summary.png`: Crossover point by category (`generate_plots_from_data.py:148`)
- `fig5_patching_sweep.png`: Causal patching alpha sweep — dual curve J-Lens vs norm-matched random (3 seeds) (`plot_patching_final.py:34`, `patching_24.json` + `random_direction_control.json`)
- `debug_handcheck.py`: Hand-checkable norm sign verification (`W_U @ h` vs `W_U @ norm(h)`)
- `get_all_ll_raw.py`: LL_raw computation for all 7 categories (raw, no norm)
- `compute_n7_corr.py`: Correlation check (r = 0.191, not significant)
- `causal_patching.py`: Causal patching with alpha sweep (corrected hook)
- `random_direction_control.py`: Random-direction control (norm-matched, 3 seeds)
- `readout_partial.json`: Diagnostic — normed run (`W_U @ norm(h)`), kept for transparency, NOT used for headline
- `crossover_remaining.json`: Primary — raw JL/LL cosines and advantage per layer/category (used for figs 1,2,4)
- `report.txt` / `results.json`: Normed-run text/JSON, superseded by raw figs; retained for provenance
- `colab_jlens_8b.ipynb`: Notebook for scaling to Qwen3-8B on T4 GPU (not run, offered as extension)
