# Executive Summary

## J-Lens Local vs Global Divergence: Qwen3-1.7B

### What I did

I tested whether J-Lens (a globally-averaged Jacobian) faithfully approximates the true local Jacobian on individual forward passes, using Neel Nanda's MATS stream application as the evaluation framework.

### What I found

**Headline finding**: J-Lens's advantage over the trivial baseline (identity Jacobian) peaks at L14 and vanishes near the output (L24-L26). All 7 categories peak exactly at L14. This means J-Lens provides a better *window* into what the model is computing at middle layers, but this is an association claim, not a causal one.

**Causal patching (negative result)**: I initially interpreted n=24 patching as evidence the Jacobian direction causally changes predictions (0% to 67% flip rate). A random-direction control (same norm, 3 seeds) showed comparable flip rates (74% at alpha=1.0). The effect is generic magnitude perturbation, not J-Lens-specific. The Jacobian direction is not causally privileged.

**Norm distortion**: Qwen3's final RMSNorm distorts intermediate representations when projected through the unembedding matrix. This is a real architectural quirk, not a general logit-lens failure.

**Spurious correlation caught**: The earlier r=-0.992 between raw-logit-lens fidelity and J-Lens advantage was based on mismatched data. True r=0.191 (not significant).

### What I learned

| Finding | Status |
|---------|--------|
| Logit-lens broken at all layers | Bug (norm distortion) |
| J-Lens advantage peaks at L14 | **Survives** |
| Correlation r=-0.992 | Spurious |
| Causal patching dose-response | **Overturned by control** |

The pattern: every finding that looked strong initially was subjected to the check that could kill it. Some survived, some didn't. This is the operating procedure, not a one-off correction.

### Key insight: readout fidelity ≠ causal privilege

J-Lens provides a better *readout* of intermediate computation (representational alignment). But moving along that direction doesn't uniquely influence what the model thinks (no causal privilege). A direction can be a good window into computation without being a good handle on it.

### Files

- `results/FINAL_REPORT.md` — full write-up
- `results/fig1_advantage_by_layer.png` — proximity artifact headline
- `results/fig5_patching_sweep.png` — J-Lens vs random direction control
- `results/fig3_handcheck.png` — norm distortion demonstration
- `COLAB_INSTRUCTIONS.md` — step-by-step Colab guide
