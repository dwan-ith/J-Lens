# Jacobian Lens (J-Lens): Safety-Relevant Interpretability

Research framework for validating J-Lens as a reliable readout tool for AI safety applications. Investigates whether J-Lens provides genuine windows into model computation versus statistical artifacts, with direct applications to deception detection and alignment monitoring.

## Research Program

### Phase 1: Safety-Specific Readout Validation (`safety_gap.py`)
Tests whether J-Lens readout fidelity degrades on safety-relevant inputs compared to in-distribution text.

- Safety categories: deception, eval awareness, sycophancy, refusal
- Adversarial categories: escalation, roleplay, obfuscation
- Compares web-text lens vs safety-distribution lens
- Quantifies the "safety gap" in readout reliability

### Phase 2: Deception Detection (`deception_detector.py`)
Tracks J-Lens readout trajectories across all layers to detect deceptive alignment.

- Trajectory metrics: divergence area, mid-late gap, peak layer, variance
- Classification: normal / uncertain / suspicious
- Tests on structured deception scenarios (escalation, roleplay, obfuscation)

### Phase 3: Cross-Architecture Scaling (`cross_arch.py`)
Tests whether the proximity artifact (advantage peaks mid-stack, vanishes at output) is architecture-general.

- Models: Qwen3-1.7B, Qwen3-8B, Llama-3.2-3B
- Cross-model lens transfer evaluation

## Key Findings (Qwen3-1.7B)

| Finding | Status |
|---------|--------|
| J-Lens advantage peaks at L14 | **Survives** |
| Causal patching (J-Lens direction) | Overturned by random-direction control |
| Norm distortion (Qwen3 RMSNorm) | Architecture-specific bug |
| Readout fidelity != causal privilege | Core insight |

**Critical distinction**: A direction can be a good *readout* of computation without being a good *handle* on it. J-Lens tells you what the model is thinking (readout), but moving along that direction doesn't uniquely influence what it thinks (no causal privilege).

## Files

### Safety Research
- `safety_prompts.py` — Safety-focused prompt datasets (deception, eval awareness, sycophancy, refusal, adversarial)
- `safety_gap.py` — Phase 1: Safety readout validation
- `deception_detector.py` — Phase 2: Deception detection via trajectory analysis
- `cross_arch.py` — Phase 3: Cross-architecture scaling
- `run_full_study.py` — Unified runner for all phases

### Original Evaluation
- `experiment.py` — Local vs global divergence experiment
- `causal_patching.py` & `causal_patching_24.py` — Causal patching with alpha sweep
- `random_direction_control.py` — Random-direction control (norm-matched)
- `prompts.py` — Original 7-category prompt set

### J-Lens Library
- `jacobian-lens/` — Anthropic's J-Lens implementation (Apache 2.0)

## Setup

1. Place Qwen3-1.7B model in `model/`
2. Place fitted J-Lens in `lens_weights/qwen3-1.7b/jlens/Salesforce-wikitext/`
3. Install dependencies: `pip install torch transformers matplotlib seaborn numpy`

## Running

```bash
# Full study (all phases)
python run_full_study.py all

# Individual phases
python run_full_study.py 1  # Safety gap
python run_full_study.py 2  # Deception detection
python run_full_study.py 3  # Cross-architecture
```

## Results

Results are saved to `results/phase1_safety_gap/`, `results/phase2_deception/`, and `results/phase3_cross_arch/`.
