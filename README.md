# Jacobian Lens (J-Lens): Causal Divergence and Representation Tracking

This repository contains the rigorous evaluation testbed for the **Jacobian Lens (J-Lens)** methodology, specifically investigating the causal divergence between globally-averaged Jacobian representations and instantaneous local Jacobians on the `Qwen3-1.7B` architecture.

## Overview

The core hypothesis underpinning this framework is whether global Jacobian approximations—learned over broad offline distributions—can accurately track and manipulate the latent space semantics computed dynamically during causal forward passes. By applying targeted interventions, this codebase mathematically and empirically dissects the validity of interpreting Language Models via singular global Jacobians.

### Core Modules

* **`experiment.py`**
  The central analytical driver. Formally assesses *Local vs. Global Divergence*, quantifying the exact misalignment between the J-Lens linear projection matrix and the actual dynamically computed derivatives at the transformer residuum.

* **`causal_patching_24.py` & `causal_patching.py`**
  Implements activation-level interventions at specific injection points (primarily $L_{14}$). It performs continuous interpolations over an $\alpha$-vector space `[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]`, substituting cleanly ablated states to trace non-linear propagation of semantics through subsequent layers.

* **`random_direction_control.py`**
  A critical falsification mechanism. By intervening using uniformly sampled random directions restricted to the exact same $L_2$ norm as the principal J-Lens projection vectors, this ensures that any observed causal changes in decoding probabilities are not mere artifacts of vector magnitude disruption, thereby securing the rigor of the "causal headline."

* **`compute_n7_corr.py` & `get_all_ll_raw.py`**
  Extracts the raw $LL$ (Logit Lens) baseline directly from intermediate residual streams (baseline correlation $r \approx 0.191$). These baseline extraction routines establish the grounding against which the Jacobian Lens's accuracy improvements are benchmarked.

* **`generate_plots_from_data.py` & `plot_patching_final.py`**
  Responsible for translating complex higher-dimensional divergence matrices, patching sweeps, and categorical metrics into rigorous visual telemetry (e.g., Figures 1-5).

## Methodology & Execution

1. **Environment Definition**: The evaluations operate inherently against the `Qwen3-1.7B` structure, assuming access to standard `transformers` APIs and the bespoke `jlens` fitting framework.
2. **Setup**: Necessary lens weights and models (such as `Qwen3-1.7B_jacobian_lens.pt`) should be provisioned into the `model/` and `lens_weights/` directories respectively (kept completely outside Git version control via structural `.gitignore` enforcement).
3. **Execution**: To reproduce core analytical results, execute the causal patching scripts followed by standard correlative analysis checks.
4. **Validation**: Rely on the random direction control as the baseline invariant. Valid interventions under J-Lens will significantly outpace random vectors bounded under identically constrained activation norms.

## Context

This rigorous evaluation infrastructure is designed specifically to ensure that the representational readouts derived from intermediate network spaces are not hallucinated statistical correlates, but true, causally potent vectors orchestrating next-token generation.
