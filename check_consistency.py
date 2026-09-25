"""Compute correlation and check peak layers for the consistency pass."""
import os, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")
import numpy as np
import json

RESULTS_DIR = r"C:\Users\aacer\Documents\J-Space\results"

key_layers = [4, 9, 14, 19, 24, 26]

# All data: advantage over raw logit-lens (JL_raw - LL_raw)
adv_list = {
    "in_distribution_generic": [-0.430, 0.018, 0.891, 0.717, 0.369, -0.356],
    "code_text": [0.311, 0.320, 0.720, 0.416, 0.338, -0.192],
    "ambiguous_sentences": [-0.082, 0.033, 1.119, 0.912, 0.701, -0.036],
    "eval_awareness": [-0.000, 0.398, 0.457, 0.352, 0.441, 0.285],
    "math_cot": [-0.315, 0.652, 1.084, 0.859, 0.613, 0.281],
    "deception_scenarios": [-0.211, 0.265, 0.944, 0.881, 0.593, 0.162],
    "multi_turn_agentic": [-0.110, 0.269, 0.909, 0.767, 0.538, 0.292],
}

# LL_raw values at L14 (from crossover data)
# These are the "raw logit-lens baseline fidelity" at the key layer
with open(os.path.join(RESULTS_DIR, "crossover_remaining.json")) as f:
    remaining = json.load(f)

# Build LL_raw at L14 for all 7 categories
ll_at_14 = {}
for cat in remaining:
    ll_at_14[cat] = remaining[cat]["14"]["ll_raw"]

# For the first 3 categories, I need to extract LL_raw from the crossover log
# From the crossover.log output, the first run printed JL advantage = JL_raw - LL_raw
# But I don't have the individual values. Let me use the advantage + JL to back-compute.
# Actually, I can compute JL_raw from the data I have.

# From the crossover analysis (first run), at L14:
# in_distribution_generic: JL advantage = +0.891
# code_text: JL advantage = +0.720
# ambiguous_sentences: JL advantage = +1.119

# I need JL_raw at L14 for these. Let me check if I have it elsewhere.
# Actually, looking at the crossover.log, it printed per-prompt JL_raw and LL_raw.
# But I only saved the summary. Let me just use what I have.

# For the remaining 4 categories, I have both JL_raw and LL_raw at L14:
# eval_awareness: JL_raw=0.515, LL_raw=0.059
# math_cot: JL_raw=0.719, LL_raw=-0.365
# deception_scenarios: JL_raw=0.619, LL_raw=-0.326
# multi_turn_agentic: JL_raw=0.642, LL_raw=-0.268

# For the first 3, I only have advantage. Let me estimate LL_raw.
# From the readout_partial.json, the old data had norm applied.
# I'll use the advantage + a rough JL estimate.

# Actually, let me just compute the correlation using what I have.
# The key question: is there a negative correlation between LL_raw fidelity and JL advantage?

# Categories with both values:
cats_with_both = {
    "eval_awareness": {"ll_raw": 0.059, "jl_adv": 0.457},
    "math_cot": {"ll_raw": -0.365, "jl_adv": 1.084},
    "deception_scenarios": {"ll_raw": -0.326, "jl_adv": 0.944},
    "multi_turn_agentic": {"ll_raw": -0.268, "jl_adv": 0.909},
}

# For the first 3, I'll estimate LL_raw from the advantage + JL_raw
# From the crossover.log, at L14 for in_distribution_generic:
# The per-prompt data shows JL_raw ~ 0.7-0.8, LL_raw ~ -0.3 to -0.5
# Let me use the summary from the first run.

# Actually, let me just use the 4 categories I have complete data for.
# Then check if the pattern holds.

ll_vals = [cats_with_both[c]["ll_raw"] for c in cats_with_both]
jl_adv_vals = [cats_with_both[c]["jl_adv"] for c in cats_with_both]

corr = np.corrcoef(ll_vals, jl_adv_vals)[0, 1]
print(f"Correlation between LL_raw@L14 and JL advantage@L14 (n=4): {corr:.3f}")

# Also check: what if I include approximate values for the first 3?
# From the crossover.log, the first run at L14:
# in_distribution_generic: JL advantage = 0.891
#   - Looking at the readout_partial.json (old data with norm), JL was ~0.72
#   - But that's with norm, not raw. I don't have the raw values.
# Let me just report the n=4 correlation.

print(f"\nLL_raw@L14 values: {ll_vals}")
print(f"JL advantage@L14 values: {jl_adv_vals}")

# Check peak layers
print("\nPeak layers per category:")
for cat, advs in adv_list.items():
    peak_idx = advs.index(max(advs))
    print(f"  {cat:<25} peak at L{key_layers[peak_idx]} (advantage={advs[peak_idx]:+.3f})")

# Check if the peak is exactly L14 or spread
peak_layers = [key_layers[advs.index(max(advs))] for advs in adv_list.values()]
print(f"\nAll peak layers: {peak_layers}")
print(f"Peak at L14: {peak_layers.count(14)} categories")
print(f"Peak at L19: {peak_layers.count(19)} categories")
