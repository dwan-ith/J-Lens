"""Compute n=7 correlation with consistent data."""
import os, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = r"C:\Users\aacer\Documents\J-Space\results"

key_layers = [4, 9, 14, 19, 24, 26]

# JL advantage data (from crossover analysis, consistent prompts from prompts.py)
adv_list = {
    "in_distribution_generic": [-0.430, 0.018, 0.891, 0.717, 0.369, -0.356],
    "code_text": [0.311, 0.320, 0.720, 0.416, 0.338, -0.192],
    "ambiguous_sentences": [-0.082, 0.033, 1.119, 0.912, 0.701, -0.036],
    "eval_awareness": [-0.000, 0.398, 0.457, 0.352, 0.441, 0.285],
    "math_cot": [-0.315, 0.652, 1.084, 0.859, 0.613, 0.281],
    "deception_scenarios": [-0.211, 0.265, 0.944, 0.881, 0.593, 0.162],
    "multi_turn_agentic": [-0.110, 0.269, 0.909, 0.767, 0.538, 0.292],
}

# LL_raw at L14 (from get_all_ll_raw.py, consistent prompts from prompts.py)
ll_at_14 = {
    "in_distribution_generic": -0.1134,
    "code_text": -0.0713,
    "ambiguous_sentences": -0.2940,
    "eval_awareness": -0.2475,
    "deception_scenarios": -0.0572,
    "math_cot": -0.0050,
    "multi_turn_agentic": +0.2108,
}

# JL advantage at L14
jl_adv_at_14 = {cat: advs[2] for cat, advs in adv_list.items()}  # index 2 = L14

# Compute n=7 correlation
cats = sorted(ll_at_14.keys())
ll_vals = [ll_at_14[c] for c in cats]
jl_vals = [jl_adv_at_14[c] for c in cats]

corr = np.corrcoef(ll_vals, jl_vals)[0, 1]
print(f"n=7 correlation between LL_raw@L14 and JL advantage@L14: r = {corr:.3f}")
print()
print("Data:")
for c in cats:
    print(f"  {c:<25} LL_raw={ll_at_14[c]:+.4f}  JL_adv={jl_adv_at_14[c]:+.3f}")

# Also compute n=7 correlation for crossover (does LL_raw predict whether advantage persists?)
# Categories that maintain advantage through L26: advantage at L26 > 0.1
maintains = {c: (adv_list[c][5] > 0.1) for c in cats}
print("\nCrossover (advantage at L26 > 0.1):")
for c in cats:
    print(f"  {c:<25} L26_adv={adv_list[c][5]:+.3f}  {'maintains' if maintains[c] else 'crosses'}")

# Point-biserial correlation: LL_raw vs maintains
from scipy.stats import pointbiserialr
maintains_binary = [1 if maintains[c] else 0 for c in cats]
pb_corr, pb_p = pointbiserialr(ll_vals, maintains_binary)
print(f"\nPoint-biserial correlation (LL_raw vs maintains): r = {pb_corr:.3f}, p = {pb_p:.3f}")
