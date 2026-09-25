"""Fix the crossover plot: find where advantage drops below 0.1 AFTER the peak."""
import os, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = r"C:\Users\aacer\Documents\J-Space\results"

key_layers = [4, 9, 14, 19, 24, 26]

advantage_matrix = {
    "in_distribution_generic": {-0.430, 0.018, 0.891, 0.717, 0.369, -0.356},
    "code_text": {0.311, 0.320, 0.720, 0.416, 0.338, -0.192},
    "ambiguous_sentences": {-0.082, 0.033, 1.119, 0.912, 0.701, -0.036},
    "eval_awareness": {-0.000, 0.398, 0.457, 0.352, 0.441, 0.285},
    "math_cot": {-0.315, 0.652, 1.084, 0.859, 0.613, 0.281},
    "deception_scenarios": {-0.211, 0.265, 0.944, 0.881, 0.593, 0.162},
    "multi_turn_agentic": {-0.110, 0.269, 0.909, 0.767, 0.538, 0.292},
}

# Fix: need lists not sets
adv_list = {
    "in_distribution_generic": [-0.430, 0.018, 0.891, 0.717, 0.369, -0.356],
    "code_text": [0.311, 0.320, 0.720, 0.416, 0.338, -0.192],
    "ambiguous_sentences": [-0.082, 0.033, 1.119, 0.912, 0.701, -0.036],
    "eval_awareness": [-0.000, 0.398, 0.457, 0.352, 0.441, 0.285],
    "math_cot": [-0.315, 0.652, 1.084, 0.859, 0.613, 0.281],
    "deception_scenarios": [-0.211, 0.265, 0.944, 0.881, 0.593, 0.162],
    "multi_turn_agentic": [-0.110, 0.269, 0.909, 0.767, 0.538, 0.292],
}

# Find crossover: where advantage drops below 0.1 AFTER the peak layer
def find_crossover(advs, layers):
    peak_idx = advs.index(max(advs))
    for i in range(peak_idx + 1, len(advs)):
        if advs[i] < 0.1:
            return layers[i]
    return None  # never drops below 0.1

crossover_points = {}
for cat, advs in adv_list.items():
    cp = find_crossover(advs, key_layers)
    crossover_points[cat] = cp
    print(f"  {cat:<25} peak at L{key_layers[advs.index(max(advs))]}, crossover at {'L'+str(cp) if cp else '>L26'}")

# Plot
fig, ax = plt.subplots(figsize=(8, 4))
cats_sorted = sorted(crossover_points.keys(), key=lambda c: (crossover_points[c] is None, crossover_points[c] or 0))
y_pos = range(len(cats_sorted))
bar_vals = [crossover_points[c] if crossover_points[c] else 28 for c in cats_sorted]
bar_colors = ["steelblue" if crossover_points[c] else "darkgreen" for c in cats_sorted]

ax.barh(list(y_pos), bar_vals, color=bar_colors)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(cats_sorted)
ax.set_xlabel("Layer where J-Lens advantage drops below 0.1 (after peak)")
ax.set_title("Crossover point by category (where advantage vanishes)")
ax.set_xticks(key_layers + [28])
ax.set_xticklabels([str(l) for l in key_layers] + [">26"])
ax.grid(True, alpha=0.3, axis="x")
ax.axvline(14, color="red", linestyle=":", alpha=0.5, label="Peak (L14)")
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig4_crossover_summary.png"), dpi=150, bbox_inches="tight")
print("\nSaved fig4_crossover_summary.png")
plt.close()
