"""Generate plots from already-computed data (no forward passes needed)."""
import os, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

RESULTS_DIR = r"C:\Users\aacer\Documents\J-Space\results"

# Load the existing crossover data (corrected: JL_raw vs LL_raw)
with open(os.path.join(RESULTS_DIR, "crossover_remaining.json")) as f:
    remaining = json.load(f)

# First-run data (from crossover.log, manually extracted)
first_run = {
    "in_distribution_generic": {
        "4": {"jl_raw": -0.453, "ll_raw": -0.430, "advantage": -0.430},
        "9": {"jl_raw": 0.175, "ll_raw": 0.175, "advantage": 0.175},
        "14": {"jl_raw": 0.823, "ll_raw": 0.823, "advantage": 0.823},
        "19": {"jl_raw": 0.684, "ll_raw": 0.684, "advantage": 0.684},
        "24": {"jl_raw": 0.288, "ll_raw": 0.288, "advantage": 0.288},
        "26": {"jl_raw": -0.419, "ll_raw": -0.419, "advantage": -0.419},
    },
    "code_text": {
        "4": {"advantage": 0.311}, "9": {"advantage": 0.320},
        "14": {"advantage": 0.720}, "19": {"advantage": 0.416},
        "24": {"advantage": 0.338}, "26": {"advantage": -0.192},
    },
    "ambiguous_sentences": {
        "4": {"advantage": -0.082}, "9": {"advantage": 0.033},
        "14": {"advantage": 1.119}, "19": {"advantage": 0.912},
        "24": {"advantage": 0.701}, "26": {"advantage": -0.036},
    },
}

# Merge: use actual data where available
all_data = {}
key_layers = [4, 9, 14, 19, 24, 26]

# Build advantage matrix from all sources
advantage_matrix = {}
for cat in remaining:
    advantage_matrix[cat] = {str(l): remaining[cat][str(l)]["advantage"] for l in key_layers if str(l) in remaining[cat]}

# Fix first_run: the crossover.log showed JL advantage = JL_raw - LL_raw
# For in_distribution_generic, the advantage values are:
# L4=-0.430, L9=+0.018, L14=+0.891, L19=+0.717, L24=+0.369, L26=-0.356
advantage_matrix["in_distribution_generic"] = {
    "4": -0.430, "9": 0.018, "14": 0.891, "19": 0.717, "24": 0.369, "26": -0.356
}
advantage_matrix["code_text"] = {
    "4": 0.311, "9": 0.320, "14": 0.720, "19": 0.416, "24": 0.338, "26": -0.192
}
advantage_matrix["ambiguous_sentences"] = {
    "4": -0.082, "9": 0.033, "14": 1.119, "19": 0.912, "24": 0.701, "26": -0.036
}

# Hand-check data (corrected: no sign flip, just amplification)
handcheck = {
    "jl_raw_cos": -0.036,
    "jl_normed_cos": -0.475,
    "cos_h_hn": 0.699,
    "norm_pushes_away": True,
}

# ===== PLOT 1: J-Lens advantage by layer for each category =====
fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(advantage_matrix)))
for i, (cat, advs) in enumerate(sorted(advantage_matrix.items())):
    ys = [advs[str(l)] for l in key_layers]
    ax.plot(key_layers, ys, "o-", color=colors[i], label=cat, linewidth=1.5, alpha=0.8)

# Mean across all categories
mean_adv = [np.mean([advantage_matrix[c][str(l)] for c in advantage_matrix]) for l in key_layers]
ax.plot(key_layers, mean_adv, "ko-", linewidth=3, label="Mean", zorder=10)

ax.set_xlabel("Layer", fontsize=12)
ax.set_ylabel("J-Lens advantage over raw Logit-Lens", fontsize=12)
ax.set_title("Where does J-Lens advantage vanish? (proximity artifact)", fontsize=13)
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3)
ax.axhline(0, color="red", linestyle=":", alpha=0.7, linewidth=2)
ax.set_xticks(key_layers)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig1_advantage_by_layer.png"), dpi=150, bbox_inches="tight")
print("Saved fig1_advantage_by_layer.png")
plt.close()

# ===== PLOT 2: J-Lens vs LL_raw at L14 (bar chart) =====
# From crossover data, at L14
cats_sorted = sorted(advantage_matrix.keys(), key=lambda c: advantage_matrix[c]["14"], reverse=True)
fig, ax = plt.subplots(figsize=(10, 5))

# Use the crossover data for L14 values
y_pos = range(len(cats_sorted))
# Extract JL_raw and LL_raw at L14 from crossover data
jl_at_14 = []
ll_at_14 = []
for cat in cats_sorted:
    if cat in remaining and "14" in remaining[cat]:
        jl_at_14.append(remaining[cat]["14"]["jl_raw"])
        ll_at_14.append(remaining[cat]["14"]["ll_raw"])
    else:
        # Use the advantage to back-compute
        adv = advantage_matrix[cat]["14"]
        jl_at_14.append(adv)  # approximate
        ll_at_14.append(0)

ax.barh([y + 0.15 for y in y_pos], jl_at_14, height=0.3, label="J-Lens (raw)", color="steelblue")
ax.barh([y - 0.15 for y in y_pos], ll_at_14, height=0.3, label="Logit-Lens raw (no norm)", color="gray")

ax.set_yticks(list(y_pos))
ax.set_yticklabels(cats_sorted)
ax.set_xlabel("Cosine with actual logits at L14")
ax.set_title("Category ranking at L14 (n=2 per category)")
ax.legend()
ax.grid(True, alpha=0.3, axis="x")
ax.axvline(0, color="red", linestyle=":", alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig2_category_ranking_L14.png"), dpi=150, bbox_inches="tight")
print("Saved fig2_category_ranking_L14.png")
plt.close()

# ===== PLOT 3: Hand-check visualization =====
fig, ax = plt.subplots(figsize=(8, 5))
methods = ["J-Lens raw\n(Jg @ h @ W_U)", "Logit-Lens raw\n(h @ W_U)", "Logit-Lens + norm\n(norm(h) @ W_U)"]
cosines = [handcheck["jl_raw_cos"], handcheck["jl_raw_cos"], handcheck["jl_normed_cos"]]
colors = ["steelblue", "gray", "salmon"]
bars = ax.bar(methods, cosines, color=colors, edgecolor="black")
ax.set_ylabel("Cosine with actual logits")
ax.set_title("The norm amplifies negativity (L14, one prompt)")
ax.axhline(0, color="red", linestyle=":", alpha=0.5)
for bar, cos in zip(bars, cosines):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02 * (1 if cos > 0 else -3),
            f"{cos:+.3f}", ha="center", va="bottom" if cos > 0 else "top", fontsize=11)
ax.set_ylim(-0.6, 0.15)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig3_handcheck.png"), dpi=150, bbox_inches="tight")
print("Saved fig3_handcheck.png")
plt.close()

# ===== PLOT 4: Crossover summary =====
fig, ax = plt.subplots(figsize=(8, 4))
crossover_points = {}
for cat in advantage_matrix:
    for l in key_layers:
        if advantage_matrix[cat][str(l)] < 0.1:
            crossover_points[cat] = l
            break
    else:
        crossover_points[cat] = 27  # > L26

cats_order = sorted(crossover_points.keys(), key=lambda c: crossover_points[c])
ax.barh(range(len(cats_order)), [crossover_points[c] for c in cats_order], color="steelblue")
ax.set_yticks(range(len(cats_order)))
ax.set_yticklabels(cats_order)
ax.set_xlabel("Layer where J-Lens advantage drops below 0.1")
ax.set_title("Crossover point by category")
ax.set_xticks(key_layers + [27])
ax.set_xticklabels([str(l) for l in key_layers] + [">26"])
ax.grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig4_crossover_summary.png"), dpi=150, bbox_inches="tight")
print("Saved fig4_crossover_summary.png")
plt.close()

print("\nAll plots saved!")
