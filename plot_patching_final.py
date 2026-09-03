"""Generate final patching figure: J-Lens vs random direction control."""
import os, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

RESULTS_DIR = r"C:\Users\aacer\Desktop\MATS\results"

with open(os.path.join(RESULTS_DIR, "patching_24.json")) as f:
    jl_results = json.load(f)
with open(os.path.join(RESULTS_DIR, "random_direction_control.json")) as f:
    rand_results = json.load(f)

alphas = jl_results[0]["alphas"]

# J-Lens flip rates
jl_flip_rates = []
for ai in range(len(alphas)):
    rate = sum(1 for r in jl_results if r["top1_flips"][ai]) / len(jl_results)
    jl_flip_rates.append(rate)

# Random direction flip rates (already averaged over 3 seeds per prompt)
rand_flip_rates = []
for ai in range(len(alphas)):
    rate = sum(r["avg_flip_rates"][ai] for r in rand_results) / len(rand_results)
    rand_flip_rates.append(rate)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1: J-Lens vs Random (the key result)
ax = axes[0]
ax.plot(alphas, jl_flip_rates, "ko-", linewidth=2.5, markersize=10, zorder=10, label="J-Lens direction")
ax.plot(alphas, rand_flip_rates, "rs--", linewidth=2.5, markersize=10, zorder=10, label="Random direction (norm-matched)")
ax.set_xlabel("Alpha (fraction of direction injected)", fontsize=12)
ax.set_ylabel("Fraction of prompts with top-1 flip", fontsize=12)
ax.set_title("Direction specificity test", fontsize=13)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)
ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)
ax.legend(fontsize=11, loc="upper left")
for a, j, r in zip(alphas, jl_flip_rates, rand_flip_rates):
    ax.annotate(f"{j:.0%}", (a, j), textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=9, fontweight="bold", color="black")
    ax.annotate(f"{r:.0%}", (a, r), textcoords="offset points", xytext=(0, -18),
                ha="center", fontsize=9, fontweight="bold", color="darkred")
ax.text(0.5, 0.02, "No difference = not J-Lens-specific",
        transform=ax.transAxes, ha="center", fontsize=10, fontstyle="italic", color="gray")

# Panel 2: By category (J-Lens at alpha=0.3)
ax = axes[1]
cat_names = ["in_dist", "code", "ambig", "eval", "safety", "math"]
cat_flips = []
for ci in range(6):
    start = ci * 4
    n_flip = sum(1 for r in jl_results[start:start+4] if r["top1_flips"][3])  # alpha=0.3
    cat_flips.append(n_flip / 4)
colors = plt.cm.Set2(np.linspace(0, 1, 6))
bars = ax.bar(cat_names, cat_flips, color=colors, edgecolor="black")
for bar, n in zip(bars, [f"{int(f*4)}/4" for f in cat_flips]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            n, ha="center", va="bottom", fontsize=11)
ax.set_ylabel("Fraction flipping at alpha=0.3")
ax.set_title("Flip rate by category (J-Lens)")
ax.set_ylim(0, 1.2)
ax.grid(True, alpha=0.3, axis="y")

# Panel 3: Difference (J-Lens minus random)
ax = axes[2]
diffs = [j - r for j, r in zip(jl_flip_rates, rand_flip_rates)]
colors_diff = ["steelblue" if d >= 0 else "darkred" for d in diffs]
ax.bar([f"{a:.1f}" for a in alphas], diffs, color=colors_diff, edgecolor="black")
ax.axhline(0, color="black", linewidth=1)
ax.set_xlabel("Alpha")
ax.set_ylabel("Flip rate difference (J-Lens - Random)")
ax.set_title("Difference: J-Lens vs Random")
ax.grid(True, alpha=0.3, axis="y")
ax.text(0.5, 0.02, "Within noise = no evidence for J-Lens specificity",
        transform=ax.transAxes, ha="center", fontsize=10, fontstyle="italic", color="gray")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig5_patching_sweep.png"), dpi=150, bbox_inches="tight")
print("Saved fig5_patching_sweep.png")
plt.close()

print(f"\nn = {len(jl_results)} prompts, 3 random seeds per prompt")
print(f"\nAlpha   J-Lens   Random   Diff")
print("-" * 40)
for a, j, r, d in zip(alphas, jl_flip_rates, rand_flip_rates, diffs):
    print(f"  {a:.1f}    {j:.0%}     {r:.0%}    {d:+.0%}")
