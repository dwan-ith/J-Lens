"""Generate patching sweep figure."""
import os, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

RESULTS_DIR = r"C:\Users\aacer\Documents\J-Space\results"

with open(os.path.join(RESULTS_DIR, "patching_sweep.json")) as f:
    results = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

colors = {"high_div_1": "steelblue", "high_div_2": "darkorange", "high_div_3": "green",
          "high_div_4": "red", "low_div_1": "purple", "low_div_2": "brown"}

# Panel 1: Top-5 overlap
ax = axes[0]
for r in results:
    ax.plot(r["alphas"], r["top5_overlaps"], "o-", color=colors[r["name"]], label=r["name"], linewidth=1.5)
ax.set_xlabel("Alpha (fraction of J-Lens direction injected)")
ax.set_ylabel("Top-5 overlap (Jaccard)")
ax.set_title("Prediction stability under patching")
ax.legend(fontsize=7, loc="lower left")
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)

# Panel 2: Probability shift (L1)
ax = axes[1]
for r in results:
    ax.plot(r["alphas"], r["prob_shifts"], "o-", color=colors[r["name"]], label=r["name"], linewidth=1.5)
ax.set_xlabel("Alpha")
ax.set_ylabel("L1 probability shift")
ax.set_title("Distribution change under patching")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Panel 3: KL divergence
ax = axes[2]
for r in results:
    ax.plot(r["alphas"], r["kl_divs"], "o-", color=colors[r["name"]], label=r["name"], linewidth=1.5)
ax.set_xlabel("Alpha")
ax.set_ylabel("KL(clean || patched)")
ax.set_title("KL divergence under patching")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig5_patching_sweep.png"), dpi=150, bbox_inches="tight")
print("Saved fig5_patching_sweep.png")
plt.close()

# Print summary table
print("\nCausal patching summary (alpha=0.3)")
print("=" * 70)
print(f"{'Prompt':<12} {'Flip?':<8} {'Overlap':<10} {'P-shift':<10} {'KL':<10}")
print("-" * 70)
for r in results:
    i = r["alphas"].index(0.3)
    flip = "YES" if r["top1_flips"][i] else "no"
    print(f"{r['name']:<12} {flip:<8} {r['top5_overlaps'][i]:<10.2f} "
          f"{r['prob_shifts'][i]:<10.4f} {r['kl_divs'][i]:<10.2f}")

flips_at_03 = sum(1 for r in results if r["top1_flips"][r["alphas"].index(0.3)])
print(f"\n{flips_at_3}/6 prompts flip at alpha=0.3")
