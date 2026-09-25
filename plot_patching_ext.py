"""Generate extended patching figure."""
import os, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

RESULTS_DIR = r"C:\Users\aacer\Documents\J-Space\results"

# Load both datasets
with open(os.path.join(RESULTS_DIR, "patching_sweep.json")) as f:
    orig = json.load(f)
with open(os.path.join(RESULTS_DIR, "patching_extended.json")) as f:
    ext = json.load(f)

# Merge (use extended which includes originals)
all_results = {r["name"]: r for r in ext}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

colors = plt.cm.tab20(np.linspace(0, 1, 12))

# Panel 1: Top-5 overlap
ax = axes[0]
for i, (name, r) in enumerate(sorted(all_results.items())):
    ax.plot(r["alphas"], r["top5_overlaps"], "o-", color=colors[i], linewidth=1.2, label=name)
ax.set_xlabel("Alpha (fraction of J-Lens direction injected)")
ax.set_ylabel("Top-5 overlap (Jaccard)")
ax.set_title("Prediction stability under patching")
ax.legend(fontsize=6, loc="lower left", ncol=2)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)

# Panel 2: Probability shift
ax = axes[1]
for i, (name, r) in enumerate(sorted(all_results.items())):
    ax.plot(r["alphas"], r["prob_shifts"], "o-", color=colors[i], linewidth=1.2, label=name)
ax.set_xlabel("Alpha")
ax.set_ylabel("L1 probability shift")
ax.set_title("Distribution change under patching")
ax.legend(fontsize=6, ncol=2)
ax.grid(True, alpha=0.3)

# Panel 3: Flip rate by alpha
ax = axes[2]
flip_rates = []
for alpha_idx in range(len(ext[0]["alphas"])):
    rate = sum(1 for r in ext if r["top1_flips"][alpha_idx]) / len(ext)
    flip_rates.append(rate)
ax.plot(ext[0]["alphas"], flip_rates, "ko-", linewidth=2, markersize=8)
ax.set_xlabel("Alpha")
ax.set_ylabel("Fraction of prompts with top-1 flip")
ax.set_title(f"Causal effect: {sum(r['top1_flips'][3] for r in ext)}/12 flip at alpha=0.3")
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)
ax.axhline(0.5, color="red", linestyle=":", alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig5_patching_sweep.png"), dpi=150, bbox_inches="tight")
print("Saved fig5_patching_sweep.png")
plt.close()

# Print summary
print("\nFlip rates:")
for alpha, rate in zip(ext[0]["alphas"], flip_rates):
    print(f"  alpha={alpha:.1f}: {rate:.0%} ({int(rate*12)}/12)")
