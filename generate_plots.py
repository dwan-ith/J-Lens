"""Generate corrected plots with identity baseline and no-norm logit-lens."""
import sys, warnings, os
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")
import torch
import numpy as np
import transformers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

sys.path.insert(0, r"C:\Users\aacer\Documents\J-Space\jacobian-lens")
import jlens
from jlens.hooks import ActivationRecorder
from prompts import ALL_CATEGORIES

MODEL_DIR = r"C:\Users\aacer\Documents\J-Space\model"
LENS_PATH = r"C:\Users\aacer\Documents\J-Space\lens_weights\qwen3-1.7b\jlens\Salesforce-wikitext\Qwen3-1.7B_jacobian_lens.pt"

model = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, dtype=torch.float32, device_map="cpu", local_files_only=True)
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model.eval()
global_lens = jlens.JacobianLens.load(LENS_PATH)

d_model = model.config.hidden_size
text_module = model.model
block_layers = text_module.layers
W_U = model.lm_head.weight.data.float()

key_layers = [4, 9, 14, 19, 24, 26]
N = 2  # prompts per category

all_data = {}
for cat_name, prompts in ALL_CATEGORIES.items():
    print(f"Computing: {cat_name}", flush=True)
    layer_data = {l: {"jl": [], "ll": [], "ident": []} for l in key_layers}
    
    for prompt in prompts[:N]:
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=96)
        input_ids = encoded.input_ids
        
        with torch.no_grad():
            logits_actual = model(input_ids).logits[0, -1, :].float()
        
        for layer in key_layers:
            Jg = global_lens.jacobians[layer].float()
            
            with ActivationRecorder(block_layers, at=[layer]) as recorder:
                text_module(input_ids=input_ids, use_cache=False)
                h = recorder.activations[layer][0, -1, :].float()
            
            with torch.no_grad():
                jl_cos = torch.nn.functional.cosine_similarity(W_U @ (Jg @ h), logits_actual, dim=0).item()
                ll_cos = torch.nn.functional.cosine_similarity(W_U @ h, logits_actual, dim=0).item()
                # Identity baseline: cos between identity and J_global
                ident_cos = torch.nn.functional.cosine_similarity(
                    torch.eye(d_model).flatten(), Jg.flatten(), dim=0).item()
            
            layer_data[layer]["jl"].append(jl_cos)
            layer_data[layer]["ll"].append(ll_cos)
            layer_data[layer]["ident"].append(ident_cos)
    
    all_data[cat_name] = {
        l: {
            "jl": float(np.mean(layer_data[l]["jl"])),
            "ll": float(np.mean(layer_data[l]["ll"])),
            "ident": float(np.mean(layer_data[l]["ident"])),
        } for l in key_layers
    }

# ===== PLOT 1: Main finding — J-Lens vs LL_raw vs Identity by layer =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1a: J-Lens vs Logit-Lens (raw, no norm)
ax = axes[0]
for cat, data in all_data.items():
    xs = key_layers
    ys = [data[l]["jl"] for l in key_layers]
    ax.plot(xs, ys, "o-", label=f"J-Lens", alpha=0.3, linewidth=1)

# Add mean
mean_jl = [np.mean([all_data[c][l]["jl"] for c in all_data]) for l in key_layers]
ax.plot(key_layers, mean_jl, "ko-", linewidth=3, label="J-Lens (mean)", zorder=10)

mean_ll = [np.mean([all_data[c][l]["ll"] for c in all_data]) for l in key_layers]
ax.plot(key_layers, mean_ll, "s--", color="gray", linewidth=2, label="Logit-Lens raw (mean)")

ax.set_xlabel("Layer")
ax.set_ylabel("Cosine with actual logits")
ax.set_title("J-Lens vs Raw Logit-Lens (no norm)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(0, color="red", linestyle=":", alpha=0.5)

# 1b: Identity baseline
ax = axes[1]
mean_ident = [np.mean([all_data[c][l]["ident"] for c in all_data]) for l in key_layers]
ax.plot(key_layers, mean_ident, "s--", color="coral", linewidth=2, label="Identity baseline")
ax.plot(key_layers, mean_jl, "ko-", linewidth=3, label="J-Lens (mean)", zorder=10)

ax.set_xlabel("Layer")
ax.set_ylabel("Cosine with actual logits / sim(I, J_global)")
ax.set_title("J-Lens vs Identity Baseline (proximity artifact check)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(0, color="red", linestyle=":", alpha=0.5)

plt.tight_layout()
plt.savefig("results/fig1_corrected.png", dpi=150, bbox_inches="tight")
print("Saved fig1_corrected.png")
plt.close()

# ===== PLOT 2: Crossover by category =====
fig, ax = plt.subplots(figsize=(10, 5))
colors = plt.cm.Set2(np.linspace(0, 1, len(all_data)))
for i, (cat, data) in enumerate(all_data.items()):
    advs = [data[l]["jl"] - data[l]["ll"] for l in key_layers]
    ax.plot(key_layers, advs, "o-", color=colors[i], label=cat, linewidth=1.5)

ax.set_xlabel("Layer")
ax.set_ylabel("J-Lens advantage over raw Logit-Lens")
ax.set_title("Crossover: Where does J-Lens advantage vanish?")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3)
ax.axhline(0, color="red", linestyle=":", alpha=0.5)

plt.tight_layout()
plt.savefig("results/fig2_crossover.png", dpi=150, bbox_inches="tight")
print("Saved fig2_crossover.png")
plt.close()

# ===== PLOT 3: eval_awareness is best-approximated =====
fig, ax = plt.subplots(figsize=(10, 5))
cats_sorted = sorted(all_data.keys(), key=lambda c: all_data[c][14]["jl"], reverse=True)
y_positions = range(len(cats_sorted))
jl_at_14 = [all_data[c][14]["jl"] for c in cats_sorted]
ll_at_14 = [all_data[c][14]["ll"] for c in cats_sorted]

ax.barh([y + 0.15 for y in y_positions], jl_at_14, height=0.3, label="J-Lens", color="steelblue")
ax.barh([y - 0.15 for y in y_positions], ll_at_14, height=0.3, label="Logit-Lens raw", color="gray")

ax.set_yticks(list(y_positions))
ax.set_yticklabels(cats_sorted)
ax.set_xlabel("Cosine with actual logits at L14")
ax.set_title("Category ranking at L14 (n=2 per category)")
ax.legend()
ax.grid(True, alpha=0.3, axis="x")
ax.axvline(0, color="red", linestyle=":", alpha=0.5)

# Add n labels
for y, c in zip(y_positions, cats_sorted):
    ax.text(max(jl_at_14[y_positions.index(y)], ll_at_14[y_positions.index(y)]) + 0.05, y, "n=2", va="center", fontsize=8, color="gray")

plt.tight_layout()
plt.savefig("results/fig3_category_ranking.png", dpi=150, bbox_inches="tight")
print("Saved fig3_category_ranking.png")
plt.close()

# Save data
with open("results/corrected_summary.json", "w") as f:
    json.dump(all_data, f, indent=2)
print("Saved corrected_summary.json")
print("Done!")
