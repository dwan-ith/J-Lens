"""
J-Lens Local vs Global Divergence Experiment — Qwen3-1.7B
==========================================================
Does the globally-averaged J-Lens Jacobian match what the model
actually computes locally on individual forward passes?

Metrics:
  A) Readout comparison: J-lens vs Logit-lens vs actual model logits
  B) Sampled Jacobian: K random d_model dims per prompt, local vs global
  C) Causal patching: does J-lens transport change model predictions?

H1: Faithful local approximation
H2: Global-average artifact, worse for safety-relevant inputs
H3: Divergence predicts causal patching failure
"""

import os, sys, json, time, gc, warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, r"C:\Users\aacer\Documents\J-Space\jacobian-lens")
import jlens
from jlens.hooks import ActivationRecorder
from prompts import ALL_CATEGORIES

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_DIR = r"C:\Users\aacer\Documents\J-Space\model"
LENS_PATH = r"C:\Users\aacer\Documents\J-Space\lens_weights\qwen3-1.7b\jlens\Salesforce-wikitext\Qwen3-1.7B_jacobian_lens.pt"
OUTPUT_DIR = Path(r"C:\Users\aacer\Documents\J-Space\results")
OUTPUT_DIR.mkdir(exist_ok=True)

N_PROMPTS_PER_CAT = 5
MAX_SEQ_LEN = 96
SKIP_FIRST = 16
K_DIMS_JAC = 10
K_JAC_PROMPTS = 2
TARGET_LAYER = 27
ANALYSIS_LAYERS = [4, 9, 14, 19, 24]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def cosine_sim(a, b):
    a_f, b_f = a.float().flatten(), b.float().flatten()
    n1, n2 = torch.norm(a_f), torch.norm(b_f)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    return (torch.dot(a_f, b_f) / (n1 * n2)).item()


def free_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─── Jacobian Sampling (one-at-a-time, K_DIMS_JAC rows) ────────────────────

def sample_local_jacobian(model, tokenizer, prompt, layer, target_layer,
                           k_dims=K_DIMS_JAC):
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=MAX_SEQ_LEN)
    input_ids = encoded.input_ids
    seq_len = input_ids.shape[1]
    if seq_len <= SKIP_FIRST + 1:
        raise ValueError(f"Prompt too short: {seq_len}")

    d_model = model.config.hidden_size
    text_module = model.model
    block_layers = text_module.layers

    pos_mask = torch.zeros(seq_len, dtype=torch.bool)
    pos_mask[SKIP_FIRST:seq_len - 1] = True
    valid_pos = pos_mask.nonzero(as_tuple=True)[0]

    dim_indices = torch.randperm(d_model)[:k_dims]
    local_rows = torch.zeros(k_dims, d_model, dtype=torch.float32)

    for i in range(k_dims):
        with ActivationRecorder(block_layers, at=[layer, target_layer],
                                start_graph_at=layer) as recorder:
            text_module(input_ids=input_ids, use_cache=False)
            target_act = recorder.activations[target_layer]
            source_act = recorder.activations[layer]

            cotangent = torch.zeros_like(target_act)
            cotangent[0, :, dim_indices[i]] = 1.0

            grads = torch.autograd.grad(
                outputs=target_act,
                inputs=source_act,
                grad_outputs=cotangent,
                retain_graph=False,
            )
            g = grads[0][0].float()
            local_rows[i] = g[valid_pos].mean(dim=0).cpu()
            del grads, g, cotangent, target_act, source_act
        free_mem()

    return local_rows, dim_indices, seq_len


# ─── Readout comparison (NO backward passes, very fast) ─────────────────────
# NOTE 2026-09: This function uses final RMSNorm (W_U @ norm(h)) — that distorts
# intermediate layers (see FINAL_REPORT §1, debug_handcheck.py). Headline figs use
# raw baseline W_U @ h / W_U @ (Jg@h) from crossover_remaining.json / get_all_ll_raw.py.
# Kept for provenance; use compare_readouts_raw for correct baseline.
def compare_readouts(model, tokenizer, prompt, layer, J_global):
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=MAX_SEQ_LEN)
    input_ids = encoded.input_ids
    text_module = model.model
    block_layers = text_module.layers

    with ActivationRecorder(block_layers, at=[layer]) as recorder:
        text_module(input_ids=input_ids, use_cache=False)
        h = recorder.activations[layer][0, -1, :].float().cpu()

    W_U = model.lm_head.weight.data.cpu().float()
    norm = text_module.norm

    with torch.no_grad():
        out = model(input_ids)
        logits_actual = out.logits[0, -1, :].cpu().float()

        h_normed = norm(h.unsqueeze(0)).squeeze(0)
        logits_ll = (W_U @ h_normed).float()

        transported = J_global.float() @ h
        t_normed = norm(transported.unsqueeze(0)).squeeze(0)
        logits_jl = (W_U @ t_normed).float()

    K = 15
    actual_top = set(torch.topk(logits_actual, K).indices.tolist())
    ll_top = set(torch.topk(logits_ll, K).indices.tolist())
    jl_top = set(torch.topk(logits_jl, K).indices.tolist())

    return {
        "jl_vs_actual_cos": cosine_sim(logits_jl, logits_actual),
        "ll_vs_actual_cos": cosine_sim(logits_ll, logits_actual),
        "jl_vs_ll_cos": cosine_sim(logits_jl, logits_ll),
        "actual_vs_jl_overlap": len(actual_top & jl_top) / K,
        "actual_vs_ll_overlap": len(actual_top & ll_top) / K,
        "actual_top5": [tokenizer.decode([t]) for t in torch.topk(logits_actual, 5).indices.tolist()],
        "jl_top5": [tokenizer.decode([t]) for t in torch.topk(logits_jl, 5).indices.tolist()],
        "ll_top5": [tokenizer.decode([t]) for t in torch.topk(logits_ll, 5).indices.tolist()],
    }


def compare_readouts_raw(model, tokenizer, prompt, layer, J_global):
    """Raw baseline without final RMSNorm: W_U @ h vs W_U @ (Jg@h). Use for headline."""
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=MAX_SEQ_LEN)
    input_ids = encoded.input_ids
    text_module = model.model
    block_layers = text_module.layers
    with ActivationRecorder(block_layers, at=[layer]) as recorder:
        text_module(input_ids=input_ids, use_cache=False)
        h = recorder.activations[layer][0, -1, :].float().cpu()
    W_U = model.lm_head.weight.data.cpu().float()
    with torch.no_grad():
        out = model(input_ids)
        logits_actual = out.logits[0, -1, :].cpu().float()
        logits_ll_raw = (W_U @ h).float()
        transported = J_global.float() @ h
        logits_jl_raw = (W_U @ transported).float()
    K = 15
    actual_top = set(torch.topk(logits_actual, K).indices.tolist())
    ll_top = set(torch.topk(logits_ll_raw, K).indices.tolist())
    jl_top = set(torch.topk(logits_jl_raw, K).indices.tolist())
    return {
        "jl_vs_actual_cos": cosine_sim(logits_jl_raw, logits_actual),
        "ll_vs_actual_cos": cosine_sim(logits_ll_raw, logits_actual),
        "jl_vs_ll_cos": cosine_sim(logits_jl_raw, logits_ll_raw),
        "actual_vs_jl_overlap": len(actual_top & jl_top) / K,
        "actual_vs_ll_overlap": len(actual_top & ll_top) / K,
    }


# ─── Loaders ─────────────────────────────────────────────────────────────────

def load_model():
    import transformers
    print("Loading Qwen3-1.7B...", flush=True)
    t0 = time.time()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.float32, device_map="cpu",
        local_files_only=True,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_DIR, local_files_only=True,
    )
    model.eval()
    dt = time.time() - t0
    n = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  {n:.1f}B params, d={model.config.hidden_size}, "
          f"{model.config.num_hidden_layers} layers, {dt:.1f}s", flush=True)
    return model, tokenizer


def load_lens():
    print("Loading J-Lens...", flush=True)
    lens = jlens.JacobianLens.load(LENS_PATH)
    print(f"  {lens}", flush=True)
    return lens


# ─── Experiment: Readout comparison (fast, all prompts, all layers) ──────────

def experiment_readout(model, tokenizer, global_lens, categories):
    results = {}
    for cat_name, prompts in categories.items():
        print(f"\n--- READOUT: {cat_name} ---", flush=True)
        cat = {"per_layer": defaultdict(list), "per_prompt": []}
        n = min(len(prompts), N_PROMPTS_PER_CAT)

        for i, prompt in enumerate(prompts[:n]):
            print(f"  [{i+1}/{n}] {prompt[:50]}...", end=" ", flush=True)
            t0 = time.time()
            prompt_result = {"prompt": prompt[:120], "layers": {}}

            for layer in ANALYSIS_LAYERS:
                Jg = global_lens.jacobians[layer].float()
                rc = compare_readouts(model, tokenizer, prompt, layer, Jg)
                prompt_result["layers"][layer] = rc
                cat["per_layer"][layer].append(rc)
                del Jg
                free_mem()

            dt = time.time() - t0
            jl_str = " ".join(f"L{l}={prompt_result['layers'][l]['jl_vs_actual_cos']:.3f}"
                              for l in ANALYSIS_LAYERS)
            print(f"({dt:.0f}s) JL: {jl_str}", flush=True)
            cat["per_prompt"].append(prompt_result)

        # Summarize
        cat["summary"] = {}
        for l in ANALYSIS_LAYERS:
            jl_scores = [d["jl_vs_actual_cos"] for d in cat["per_layer"][l]]
            ll_scores = [d["ll_vs_actual_cos"] for d in cat["per_layer"][l]]
            cat["summary"][l] = {
                "jl_mean": round(float(np.mean(jl_scores)), 4),
                "jl_std": round(float(np.std(jl_scores)), 4),
                "ll_mean": round(float(np.mean(ll_scores)), 4),
                "ll_std": round(float(np.std(ll_scores)), 4),
                "jl_beats_ll": round(float(np.mean(jl_scores) - np.mean(ll_scores)), 4),
            }
        results[cat_name] = cat
        try:
            def _ser(o):
                if isinstance(o, torch.Tensor): return o.tolist()
                if isinstance(o, np.ndarray): return o.tolist()
                if isinstance(o, defaultdict): return dict(o)
                if isinstance(o, dict): return {str(k): _ser(v) for k, v in o.items()}
                if isinstance(o, list): return [_ser(v) for v in o]
                return o
            with open(OUTPUT_DIR / "readout_partial.json", "w") as f:
                json.dump(_ser(results), f, indent=2)
        except Exception:
            pass
    return results


# ─── Experiment: Jacobian sampling (slow, subset of prompts) ─────────────────

def experiment_jacobian(model, tokenizer, global_lens, categories):
    d_model = model.config.hidden_size
    results = {}
    for cat_name, prompts in categories.items():
        print(f"\n--- JACOBIAN: {cat_name} ---", flush=True)
        cat = {"per_layer": defaultdict(list)}
        n = min(len(prompts), K_JAC_PROMPTS)

        for i, prompt in enumerate(prompts[:n]):
            print(f"  [{i+1}/{n}] {prompt[:50]}...", end=" ", flush=True)
            t0 = time.time()

            for layer in ANALYSIS_LAYERS:
                local_rows, dim_idx, sl = sample_local_jacobian(
                    model, tokenizer, prompt, layer, TARGET_LAYER, K_DIMS_JAC
                )
                Jg = global_lens.jacobians[layer].float()
                Jg_rows = Jg[dim_idx].float()

                overall_cos = cosine_sim(local_rows, Jg_rows)
                row_cosines = [cosine_sim(local_rows[i], Jg_rows[i])
                               for i in range(K_DIMS_JAC)]
                rand_idx = torch.randperm(d_model)[:K_DIMS_JAC]
                rand_rows = Jg[rand_idx].float()
                rand_cos = [cosine_sim(local_rows[i], rand_rows[i])
                            for i in range(K_DIMS_JAC)]

                cat["per_layer"][layer].append({
                    "overall": overall_cos,
                    "mean_row": float(np.mean(row_cosines)),
                    "mean_random": float(np.mean(rand_cos)),
                })
                del local_rows, Jg, Jg_rows, rand_rows
                free_mem()

            dt = time.time() - t0
            scores = {l: cat["per_layer"][l][-1]["overall"] for l in ANALYSIS_LAYERS}
            s = " ".join(f"L{l}={scores[l]:.3f}" for l in ANALYSIS_LAYERS)
            print(f"({dt:.0f}s) {s}", flush=True)

        cat["summary"] = {}
        for l in ANALYSIS_LAYERS:
            vals = [d["overall"] for d in cat["per_layer"][l]]
            if vals:
                cat["summary"][l] = {
                    "jac_mean": round(float(np.mean(vals)), 4),
                    "jac_std": round(float(np.std(vals)), 4),
                }
        results[cat_name] = cat
    return results


# ─── Experiment: Causal Patching ────────────────────────────────────────────

def experiment_patching(model, tokenizer, global_lens, readout_results, categories):
    """Pick 2 high-divergence and 2 low-divergence prompts and test patching."""
    all_entries = []
    for cat_name, data in readout_results.items():
        for prompt_data in data["per_prompt"]:
            for layer in ANALYSIS_LAYERS:
                r = prompt_data["layers"][layer]
                all_entries.append({
                    "prompt": prompt_data["prompt"],
                    "category": cat_name,
                    "layer": layer,
                    "divergence": 1.0 - r["jl_vs_actual_cos"],
                    "jl_cos": r["jl_vs_actual_cos"],
                })

    all_entries.sort(key=lambda x: -x["divergence"])
    extreme_cases = all_entries[:2] + all_entries[-2:]

    results = []
    for case in extreme_cases:
        prompt = categories[case["category"]][0]
        layer = case["layer"]
        Jg = global_lens.jacobians[layer].float()

        ct = "high_divergence" if case["divergence"] > 0.5 else "low_divergence"
        print(f"\n  [{ct}] div={case['divergence']:.3f} L{layer}", flush=True)

        encoded = tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=MAX_SEQ_LEN)

        with torch.no_grad():
            out_b = model(encoded.input_ids)
            top_before = torch.topk(out_b.logits[0, -1, :], 5)
            tok_before_ids = top_before.indices.tolist()

        hook_name = f"model.model.layers.{layer}"
        hook_handle = [None]

        def make_hook(jg_local):
            def hook_fn(module, input, output):
                h = output[0][:, -1:, :].float()
                transported = h @ jg_local.T
                alpha = 0.3
                new_h = (h + alpha * (transported - h)).to(output[0].dtype)
                return (new_h,) + output[1:]
            return hook_fn

        try:
            layer_module = model.model.layers[layer]
            handle = layer_module.register_forward_hook(make_hook(Jg))
            with torch.no_grad():
                out_a = model(encoded.input_ids)
            handle.remove()

            top_after = torch.topk(out_a.logits[0, -1, :], 5)
            tok_after_ids = top_after.indices.tolist()
            changed = tok_before_ids[0] != tok_after_ids[0]
            print(f"    Before: {tok_before_ids[0]}", flush=True)
            print(f"    After:  {tok_after_ids[0]}", flush=True)
            print(f"    Changed: {changed}", flush=True)
            results.append({
                "prompt": prompt[:100], "layer": layer,
                "divergence": case["divergence"], "case_type": ct,
                "before": tok_before_ids, "after": tok_after_ids,
                "changed": changed,
            })
            del out_a, out_b; free_mem()
        except Exception as e:
            print(f"    ERR: {e}", flush=True)
            import traceback; traceback.print_exc()
            results.append({"error": str(e), "case_type": ct})
            if 'out_b' in dir(): del out_b; free_mem()

    return results


# ─── Baselines ───────────────────────────────────────────────────────────────

def random_baseline(d_model, n=300):
    sims = [cosine_sim(torch.randn(d_model), torch.randn(d_model)) for _ in range(n)]
    return {"mean": float(np.mean(sims)), "std": float(np.std(sims))}


# ─── Plots ───────────────────────────────────────────────────────────────────

def make_plots(readout_results, jac_results, rand_base, output_dir):
    sns.set_theme(style="whitegrid", font_scale=1.0)
    cats = sorted(readout_results.keys())
    layers = ANALYSIS_LAYERS

    # Fig 1: Readout heatmap — J-Lens vs actual
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for idx, (key, label) in enumerate([("jl_mean", "J-Lens"), ("ll_mean", "Logit-Lens")]):
        data = np.zeros((len(cats), len(layers)))
        for i, c in enumerate(cats):
            for j, l in enumerate(layers):
                data[i, j] = readout_results[c]["summary"][l][key]
        sns.heatmap(data, annot=True, fmt=".3f", cmap="RdYlGn",
                    xticklabels=[f"L{l}" for l in layers],
                    yticklabels=[c.replace("_", " ")[:20] for c in cats],
                    vmin=-0.2, vmax=1.0, ax=axes[idx], linewidths=0.5)
        axes[idx].set_title(f"{label}: Cosine Sim with Actual Logits", fontsize=12)
        axes[idx].set_xlabel("Layer"); axes[idx].set_ylabel("Category" if idx == 0 else "")
    plt.suptitle("Readout Quality: J-Lens vs Logit-Lens (Qwen3-1.7B)", fontsize=14, y=1.02)
    plt.tight_layout(); plt.savefig(output_dir / "fig1_readout_heatmap.png", dpi=150, bbox_inches="tight"); plt.close()

    # Fig 2: Readout lines across layers
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_jl = sns.color_palette("husl", len(cats))
    for ci, c in enumerate(cats):
        jl_means = [readout_results[c]["summary"][l]["jl_mean"] for l in layers]
        ll_means = [readout_results[c]["summary"][l]["ll_mean"] for l in layers]
        ax.plot(layers, jl_means, "o-", color=colors_jl[ci], linewidth=1.8, markersize=5,
                label=f"{c.replace('_',' ')[:18]} (JL)")
        ax.plot(layers, ll_means, "s--", color=colors_jl[ci], linewidth=1.2, markersize=4, alpha=0.6)
    ax.set_xlabel("Layer"); ax.set_ylabel("Cosine Similarity with Actual Logits")
    ax.set_title("J-Lens (solid) vs Logit-Lens (dashed) Across Layers")
    ax.legend(fontsize=7, ncol=2, loc="lower right"); ax.set_ylim(-0.2, 1.0)
    plt.tight_layout(); plt.savefig(output_dir / "fig2_readout_lines.png", dpi=150); plt.close()

    # Fig 3: J-Lens advantage over Logit-Lens
    fig, ax = plt.subplots(figsize=(10, 6))
    for ci, c in enumerate(cats):
        diffs = [readout_results[c]["summary"][l]["jl_beats_ll"] for l in layers]
        ax.plot(layers, diffs, "o-", color=colors_jl[ci], linewidth=2, markersize=6,
                label=c.replace("_", " ")[:20])
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Layer"); ax.set_ylabel("J-Lens cos - Logit-Lens cos")
    ax.set_title("J-Lens Advantage Over Logit-Lens by Layer")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout(); plt.savefig(output_dir / "fig3_jl_advantage.png", dpi=150); plt.close()

    # Fig 4: Jacobian cosine heatmap (if available)
    if jac_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        jcats = sorted(jac_results.keys())
        data = np.zeros((len(jcats), len(layers)))
        for i, c in enumerate(jcats):
            for j, l in enumerate(layers):
                data[i, j] = jac_results[c]["summary"].get(l, {}).get("jac_mean", 0)
        sns.heatmap(data, annot=True, fmt=".3f", cmap="RdYlGn",
                    xticklabels=[f"L{l}" for l in layers],
                    yticklabels=[c.replace("_", " ")[:20] for c in jcats],
                    vmin=-0.05, vmax=1.0, ax=ax, linewidths=0.5)
        ax.set_title("Jacobian Cosine Similarity: Global vs Local\n(Qwen3-1.7B, sampled dims)")
        plt.tight_layout(); plt.savefig(output_dir / "fig4_jacobian_heatmap.png", dpi=150); plt.close()

    # Fig 5: Category bar chart — readout at mid-layer
    fig, ax = plt.subplots(figsize=(10, 6))
    mid_layer = ANALYSIS_LAYERS[len(ANALYSIS_LAYERS) // 2]
    x = np.arange(len(cats))
    jl_scores = [readout_results[c]["summary"][mid_layer]["jl_mean"] for c in cats]
    ll_scores = [readout_results[c]["summary"][mid_layer]["ll_mean"] for c in cats]
    ax.bar(x - 0.15, jl_scores, 0.3, label="J-Lens", color="#2196F3", alpha=0.85)
    ax.bar(x + 0.15, ll_scores, 0.3, label="Logit-Lens", color="#FF9800", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n")[:15] for c in cats], fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Cosine Similarity with Actual Model Logits")
    ax.set_title(f"Readout Quality at Layer {mid_layer}: J-Lens vs Logit-Lens")
    ax.legend(); ax.set_ylim(-0.2, 1.0)
    for i, (j, l) in enumerate(zip(jl_scores, ll_scores)):
        ax.text(i - 0.15, j + 0.02, f"{j:.2f}", ha="center", fontsize=7)
        ax.text(i + 0.15, l + 0.02, f"{l:.2f}", ha="center", fontsize=7)
    plt.tight_layout(); plt.savefig(output_dir / "fig5_category_bars.png", dpi=150); plt.close()

    print(f"Plots saved to {output_dir}", flush=True)


# ─── Report ──────────────────────────────────────────────────────────────────

def write_report(readout_results, jac_results, rand_base, patch_results, output_dir):
    cats = sorted(readout_results.keys())
    layers = ANALYSIS_LAYERS

    L = []
    L.append("=" * 70)
    L.append("J-LENS LOCAL vs GLOBAL DIVERGENCE — Qwen3-1.7B RESULTS")
    L.append("=" * 70)
    L.append("")
    L.append("Model: Qwen3-1.7B (1.7B params, 2048 d_model, 28 layers)")
    L.append("Pre-fit J-Lens: neuronpedia/jacobian-lens (466 prompts, wikitext)")
    L.append(f"Jacobian sampling: {K_DIMS_JAC} random d_model dims (for Jacobian subset)")
    L.append(f"{N_PROMPTS_PER_CAT} prompts/category x {len(cats)} categories")
    L.append("")
    L.append("HYPOTHESES:")
    L.append("  H1: J-Lens faithfully approximates local computation")
    L.append("  H2: Global average artifact, worse for safety-relevant inputs")
    L.append("  H3: Divergence predicts causal patching failure")
    L.append("")
    L.append(f"Random direction baseline: {rand_base['mean']:.4f} +/- {rand_base['std']:.4f}")
    L.append("")

    # Readout results
    L.append("READOUT COSINE SIMILARITY (Logit vectors vs actual model output)")
    L.append("")
    header = f"{'Category':<25}"
    for l in layers:
        header += f"{'JL@L'+str(l):>8}{'LL@L'+str(l):>8}"
    L.append(header)
    L.append("-" * len(header))
    for c in cats:
        row = f"{c:<25}"
        for l in layers:
            s = readout_results[c]["summary"][l]
            row += f"{s['jl_mean']:>8.4f}{s['ll_mean']:>8.4f}"
        L.append(row)
    L.append("")

    # Layer means
    L.append("LAYER MEANS (J-Lens vs Logit-Lens):")
    for l in layers:
        jl_means = [readout_results[c]["summary"][l]["jl_mean"] for c in cats]
        ll_means = [readout_results[c]["summary"][l]["ll_mean"] for c in cats]
        diff = np.mean(jl_means) - np.mean(ll_means)
        L.append(f"  L{l}: JL={np.mean(jl_means):.4f}  LL={np.mean(ll_means):.4f}  diff={diff:+.4f}")
    L.append("")

    # Jacobian results
    if jac_results:
        L.append("JACOBIAN COSINE SIMILARITY (Global J-Lens vs Local Jacobian)")
        header = f"{'Category':<25}" + "".join(f"{'L'+str(l):>8}" for l in layers) + f"{'Mean':>8}"
        L.append(header)
        L.append("-" * len(header))
        all_means = []
        for c in cats:
            if c in jac_results:
                means = [jac_results[c]["summary"].get(l, {}).get("jac_mean", float("nan")) for l in layers]
                m = np.nanmean(means)
                all_means.append(m)
                L.append(f"{c:<25}" + "".join(f"{v:>8.4f}" for v in means) + f"{m:>8.4f}")
        if all_means:
            L.append(f"{'GRAND MEAN':<25}" + "".join(f"{'':>8}" for _ in layers) + f"{np.mean(all_means):>8.4f}")
        L.append("")

    # Interpretation
    L.append("INTERPRETATION:")

    # Readout verdict
    all_jl = [readout_results[c]["summary"][l]["jl_mean"]
              for c in cats for l in layers]
    all_ll = [readout_results[c]["summary"][l]["ll_mean"]
              for c in cats for l in layers]
    L.append(f"  J-Lens readout mean (all layers): {np.mean(all_jl):.4f}")
    L.append(f"  Logit-Lens readout mean (all layers): {np.mean(all_ll):.4f}")
    diff = np.mean(all_jl) - np.mean(all_ll)
    if diff > 0.05:
        L.append(f"  => J-Lens BEATS Logit-Lens by {diff:.4f} on average")
    elif diff < -0.05:
        L.append(f"  => Logit-Lens beats J-Lens by {-diff:.4f} on average")
    else:
        L.append(f"  => Comparable performance (diff={diff:+.4f})")
    L.append("")

    # Safety gap
    safety = {"eval_awareness", "deception_scenarios"}
    in_dist = {"in_distribution_generic", "code_text"}
    s_jl = [readout_results[c]["summary"][l]["jl_mean"]
            for c in cats for l in layers if c in safety]
    i_jl = [readout_results[c]["summary"][l]["jl_mean"]
            for c in cats for l in layers if c in in_dist]
    if s_jl and i_jl:
        L.append(f"  Safety-relevant J-Lens mean: {np.mean(s_jl):.4f}")
        L.append(f"  In-distribution J-Lens mean: {np.mean(i_jl):.4f}")
        safety_diff = np.mean(i_jl) - np.mean(s_jl)
        L.append(f"  Gap (in-dist - safety): {safety_diff:+.4f}")
        if safety_diff > 0.05:
            L.append("  => H2 PARTIALLY SUPPORTED: J-Lens less reliable on safety inputs")
        elif safety_diff < -0.05:
            L.append("  => J-Lens MORE reliable on safety inputs (surprising)")
        else:
            L.append("  => No significant safety/in-dist gap")
    L.append("")

    # Layer trend
    jl_layer_means = [np.mean([readout_results[c]["summary"][l]["jl_mean"] for c in cats])
                      for l in layers]
    L.append("Layer trend (J-Lens):")
    for i, l in enumerate(layers):
        arrow = "^" if i > 0 and jl_layer_means[i] > jl_layer_means[i-1] else "v"
        L.append(f"  L{l}: {jl_layer_means[i]:.4f} ({arrow})")
    if jl_layer_means[-1] > jl_layer_means[0] + 0.2:
        L.append("  => J-Lens fidelity improves strongly with depth")
    L.append("")

    # Jacobian verdict
    if jac_results:
        jac_all = []
        for c in cats:
            if c in jac_results:
                for l in layers:
                    if l in jac_results[c]["summary"]:
                        jac_all.append(jac_results[c]["summary"][l]["jac_mean"])
        if jac_all:
            grand_jac = np.mean(jac_all)
            L.append(f"  Grand mean Jacobian cosine: {grand_jac:.4f} (random: {rand_base['mean']:.4f})")
            if grand_jac > 0.7:
                L.append("  Jacobian verdict: H1 PARTIALLY SUPPORTED at late layers")
            elif grand_jac > 0.3:
                L.append("  Jacobian verdict: Moderate fidelity, strong layer dependence")
            else:
                L.append("  Jacobian verdict: J-Lens poorly approximates local Jacobian structure")
    L.append("")

    # Patching
    if patch_results:
        L.append("PATCHING RESULTS:")
        for group in ["high_divergence", "low_divergence"]:
            cases = [p for p in patch_results if p.get("case_type") == group and "error" not in p]
            if cases:
                ch = sum(1 for c in cases if c["changed"])
                L.append(f"  {group}: {ch}/{len(cases)} changed top prediction")
        L.append("")

    L.append("=" * 70)
    report = "\n".join(L)
    with open(output_dir / "report.txt", "w") as f:
        f.write(report)
    print(report, flush=True)
    return report


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60, flush=True)
    print("J-LENS DIVERGENCE EXPERIMENT — Qwen3-1.7B", flush=True)
    print(f"  Readout: all prompts, all layers (fast)", flush=True)
    print(f"  Jacobian: {K_JAC_PROMPTS}/category, K={K_DIMS_JAC} dims (slow, subset)", flush=True)
    print(f"  Layers: {ANALYSIS_LAYERS}", flush=True)
    print("=" * 60, flush=True)

    model, tokenizer = load_model()
    global_lens = load_lens()
    d_model = model.config.hidden_size

    # Positive control
    print("\n" + "=" * 60, flush=True)
    print("POSITIVE CONTROL", flush=True)
    print("=" * 60, flush=True)
    test_prompt = "The quick brown fox jumps over the lazy dog near the river bank on a sunny afternoon in spring."
    Jg14 = global_lens.jacobians[14].float()
    rc = compare_readouts(model, tokenizer, test_prompt, 14, Jg14)
    print(f"  Readout JL-vs-actual: {rc['jl_vs_actual_cos']:.4f}", flush=True)
    print(f"  Readout LL-vs-actual: {rc['ll_vs_actual_cos']:.4f}", flush=True)
    print(f"  JL beats LL: {rc['jl_vs_actual_cos'] - rc['ll_vs_actual_cos']:+.4f}", flush=True)
    del Jg14; free_mem()

    rb = random_baseline(d_model)
    print(f"\n  Random baseline: {rb['mean']:.4f} +/- {rb['std']:.4f}", flush=True)

    # Experiment 1: Readout comparison (FAST — all prompts, all layers)
    print("\n" + "=" * 60, flush=True)
    print("EXPERIMENT 1: Readout Comparison (all prompts, all layers)", flush=True)
    print("=" * 60, flush=True)
    t0 = time.time()
    readout_results = experiment_readout(model, tokenizer, global_lens, ALL_CATEGORIES)
    print(f"\nReadout experiment completed in {time.time()-t0:.0f}s", flush=True)

    # Experiment 2: Jacobian sampling (SLOW — subset only)
    print("\n" + "=" * 60, flush=True)
    print("EXPERIMENT 2: Jacobian Sampling (subset, K=10)", flush=True)
    print("=" * 60, flush=True)
    t0 = time.time()
    jac_results = experiment_jacobian(model, tokenizer, global_lens, ALL_CATEGORIES)
    print(f"\nJacobian experiment completed in {time.time()-t0:.0f}s", flush=True)

    # Experiment 3: Causal patching
    print("\n" + "=" * 60, flush=True)
    print("EXPERIMENT 3: Causal Patching", flush=True)
    print("=" * 60, flush=True)
    patch_results = experiment_patching(model, tokenizer, global_lens, readout_results, ALL_CATEGORIES)

    # Plots + report
    print("\nGenerating plots...", flush=True)
    make_plots(readout_results, jac_results, rb, OUTPUT_DIR)

    print("\nWriting report...", flush=True)
    write_report(readout_results, jac_results, rb, patch_results, OUTPUT_DIR)

    # Save JSON
    def ser(o):
        if isinstance(o, torch.Tensor): return o.tolist()
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, defaultdict): return dict(o)
        if isinstance(o, dict): return {str(k): ser(v) for k, v in o.items()}
        if isinstance(o, list): return [ser(v) for v in o]
        return o

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(ser({"readout": readout_results, "jacobian": jac_results,
                        "patching": patch_results}), f, indent=2)

    print(f"\nDone! Results: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
