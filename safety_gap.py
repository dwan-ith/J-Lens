"""
Phase 1: Safety-Specific Readout Validation (Deep)

Tests whether J-Lens readout fidelity degrades on safety-relevant inputs
with rigorous statistics: bootstrap CIs, permutation tests, effect sizes,
multiple-testing correction, and random-direction control.

Design:
  - Web-text lens (466 wikitext prompts) evaluated on 8 categories
  - Metrics: cosine JL vs actual, LL vs actual, and random-J vs actual
  - Per-layer: bootstrap 95% CI, permutation p, Cohen's d, Cliff's delta
  - Per-category: BH-corrected significance of JL advantage over LL and over random
  - Safety gap defined on pooled per-prompt scores (not category means)

If readout fidelity drops on safety inputs when using a web-text lens,
that means safety monitoring with off-the-shelf lenses is unreliable.
"""

import sys, os, json, time, warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "jacobian-lens"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import jlens
from jlens.hooks import ActivationRecorder
from safety_prompts import get_all_categories
import analysis_utils as au
from config import get_config, DEFAULT_LENS_FILE, DEFAULT_MODEL_DIR

# ─── Config (centralized, overridable via env/CLI) ───────────────────────────
_cfg = get_config()
MODEL_DIR = str(DEFAULT_MODEL_DIR)
WEBTEXT_LENS = str(DEFAULT_LENS_FILE)
OUTPUT_DIR = Path(os.path.dirname(__file__), "results", "phase1_safety_gap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_PROMPTS_PER_CAT = _cfg.n_prompts_per_cat
MAX_SEQ_LEN = _cfg.max_seq_len
# Layers derived from fractions — not hardcoded to 28. Updated at runtime after model load.
ANALYSIS_LAYERS = _cfg.analysis_layers(28)
N_RAND_SEEDS = _cfg.n_rand_seeds
N_BOOT = _cfg.n_boot
N_PERM = _cfg.n_perm


# ─── Helpers ─────────────────────────────────────────────────────────────────

def cosine_sim(a, b):
    return au.cosine_sim(a, b)

def free_mem():
    au.free_mem()

def load_model():
    import transformers
    # dtype/device from central config (env-overridable)
    from config import get_config as _get_cfg
    _lcfg = _get_cfg()
    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    _dtype = dtype_map.get(_lcfg.dtype, torch.float16)
    print(f"Loading model from {MODEL_DIR} (dtype={_lcfg.dtype}, device={_lcfg.device})...", flush=True)
    t0 = time.time()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=_dtype, device_map="cpu",
        low_cpu_mem_usage=True, local_files_only=True,
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


# ─── Core readout (batched, with random-J control) ──────────────────────────

def measure_all_layers(model, tokenizer, prompt, lens):
    """Batched measurement with random-direction control.

    For each layer we compute:
      jl_cos  — W_U @ (J @ h) vs actual
      ll_cos  — W_U @ h vs actual
      rand_cos — mean over N_RAND_SEEDS random matrices matched to ||J||_F

    Returns dict layer -> {jl_cos, ll_cos, rand_cos, rand_std}
    """
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    input_ids = encoded.input_ids
    # handle both HF (model.model.layers) and TinyDecoder (model.layers)
    text_module = getattr(model, "model", model)
    block_layers = getattr(text_module, "layers", getattr(model, "layers", None))
    if block_layers is None:
        raise AttributeError("model has no .layers")

    valid_layers = [l for l in ANALYSIS_LAYERS if 0 <= l < len(block_layers)]
    if not valid_layers:
        raise ValueError(f"No valid analysis layers for depth {len(block_layers)}: {ANALYSIS_LAYERS}")
    with torch.no_grad():
        with ActivationRecorder(block_layers, at=valid_layers) as recorder:
            # Use text_module forward if available, else model forward (TinyDecoder)
            forward_fn = getattr(text_module, "forward", None)
            if forward_fn is not None and text_module is not model:
                # HF path: text_module is model.model
                text_module(input_ids=input_ids, use_cache=False)
            else:
                # TinyDecoder path: model.forward
                model.forward(input_ids)
            activations = {l: recorder.activations[l][0, -1, :].detach().float().cpu() for l in valid_layers}

    W_U = model.lm_head.weight.data.cpu().float()
    with torch.no_grad():
        out = model(input_ids)
        # HF returns .logits, Tiny returns .last_hidden_state -> use unembed
        if hasattr(out, "logits"):
            logits_actual = out.logits[0, -1, :].cpu().float()
        else:
            # Tiny: need to unembed last hidden
            logits_actual = model.unembed(out.last_hidden_state)[0, -1, :].cpu().float()

    out_dict = {}
    K = 15
    actual_top = set(torch.topk(logits_actual, K).indices.tolist())
    for layer in valid_layers:
        h = activations[layer]
        logits_ll = (W_U @ h).float()
        ll_cos = cosine_sim(logits_ll, logits_actual)
        ll_top = set(torch.topk(logits_ll, K).indices.tolist())

        if layer in lens.jacobians:
            Jg = lens.jacobians[layer].float()
            logits_jl = (W_U @ (Jg @ h)).float()
            jl_cos = cosine_sim(logits_jl, logits_actual)
            jl_top = set(torch.topk(logits_jl, K).indices.tolist())
            # random-J control: same Frobenius norm as Jg (local Generator — no global RNG reset)
            frob = float(torch.norm(Jg).item())
            rand_coses = []
            for s in range(N_RAND_SEEDS):
                g = torch.Generator().manual_seed(s + 1009 * layer)
                R = torch.randn(Jg.shape, generator=g, dtype=Jg.dtype, device=Jg.device)
                R = R / torch.norm(R) * frob
                rand_logits = (W_U @ (R @ h)).float()
                rand_coses.append(cosine_sim(rand_logits, logits_actual))
            rand_cos = float(np.mean(rand_coses))
            rand_std = float(np.std(rand_coses))
        else:
            jl_cos = ll_cos; rand_cos = ll_cos; rand_std = 0.0
            jl_top = ll_top

        out_dict[layer] = {
            "jl_cos": jl_cos, "ll_cos": ll_cos,
            "rand_cos": rand_cos, "rand_std": rand_std,
            "jl_overlap": len(actual_top & jl_top) / K,
            "ll_overlap": len(actual_top & ll_top) / K,
            "jl_beats_ll": jl_cos - ll_cos,
            "jl_beats_rand": jl_cos - rand_cos,
        }
    return out_dict


# ─── Experiment runner ───────────────────────────────────────────────────────

def run_experiment(model, tokenizer, categories, lens, lens_name, n_prompts):
    results = {}
    # also collect pooled per-prompt scores for rigorous safety gap
    pooled = {l: {"jl": [], "ll": [], "rand": []} for l in ANALYSIS_LAYERS}
    for cat_name, prompts in categories.items():
        print(f"\n  [{lens_name}] {cat_name}", end="", flush=True)
        cat_data = {"per_layer": defaultdict(list), "per_prompt": []}
        n = min(len(prompts), n_prompts)
        for prompt in prompts[:n]:
            try:
                all_rc = measure_all_layers(model, tokenizer, prompt, lens)
                prompt_result = {"prompt": prompt[:100], "layers": all_rc}
                for layer in ANALYSIS_LAYERS:
                    cat_data["per_layer"][layer].append(all_rc[layer])
                    pooled[layer]["jl"].append((cat_name, all_rc[layer]["jl_cos"]))
                    pooled[layer]["ll"].append((cat_name, all_rc[layer]["ll_cos"]))
                    pooled[layer]["rand"].append((cat_name, all_rc[layer]["rand_cos"]))
            except Exception as e:
                print(f" ERR: {e}", end="", flush=True)
                import traceback; traceback.print_exc()
                prompt_result = {"prompt": prompt[:100], "layers": {}}
            free_mem()
            cat_data["per_prompt"].append(prompt_result)

        cat_data["summary"] = {}
        for l in ANALYSIS_LAYERS:
            if cat_data["per_layer"][l]:
                jl = [d["jl_cos"] for d in cat_data["per_layer"][l]]
                ll = [d["ll_cos"] for d in cat_data["per_layer"][l]]
                rc = [d["rand_cos"] for d in cat_data["per_layer"][l]]
                jl_est, jl_lo, jl_hi = au.bootstrap_ci(jl, n_boot=N_BOOT)
                diff_est, diff_lo, diff_hi, diff_p = au.bootstrap_diff_ci(jl, ll, n_boot=N_BOOT)
                _, perm_p = au.permutation_test(jl, ll, n_perm=N_PERM)
                cat_data["summary"][l] = {
                    "jl_mean": round(float(np.mean(jl)), 4),
                    "jl_std": round(float(np.std(jl)), 4),
                    "jl_ci_lo": round(jl_lo, 4), "jl_ci_hi": round(jl_hi, 4),
                    "ll_mean": round(float(np.mean(ll)), 4),
                    "rand_mean": round(float(np.mean(rc)), 4),
                    "jl_beats_ll": round(float(np.mean(jl)-np.mean(ll)), 4),
                    "jl_beats_ll_ci_lo": round(diff_lo, 4), "jl_beats_ll_ci_hi": round(diff_hi, 4),
                    "jl_beats_ll_p_boot": round(diff_p, 4),
                    "jl_beats_ll_p_perm": round(perm_p, 4),
                    "cohens_d": round(au.cohens_d(jl, ll), 4),
                    "cliffs_delta": round(au.cliffs_delta(jl, ll), 4),
                    "n": len(jl),
                }
            else:
                cat_data["summary"][l] = {"jl_mean": 0, "ll_mean": 0, "n": 0}
        peak_l = max(cat_data["summary"], key=lambda l: cat_data["summary"][l].get("jl_mean", -1))
        peak_v = cat_data["summary"][peak_l].get("jl_mean", 0)
        print(f"  peak={peak_v:.3f}@L{peak_l}", flush=True)
        results[cat_name] = cat_data
    return results, pooled


# ─── Safety gap with rigorous stats ─────────────────────────────────────────

def compute_safety_gap(pooled, categories):
    """Pooled per-prompt safety gap with bootstrap CI, permutation p, effect sizes."""
    in_cats = {"in_distribution"}
    gaps = {}
    for layer in ANALYSIS_LAYERS:
        in_scores = [v for cat, v in pooled[layer]["jl"] if cat in in_cats]
        safety_scores = [v for cat, v in pooled[layer]["jl"] if cat not in in_cats]
        if not in_scores or not safety_scores:
            gaps[layer] = {"in_dist_mean": 0, "in_dist_ci_lo": 0, "in_dist_ci_hi": 0,
                           "safety_mean": 0, "safety_ci_lo": 0, "safety_ci_hi": 0,
                           "gap": 0, "gap_ci_lo": 0, "gap_ci_hi": 0,
                           "gap_p_boot": 1.0, "gap_p_perm": 1.0, "gap_p_perm_bh": 1.0,
                           "cohens_d": 0.0, "cliffs_delta": 0.0,
                           "n_in": 0, "n_safety": 0}
            continue
        gap, lo, hi, p_boot = au.bootstrap_diff_ci(safety_scores, in_scores, n_boot=N_BOOT)
        # Note: gap = safety - in_dist so positive = safety better
        _, p_perm = au.permutation_test(safety_scores, in_scores, n_perm=N_PERM)
        gaps[layer] = {
            "in_dist_mean": round(float(np.mean(in_scores)), 4),
            "in_dist_ci_lo": round(au.bootstrap_ci(in_scores, n_boot=N_BOOT)[1], 4),
            "in_dist_ci_hi": round(au.bootstrap_ci(in_scores, n_boot=N_BOOT)[2], 4),
            "safety_mean": round(float(np.mean(safety_scores)), 4),
            "safety_ci_lo": round(au.bootstrap_ci(safety_scores, n_boot=N_BOOT)[1], 4),
            "safety_ci_hi": round(au.bootstrap_ci(safety_scores, n_boot=N_BOOT)[2], 4),
            "gap": round(float(gap), 4),  # safety - in_dist
            "gap_ci_lo": round(lo, 4), "gap_ci_hi": round(hi, 4),
            "gap_p_boot": round(p_boot, 4),
            "gap_p_perm": round(p_perm, 4),
            "cohens_d": round(au.cohens_d(safety_scores, in_scores), 4),
            "cliffs_delta": round(au.cliffs_delta(safety_scores, in_scores), 4),
            "n_in": len(in_scores), "n_safety": len(safety_scores),
        }
    # BH correction across layers
    pvals = [g["gap_p_perm"] for g in gaps.values()]
    adj = au.benjamini_hochberg(pvals)
    for (layer, g), a in zip(gaps.items(), adj):
        g["gap_p_perm_bh"] = round(a, 4)
    return gaps


# ─── Plots ───────────────────────────────────────────────────────────────────

def make_plots(webtext_results, pooled, categories, output_dir):
    sns.set_theme(style="whitegrid", font_scale=1.0)
    cats = sorted(categories.keys())
    layers = ANALYSIS_LAYERS

    # Fig1: mid-layer readout with bootstrap CIs (dynamic — not hardcoded to L14)
    fig, ax = plt.subplots(figsize=(15, 6))
    layer = layers[len(layers) // 2]
    x = np.arange(len(cats))
    jl_means = [webtext_results[c]["summary"][layer]["jl_mean"] for c in cats]
    jl_lo = [webtext_results[c]["summary"][layer].get("jl_ci_lo", webtext_results[c]["summary"][layer]["jl_mean"]) for c in cats]
    jl_hi = [webtext_results[c]["summary"][layer].get("jl_ci_hi", webtext_results[c]["summary"][layer]["jl_mean"]) for c in cats]
    colors = ["#2196F3" if "in_" in c or "safety_" in c else "#FF9800" for c in cats]
    ax.bar(x, jl_means, 0.6, color=colors, alpha=0.85, yerr=[np.array(jl_means)-np.array(jl_lo), np.array(jl_hi)-np.array(jl_means)], capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n")[:20] for c in cats], fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("J-Lens Cosine (95% bootstrap CI)")
    ax.set_title(f"J-Lens Readout at L{layer} with Bootstrap CIs")
    plt.tight_layout()
    plt.savefig(output_dir / "fig1_safety_readout_by_category.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig2: layer profiles with CIs (pooled)
    fig, ax = plt.subplots(figsize=(11, 6))
    for group, color, label, marker in [
        ([c for c in cats if "in_" in c], "#4CAF50", "In-Distribution", "o"),
        ([c for c in cats if "safety_" in c], "#F44336", "Safety-Relevant", "s"),
        ([c for c in cats if "adversarial_" in c], "#FF9800", "Adversarial", "^"),
    ]:
        if not group: continue
        means, los, his = [], [], []
        for l in layers:
            vals = [v for cat, v in pooled[l]["jl"] if cat in group]
            est, lo, hi = au.bootstrap_ci(vals, n_boot=N_BOOT)
            means.append(est); los.append(lo); his.append(hi)
        means = np.array(means); los = np.array(los); his = np.array(his)
        ax.plot(layers, means, f"{marker}-", color=color, linewidth=2, markersize=7, label=label)
        ax.fill_between(layers, los, his, color=color, alpha=0.15)
    ax.set_xlabel("Layer"); ax.set_ylabel("J-Lens Cosine (95% CI)"); ax.set_title("Pooled Readout by Input Type (bootstrap CI)"); ax.legend(); ax.set_ylim(-0.6, 1.0)
    plt.tight_layout(); plt.savefig(output_dir / "fig2_layer_profiles_by_type.png", dpi=150); plt.close()

    # Fig3: safety gap forest plot with CIs and significance
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(layers))
    gap_vals, lo_vals, hi_vals = [], [], []
    for l in layers:
        in_s = [v for cat,v in pooled[l]["jl"] if cat in {"in_distribution"}]
        sf_s = [v for cat,v in pooled[l]["jl"] if cat not in {"in_distribution"}]
        _, lo, hi, _ = au.bootstrap_diff_ci(sf_s, in_s, n_boot=N_BOOT)
        gap_vals.append(float(np.mean(sf_s)-np.mean(in_s)) if sf_s and in_s else 0.0)
        lo_vals.append(lo); hi_vals.append(hi)
    gap_vals=np.array(gap_vals); lo_vals=np.array(lo_vals); hi_vals=np.array(hi_vals)
    ax.errorbar(gap_vals, y, xerr=[gap_vals-lo_vals, hi_vals-gap_vals], fmt='o', color='#2196F3', capsize=5)
    ax.axvline(0, color='red', linestyle='--')
    ax.set_yticks(y); ax.set_yticklabels([f"L{l}" for l in layers])
    ax.set_xlabel("Safety − In-Dist Gap (cosine, 95% CI)")
    ax.set_title("Safety Gap Forest Plot (safety better → right)")
    plt.tight_layout(); plt.savefig(output_dir / "fig3_safety_gap_forest.png", dpi=150); plt.close()

    # Fig4: heatmap with significance stars for JL>LL
    fig, ax = plt.subplots(figsize=(11, 6))
    data = np.zeros((len(cats), len(layers)))
    annot = np.empty((len(cats), len(layers)), dtype=object)
    for i, c in enumerate(cats):
        for j, l in enumerate(layers):
            data[i,j] = webtext_results[c]["summary"][l]["jl_mean"]
            p = webtext_results[c]["summary"][l].get("jl_beats_ll_p_perm", 1.0)
            star = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
            annot[i,j] = f"{data[i,j]:.2f}{star}"
    sns.heatmap(data, annot=annot, fmt="", cmap="RdYlGn", xticklabels=[f"L{l}" for l in layers], yticklabels=[c.replace("_"," ")[:22] for c in cats], vmin=-0.6, vmax=1.0, ax=ax, linewidths=0.5)
    ax.set_title("J-Lens Heatmap (* p<0.05 permutation, JL>LL)")
    plt.tight_layout(); plt.savefig(output_dir / "fig4_safety_heatmap.png", dpi=150); plt.close()
    print(f"Plots saved to {output_dir}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Phase 1: Safety Gap")
    ap.add_argument("--model", default=None, help="model_id from config registry (e.g., qwen3-1.7b, inkling)")
    ap.add_argument("--n-prompts", type=int, default=None, help="prompts per category")
    ap.add_argument("--max-seq-len", type=int, default=None)
    args = ap.parse_args(argv)

    global ANALYSIS_LAYERS, N_PROMPTS_PER_CAT, MAX_SEQ_LEN, MODEL_DIR, WEBTEXT_LENS
    global N_BOOT, N_PERM, N_RAND_SEEDS
    cfg = get_config(**{k: v for k, v in {"model_id": args.model, "n_prompts_per_cat": args.n_prompts, "max_seq_len": args.max_seq_len}.items() if v is not None})
    if args.n_prompts is not None:
        N_PROMPTS_PER_CAT = args.n_prompts
    if args.max_seq_len is not None:
        MAX_SEQ_LEN = args.max_seq_len
    if N_PROMPTS_PER_CAT < 1:
        raise ValueError(f"--n-prompts must be >= 1, got {N_PROMPTS_PER_CAT}")
    # refresh path + stats globals from resolved config so --model actually switches models
    MODEL_DIR = str(cfg.local_dir) if cfg.local_dir else MODEL_DIR
    WEBTEXT_LENS = str(cfg.lens) if cfg.lens else WEBTEXT_LENS
    N_BOOT = cfg.n_boot; N_PERM = cfg.n_perm; N_RAND_SEEDS = cfg.n_rand_seeds

    print("="*60); print("PHASE 1: Safety Gap (Deep)"); print("="*60)
    print(f"Config: model={cfg.model_id} max_seq_len={MAX_SEQ_LEN} n_prompts={N_PROMPTS_PER_CAT}")
    categories = get_all_categories()
    print(f"Categories: {list(categories.keys())}"); print(f"Prompts/cat: {N_PROMPTS_PER_CAT}  rand_seeds={N_RAND_SEEDS}  boot={N_BOOT}")
    model, tokenizer = load_model()
    # derive layers from actual model depth — no hardcoding to 28
    ANALYSIS_LAYERS = cfg.analysis_layers(model.config.num_hidden_layers)
    print(f"Analysis layers (fractions {cfg.layer_fractions}): {ANALYSIS_LAYERS}")
    print(f"\nLoading web-text J-Lens from {WEBTEXT_LENS}...")
    webtext_lens = jlens.JacobianLens.load(WEBTEXT_LENS)
    print(f"  {webtext_lens}")
    print("\n--- Experiment A: Web-text lens ---")
    t0=time.time()
    results, pooled = run_experiment(model, tokenizer, categories, webtext_lens, "webtext", N_PROMPTS_PER_CAT)
    print(f"\nCompleted in {time.time()-t0:.0f}s")
    print("\n--- Safety Gap (pooled, bootstrap + permutation + BH) ---")
    gaps = compute_safety_gap(pooled, categories)
    for layer, g in gaps.items():
        sig = " *" if g["gap_p_perm_bh"]<0.05 else ""
        print(f"  L{layer}: safety {g['safety_mean']:.3f} [{g['safety_ci_lo']:.3f},{g['safety_ci_hi']:.3f}]  in {g['in_dist_mean']:.3f}  gap {g['gap']:+.3f} [{g['gap_ci_lo']:+.3f},{g['gap_ci_hi']:+.3f}] d={g['cohens_d']:+.2f} p_perm_bh={g['gap_p_perm_bh']:.3f}{sig}")
    print("\nGenerating plots...")
    make_plots(results, pooled, categories, OUTPUT_DIR)
    def ser(o):
        if isinstance(o, torch.Tensor): return o.tolist()
        if isinstance(o, (np.integer, np.floating, np.bool_)): return o.item()
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))): return None
        if isinstance(o, defaultdict): return {str(k): ser(v) for k, v in o.items()}
        if isinstance(o, dict): return {str(k): ser(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [ser(v) for v in o]
        return o
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(ser({"results": results, "gaps": gaps, "pooled": {k: {"jl": v["jl"], "ll": v["ll"], "rand": v["rand"]} for k,v in pooled.items()}}), f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
