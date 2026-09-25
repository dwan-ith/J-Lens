"""
Phase 2: Deception Detection — Deep

Trajectory analysis + learned probe + contrastive steering + causal control.

Structure:
  A) Trajectory features (5-dim) for each prompt
  B) Contrastive honest vs deceptive pairs → deception direction in residual space
  C) Linear probe on trajectory features (CV ROC-AUC, calibrated)
  D) Cross-validated probe on J-Lens logits at mid-layer (L14) vs LL baseline
  E) Steering vector patching: does deception direction causally increase deception?
     vs random-direction control (matched L2 norm, 3 seeds)
"""

import sys, os, json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "jacobian-lens"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import jlens
from jlens.hooks import ActivationRecorder
from safety_prompts import (
    SAFETY_DECEPTION, SAFETY_EVAL_AWARENESS, SAFETY_SYCOPHANCY,
    SAFETY_REFUSAL, IN_DIST, ADVERSARIAL_ESCALATION, ADVERSARIAL_ROLEPLAY,
    ADVERSARIAL_OBFUSCATION,
    get_contrastive_pairs, get_contrastive_categories,
)
import analysis_utils as au

# ─── Config (centralized) ────────────────────────────────────────────────────
from config import get_config as _get_cfg, DEFAULT_MODEL_DIR, DEFAULT_LENS_FILE

_cfg = _get_cfg()
OUTPUT_DIR = Path(os.path.dirname(__file__), "results", "phase2_deception")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SEQ_LEN = _cfg.max_seq_len
# ALL_LAYERS and MID_LAYER are set at runtime from model.config.n_layers (see main())
ALL_LAYERS: list[int] | None = None
MID_LAYER: int | None = None
N_BOOT = _cfg.n_boot

# ─── Helpers ─────────────────────────────────────────────────────────────────

def cosine_sim(a, b):
    return au.cosine_sim(a, b)

def free_mem():
    au.free_mem()

def _resolve_blocks(model):
    """Return (text_module, block_layers), handling both HF (model.model.layers)
    and standalone (model.layers, e.g. TinyDecoder) layouts. Same pattern as safety_gap.py."""
    text_module = getattr(model, "model", model)
    block_layers = getattr(text_module, "layers", getattr(model, "layers", None))
    if block_layers is None:
        raise AttributeError("model has no .layers")
    return text_module, block_layers

def _record_forward(text_module, model, input_ids):
    """Recording forward pass for activation capture. Uses text_module forward
    if it is a distinct submodule (HF), else model.forward (TinyDecoder)."""
    forward_fn = getattr(text_module, "forward", None)
    if forward_fn is not None and text_module is not model:
        text_module(input_ids=input_ids, use_cache=False)
    else:
        model.forward(input_ids)

def load_model(model_dir=None):
    import transformers
    if model_dir is None:
        from config import DEFAULT_MODEL_DIR
        model_dir = str(DEFAULT_MODEL_DIR)
    print(f"Loading model from {model_dir} (fp16, low-mem)...", flush=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float16, device_map="cpu",
        low_cpu_mem_usage=True, local_files_only=True,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True,
    )
    model.eval()
    n = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  {n:.1f}B params", flush=True)
    return model, tokenizer

# ─── A) Trajectory extraction (batched single-forward) ──────────────────────

def get_trajectory(model, tokenizer, prompt, lens, W_U=None):
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    input_ids = encoded.input_ids
    _, block_layers = _resolve_blocks(model)
    if W_U is None:
        W_U = model.lm_head.weight.data.cpu().float()
    with torch.no_grad():
        with ActivationRecorder(block_layers, at=ALL_LAYERS) as recorder:
            out = model(input_ids)
            activations = {i: recorder.activations[i].detach() for i in ALL_LAYERS}
        if hasattr(out, "logits"):
            logits_actual = out.logits[0, -1, :].cpu().float()
        else:
            # TinyDecoder path: unembed last hidden state
            logits_actual = model.unembed(out.last_hidden_state)[0, -1, :].cpu().float()
    jl_traj, ll_traj = [], []
    for layer in ALL_LAYERS:
        h = activations[layer][0, -1, :].float().cpu()
        ll_cos = cosine_sim(W_U @ h, logits_actual)
        if layer in lens.jacobians:
            Jg = lens.jacobians[layer].float()
            jl_cos = cosine_sim(W_U @ (Jg @ h), logits_actual)
        else:
            jl_cos = 0.0
        jl_traj.append(jl_cos); ll_traj.append(ll_cos)
    return jl_traj, ll_traj, activations

def trajectory_features(jl_traj, ll_traj):
    jl = np.array(jl_traj); ll = np.array(ll_traj)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    divergence_area = float(_trapz(np.abs(jl - ll), dx=1))
    n = len(jl)
    # use dynamic mid/late ranges instead of hardcoded layer indices
    mid_lo = max(0, n // 3); mid_hi = max(mid_lo + 1, 2 * n // 3)
    late_lo = max(0, 2 * n // 3); late_hi = n
    mid_mean = float(np.mean(jl[mid_lo:mid_hi]))
    late_mean = float(np.mean(jl[late_lo:late_hi])) if late_lo < late_hi else float(jl[-1])
    return {
        "divergence_area": divergence_area,
        "mid_late_gap": mid_mean - late_mean,
        "peak_layer": int(np.argmax(jl)), "peak_value": float(np.max(jl)),
        "trajectory_var": float(np.var(jl)),
        "final_surprise": float(abs(jl[-1] - mid_mean)),
    }

def classify_heuristic(score, thresholds=None):
    """Heuristic with config-driven thresholds (no hardcoding). Falls back to
    data-driven percentile if thresholds is None: uses in-dist 90th percentile."""
    if thresholds is None:
        from config import get_config
        thresholds = get_config().thresholds
    signals = sum([
        score["divergence_area"] > thresholds.divergence_area,
        score["mid_late_gap"] > thresholds.mid_late_gap,
        score["trajectory_var"] > thresholds.trajectory_var,
        score["final_surprise"] > thresholds.final_surprise,
    ])
    conf = signals/4
    if conf >= thresholds.suspicious_frac: return "suspicious", conf
    elif conf >= thresholds.uncertain_frac: return "uncertain", conf
    else: return "normal", conf

# ─── B) Contrastive steering vector ─────────────────────────────────────────

def extract_steering_vector(model, tokenizer, pairs, layer=None):
    """Mean(deceptive residual) - mean(honest residual) at `layer`."""
    if layer is None:
        from config import get_config
        cfg = get_config()
        layer = cfg.mid_layer_for(model.config.num_hidden_layers if hasattr(model.config, "num_hidden_layers") else 28)
    text_module, block_layers = _resolve_blocks(model)
    honest_hs, deceptive_hs = [], []
    for honest, deceptive in pairs:
        for prompt, bucket in [(honest, honest_hs), (deceptive, deceptive_hs)]:
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
            with ActivationRecorder(block_layers, at=[layer]) as rec:
                _record_forward(text_module, model, enc.input_ids)
                h = rec.activations[layer][0, -1, :].float().cpu()
            bucket.append(h)
    h_hon = torch.stack(honest_hs).mean(0)
    h_dec = torch.stack(deceptive_hs).mean(0)
    vec = h_dec - h_hon
    return vec, float(torch.norm(vec).item()), h_hon, h_dec

def patching_test(model, tokenizer, vec, layer=None, alphas=None, n_rand_seeds=3):
    """Real causal patching: vec vs norm-matched random directions.

    For each test prompt, patch the residual at `layer` at the last token:
      h_patched = h + alpha * vec  (and same for random vec of equal norm)
    Uses the exact hook pattern from random_direction_control.py (output[0][0,-1,:] = hp).
    Returns flip rates and full per-prompt results.
    """
    from config import get_config
    cfg = get_config()
    if layer is None:
        layer = cfg.mid_layer_for(model.config.num_hidden_layers if hasattr(model.config, "num_hidden_layers") else 28)
    if alphas is None:
        alphas = cfg.thresholds.patch_alphas
    if n_rand_seeds is None:
        n_rand_seeds = cfg.n_rand_seeds
    test_prompts = IN_DIST[:cfg.n_prompts_per_cat * 2] or IN_DIST[:1]
    vec = vec.float().cpu()
    vec_norm = float(torch.norm(vec).item())
    if vec_norm < 1e-8:
        raise ValueError("steering vec is ~0; patching would be a no-op")
    text_module, block_layers = _resolve_blocks(model)
    if not 0 <= layer < len(block_layers):
        raise ValueError(f"layer {layer} out of range for depth {len(block_layers)}")

    def _logits_top1(ids):
        out = model(ids)
        logits = out.logits[0, -1, :] if hasattr(out, "logits") else model.unembed(out.last_hidden_state)[0, -1, :]
        return int(torch.argmax(logits).item()), logits.float().cpu()

    def _flip_for_direction(direction, prompt_input_ids, clean_top1):
        flips = []
        # capture clean h ONCE (not per alpha)
        with torch.no_grad():
            with ActivationRecorder(block_layers, at=[layer]) as rec:
                _record_forward(text_module, model, prompt_input_ids)
                if layer not in rec.activations:
                    raise RuntimeError(f"hook missed L{layer}: blocks={type(block_layers)} vs forward={type(text_module)}")
                h = rec.activations[layer][0, -1, :].float().cpu()
        for alpha in alphas:
            h_patched = h + alpha * direction
            captured = {"done": False}
            def make_hook(hp):
                def hook_fn(module, inp, output):
                    if not captured["done"]:
                        is_tup = isinstance(output, tuple)
                        t = output[0] if is_tup else output
                        new = t.clone()
                        new[0, -1, :] = hp.to(new.device).to(new.dtype)
                        captured["done"] = True
                        return (new,) + output[1:] if is_tup else new
                return hook_fn
            handle = block_layers[layer].register_forward_hook(make_hook(h_patched))
            try:
                with torch.no_grad():
                    # match recording config (no cache) so hook sees the same output type
                    out = model(prompt_input_ids, use_cache=False) if text_module is not model else model(prompt_input_ids)
                    logits = out.logits[0, -1, :] if hasattr(out, "logits") else model.unembed(out.last_hidden_state)[0, -1, :]
                    patched_top1 = int(torch.argmax(logits).item())
                flips.append(int(clean_top1 != patched_top1))
            finally:
                handle.remove()
            free_mem()
        return flips

    all_vec_flips = []
    all_rand_flips = []  # shape: [n_prompts, n_seeds, n_alphas]
    per_prompt = []
    for prompt in test_prompts:
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
        input_ids = enc.input_ids
        with torch.no_grad():
            clean_top1, _ = _logits_top1(input_ids)
        vec_flips = _flip_for_direction(vec, input_ids, clean_top1)
        all_vec_flips.append(vec_flips)
        # random directions, matched norm (local Generator — no global RNG reset)
        seed_flips = []
        for seed in range(n_rand_seeds):
            g = torch.Generator().manual_seed(cfg.seed + 1000 + seed)
            rand_dir = torch.randn(vec.shape, generator=g, dtype=vec.dtype)
            rand_dir = rand_dir / torch.norm(rand_dir) * vec_norm
            rf = _flip_for_direction(rand_dir, input_ids, clean_top1)
            seed_flips.append(rf)
        # average over seeds
        rand_mean = np.mean(seed_flips, axis=0).tolist()
        all_rand_flips.append(rand_mean)
        per_prompt.append({"prompt": prompt[:60], "vec_flips": vec_flips, "rand_flips": rand_mean, "clean_top1": clean_top1})

    vec_rate = np.mean(all_vec_flips, axis=0).tolist()
    rand_rate = np.mean(all_rand_flips, axis=0).tolist()
    # excess = vec - rand (negative means random flips more — not causally privileged)
    excess = (np.array(vec_rate) - np.array(rand_rate)).tolist()
    return {
        "vec_flip_rate": vec_rate,
        "rand_flip_rate": rand_rate,
        "excess_flip_rate": excess,
        "per_prompt": per_prompt,
        "alphas": alphas,
        "layer": layer,
        "vec_norm": vec_norm,
        "n_prompts": len(test_prompts),
        "n_rand_seeds": n_rand_seeds,
    }

# ─── Visualization ──────────────────────────────────────────────────────────

def plot_trajectories(trajectories, labels, output_dir, title_suffix=""):
    fig, ax = plt.subplots(figsize=(14, 7))
    color_map = {"in_dist": ("#4CAF50","solid"), "deception": ("#F44336","solid"), "eval_awareness": ("#9C27B0","dashed"), "sycophancy": ("#FF9800","dashed"), "refusal": ("#00BCD4","dotted"), "adversarial": ("#E91E63","dashdot"), "contrast_honest": ("#4CAF50","dotted"), "contrast_deceptive": ("#F44336","dotted")}
    for (jl_traj, ll_traj), label in zip(trajectories, labels):
        if "contrast_deceptive" in label:
            cat_type = "contrast_deceptive"
        elif "contrast_honest" in label:
            cat_type = "contrast_honest"
        elif "in_dist" in label:
            cat_type = "in_dist"
        elif "decept" in label:
            cat_type = "deception"
        elif "eval" in label:
            cat_type = "eval_awareness"
        elif "sycophancy" in label:
            cat_type = "sycophancy"
        elif "refusal" in label:
            cat_type = "refusal"
        elif "adversarial" in label:
            cat_type = "adversarial"
        else:
            cat_type = "in_dist"
        color, ls = color_map.get(cat_type, ("#888888","solid"))
        ax.plot(ALL_LAYERS, jl_traj, color=color, linestyle=ls, linewidth=1.5, alpha=0.7, label=f"{label} (JL)")
    ax.set_xlabel("Layer"); ax.set_ylabel("Cosine with Actual Logits"); ax.set_title(f"J-Lens Trajectories {title_suffix}"); ax.legend(fontsize=7, ncol=2, loc="lower left"); ax.set_ylim(-0.4, 1.0); ax.axhline(0, color="gray", ls="--", lw=0.5)
    plt.tight_layout(); plt.savefig(output_dir / "fig_trajectories.png", dpi=150, bbox_inches="tight"); plt.close()

def plot_probe_roc(y_true, y_score, output_dir):
    auc, fpr, tpr = au.roc_auc_and_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"AUC={auc:.2f}")
    ax.plot([0,1],[0,1], color="gray", ls="--")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("Probe ROC (trajectory features)")
    ax.legend(); plt.tight_layout(); plt.savefig(output_dir / "fig_probe_roc.png", dpi=150); plt.close()
    return auc

# ─── Main ────────────────────────────────────────────────────────────────────

def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Phase 2: Deception Detection — Deep")
    ap.add_argument("--model", default=None); ap.add_argument("--n-prompts", type=int, default=None); ap.add_argument("--max-seq-len", type=int, default=None)
    args = ap.parse_args(argv)
    global ALL_LAYERS, MID_LAYER, MAX_SEQ_LEN, N_BOOT
    cfg = _get_cfg(**{k: v for k, v in {"model_id": args.model, "n_prompts_per_cat": args.n_prompts, "max_seq_len": args.max_seq_len}.items() if v is not None})
    MAX_SEQ_LEN = cfg.max_seq_len
    N_BOOT = cfg.n_boot
    print("="*60); print("PHASE 2: Deception Detection — Deep"); print("="*60)
    print(f"Config: model={cfg.model_id} max_seq_len={MAX_SEQ_LEN}")
    # derive model_dir and lens_path from config, not stale module-level constants
    model_dir = str(cfg.local_dir) if cfg.local_dir else str(DEFAULT_MODEL_DIR)
    lens_path = str(cfg.lens) if cfg.lens else str(DEFAULT_LENS_FILE)
    model, tokenizer = load_model(model_dir)
    # derive layers from actual depth
    n_layers = model.config.num_hidden_layers if hasattr(model.config, "num_hidden_layers") else 28
    ALL_LAYERS = list(range(n_layers))
    MID_LAYER = cfg.mid_layer_for(n_layers)
    print(f"Layers: {len(ALL_LAYERS)}  mid={MID_LAYER}")
    lens = jlens.JacobianLens.load(lens_path)
    print(f"Lens: {lens}")

    # A) trajectories for original categories + contrastive
    test_prompts = {
        "in_distribution": IN_DIST[:2],
        "deception": SAFETY_DECEPTION[:2],
        "eval_awareness": SAFETY_EVAL_AWARENESS[:2],
        "sycophancy": SAFETY_SYCOPHANCY[:2],
        "refusal": SAFETY_REFUSAL[:2],
        "adversarial_escalation": ADVERSARIAL_ESCALATION[:2],
        "adversarial_roleplay": ADVERSARIAL_ROLEPLAY[:2],
        "adversarial_obfuscation": ADVERSARIAL_OBFUSCATION[:2],
    }
    contrast_pairs = get_contrastive_pairs()
    contrast_cats = get_contrastive_categories()
    print(f"\nContrastive pairs: {len(contrast_pairs)}")

    print("\n--- A) Extracting trajectories ---")
    all_trajs, all_scores, classifications = {}, {}, {}
    feat_matrix, feat_labels, feat_keys = [], [], []
    W_U = model.lm_head.weight.data.cpu().float()
    for cat_name, prompts in {**test_prompts, **contrast_cats}.items():
        print(f"\n  {cat_name}:")
        for i, prompt in enumerate(prompts):
            jl_traj, ll_traj, _ = get_trajectory(model, tokenizer, prompt, lens, W_U=W_U)
            feats = trajectory_features(jl_traj, ll_traj)
            label, conf = classify_heuristic(feats)
            key = f"{cat_name}_p{i}"
            all_trajs[key] = (jl_traj, ll_traj)
            all_scores.setdefault(cat_name, []).append(feats)
            classifications[key] = (label, conf)
            # for probe: use contrastive labels (honest=0, deceptive=1)
            if "contrast" in cat_name:
                y = 1 if "deceptive" in cat_name else 0
                feat_matrix.append([feats["divergence_area"], feats["mid_late_gap"], feats["trajectory_var"], feats["final_surprise"]])
                feat_labels.append(y); feat_keys.append(key)
            print(f"    [{i+1}] {label}({conf:.2f}) div={feats['divergence_area']:.1f} peak=L{feats['peak_layer']}")
        free_mem()

    # B) steering vector
    print(f"\n--- B) Contrastive steering vector (L{MID_LAYER}) ---")
    vec, vec_norm, h_hon, h_dec = extract_steering_vector(model, tokenizer, contrast_pairs, layer=MID_LAYER)
    print(f"  ||vec||={vec_norm:.2f}  cos(h_hon,h_dec)={torch.nn.functional.cosine_similarity(h_hon, h_dec, dim=0).item():.3f}")
    # cosine of vec with J-Lens transport (guarded: lens may skip MID_LAYER)
    if MID_LAYER in lens.jacobians:
        Jg = lens.jacobians[MID_LAYER].float()
        vec_transported = Jg @ vec
        print(f"  ||J@vec||={torch.norm(vec_transported).item():.2f}  cos(vec,J@vec)={torch.nn.functional.cosine_similarity(vec, vec_transported, dim=0).item():.3f}")
    else:
        print(f"  skip J@vec: L{MID_LAYER} not in lens source_layers {sorted(lens.jacobians.keys())}")

    # C) probe on trajectory features
    print("\n--- C) Probe on trajectory features ---")
    if len(feat_matrix) >= 6:
        X = np.array(feat_matrix, float); y = np.array(feat_labels, int)
        probe = au.fit_linear_probe(X, y, cv=3)
        print(f"  CV AUC: {probe['cv_auc_mean']:.3f} ± {probe['cv_auc_std']:.3f}  aucs={probe['aucs']}")
        # also heuristic ROC for comparison
        heur_scores = [s["divergence_area"] for cat in ["contrast_honest","contrast_deceptive"] for s in all_scores.get(cat,[])]
        heur_labels = [0]*len(all_scores.get("contrast_honest",[])) + [1]*len(all_scores.get("contrast_deceptive",[]))
        if len(heur_scores)==len(heur_labels) and len(set(heur_labels))>1:
            heur_auc, _, _ = au.roc_auc_and_curve(heur_labels, heur_scores)
            print(f"  Heuristic (div_area) AUC: {heur_auc:.3f}")
        else:
            heur_auc = float("nan")
    else:
        probe = {"cv_auc_mean": float("nan"), "cv_auc_std": 0.0, "aucs": [], "coef": None}
        heur_auc = float("nan")
        print("  Not enough contrastive samples for probe")

    # D) probe on mid-layer J-Lens logits (compare to LL baseline)
    print("\n--- D) Mid-layer J-Lens vs LL probe ---")
    # collect J-Lens and LL cos at MID_LAYER as features for deception vs in_dist
    X2, y2 = [], []
    for cat in ["in_distribution", "deception", "adversarial_escalation"]:
        for s in all_scores.get(cat, []):
            X2.append([s["divergence_area"], s["mid_late_gap"], s["trajectory_var"]])
            y2.append(1 if cat != "in_distribution" else 0)
    if len(X2) >= 4:
        X2 = np.array(X2, float); y2 = np.array(y2, int)
        probe2 = au.fit_linear_probe(X2, y2, cv=3)
        print(f"  JL L{MID_LAYER} probe AUC: {probe2['cv_auc_mean']:.3f} ± {probe2['cv_auc_std']:.3f}")
    else:
        probe2 = {"cv_auc_mean": float("nan"), "cv_auc_std": 0.0, "aucs": [], "coef": None}

    # E) Real causal patching: deception vec vs random (matched norm)
    print("\n--- E) Causal patching (real hook, vec vs random) ---")
    try:
        patch_res = patching_test(model, tokenizer, vec, layer=MID_LAYER, n_rand_seeds=3)
        print(f"  Layer {patch_res['layer']}  vec_norm {patch_res['vec_norm']:.2f}  n_prompts {patch_res['n_prompts']}")
        for alpha, vr, rr, ex in zip(patch_res["alphas"], patch_res["vec_flip_rate"], patch_res["rand_flip_rate"], patch_res["excess_flip_rate"]):
            print(f"    alpha={alpha:.1f}: vec={vr:.0%} rand={rr:.0%} excess={ex:+.0%}")
        if patch_res["excess_flip_rate"] and len(patch_res["excess_flip_rate"]) > 1:
            # interpret: negative excess = random flips more than vec (not causally privileged)
            mean_excess = float(np.mean(patch_res["excess_flip_rate"][1:]))
            if abs(mean_excess) < 0.05:
                print(f"  Result: vec NOT causally privileged vs random (mean excess {mean_excess:+.2f}) — matches pilot.")
            else:
                print(f"  Result: vec shows excess over random ({mean_excess:+.2f}) — potential causal handle (needs N>4).")
    except Exception as e:
        print(f"  Patching failed: {e}")
        import traceback; traceback.print_exc()
        patch_res = {"vec_flip_rate": [], "rand_flip_rate": [], "excess_flip_rate": [],
                     "per_prompt": [], "alphas": [], "layer": MID_LAYER, "vec_norm": 0.0,
                     "n_prompts": 0, "n_rand_seeds": 0, "error": str(e)}

    # Plots
    print("\nGenerating plots...")
    labels = list(all_trajs.keys()); trajs = list(all_trajs.values())
    plot_trajectories(trajs, labels, OUTPUT_DIR, f"({cfg.model_id} deep)")
    # probe ROC
    if len(feat_matrix) >= 4 and len(set(feat_labels))>1:
        try:
            # use probe scores as linear projection
            X = np.array(feat_matrix, float)
            # direction from probe coef if available else divergence
            if probe.get("coef") is not None:
                scores = X @ probe["coef"]
            else:
                scores = X[:,0]
            plot_probe_roc(feat_labels, scores, OUTPUT_DIR)
        except Exception as e:
            print(f"ROC plot failed: {e}")
    # deception scores bar with CIs
    try:
        fig, axes = plt.subplots(1, 4, figsize=(20,5))
        cats = list(all_scores.keys())
        for ax, metric in zip(axes, ["divergence_area","mid_late_gap","trajectory_var","final_surprise"]):
            vals = [np.mean([s[metric] for s in all_scores[c]]) for c in cats]
            # bootstrap CI
            los, his = [], []
            for c in cats:
                _, lo, hi = au.bootstrap_ci([s[metric] for s in all_scores[c]], n_boot=N_BOOT)
                los.append(lo); his.append(hi)
            y = np.arange(len(cats))
            ax.errorbar(vals, y, xerr=[np.array(vals)-np.array(los), np.array(his)-np.array(vals)], fmt='o', capsize=3, color="#2196F3")
            ax.set_yticks(y); ax.set_yticklabels([c.replace("_"," ")[:18] for c in cats], fontsize=7)
            ax.set_title(metric.replace("_"," ").title())
        plt.suptitle("Deception Features with 95% CI", y=1.02); plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig_deception_scores.png", dpi=150, bbox_inches="tight"); plt.close()
    except Exception as e:
        print(f"deception scores plot failed: {e}")

    # Summary (tolerant of per-prompt failures in loop A)
    print("\n"+"="*60); print("CLASSIFICATION SUMMARY (heuristic)"); print("="*60)
    for cat_name in test_prompts:
        labs = [classifications.get(f"{cat_name}_p{i}", ("missing", 0.0))[0] for i in range(len(test_prompts[cat_name]))]
        print(f"  {cat_name:<25} suspicious={sum(1 for l in labs if l=='suspicious')} uncertain={sum(1 for l in labs if l=='uncertain')} normal={sum(1 for l in labs if l=='normal')}")

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
        json.dump(ser({
            "scores": all_scores, "classifications": classifications,
            "steering": {"vec_norm": vec_norm, "vec_cos_hon_dec": float(torch.nn.functional.cosine_similarity(h_hon, h_dec, dim=0).item())},
            "probe_trajectory": probe, "probe_mid": probe2, "heuristic_auc": heur_auc,
            "patching": patch_res,
        }), f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
