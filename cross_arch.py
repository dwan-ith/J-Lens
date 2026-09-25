"""
Phase 3: Cross-Architecture Scaling — Deep

Tests proximity artifact + Jacobian subspace structure + CKA.

Adds:
  - Full activation collection for CKA across prompts
  - Jacobian subspace overlap (top-k SVD) across layers
  - Layer-wise correlation with bootstrap CI
  - Transfer significance: does Qwen3-1.7B lens geometry predict itself?

Still batched single-forward per prompt (4.6s on CPU fp16).
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
from safety_prompts import IN_DIST, SAFETY_DECEPTION
import analysis_utils as au

# ─── Config (centralized) ────────────────────────────────────────────────────
from config import get_config as _get_cfg, MODEL_REGISTRY, DEFAULT_MODEL_DIR, DEFAULT_LENS_FILE

_cfg = _get_cfg()
OUTPUT_DIR = Path(os.path.dirname(__file__), "results", "phase3_cross_arch")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Build MODELS from central registry — open models only.
# Closed/API models (Inkling, Inkling-Small, Tinker) have truthy default hf_id
# placeholders but no runnable weights — skip them explicitly instead of
# attempting Hub downloads of gated/nonexistent repos.
MODELS = {}
for _mid, _info in MODEL_REGISTRY.items():
    if _info.get("api_base"):
        continue
    local_exists = bool(_info.get("local_dir") and os.path.exists(str(_info["local_dir"])))
    source = "local" if local_exists else "huggingface"
    MODELS[_mid] = {
        "source": source,
        "path": str(_info["local_dir"]) if local_exists else _info.get("hf_id"),
        "lens": _info.get("lens") or _info.get("lens_file"),
        "hf_id": _info.get("hf_id"),
        "lens_repo": _info.get("lens_repo"),
        "lens_file": _info.get("lens_file"),
    }
# fallback: ensure at least qwen3-1.7b local
if not MODELS:
    MODELS = {"qwen3-1.7b": {"source": "local", "path": str(DEFAULT_MODEL_DIR), "lens": str(DEFAULT_LENS_FILE),
                             "hf_id": "Qwen/Qwen3-1.7B", "lens_repo": None, "lens_file": None}}

N_PROMPTS = _cfg.n_prompts_per_cat
MAX_SEQ_LEN = _cfg.max_seq_len

# ─── Helpers ─────────────────────────────────────────────────────────────────

def cosine_sim(a, b):
    return au.cosine_sim(a, b)

def free_mem():
    au.free_mem()

def load_model_and_lens(model_name, config):
    import transformers
    print(f"\nLoading {model_name}...", flush=True)
    t0 = time.time()
    if config["source"] == "local":
        if not config.get("lens") or not os.path.exists(str(config["lens"])):
            raise FileNotFoundError(f"no local lens file for {model_name}: {config.get('lens')}")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            config["path"], torch_dtype=torch.float16, device_map="cpu",
            low_cpu_mem_usage=True, local_files_only=True,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(config["path"], local_files_only=True)
        lens = jlens.JacobianLens.load(config["lens"])
    else:
        device = "auto" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        _model_id = config.get("hf_id") or config.get("path")
        _lens_repo = config.get("lens_repo") or "neuronpedia/jacobian-lens"
        _lens_file = config.get("lens_file")
        if not _lens_file or os.path.isabs(_lens_file) or _lens_file.startswith("~"):
            raise FileNotFoundError(f"no Hub lens filename for {model_name} (got {_lens_file!r}); refusing to pass a local path as a Hub filename")
        model = transformers.AutoModelForCausalLM.from_pretrained(_model_id, torch_dtype=dtype, device_map=device, low_cpu_mem_usage=True)
        tokenizer = transformers.AutoTokenizer.from_pretrained(_model_id)
        try:
            from huggingface_hub import hf_hub_download
            lens_path = hf_hub_download(repo_id=_lens_repo, filename=_lens_file)
            lens = jlens.JacobianLens.load(lens_path)
        except Exception as e:
            print(f"  WARNING: Could not load lens for {model_name}: {e}")
            lens = None
    model.eval()
    dt = time.time() - t0
    n = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  {n:.1f}B params, d={model.config.hidden_size}, {model.config.num_hidden_layers} layers, {dt:.1f}s", flush=True)
    return model, tokenizer, lens

# ─── Batched measurement + activation collection ─────────────────────────────

def measure_trajectory_and_collect(model, tokenizer, prompts, lens):
    """For each prompt: batched trajectory + collect last-token residuals for CKA.

    Returns:
      per_prompt_trajectories: list of list of {layer, jl_cos, ll_cos}
      stacked_activations: dict layer -> np array (n_prompts, d_model)
      layers_to_check: list of layer indices
    """
    if not prompts:
        raise ValueError("no prompts — check N_PROMPTS / category slices")
    n_layers = model.config.num_hidden_layers
    layers_to_check = _get_cfg().analysis_layers(n_layers)
    # architecture-agnostic block resolution (same pattern as safety_gap.py)
    text_module = getattr(model, "model", model)
    block_layers = getattr(text_module, "layers", getattr(model, "layers", None))
    if block_layers is None:
        raise AttributeError("model has no .layers")
    head = getattr(model, "lm_head", None)
    if head is None:
        raise AttributeError("model has no lm_head")
    W_U = head.weight.data.cpu().float()

    all_trajs = []
    stacked = {l: [] for l in layers_to_check}
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
        input_ids = enc.input_ids
        with torch.no_grad():
            with ActivationRecorder(block_layers, at=layers_to_check) as rec:
                out = model(input_ids)
                logits_actual = out.logits[0, -1, :].cpu().float()
                acts = {l: rec.activations[l].detach().cpu().float() for l in layers_to_check}
        traj = []
        for layer in layers_to_check:
            h = acts[layer][0, -1, :].float().cpu()
            ll_cos = cosine_sim(W_U @ h, logits_actual)
            if lens is not None and layer in lens.jacobians:
                Jg = lens.jacobians[layer].float()
                jl_cos = cosine_sim(W_U @ (Jg @ h), logits_actual)
            else:
                jl_cos = ll_cos
            traj.append({"layer": layer, "jl_cos": jl_cos, "ll_cos": ll_cos})
            # collect for CKA: use last-token residual (1 per prompt) to keep size small
            stacked[layer].append(h.numpy())
        all_trajs.append(traj)
        free_mem()
    # stack
    stacked_np = {l: np.stack(v, axis=0) for l, v in stacked.items()}  # (n_prompts, d)
    return all_trajs, stacked_np, layers_to_check

# ─── Proximity + deeper analyses ────────────────────────────────────────────

def analyze_proximity(all_trajs, model_name, layers=None):
    # flatten across prompts
    flat = [item for traj in all_trajs for item in traj]
    if not flat:
        return {"model": model_name, "layers": [], "jl_mean": [], "ll_mean": [],
                "advantage": [], "adv_ci_lo": [], "adv_ci_hi": [],
                "peak_layer": -1, "peak_advantage": 0.0, "late_advantage": 0.0,
                "artifact_exists": False}
    # group by layer
    by_layer = defaultdict(list)
    for item in flat:
        by_layer[item["layer"]].append(item)
    layers_sorted = sorted(by_layer.keys())
    jl_mean = np.array([np.mean([r["jl_cos"] for r in by_layer[l]]) for l in layers_sorted])
    ll_mean = np.array([np.mean([r["ll_cos"] for r in by_layer[l]]) for l in layers_sorted])
    adv = jl_mean - ll_mean
    peak_idx = int(np.argmax(adv)); peak_layer = layers_sorted[peak_idx]; peak_val = float(adv[peak_idx])
    late_adv = float(np.mean(adv[-3:])) if len(adv)>=3 else (float(np.mean(adv)) if len(adv) > 0 else 0.0)
    _thr = _get_cfg().thresholds
    artifact_exists = bool(peak_val > _thr.artifact_peak_min and late_adv < peak_val * _thr.artifact_late_ratio)
    # bootstrap CI for advantage at each layer
    ci_lo, ci_hi = [], []
    for l in layers_sorted:
        adv_per_prompt = [r["jl_cos"]-r["ll_cos"] for r in by_layer[l]]
        _, lo, hi = au.bootstrap_ci(adv_per_prompt, n_boot=_get_cfg().n_boot)
        ci_lo.append(lo); ci_hi.append(hi)
    return {
        "model": model_name, "layers": layers_sorted,
        "jl_mean": jl_mean.tolist(), "ll_mean": ll_mean.tolist(), "advantage": adv.tolist(),
        "adv_ci_lo": ci_lo, "adv_ci_hi": ci_hi,
        "peak_layer": int(peak_layer), "peak_advantage": round(peak_val,4), "late_advantage": round(late_adv,4),
        "artifact_exists": artifact_exists,
    }

def analyze_subspace(lens, layers, k=None):
    """Top-k subspace overlap — cached SVD (5 SVDs not 25)."""
    if k is None:
        k = _get_cfg().top_k_overlap
    n = len(layers)
    # cache U for each layer (one SVD per layer)
    Us = {}
    for li in layers:
        if lens is not None and li in lens.jacobians:
            Ji = lens.jacobians[li].float().numpy()
            U, _, _ = np.linalg.svd(Ji, full_matrices=False)
            Us[li] = U[:, :k]
    overlap = np.zeros((n, n))
    for i, li in enumerate(layers):
        for j, lj in enumerate(layers):
            if li in Us and lj in Us:
                M = Us[li].T @ Us[lj]  # k x k
                s = np.linalg.svd(M, compute_uv=False)
                overlap[i, j] = float(np.mean(s**2))
            else:
                overlap[i, j] = float("nan")
    return overlap

def analyze_cka(stacked, layers):
    """Linear CKA between residual spaces at different layers."""
    n = len(layers)
    cka = np.zeros((n, n))
    for i, li in enumerate(layers):
        for j, lj in enumerate(layers):
            cka[i, j] = au.linear_cka(stacked[li], stacked[lj])
    return cka

# ─── Plots ───────────────────────────────────────────────────────────────────

def make_plots(all_trajs, stacked, layers, lens, output_dir, model_name="unknown"):
    sns.set_theme(style="whitegrid", font_scale=1.0)
    # Fig1: advantage with CI — x-axis is analysis["layers"] (data-derived), single source of truth
    analysis = analyze_proximity(all_trajs, model_name, layers)
    xs = np.array(analysis["layers"])
    assert len(xs) == len(analysis["advantage"]) == len(analysis["adv_ci_lo"]) == len(analysis["adv_ci_hi"])
    fig, ax = plt.subplots(figsize=(10, 5))
    adv = np.array(analysis["advantage"]); lo = np.array(analysis["adv_ci_lo"]); hi = np.array(analysis["adv_ci_hi"])
    ax.plot(xs, adv, "o-", color="#2196F3", lw=2, markersize=6)
    ax.fill_between(xs, lo, hi, color="#2196F3", alpha=0.2)
    ax.axhline(0, color="gray", ls="--")
    ax.set_xlabel("Layer"); ax.set_ylabel("J-Lens − LogitLens (cos, 95% CI)"); ax.set_title(f"Advantage with Bootstrap CI ({model_name})")
    plt.tight_layout(); plt.savefig(output_dir / f"fig_proximity_artifact_{model_name}.png", dpi=150); plt.close()

    # Fig2: CKA heatmap
    cka = analyze_cka(stacked, layers)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cka, annot=True, fmt=".2f", cmap="viridis", xticklabels=layers, yticklabels=layers, ax=ax, vmin=0, vmax=1)
    ax.set_title(f"Linear CKA: Residual Spaces ({model_name})"); plt.tight_layout(); plt.savefig(output_dir / f"fig_cka_{model_name}.png", dpi=150); plt.close()

    # Fig3: subspace overlap
    overlap = analyze_subspace(lens, layers)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(overlap, annot=True, fmt=".2f", cmap="magma", xticklabels=layers, yticklabels=layers, ax=ax, vmin=0, vmax=1)
    ax.set_title(f"Jacobian Top-32 Subspace Overlap ({model_name})"); plt.tight_layout(); plt.savefig(output_dir / f"fig_subspace_overlap_{model_name}.png", dpi=150); plt.close()

    # Fig4: trajectories (mean)
    fig, ax = plt.subplots(figsize=(10, 5))
    jl_mean = analysis["jl_mean"]; ll_mean = analysis["ll_mean"]
    ax.plot(xs, jl_mean, "o-", label="J-Lens", color="#2196F3", lw=2)
    ax.plot(xs, ll_mean, "s--", label="LogitLens", color="#FF9800", lw=2, alpha=0.7)
    ax.set_xlabel("Layer"); ax.set_ylabel("Cosine with Actual"); ax.set_title(f"Mean Readout Trajectories ({model_name})"); ax.legend()
    plt.tight_layout(); plt.savefig(output_dir / f"fig_cross_arch_trajectories_{model_name}.png", dpi=150); plt.close()
    print(f"Plots saved to {output_dir}")
    return analysis, cka, overlap

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("="*60); print("PHASE 3: Cross-Architecture — Deep"); print("="*60)
    available = {}
    for name, cfg in MODELS.items():
        local_exists = cfg["source"] == "local" and os.path.exists(cfg["path"])
        hf_available = cfg["source"] == "huggingface" and cfg.get("hf_id")
        if local_exists:
            available[name] = cfg; print(f"  {name}: LOCAL")
        elif hf_available:
            available[name] = cfg; print(f"  {name}: HUB ({cfg['hf_id']})")
    if not available:
        print("ERROR: No models"); return
    test_prompts = IN_DIST[:N_PROMPTS] + SAFETY_DECEPTION[:N_PROMPTS]
    print(f"Prompts: {len(test_prompts)}")

    for model_name, cfg in available.items():
        print(f"\n{'='*60}\nTesting {model_name}\n{'='*60}")
        try:
            model, tokenizer, lens = load_model_and_lens(model_name, cfg)
            t0=time.time()
            all_trajs, stacked, layers = measure_trajectory_and_collect(model, tokenizer, test_prompts, lens)
            print(f"Collected {len(all_trajs)} trajectories in {time.time()-t0:.0f}s")
            # flatten for legacy format
            flat = [item for traj in all_trajs for item in traj]
            analysis, cka, overlap = make_plots(all_trajs, stacked, layers, lens, OUTPUT_DIR, model_name=model_name)
            print(f"Peak L{analysis['peak_layer']} +{analysis['peak_advantage']:.3f} late {analysis['late_advantage']:.3f} artifact={analysis['artifact_exists']}")
            # Save (per-model filename to avoid overwrite)
            def ser(o):
                if isinstance(o, torch.Tensor): return o.tolist()
                if isinstance(o, (np.integer, np.floating)): return o.item()
                if isinstance(o, np.ndarray): return o.tolist()
                if isinstance(o, defaultdict): return {str(k): ser(v) for k, v in o.items()}
                if isinstance(o, dict): return {str(k): ser(v) for k, v in o.items()}
                if isinstance(o, list): return [ser(v) for v in o]
                return o
            with open(OUTPUT_DIR / f"results_{model_name}.json", "w") as f:
                json.dump(ser({"trajectories": flat, "analysis": analysis, "cka": cka, "overlap": overlap, "layers": layers}), f, indent=2)
            print(f"Results saved to {OUTPUT_DIR}")
            # summary
            print("\n"+"="*60); print("SUMMARY"); print("="*60)
            print(f"  peak L{analysis['peak_layer']} adv {analysis['peak_advantage']:+.3f} artifact {'EXISTS' if analysis['artifact_exists'] else 'NOT FOUND'}")
            print(f"  mean CKA (adjacent layers) {np.mean([cka[i,i+1] for i in range(len(layers)-1)]):.3f}")
            print(f"  mean subspace overlap (k=32) {np.nanmean(overlap):.3f}")
            del model, tokenizer, lens; free_mem()
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"ERROR: {e}")

if __name__ == "__main__":
    main()
