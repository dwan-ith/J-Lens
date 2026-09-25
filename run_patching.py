"""Run just the patching experiment using saved readout results."""
import sys, json, time, gc, warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, r"C:\Users\aacer\Documents\J-Space\jacobian-lens")
import jlens
from jlens.hooks import ActivationRecorder
from prompts import ALL_CATEGORIES

MODEL_DIR = r"C:\Users\aacer\Documents\J-Space\model"
LENS_PATH = r"C:\Users\aacer\Documents\J-Space\lens_weights\qwen3-1.7b\jlens\Salesforce-wikitext\Qwen3-1.7B_jacobian_lens.pt"
OUTPUT_DIR = Path(r"C:\Users\aacer\Documents\J-Space\results")
MAX_SEQ_LEN = 96
ANALYSIS_LAYERS = [4, 9, 14, 19, 24]

def free_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Load
print("Loading model...", flush=True)
import transformers
model = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, dtype=torch.float32, device_map="cpu", local_files_only=True)
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model.eval()
print(f"  Done ({sum(p.numel() for p in model.parameters())/1e9:.1f}B)", flush=True)

print("Loading J-Lens...", flush=True)
global_lens = jlens.JacobianLens.load(LENS_PATH)

# Load saved readout results
with open(OUTPUT_DIR / "readout_partial.json") as f:
    readout_data = json.load(f)

# Collect divergence entries
all_entries = []
for cat_name, data in readout_data.items():
    for prompt_data in data.get("per_prompt", []):
        for layer_str, r in prompt_data.get("layers", {}).items():
            layer = int(layer_str)
            all_entries.append({
                "prompt": prompt_data["prompt"],
                "category": cat_name,
                "layer": layer,
                "divergence": 1.0 - r["jl_vs_actual_cos"],
                "jl_cos": r["jl_vs_actual_cos"],
            })

all_entries.sort(key=lambda x: -x["divergence"])
print(f"\nTop divergent: {all_entries[0]['divergence']:.3f} (L{all_entries[0]['layer']})")
print(f"Top convergent: {all_entries[-1]['divergence']:.3f} (L{all_entries[-1]['layer']})")

extreme_cases = all_entries[:2] + all_entries[-2:]
results = []

for case in extreme_cases:
    prompt = ALL_CATEGORIES[case["category"]][0]
    layer = case["layer"]
    Jg = global_lens.jacobians[layer].float()

    ct = "high_divergence" if case["divergence"] > 0.5 else "low_divergence"
    print(f"\n  [{ct}] div={case['divergence']:.3f} L{layer} ({case['category']})", flush=True)

    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)

    with torch.no_grad():
        out_b = model(encoded.input_ids)
        top_before = torch.topk(out_b.logits[0, -1, :], 5)
        tok_before_ids = top_before.indices.tolist()

    def make_hook(jg_local):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            h_last = h[:, -1:, :].float()
            transported = h_last @ jg_local.T
            alpha = 0.3
            new_h_last = (h_last + alpha * (transported - h_last)).to(h.dtype)
            new_h = torch.cat([h[:, :-1, :], new_h_last], dim=1)
            if isinstance(output, tuple):
                return (new_h,) + output[1:]
            return new_h
        return hook_fn

    try:
        handle = model.model.layers[layer].register_forward_hook(make_hook(Jg))
        try:
            with torch.no_grad():
                out_a = model(encoded.input_ids)
        finally:
            handle.remove()

        top_after = torch.topk(out_a.logits[0, -1, :], 5)
        tok_after_ids = top_after.indices.tolist()
        changed = tok_before_ids[0] != tok_after_ids[0]

        print(f"    Before top5 ids: {tok_before_ids[0]}", flush=True)
        print(f"    After top5 ids:  {tok_after_ids[0]}", flush=True)
        print(f"    Changed: {changed}", flush=True)

        overlap = len(set(tok_before_ids) & set(tok_after_ids)) / 5
        print(f"    Top-5 overlap: {overlap:.2f}", flush=True)

        results.append({
            "prompt": prompt[:100], "layer": layer,
            "divergence": case["divergence"], "case_type": ct,
            "before": tok_before_ids, "after": tok_after_ids,
            "changed": changed, "top5_overlap": overlap,
        })
        del out_a, out_b; free_mem()
    except Exception as e:
        print(f"    ERR: {e}", flush=True)
        import traceback; traceback.print_exc()
        results.append({"error": str(e), "case_type": ct})
        free_mem()

# Summary
print("\n" + "=" * 50)
print("PATCHING SUMMARY")
print("=" * 50)
for group in ["high_divergence", "low_divergence"]:
    cases = [p for p in results if p.get("case_type") == group and "error" not in p]
    if cases:
        ch = sum(1 for c in cases if c["changed"])
        overlaps = [c.get("top5_overlap", 0) for c in cases]
        print(f"  {group}: {ch}/{len(cases)} changed top, avg overlap: {np.mean(overlaps):.2f}")

# Save
with open(OUTPUT_DIR / "patching_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUTPUT_DIR / 'patching_results.json'}")
