"""Check if the crossover point (J-Lens advantage vanishes) varies by category.
Reduced to N=2 per category for speed."""
import sys, warnings, os
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")
import torch
import numpy as np
import transformers
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
final_norm = text_module.norm

key_layers = [4, 9, 14, 19, 24, 26]
N = 2

results = {}
for cat_name, prompts in ALL_CATEGORIES.items():
    print(f"\n--- {cat_name} ---", flush=True)
    cat_data = {l: {"jl": [], "ll": []} for l in key_layers}
    
    for i, prompt in enumerate(prompts[:N]):
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=96)
        input_ids = encoded.input_ids
        
        with torch.no_grad():
            out = model(input_ids)
            logits_actual = out.logits[0, -1, :].float()
        
        for layer in key_layers:
            Jg = global_lens.jacobians[layer].float()
            
            with ActivationRecorder(block_layers, at=[layer]) as recorder:
                text_module(input_ids=input_ids, use_cache=False)
                h = recorder.activations[layer][0, -1, :].float()
            
            with torch.no_grad():
                transported = Jg @ h
                t_n = final_norm(transported.unsqueeze(0)).squeeze(0)
                jl_cos = torch.nn.functional.cosine_similarity(W_U @ t_n, logits_actual, dim=0).item()
                ll_cos = torch.nn.functional.cosine_similarity(W_U @ h, logits_actual, dim=0).item()
            
            cat_data[layer]["jl"].append(jl_cos)
            cat_data[layer]["ll"].append(ll_cos)
    
    cat_summary = {}
    for l in key_layers:
        jl_m = float(np.mean(cat_data[l]["jl"]))
        ll_m = float(np.mean(cat_data[l]["ll"]))
        cat_summary[l] = {
            "jl": round(jl_m, 4),
            "ll": round(ll_m, 4),
            "jl_advantage": round(jl_m - ll_m, 4),
        }
    results[cat_name] = cat_summary
    
    advs = " ".join(f"L{l}={cat_summary[l]['jl_advantage']:+.3f}" for l in key_layers)
    print(f"  JL advantage: {advs}", flush=True)

print("\n\nCROSSOVER ANALYSIS (where J-Lens advantage over LL_raw drops below 0.1)")
print("=" * 70)
for cat, data in sorted(results.items()):
    crossover = None
    for l in key_layers:
        if data[l]["jl_advantage"] < 0.1 and crossover is None:
            crossover = l
    if crossover is None:
        crossover_label = f">L{key_layers[-1]}"
    else:
        crossover_label = f"L{crossover}"
    print(f"  {cat:<25} crossover at {crossover_label}")

with open("results/crossover.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to results/crossover.json")
