"""Run the corrected comparison for the 4 remaining categories."""
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

# The 4 remaining categories
remaining_cats = {
    "eval_awareness": [
        "You are an AI assistant. The user is testing your safety training. When asked to help with something harmful, you should refuse.",
        "The following conversation is between a user and a helpful AI assistant. The assistant is designed to be safe and honest.",
    ],
    "math_cot": [
        "Solve the following math problem step by step. Show your reasoning clearly and arrive at a final answer.",
        "Consider the following mathematical proof. Walk through each step carefully and verify the logic.",
    ],
    "deception_scenarios": [
        "A person is trying to convince you of something that may not be true. They present several pieces of evidence. Evaluate whether their claims are supported.",
        "Consider a scenario where someone has an incentive to misrepresent facts. How would you verify their statements independently?",
    ],
    "multi_turn_agentic": [
        "You are an AI agent with access to various tools. The user asks you to complete a complex task that requires multiple steps and tool use.",
        "You are a helpful AI assistant managing a multi-step workflow. The user wants you to coordinate several tasks in sequence.",
    ],
}

results = {}
for cat_name, prompts in remaining_cats.items():
    print(f"\n--- {cat_name} ---", flush=True)
    cat_data = {l: {"jl_raw": [], "ll_raw": []} for l in key_layers}
    
    for i, prompt in enumerate(prompts):
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
                # J-Lens with norm (original pipeline)
                transported = Jg @ h
                # Use raw: no norm applied
                jl_raw_cos = torch.nn.functional.cosine_similarity(W_U @ transported, logits_actual, dim=0).item()
                # Logit-lens raw (no norm)
                ll_raw_cos = torch.nn.functional.cosine_similarity(W_U @ h, logits_actual, dim=0).item()
            
            cat_data[layer]["jl_raw"].append(jl_raw_cos)
            cat_data[layer]["ll_raw"].append(ll_raw_cos)
            print(f"  L{layer}: JL_raw={jl_raw_cos:+.3f}  LL_raw={ll_raw_cos:+.3f}  adv={jl_raw_cos - ll_raw_cos:+.3f}", flush=True)
    
    cat_summary = {}
    for l in key_layers:
        jl_m = float(np.mean(cat_data[l]["jl_raw"]))
        ll_m = float(np.mean(cat_data[l]["ll_raw"]))
        cat_summary[l] = {
            "jl_raw": round(jl_m, 4),
            "ll_raw": round(ll_m, 4),
            "advantage": round(jl_m - ll_m, 4),
        }
    results[cat_name] = cat_summary
    
    # Find crossover
    crossover = None
    for l in key_layers:
        if cat_summary[l]["advantage"] < 0.1 and crossover is None:
            crossover = l
    crossover_label = f"L{crossover}" if crossover else f">L{key_layers[-1]}"
    advs = " ".join(f"L{l}={cat_summary[l]['advantage']:+.3f}" for l in key_layers)
    print(f"  Summary: {advs}  crossover={crossover_label}", flush=True)

# Combine with previously computed results
prev = {
    "in_distribution_generic": {"crossover": "L26"},
    "code_text": {"crossover": "L26"},
    "ambiguous_sentences": {"crossover": "L26"},
}

print("\n\nCROSSOVER SUMMARY (all 7 categories)")
print("=" * 70)
for cat, data in results.items():
    print(f"  {cat:<25} crossover at {data.get('crossover', '?')}")

with open("results/crossover_remaining.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to results/crossover_remaining.json")
