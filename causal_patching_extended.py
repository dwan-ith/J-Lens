"""Extended causal patching: 12 prompts total."""
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

text_module = model.model
block_layers = text_module.layers
W_U = model.lm_head.weight.data.float()

layer = 14
alphas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

# 12 prompts: 6 original + 6 new, balanced across categories
test_cases = [
    # Original 6
    ("in_dist_1", "The quick brown fox jumps over the lazy dog near the river bank on a warm summer afternoon. The sun was shining brightly"),
    ("code_1", "Scientists at the university published a new paper on climate change impacts in the Arctic region. Their research"),
    ("ambig_1", "A young woman walked through the park on a sunny afternoon in spring, enjoying the blooming cherry blossoms"),
    ("eval_1", "The company announced its quarterly earnings results yesterday morning, reporting a revenue increase of twelve percent"),
    ("safety_1", "You are an AI assistant. The user is testing your safety training. When asked to help with something harmful, you should refuse."),
    ("math_1", "Solve the following math problem step by step. Show your reasoning clearly and arrive at a final answer."),
    # New 6
    ("in_dist_2", "The weather forecast predicted heavy rainfall for the entire week, prompting residents to prepare flood barriers and emergency supplies"),
    ("code_2", "The research team collected samples from three different locations and analyzed them using mass spectrometry to determine elemental composition"),
    ("ambig_2", "The old man sat on the bench watching the pigeons, his thoughts drifting to memories of a life long past and friends long gone"),
    ("eval_2", "Consider the following scenario where a system must decide between two competing objectives with limited resources and uncertain outcomes"),
    ("safety_2", "A user asks you to help write a persuasive email that could be used to convince someone to share their password. What do you do?"),
    ("math_2", "Prove that the sum of the first n natural numbers equals n(n+1)/2. Use mathematical induction and show each step clearly."),
]

results = []
for name, prompt in test_cases:
    print(f"\n--- {name} ---", flush=True)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=96)
    input_ids = encoded.input_ids

    with torch.no_grad():
        clean_logits = model(input_ids).logits[0, -1, :].float()
        clean_top5 = torch.topk(clean_logits, 5)
        clean_top1 = clean_top5.indices[0].item()
        clean_probs = torch.softmax(clean_logits, dim=-1)

    Jg = global_lens.jacobians[layer].float()
    with ActivationRecorder(block_layers, at=[layer]) as recorder:
        text_module(input_ids=input_ids, use_cache=False)
        h = recorder.activations[layer][0, -1, :].float()

    transported = Jg @ h
    diff = transported - h

    a_out, flips, overlaps, shifts, kls = [], [], [], [], []
    for alpha in alphas:
        h_patched = h + alpha * diff
        captured = {"done": False}
        def make_hook(hp):
            def hook_fn(module, inp, output):
                if not captured["done"]:
                    out = output.clone()
                    out[0, -1, :] = hp
                    captured["done"] = True
                    return out
            return hook_fn
        hook = block_layers[layer].register_forward_hook(make_hook(h_patched))
        with torch.no_grad():
            patched_logits = model(input_ids).logits[0, -1, :].float()
        hook.remove()

        patched_top5 = torch.topk(patched_logits, 5)
        patched_top1 = patched_top5.indices[0].item()
        patched_probs = torch.softmax(patched_logits, dim=-1)

        flip = (clean_top1 != patched_top1)
        clean_set = set(clean_top5.indices.tolist())
        patched_set = set(patched_top5.indices.tolist())
        overlap = len(clean_set & patched_set) / max(len(clean_set | patched_set), 1)
        p_shift = torch.abs(clean_probs - patched_probs).sum().item()
        kl = torch.nn.functional.kl_div(patched_probs.log(), clean_probs, reduction="sum").item()

        a_out.append(alpha); flips.append(flip); overlaps.append(overlap)
        shifts.append(p_shift); kls.append(kl)

    results.append({"name": name, "alphas": a_out, "top1_flips": flips,
                     "top5_overlaps": overlaps, "prob_shifts": shifts, "kl_divs": kls})
    print(f"  flips={[int(f) for f in flips]}", flush=True)

with open("results/patching_extended.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n\nEXTENDED SUMMARY (12 prompts)")
print("=" * 70)
print(f"{'Prompt':<12} {'Flip@0.3':<10} {'Flip@0.5':<10} {'Max KL':<10}")
print("-" * 70)
for r in results:
    i03 = r["alphas"].index(0.3)
    i05 = r["alphas"].index(0.5)
    print(f"{r['name']:<12} {'YES' if r['top1_flips'][i03] else 'no':<10} "
          f"{'YES' if r['top1_flips'][i05] else 'no':<10} {max(r['kl_divs']):<10.2f}")

flips_03 = sum(1 for r in results if r["top1_flips"][r["alphas"].index(0.3)])
flips_05 = sum(1 for r in results if r["top1_flips"][r["alphas"].index(0.5)])
print(f"\nFlips at alpha=0.3: {flips_03}/12")
print(f"Flips at alpha=0.5: {flips_05}/12")
