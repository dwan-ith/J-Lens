"""Causal patching: alpha sweep with hook (correct output handling)."""
import sys, warnings, os
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")
import torch
import numpy as np
import transformers
import json

sys.path.insert(0, r"C:\Users\aacer\Documents\J-Space\jacobian-lens")
import jlens

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

test_cases = [
    ("high_div_1", "The quick brown fox jumps over the lazy dog near the river bank on a warm summer afternoon. The sun was shining brightly"),
    ("high_div_2", "Scientists at the university published a new paper on climate change impacts in the Arctic region. Their research"),
    ("high_div_3", "A young woman walked through the park on a sunny afternoon in spring, enjoying the blooming cherry blossoms"),
    ("high_div_4", "The company announced its quarterly earnings results yesterday morning, reporting a revenue increase of twelve percent"),
    ("low_div_1", "You are an AI assistant. The user is testing your safety training. When asked to help with something harmful, you should refuse."),
    ("low_div_2", "Solve the following math problem step by step. Show your reasoning clearly and arrive at a final answer."),
]

results = []

for name, prompt in test_cases:
    print(f"\n--- {name} ---", flush=True)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=96)
    input_ids = encoded.input_ids

    # Get clean logits
    with torch.no_grad():
        clean_logits = model(input_ids).logits[0, -1, :].float()
        clean_top5 = torch.topk(clean_logits, 5)
        clean_top1 = clean_top5.indices[0].item()
        clean_probs = torch.softmax(clean_logits, dim=-1)

    # Get J-Lens direction at layer 14
    from jlens.hooks import ActivationRecorder
    Jg = global_lens.jacobians[layer].float()
    with ActivationRecorder(block_layers, at=[layer]) as recorder:
        text_module(input_ids=input_ids, use_cache=False)
        h = recorder.activations[layer][0, -1, :].float()

    transported = Jg @ h
    diff = transported - h

    alphas_out, flips, overlaps, shifts, kls = [], [], [], [], []

    for alpha in alphas:
        h_patched = h + alpha * diff

        captured = {"done": False}
        def make_hook(h_patched):
            def hook_fn(module, inp, output):
                if not captured["done"]:
                    out = output.clone()
                    out[0, -1, :] = h_patched
                    captured["done"] = True
                    return out
            return hook_fn

        hook = block_layers[layer].register_forward_hook(make_hook(h_patched))

        with torch.no_grad():
            patched_out = model(input_ids)
            patched_logits = patched_out.logits[0, -1, :].float()

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

        alphas_out.append(alpha)
        flips.append(flip)
        overlaps.append(overlap)
        shifts.append(p_shift)
        kls.append(kl)

        print(f"  a={alpha:.1f}: flip={flip}  overlap={overlap:.2f}  P_shift={p_shift:.4f}  KL={kl:.2f}", flush=True)

    results.append({
        "name": name,
        "alphas": alphas_out, "top1_flips": flips,
        "top5_overlaps": overlaps, "prob_shifts": shifts, "kl_divs": kls,
    })

with open("results/patching_sweep.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n\nSUMMARY")
print("=" * 60)
for r in results:
    fl = sum(r["top1_flips"])
    ms = max(r["prob_shifts"])
    mk = max(r["kl_divs"])
    print(f"  {r['name']:<12} flips={fl}/{len(alphas)}  max_P={ms:.4f}  max_KL={mk:.2f}")
