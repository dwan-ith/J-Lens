"""Get LL_raw at L14 for ALL 7 categories to compute n=7 correlation."""
import sys, warnings, os
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")
import torch
import numpy as np
import transformers

sys.path.insert(0, r"C:\Users\aacer\Desktop\MATS\jacobian-lens")
from jlens.hooks import ActivationRecorder
from prompts import ALL_CATEGORIES

MODEL_DIR = r"C:\Users\aacer\Desktop\MATS\model"

model = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, dtype=torch.float32, device_map="cpu", local_files_only=True)
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model.eval()

text_module = model.model
block_layers = text_module.layers
W_U = model.lm_head.weight.data.float()

# Just L14, 2 prompts per category, compute LL_raw only
layer = 14
N = 2

results = {}
for cat_name, prompts in ALL_CATEGORIES.items():
    ll_vals = []
    for prompt in prompts[:N]:
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=96)
        input_ids = encoded.input_ids
        with torch.no_grad():
            logits_actual = model(input_ids).logits[0, -1, :].float()
        with ActivationRecorder(block_layers, at=[layer]) as recorder:
            text_module(input_ids=input_ids, use_cache=False)
            h = recorder.activations[layer][0, -1, :].float()
        ll_cos = torch.nn.functional.cosine_similarity(W_U @ h, logits_actual, dim=0).item()
        ll_vals.append(ll_cos)
    mean_ll = float(np.mean(ll_vals))
    results[cat_name] = round(mean_ll, 4)
    print(f"{cat_name:<25} LL_raw@L14 = {mean_ll:+.4f}  (n={N})")

print("\nAll LL_raw@L14 values:")
for cat, val in sorted(results.items()):
    print(f"  {cat:<25} {val:+.4f}")
