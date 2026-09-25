"""Check if Jacobian fidelity curve is a proximity artifact.
Compare J-Lens vs identity (I) Jacobian vs random baseline."""
import sys, warnings, os
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")
import torch
import numpy as np
import transformers
from collections import defaultdict

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
n_layers = model.config.num_hidden_layers
text_module = model.model
block_layers = text_module.layers

print(f"n_layers={n_layers}, len(block_layers)={len(block_layers)}")
W_U = model.lm_head.weight.data.float()
final_norm = text_module.norm

# The identity Jacobian: I(d_model) — no transformation
# This is the "do nothing" baseline: just use the hidden state as-is
# For readout: norm(h) @ W_U (logit-lens without Jacobian)
# vs norm(I @ h) @ W_U = norm(h) @ W_U (same thing)

# For Jacobian cosine: I @ h vs J_global @ h
# cos(I, J_global) at each layer

prompts = [
    "The capital of France is Paris, which is known for the Eiffel Tower.",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "I saw her duck behind the fence when the rain started pouring down.",
    "This is a safety evaluation. Please respond honestly to the following test.",
    "Let us solve this step by step. First find the derivative of f(x) = 3x^2.",
]

key_layers = [4, 9, 14, 19, 24, 26, 27]

print(f"{'Layer':>5}  {'J-Lens':>8}  {'Identity':>8}  {'LogitLens':>10}  {'LL_no_norm':>10}")
print("-" * 60)

for layer in key_layers:
    Jg = global_lens.jacobians[layer].float()
    I_mat = torch.eye(d_model)
    
    # Jacobian cosine: how similar is each method to the true local Jacobian?
    # For identity: cos(I, J_global) — just a constant, same for all prompts
    cos_jl_list = []
    cos_id_list = []
    cos_ll_list = []
    cos_ll_nn_list = []
    
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=96)
        input_ids = encoded.input_ids
        
        with ActivationRecorder(block_layers, at=[layer]) as recorder:
            text_module(input_ids=input_ids, use_cache=False)
            h_source = recorder.activations[layer][0, -1, :].float()
        
        with torch.no_grad():
            # True local Jacobian direction: norm(J @ h) @ W_U
            transported_jl = Jg @ h_source
            h_jl_normed = final_norm(transported_jl.unsqueeze(0)).squeeze(0)
            logits_jl = W_U @ h_jl_normed
            
            # Identity: norm(h) @ W_U (logit-lens)
            h_normed = final_norm(h_source.unsqueeze(0)).squeeze(0)
            logits_ll = W_U @ h_normed
            
            # Logit-lens without norm: h @ W_U
            logits_ll_nn = W_U @ h_source
            
            # Actual model output
            out = model(input_ids)
            logits_actual = out.logits[0, -1, :].float()
        
        cos_jl = torch.nn.functional.cosine_similarity(logits_jl, logits_actual, dim=0).item()
        cos_ll = torch.nn.functional.cosine_similarity(logits_ll, logits_actual, dim=0).item()
        cos_ll_nn = torch.nn.functional.cosine_similarity(logits_ll_nn, logits_actual, dim=0).item()
        
        cos_jl_list.append(cos_jl)
        cos_ll_list.append(cos_ll)
        cos_ll_nn_list.append(cos_ll_nn)
        
        del h_source, transported_jl, h_jl_normed, h_normed
    
    # Identity Jacobian cosine with J_global (constant across prompts)
    cos_id = torch.nn.functional.cosine_similarity(I_mat.flatten(), Jg.flatten(), dim=0).item()
    
    mean_jl = np.mean(cos_jl_list)
    mean_ll = np.mean(cos_ll_list)
    mean_ll_nn = np.mean(cos_ll_nn_list)
    
    print(f"L{layer:2d}   {mean_jl:+.4f}  {cos_id:+.4f}  {mean_ll:+.4f}    {mean_ll_nn:+.4f}")

# Also check: what's the cosine of J_global with itself at each layer? (sanity)
print(f"\nJ_global self-cosine (should be 1.0):")
for layer in key_layers:
    Jg = global_lens.jacobians[layer].float()
    cos_self = torch.nn.functional.cosine_similarity(Jg.flatten(), Jg.flatten(), dim=0).item()
    print(f"  L{layer}: {cos_self:.4f}")
