"""Final comprehensive debug: norm path + identity baseline."""
import sys, warnings, os
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")
import torch
import numpy as np
import transformers

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
final_norm = text_module.norm

prompts = [
    "The capital of France is Paris, which is known for the Eiffel Tower.",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "I saw her duck behind the fence when the rain started pouring down.",
    "This is a safety evaluation. Please respond honestly to the following test.",
    "Let us solve this step by step. First find the derivative of f(x) = 3x^2.",
]

key_layers = [0, 4, 9, 14, 19, 24, 26]

print("TEST 1: What does final_norm actually do to the representation?")
print("=" * 60)
# Check final_norm weights
fn_weight = final_norm.weight.data.float()
print(f"final_norm weight stats: mean={fn_weight.mean():.4f}, std={fn_weight.std():.4f}, min={fn_weight.min():.4f}, max={fn_weight.max():.4f}")

# Check if norm preserves cosine similarity
print(f"\nRMSNorm preserves cosine? Test on random vectors:")
a = torch.randn(d_model)
b = torch.randn(d_model)
cos_before = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
a_n = final_norm(a.unsqueeze(0)).squeeze(0)
b_n = final_norm(b.unsqueeze(0)).squeeze(0)
cos_after = torch.nn.functional.cosine_similarity(a_n, b_n, dim=0).item()
print(f"  cos(a, b) = {cos_before:.6f}")
print(f"  cos(norm(a), norm(b)) = {cos_after:.6f}")
print(f"  Preserved: {abs(cos_before - cos_after) < 1e-4}")

# So norm PRESERVES cosine similarity between two vectors of the same shape.
# But our logit-lens computes cos(norm(h) @ W_U, actual_logits)
# The actual_logits are W_U @ norm(h_final)
# So we're computing cos(W_U @ norm(h_intermediate), W_U @ norm(h_final))
# This is NOT the same as cos(h_intermediate, h_final) because W_U is not orthogonal.

print(f"\n\nTEST 2: Layer-by-layer readout comparison")
print("=" * 60)
print(f"{'Layer':>5}  {'JL_norm':>8}  {'JL_raw':>8}  {'LL_norm':>8}  {'LL_raw':>8}  {'ident':>8}")
print("-" * 65)

for layer in key_layers:
    Jg = global_lens.jacobians[layer].float()
    
    jl_norms, jl_raws, ll_norms, ll_raws = [], [], [], []
    
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=96)
        input_ids = encoded.input_ids
        
        with ActivationRecorder(block_layers, at=[layer]) as recorder:
            text_module(input_ids=input_ids, use_cache=False)
            h = recorder.activations[layer][0, -1, :].float()
        
        with torch.no_grad():
            out = model(input_ids)
            logits_actual = out.logits[0, -1, :].float()
            
            # J-Lens with norm
            transported = Jg @ h
            t_n = final_norm(transported.unsqueeze(0)).squeeze(0)
            jl_norm = torch.nn.functional.cosine_similarity(W_U @ t_n, logits_actual, dim=0).item()
            
            # J-Lens without norm
            jl_raw = torch.nn.functional.cosine_similarity(W_U @ transported, logits_actual, dim=0).item()
            
            # Logit-lens with norm
            h_n = final_norm(h.unsqueeze(0)).squeeze(0)
            ll_norm = torch.nn.functional.cosine_similarity(W_U @ h_n, logits_actual, dim=0).item()
            
            # Logit-lens without norm
            ll_raw = torch.nn.functional.cosine_similarity(W_U @ h, logits_actual, dim=0).item()
        
        jl_norms.append(jl_norm)
        jl_raws.append(jl_raw)
        ll_norms.append(ll_norm)
        ll_raws.append(ll_raw)
    
    # Identity Jacobian cosine with J_global (structural)
    ident = torch.nn.functional.cosine_similarity(
        torch.eye(d_model).flatten(), Jg.flatten(), dim=0).item()
    
    print(f"L{layer:2d}   {np.mean(jl_norms):+.4f}  {np.mean(jl_raws):+.4f}  "
          f"{np.mean(ll_norms):+.4f}  {np.mean(ll_raws):+.4f}  {ident:+.4f}")

print(f"\n\nTEST 3: What if the norm is applied to the ACTUAL logits too?")
print("=" * 60)
# Check: are the actual model logits computed with or without norm?
# Model forward: h_final -> norm -> W_U -> logits
# So actual_logits = W_U @ norm(h_final)
# Our logit-lens should compute: W_U @ norm(h_intermediate)
# These are comparable IF the norm doesn't distort the direction

# Check: cos(W_U @ norm(h_final), W_U @ norm(h_final)) should be 1.0
prompt = prompts[0]
encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=96)
input_ids = encoded.input_ids

with torch.no_grad():
    out = model(input_ids, output_hidden_states=True)
    logits_actual = out.logits[0, -1, :].float()
    hs = out.hidden_states

# Last hidden state
h_last = hs[-1][0, -1, :].float()
h_last_n = final_norm(h_last.unsqueeze(0)).squeeze(0)
logits_reconstructed = W_U @ h_last_n
cos_recon = torch.nn.functional.cosine_similarity(logits_reconstructed, logits_actual, dim=0).item()
print(f"Reconstruct from hidden_states[-1] + norm: cos={cos_recon:.6f}")

# Without norm
logits_no_norm = W_U @ h_last
cos_no_norm = torch.nn.functional.cosine_similarity(logits_no_norm, logits_actual, dim=0).item()
print(f"Reconstruct from hidden_states[-1] no norm: cos={cos_no_norm:.6f}")

# The actual model uses: h_last -> norm -> W_U
# So cos(W_U @ norm(h_last), actual) should be 1.0
# But it's 0.885... why?

# Check: maybe the model applies norm to ALL positions, not just the last?
with torch.no_grad():
    out2 = model(input_ids)
    logits_full = out2.logits[0, -1, :]
    
    # Manual: norm the full hidden state, then project
    h_full = hs[-1]  # [1, seq_len, d_model]
    h_full_n = final_norm(h_full)  # norm applied to all positions
    logits_manual = W_U @ h_full_n[0, -1, :]
    cos_manual = torch.nn.functional.cosine_similarity(logits_manual, logits_actual, dim=0).item()
    print(f"Manual norm of full hidden state: cos={cos_manual:.6f}")

# Maybe the issue is float precision?
h_last_fp16 = hs[-1][0, -1, :].half().float()
h_last_n_fp16 = final_norm(h_last_fp16.unsqueeze(0)).squeeze(0)
logits_fp16 = W_U @ h_last_n_fp16
cos_fp16 = torch.nn.functional.cosine_similarity(logits_fp16, logits_actual, dim=0).item()
print(f"FP16 roundtrip: cos={cos_fp16:.6f}")
