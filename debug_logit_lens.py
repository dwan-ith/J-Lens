"""Debug: check logit lens at key layers only (fast)."""
import sys, warnings, os
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")
import torch
import transformers

sys.path.insert(0, r"C:\Users\aacer\Documents\J-Space\jacobian-lens")
from jlens.hooks import ActivationRecorder

MODEL_DIR = r"C:\Users\aacer\Documents\J-Space\model"

model = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, dtype=torch.float32, device_map="cpu", local_files_only=True)
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model.eval()

prompt = "The capital of France is"
encoded = tokenizer(prompt, return_tensors="pt")
input_ids = encoded.input_ids

# Actual model output
with torch.no_grad():
    out = model(input_ids)
    logits_actual = out.logits[0, -1, :]
    top5_actual = torch.topk(logits_actual, 5)
    print(f"Actual top-5 ids: {top5_actual.indices.tolist()}")

text_module = model.model
block_layers = text_module.layers
W_U = model.lm_head.weight.data.float()
final_norm = text_module.norm

# Also get hidden states via output_hidden_states
with torch.no_grad():
    out_full = model(input_ids, output_hidden_states=True)
    hs = out_full.hidden_states  # (n_layers+1,) tensors

print(f"\nhidden_states count: {len(hs)} (0=embed, 1..28=layer outputs)")

# Check key layers
key_layers = [0, 4, 8, 13, 18, 22, 25, 26, 27]
print(f"\n{'Layer':>5}  {'norm(h)':>8}  {'h_only':>8}  {'hs_norm':>8}  {'hs_only':>8}")
print("-" * 50)

for L in key_layers:
    # Via ActivationRecorder (our pipeline's path)
    with ActivationRecorder(block_layers, at=[L]) as recorder:
        text_module(input_ids=input_ids, use_cache=False)
        h_rec = recorder.activations[L][0, -1, :].float()

    h_nA = final_norm(h_rec.unsqueeze(0)).squeeze(0)
    logits_A = W_U @ h_nA
    cos_A = torch.nn.functional.cosine_similarity(logits_A, logits_actual, dim=0).item()

    logits_B = W_U @ h_rec
    cos_B = torch.nn.functional.cosine_similarity(logits_B, logits_actual, dim=0).item()

    # Via hidden_states (ground truth path)
    h_hs = hs[L + 1][0, -1, :].float()  # +1 because hs[0] is embedding
    h_hs_n = final_norm(h_hs.unsqueeze(0)).squeeze(0)
    logits_hs_n = W_U @ h_hs_n
    cos_hs_n = torch.nn.functional.cosine_similarity(logits_hs_n, logits_actual, dim=0).item()

    logits_hs = W_U @ h_hs
    cos_hs = torch.nn.functional.cosine_similarity(logits_hs, logits_actual, dim=0).item()

    # Check if recorder and hidden_states agree
    diff = (h_rec - h_hs).abs().max().item()

    print(f"L{L:2d}   {cos_A:+.4f}  {cos_B:+.4f}  {cos_hs_n:+.4f}  {cos_hs:+.4f}   (diff={diff:.6f})")

# Final check: the actual last hidden state
print(f"\n=== FINAL LAYER CHECK ===")
h_last = hs[-1][0, -1, :].float()
h_last_n = final_norm(h_last.unsqueeze(0)).squeeze(0)
logits_final = W_U @ h_last_n
cos_final = torch.nn.functional.cosine_similarity(logits_final, logits_actual, dim=0).item()
print(f"Last hidden state + norm: cos={cos_final:.6f} (should be ~1.0)")

logits_final_no_norm = W_U @ h_last
cos_final_no_norm = torch.nn.functional.cosine_similarity(logits_final_no_norm, logits_actual, dim=0).item()
print(f"Last hidden state no norm: cos={cos_final_no_norm:.6f}")

# Check: is the ActivationRecorder output at layer 27 the same as hidden_states[28]?
with ActivationRecorder(block_layers, at=[27]) as recorder:
    text_module(input_ids=input_ids, use_cache=False)
    h_rec_27 = recorder.activations[27][0, -1, :].float()

h_hs_27 = hs[28][0, -1, :].float()
diff_27 = (h_rec_27 - h_hs_27).abs().max().item()
print(f"\nRecorder L27 vs hidden_states[28]: diff={diff_27:.6f}")
