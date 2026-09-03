"""Hand-check: show the norm sign flip in front of the reader."""
import sys, warnings, os
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")
import torch
import transformers

sys.path.insert(0, r"C:\Users\aacer\Desktop\MATS\jacobian-lens")
from jlens.hooks import ActivationRecorder

MODEL_DIR = r"C:\Users\aacer\Desktop\MATS\model"

model = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, dtype=torch.float32, device_map="cpu", local_files_only=True)
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model.eval()

prompt = "The capital of France is"
encoded = tokenizer(prompt, return_tensors="pt")
input_ids = encoded.input_ids

# Get actual model logits
with torch.no_grad():
    out = model(input_ids)
    logits_actual = out.logits[0, -1, :].float()

text_module = model.model
block_layers = text_module.layers
W_U = model.lm_head.weight.data.float()
final_norm = text_module.norm

print("=" * 70)
print("HAND CHECK: Does final_norm flip the cosine sign?")
print("=" * 70)
print(f"Prompt: '{prompt}'")
print(f"Target position: last token ('is')")
print()

# Check at layer 14 (middle of network)
check_layer = 14

with ActivationRecorder(block_layers, at=[check_layer]) as recorder:
    text_module(input_ids=input_ids, use_cache=False)
    h = recorder.activations[check_layer][0, -1, :].float()

print(f"Layer {check_layer} activation h:")
print(f"  shape: {h.shape}")
print(f"  stats: mean={h.mean():.4f}, std={h.std():.4f}")
print()

# Method A: raw h @ W_U (no norm)
logits_raw = W_U @ h
cos_raw = torch.nn.functional.cosine_similarity(logits_raw, logits_actual, dim=0).item()
top5_raw = torch.topk(logits_raw, 5)

# Method B: norm(h) @ W_U (with norm) 
h_normed = final_norm(h.unsqueeze(0)).squeeze(0)
logits_normed = W_U @ h_normed
cos_normed = torch.nn.functional.cosine_similarity(logits_normed, logits_actual, dim=0).item()
top5_normed = torch.topk(logits_normed, 5)

# Method C: norm then check what happens to individual dimensions
# Show the effect of norm on a few specific dimensions
print("Effect of final_norm on 5 specific dimensions of h:")
print(f"  {'Dim':>5}  {'h_raw':>10}  {'h_normed':>10}  {'weight':>8}  {'ratio':>8}")
for dim in [0, 100, 500, 1000, 2047]:
    h_d = h[dim].item()
    hn_d = h_normed[dim].item()
    w_d = final_norm.weight.data[dim].item()
    ratio = hn_d / h_d if abs(h_d) > 1e-8 else float('inf')
    print(f"  {dim:5d}  {h_d:10.4f}  {hn_d:10.4f}  {w_d:8.4f}  {ratio:8.4f}")

print()
print(f"Method A (h @ W_U, no norm):")
print(f"  cosine with actual logits: {cos_raw:+.4f}")
print(f"  top-5 token ids: {top5_raw.indices.tolist()}")
print()
print(f"Method B (norm(h) @ W_U, with norm):")
print(f"  cosine with actual logits: {cos_normed:+.4f}")
print(f"  top-5 token ids: {top5_normed.indices.tolist()}")
print()
print(f"SIGN FLIP: cos went from {cos_raw:+.4f} to {cos_normed:+.4f}")
print(f"  Direction: {'SAME' if cos_raw * cos_normed > 0 else 'FLIPPED'}")
print()

# Now show: is it the norm itself, or the interaction with W_U?
# Compute cos(h, h_normed) — do they point in the same direction?
cos_h_hn = torch.nn.functional.cosine_similarity(h, h_normed, dim=0).item()
print(f"cos(h, norm(h)) = {cos_h_hn:.4f}")
print(f"  The norm changes the direction by {1 - cos_h_hn:.4f} (1.0 = no change)")
print()

# Compute the dot product of norm weights with direction of change
direction_change = h_normed - h
print(f"Direction of norm-induced change:")
print(f"  mean change: {direction_change.mean():.6f}")
print(f"  std of change: {direction_change.std():.4f}")
print(f"  max abs change: {direction_change.abs().max():.4f}")

# Does the change align with or oppose the actual logit direction?
# Project the change into logit space first
change_in_logit_space = W_U @ direction_change
cos_change_logits = torch.nn.functional.cosine_similarity(change_in_logit_space, logits_actual, dim=0).item()
print(f"  cos(change_in_logit_space, actual_logits) = {cos_change_logits:+.4f}")
print(f"  => The norm pushes logit representation {'TOWARD' if cos_change_logits > 0 else 'AWAY FROM'} the output")
