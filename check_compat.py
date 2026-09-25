import json
config = json.load(open(r"C:\Users\aacer\Documents\J-Space\model\config.json"))
print(f"Model: {config.get('model_type', '?')}")
print(f"Layers: {config.get('num_hidden_layers', '?')}")
print(f"d_model: {config.get('hidden_size', '?')}")
print(f"Vocab: {config.get('vocab_size', '?')}")
print(f"Head dim: {config.get('head_dim', '?')}")
print(f"Num heads: {config.get('num_attention_heads', '?')}")

import torch
lens = torch.load(
    r"C:\Users\aacer\Documents\J-Space\lens_weights\qwen3-1.7b\jlens\Salesforce-wikitext\Qwen3-1.7B_jacobian_lens.pt",
    map_location="cpu", weights_only=True
)
print(f"\nLens d_model: {lens['d_model']}")
print(f"Lens layers: {len(lens['source_layers'])} (indices {lens['source_layers'][0]}-{lens['source_layers'][-1]})")
print(f"Lens n_prompts: {lens['n_prompts']}")

assert lens["d_model"] == config["hidden_size"], "d_model mismatch!"
assert len(lens["source_layers"]) == config["num_hidden_layers"] - 1, "layer count mismatch!"
print("\nCOMPATIBLE")
