"""Debug: what does a decoder layer actually return?"""
import sys, warnings, os
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")
import torch
import transformers

MODEL_DIR = r"C:\Users\aacer\Documents\J-Space\model"
model = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, dtype=torch.float32, device_map="cpu", local_files_only=True)
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model.eval()

block = model.model.layers[14]
encoded = tokenizer("Hello world", return_tensors="pt")
input_ids = encoded.input_ids

# Get position embeddings
h = model.model.embed_tokens(input_ids)
pos_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
pos_emb = model.model.rotary_emb(h, pos_ids)

# Run the block
with torch.no_grad():
    out = block(h, position_embeddings=pos_emb)

print(f"Type: {type(out)}")
if isinstance(out, tuple):
    for i, o in enumerate(out):
        if isinstance(o, torch.Tensor):
            print(f"  [{i}] shape={o.shape}")
        elif o is None:
            print(f"  [{i}] None")
        else:
            print(f"  [{i}] {type(o)}")
            if hasattr(o, 'shape'):
                print(f"       shape={o.shape}")
