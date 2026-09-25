"""Debug: check decoder layer output shape."""
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

text_module = model.model
block_layers = text_module.layers

encoded = tokenizer("Hello world", return_tensors="pt")
input_ids = encoded.input_ids

def debug_hook(module, input, output):
    print(f"  output type: {type(output)}")
    if isinstance(output, tuple):
        for i, o in enumerate(output):
            if isinstance(o, torch.Tensor):
                print(f"  output[{i}]: shape={o.shape}, dtype={o.dtype}")
            elif o is None:
                print(f"  output[{i}]: None")
            else:
                print(f"  output[{i}]: type={type(o)}")
    elif isinstance(output, torch.Tensor):
        print(f"  output: shape={output.shape}")
    return output

hook = block_layers[14].register_forward_hook(debug_hook)
with torch.no_grad():
    out = model(input_ids)
hook.remove()
print("Done")
