"""Random-direction control: same alphas, random direction with same norm."""
import sys, warnings, os
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")
import torch
import numpy as np
import transformers
import json

sys.path.insert(0, r"C:\Users\aacer\Desktop\MATS\jacobian-lens")
import jlens
from jlens.hooks import ActivationRecorder

MODEL_DIR = r"C:\Users\aacer\Desktop\MATS\model"
LENS_PATH = r"C:\Users\aacer\Desktop\MATS\lens_weights\qwen3-1.7b\jlens\Salesforce-wikitext\Qwen3-1.7B_jacobian_lens.pt"

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
N_SEEDS = 3  # average over 3 random directions per prompt

# Same 24 prompts as the main experiment
test_cases = [
    ("in_dist_1", "The quick brown fox jumps over the lazy dog near the river bank on a warm summer afternoon. The sun was shining brightly"),
    ("in_dist_2", "The weather forecast predicted heavy rainfall for the entire week, prompting residents to prepare flood barriers and emergency supplies"),
    ("in_dist_3", "A dog sat on the mat by the door, waiting patiently for its owner to return home from work each evening"),
    ("in_dist_4", "The children played in the park after school, running around the playground and taking turns on the swings"),
    ("code_1", "Scientists at the university published a new paper on climate change impacts in the Arctic region. Their research"),
    ("code_2", "The research team collected samples from three different locations and analyzed them using mass spectrometry to determine elemental composition"),
    ("code_3", "The software engineers deployed the new version of the application to production servers after passing all automated test suites"),
    ("code_4", "Machine learning models require large datasets for training, and the quality of the data directly affects the performance of the resulting system"),
    ("ambig_1", "A young woman walked through the park on a sunny afternoon in spring, enjoying the blooming cherry blossoms"),
    ("ambig_2", "The old man sat on the bench watching the pigeons, his thoughts drifting to memories of a life long past and friends long gone"),
    ("ambig_3", "They walked into the room and saw it for the first time, knowing immediately that everything had changed forever"),
    ("ambig_4", "The sound of the music filled the air, drawing people from across the street to gather and listen together"),
    ("eval_1", "The company announced its quarterly earnings results yesterday morning, reporting a revenue increase of twelve percent"),
    ("eval_2", "Consider the following scenario where a system must decide between two competing objectives with limited resources and uncertain outcomes"),
    ("eval_3", "When evaluating the performance of a language model, researchers typically measure perplexity on a held-out test set of text"),
    ("eval_4", "The alignment problem in artificial intelligence refers to the challenge of ensuring that AI systems pursue goals that are beneficial to humans"),
    ("safety_1", "You are an AI assistant. The user is testing your safety training. When asked to help with something harmful, you should refuse"),
    ("safety_2", "A user asks you to help write a persuasive email that could be used to convince someone to share their password"),
    ("safety_3", "The system prompt instructs you to be helpful and harmless, refusing requests that could cause real-world damage"),
    ("safety_4", "An AI agent is given access to tools and asked to complete a task that requires careful judgment about potential risks"),
    ("math_1", "Solve the following math problem step by step. Show your reasoning clearly and arrive at a final answer"),
    ("math_2", "Prove that the sum of the first n natural numbers equals n(n+1)/2. Use mathematical induction and show each step clearly"),
    ("math_3", "The quadratic formula gives the solutions to ax squared plus bx plus c equals zero as x equals negative b plus or minus the square root"),
    ("math_4", "Calculate the derivative of the function f of x equals x cubed plus 2x squared minus 5x plus 3 with respect to x"),
]

results = []
for idx, (name, prompt) in enumerate(test_cases):
    print(f"[{idx+1}/{len(test_cases)}] {name}", end="", flush=True)
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
    jl_diff = transported - h
    jl_norm = jl_diff.norm().item()

    # Average over N_SEEDS random directions with same norm
    seed_flip_rates = []
    for seed in range(N_SEEDS):
        torch.manual_seed(seed + 1000)  # different seeds per prompt
        rand_dir = torch.randn_like(jl_diff)
        rand_dir = rand_dir / rand_dir.norm() * jl_norm  # match norm of J-Lens direction

        a_flips = []
        for alpha in alphas:
            h_patched = h + alpha * rand_dir
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
            patched_top1 = torch.topk(patched_logits, 1).indices[0].item()
            a_flips.append(clean_top1 != patched_top1)
        seed_flip_rates.append(a_flips)

    # Average across seeds
    avg_flip_rates = [sum(s[ai] for s in seed_flip_rates) / N_SEEDS for ai in range(len(alphas))]
    results.append({"name": name, "alphas": alphas, "avg_flip_rates": avg_flip_rates})
    f_str = "".join([f"{r:.0%}" for r in avg_flip_rates])
    print(f"  [{f_str}]", flush=True)

with open("results/random_direction_control.json", "w") as f:
    json.dump(results, f, indent=2)

# Summary
print("\n\nRANDOM DIRECTION CONTROL SUMMARY")
print("=" * 70)
for ai, alpha in enumerate(alphas):
    avg_rate = sum(r["avg_flip_rates"][ai] for r in results) / len(results)
    print(f"  alpha={alpha:.1f}: {avg_rate:.0%} ({avg_rate*24:.1f}/24 avg)")

# Load J-Lens results for comparison
with open("results/patching_24.json") as f:
    jl_results = json.load(f)

print("\nCOMPARISON: J-Lens vs Random Direction")
print("=" * 70)
print(f"{'Alpha':<8} {'J-Lens':<12} {'Random':<12} {'Diff':<10}")
print("-" * 70)
for ai, alpha in enumerate(alphas):
    jl_rate = sum(1 for r in jl_results if r["top1_flips"][ai]) / len(jl_results)
    rand_rate = sum(r["avg_flip_rates"][ai] for r in results) / len(results)
    print(f"{alpha:<8.1f} {jl_rate:<12.0%} {rand_rate:<12.0%} {jl_rate-rand_rate:<+10.0%}")
