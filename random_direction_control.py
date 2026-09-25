"""
Random-direction control: same alphas, random direction with same norm.

Legacy script — now uses config.py for all paths and layer selection.
Run via: python random_direction_control.py [--model qwen3-1.7b]
"""
import sys, os, warnings, argparse
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "jacobian-lens"))
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import json
from pathlib import Path

from config import get_config, DEFAULT_MODEL_DIR, DEFAULT_LENS_FILE

def main():
    parser = argparse.ArgumentParser(description="Random-direction control experiment")
    parser.add_argument("--model", default="qwen3-1.7b")
    parser.add_argument("--n-seeds", type=int, default=3)
    args = parser.parse_args()

    cfg = get_config(model_id=args.model)
    model_dir = str(cfg.local_dir) if cfg.local_dir else str(DEFAULT_MODEL_DIR)
    lens_path = str(cfg.lens) if cfg.lens else str(DEFAULT_LENS_FILE)

    import transformers
    import jlens
    from jlens.hooks import ActivationRecorder

    print(f"Loading model from {model_dir}...")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, device_map="cpu", local_files_only=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    global_lens = jlens.JacobianLens.load(lens_path)

    text_module = model.model
    block_layers = text_module.layers
    W_U = model.lm_head.weight.data.float()

    n_layers = model.config.num_hidden_layers
    layer = cfg.mid_layer_for(n_layers)
    alphas = cfg.thresholds.patch_alphas
    N_SEEDS = args.n_seeds

    print(f"Layer: {layer}  Alphas: {alphas}  Seeds: {N_SEEDS}")

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

    output_dir = Path(__file__).parent / "results" / "random_direction_control"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for idx, (name, prompt) in enumerate(test_cases):
        print(f"[{idx+1}/{len(test_cases)}] {name}", end="", flush=True)
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=cfg.max_seq_len)
        input_ids = encoded.input_ids

        with torch.no_grad():
            clean_logits = model(input_ids).logits[0, -1, :].float()
            clean_top1 = int(torch.argmax(clean_logits).item())

        Jg = global_lens.jacobians[layer].float()
        with ActivationRecorder(block_layers, at=[layer]) as recorder:
            text_module(input_ids=input_ids, use_cache=False)
            h = recorder.activations[layer][0, -1, :].float()

        transported = Jg @ h
        jl_diff = transported - h
        jl_norm = jl_diff.norm().item()

        seed_flip_rates = []
        for seed in range(N_SEEDS):
            g = torch.Generator()
            g.manual_seed(seed + 1000 + idx * 100)  # unique seed per (seed, prompt) pair
            rand_dir = torch.randn(jl_diff.shape, generator=g)
            rand_dir = rand_dir / rand_dir.norm() * jl_norm

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
                patched_top1 = int(torch.argmax(patched_logits).item())
                a_flips.append(int(clean_top1 != patched_top1))
            seed_flip_rates.append(a_flips)

        avg_flip_rates = [sum(s[ai] for s in seed_flip_rates) / N_SEEDS for ai in range(len(alphas))]
        results.append({"name": name, "alphas": alphas, "avg_flip_rates": avg_flip_rates})
        f_str = "".join([f"{r:.0%}" for r in avg_flip_rates])
        print(f"  [{f_str}]", flush=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n\nRANDOM DIRECTION CONTROL SUMMARY")
    print("=" * 70)
    for ai, alpha in enumerate(alphas):
        avg_rate = sum(r["avg_flip_rates"][ai] for r in results) / len(results)
        print(f"  alpha={alpha:.1f}: {avg_rate:.0%} ({avg_rate*len(test_cases):.1f}/{len(test_cases)} avg)")

    jl_results_path = Path(__file__).parent / "results" / "patching_24.json"
    if jl_results_path.exists():
        with open(jl_results_path) as f:
            jl_results = json.load(f)
        print("\nCOMPARISON: J-Lens vs Random Direction")
        print("=" * 70)
        print(f"{'Alpha':<8} {'J-Lens':<12} {'Random':<12} {'Diff':<10}")
        print("-" * 70)
        for ai, alpha in enumerate(alphas):
            jl_rate = sum(1 for r in jl_results if r["top1_flips"][ai]) / len(jl_results)
            rand_rate = sum(r["avg_flip_rates"][ai] for r in results) / len(results)
            print(f"{alpha:<8.1f} {jl_rate:<12.0%} {rand_rate:<12.0%} {jl_rate-rand_rate:<+10.0%}")
    else:
        print(f"\n(results/patching_24.json not found — skipping J-Lens comparison)")

    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()
